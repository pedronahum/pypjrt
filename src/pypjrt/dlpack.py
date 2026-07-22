"""DLPack: share device buffers with torch / jax / numpy / cupy, no host copy.

**Zero-copy import is the half that usually goes unimplemented**: export alone
is much easier, and PyCapsule construction is the fiddly part. Yet import is
what makes pypjrt *compose with* the ecosystem instead of competing with it,
which is the posture PJRT's first goal describes.

Export pins device memory with ``PJRT_Buffer_IncreaseExternalReferenceCount``
and releases it in the consumer's deleter. The lifetime hazards are real:

  - the external reference must be released *after* everything that depends on
    it, or the underlying buffer's destructor trips a CHECK;
  - the consumer's deleter runs on an arbitrary thread (ctypes reacquires the
    GIL for us, which is one place Python is easier than C++);
  - an unconsumed capsule must still free the tensor, and a consumed one --
    renamed ``used_dltensor`` -- must not be freed twice.

Import goes through ``PJRT_Client_CreateViewOfDeviceBuffer``, whose
``on_delete_callback`` is where we hand the producer's memory back.
"""

from __future__ import annotations

import ctypes
from typing import Any

from . import errors

_VOIDP = ctypes.c_void_p

# ---------------------------------------------------------------------------
# DLPack ABI (dlpack.h). Classic unversioned form; the versioned form is
# accepted on import.

kDLCPU, kDLCUDA, kDLCUDAHost, kDLROCM = 1, 2, 3, 10
kDLInt, kDLUInt, kDLFloat, kDLOpaque, kDLBfloat, kDLComplex, kDLBool = 0, 1, 2, 3, 4, 5, 6


class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16)]


class DLTensor(ctypes.Structure):
    _fields_ = [("data", _VOIDP), ("device", DLDevice), ("ndim", ctypes.c_int32),
                ("dtype", DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64)]


class DLManagedTensor(ctypes.Structure):
    pass


_DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))
DLManagedTensor._fields_ = [("dl_tensor", DLTensor), ("manager_ctx", _VOIDP),
                            ("deleter", _DELETER)]


class DLPackVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]


class DLManagedTensorVersioned(ctypes.Structure):
    pass


_DELETER_V = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensorVersioned))
DLManagedTensorVersioned._fields_ = [
    ("version", DLPackVersion), ("manager_ctx", _VOIDP), ("deleter", _DELETER_V),
    ("flags", ctypes.c_uint64), ("dl_tensor", DLTensor)]

#: PJRT_Buffer_Type -> (DLPack code, bits). Built from the dtype markers, which
#: take their codes from the generated ABI -- see typing.py for why a literal
#: table here was wrong.
from .typing import BF16, F16, F32, F64, PRED, S8, S16, S32, S64, U8, U16, U32, U64

_PJRT_TO_DL = {
    PRED.code: (kDLBool, 8),
    S8.code: (kDLInt, 8), S16.code: (kDLInt, 16),
    S32.code: (kDLInt, 32), S64.code: (kDLInt, 64),
    U8.code: (kDLUInt, 8), U16.code: (kDLUInt, 16),
    U32.code: (kDLUInt, 32), U64.code: (kDLUInt, 64),
    F16.code: (kDLFloat, 16), F32.code: (kDLFloat, 32), F64.code: (kDLFloat, 64),
    BF16.code: (kDLBfloat, 16),
}
_DL_TO_PJRT = {v: k for k, v in _PJRT_TO_DL.items()}

# ---------------------------------------------------------------------------
# CPython capsule API through ctypes

_py = ctypes.pythonapi

# The capsule API is bound over RAW PyObject pointers, never ctypes.py_object.
# A capsule destructor receives an object whose refcount has already reached
# zero; letting ctypes convert it to py_object INCREFs a dying object and
# resurrects it, which segfaults. Passing addresses avoids every refcount
# interaction. `id(obj)` is the PyObject* in CPython.
_CAPSULE_DTOR = ctypes.CFUNCTYPE(None, _VOIDP)
_py.PyCapsule_New.restype = _VOIDP
_py.PyCapsule_New.argtypes = [_VOIDP, ctypes.c_char_p, _CAPSULE_DTOR]
_py.PyCapsule_GetPointer.restype = _VOIDP
_py.PyCapsule_GetPointer.argtypes = [_VOIDP, ctypes.c_char_p]
_py.PyCapsule_IsValid.restype = ctypes.c_int
_py.PyCapsule_IsValid.argtypes = [_VOIDP, ctypes.c_char_p]
_py.PyCapsule_GetName.restype = ctypes.c_char_p
_py.PyCapsule_GetName.argtypes = [_VOIDP]
_py.PyCapsule_SetName.restype = ctypes.c_int
_py.PyCapsule_SetName.argtypes = [_VOIDP, ctypes.c_char_p]


def _obj(o) -> _VOIDP:
    """The PyObject* of a live Python object."""
    return _VOIDP(id(o))


_py.Py_DecRef.argtypes = [_VOIDP]


def _steal(ptr):
    """Adopt a PyObject* that we already own a reference to.

    ``ctypes.cast(ptr, py_object).value`` INCREFs, so combined with the
    reference PyCapsule_New already returned the object would sit at 2 and
    never die -- its destructor would never run and the pin would leak.
    Balance it explicitly.
    """
    obj = ctypes.cast(ptr, ctypes.py_object).value
    _py.Py_DecRef(ptr)
    return obj

_UNUSED = b"dltensor"
_USED = b"used_dltensor"
_UNUSED_V = b"dltensor_versioned"
_USED_V = b"used_dltensor_versioned"

#: Everything alive on behalf of an exported tensor, keyed by its address.
#: Entries are removed by the consumer's deleter, on an arbitrary thread.
_EXPORTS: dict[int, dict[str, Any]] = {}
#: Callback objects the plugin retains for the life of an imported view.
_IMPORT_KEEP: list = []


# ---------------------------------------------------------------------------
# export


def _release_export(addr: int) -> None:
    entry = _EXPORTS.pop(addr, None)
    if entry is None:
        return
    # Order matters: drop everything that depends on the pin, *then* unpin.
    # The mirror image of this in C++ is field-declaration order -- get it
    # wrong and the buffer's destructor trips a CHECK.
    plugin, buf_ptr = entry["plugin"], entry["buffer_ptr"]
    entry.pop("shape", None)
    entry.pop("strides", None)
    entry.pop("tensor", None)
    try:
        a = plugin.args("PJRT_Buffer_DecreaseExternalReferenceCount_Args", buffer=buf_ptr)
        plugin.call("PJRT_Buffer_DecreaseExternalReferenceCount", a)
    except errors.PjrtError:
        pass


@_DELETER
def _managed_deleter(ptr):
    """Called by the consumer, from a thread we do not control."""
    if ptr:
        _release_export(ctypes.addressof(ptr.contents))


@_CAPSULE_DTOR
def _capsule_destructor(capsule_ptr):
    """Runs only if the capsule was never consumed.

    `capsule_ptr` is a raw PyObject* whose refcount is already zero. Do not
    convert it to a Python object.
    """
    try:
        if not _py.PyCapsule_IsValid(capsule_ptr, _UNUSED):
            return          # consumed: renamed to used_dltensor, consumer owns it
        p = _py.PyCapsule_GetPointer(capsule_ptr, _UNUSED)
        if p:
            _release_export(int(p))
    except Exception:
        pass


def buffer_to_dlpack(buffer, *, stream: int | None = None):
    """Wrap a :class:`~pypjrt.client.Buffer` in an unconsumed DLPack capsule."""
    plugin = buffer._plugin
    ptr = buffer._check()

    dims = tuple(buffer.dimensions)
    dtype = buffer.element_type
    if dtype not in _PJRT_TO_DL:
        raise errors.InvalidArgument(f"no DLPack dtype for PJRT_Buffer_Type {dtype}")
    code, bits = _PJRT_TO_DL[dtype]

    data = buffer.device_pointer()
    if not data:
        raise errors.UnsupportedByPlugin(
            "plugin returned a null device pointer; zero-copy export unavailable")

    # Pin the device memory for as long as the consumer holds the tensor.
    plugin.call("PJRT_Buffer_IncreaseExternalReferenceCount",
                plugin.args("PJRT_Buffer_IncreaseExternalReferenceCount_Args", buffer=ptr))

    try:
        shape = (ctypes.c_int64 * max(len(dims), 1))(*dims) if dims else (ctypes.c_int64 * 1)()
        mt = DLManagedTensor()
        ctypes.memset(ctypes.byref(mt), 0, ctypes.sizeof(mt))
        mt.dl_tensor.data = data
        mt.dl_tensor.device = DLDevice(*buffer_dlpack_device(buffer))
        mt.dl_tensor.ndim = len(dims)
        mt.dl_tensor.dtype = DLDataType(code, bits, 1)
        mt.dl_tensor.shape = ctypes.cast(shape, ctypes.POINTER(ctypes.c_int64))
        mt.dl_tensor.strides = None          # dense, major-to-minor
        mt.dl_tensor.byte_offset = 0
        mt.deleter = _managed_deleter

        addr = ctypes.addressof(mt)
        _EXPORTS[addr] = {"plugin": plugin, "buffer_ptr": ptr, "tensor": mt,
                          "shape": shape, "buffer": buffer}
        cap_ptr = _py.PyCapsule_New(_VOIDP(addr), _UNUSED, _capsule_destructor)
        if not cap_ptr:
            raise errors.Internal("PyCapsule_New failed")
        return _steal(cap_ptr)
    except BaseException:
        plugin.call("PJRT_Buffer_DecreaseExternalReferenceCount",
                    plugin.args("PJRT_Buffer_DecreaseExternalReferenceCount_Args", buffer=ptr))
        raise


def buffer_dlpack_device(buffer) -> tuple[int, int]:
    """``(device_type, device_id)`` for ``__dlpack_device__``."""
    plugin = buffer._plugin
    a = plugin.args("PJRT_Buffer_Device_Args", buffer=buffer._check())
    plugin.call("PJRT_Buffer_Device", a)
    d = plugin.args("PJRT_Device_LocalHardwareId_Args", device=a.device)
    try:
        plugin.call("PJRT_Device_LocalHardwareId", d)
        ordinal = max(int(d.local_hardware_id), 0)
    except errors.PjrtError:
        ordinal = 0
    on_cpu = plugin.args("PJRT_Buffer_IsOnCpu_Args", buffer=buffer._check())
    try:
        plugin.call("PJRT_Buffer_IsOnCpu", on_cpu)
        cpu = bool(on_cpu.is_on_cpu)
    except errors.PjrtError:
        cpu = not plugin.is_accelerator
    return (kDLCPU, 0) if cpu else (kDLCUDA, ordinal)


# ---------------------------------------------------------------------------
# import


def _extract(capsule) -> tuple[DLTensor, int, Any, bytes]:
    """Return ``(dl_tensor, manager_addr, deleter_fn, consumed_name)``."""
    cap = _obj(capsule)
    if _py.PyCapsule_IsValid(cap, _UNUSED):
        p = _py.PyCapsule_GetPointer(cap, _UNUSED)
        mt = ctypes.cast(p, ctypes.POINTER(DLManagedTensor)).contents
        return mt.dl_tensor, int(p), mt.deleter, _USED
    if _py.PyCapsule_IsValid(cap, _UNUSED_V):
        p = _py.PyCapsule_GetPointer(cap, _UNUSED_V)
        mv = ctypes.cast(p, ctypes.POINTER(DLManagedTensorVersioned)).contents
        if mv.version.major > 1:
            raise errors.InvalidArgument(
                f"DLPack version {mv.version.major}.{mv.version.minor} is newer than 1.x")
        return mv.dl_tensor, int(p), mv.deleter, _USED_V
    name = _py.PyCapsule_GetName(cap)
    if name in (_USED, _USED_V):
        raise errors.InvalidArgument(
            "this DLPack capsule was already consumed; a capsule may be used once")
    raise errors.InvalidArgument(f"not a DLPack capsule (name {name!r})")


def buffer_from_dlpack(client, obj, *, device=None):
    """Adopt another framework's device buffer with no copy.

    Accepts anything exposing ``__dlpack__``, or a capsule directly.
    """
    from .client import Buffer, Device

    capsule = obj.__dlpack__() if hasattr(obj, "__dlpack__") else obj
    dl, manager_addr, deleter, consumed_name = _extract(capsule)

    ndim = int(dl.ndim)
    dims = tuple(int(dl.shape[i]) for i in range(ndim)) if dl.shape else ()
    if dl.strides:
        expected, stride = [], 1
        for d in reversed(dims):
            expected.append(stride)
            stride *= d
        expected.reverse()
        actual = [int(dl.strides[i]) for i in range(ndim)]
        if actual != expected:
            raise errors.InvalidArgument(
                f"only dense major-to-minor tensors can be adopted; got strides {actual}, "
                f"expected {expected}")
    key = (int(dl.dtype.code), int(dl.dtype.bits))
    if int(dl.dtype.lanes) != 1 or key not in _DL_TO_PJRT:
        raise errors.InvalidArgument(
            f"unsupported DLPack dtype code={dl.dtype.code} bits={dl.dtype.bits} "
            f"lanes={dl.dtype.lanes}")
    pjrt_dtype = _DL_TO_PJRT[key]
    data = int(dl.data or 0) + int(dl.byte_offset)
    if not data:
        raise errors.InvalidArgument("DLPack tensor has a null data pointer")

    plugin = client._plugin
    if device is None:
        device = Device(plugin, client._addressable()[0])

    # We own the capsule from here; mark it consumed before handing memory over.
    _py.PyCapsule_SetName(_obj(capsule), consumed_name)

    holder: dict[str, Any] = {"deleter": deleter, "manager": manager_addr}

    def _on_delete(_ptr, _arg):
        d = holder.pop("deleter", None)
        m = holder.pop("manager", None)
        if d and m:
            try:
                d(ctypes.cast(_VOIDP(m), ctypes.POINTER(DLManagedTensor)))
            except Exception:
                pass

    cb = ctypes.CFUNCTYPE(None, _VOIDP, _VOIDP)(_on_delete)
    _IMPORT_KEEP.append((cb, holder))

    dim_arr = (ctypes.c_int64 * max(len(dims), 1))(*dims) if dims else (ctypes.c_int64 * 1)()
    a = plugin.args(
        "PJRT_Client_CreateViewOfDeviceBuffer_Args", client=client._check(),
        device_buffer_ptr=data, dims=ctypes.cast(dim_arr, _VOIDP), num_dims=len(dims),
        element_type=pjrt_dtype, device=device.address,
        on_delete_callback=ctypes.cast(cb, _VOIDP))
    try:
        plugin.call("PJRT_Client_CreateViewOfDeviceBuffer", a)
    except errors.PjrtError as e:
        # We never took ownership, so give the producer's memory straight back.
        _py.PyCapsule_SetName(_obj(capsule), _UNUSED if consumed_name == _USED else _UNUSED_V)
        holder.clear()
        _IMPORT_KEEP.remove((cb, holder))
        if "align" in e.message:
            raise type(e)(
                f"{e.message}\n  hint: zero-copy import requires the producer's allocation to "
                f"meet the backend's alignment. numpy's default allocator does not "
                f"(this pointer is {data % 64} bytes past a 64-byte boundary); buffers from "
                f"jax, torch or cupy do. Use Client.buffer_from_host() to copy instead.",
                e.code) from e
        raise
    return Buffer(plugin, a.buffer, keepalive=(cb, holder, dim_arr))


def live_exports() -> int:
    """Exported tensors whose consumer has not released them yet."""
    return len(_EXPORTS)
