"""Decoding an ``XLA_FFI_CallFrame``.

Layouts come from the generated ABI module, so no offset is hand-written. They
agree exactly with the values `spike/pjrt_ffi.py` derived by hand from the
header -- frame 176 B with args@40 rets@80 attrs@120 future@168, buffer 48 B
with dims@40, ``Error_Create``@48 and ``Stream_Get``@80 in ``XLA_FFI_Api``.
"""

from __future__ import annotations

import ctypes
from typing import Any

_VOIDP = ctypes.c_void_p

#: XLA_FFI_DataType -> (ctypes scalar, itemsize). NOTE this is *not*
#: PJRT_Buffer_Type: the two enums number the same concepts differently (BF16
#: is 16 here and 13 there). Sub-byte and exotic float types are deliberately
#: absent -- we expose their raw address instead.
_CTYPE = {
    1: (ctypes.c_bool, 1),      # PRED
    2: (ctypes.c_int8, 1), 3: (ctypes.c_int16, 2),
    4: (ctypes.c_int32, 4), 5: (ctypes.c_int64, 8),
    6: (ctypes.c_uint8, 1), 7: (ctypes.c_uint16, 2),
    8: (ctypes.c_uint32, 4), 9: (ctypes.c_uint64, 8),
    11: (ctypes.c_float, 4), 12: (ctypes.c_double, 8),
}


class FfiBuffer:
    """One argument or result. ``data`` is the raw device address."""

    __slots__ = ("dtype", "data", "dims")

    def __init__(self, dtype: int, data: int, dims: tuple[int, ...]):
        self.dtype, self.data, self.dims = dtype, data, dims

    @property
    def size(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def itemsize(self) -> int:
        return _CTYPE[self.dtype][1] if self.dtype in _CTYPE else 0

    @property
    def nbytes(self) -> int:
        return self.size * self.itemsize

    def as_ctypes(self):
        """A ctypes array over the buffer. Host-addressable memory only.

        On an accelerator this address is a device pointer: pass it to a kernel
        launch, do not dereference it.
        """
        if self.dtype not in _CTYPE:
            raise TypeError(f"no ctypes mapping for XLA_FFI_DataType {self.dtype}")
        cty, _ = _CTYPE[self.dtype]
        return (cty * self.size).from_address(self.data)

    def __repr__(self) -> str:
        return f"<FfiBuffer dtype={self.dtype} dims={self.dims} data=0x{self.data:x}>"


def _buffers(abi, args_struct, field: str) -> list[FfiBuffer]:
    # XLA_FFI_Args names its pointer array `args`; XLA_FFI_Rets names its `rets`.
    n = int(args_struct.size)
    ptr = getattr(args_struct, field)
    if not n or not ptr:
        return []
    vals = ctypes.cast(ptr, ctypes.POINTER(_VOIDP))
    out = []
    for i in range(n):
        b = ctypes.cast(vals[i], ctypes.POINTER(abi.XLA_FFI_Buffer)).contents
        rank = int(b.rank)
        dims = ctypes.cast(b.dims, ctypes.POINTER(ctypes.c_int64))
        out.append(FfiBuffer(int(b.dtype), int(b.data or 0),
                             tuple(int(dims[j]) for j in range(rank))))
    return out


def _decode_attrs(abi, attrs) -> dict[str, Any]:
    """``backend_config`` as a dict.

    The spelling matters: with ``api_version = 4`` the attributes must be a
    dictionary attribute *on the op*. The `mhlo.backend_config` spelling
    compiles fine and delivers an empty dict (verified against jaxlib 0.10).
    """
    n = int(attrs.size)
    if not n:
        return {}
    types = ctypes.cast(attrs.types, ctypes.POINTER(ctypes.c_int32))
    names = ctypes.cast(attrs.names, ctypes.POINTER(_VOIDP))
    vals = ctypes.cast(attrs.attrs, ctypes.POINTER(_VOIDP))
    out: dict[str, Any] = {}
    for i in range(n):
        span = ctypes.cast(names[i], ctypes.POINTER(abi.XLA_FFI_ByteSpan)).contents
        key = ctypes.string_at(span.ptr, span.len).decode(errors="replace")
        t = int(types[i])
        if t == abi.XLA_FFI_AttrType_SCALAR:
            sc = ctypes.cast(vals[i], ctypes.POINTER(abi.XLA_FFI_Scalar)).contents
            dt = int(sc.dtype)
            out[key] = (_CTYPE[dt][0].from_address(sc.value).value
                        if dt in _CTYPE else int(sc.value or 0))
        elif t == abi.XLA_FFI_AttrType_STRING:
            sp = ctypes.cast(vals[i], ctypes.POINTER(abi.XLA_FFI_ByteSpan)).contents
            out[key] = ctypes.string_at(sp.ptr, sp.len).decode(errors="replace")
        elif t == abi.XLA_FFI_AttrType_ARRAY:
            ar = ctypes.cast(vals[i], ctypes.POINTER(abi.XLA_FFI_Array)).contents
            dt, sz = int(ar.dtype), int(ar.size)
            out[key] = (list((_CTYPE[dt][0] * sz).from_address(ar.data))
                        if dt in _CTYPE and ar.data else [])
        elif t == abi.XLA_FFI_AttrType_DICTIONARY:
            out[key] = _decode_attrs(
                abi, ctypes.cast(vals[i], ctypes.POINTER(abi.XLA_FFI_Attrs)).contents)
    return out


class CallFrame:
    """What a handler receives. Read-only apart from writing into ``rets``."""

    def __init__(self, abi, ptr):
        self._abi = abi
        self._f = ctypes.cast(ptr, ctypes.POINTER(abi.XLA_FFI_CallFrame)).contents
        self.stage = int(self._f.stage)
        self.args = _buffers(abi, self._f.args, "args")
        self.rets = _buffers(abi, self._f.rets, "rets")
        self.attrs = _decode_attrs(abi, self._f.attrs)

    @property
    def api(self) -> int:
        return int(self._f.api or 0)

    def stream(self) -> int:
        """XLA's stream for this execution.

        Launch on it; never synchronize it and never switch context. Returns 0
        on backends without one (CPU).
        """
        api = self.api
        if not api:
            return 0
        fn_ptr = ctypes.cast(api + self._abi.XLA_FFI_Api.XLA_FFI_Stream_Get.offset,
                             ctypes.POINTER(_VOIDP))[0]
        if not fn_ptr:
            return 0
        a = self._abi.XLA_FFI_Stream_Get_Args()
        ctypes.memset(ctypes.byref(a), 0, ctypes.sizeof(a))
        a.struct_size = self._abi.XLA_FFI_Stream_Get_Args_STRUCT_SIZE
        a.ctx = self._f.ctx
        err = ctypes.CFUNCTYPE(_VOIDP, _VOIDP)(fn_ptr)(ctypes.byref(a))
        return 0 if err else int(a.stream or 0)

    def __repr__(self) -> str:
        return (f"<CallFrame stage={self.stage} args={len(self.args)} "
                f"rets={len(self.rets)} attrs={sorted(self.attrs)}>")
