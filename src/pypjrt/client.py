"""Client, Device, Buffer, Event, Executable.

Two API shapes are fixed here and are deliberately not deferrable:

  Shape 1 -- Event is FUTURE-SHAPED. ``result()``/``done()``/
  ``add_done_callback()`` ship now with a *blocking* implementation, so the
  ``PJRT_Event_OnReady`` upgrade (M9) changes the implementation and not the
  API. All three source projects shipped ``await()`` and deferred async; every
  caller written against that has to be rewritten the day it lands.

  Shape 2 -- Execute is DEVICE-LIST-SHAPED. The C API's central structure is
  ``argument_lists[num_devices][num_args]``. We model that internally and put
  the single-device convenience on top, never the reverse. A client that
  hardcodes ``num_devices = 1`` can never grow a sharding story.
"""

from __future__ import annotations

import array
import ctypes
import enum
import threading
import warnings
from contextlib import contextmanager
from typing import Any, Generic, Iterator, Sequence

from . import errors
from ._lock import dispatching
from ._loader import Plugin
from .compile_options import CompileOptions
from .typing import DT, DType

_VOIDP = ctypes.c_void_p
_I64 = ctypes.c_int64


def _fmt_bytes(n: int) -> str:
    if n < 0:
        return str(n)
    v = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if v < 1024 or unit == "TiB":
            return f"{v:.1f} {unit}" if unit != "B" else f"{int(v)} B"
        v /= 1024.0
    return str(v)


#: Reported first in an allocation-failure message; the rest follow.
_OOM_KEYS = ("bytes_in_use", "bytes_limit", "largest_free_block_bytes",
             "peak_bytes_in_use", "bytes_reserved", "pool_bytes",
             "num_allocs", "largest_alloc_size")


def _dense_layout(abi, dims: Sequence[int]):
    """A row-major ``PJRT_Buffer_MemoryLayout``, and everything keeping it alive.

    Some entry points ``CHECK``-fail on a shape without a layout -- and a CHECK
    in XLA *aborts the process*, it does not return an error. So anything that
    takes a shape gets a real layout rather than NULL.
    """
    rank = len(dims)
    m2m = (ctypes.c_int64 * max(rank, 1))(*range(rank - 1, -1, -1))
    lay = abi.PJRT_Buffer_MemoryLayout()
    ctypes.memset(ctypes.byref(lay), 0, ctypes.sizeof(lay))
    lay.struct_size = abi.PJRT_Buffer_MemoryLayout_STRUCT_SIZE
    lay.type = abi.PJRT_Buffer_MemoryLayout_Type_Tiled
    lay.tiled.struct_size = abi.PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE
    lay.tiled.minor_to_major = ctypes.cast(m2m, _VOIDP)
    lay.tiled.minor_to_major_size = rank
    return lay, [lay, m2m]


def _require_dtype(dtype: int, where: str) -> None:
    """Reject PJRT_Buffer_Type_INVALID before it reaches the plugin.

    ``pjrt_c_api_helpers.cc`` does ``CHECK(false) << "Buffer type is not
    supported in C API layer"`` on an unknown element type, which *aborts the
    process*. A zeroed shape field is an easy way to hit it, so validate here
    rather than hand XLA a 0.
    """
    from .typing import by_code
    if by_code(dtype) is None:
        raise errors.InvalidArgument(
            f"{where}: element type {dtype} is not a valid PJRT_Buffer_Type; "
            f"passing it would abort the process inside XLA")


def _byte_view(data: Any) -> memoryview:
    """A contiguous byte view of any buffer-protocol object.

    ``memoryview`` refuses formats it does not know -- notably bfloat16 and the
    f8 types, which numpy exposes through ml_dtypes as format ``'E'``. Those are
    exactly the dtypes an accelerator cares most about, so reinterpret through
    the array's own ``.view()`` (zero-copy) before giving up and copying.
    """
    if isinstance(data, memoryview):
        return data.cast("B") if data.format != "B" else data
    try:
        return memoryview(data).cast("B")
    except (TypeError, ValueError):
        pass
    view = getattr(data, "view", None)
    if callable(view):
        try:
            raw: Any = view("u1")          # callable() narrows the return to object
            return memoryview(raw).cast("B")
        except Exception:
            pass
    raise TypeError(
        f"{type(data).__name__} does not expose a contiguous buffer; pass bytes, "
        f"an array, or an object supporting the buffer protocol")


class State(enum.Enum):
    LIVE = "live"
    DONATED = "donated"     # device memory consumed by a donating execute
    CLOSED = "closed"


class _Owned:
    """A handle that must be destroyed exactly once.

    Python has no linear types, so this is a runtime state machine -- which is
    is what statically-typed clients end up shipping anyway: an idempotent
    ``destroy`` and a runtime-checked await. Not a degradation.
    """

    _destroy_fn: str = ""
    _destroy_args: str = ""
    _arg_field: str = ""

    def __init__(self, plugin: Plugin, ptr: int):
        self._plugin = plugin
        self._ptr = ptr
        self._state = State.LIVE

    @property
    def address(self) -> int:
        """Raw ``PJRT_*`` pointer. Needed for donation aliasing checks."""
        return self._ptr

    def _check(self) -> int:
        if self._state is not State.LIVE:
            raise errors.HandleClosed(
                f"{type(self).__name__} is {self._state.value}")
        return self._ptr

    def close(self) -> None:
        """Idempotent.

        A donated handle is still destroyed: only its *device memory* was
        consumed, the PJRT_Buffer wrapper still has to be freed. Verified safe
        on both plugins here.
        """
        if self._state is State.CLOSED:
            return
        self._state = State.CLOSED
        if self._destroy_fn:
            a = self._plugin.args(self._destroy_args, **{self._arg_field: self._ptr})
            self._plugin.call(self._destroy_fn, a)

    def _mark_donated(self) -> None:
        self._state = State.DONATED

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        # Runs at arbitrary points, including interpreter shutdown -- where
        # imports fail and module globals may already be torn down. Everything
        # here must be import-free and total.
        if getattr(self, "_state", None) is not State.LIVE:
            return
        try:
            warnings.warn(f"{type(self).__name__} was never closed", ResourceWarning,
                          stacklevel=2)
        except Exception:
            pass
        try:
            self.close()
        except Exception:
            pass


# --------------------------------------------------------------------------


#: OnReady trampolines, held until they fire. Keyed by id(entry).
_ONREADY: dict[int, Any] = {}
_ONREADY_LOCK = threading.Lock()
_ONREADY_T = ctypes.CFUNCTYPE(None, _VOIDP, _VOIDP)


class Event(_Owned):
    """Completion of an async device operation -- shaped as a future.

    The shape was fixed in M0 with a blocking implementation precisely so this
    milestone could change the implementation without touching the API.
    ``add_done_callback`` now goes through
    ``PJRT_Event_OnReady``; ``result()`` still blocks, which is what a caller
    asking for a value wants.
    """

    _destroy_fn = "PJRT_Event_Destroy"
    _destroy_args = "PJRT_Event_Destroy_Args"
    _arg_field = "event"

    @classmethod
    def create(cls, plugin: Plugin) -> "Event":
        """A host-minted event, completed later with :meth:`set`.

        *A future you can create before the computation exists* -- which is
        also Pathways' control-plane currency. ``PJRT_Event_Create`` and
        ``PJRT_Event_Set`` are rarely bound by a client at all.
        """
        a = plugin.args("PJRT_Event_Create_Args")
        plugin.call("PJRT_Event_Create", a)
        return cls(plugin, a.event)

    def set(self, error_code: int = 0, message: str = "") -> None:
        """Complete a host-minted event, optionally with an error."""
        msg = message.encode()
        buf = ctypes.create_string_buffer(msg, len(msg)) if msg else None
        a = self._plugin.args(
            "PJRT_Event_Set_Args", event=self._check(), error_code=error_code,
            error_message=ctypes.cast(buf, _VOIDP) if buf else None,
            error_message_size=len(msg))
        self._plugin.call("PJRT_Event_Set", a)

    def done(self) -> bool:
        a = self._plugin.args("PJRT_Event_IsReady_Args", event=self._check())
        self._plugin.call("PJRT_Event_IsReady", a)
        return bool(a.is_ready)

    def result(self, timeout: float | None = None) -> None:
        """Block until complete; raise if the operation failed.

        ``PJRT_Event_Error`` is the difference between "it finished" and "it
        succeeded". No source repo binds it, so a failed transfer there surfaces
        as wrong data rather than an exception.
        """
        p = self._check()
        self._plugin.call("PJRT_Event_Await", self._plugin.args("PJRT_Event_Await_Args", event=p))
        self._plugin.call("PJRT_Event_Error", self._plugin.args("PJRT_Event_Error_Args", event=p))

    def add_done_callback(self, fn) -> None:
        """Invoke ``fn(error_message_or_None)`` when the event completes.

        Non-blocking where the plugin offers ``PJRT_Event_OnReady``; the
        callback runs on one of XLA's threads, and ctypes acquires the GIL for
        us. Falls back to blocking + calling inline when the entry point is
        absent, so the contract holds either way.

        The callback must not call back into PJRT: it can run while another
        thread holds the process lock waiting on this very event.
        """
        try:
            self._plugin.fn("PJRT_Event_OnReady")
        except errors.PjrtError:
            try:
                self.result()
                fn(None)
            except errors.PjrtError as e:
                fn(e.message)
            return

        entry: dict[str, Any] = {"fn": fn, "plugin": self._plugin}

        def _on_ready(err_ptr, _user_arg):
            key = entry.pop("key", None)
            if key is not None:
                with _ONREADY_LOCK:
                    _ONREADY.pop(key, None)
            message = None
            if err_ptr:
                # The header is explicit: ownership of `error` passes to us and
                # we must destroy it. Forgetting leaks plugin memory per event.
                try:
                    message = self._plugin._to_exception(int(err_ptr)).message
                except errors.PjrtError as e:
                    message = e.message
            try:
                with dispatching():
                    fn(message)
            except Exception:
                pass  # never unwind into the plugin's thread

        trampoline = _ONREADY_T(_on_ready)
        entry["cb"] = trampoline
        key = id(entry)
        entry["key"] = key
        with _ONREADY_LOCK:
            _ONREADY[key] = entry

        a = self._plugin.args("PJRT_Event_OnReady_Args", event=self._check(),
                              callback=ctypes.cast(trampoline, _VOIDP))
        try:
            self._plugin.call("PJRT_Event_OnReady", a)
        except errors.PjrtError:
            with _ONREADY_LOCK:
                _ONREADY.pop(key, None)
            raise

    def consume(self) -> None:
        """Await then destroy -- the common one-shot pattern."""
        try:
            self.result()
        finally:
            self.close()


class Memory:
    """A memory space: where a buffer lives, not just which device owns it.

    Accelerators expose several -- device HBM, pinned host, unpinned host --
    and moving between them is how you stage weights or offload. Bound at 1/6
    before this; `Buffer.copy_to_memory` is the reason the rest matters.
    """

    def __init__(self, plugin: Plugin, ptr: int):
        self._plugin, self._ptr, self._valid = plugin, ptr, True

    def _check(self) -> int:
        if not self._valid:
            raise errors.HandleClosed("Memory escaped its owning block")
        return self._ptr

    def _invalidate(self) -> None:
        self._valid = False

    @property
    def address(self) -> int:
        return self._check()

    @property
    def id(self) -> int:
        a = self._plugin.args("PJRT_Memory_Id_Args", memory=self._check())
        self._plugin.call("PJRT_Memory_Id", a)
        return int(a.id)

    @property
    def kind(self) -> str:
        a = self._plugin.args("PJRT_Memory_Kind_Args", memory=self._check())
        self._plugin.call("PJRT_Memory_Kind", a)
        return ctypes.string_at(a.kind, a.kind_size).decode(errors="replace")

    @property
    def kind_id(self) -> int:
        a = self._plugin.args("PJRT_Memory_Kind_Id_Args", memory=self._check())
        self._plugin.call("PJRT_Memory_Kind_Id", a)
        return int(a.kind_id)

    @property
    def debug_string(self) -> str:
        a = self._plugin.args("PJRT_Memory_DebugString_Args", memory=self._check())
        self._plugin.call("PJRT_Memory_DebugString", a)
        return ctypes.string_at(a.debug_string, a.debug_string_size).decode(errors="replace")

    def __str__(self) -> str:
        a = self._plugin.args("PJRT_Memory_ToString_Args", memory=self._check())
        self._plugin.call("PJRT_Memory_ToString", a)
        return ctypes.string_at(a.to_string, a.to_string_size).decode(errors="replace")

    def addressable_by(self) -> list["Device"]:
        """Devices that can reach this memory space."""
        a = self._plugin.args("PJRT_Memory_AddressableByDevices_Args", memory=self._check())
        self._plugin.call("PJRT_Memory_AddressableByDevices", a)
        ptrs = ctypes.cast(a.devices, ctypes.POINTER(_VOIDP))
        return [Device(self._plugin, int(ptrs[i])) for i in range(a.num_devices)]

    def __repr__(self) -> str:
        try:
            return f"<Memory id={self.id} kind={self.kind!r}>"
        except errors.PjrtError:
            return f"<Memory 0x{self._ptr:x}>"


class Device:
    """A borrowed child of a Client.

    Only reachable inside ``client.device(i)``; invalidated on exit. This is the
    one borrow-style guarantee worth enforcing here: a device handle must not
    outlive the scope that vended it.
    """

    def __init__(self, plugin: Plugin, ptr: int):
        self._plugin, self._ptr, self._valid = plugin, ptr, True

    def _check(self) -> int:
        if not self._valid:
            raise errors.HandleClosed("Device escaped its `with client.device(...)` block")
        return self._ptr

    def _invalidate(self) -> None:
        self._valid = False

    @property
    def address(self) -> int:
        return self._check()

    def _description(self) -> int:
        a = self._plugin.args("PJRT_Device_GetDescription_Args", device=self._check())
        self._plugin.call("PJRT_Device_GetDescription", a)
        return int(a.device_description)

    @property
    def id(self) -> int:
        a = self._plugin.args("PJRT_DeviceDescription_Id_Args",
                              device_description=self._description())
        self._plugin.call("PJRT_DeviceDescription_Id", a)
        return int(a.id)

    @property
    def process_index(self) -> int:
        a = self._plugin.args("PJRT_DeviceDescription_ProcessIndex_Args",
                              device_description=self._description())
        self._plugin.call("PJRT_DeviceDescription_ProcessIndex", a)
        return int(a.process_index)

    @property
    def kind(self) -> str:
        a = self._plugin.args("PJRT_DeviceDescription_Kind_Args",
                              device_description=self._description())
        self._plugin.call("PJRT_DeviceDescription_Kind", a)
        return ctypes.string_at(a.device_kind, a.device_kind_size).decode(errors="replace")

    @property
    def debug_string(self) -> str:
        a = self._plugin.args("PJRT_DeviceDescription_DebugString_Args",
                              device_description=self._description())
        self._plugin.call("PJRT_DeviceDescription_DebugString", a)
        return ctypes.string_at(a.debug_string, a.debug_string_size).decode(errors="replace")

    @property
    def local_hardware_id(self) -> int:
        a = self._plugin.args("PJRT_Device_LocalHardwareId_Args", device=self._check())
        self._plugin.call("PJRT_Device_LocalHardwareId", a)
        return int(a.local_hardware_id)

    @property
    def attributes(self) -> dict[str, Any]:
        """Vendor-specific facts about this device.

        The only route to a TPU's mesh position: libtpu reports ``coords``,
        ``core_on_chip``, ``num_cores`` and (on multi-slice) ``slice_index``
        here and nowhere else, and clients rarely bind it.
        """
        from ._loader import read_named_values
        a = self._plugin.args("PJRT_DeviceDescription_Attributes_Args",
                              device_description=self._description())
        self._plugin.call("PJRT_DeviceDescription_Attributes", a)
        return read_named_values(self._plugin.abi, a.attributes, a.num_attributes)

    @property
    def coords(self) -> tuple[int, ...] | None:
        """N-D chip coordinates within a slice, when the plugin reports them."""
        v = self.attributes.get("coords")
        return tuple(int(x) for x in v) if isinstance(v, (tuple, list)) else None

    def to_string(self) -> str:
        a = self._plugin.args("PJRT_DeviceDescription_ToString_Args",
                              device_description=self._description())
        self._plugin.call("PJRT_DeviceDescription_ToString", a)
        return ctypes.string_at(a.to_string, a.to_string_size).decode(errors="replace")

    def live_attributes(self) -> dict[str, Any]:
        """``PJRT_Device_GetAttributes`` -- the live device, as opposed to
        :attr:`attributes`, which reads its (static) description."""
        from ._loader import read_named_values
        a = self._plugin.args("PJRT_Device_GetAttributes_Args", device=self._check())
        self._plugin.call("PJRT_Device_GetAttributes", a)
        return read_named_values(self._plugin.abi, a.attributes, a.num_attributes)

    def poison_execution(self, launch_id: int, code: int = 13,
                         message: str = "poisoned") -> bool:
        """Fail an in-flight launch on purpose. Fault-injection for resiliency
        tests -- the only reason to call it deliberately."""
        msg = message.encode()
        buf = ctypes.create_string_buffer(msg, len(msg))
        a = self._plugin.args("PJRT_Device_PoisonExecution_Args", device=self._check(),
                              launch_id=launch_id, error_code=code,
                              error_message=ctypes.cast(buf, _VOIDP),
                              error_message_size=len(msg))
        self._plugin.call("PJRT_Device_PoisonExecution", a)
        return bool(a.success)

    def create_async_tracking_event(self, description: str = "") -> "AsyncTrackingEvent":
        d = description.encode()
        buf = ctypes.create_string_buffer(d, len(d)) if d else None
        a = self._plugin.args(
            "PJRT_Device_CreateAsyncTrackingEvent_Args", device=self._check(),
            description=ctypes.cast(buf, _VOIDP) if buf else None,
            description_size=len(d))
        self._plugin.call("PJRT_Device_CreateAsyncTrackingEvent", a)
        return AsyncTrackingEvent(self._plugin, a.event)

    def memories(self) -> list[Memory]:
        """Memory spaces this device can address."""
        a = self._plugin.args("PJRT_Device_AddressableMemories_Args", device=self._check())
        self._plugin.call("PJRT_Device_AddressableMemories", a)
        ptrs = ctypes.cast(a.memories, ctypes.POINTER(_VOIDP))
        return [Memory(self._plugin, int(ptrs[i])) for i in range(a.num_memories)]

    def default_memory(self) -> Memory:
        a = self._plugin.args("PJRT_Device_DefaultMemory_Args", device=self._check())
        self._plugin.call("PJRT_Device_DefaultMemory", a)
        return Memory(self._plugin, a.memory)

    def memory_stats(self) -> dict[str, int]:
        """``PJRT_Device_MemoryStats`` -- bound by no source repo, despite all
        three suffering catastrophic OOMs on this exact box."""
        a = self._plugin.args("PJRT_Device_MemoryStats_Args", device=self._check())
        self._plugin.call("PJRT_Device_MemoryStats", a)
        out = {"bytes_in_use": int(a.bytes_in_use)}
        for name in ("peak_bytes_in_use", "num_allocs", "largest_alloc_size", "bytes_limit",
                     "bytes_reserved", "peak_bytes_reserved", "bytes_reservable_limit",
                     "largest_free_block_bytes", "pool_bytes", "peak_pool_bytes"):
            if getattr(a, f"{name}_is_set", False):
                out[name] = int(getattr(a, name))
        return out

    def clear_memory_stats(self) -> None:
        """Reset peak counters, e.g. between phases of a run."""
        a = self._plugin.args("PJRT_Device_ClearMemoryStats_Args", device=self._check())
        self._plugin.call("PJRT_Device_ClearMemoryStats", a)


class Buffer(_Owned, Generic[DT]):
    """A device buffer. The ``DT`` parameter is erased at runtime and exists
    only so a type checker can catch dtype mixups."""

    _destroy_fn = "PJRT_Buffer_Destroy"
    _destroy_args = "PJRT_Buffer_Destroy_Args"
    _arg_field = "buffer"

    def __init__(self, plugin: Plugin, ptr: int, *, keepalive: Any = None):
        super().__init__(plugin, ptr)
        self._keepalive = keepalive
        self._fulfill_cb = 0

    @property
    def element_type(self) -> int:
        a = self._plugin.args("PJRT_Buffer_ElementType_Args", buffer=self._check())
        self._plugin.call("PJRT_Buffer_ElementType", a)
        return int(a.type)

    @property
    def dimensions(self) -> tuple[int, ...]:
        a = self._plugin.args("PJRT_Buffer_Dimensions_Args", buffer=self._check())
        self._plugin.call("PJRT_Buffer_Dimensions", a)
        if not a.dims or a.num_dims == 0:
            return ()
        return tuple(ctypes.cast(a.dims, ctypes.POINTER(_I64))[i] for i in range(a.num_dims))

    @property
    def nbytes(self) -> int:
        a = self._plugin.args("PJRT_Buffer_OnDeviceSizeInBytes_Args", buffer=self._check())
        self._plugin.call("PJRT_Buffer_OnDeviceSizeInBytes", a)
        return int(a.on_device_size_in_bytes)

    @property
    def is_deleted(self) -> bool:
        """Whether the runtime has released this buffer's device memory.

        The authoritative donation signal. Comparing ``PJRT_Buffer*`` handles
        against the output pointers looks right and is not: on both plugins
        here handles never alias -- the *device* pointer does, and the input
        ``IsDeleted``. Measured, not assumed.
        """
        a = self._plugin.args("PJRT_Buffer_IsDeleted_Args", buffer=self._ptr)
        self._plugin.call("PJRT_Buffer_IsDeleted", a)
        return bool(a.is_deleted)

    def delete(self) -> None:
        """Release device memory now, keeping the handle valid.

        Distinct from close(): on a 128 GB unified-memory box this is the
        difference between finishing and OOMing.
        """
        a = self._plugin.args("PJRT_Buffer_Delete_Args", buffer=self._check())
        self._plugin.call("PJRT_Buffer_Delete", a)
        self._mark_donated()

    def device_pointer(self) -> int:
        a = self._plugin.args("PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args",
                              buffer=self._check())
        self._plugin.call("PJRT_Buffer_OpaqueDeviceMemoryDataPointer", a)
        return int(a.device_memory_ptr or 0)

    def __dlpack__(self, stream: Any = None, **kwargs):
        """Export for zero-copy consumption by torch / jax / numpy / cupy."""
        from .dlpack import buffer_to_dlpack
        return buffer_to_dlpack(self, stream=stream)

    def __dlpack_device__(self) -> tuple[int, int]:
        from .dlpack import buffer_dlpack_device
        return buffer_dlpack_device(self)

    def memory(self) -> Memory:
        """The memory space this buffer occupies."""
        a = self._plugin.args("PJRT_Buffer_Memory_Args", buffer=self._check())
        self._plugin.call("PJRT_Buffer_Memory", a)
        return Memory(self._plugin, a.memory)

    def device(self) -> Device:
        a = self._plugin.args("PJRT_Buffer_Device_Args", buffer=self._check())
        self._plugin.call("PJRT_Buffer_Device", a)
        return Device(self._plugin, a.device)

    def is_on_cpu(self) -> bool:
        a = self._plugin.args("PJRT_Buffer_IsOnCpu_Args", buffer=self._check())
        self._plugin.call("PJRT_Buffer_IsOnCpu", a)
        return bool(a.is_on_cpu)

    def copy_to_memory(self, memory: Memory) -> "Buffer[DT]":
        """Move to another memory space -- staging, offload, pinned host."""
        a = self._plugin.args("PJRT_Buffer_CopyToMemory_Args", buffer=self._check(),
                              dst_memory=memory.address)
        self._plugin.call("PJRT_Buffer_CopyToMemory", a)
        return Buffer(self._plugin, a.dst_buffer)

    def copy_to_device(self, device: Device) -> "Buffer[DT]":
        a = self._plugin.args("PJRT_Buffer_CopyToDevice_Args", buffer=self._check(),
                              dst_device=device.address)
        self._plugin.call("PJRT_Buffer_CopyToDevice", a)
        return Buffer(self._plugin, a.dst_buffer)

    @property
    def unpadded_dimensions(self) -> tuple[int, ...]:
        """Logical dims, which differ from :attr:`dimensions` under dynamic shapes."""
        a = self._plugin.args("PJRT_Buffer_UnpaddedDimensions_Args", buffer=self._check())
        self._plugin.call("PJRT_Buffer_UnpaddedDimensions", a)
        if not a.unpadded_dims:
            return ()
        p64 = ctypes.cast(a.unpadded_dims, ctypes.POINTER(_I64))
        return tuple(int(p64[i]) for i in range(a.num_dims))

    @property
    def dynamic_dimension_indices(self) -> tuple[int, ...]:
        a = self._plugin.args("PJRT_Buffer_DynamicDimensionIndices_Args", buffer=self._check())
        self._plugin.call("PJRT_Buffer_DynamicDimensionIndices", a)
        if not a.dynamic_dim_indices:
            return ()
        pz = ctypes.cast(a.dynamic_dim_indices, ctypes.POINTER(ctypes.c_size_t))
        return tuple(int(pz[i]) for i in range(a.num_dynamic_dims))

    def unsafe_pointer(self) -> int:
        a = self._plugin.args("PJRT_Buffer_UnsafePointer_Args", buffer=self._check())
        self._plugin.call("PJRT_Buffer_UnsafePointer", a)
        return int(a.buffer_pointer)

    def copy_raw_to_host(self, out: Any, *, offset: int = 0,
                         size: int | None = None) -> None:
        """Read a byte range without allocating for the whole buffer."""
        view = _byte_view(out)
        n = view.nbytes if size is None else size
        a = self._plugin.args(
            "PJRT_Buffer_CopyRawToHost_Args", buffer=self._check(),
            dst=ctypes.addressof(ctypes.c_char.from_buffer(view)),
            offset=offset, transfer_size=n)
        self._plugin.call("PJRT_Buffer_CopyRawToHost", a)
        if a.event:
            Event(self._plugin, a.event).consume()

    def bitcast(self, dtype: type[DT], dims: Sequence[int] | None = None) -> "Buffer[DT]":
        """Reinterpret without copying."""
        d = tuple(self.dimensions) if dims is None else tuple(dims)
        arr = (_I64 * max(len(d), 1))(*d)
        a = self._plugin.args("PJRT_Buffer_Bitcast_Args", buffer=self._check(),
                              element_type=dtype.code,
                              dims=ctypes.cast(arr, _VOIDP), num_dims=len(d))
        self._plugin.call("PJRT_Buffer_Bitcast", a)
        return Buffer(self._plugin, a.out_buffer)

    def astype(self, dtype: type[DT]) -> "Buffer[DT]":
        """Refine an erased buffer, checking the runtime tag.

        Without GADTs this is a runtime check in every host language -- Swift
        reaches the same conclusion.
        """
        got = self.element_type
        if got != dtype.code:
            from .typing import by_code
            have = by_code(got)
            raise errors.InvalidArgument(
                f"buffer holds {have.name if have else got}, not {dtype.name}")
        return self  # type: ignore[return-value]

    def to_host(self, out: Any = None) -> Any:
        """Copy to host. ``out`` is any writable buffer-protocol object."""
        p = self._check()
        if out is None:
            out = bytearray(self.nbytes)
        view = _byte_view(out)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(view))
        a = self._plugin.args("PJRT_Buffer_ToHostBuffer_Args", src=p, dst=addr,
                              dst_size=view.nbytes)
        self._plugin.call("PJRT_Buffer_ToHostBuffer", a)
        if a.event:
            Event(self._plugin, a.event).consume()
        return out


class CopyToDeviceStream(_Owned):
    """The receiving end of a ``recv`` op's streamed transfer.

    You do not construct one: XLA hands it to a ``recv`` callback registered in
    ``PJRT_ExecuteOptions``. Exposed so that when a program *does* use send/recv
    the chunks can be consumed, rather than the family sitting at 0/5.
    """

    _destroy_fn = "PJRT_CopyToDeviceStream_Destroy"
    _destroy_args = "PJRT_CopyToDeviceStream_Destroy_Args"
    _arg_field = "stream"

    @property
    def total_bytes(self) -> int:
        a = self._plugin.args("PJRT_CopyToDeviceStream_TotalBytes_Args", stream=self._check())
        self._plugin.call("PJRT_CopyToDeviceStream_TotalBytes", a)
        return int(a.total_bytes)

    @property
    def granule_size(self) -> int:
        a = self._plugin.args("PJRT_CopyToDeviceStream_GranuleSize_Args", stream=self._check())
        self._plugin.call("PJRT_CopyToDeviceStream_GranuleSize", a)
        return int(a.granule_size_in_bytes)

    @property
    def current_bytes(self) -> int:
        a = self._plugin.args("PJRT_CopyToDeviceStream_CurrentBytes_Args", stream=self._check())
        self._plugin.call("PJRT_CopyToDeviceStream_CurrentBytes", a)
        return int(a.current_bytes)

    def add_chunk(self, chunk_ptr: int) -> "Event":
        """Feed one ``PJRT_Chunk*`` in; the returned event completes the copy."""
        a = self._plugin.args("PJRT_CopyToDeviceStream_AddChunk_Args",
                              stream=self._check(), chunk=chunk_ptr)
        self._plugin.call("PJRT_CopyToDeviceStream_AddChunk", a)
        return Event(self._plugin, a.transfer_complete)


class AsyncTrackingEvent(_Owned):
    """A device-created event used to track asynchronous work."""

    _destroy_fn = "PJRT_AsyncTrackingEvent_Destroy"
    _destroy_args = "PJRT_AsyncTrackingEvent_Destroy_Args"
    _arg_field = "event"


class ExecuteContext(_Owned):
    """Per-execution state shared with FFI handlers.

    Required by the FFI extension's ``user_data_add``, i.e. by any *stateful*
    handler. Bound by no source repo.
    """

    _destroy_fn = "PJRT_ExecuteContext_Destroy"
    _destroy_args = "PJRT_ExecuteContext_Destroy_Args"
    _arg_field = "context"

    @classmethod
    def create(cls, plugin: Plugin) -> "ExecuteContext":
        a = plugin.args("PJRT_ExecuteContext_Create_Args")
        plugin.call("PJRT_ExecuteContext_Create", a)
        return cls(plugin, a.context)


class Executable(_Owned):
    _destroy_fn = "PJRT_LoadedExecutable_Destroy"
    _destroy_args = "PJRT_LoadedExecutable_Destroy_Args"
    _arg_field = "executable"

    def __init__(self, plugin: Plugin, ptr: int, client: "Client"):
        super().__init__(plugin, ptr)
        self._client = client
        self._inner: int | None = None
        self._num_outputs: int | None = None
        #: How many input buffers the runtime has consumed via donation.
        #: Zero when you expected donation means it silently did not happen.
        self.donate_alias_count = 0

    def _executable(self) -> int:
        """The unloaded ``PJRT_Executable*``, cached for this object's life."""
        if self._inner is None:
            e = self._plugin.args("PJRT_LoadedExecutable_GetExecutable_Args",
                                  loaded_executable=self._check())
            self._plugin.call("PJRT_LoadedExecutable_GetExecutable", e)
            self._inner = int(e.executable)
        return self._inner

    def close(self) -> None:
        if self._state is State.LIVE and self._inner is not None:
            self._plugin.call("PJRT_Executable_Destroy",
                              self._plugin.args("PJRT_Executable_Destroy_Args",
                                                executable=self._inner))
            self._inner = None
        super().close()

    @property
    def num_outputs(self) -> int:
        if self._num_outputs is None:
            a = self._plugin.args("PJRT_Executable_NumOutputs_Args",
                                  executable=self._executable())
            self._plugin.call("PJRT_Executable_NumOutputs", a)
            self._num_outputs = int(a.num_outputs)
        return self._num_outputs

    @property
    def num_replicas(self) -> int:
        a = self._plugin.args("PJRT_Executable_NumReplicas_Args",
                              executable=self._executable())
        self._plugin.call("PJRT_Executable_NumReplicas", a)
        return int(a.num_replicas)

    @property
    def num_partitions(self) -> int:
        a = self._plugin.args("PJRT_Executable_NumPartitions_Args",
                              executable=self._executable())
        self._plugin.call("PJRT_Executable_NumPartitions", a)
        return int(a.num_partitions)

    @property
    def addressable_device_count(self) -> int:
        """How many devices this executable expects per launch."""
        a = self._plugin.args("PJRT_LoadedExecutable_AddressableDevices_Args",
                              executable=self._check())
        self._plugin.call("PJRT_LoadedExecutable_AddressableDevices", a)
        return int(a.num_addressable_devices)

    def addressable_devices(self) -> list[int]:
        """Raw device pointers, in launch order."""
        a = self._plugin.args("PJRT_LoadedExecutable_AddressableDevices_Args",
                              executable=self._check())
        self._plugin.call("PJRT_LoadedExecutable_AddressableDevices", a)
        ptrs = ctypes.cast(a.addressable_devices, ctypes.POINTER(_VOIDP))
        return [int(ptrs[i]) for i in range(a.num_addressable_devices)]

    def device_assignment(self) -> list[tuple[int, int]]:
        """``[(replica, partition)]`` per addressable device, in launch order."""
        a = self._plugin.args("PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args",
                              executable=self._check())
        self._plugin.call("PJRT_LoadedExecutable_AddressableDeviceLogicalIds", a)
        ids = ctypes.cast(a.addressable_device_logical_ids,
                          ctypes.POINTER(self._plugin.abi.PJRT_LogicalDeviceIds))
        n = int(a.num_addressable_device_logical_ids)
        return [(int(ids[i].replica), int(ids[i].partition)) for i in range(n)]

    def fingerprint(self) -> bytes:
        """A cache key the *plugin* computed -- better than hashing MLIR text."""
        a = self._plugin.args("PJRT_Executable_Fingerprint_Args",
                              executable=self._executable())
        self._plugin.call("PJRT_Executable_Fingerprint", a)
        if not a.executable_fingerprint:
            return b""
        return ctypes.string_at(a.executable_fingerprint, a.executable_fingerprint_size)

    def serialize(self) -> bytes:
        """The compiled executable, ready to reload without recompiling."""
        a = self._plugin.args("PJRT_Executable_Serialize_Args",
                              executable=self._executable())
        self._plugin.call("PJRT_Executable_Serialize", a)
        try:
            return ctypes.string_at(a.serialized_bytes, a.serialized_bytes_size)
        finally:
            if a.serialized_executable and a.serialized_executable_deleter:
                ctypes.CFUNCTYPE(None, _VOIDP)(a.serialized_executable_deleter)(
                    a.serialized_executable)

    def output_types(self) -> list[int]:
        a = self._plugin.args("PJRT_Executable_OutputElementTypes_Args",
                              executable=self._executable())
        self._plugin.call("PJRT_Executable_OutputElementTypes", a)
        p32 = ctypes.cast(a.output_types, ctypes.POINTER(ctypes.c_int32))
        return [int(p32[i]) for i in range(a.num_output_types)]

    def to_artifact(self, *, source: bytes = b"", compile_options: bytes = b"",
                    metadata: dict[str, Any] | None = None):
        """Package this executable for reuse on a compatible machine."""
        from .artifact import AbiVersion, Artifact, sha256_hex
        abi_proto = b""
        if (av := AbiVersion.probe(self._plugin)) is not None:
            try:
                abi_proto = av.executable_proto(self._executable())
            except errors.PjrtError:
                pass
        try:
            outs = self.output_types()
        except errors.PjrtError:
            outs = []
        return Artifact(
            executable=self.serialize(),
            platform=self._client.platform_name,
            api_version=self._plugin.api_version,
            xla_version=self._plugin.xla_version,
            fingerprint=self.fingerprint().hex(),
            abi_proto=abi_proto.hex(),
            compile_options=compile_options.hex(),
            source=source,
            source_sha256=sha256_hex(source) if source else "",
            output_types=outs,
            metadata=metadata or {},
        )

    def abi_compatibility(self) -> str | None:
        """``None`` if the plugin considers this executable runnable here."""
        from .artifact import AbiVersion
        av = AbiVersion.probe(self._plugin)
        if av is None:
            return None
        return av.check(self._client.address, self._executable())

    @property
    def name(self) -> str:
        a = self._plugin.args("PJRT_Executable_Name_Args", executable=self._executable())
        self._plugin.call("PJRT_Executable_Name", a)
        return ctypes.string_at(a.executable_name, a.executable_name_size).decode()

    @property
    def code_size_bytes(self) -> int:
        a = self._plugin.args("PJRT_Executable_SizeOfGeneratedCodeInBytes_Args",
                              executable=self._executable())
        self._plugin.call("PJRT_Executable_SizeOfGeneratedCodeInBytes", a)
        return int(a.size_in_bytes)

    def _memory_kinds(self, which: str) -> list[str]:
        name = ("PJRT_Executable_OutputMemoryKinds_Args" if which == "outputs"
                else "PJRT_Executable_ParameterMemoryKinds_Args")
        fn = ("PJRT_Executable_OutputMemoryKinds" if which == "outputs"
              else "PJRT_Executable_ParameterMemoryKinds")
        a = self._plugin.args(name, executable=self._executable())
        self._plugin.call(fn, a)
        n = int(a.num_outputs if which == "outputs" else a.num_parameters)
        kinds = ctypes.cast(a.memory_kinds, ctypes.POINTER(_VOIDP))
        sizes = ctypes.cast(a.memory_kind_sizes, ctypes.POINTER(ctypes.c_size_t))
        return [ctypes.string_at(kinds[i], sizes[i]).decode(errors="replace")
                for i in range(n)]

    def output_memory_kinds(self) -> list[str]:
        """Which memory space each output lands in."""
        return self._memory_kinds("outputs")

    def parameter_memory_kinds(self) -> list[str]:
        return self._memory_kinds("parameters")

    def compile_options(self) -> bytes:
        """The serialized CompileOptionsProto the plugin actually applied."""
        a = self._plugin.args("PJRT_Executable_GetCompileOptions_Args",
                              executable=self._executable())
        self._plugin.call("PJRT_Executable_GetCompileOptions", a)
        return ctypes.string_at(a.serialized_bytes, a.serialized_bytes_size)

    def device_assignment_proto(self) -> bytes:
        a = self._plugin.args("PJRT_LoadedExecutable_GetDeviceAssignment_Args",
                              executable=self._check())
        self._plugin.call("PJRT_LoadedExecutable_GetDeviceAssignment", a)
        return ctypes.string_at(a.serialized_bytes, a.serialized_bytes_size)

    def loaded_fingerprint(self) -> bytes:
        a = self._plugin.args("PJRT_LoadedExecutable_Fingerprint_Args",
                              executable=self._check())
        self._plugin.call("PJRT_LoadedExecutable_Fingerprint", a)
        if not a.executable_fingerprint:
            return b""
        return ctypes.string_at(a.executable_fingerprint, a.executable_fingerprint_size)

    def release_device_memory(self) -> None:
        """Free device resources while keeping the handle valid."""
        self._plugin.call("PJRT_LoadedExecutable_Delete",
                          self._plugin.args("PJRT_LoadedExecutable_Delete_Args",
                                            executable=self._check()))

    @property
    def is_deleted(self) -> bool:
        a = self._plugin.args("PJRT_LoadedExecutable_IsDeleted_Args", executable=self._check())
        self._plugin.call("PJRT_LoadedExecutable_IsDeleted", a)
        return bool(a.is_deleted)

    def compiled_memory_stats(self) -> dict[str, int]:
        """What this program will need, from the compiler -- before you run it.

        Bound by no source repo, so all three sized workloads by trial and OOM.
        """
        a = self._plugin.args("PJRT_Executable_GetCompiledMemoryStats_Args",
                              executable=self._executable())
        self._plugin.call("PJRT_Executable_GetCompiledMemoryStats", a)
        return {f: int(getattr(a, f)) for f, _ in a._fields_
                if f.endswith("_in_bytes") or f in ("total_allocation_bytes",
                                                    "indefinite_allocations")}

    def cost_analysis(self) -> dict[str, Any]:
        """The plugin's own FLOP/byte estimates.

        Exposing this is in scope; *building* a roofline model is not. Hand-
        transcribed datasheet numbers are how you get an uncalibrated model;
        these come from the plugin.
        """
        from ._loader import read_named_values
        a = self._plugin.args("PJRT_Executable_GetCostAnalysis_Args",
                              executable=self._executable())
        self._plugin.call("PJRT_Executable_GetCostAnalysis", a)
        return read_named_values(self._plugin.abi, a.properties, a.num_properties)

    def execute_sharded(self, arguments: Sequence[Sequence[Buffer]], *,
                        launch_id: int = 0,
                        non_donatable: Sequence[int] = (),
                        context: "ExecuteContext | None" = None) -> list[list[Buffer]]:
        """Execute across N devices. ``arguments[device][arg]``.

        This is the native shape (Shape 2). ``__call__`` is sugar over it.

        ``launch_id`` identifies this execution as part of a multi-device launch;
        the header says the runtime uses it to *detect* hosts launching in
        different orders, so it is worth actually writing.

        Donation is declared by the *program* (``tf.aliasing_output`` on a func
        argument, which is what jax emits for ``donate_argnums``);
        ``non_donatable`` suppresses it per call. Afterwards we ask the runtime
        which inputs it actually consumed and mark those handles, so a later use
        raises instead of being a use-after-free.
        """
        p = self._check()
        num_devices = len(arguments)
        if num_devices == 0:
            raise errors.InvalidArgument("no devices")
        want = self.addressable_device_count
        if num_devices != want:
            raise errors.InvalidArgument(
                f"executable expects {want} device(s) per launch, got {num_devices}. "
                f"It was compiled with num_replicas={self.num_replicas}, "
                f"num_partitions={self.num_partitions}.")
        num_args = len(arguments[0])
        if any(len(row) != num_args for row in arguments):
            raise errors.InvalidArgument("every device must receive the same number of arguments")
        num_outputs = self.num_outputs

        rows = [(_VOIDP * num_args)(*[b._check() for b in row]) for row in arguments]
        arglists = (_VOIDP * num_devices)(*[ctypes.cast(r, _VOIDP) for r in rows])
        outrows = [(_VOIDP * num_outputs)() for _ in range(num_devices)]
        outlists = (_VOIDP * num_devices)(*[ctypes.cast(r, _VOIDP) for r in outrows])
        events = (_VOIDP * num_devices)()

        opts = self._plugin.args("PJRT_ExecuteOptions", launch_id=launch_id)
        if context is not None:
            opts.context = context.address
        _keep_nd = None
        if non_donatable:
            _keep_nd = (_I64 * len(non_donatable))(*non_donatable)
            opts.non_donatable_input_indices = ctypes.cast(_keep_nd, _VOIDP)
            opts.num_non_donatable_input_indices = len(non_donatable)
        a = self._plugin.args(
            "PJRT_LoadedExecutable_Execute_Args", executable=p,
            options=ctypes.addressof(opts),
            argument_lists=ctypes.cast(arglists, _VOIDP),
            num_devices=num_devices, num_args=num_args,
            output_lists=ctypes.cast(outlists, _VOIDP),
            device_complete_events=ctypes.cast(events, _VOIDP),
        )
        # XLA may dispatch FFI handlers synchronously inside Execute.
        with self._client.diagnose_allocation():
            self._plugin.call("PJRT_LoadedExecutable_Execute", a, reentrant=True)

        # Await completion BEFORE touching any input:
        # destroying a donated input while the kernels are still reading it is a
        # use-after-free that shows up non-deterministically after many steps.
        for i in range(num_devices):
            if events[i]:
                Event(self._plugin, events[i]).consume()

        for row in arguments:
            for buf in row:
                if buf._state is State.LIVE and buf.is_deleted:
                    buf._mark_donated()
                    self.donate_alias_count += 1

        del _keep_nd
        return [[Buffer(self._plugin, outrows[d][j]) for j in range(num_outputs)]
                for d in range(num_devices)]

    def __call__(self, *arguments: Buffer, non_donatable: Sequence[int] = ()) -> list[Buffer]:
        """Single-device convenience over :meth:`execute_sharded`."""
        return self.execute_sharded([list(arguments)], non_donatable=non_donatable)[0]


class Client(_Owned):
    _destroy_fn = "PJRT_Client_Destroy"
    _destroy_args = "PJRT_Client_Destroy_Args"
    _arg_field = "client"

    #: Defaults that stop the GB10 hanging. On unified memory the CUDA plugin's
    #: own defaults (preallocate=true, fraction=0.75) make each client reserve
    #: ~98 GB of system RAM, which hard-hangs the machine rather than failing.
    GPU_DEFAULTS = {"preallocate": False, "memory_fraction": 0.5}

    _live_accelerators = 0
    _live_lock = threading.Lock()

    def __init__(self, plugin: Plugin, *, options: dict[str, Any] | None = None,
                 allow_multiple: bool = False, kv_store: Any = None):
        self._plugin = plugin
        accel = plugin.is_accelerator
        # Allocator options are GPU-family only: a CPU plugin rejects unknown
        # create-options and a TPU plugin has no such knobs. The live-client
        # guard, by contrast, applies to any accelerator.
        if options is None:
            options = dict(self.GPU_DEFAULTS) if plugin.is_gpu else {}
        opts = dict(options)

        self._counted = False
        if accel and not allow_multiple:
            with Client._live_lock:
                if Client._live_accelerators:
                    raise errors.ResourceExhausted(
                        "an accelerator PJRT client is already live in this process. "
                        "Each client reserves its own slice of device memory, and on "
                        "unified-memory hardware a second one can hang the machine. "
                        "Close the first, or pass allow_multiple=True."
                    )

        # Multi-controller SPMD: without these callbacks a client cannot
        # rendezvous with its peers.
        self._kv = None
        named, keep = _named_values(plugin, opts)
        a = plugin.args("PJRT_Client_Create_Args")
        if kv_store is not None:
            from .kv import KvBridge
            self._kv = KvBridge(plugin, kv_store)
            self._kv.apply(a)
        if named is not None:
            a.create_options = ctypes.cast(named, _VOIDP)
            a.num_options = len(opts)
        plugin.call("PJRT_Client_Create", a)
        super().__init__(plugin, a.client)
        self._keep = keep
        self._is_accelerator = accel
        self._is_gpu = plugin.is_gpu
        self._create_options = opts
        if accel:
            with Client._live_lock:
                Client._live_accelerators += 1
                self._counted = True

    def close(self) -> None:
        if self._state is State.LIVE and getattr(self, "_counted", False):
            with Client._live_lock:
                Client._live_accelerators = max(0, Client._live_accelerators - 1)
            self._counted = False
        super().close()

    @classmethod
    def create(cls, plugin: Plugin | str | None = None, *,
               options: dict[str, Any] | None = None,
               allow_multiple: bool = False, kv_store: Any = None) -> "Client":
        if not isinstance(plugin, Plugin):
            plugin = Plugin(plugin)
        plugin.initialize()
        return cls(plugin, options=options, allow_multiple=allow_multiple,
                   kv_store=kv_store)

    @property
    def kv_calls(self) -> dict[str, int]:
        """How many times the plugin used the rendezvous store."""
        return dict(self._kv.calls) if self._kv else {}

    # -- diagnostics -------------------------------------------------------

    def memory_summary(self) -> str:
        """Per-device allocator state, formatted for a human."""
        parts = []
        for i, ptr in enumerate(self._addressable()):
            d = Device(self._plugin, ptr)
            try:
                stats = d.memory_stats()
            except errors.PjrtError:
                continue
            fields = [f"{k}={_fmt_bytes(stats[k])}" for k in _OOM_KEYS
                      if k in stats and k not in ("num_allocs",)]
            if "num_allocs" in stats:
                fields.append(f"num_allocs={stats['num_allocs']}")
            if fields:
                parts.append(f"device {i}: " + ", ".join(fields))
        return "; ".join(parts)

    @contextmanager
    def diagnose_allocation(self) -> Iterator[None]:
        """Attach allocator state to any allocation failure raised inside.

        All three source projects hit catastrophic OOMs on this hardware and
        debugged them blind: none binds PJRT_Device_MemoryStats.
        A bare "RESOURCE_EXHAUSTED" is not an actionable message.
        """
        try:
            yield
        except errors.ResourceExhausted as e:
            try:
                summary = self.memory_summary()
            except Exception:
                summary = ""
            if not summary:
                raise
            hint = ""
            if getattr(self, "_is_gpu", False):
                opts = getattr(self, "_create_options", {})
                tips = []
                if opts.get("preallocate", True):
                    tips.append("set preallocate=False")
                frac = opts.get("memory_fraction")
                if frac is None:
                    tips.append("cap memory_fraction")
                elif frac > 0.05:
                    tips.append(f"lower memory_fraction (currently {frac})")
                else:
                    tips.append("this client is already capped; the workload needs more memory "
                                "than it was given")
                hint = (f"\n  client create-options: {opts or '{}'}"
                        f"\n  hint: {'; '.join(tips)}")
            raise errors.ResourceExhausted(
                f"{e.message}\n  device memory: {summary}{hint}", e.code) from e

    @property
    def platform_name(self) -> str:
        a = self._plugin.args("PJRT_Client_PlatformName_Args", client=self._check())
        self._plugin.call("PJRT_Client_PlatformName", a)
        return ctypes.string_at(a.platform_name, a.platform_name_size).decode()

    @property
    def process_index(self) -> int:
        a = self._plugin.args("PJRT_Client_ProcessIndex_Args", client=self._check())
        self._plugin.call("PJRT_Client_ProcessIndex", a)
        return int(a.process_index)

    def _addressable(self) -> tuple[int, ...]:
        a = self._plugin.args("PJRT_Client_AddressableDevices_Args", client=self._check())
        self._plugin.call("PJRT_Client_AddressableDevices", a)
        ptrs = ctypes.cast(a.addressable_devices, ctypes.POINTER(_VOIDP))
        return tuple(int(ptrs[i]) for i in range(a.num_addressable_devices))

    @property
    def device_count(self) -> int:
        return len(self._addressable())

    @contextmanager
    def device(self, index: int = 0) -> Iterator[Device]:
        """Borrow a device for the duration of the block."""
        d = Device(self._plugin, self._addressable()[index])
        try:
            yield d
        finally:
            d._invalidate()

    @contextmanager
    def devices(self) -> Iterator[list[Device]]:
        """Borrow every addressable device for the duration of the block."""
        ds = [Device(self._plugin, ptr) for ptr in self._addressable()]
        try:
            yield ds
        finally:
            for d in ds:
                d._invalidate()

    def default_device_assignment(self, num_replicas: int,
                                  num_partitions: int = 1) -> list[int]:
        """The plugin's own device ids for a ``(replicas, partitions)`` mesh."""
        n = num_replicas * num_partitions
        ids = (ctypes.c_int32 * n)()
        a = self._plugin.args(
            "PJRT_Client_DefaultDeviceAssignment_Args", client=self._check(),
            num_replicas=num_replicas, num_partitions=num_partitions, num_devices=n,
            default_assignment=ctypes.cast(ids, _VOIDP), default_assignment_size=n)
        self._plugin.call("PJRT_Client_DefaultDeviceAssignment", a)
        return [int(x) for x in ids]

    def from_dlpack(self, obj: Any, *, device: "Device | None" = None) -> Buffer:
        """Adopt another framework's device buffer with no host copy.

        The import half of DLPack, which no source repo implements.
        """
        from .dlpack import buffer_from_dlpack
        return buffer_from_dlpack(self, obj, device=device)

    def typed_buffer(self, dtype: type[DT], data: Any, dims: Sequence[int],
                     device: Device) -> "Buffer[DT]":
        """``buffer_from_host`` with the dtype in the static type as well."""
        return self.buffer_from_host(data, dtype.code, dims, device)

    def lookup_device(self, device_id: int, *, addressable: bool = False) -> Device:
        """Find a device by its global id."""
        name = ("PJRT_Client_LookupAddressableDevice_Args" if addressable
                else "PJRT_Client_LookupDevice_Args")
        fn = ("PJRT_Client_LookupAddressableDevice" if addressable
              else "PJRT_Client_LookupDevice")
        key = "local_hardware_id" if addressable else "id"
        a = self._plugin.args(name, client=self._check(), **{key: device_id})
        self._plugin.call(fn, a)
        return Device(self._plugin, a.device)

    def uninitialized_buffer(self, dtype: int, dims: Sequence[int],
                             memory: "Memory") -> Buffer:
        """Allocate device space without writing to it -- an output slot."""
        _require_dtype(dtype, "uninitialized_buffer")
        arr = (_I64 * max(len(dims), 1))(*dims)
        a = self._plugin.args(
            "PJRT_Client_CreateUninitializedBuffer_Args", client=self._check(),
            shape_dims=ctypes.cast(arr, _VOIDP), shape_num_dims=len(dims),
            shape_element_type=dtype, memory=memory.address)
        with self.diagnose_allocation():
            self._plugin.call("PJRT_Client_CreateUninitializedBuffer", a)
        return Buffer(self._plugin, a.buffer)

    def error_buffer(self, memory: "Memory", code: int, message: str, *,
                     dtype: int, dims: Sequence[int]) -> Buffer:
        """A buffer that *is* a failure, so a consumer sees the error rather
        than blocking or reading garbage.

        ``dtype`` and ``dims`` are required: XLA's C-API layer ``CHECK``-fails
        -- aborting the process, not returning an error -- when the element
        type is INVALID, so a zeroed shape here kills the interpreter.
        """
        _require_dtype(dtype, "error_buffer")
        msg = message.encode()
        buf = ctypes.create_string_buffer(msg, len(msg))
        arr = (_I64 * max(len(dims), 1))(*dims)
        a = self._plugin.args(
            "PJRT_Client_CreateErrorBuffer_Args", client=self._check(),
            error_code=code, error_message=ctypes.cast(buf, _VOIDP),
            error_message_size=len(msg), memory=memory.address,
            shape_dims=ctypes.cast(arr, _VOIDP), shape_num_dims=len(dims),
            shape_element_type=dtype)
        _lay, _keep = _dense_layout(self._plugin.abi, dims)
        a.shape_layout = ctypes.addressof(_lay)
        self._plugin.call("PJRT_Client_CreateErrorBuffer", a)
        del _keep
        return Buffer(self._plugin, a.buffer)

    def alias_buffer(self, memory: "Memory", dtype: int, dims: Sequence[int]) -> Buffer:
        """A promise: a buffer whose contents arrive later via
        :meth:`fulfill_alias_buffer`."""
        _require_dtype(dtype, "alias_buffer")
        arr = (_I64 * max(len(dims), 1))(*dims)
        a = self._plugin.args(
            "PJRT_Client_CreateAliasBuffer_Args", client=self._check(),
            memory=memory.address, shape_dims=ctypes.cast(arr, _VOIDP),
            shape_num_dims=len(dims), shape_element_type=dtype)
        _lay, _keep = _dense_layout(self._plugin.abi, dims)
        a.shape_layout = ctypes.addressof(_lay)
        self._plugin.call("PJRT_Client_CreateAliasBuffer", a)
        del _keep
        buf = Buffer(self._plugin, a.alias_buffer)
        # Creation hands back the callback that fulfilment must carry; the
        # promise is only completable through it.
        buf._fulfill_cb = int(a.fulfill_alias_buffer_cb or 0)
        return buf

    def fulfill_alias_buffer(self, alias: Buffer, *, code: int = 0,
                             message: str = "") -> None:
        """Complete a promised buffer, or fail it with ``code``/``message``."""
        cb = getattr(alias, "_fulfill_cb", 0)
        if not cb:
            raise errors.InvalidArgument(
                "this buffer was not created by alias_buffer(), so it carries no "
                "fulfilment callback")
        msg = message.encode()
        buf = ctypes.create_string_buffer(msg, len(msg)) if msg else None
        a = self._plugin.args(
            "PJRT_Client_FulfillAliasBuffer_Args", client=self._check(),
            buffer=alias._check(), status_code=code,
            error_message=ctypes.cast(buf, _VOIDP) if buf else None,
            error_message_size=len(msg), fulfill_alias_buffer_cb=cb)
        self._plugin.call("PJRT_Client_FulfillAliasBuffer", a)

    def dma_map(self, data: Any) -> memoryview:
        """Pin host memory for fast transfer. Keep the returned view alive."""
        view = _byte_view(data)
        a = self._plugin.args(
            "PJRT_Client_DmaMap_Args", client=self._check(),
            data=ctypes.addressof(ctypes.c_char.from_buffer(view)), size=view.nbytes)
        self._plugin.call("PJRT_Client_DmaMap", a)
        return view

    def dma_unmap(self, view: memoryview) -> None:
        a = self._plugin.args(
            "PJRT_Client_DmaUnmap_Args", client=self._check(),
            data=ctypes.addressof(ctypes.c_char.from_buffer(view)))
        self._plugin.call("PJRT_Client_DmaUnmap", a)

    def update_global_process_info(self, infos: Sequence[tuple[int, int, int]]) -> None:
        """Report peer liveness in a multi-process run.

        ``infos`` is ``[(task_id, incarnation_id, state)]``. Part of the
        multi-controller story alongside the KV rendezvous.
        """
        abi = self._plugin.abi
        arr = (abi.PJRT_ProcessInfo * max(len(infos), 1))()
        for i, (task, inc, state) in enumerate(infos):
            arr[i].struct_size = abi.PJRT_ProcessInfo_STRUCT_SIZE
            arr[i].task_id = task
            arr[i].incarnation_id = inc
            arr[i].state = state
        a = self._plugin.args("PJRT_Client_UpdateGlobalProcessInfo_Args",
                              client=self._check(),
                              process_infos=ctypes.cast(arr, _VOIDP),
                              num_process_infos=len(infos))
        self._plugin.call("PJRT_Client_UpdateGlobalProcessInfo", a)

    def async_transfer(self, specs: Sequence[Any], memory: "Memory | None" = None,
                       device: "Device | None" = None):
        """Allocate device buffers now, stream data into them in chunks.

        Pass a memory space, or a device to use its default one.
        """
        from .transfer import AsyncTransfer
        if memory is None:
            if device is None:
                raise errors.InvalidArgument("pass memory= or device=")
            memory = device.default_memory()
        return AsyncTransfer(self, specs, memory)

    def buffers_from_host(self, shards: Sequence[Any], dtype: int,
                          dims: Sequence[int], devices: Sequence[Device]) -> list[Buffer]:
        """Place one shard per device. Buffer placement, not array semantics."""
        if len(shards) != len(devices):
            raise errors.InvalidArgument(
                f"{len(shards)} shard(s) for {len(devices)} device(s)")
        return [self.buffer_from_host(s, dtype, dims, d)
                for s, d in zip(shards, devices)]

    def compile(self, program: str | bytes, *,
                options: CompileOptions | None = None,
                cache: "Any | None" = None) -> Executable:
        """Compile StableHLO. Accepts text or portable-artifact bytes.

        We consume StableHLO; we never produce it.
        """
        code = program.encode() if isinstance(program, str) else bytes(program)
        fmt = b"mlir"
        if options is None:
            # A module carries a partitioner assumption. jax >= 0.11 lowers
            # targeting Shardy and emits `xla.sdy.*` custom calls; handing that
            # to XLA with Shardy disabled fails deep inside the GSPMD
            # partitioner. Match the producer when the caller expressed no
            # preference; an explicit CompileOptions is always honoured as given.
            options = CompileOptions(use_shardy_partitioner=b"sdy." in code)
        opts_bytes = options.encode(self._plugin.abi)
        code_buf = ctypes.create_string_buffer(code, len(code))
        fmt_buf = ctypes.create_string_buffer(fmt, len(fmt))
        opt_buf = ctypes.create_string_buffer(opts_bytes, len(opts_bytes))
        prog = self._plugin.args(
            "PJRT_Program", code=ctypes.cast(code_buf, _VOIDP), code_size=len(code),
            format=ctypes.cast(fmt_buf, _VOIDP), format_size=len(fmt))
        a = self._plugin.args(
            "PJRT_Client_Compile_Args", client=self._check(),
            program=ctypes.addressof(prog),
            compile_options=ctypes.cast(opt_buf, _VOIDP),
            compile_options_size=len(opts_bytes))
        ckey = None
        if cache is not None:
            ckey = cache.key(code, opts_bytes, self.platform_name,
                             self._plugin.xla_version, self._plugin.api_version[0])
            if (hit := cache.load(ckey)) is not None:
                try:
                    return self.deserialize_executable(hit.executable)
                except errors.PjrtError:
                    pass  # stale entry: fall through and recompile
        try:
            self._plugin.call("PJRT_Client_Compile", a)
        except errors.PjrtError as e:
            if "xla.sdy" in e.message and not options.use_shardy_partitioner:
                raise type(e)(
                    f"{e.message}\n  hint: this module was lowered targeting Shardy "
                    f"(it contains xla.sdy custom calls). Pass "
                    f"CompileOptions(..., use_shardy_partitioner=True).", e.code) from e
            raise
        exe = Executable(self._plugin, a.executable, self)
        if cache is not None and ckey is not None:
            try:
                cache.store(ckey, exe.to_artifact(source=code, compile_options=opts_bytes))
            except (errors.PjrtError, OSError):
                pass  # a cache that cannot write must never break a compile
        return exe

    def deserialize_executable(self, blob: bytes, *,
                               overridden_options: bytes | None = None) -> Executable:
        """Load a previously serialized executable, skipping XLA compilation.

        ``overridden_serialized_compile_options`` arrived in PJRT 0.111; it is
        passed through when the plugin is new enough to have the field.
        """
        buf = ctypes.create_string_buffer(blob, len(blob))
        kw: dict[str, Any] = dict(
            client=self._check(),
            serialized_executable=ctypes.cast(buf, _VOIDP),
            serialized_executable_size=len(blob))
        keep = [buf]
        if overridden_options:
            ob = ctypes.create_string_buffer(overridden_options, len(overridden_options))
            keep.append(ob)
            kw["overridden_serialized_compile_options"] = ctypes.cast(ob, _VOIDP)
            kw["overridden_serialized_compile_options_size"] = len(overridden_options)
        a = self._plugin.args("PJRT_Executable_DeserializeAndLoad_Args", **kw)
        with self.diagnose_allocation():
            self._plugin.call("PJRT_Executable_DeserializeAndLoad", a)
        del keep
        return Executable(self._plugin, a.loaded_executable, self)

    def load_artifact(self, artifact_or_path, *, strict: bool = True) -> Executable:
        """Load an :class:`~pypjrt.artifact.Artifact`, checking its guards first.

        A mismatched artifact fails with a diagnostic rather than crashing
        inside XLA -- the reason an AOT artifact should carry arch fields.
        """
        from .artifact import Artifact
        art = (artifact_or_path if isinstance(artifact_or_path, Artifact)
               else Artifact.read(artifact_or_path))
        art.check_compatible(self._plugin, platform=self.platform_name, strict=strict)
        return self.deserialize_executable(art.executable)

    def buffer_from_host(self, data: Any, dtype: int, dims: Sequence[int],
                         device: Device) -> Buffer:
        view = _byte_view(data)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(view))
        dim_arr = (_I64 * len(dims))(*dims)
        a = self._plugin.args(
            "PJRT_Client_BufferFromHostBuffer_Args", client=self._check(),
            data=addr, type=dtype,
            dims=ctypes.cast(dim_arr, _VOIDP), num_dims=len(dims),
            host_buffer_semantics=self._plugin.abi.PJRT_HostBufferSemantics_kImmutableOnlyDuringCall,
            device=device.address)
        with self.diagnose_allocation():
            self._plugin.call("PJRT_Client_BufferFromHostBuffer", a)
        if a.done_with_host_buffer:
            Event(self._plugin, a.done_with_host_buffer).consume()
        return Buffer(self._plugin, a.buffer)


def _named_values(plugin: Plugin, opts: dict[str, Any]):
    """Marshal client create-options into ``PJRT_NamedValue[]``.

    Structural, not env vars, so one process can hold different settings per
    client.
    """
    if not opts:
        return None, None
    abi = plugin.abi
    NV = abi.PJRT_NamedValue
    arr = (NV * len(opts))()
    keep: list[Any] = [arr]
    for i, (k, v) in enumerate(opts.items()):
        nv = arr[i]
        nv.struct_size = abi.PJRT_NamedValue_STRUCT_SIZE
        name = ctypes.create_string_buffer(k.encode())
        keep.append(name)
        nv.name = ctypes.cast(name, _VOIDP)
        nv.name_size = len(k)
        if isinstance(v, bool):
            nv.type = abi.PJRT_NamedValue_kBool
            nv.bool_value = v
        elif isinstance(v, int):
            nv.type = abi.PJRT_NamedValue_kInt64
            nv.int64_value = v
        elif isinstance(v, float):
            nv.type = abi.PJRT_NamedValue_kFloat
            nv.float_value = v
        elif isinstance(v, str):
            sv = ctypes.create_string_buffer(v.encode())
            keep.append(sv)
            nv.type = abi.PJRT_NamedValue_kString
            nv.string_value = ctypes.cast(sv, _VOIDP)
            nv.value_size = len(v)
        else:
            raise errors.InvalidArgument(f"unsupported create-option type for {k!r}: {type(v)}")
    return arr, keep
