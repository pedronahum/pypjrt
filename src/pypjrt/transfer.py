"""Chunked, overlapped host-to-device transfer.

``PJRT_Client_CreateBuffersForAsyncHostToDevice`` allocates device buffers from
shape specs *before* the data exists, then streams into them in pieces. That
lets a large checkpoint move without a full host staging copy, and lets the
first chunks start landing while later ones are still being read.

Rarely bound. It is the kind of API that gets declared and then documented as
"use the blocking host copy instead", which gives up the overlap entirely.

    with client.async_transfer([ShapeSpec(F32, (1024, 1024))], memory) as t:
        for off, chunk in enumerate_chunks(...):
            t.transfer(0, chunk, offset=off, last=...)
        (weights,) = t.buffers()
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Sequence

from . import errors

_VOIDP = ctypes.c_void_p
_I64 = ctypes.c_int64


@dataclass(frozen=True)
class ShapeSpec:
    """What to allocate, before the bytes exist."""

    dtype: int
    dims: tuple[int, ...]

    @property
    def nbytes_hint(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


class AsyncTransfer:
    """A set of device buffers being filled incrementally.

    Ordering rule from the C API: mark the final chunk of each buffer with
    ``last=True``. A buffer is only retrievable once its transfers are done.
    """

    def __init__(self, client, specs: Sequence[ShapeSpec], memory):
        self._client = client
        self._plugin = client._plugin
        self._specs = list(specs)
        self._closed = False
        abi = self._plugin.abi

        arr = (abi.PJRT_ShapeSpec * len(self._specs))()
        self._keep: list[Any] = [arr]
        for i, sp in enumerate(self._specs):
            s = arr[i]
            s.struct_size = abi.PJRT_ShapeSpec_STRUCT_SIZE
            dims = (_I64 * max(len(sp.dims), 1))(*sp.dims)
            self._keep.append(dims)
            s.dims = ctypes.cast(dims, _VOIDP)
            s.num_dims = len(sp.dims)
            s.element_type = sp.dtype

        a = self._plugin.args(
            "PJRT_Client_CreateBuffersForAsyncHostToDevice_Args",
            client=client._check(), shape_specs=ctypes.cast(arr, _VOIDP),
            num_shape_specs=len(self._specs), memory=memory.address)
        with client.diagnose_allocation():
            self._plugin.call("PJRT_Client_CreateBuffersForAsyncHostToDevice", a)
        self._ptr = int(a.transfer_manager)

    # -- inspection --------------------------------------------------------

    def _check(self) -> int:
        if self._closed:
            raise errors.HandleClosed("AsyncTransfer is closed")
        return self._ptr

    def __len__(self) -> int:
        a = self._plugin.args("PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args",
                              transfer_manager=self._check())
        self._plugin.call("PJRT_AsyncHostToDeviceTransferManager_BufferCount", a)
        return int(a.buffer_count)

    def buffer_size(self, index: int) -> int:
        """Device bytes reserved for one buffer -- may exceed the logical size."""
        a = self._plugin.args("PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args",
                              transfer_manager=self._check(), buffer_index=index)
        self._plugin.call("PJRT_AsyncHostToDeviceTransferManager_BufferSize", a)
        return int(a.buffer_size)

    # -- streaming ---------------------------------------------------------

    def transfer(self, index: int, data: Any, *, offset: int = 0,
                 last: bool = True, wait: bool = True) -> None:
        """Copy one chunk into buffer ``index`` at ``offset``.

        ``last=True`` closes that buffer. With ``wait=False`` the completion
        event is awaited anyway before returning, because ``data`` must outlive
        the copy and we do not own the caller's memory.
        """
        from .client import Event, _byte_view
        view = _byte_view(data)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(view))
        a = self._plugin.args(
            "PJRT_AsyncHostToDeviceTransferManager_TransferData_Args",
            transfer_manager=self._check(), buffer_index=index, data=addr,
            offset=offset, transfer_size=view.nbytes, is_last_transfer=last)
        self._plugin.call("PJRT_AsyncHostToDeviceTransferManager_TransferData", a)
        if a.done_with_h2d_transfer:
            ev = Event(self._plugin, a.done_with_h2d_transfer)
            # The host buffer is the caller's; it must stay valid until the copy
            # completes, so we always await rather than hand back a live event.
            ev.consume()
        del view

    def retrieve(self, index: int):
        """The device buffer for ``index``, once its transfers are complete."""
        from .client import Buffer
        a = self._plugin.args("PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args",
                              transfer_manager=self._check(), buffer_index=index)
        self._plugin.call("PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer", a)
        return Buffer(self._plugin, a.buffer_out)

    def buffers(self) -> list:
        return [self.retrieve(i) for i in range(len(self._specs))]

    def device(self):
        """The device these buffers are being filled on."""
        from .client import Device
        a = self._plugin.args("PJRT_AsyncHostToDeviceTransferManager_Device_Args",
                              transfer_manager=self._check())
        self._plugin.call("PJRT_AsyncHostToDeviceTransferManager_Device", a)
        return Device(self._plugin, a.device_out)

    def add_metadata(self, metadata: dict[str, Any]) -> None:
        """Attach named values to the in-flight transfer."""
        from .client import _named_values
        named, keep = _named_values(self._plugin, dict(metadata))
        a = self._plugin.args("PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args",
                              transfer_manager=self._check(),
                              transfer_metadata=ctypes.cast(named, _VOIDP) if named else None,
                              num_metadata=len(metadata))
        self._plugin.call("PJRT_AsyncHostToDeviceTransferManager_AddMetadata", a)
        del keep

    def set_error(self, index: int, code: int, message: str) -> None:
        """Abort one buffer, so a consumer sees a failure instead of garbage."""
        msg = message.encode()
        buf = ctypes.create_string_buffer(msg, len(msg))
        a = self._plugin.args(
            "PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args",
            transfer_manager=self._check(), buffer_index=index, error_code=code,
            error_message=ctypes.cast(buf, _VOIDP), error_message_size=len(msg))
        self._plugin.call("PJRT_AsyncHostToDeviceTransferManager_SetBufferError", a)

    # -- lifetime ----------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._plugin.call(
            "PJRT_AsyncHostToDeviceTransferManager_Destroy",
            self._plugin.args("PJRT_AsyncHostToDeviceTransferManager_Destroy_Args",
                              transfer_manager=self._ptr))

    def __enter__(self) -> "AsyncTransfer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<AsyncTransfer {len(self._specs)} buffer(s)>"
