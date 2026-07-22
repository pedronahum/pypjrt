"""Thin wrappers over the smaller PJRT extensions.

Each is behind a capability probe: plugins legitimately differ, and the core
must never assume one is present.

* **Layouts** -- what a buffer's device layout actually is, and what layout a
  program expects. Both plugins here advertise it, and executable
  parameter/output layouts are almost never read.
* **Stream** -- the bridge for *external* stream interop: get the stream a
  device signals readiness on, or make a foreign stream wait on a buffer. This
  is what lets a torch or cupy stream synchronise with a pypjrt buffer without
  a host round-trip. Unlike `pypjrt.cuda` these are vendor-neutral: the stream
  is an opaque ``intptr_t``.
"""

from __future__ import annotations

import ctypes
from typing import Any

from . import errors
from ._loader import Plugin

_VOIDP = ctypes.c_void_p
_I64 = ctypes.c_int64
_FN = ctypes.CFUNCTYPE(_VOIDP, _VOIDP)


def _entry(plugin: Plugin, ext_name: str, offset: int):
    ext = plugin.require_extension(ext_name)
    ptr = ctypes.cast(ext.address + offset, ctypes.POINTER(_VOIDP))[0]
    if not ptr:
        raise errors.UnsupportedByPlugin(f"{ext_name} extension entry at +{offset} is NULL")
    return _FN(ptr)


def _invoke(plugin: Plugin, fn, args) -> None:
    err = fn(ctypes.byref(args))
    if err:
        raise plugin._to_exception(err)


# ---------------------------------------------------------------------------
# Layouts. PJRT_Layouts_Extension: base[24] then the table.

_LAYOUTS = {
    "destroy": 24, "serialize": 32, "client_default": 40, "buffer": 48,
    "topology_default": 56, "executable_outputs": 64, "executable_parameters": 72,
}


class Layout:
    """An opaque device layout. Serialize it to see what it is.

    Ownership is not uniform. ``buffer_layout`` and ``default_layout`` return a
    freshly allocated layout the caller must destroy; an executable's
    parameter/output layouts live *inside* the ``PJRT_Executable`` (XLA stores
    them in a vector and hands out interior pointers), so destroying one frees
    memory that was never separately allocated -- ``free(): invalid size``,
    found the hard way. Same shape as ``Topology.from_client``.
    """

    def __init__(self, plugin: Plugin, ptr: int, *, owned: bool = True):
        self._plugin, self._ptr, self._owned = plugin, ptr, owned

    def serialize(self) -> bytes:
        a = self._plugin.args("PJRT_Layouts_MemoryLayout_Serialize_Args", layout=self._ptr)
        _invoke(self._plugin, _entry(self._plugin, "Layouts", _LAYOUTS["serialize"]), a)
        try:
            return ctypes.string_at(a.serialized_bytes, a.serialized_bytes_size)
        finally:
            if a.serialized_layout and a.serialized_layout_deleter:
                ctypes.CFUNCTYPE(None, _VOIDP)(a.serialized_layout_deleter)(a.serialized_layout)

    def close(self) -> None:
        if self._ptr and self._owned:
            a = self._plugin.args("PJRT_Layouts_MemoryLayout_Destroy_Args", layout=self._ptr)
            _invoke(self._plugin, _entry(self._plugin, "Layouts", _LAYOUTS["destroy"]), a)
        self._ptr = 0 if self._owned else self._ptr

    def __enter__(self) -> "Layout":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        try:
            return f"<Layout {self.serialize().decode(errors='replace')[:48]!r}>"
        except errors.PjrtError:
            return f"<Layout 0x{self._ptr:x}>"


def layouts_available(plugin: Plugin) -> bool:
    return plugin.extension("Layouts") is not None


def buffer_layout(buffer) -> Layout:
    p = buffer._plugin
    a = p.args("PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args", buffer=buffer._check())
    _invoke(p, _entry(p, "Layouts", _LAYOUTS["buffer"]), a)
    return Layout(p, a.layout)


def default_layout(client, dtype: int, dims: tuple[int, ...]) -> Layout:
    """The layout this client would choose for a shape."""
    p = client._plugin
    d = (_I64 * max(len(dims), 1))(*dims)
    a = p.args("PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args", client=client._check(),
               type=dtype, dims=ctypes.cast(d, _VOIDP), num_dims=len(dims))
    _invoke(p, _entry(p, "Layouts", _LAYOUTS["client_default"]), a)
    return Layout(p, a.layout)


def executable_layouts(executable, which: str = "parameters") -> list[Layout]:
    """Layouts a compiled program expects (``"parameters"``) or produces
    (``"outputs"``). Bound by no source repo."""
    p = executable._plugin
    key = "executable_parameters" if which == "parameters" else "executable_outputs"
    name = ("PJRT_Layouts_PJRT_Executable_GetParameterLayouts_Args" if which == "parameters"
            else "PJRT_Layouts_PJRT_Executable_GetOutputLayouts_Args")
    a = p.args(name, executable=executable._executable())
    _invoke(p, _entry(p, "Layouts", _LAYOUTS[key]), a)
    n = int(a.num_parameters if which == "parameters" else a.num_outputs)
    ptrs = ctypes.cast(a.layouts, ctypes.POINTER(_VOIDP))
    # Owned by the executable -- see Layout's docstring.
    return [Layout(p, int(ptrs[i]), owned=False) for i in range(n)]


# ---------------------------------------------------------------------------
# Stream. PJRT_Stream_Extension: base[24], get_stream@24, wait_stream@32.


def stream_available(plugin: Plugin) -> bool:
    return plugin.extension("Stream") is not None


def device_stream(device) -> int:
    """The stream this device signals external readiness on.

    Vendor-neutral: an opaque ``intptr_t``, not a ``CUstream`` type.
    """
    p = device._plugin
    a = p.args("PJRT_Get_Stream_For_External_Ready_Events_Args", device=device.address)
    _invoke(p, _entry(p, "Stream", 24), a)
    return int(a.stream)


def wait_for_buffer(buffer, stream: int) -> None:
    """Make a foreign stream wait until ``buffer`` is ready.

    The synchronisation primitive behind zero-copy hand-off: a torch or cupy
    stream can queue work behind a pypjrt buffer with no host round-trip.
    """
    p = buffer._plugin
    a = p.args("PJRT_Wait_Until_Buffer_Ready_On_Stream_Args",
               stream=stream, buffer=buffer._check())
    _invoke(p, _entry(p, "Stream", 32), a)
