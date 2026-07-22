"""Registering Python functions as XLA FFI handlers.

Four contracts, each learned the hard way by a source project:

1. **Service the metadata probe first, centrally.** Before first use XLA calls
   the handler with an ``XLA_FFI_Extension_Metadata`` on the frame; the handler
   must fill in ``api_version`` and return. *If we skip
   this dance, XLA silently drops the registration -- the handler will never be
   found by name at compile time."* Doing it here means user handlers only ever
   see EXECUTE.
2. **Use the FFI extension**, not ``PJRT_Gpu_Register_Custom_Call`` with
   ``api_version = 1``, which *"deadlocks in JAX 0.10.1's CUDA plugin"*.
3. **Never let a Python exception reach XLA.** Convert to ``XLA_FFI_Error``, so
   a bad handler fails one execution instead of the process.
4. **Never ``dlclose`` the plugin** and never free the trampoline: the plugin
   keeps these pointers for the process lifetime. ``ctypes.CFUNCTYPE`` wraps a
   real closure, so no pool of pre-compiled trampolines with GC-rooted slots is
   needed, as a C or Swift binding would require.

Registration also happens to be re-entrant: the plugin invokes the handler
*synchronously* inside ``PJRT_FFI_Register_Handler``. Handlers therefore must
not call back into PJRT -- see ``pypjrt._lock``.
"""

from __future__ import annotations

import ctypes
from typing import Callable

from .. import errors
from .._lock import dispatching
from .._loader import Plugin
from .frame import CallFrame

_VOIDP = ctypes.c_void_p
_HANDLER_T = ctypes.CFUNCTYPE(_VOIDP, _VOIDP)

#: Everything the plugin retains for the life of the process. Never freed.
_ARENA: list = []
_REGISTERED: dict[tuple[int, str, str], Callable] = {}

INTERNAL = 13  # XLA_FFI_Error_Code_INTERNAL


def _make_error(abi, api: int, message: str, code: int = INTERNAL) -> int:
    """Build an XLA_FFI_Error. Returning 0 means success."""
    if not api:
        return 0
    fn_ptr = ctypes.cast(api + abi.XLA_FFI_Api.XLA_FFI_Error_Create.offset,
                         ctypes.POINTER(_VOIDP))[0]
    if not fn_ptr:
        return 0
    msg = ctypes.create_string_buffer(message.encode()[:4000])
    _ARENA.append(msg)  # XLA copies, but outliving the call costs nothing
    a = abi.XLA_FFI_Error_Create_Args()
    ctypes.memset(ctypes.byref(a), 0, ctypes.sizeof(a))
    a.struct_size = abi.XLA_FFI_Error_Create_Args_STRUCT_SIZE
    a.message = ctypes.cast(msg, _VOIDP)
    a.errc = code
    return int(ctypes.CFUNCTYPE(_VOIDP, _VOIDP)(fn_ptr)(ctypes.byref(a)) or 0)


def _dispatch(abi, fn: Callable[[CallFrame], None], frame_ptr: int) -> int:
    f = ctypes.cast(frame_ptr, ctypes.POINTER(abi.XLA_FFI_CallFrame)).contents

    # (1) the metadata probe, before anything else
    ext = f.extension_start
    while ext:
        base = ctypes.cast(ext, ctypes.POINTER(abi.XLA_FFI_Extension_Base)).contents
        if int(base.type) == abi.XLA_FFI_Extension_Metadata:
            md_ptr = ctypes.cast(ext + abi.XLA_FFI_Metadata_Extension.metadata.offset,
                                 ctypes.POINTER(_VOIDP))[0]
            md = ctypes.cast(md_ptr, ctypes.POINTER(abi.XLA_FFI_Metadata)).contents
            md.api_version.major_version = abi.XLA_FFI_API_MAJOR
            md.api_version.minor_version = abi.XLA_FFI_API_MINOR
            md.traits = 0
            return 0
        ext = int(base.next or 0)

    if int(f.stage) != abi.XLA_FFI_ExecutionStage_EXECUTE:
        return 0

    try:
        fn(CallFrame(abi, frame_ptr))
    except BaseException as e:  # noqa: BLE001 -- must never unwind into XLA
        return _make_error(abi, int(f.api or 0),
                           f"{type(e).__name__} in FFI handler: {e}")
    return 0


def register(plugin: Plugin, name: str, fn: Callable[[CallFrame], None], *,
             platform: str | None = None, traits: int = 0) -> None:
    """Register ``fn`` as the handler for ``stablehlo.custom_call @name``.

    The module must spell it as typed FFI::

        stablehlo.custom_call @name(%x) {api_version = 4 : i32,
                                        backend_config = {k = 1 : i64}}
    """
    ext = plugin.require_extension("FFI")
    abi = plugin.abi
    plat = platform or ("CUDA" if plugin.is_accelerator else "Host")
    key = (plugin.api, name, plat)
    if key in _REGISTERED:
        raise errors.AlreadyExists(
            f"an FFI handler named {name!r} is already registered for platform {plat!r}")

    trampoline = _HANDLER_T(lambda p: _dispatch(abi, fn, p))
    name_b = ctypes.create_string_buffer(name.encode())
    plat_b = ctypes.create_string_buffer(plat.encode())
    _ARENA.extend((trampoline, name_b, plat_b))

    a = plugin.args(
        "PJRT_FFI_Register_Handler_Args",
        target_name=ctypes.cast(name_b, _VOIDP), target_name_size=len(name),
        handler=ctypes.cast(trampoline, _VOIDP),
        platform_name=ctypes.cast(plat_b, _VOIDP), platform_name_size=len(plat),
        traits=traits)

    # PJRT_FFI_Extension: base[24], type_register@24, user_data_add@32,
    # register_handler@40.
    fn_ptr = ctypes.cast(ext.address + 40, ctypes.POINTER(_VOIDP))[0]
    if not fn_ptr:
        raise errors.UnsupportedByPlugin("FFI extension has no register_handler")
    # The plugin calls the handler synchronously from inside this call.
    with dispatching():
        err = ctypes.CFUNCTYPE(_VOIDP, _VOIDP)(fn_ptr)(ctypes.byref(a))
    if err:
        raise plugin._to_exception(err)
    _REGISTERED[key] = fn


def handler(plugin: Plugin, name: str, *, platform: str | None = None, traits: int = 0):
    """Decorator form of :func:`register`."""
    def deco(fn):
        register(plugin, name, fn, platform=platform, traits=traits)
        return fn
    return deco


def registered(plugin: Plugin | None = None) -> list[str]:
    if plugin is None:
        return sorted({n for _, n, _ in _REGISTERED})
    return sorted({n for api, n, _ in _REGISTERED if api == plugin.api})
