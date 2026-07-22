"""Compile Triton IR through the plugin -- **NVIDIA-only by construction**.

Like `pypjrt.ffi` and `pypjrt.cuda`, this module is named to be visible: core
never mentions a device, and importing this trades portability.

Why it matters: the usual way to reach Triton from a non-Python runtime is to
*discover a Python interpreter with `triton` importable, write a driver script,
and shell out to it*. The CUDA plugin advertises
``PJRT_Extension_Type_Triton`` with a ``compile`` entry point that does the same
job in-process, with no Python subprocess and no `triton` package. Nothing binds it.

Input is Triton MLIR (the ``tt`` dialect); output is PTX plus the shared-memory
requirement, ready to hand to ``cuModuleLoadData`` -- see `pypjrt.cuda` and the
FFI dispatch pattern.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

from . import errors
from ._loader import Plugin

_VOIDP = ctypes.c_void_p


@dataclass(frozen=True)
class CompiledKernel:
    """The plugin's output for one Triton module."""

    asm: bytes           #: PTX (or cubin, plugin-dependent)
    smem_bytes: int      #: dynamic shared memory the launch must reserve
    path: str            #: plugin-side artifact path, when it reports one

    def __repr__(self) -> str:
        return (f"<CompiledKernel {len(self.asm)} bytes, smem={self.smem_bytes}"
                + (f", path={self.path!r}" if self.path else "") + ">")


def available(plugin: Plugin) -> bool:
    return plugin.extension("Triton") is not None


def arch_of(device) -> str:
    """The arch string this extension wants, from a device.

    ``compute_capability`` is reported as ``"12.1"``, and that dotted form is
    what the plugin accepts; ``sm_121a`` and ``sm_121`` are both rejected.
    """
    cc = device.attributes.get("compute_capability")
    if not cc:
        raise errors.UnsupportedByPlugin(
            "device reports no compute_capability; pass arch= explicitly")
    return str(cc)


def compile(plugin: Plugin, module: str | bytes, *, arch: str,
            num_warps: int = 4, num_ctas: int = 1, num_stages: int = 3) -> CompiledKernel:
    """Compile Triton MLIR to PTX using the plugin's own Triton pipeline.

    ``arch`` is the **dotted compute capability**, e.g. ``"12.1"`` -- not the
    ``sm_121a`` spelling ptxas uses, which this entry point rejects. That is
    exactly the string ``Device.attributes["compute_capability"]`` reports, so
    :func:`arch_of` derives it for you.
    """
    ext = plugin.require_extension("Triton")
    src = module.encode() if isinstance(module, str) else bytes(module)
    mb = ctypes.create_string_buffer(src, len(src))
    ab = ctypes.create_string_buffer(arch.encode(), len(arch))

    a = plugin.args("PJRT_Triton_Compile_Args",
                    module=ctypes.cast(mb, _VOIDP), module_size=len(src),
                    arch_name=ctypes.cast(ab, _VOIDP), arch_name_size=len(arch),
                    num_warps=num_warps, num_ctas=num_ctas, num_stages=num_stages)

    # PJRT_Triton_Extension: base[24] then `compile` at 24.
    fn_ptr = ctypes.cast(ext.address + 24, ctypes.POINTER(_VOIDP))[0]
    if not fn_ptr:
        raise errors.UnsupportedByPlugin("Triton extension has no compile entry point")
    err = ctypes.CFUNCTYPE(_VOIDP, _VOIDP)(fn_ptr)(ctypes.byref(a))
    if err:
        raise plugin._to_exception(err)

    asm = ctypes.string_at(a.out_asm, a.out_asm_size) if a.out_asm else b""
    path = (ctypes.string_at(a.out_path, a.out_path_size).decode(errors="replace")
            if a.out_path else "")
    return CompiledKernel(asm=asm, smem_bytes=int(a.out_smem_bytes), path=path)
