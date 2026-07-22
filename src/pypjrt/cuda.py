"""A minimal CUDA driver binding -- **NVIDIA-only by construction**.

Like `pypjrt.ffi`, this module exists to be visible: `pypjrt` core never names a
device, and importing `pypjrt.cuda` is an explicit trade of portability
Nothing in `pypjrt` core imports it.

Scope is deliberately tiny: enough to move bytes and launch a kernel on the
stream XLA hands an FFI handler. `libcuda.so.1` is opened lazily on first use
and held for the process lifetime.

The rules for operating on XLA's stream, from `KptxKernelRegistry`:
  - launch on the stream you were given; never synchronize it;
  - never switch CUDA context;
  - load modules lazily *in the handler thread*, because CUmodules are
    per-CUcontext and XLA's context is only current there.
"""

from __future__ import annotations

import ctypes
import threading

_lib = None
_lock = threading.Lock()

CUDA_SUCCESS = 0


class CudaError(RuntimeError):
    pass


def lib():
    """`libcuda.so.1`, opened once. Raises if the driver is absent."""
    global _lib
    if _lib is not None:
        return _lib
    with _lock:
        if _lib is None:
            try:
                _lib = ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                raise CudaError(f"libcuda.so.1 not loadable: {e}") from e
            _lib.cuInit(0)
    return _lib


def available() -> bool:
    try:
        lib()
        return True
    except CudaError:
        return False


def _check(rc: int, what: str) -> None:
    if rc != CUDA_SUCCESS:
        name = ctypes.c_char_p()
        try:
            lib().cuGetErrorName(rc, ctypes.byref(name))
            detail = name.value.decode() if name.value else str(rc)
        except Exception:
            detail = str(rc)
        raise CudaError(f"{what} failed: {detail} ({rc})")


def memcpy_dtod_async(dst: int, src: int, nbytes: int, stream: int) -> None:
    """Device-to-device copy enqueued on ``stream``. Does not synchronize."""
    fn = lib().cuMemcpyDtoDAsync_v2
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    _check(fn(ctypes.c_void_p(dst), ctypes.c_void_p(src),
              ctypes.c_size_t(nbytes), ctypes.c_void_p(stream)), "cuMemcpyDtoDAsync")


def memcpy_dtoh(dst_addr: int, src: int, nbytes: int) -> None:
    """Blocking device-to-host copy. For tests and slow paths only."""
    fn = lib().cuMemcpyDtoH_v2
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    _check(fn(ctypes.c_void_p(dst_addr), ctypes.c_void_p(src),
              ctypes.c_size_t(nbytes)), "cuMemcpyDtoH")


def memcpy_htod(dst: int, src_addr: int, nbytes: int) -> None:
    fn = lib().cuMemcpyHtoD_v2
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    _check(fn(ctypes.c_void_p(dst), ctypes.c_void_p(src_addr),
              ctypes.c_size_t(nbytes)), "cuMemcpyHtoD")


def driver_version() -> int:
    v = ctypes.c_int()
    _check(lib().cuDriverGetVersion(ctypes.byref(v)), "cuDriverGetVersion")
    return v.value
