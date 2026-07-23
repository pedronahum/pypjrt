"""A minimal CUDA driver binding -- **NVIDIA-only by construction**.

Like `pypjrt.ffi`, this module exists to be visible: `pypjrt` core never names a
device, and importing `pypjrt.cuda` is an explicit trade of portability
Nothing in `pypjrt` core imports it.

Scope is deliberately tiny: enough to move bytes and launch a kernel on the
stream XLA hands an FFI handler. `libcuda.so.1` is opened lazily on first use
and held for the process lifetime.

The rules for operating on XLA's stream:
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


# --- modules and kernels ---------------------------------------------------
# CUmodules are per-CUcontext. When XLA owns the context (the usual case, e.g.
# a kernel launched from an FFI handler) these must be called on XLA's thread,
# where its context is current -- see this module's docstring.


def module_load_data(image: bytes) -> int:
    """Load a compiled module (PTX or cubin) and return its handle.

    ``image`` is the plugin's Triton output or any ``cuModuleLoadData``-able
    blob. PTX is JIT-compiled by the driver here, so a malformed or
    wrong-arch image fails at *this* call, not at launch.
    """
    mod = ctypes.c_void_p()
    fn = lib().cuModuleLoadData
    fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    # PTX is passed to the driver JIT as a NUL-terminated C string; a buffer
    # sized exactly to the image leaves the JIT reading past the end, which
    # surfaces as an intermittent CUDA_ERROR_INVALID_PTX. Passing just the
    # bytes (no explicit size) appends the terminator, and preserves any
    # internal NULs a cubin may contain.
    buf = ctypes.create_string_buffer(bytes(image))
    _check(fn(ctypes.byref(mod), ctypes.cast(buf, ctypes.c_void_p)),
           "cuModuleLoadData")
    return int(mod.value or 0)


def module_get_function(module: int, name: str) -> int:
    """Resolve a ``.visible .entry`` symbol in ``module`` to a function handle."""
    f = ctypes.c_void_p()
    fn = lib().cuModuleGetFunction
    fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    _check(fn(ctypes.byref(f), ctypes.c_void_p(module), name.encode()),
           "cuModuleGetFunction")
    return int(f.value or 0)


def module_unload(module: int) -> None:
    fn = lib().cuModuleUnload
    fn.argtypes = [ctypes.c_void_p]
    _check(fn(ctypes.c_void_p(module)), "cuModuleUnload")


def _dim3(v) -> tuple[int, int, int]:
    """Normalize an int or 1-3 element sequence to a (x, y, z) triple."""
    if isinstance(v, int):
        return (v, 1, 1)
    t = [int(x) for x in v]
    if not 1 <= len(t) <= 3:
        raise ValueError(f"dim must be an int or 1-3 ints, got {v!r}")
    t += [1] * (3 - len(t))
    return (t[0], t[1], t[2])


def _pack_params(params):
    """Build the ``void**`` kernel-parameter array the driver expects.

    Each element must be a **ctypes value object**, because CUDA reads the
    argument's raw bytes and the *width* matters: a device pointer is
    ``ctypes.c_void_p(addr)`` (8 bytes), a ``.param .u32`` scalar is
    ``ctypes.c_uint32(v)`` (4 bytes). Passing a bare Python ``int`` is
    rejected rather than silently widened -- silent widening is exactly how a
    kernel reads a scalar from the wrong offset and returns quiet garbage.

    Returns ``(array_or_None, keepalive)``. The caller must keep ``keepalive``
    alive until ``cuLaunchKernel`` returns: the array holds raw addresses into
    the param objects, not references to them.
    """
    params = list(params)
    if not params:
        return None, []
    ptrs = []
    for i, p in enumerate(params):
        try:
            ptrs.append(ctypes.cast(ctypes.byref(p), ctypes.c_void_p))
        except TypeError:
            raise TypeError(
                f"launch param {i} must be a ctypes value object "
                f"(e.g. ctypes.c_void_p(device_ptr) or ctypes.c_uint32(n)); "
                f"got {type(p).__name__}") from None
    arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
    return arr, [params, arr]


def launch_kernel(func: int, grid, block, params, stream: int,
                  *, shared_bytes: int = 0) -> None:
    """Enqueue ``func`` on ``stream``. Does not synchronize.

    ``grid`` and ``block`` are each an int (1-D) or a 1-3 element sequence.
    ``params`` is a sequence of ctypes value objects, one per kernel parameter
    -- see :func:`_pack_params` for why bare ints are refused. ``func`` and
    ``stream`` are the integer handles from :func:`module_get_function` and
    ``call.stream()``.

    The rules from this module's docstring apply: launch on the stream you were
    given, never synchronize it, never switch context.
    """
    gx, gy, gz = _dim3(grid)
    bx, by, bz = _dim3(block)
    arr, _keep = _pack_params(params)
    fn = lib().cuLaunchKernel
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
    ]
    _check(fn(ctypes.c_void_p(func), gx, gy, gz, bx, by, bz,
              shared_bytes, ctypes.c_void_p(stream), arr, None), "cuLaunchKernel")
    del _keep
