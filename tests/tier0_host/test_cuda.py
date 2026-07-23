"""Tier 0: the parts of pypjrt.cuda that need no device.

Launching a kernel is a GPU-only integration test (tier2), but the argument
marshalling in front of ``cuLaunchKernel`` is pure logic, and it is exactly
where a mistake is silent -- a wrong dim or a mis-sized scalar returns quiet
garbage rather than an error. So it is pinned here, on any machine.
"""
import ctypes

import pytest

import pypjrt.cuda as cuda

pytestmark = pytest.mark.tier0


# --- grid/block normalization ---------------------------------------------

@pytest.mark.parametrize("value, expected", [
    (64, (64, 1, 1)),
    ((8,), (8, 1, 1)),
    ((8, 4), (8, 4, 1)),
    ((8, 4, 2), (8, 4, 2)),
    ([16, 16], (16, 16, 1)),
])
def test_dim3_normalizes(value, expected):
    assert cuda._dim3(value) == expected


@pytest.mark.parametrize("bad", [(), (1, 2, 3, 4), (0, 0, 0, 0)])
def test_dim3_rejects_wrong_arity(bad):
    with pytest.raises(ValueError):
        cuda._dim3(bad)


# --- kernel-parameter packing ---------------------------------------------

def test_pack_params_empty_is_null():
    arr, keep = cuda._pack_params([])
    assert arr is None and keep == []


def test_pack_params_builds_pointer_array():
    a = ctypes.c_void_p(0xdead0000)
    b = ctypes.c_uint32(7)
    arr, keep = cuda._pack_params([a, b])

    assert len(arr) == 2
    # Each slot points AT the value object, not at its contents.
    assert arr[0] == ctypes.addressof(a)
    assert arr[1] == ctypes.addressof(b)
    # The originals must be kept alive, or the addresses dangle.
    assert a in keep[0] and b in keep[0]


def test_pack_params_preserves_scalar_width():
    """A u32 scalar must occupy 4 bytes, not be widened to 8. Reading through
    the packed pointer as the original type must give the original value --
    this is the silent-garbage bug the ctypes-object requirement prevents."""
    n = ctypes.c_uint32(0x11223344)
    arr, _keep = cuda._pack_params([ctypes.c_void_p(0), n])
    seen = ctypes.cast(arr[1], ctypes.POINTER(ctypes.c_uint32))[0]
    assert seen == 0x11223344


def test_pack_params_rejects_bare_int():
    """Refusing a bare Python int is deliberate: silently treating it as a
    pointer-width value is how a kernel reads its scalars from wrong offsets."""
    with pytest.raises(TypeError, match="ctypes value object"):
        cuda._pack_params([ctypes.c_void_p(0), 1234])


# --- degradation when there is no driver ----------------------------------

def test_public_surface_exists():
    for name in ("launch_kernel", "module_load_data", "module_get_function",
                 "module_unload", "memcpy_dtod_async", "available", "lib"):
        assert callable(getattr(cuda, name))


def test_calls_raise_cleanly_without_a_driver(monkeypatch):
    """On a box with no CUDA driver every entry point raises CudaError, not a
    bare AttributeError or a segfault."""
    def no_driver():
        raise cuda.CudaError("libcuda.so.1 not loadable: simulated")
    monkeypatch.setattr(cuda, "lib", no_driver)

    assert cuda.available() is False
    with pytest.raises(cuda.CudaError):
        cuda.module_load_data(b"//dummy")
    with pytest.raises(cuda.CudaError):
        cuda.launch_kernel(0, 1, 1, [ctypes.c_void_p(0)], 0)
