"""Tier 1: M7 -- zero-copy interop via DLPack."""
import gc, pytest, pypjrt
from pypjrt import errors
from pypjrt.dlpack import kDLCPU, live_exports

pytestmark = pytest.mark.tier1
np = pytest.importorskip("numpy")
jax = pytest.importorskip("jax")
import jax.numpy as jnp

F32, N = 11, 16


@pytest.fixture(scope="module")
def client(cpu_plugin_path):
    c = pypjrt.Client.create(pypjrt.Plugin(cpu_plugin_path))
    yield c
    c.close()


@pytest.fixture
def buf(client):
    with client.devices() as devs:
        b = client.buffer_from_host(np.arange(N, dtype=np.float32) * 1.5, F32, [N], devs[0])
    yield b
    b.close()


def _misaligned(n=N):
    """A float32 view guaranteed NOT to sit on a 64-byte boundary.

    Relying on numpy's allocator to be unaligned made these tests skip roughly
    at random -- a test that sometimes tests nothing is the failure mode this
    suite exists to avoid.
    """
    base = np.empty(n + 32, dtype=np.float32)
    pad = (64 - (base.__array_interface__["data"][0] % 64)) // 4
    view = base[pad + 1: pad + 1 + n]
    assert view.__array_interface__["data"][0] % 64 != 0
    view[:] = np.arange(n, dtype=np.float32) + 100.0
    return view


def test_dlpack_device(buf):
    assert buf.__dlpack_device__() == (kDLCPU, 0)


def test_m7_gate_export_is_zero_copy(buf):
    """numpy's view must be the *same memory*, not a copy."""
    arr = np.from_dlpack(buf)
    assert arr.dtype == np.float32 and arr.shape == (N,)
    assert arr.__array_interface__["data"][0] == buf.device_pointer()
    assert np.array_equal(arr, np.arange(N, dtype=np.float32) * 1.5)
    del arr
    gc.collect()


def test_export_to_jax(buf):
    j = jax.dlpack.from_dlpack(buf)
    assert np.array_equal(np.asarray(j), np.arange(N, dtype=np.float32) * 1.5)
    del j
    gc.collect()


def test_m7_gate_import_from_jax(client):
    """jax allocates with the alignment XLA's view API requires."""
    src = jnp.arange(N, dtype=jnp.float32) * 3.0
    adopted = client.from_dlpack(src)
    assert adopted.dimensions == (N,) and adopted.element_type == F32
    got = np.frombuffer(adopted.to_host(), dtype=np.float32)
    assert np.array_equal(got, np.asarray(src))
    adopted.close()


def test_m7_gate_full_roundtrip_with_compute(client):
    """jax -> pypjrt -> execute -> jax, never touching the host."""
    src = jnp.arange(N, dtype=jnp.float32)
    adopted = client.from_dlpack(src)
    exe = client.compile("""
module @m {
  func.func public @main(%a: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.multiply %a, %a : tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}""")
    (out,) = exe(adopted)
    back = jax.dlpack.from_dlpack(out)
    assert np.array_equal(np.asarray(back), np.arange(N, dtype=np.float32) ** 2)
    del back
    gc.collect()
    out.close(); adopted.close(); exe.close()


def test_unaligned_producer_is_refused_with_a_hint(client):
    """numpy's default allocator does not meet the backend's alignment. The
    error has to say so -- 'unaligned data' alone is not actionable."""
    with pytest.raises(errors.PjrtError, match="hint: zero-copy import requires"):
        client.from_dlpack(_misaligned())


def test_pin_is_released_when_the_consumer_drops_it(buf):
    before = live_exports()
    arr = np.from_dlpack(buf)
    assert live_exports() == before + 1, "export was not pinned"
    del arr
    for _ in range(3):
        gc.collect()
    assert live_exports() == before, "external reference leaked"


def test_unconsumed_capsule_still_releases(buf):
    before = live_exports()
    cap = buf.__dlpack__()
    assert live_exports() == before + 1
    del cap                       # never handed to a consumer
    for _ in range(3):
        gc.collect()
    assert live_exports() == before, "dropping an unconsumed capsule leaked the pin"


def test_a_capsule_may_be_consumed_only_once(client, buf):
    cap = buf.__dlpack__()
    adopted = client.from_dlpack(cap)
    with pytest.raises(errors.InvalidArgument, match="already consumed"):
        client.from_dlpack(cap)
    adopted.close()


def test_failed_import_does_not_consume_the_capsule(client):
    """If CreateViewOfDeviceBuffer refuses, the producer keeps its memory."""
    host = _misaligned()
    cap = host.__dlpack__()
    with pytest.raises(errors.PjrtError):
        client.from_dlpack(cap)
    # the capsule must be left unconsumed, so the producer keeps its memory
    import ctypes
    from pypjrt.dlpack import _obj, _py, _UNUSED
    assert _py.PyCapsule_IsValid(_obj(cap), _UNUSED), \
        "a failed import consumed the capsule and stranded the producer's buffer"
