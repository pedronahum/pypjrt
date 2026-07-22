"""Tier 2: M7 on device memory. jax here is CPU-only, so this is a self
round-trip: export a GPU buffer and adopt it back as a view, with no copy."""
import gc, pytest, pypjrt
from pypjrt.dlpack import _extract, kDLCUDA, live_exports

pytestmark = pytest.mark.tier2
np = pytest.importorskip("numpy")
F32, N = 11, 32
TIGHT = {"preallocate": False, "memory_fraction": 0.05}


@pytest.fixture(scope="module")
def client(gpu_plugin_path):
    c = pypjrt.Client.create(pypjrt.Plugin(gpu_plugin_path), options=TIGHT)
    yield c
    c.close()


def test_m7_gpu_gate_self_roundtrip(client):
    src = np.arange(N, dtype=np.float32) * 2.0
    with client.devices() as devs:
        buf = client.buffer_from_host(src, F32, [N], devs[0])

    assert buf.__dlpack_device__()[0] == kDLCUDA
    cap = buf.__dlpack__()
    dl, _, _, _ = _extract(cap)
    assert dl.device.device_type == kDLCUDA
    assert int(dl.data) == buf.device_pointer(), "capsule does not point at the buffer"
    assert (dl.dtype.code, dl.dtype.bits, dl.ndim) == (2, 32, 1)

    view = client.from_dlpack(cap)
    assert view.device_pointer() == buf.device_pointer(), "view is not the same memory"

    exe = client.compile("""
module @m {
  func.func public @main(%a: tensor<32xf32>) -> tensor<32xf32> {
    %0 = stablehlo.multiply %a, %a : tensor<32xf32>
    return %0 : tensor<32xf32>
  }
}""")
    (out,) = exe(view)
    assert np.array_equal(np.frombuffer(out.to_host(), dtype=np.float32), src * src)
    out.close(); view.close(); exe.close(); buf.close()


def test_gpu_pin_released_on_gc(client):
    with client.devices() as devs:
        buf = client.buffer_from_host(np.ones(N, dtype=np.float32), F32, [N], devs[0])
    before = live_exports()
    cap = buf.__dlpack__()
    assert live_exports() == before + 1
    del cap
    for _ in range(3):
        gc.collect()
    assert live_exports() == before
    buf.close()
