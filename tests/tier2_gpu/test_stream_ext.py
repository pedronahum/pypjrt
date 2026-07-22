"""Tier 2: the Stream extension, which only the GPU plugin advertises."""
import pytest, pypjrt
from pypjrt import extensions as ext
from pypjrt.typing import F32

pytestmark = pytest.mark.tier2
np = pytest.importorskip("numpy")
TIGHT = {"preallocate": False, "memory_fraction": 0.05}


@pytest.fixture(scope="module")
def plugin(gpu_plugin_path):
    p = pypjrt.Plugin(gpu_plugin_path)
    if not ext.stream_available(p):
        pytest.skip("plugin does not advertise Stream")
    return p


def test_external_stream_handshake(plugin):
    """The primitive behind zero-copy hand-off: a foreign stream can wait on a
    pypjrt buffer with no host round-trip."""
    with pypjrt.Client.create(plugin, options=TIGHT) as c, c.devices() as devs:
        stream = ext.device_stream(devs[0])
        assert stream != 0
        b = c.typed_buffer(F32, np.arange(8, dtype=np.float32), [8], devs[0])
        ext.wait_for_buffer(b, stream)
        got = np.empty(8, np.float32)
        b.to_host(got)
        assert np.array_equal(got, np.arange(8))
        b.close()
