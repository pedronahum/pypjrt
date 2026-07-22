"""Tier 1: bf16 end to end, and StableHLO portable artifacts."""
import pytest, pypjrt
from pypjrt.typing import BF16, F32

pytestmark = pytest.mark.tier1
np = pytest.importorskip("numpy")
ml_dtypes = pytest.importorskip("ml_dtypes")
N = 8
BF16_MUL = """
module @m {
  func.func public @main(%a: tensor<8xbf16>, %b: tensor<8xbf16>) -> tensor<8xbf16> {
    %0 = stablehlo.multiply %a, %b : tensor<8xbf16>
    return %0 : tensor<8xbf16>
  }
}
"""


@pytest.fixture(scope="module")
def plugin(cpu_plugin_path):
    return pypjrt.Plugin(cpu_plugin_path)


@pytest.fixture(scope="module")
def client(plugin):
    c = pypjrt.Client.create(plugin)
    yield c
    c.close()


def test_bf16_roundtrip_is_byte_identical(client):
    """bf16 is the accelerator dtype and had never executed here."""
    xs = np.arange(1, N + 1, dtype=ml_dtypes.bfloat16)
    ys = (np.arange(1, N + 1) * 0.5).astype(ml_dtypes.bfloat16)
    want = (xs * ys).astype(ml_dtypes.bfloat16)
    with client.devices() as devs:
        a = client.typed_buffer(BF16, xs, [N], devs[0])
        b = client.typed_buffer(BF16, ys, [N], devs[0])
        assert a.element_type == BF16.code == 13
        assert a.nbytes == N * 2
        e = client.compile(BF16_MUL)
        (o,) = e(a, b)
        got = np.empty(N, dtype=ml_dtypes.bfloat16)
        o.to_host(got)
        assert got.tobytes() == want.tobytes()
        for h in (o, a, b):
            h.close()
        e.close()


def test_byte_view_handles_dtypes_memoryview_rejects(client):
    """memoryview refuses format 'E' (bfloat16, f8) -- exactly the dtypes an
    accelerator cares about. _byte_view reinterprets instead of failing."""
    from pypjrt.client import _byte_view
    arr = np.arange(4, dtype=ml_dtypes.bfloat16)
    with pytest.raises((TypeError, ValueError)):
        memoryview(arr).cast("B")
    assert _byte_view(arr).nbytes == 8


def test_stablehlo_target_is_within_the_plugins_range(plugin):
    lo, hi = plugin.stablehlo_version_range
    assert plugin.stablehlo_target("min") == ".".join(map(str, lo))
    assert plugin.stablehlo_target("max") == ".".join(map(str, hi))


def test_portable_artifact_compiles_and_matches_text(client, plugin):
    """Ship versioned portable artifacts rather
    than raw text, and flagged it unverified. This verifies it."""
    jax = pytest.importorskip("jax")
    sh = pytest.importorskip("jaxlib.mlir.dialects.stablehlo")
    import jax.numpy as jnp

    f = lambda a, b: jnp.tanh(a * b + 1.0)
    x = np.arange(1, N + 1, dtype=np.float32)
    y = (x * 0.5).astype(np.float32)
    text = jax.jit(f).lower(x, y).as_text()

    def run(prog):
        with client.devices() as devs:
            e = client.compile(prog)
            a = client.buffer_from_host(x, 11, [N], devs[0])
            b = client.buffer_from_host(y, 11, [N], devs[0])
            (o,) = e(a, b)
            g = np.empty(N, dtype=np.float32)
            o.to_host(g)
            for h in (o, a, b):
                h.close()
            e.close()
            return g.tobytes()

    ref = run(text)
    for target in (plugin.stablehlo_target("min"), "1.0.0", plugin.stablehlo_target("max")):
        blob = sh.serialize_portable_artifact_str(text, target)
        assert isinstance(blob, bytes) and blob[:1] != b"m", "expected bytecode, not text"
        assert run(blob) == ref, f"portable artifact at {target} differs from text"
