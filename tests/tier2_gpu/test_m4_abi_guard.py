"""Tier 2: the AbiVersion artifact guard, which only the CUDA plugin advertises."""
import time, pytest, pypjrt
from pypjrt.artifact import AbiVersion

pytestmark = pytest.mark.tier2
TIGHT = {"preallocate": False, "memory_fraction": 0.05}
PROG = """
module @m {
  func.func public @main(%a: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = stablehlo.tanh %a : tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}
"""


@pytest.fixture(scope="module")
def plugin(gpu_plugin_path):
    return pypjrt.Plugin(gpu_plugin_path)


def test_abi_version_extension_advertised(plugin):
    assert AbiVersion.probe(plugin) is not None


def test_abi_guard_returns_the_plugins_own_verdict(plugin):
    """`IsCompatibleWithExecutable` is answered by the plugin, not guessed by
    us. On this box it reports a genuine CUDA toolkit skew."""
    with pypjrt.Client.create(plugin, options=TIGHT) as c:
        e = c.compile(PROG)
        verdict = e.abi_compatibility()
        assert verdict is None or isinstance(verdict, str)
        if verdict:
            assert "version" in verdict.lower()
        e.close()


def test_artifact_carries_the_abi_proto(plugin):
    with pypjrt.Client.create(plugin, options=TIGHT) as c:
        e = c.compile(PROG)
        art = e.to_artifact(source=PROG.encode())
        e.close()
        assert art.platform == "cuda"
        assert art.abi_proto, "AbiVersion extension present but no proto recorded"
        assert bytes.fromhex(art.abi_proto)


def test_cache_saves_real_time_on_gpu(plugin, tmp_path):
    cache = pypjrt.CompileCache(tmp_path)
    with pypjrt.Client.create(plugin, options=TIGHT) as c:
        t0 = time.perf_counter(); a = c.compile(PROG, cache=cache); cold = time.perf_counter() - t0
        a.close()
        t0 = time.perf_counter(); b = c.compile(PROG, cache=cache); warm = time.perf_counter() - t0
        b.close()
    assert cache.hits == 1
    assert warm < cold / 2, f"cold {cold*1e3:.0f}ms warm {warm*1e3:.0f}ms"
