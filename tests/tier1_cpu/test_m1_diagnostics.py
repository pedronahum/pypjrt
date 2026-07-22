"""Tier 1: M1 diagnostics against a CPU plugin, including graceful degradation."""
import array, pytest, pypjrt
from pypjrt import errors

pytestmark = pytest.mark.tier1
F32 = 11
DOT = """
module @m {
  func.func public @main(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
"""


@pytest.fixture(scope="module")
def plugin(cpu_plugin_path):
    return pypjrt.Plugin(cpu_plugin_path)


def test_plugin_attributes(plugin):
    attrs = plugin.attributes
    assert isinstance(attrs.get("xla_version"), int)
    assert plugin.xla_version == attrs["xla_version"]


def test_stablehlo_version_range(plugin):
    """The target range for portable artifacts."""
    lo, hi = plugin.stablehlo_version_range
    assert len(lo) == 3 and len(hi) == 3
    assert lo <= hi


def test_cpu_plugin_is_not_an_accelerator(plugin):
    """Detected from attributes, not from the filename."""
    assert plugin.is_accelerator is False


def test_no_gpu_options_sent_to_cpu_plugin(plugin):
    """CPU plugins reject unknown create-options, so defaults must not apply."""
    with pypjrt.Client.create(plugin) as c:
        assert c._create_options == {}


def test_multiple_cpu_clients_allowed(plugin):
    """The live-client guard is accelerator-only."""
    with pypjrt.Client.create(plugin) as a, pypjrt.Client.create(plugin) as b:
        assert a.platform_name == b.platform_name == "cpu"


def test_compiled_memory_stats(plugin):
    with pypjrt.Client.create(plugin) as client:
        exe = client.compile(DOT)
        stats = exe.compiled_memory_stats()
        assert stats["argument_size_in_bytes"] == 2 * 64 * 64 * 4
        assert stats["output_size_in_bytes"] == 64 * 64 * 4
        assert stats["peak_memory_in_bytes"] > 0
        exe.close()


def test_optional_apis_degrade_cleanly(plugin):
    """A plugin that lacks an optional entry point must raise, not corrupt."""
    with pypjrt.Client.create(plugin) as client:
        exe = client.compile(DOT)
        try:
            exe.cost_analysis()
        except errors.Unimplemented:
            pass
        exe.close()
        with client.device(0) as dev:
            try:
                dev.memory_stats()
            except errors.PjrtError:
                pass


def test_memory_summary_is_safe_without_stats(plugin):
    """No stats -> empty summary, and diagnose_allocation re-raises untouched."""
    with pypjrt.Client.create(plugin) as client:
        assert isinstance(client.memory_summary(), str)
        with pytest.raises(errors.ResourceExhausted):
            with client.diagnose_allocation():
                raise errors.ResourceExhausted("original", 8)
