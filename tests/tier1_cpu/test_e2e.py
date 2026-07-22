"""Tier 1: the M0 gate. Needs a CPU PJRT plugin."""
import array, pytest, pypjrt
from pypjrt import errors

pytestmark = pytest.mark.tier1

MLIR = """
module @m {
  func.func public @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.add %a, %b : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""
F32 = 11


@pytest.fixture(scope="module")
def plugin(cpu_plugin_path):
    return pypjrt.Plugin(cpu_plugin_path)


def test_version_negotiation(plugin):
    major, minor = plugin.api_version
    assert major == 0
    # headers are newer than this plugin; we must degrade, not assert
    assert plugin.abi.PJRT_API_MINOR >= minor
    assert plugin.n_slots >= 1
    # slot count comes from the PLUGIN's struct_size, not our header's
    assert plugin.n_slots <= len(plugin.abi.SLOT)


def test_extension_probe_never_raises(plugin):
    assert plugin.extension("FFI") is not None
    assert plugin.extension("Megascale") is None      # absent -> None, not a crash
    assert plugin.extension(9999) is None
    with pytest.raises(errors.UnsupportedByPlugin):
        plugin.require_extension("Megascale")


def test_unavailable_slot_is_a_clean_error(plugin):
    """A function past the plugin's vtable must error, never jump wild."""
    newest = max(plugin.abi.SLOT.values())
    name = next(k for k, v in plugin.abi.SLOT.items() if v == newest)
    if newest < plugin.n_slots:
        pytest.skip("this plugin exposes the full vtable")
    with pytest.raises(errors.UnsupportedByPlugin):
        plugin.fn(name)


def test_m0_gate_end_to_end(plugin):
    with pypjrt.Client.create(plugin) as client:
        assert client.platform_name == "cpu"
        assert client.device_count >= 1
        exe = client.compile(MLIR)
        with client.device(0) as dev:
            a = client.buffer_from_host(array.array("f", [1, 2, 3, 4]), F32, [4], dev)
            b = client.buffer_from_host(array.array("f", [10, 20, 30, 40]), F32, [4], dev)
            assert a.dimensions == (4,) and a.element_type == F32 and a.nbytes == 16
            (out,) = exe(a, b)
            got = array.array("f"); got.frombytes(out.to_host())
            assert list(got) == [11.0, 22.0, 33.0, 44.0]
            for h in (out, a, b):
                h.close()
        exe.close()


def test_borrowed_device_cannot_escape(plugin):
    """The @local guarantee, one moment later."""
    with pypjrt.Client.create(plugin) as client:
        with client.device(0) as dev:
            assert dev.address
        with pytest.raises(errors.HandleClosed):
            dev.address


def test_owned_handle_close_is_idempotent(plugin):
    with pypjrt.Client.create(plugin) as client:
        with client.device(0) as dev:
            buf = client.buffer_from_host(array.array("f", [1.0]), F32, [1], dev)
        buf.close(); buf.close()          # idempotent
        with pytest.raises(errors.HandleClosed):
            buf.nbytes                     # but use-after-close raises
