"""Tier 2: the M1 gate. Needs a GPU PJRT plugin."""
import array, pytest, pypjrt
from pypjrt import errors

pytestmark = pytest.mark.tier2
F32 = 11
TIGHT = {"preallocate": False, "memory_fraction": 0.02}


@pytest.fixture(scope="module")
def plugin(gpu_plugin_path):
    return pypjrt.Plugin(gpu_plugin_path)


def test_accelerator_detected_from_attributes(plugin):
    """Detection is layered so it also classifies a TPU plugin, which
    publishes none of the GPU markers."""
    assert plugin.platform_hint == "gpu"
    assert plugin.is_accelerator is True and plugin.is_gpu is True
    assert any(k in plugin.attributes for k in plugin._GPU_MARKERS)


def test_gpu_defaults_applied_automatically(plugin):
    with pypjrt.Client.create(plugin) as c:
        assert c._create_options["preallocate"] is False
        assert c._create_options["memory_fraction"] == pytest.approx(0.5)


def test_second_accelerator_client_refused(plugin):
    with pypjrt.Client.create(plugin, options=TIGHT):
        with pytest.raises(errors.ResourceExhausted, match="already live"):
            pypjrt.Client.create(plugin, options=TIGHT)


def test_allow_multiple_is_an_escape_hatch(plugin):
    with pypjrt.Client.create(plugin, options=TIGHT):
        second = pypjrt.Client.create(plugin, options=TIGHT, allow_multiple=True)
        second.close()


def test_m1_gate_oom_message_is_actionable(plugin):
    """The gate: exhaust a deliberately tight allocator and demand a message
    carrying bytes_in_use / bytes_limit / largest_free_block_bytes."""
    with pypjrt.Client.create(plugin, options=TIGHT) as client:
        chunk = array.array("f", bytes(64 << 20))
        held = []
        try:
            with client.device(0) as dev:
                with pytest.raises(errors.ResourceExhausted) as ei:
                    for _ in range(400):
                        held.append(client.buffer_from_host(chunk, F32, [len(chunk)], dev))
            msg = str(ei.value)
            for field in ("bytes_in_use", "bytes_limit", "largest_free_block_bytes"):
                assert field in msg, f"{field} missing from OOM message"
            assert "create-options" in msg and "hint:" in msg
        finally:
            for b in held:
                b.close()


def test_device_memory_stats(plugin):
    with pypjrt.Client.create(plugin, options=TIGHT) as client:
        with client.device(0) as dev:
            stats = dev.memory_stats()
            assert "bytes_in_use" in stats and "bytes_limit" in stats


def test_newer_api_degrades_not_crashes(plugin):
    """PJRT_Device_ClearMemoryStats landed in 0.106. This box's CUDA plugin is
    0.104, so calling it must raise UnsupportedByPlugin -- not jump past the
    end of the plugin's vtable. Version negotiation, observed on real hardware.
    """
    with pypjrt.Client.create(plugin, options=TIGHT) as client:
        with client.device(0) as dev:
            if plugin.api_version >= (0, 106):
                dev.clear_memory_stats()
            else:
                with pytest.raises(errors.UnsupportedByPlugin, match="slots"):
                    dev.clear_memory_stats()


def test_cost_analysis(plugin):
    with pypjrt.Client.create(plugin, options=TIGHT) as client:
        exe = client.compile("""
module @m {
  func.func public @main(%a: tensor<128x128xf32>, %b: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}
""")
        ca = exe.cost_analysis()
        assert ca.get("flops", 0) > 0
        exe.close()
