"""Tier 1: memory spaces, buffer movement, ExecuteContext."""
import pytest, pypjrt
from pypjrt import errors
from pypjrt.client import ExecuteContext, Memory
from pypjrt.typing import F32

pytestmark = pytest.mark.tier1
np = pytest.importorskip("numpy")
N = 8


@pytest.fixture(scope="module")
def plugin(cpu_plugin_path):
    return pypjrt.Plugin(cpu_plugin_path)


@pytest.fixture(scope="module")
def client(plugin):
    c = pypjrt.Client.create(plugin)
    yield c
    c.close()


def test_device_exposes_memory_spaces(client):
    with client.devices() as devs:
        mems = devs[0].memories()
        assert mems, "no memory spaces reported"
        kinds = {m.kind for m in mems}
        assert "device" in kinds
        for m in mems:
            assert isinstance(m.id, int)
            assert m.kind and isinstance(m.kind_id, int)
            assert m.debug_string


def test_default_memory_is_one_of_them(client):
    with client.devices() as devs:
        d = devs[0]
        assert d.default_memory().kind in {m.kind for m in d.memories()}


def test_buffer_reports_its_placement(client):
    with client.devices() as devs:
        b = client.typed_buffer(F32, np.arange(N, dtype=np.float32), [N], devs[0])
        assert b.memory().kind == devs[0].default_memory().kind
        assert isinstance(b.is_on_cpu(), bool)
        assert b.device().id == devs[0].id
        b.close()


def test_copy_between_memory_spaces_preserves_data(client):
    """Staging to host memory is how weights move without a host roundtrip."""
    src = np.arange(N, dtype=np.float32)
    with client.devices() as devs:
        b = client.typed_buffer(F32, src, [N], devs[0])
        others = [m for m in devs[0].memories() if m.kind != b.memory().kind]
        if not others:
            pytest.skip("plugin exposes a single memory space")
        moved = 0
        for m in others:
            try:
                b2 = b.copy_to_memory(m)
            except errors.PjrtError:
                continue          # a space this plugin cannot target
            got = np.empty(N, dtype=np.float32)
            b2.to_host(got)
            assert np.array_equal(got, src), f"data lost moving to {m.kind}"
            assert b2.memory().kind == m.kind
            b2.close()
            moved += 1
        assert moved, "no reachable alternate memory space"
        b.close()


def test_copy_to_device(client):
    if client.device_count < 2:
        pytest.skip("need >= 2 devices")
    src = np.arange(N, dtype=np.float32)
    with client.devices() as devs:
        b = client.typed_buffer(F32, src, [N], devs[0])
        b2 = b.copy_to_device(devs[1])
        got = np.empty(N, dtype=np.float32)
        b2.to_host(got)
        assert np.array_equal(got, src)
        b2.close(); b.close()


def test_memory_handle_invalidation(client):
    with client.devices() as devs:
        m = devs[0].default_memory()
        assert m.kind
        m._invalidate()
        with pytest.raises(errors.HandleClosed):
            m.kind


def test_execute_context_can_be_created_and_passed(client, plugin):
    """Required by the FFI extension's user_data_add, i.e. stateful handlers."""
    ctx = ExecuteContext.create(plugin)
    assert ctx.address
    with client.devices() as devs:
        e = client.compile("""
module @m {
  func.func public @main(%a: tensor<8xf32>) -> tensor<8xf32> {
    %0 = stablehlo.add %a, %a : tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}""")
        b = client.typed_buffer(F32, np.arange(N, dtype=np.float32), [N], devs[0])
        (o,) = e.execute_sharded([[b]], context=ctx)[0]
        got = np.empty(N, dtype=np.float32)
        o.to_host(got)
        assert np.array_equal(got, np.arange(N) * 2)
        o.close(); b.close(); e.close()
    ctx.close()
    ctx.close()          # idempotent
