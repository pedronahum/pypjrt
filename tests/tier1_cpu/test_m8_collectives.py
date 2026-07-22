"""Tier 1: collectives across devices (the compute half of M8)."""
import array, pytest, pypjrt
from pypjrt.compile_options import CompileOptions

pytestmark = pytest.mark.tier1
F32, N = 11, 4
ALLREDUCE = """
module @m {
  func.func public @main(%a: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "stablehlo.all_reduce"(%a) ({
      ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
        %s = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %s : tensor<f32>
    }) {replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""


@pytest.fixture(scope="module")
def client(cpu_plugin_path):
    c = pypjrt.Client.create(pypjrt.Plugin(cpu_plugin_path))
    if c.device_count < 2:
        c.close()
        pytest.skip("need >= 2 devices")
    yield c
    c.close()


def test_all_reduce_across_replicas(client):
    with client.devices() as devs:
        e = client.compile(ALLREDUCE, options=CompileOptions(num_replicas=2))
        assert e.num_replicas == 2 and e.addressable_device_count == 2
        assert e.device_assignment() == [(0, 0), (1, 0)]
        a = client.buffer_from_host(array.array("f", [1] * N), F32, [N], devs[0])
        b = client.buffer_from_host(array.array("f", [10] * N), F32, [N], devs[1])
        outs = e.execute_sharded([[a], [b]])
        for row in outs:
            g = array.array("f"); g.frombytes(row[0].to_host())
            assert list(g) == [11.0] * N, "all_reduce did not sum across replicas"
            row[0].close()
        a.close(); b.close(); e.close()


def test_kv_store_is_optional(client):
    """A single-process client needs no rendezvous."""
    assert client.kv_calls == {}


def test_client_accepts_a_kv_store_even_when_unused(cpu_plugin_path):
    """The CPU plugin at this pin reads only cpu_device_count and ignores the
    rendezvous callbacks; installing them must still be harmless."""
    store = pypjrt.InMemoryStore()
    c = pypjrt.Client.create(pypjrt.Plugin(cpu_plugin_path), kv_store=store)
    assert c.process_index == 0
    assert c.kv_calls == {"get": 0, "try_get": 0, "put": 0}
    c.close()
