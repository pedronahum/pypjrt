"""Tier 1: M3 -- multi-device execute. No GPU needed; the CPU plugin exposes 4."""
import array, pytest, pypjrt
from pypjrt import errors
from pypjrt.compile_options import CompileOptions

pytestmark = pytest.mark.tier1
F32, N = 11, 8

MUL = """
module @m {
  func.func public @main(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
    %0 = stablehlo.multiply %a, %b : tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
"""
MUL_SHARDED = """
module @m {
  func.func public @main(
      %a: tensor<8xf32> {mhlo.sharding = "{devices=[2]<=[2]}"},
      %b: tensor<8xf32> {mhlo.sharding = "{devices=[2]<=[2]}"}
    ) -> (tensor<8xf32> {mhlo.sharding = "{devices=[2]<=[2]}"}) {
    %0 = stablehlo.multiply %a, %b : tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
"""
XS = [float(i + 1) for i in range(N)]
YS = [float(10 * (i + 1)) for i in range(N)]
EXPECT = [x * y for x, y in zip(XS, YS)]


@pytest.fixture(scope="module")
def client(cpu_plugin_path):
    c = pypjrt.Client.create(pypjrt.Plugin(cpu_plugin_path))
    if c.device_count < 2:
        c.close()
        pytest.skip("need >= 2 addressable devices")
    yield c
    c.close()


def test_default_device_assignment(client):
    assert client.default_device_assignment(2, 1) == [0, 1]
    assert client.default_device_assignment(1, 2) == [0, 1]
    assert len(client.default_device_assignment(2, 2)) == 4


def test_compile_options_drive_the_mesh(client):
    """Proof the proto encoder is real: the plugin reports what we asked for."""
    for nr, np_ in ((2, 1), (1, 2)):
        opts = CompileOptions(num_replicas=nr, num_partitions=np_,
                              use_spmd_partitioning=np_ > 1)
        e = client.compile(MUL if np_ == 1 else MUL_SHARDED, options=opts)
        assert (e.num_replicas, e.num_partitions) == (nr, np_)
        assert e.addressable_device_count == nr * np_
        e.close()


def test_replicated_execution_across_two_devices(client):
    with client.devices() as devs:
        e = client.compile(MUL, options=CompileOptions(num_replicas=2))
        rows = [[client.buffer_from_host(array.array("f", XS), F32, [N], devs[0]),
                 client.buffer_from_host(array.array("f", YS), F32, [N], devs[0])],
                [client.buffer_from_host(array.array("f", YS), F32, [N], devs[1]),
                 client.buffer_from_host(array.array("f", XS), F32, [N], devs[1])]]
        outs = e.execute_sharded(rows, launch_id=3)
        assert len(outs) == 2
        for row in outs:
            g = array.array("f"); g.frombytes(row[0].to_host())
            assert list(g) == EXPECT      # a*b == b*a
            row[0].close()
        for r in rows:
            for b in r:
                b.close()
        e.close()


@pytest.mark.parametrize("shardy", [False, True])
def test_m3_gate_sharded_matches_single_device(client, shardy):
    """The gate: a 2-way-sharded run is byte-identical to the single-device run."""
    with client.devices() as devs:
        ref_exe = client.compile(MUL, options=CompileOptions())
        a = client.buffer_from_host(array.array("f", XS), F32, [N], devs[0])
        b = client.buffer_from_host(array.array("f", YS), F32, [N], devs[0])
        (o,) = ref_exe(a, b)
        ref = o.to_host()
        for h in (o, a, b):
            h.close()
        ref_exe.close()

        e = client.compile(MUL_SHARDED, options=CompileOptions(
            num_partitions=2, use_spmd_partitioning=True, use_shardy_partitioner=shardy))
        half = N // 2
        ba = client.buffers_from_host(
            [array.array("f", XS[:half]), array.array("f", XS[half:])], F32, [half], devs[:2])
        bb = client.buffers_from_host(
            [array.array("f", YS[:half]), array.array("f", YS[half:])], F32, [half], devs[:2])
        outs = e.execute_sharded([[ba[0], bb[0]], [ba[1], bb[1]]])
        got = b"".join(row[0].to_host() for row in outs)
        for row in outs:
            row[0].close()
        for h in ba + bb:
            h.close()
        e.close()
        assert got == bytes(ref), "sharded result differs from the single-device reference"


def test_device_assignment_reported_per_device(client):
    e = client.compile(MUL_SHARDED, options=CompileOptions(
        num_partitions=2, use_spmd_partitioning=True))
    assert e.device_assignment() == [(0, 0), (0, 1)]
    e.close()
    e = client.compile(MUL, options=CompileOptions(num_replicas=2))
    assert e.device_assignment() == [(0, 0), (1, 0)]
    e.close()


def test_arity_guard(client):
    """Passing the wrong number of device rows must be a clear error."""
    with client.devices() as devs:
        e = client.compile(MUL, options=CompileOptions(num_replicas=2))
        a = client.buffer_from_host(array.array("f", XS), F32, [N], devs[0])
        with pytest.raises(errors.InvalidArgument, match="expects 2 device"):
            e.execute_sharded([[a, a]])
        a.close()
        e.close()
