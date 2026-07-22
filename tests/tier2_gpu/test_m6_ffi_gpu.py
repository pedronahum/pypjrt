"""Tier 2: M6 on a GPU -- real device pointers, XLA's stream, real device work."""
import array, pytest, pypjrt
import pypjrt.ffi as ffi

pytestmark = pytest.mark.tier2
F32, N = 11, 1024
cuda = pytest.importorskip("pypjrt.cuda")

FUSED = """
module @m {
  func.func public @main(%x: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = stablehlo.custom_call @t_dev_copy(%x) {
        api_version = 4 : i32, backend_config = {scale = 1 : i64}
      } : (tensor<1024xf32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
}
"""
IDENTITY = """
module @m {
  func.func public @main(%x: tensor<1024xf32>) -> tensor<1024xf32> {
    return %x : tensor<1024xf32>
  }
}
"""
_seen: dict = {}


@pytest.fixture(scope="module")
def plugin(gpu_plugin_path):
    if not cuda.available():
        pytest.skip("libcuda.so.1 not loadable")
    p = pypjrt.Plugin(gpu_plugin_path)

    @ffi.handler(p, "t_dev_copy")
    def _(call):
        (x,), (y,) = call.args, call.rets
        st = call.stream()
        _seen.update(stage=call.stage, stream=st, attrs=dict(call.attrs),
                     in_ptr=x.data, out_ptr=y.data, nbytes=x.nbytes)
        cuda.memcpy_dtod_async(y.data, x.data, x.nbytes, st)

    return p


def test_m6_gpu_gate(plugin):
    """A handler doing genuine device work on the stream XLA gave it, checked
    against a decompose oracle."""
    with pypjrt.Client.create(plugin, options={"preallocate": False,
                                               "memory_fraction": 0.05}) as c, \
            c.devices() as devs:
        xs = array.array("f", [i * 0.5 for i in range(N)])

        def run(src):
            e = c.compile(src)
            a = c.buffer_from_host(xs, F32, [N], devs[0])
            (o,) = e(a)
            g = array.array("f"); g.frombytes(o.to_host())
            o.close(); a.close(); e.close()
            return g

        got, want = run(FUSED), run(IDENTITY)

    assert _seen["stage"] == 3
    assert _seen["stream"] != 0, "XLA gave no stream"
    assert _seen["in_ptr"] and _seen["out_ptr"] and _seen["in_ptr"] != _seen["out_ptr"]
    assert _seen["nbytes"] == N * 4
    assert _seen["attrs"] == {"scale": 1}
    assert got.tobytes() == want.tobytes()
