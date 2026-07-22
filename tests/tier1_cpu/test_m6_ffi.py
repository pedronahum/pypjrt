"""Tier 1: M6 -- a Python function as an XLA FFI handler, on a CPU plugin."""
import array, math, pytest, pypjrt
import pypjrt.ffi as ffi
from pypjrt import errors

pytestmark = pytest.mark.tier1
F32, N = 11, 256

FUSED = """
module @m {
  func.func public @main(%x: tensor<256xf32>, %s: tensor<256xf32>) -> tensor<256xf32> {
    %0 = stablehlo.custom_call @%NAME%(%x, %s) {
        api_version = 4 : i32,
        backend_config = {alpha = 1.7 : f64, tag = "gelu", taps = array<i64: 3, 5>}
      } : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    return %0 : tensor<256xf32>
  }
}
"""
DECOMPOSED = """
module @m {
  func.func public @main(%x: tensor<256xf32>, %s: tensor<256xf32>) -> tensor<256xf32> {
    %half = stablehlo.constant dense<0.5> : tensor<256xf32>
    %one  = stablehlo.constant dense<1.0> : tensor<256xf32>
    %a    = stablehlo.constant dense<1.7> : tensor<256xf32>
    %sc   = stablehlo.multiply %x, %s : tensor<256xf32>
    %t    = stablehlo.tanh %sc : tensor<256xf32>
    %p    = stablehlo.add %t, %one : tensor<256xf32>
    %g    = stablehlo.multiply %sc, %p : tensor<256xf32>
    %h    = stablehlo.multiply %g, %half : tensor<256xf32>
    %r    = stablehlo.multiply %h, %a : tensor<256xf32>
    return %r : tensor<256xf32>
  }
}
"""
_seen: dict = {}


@pytest.fixture(scope="module")
def plugin(cpu_plugin_path):
    p = pypjrt.Plugin(cpu_plugin_path)

    @ffi.handler(p, "t_scaled_gelu")
    def _(call):
        _seen.update(stage=call.stage, attrs=dict(call.attrs), stream=call.stream())
        x, s = call.args
        (y,) = call.rets
        alpha = call.attrs["alpha"]
        xa, sa, ya = x.as_ctypes(), s.as_ctypes(), y.as_ctypes()
        for i in range(x.size):
            v = xa[i] * sa[i]
            ya[i] = alpha * 0.5 * v * (math.tanh(v) + 1.0)

    @ffi.handler(p, "t_boom")
    def _(call):
        raise ValueError("deliberate")

    return p


@pytest.fixture(scope="module")
def client(plugin):
    c = pypjrt.Client.create(plugin)
    yield c
    c.close()


def _run(client, src, name):
    with client.devices() as devs:
        e = client.compile(src.replace("%NAME%", name))
        xs = array.array("f", [i / N for i in range(N)])
        ss = array.array("f", [1.0 + i / (2 * N) for i in range(N)])
        a = client.buffer_from_host(xs, F32, [N], devs[0])
        b = client.buffer_from_host(ss, F32, [N], devs[0])
        (o,) = e(a, b)
        g = array.array("f"); g.frombytes(o.to_host())
        for h in (o, a, b):
            h.close()
        e.close()
        return g


def test_handler_is_registered(plugin):
    assert "t_scaled_gelu" in ffi.registered(plugin)


def test_duplicate_registration_refused(plugin):
    with pytest.raises(errors.AlreadyExists):
        ffi.register(plugin, "t_scaled_gelu", lambda call: None)


def test_handler_sees_execute_stage_and_decoded_attributes(client):
    _run(client, FUSED, "t_scaled_gelu")
    assert _seen["stage"] == 3                       # EXECUTE only; probe handled centrally
    assert _seen["attrs"] == {"alpha": 1.7, "tag": "gelu", "taps": [3, 5]}
    assert _seen["stream"] == 0                      # no stream on CPU


def test_m6_gate_decompose_oracle(client):
    """The gate: the fused custom call agrees with the same math in plain
    StableHLO, compiled on the same client."""
    got = _run(client, FUSED, "t_scaled_gelu")
    want = _run(client, DECOMPOSED, "unused")
    rel = max(abs(x - y) / max(abs(y), 1e-6) for x, y in zip(got, want))
    assert rel < 1e-4, f"max relative error {rel:.3e}"


def test_m6_gate_negative_control(client):
    """Without this, the positive result proves nothing."""
    with pytest.raises(errors.PjrtError, match="No FFI handler registered"):
        client.compile(FUSED.replace("%NAME%", "t_NEVER_REGISTERED"))


def test_handler_exception_fails_one_execution_not_the_process(client):
    with pytest.raises(errors.PjrtError, match="ValueError in FFI handler"):
        _run(client, FUSED, "t_boom")
    # the process, and the client, survive
    assert client.platform_name == "cpu"
    _run(client, FUSED, "t_scaled_gelu")


def test_handlers_may_not_reenter_pjrt(plugin, client):
    """Execute holds the process lock and XLA dispatches handlers inside it."""
    @ffi.handler(plugin, "t_reenter")
    def _(call):
        plugin.call("PJRT_Plugin_Initialize", plugin.args("PJRT_Plugin_Initialize_Args"))

    with pytest.raises(errors.PjrtError, match="RuntimeError in FFI handler"):
        _run(client, FUSED, "t_reenter")
