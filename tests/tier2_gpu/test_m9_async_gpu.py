"""Tier 2: M9 on a GPU plugin."""
import array, threading, pytest, pypjrt
from pypjrt.client import Event

pytestmark = pytest.mark.tier2
F32, N = 11, 256
TIGHT = {"preallocate": False, "memory_fraction": 0.05}
MUL = """
module @m {
  func.func public @main(%a: tensor<256xf32>, %b: tensor<256xf32>) -> tensor<256xf32> {
    %0 = stablehlo.multiply %a, %b : tensor<256xf32>
    return %0 : tensor<256xf32>
  }
}
"""


@pytest.fixture(scope="module")
def plugin(gpu_plugin_path):
    return pypjrt.Plugin(gpu_plugin_path)


def test_host_minted_event_and_cross_thread_delivery(plugin):
    ev = Event.create(plugin)
    fired = threading.Event()
    tids = {"main": threading.get_ident()}
    ev.add_done_callback(lambda m: (tids.update(cb=threading.get_ident()), fired.set()))
    t = threading.Thread(target=lambda: (tids.update(setter=threading.get_ident()), ev.set()))
    t.start(); t.join()
    assert fired.wait(5)
    assert tids["cb"] == tids["setter"] != tids["main"]
    ev.close()


def test_device_event_callback(plugin):
    with pypjrt.Client.create(plugin, options=TIGHT) as c, c.devices() as devs:
        exe = c.compile(MUL)
        xs = array.array("f", [2.0] * N)
        a = c.buffer_from_host(xs, F32, [N], devs[0])
        b = c.buffer_from_host(xs, F32, [N], devs[0])
        (o,) = exe(a, b)
        r = plugin.args("PJRT_Buffer_ReadyEvent_Args", buffer=o.address)
        plugin.call("PJRT_Buffer_ReadyEvent", r)
        ev = Event(plugin, r.event)
        fired = threading.Event()
        seen = {}
        ev.add_done_callback(lambda m: (seen.update(msg=m), fired.set()))
        assert fired.wait(30), "device ready-event callback never fired"
        assert seen["msg"] is None
        ev.close()
        for h in (o, a, b):
            h.close()
        exe.close()
