"""Tier 1: M9 -- async completion behind the future shape fixed in M0."""
import array, threading, pytest, pypjrt
from pypjrt import errors
from pypjrt.client import Event, _ONREADY

pytestmark = pytest.mark.tier1
F32, N = 11, 64
MUL = """
module @m {
  func.func public @main(%a: tensor<64xf32>, %b: tensor<64xf32>) -> tensor<64xf32> {
    %0 = stablehlo.multiply %a, %b : tensor<64xf32>
    return %0 : tensor<64xf32>
  }
}
"""


@pytest.fixture(scope="module")
def plugin(cpu_plugin_path):
    return pypjrt.Plugin(cpu_plugin_path)


@pytest.fixture(scope="module")
def client(plugin):
    c = pypjrt.Client.create(plugin)
    yield c
    c.close()


# -- the M0 shape is unchanged ---------------------------------------------


def test_future_api_surface_is_the_one_shipped_in_m0():
    for m in ("result", "done", "add_done_callback", "consume"):
        assert callable(getattr(Event, m)), m


# -- host-minted events ------------------------------------------------------


def test_event_create_starts_incomplete_and_completes_on_set(plugin):
    ev = Event.create(plugin)
    assert ev.done() is False
    ev.set()
    assert ev.done() is True
    ev.result()          # must not raise
    ev.close()


def test_callback_registered_before_completion_fires_after_set(plugin):
    ev = Event.create(plugin)
    fired = threading.Event()
    seen = {}
    ev.add_done_callback(lambda msg: (seen.update(msg=msg), fired.set()))
    assert not fired.is_set(), "callback fired before the event completed"
    ev.set()
    assert fired.wait(5), "callback never fired"
    assert seen["msg"] is None
    ev.close()


def test_callback_lands_on_the_completing_thread(plugin):
    """The contract: the callback runs wherever the event completes, not on
    the thread that registered it."""
    ev = Event.create(plugin)
    fired = threading.Event()
    tids = {"main": threading.get_ident()}
    ev.add_done_callback(lambda m: (tids.update(cb=threading.get_ident()), fired.set()))

    def setter():
        tids["setter"] = threading.get_ident()
        ev.set()

    t = threading.Thread(target=setter)
    t.start(); t.join()
    assert fired.wait(5)
    assert tids["cb"] == tids["setter"] != tids["main"]
    ev.close()


def test_error_event_raises_and_reports_through_the_callback(plugin):
    ev = Event.create(plugin)
    ev.set(error_code=3, message="deliberate failure")
    with pytest.raises(errors.InvalidArgument, match="deliberate failure"):
        ev.result()
    ev.close()

    ev2 = Event.create(plugin)
    fired = threading.Event()
    seen = {}
    ev2.add_done_callback(lambda msg: (seen.update(msg=msg), fired.set()))
    ev2.set(error_code=3, message="callback sees this")
    assert fired.wait(5)
    assert seen["msg"] and "callback sees this" in seen["msg"]
    ev2.close()


def test_callback_on_an_already_complete_event_still_fires(plugin):
    ev = Event.create(plugin)
    ev.set()
    fired = threading.Event()
    ev.add_done_callback(lambda m: fired.set())
    assert fired.wait(5)
    ev.close()


def test_trampolines_do_not_accumulate(plugin):
    before = len(_ONREADY)
    for _ in range(20):
        ev = Event.create(plugin)
        done = threading.Event()
        ev.add_done_callback(lambda m: done.set())
        ev.set()
        assert done.wait(5)
        ev.close()
    assert len(_ONREADY) == before, "OnReady trampolines leaked"


def test_a_raising_callback_does_not_break_the_plugin(plugin):
    ev = Event.create(plugin)
    ev.add_done_callback(lambda m: (_ for _ in ()).throw(ValueError("boom")))
    ev.set()             # must not propagate into the plugin's thread
    ev.close()
    ev2 = Event.create(plugin)
    ev2.set()
    assert ev2.done()
    ev2.close()


# -- real device events -----------------------------------------------------


def test_device_ready_event_completes(client, plugin):
    with client.devices() as devs:
        exe = client.compile(MUL)
        xs = array.array("f", [1.5] * N)
        a = client.buffer_from_host(xs, F32, [N], devs[0])
        b = client.buffer_from_host(xs, F32, [N], devs[0])
        (o,) = exe(a, b)
        r = plugin.args("PJRT_Buffer_ReadyEvent_Args", buffer=o.address)
        plugin.call("PJRT_Buffer_ReadyEvent", r)
        ev = Event(plugin, r.event)
        fired = threading.Event()
        seen = {}
        ev.add_done_callback(lambda m: (seen.update(msg=m), fired.set()))
        assert fired.wait(10)
        assert seen["msg"] is None
        ev.close()
        for h in (o, a, b):
            h.close()
        exe.close()
