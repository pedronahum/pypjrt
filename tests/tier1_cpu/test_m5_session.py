"""Tier 1: M5 -- session, slots, donation, kill -9 resume."""
import array, hashlib, json, os, subprocess, sys, textwrap
import pytest, pypjrt
from pypjrt import errors
from pypjrt.session import Session, Slot

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
DONATING = """
module @m {
  func.func public @main(%theta: tensor<64xf32> {tf.aliasing_output = 0 : i32},
                         %grad: tensor<64xf32>) -> tensor<64xf32> {
    %0 = stablehlo.subtract %theta, %grad : tensor<64xf32>
    return %0 : tensor<64xf32>
  }
}
"""


@pytest.fixture(scope="module")
def client(cpu_plugin_path):
    c = pypjrt.Client.create(pypjrt.Plugin(cpu_plugin_path))
    yield c
    c.close()


def _buf(c, dev, vals, dims=(N,)):
    return c.buffer_from_host(array.array("f", vals), F32, list(dims), dev)


# -- donation ---------------------------------------------------------------


def test_donation_is_detected_and_counted(client):
    with client.devices() as devs:
        e = client.compile(DONATING)
        theta = _buf(client, devs[0], [1.0] * N)
        grad = _buf(client, devs[0], [0.25] * N)
        (out,) = e(theta, grad)
        assert e.donate_alias_count == 1
        assert theta.is_deleted, "runtime did not consume the donated input"
        g = array.array("f"); g.frombytes(out.to_host())
        assert g[0] == pytest.approx(0.75)
        for h in (out, theta, grad):
            h.close()
        e.close()


def test_donated_input_cannot_be_reused(client):
    """A use-after-donate is a clear error, not a use-after-free."""
    with client.devices() as devs:
        e = client.compile(DONATING)
        theta = _buf(client, devs[0], [1.0] * N)
        grad = _buf(client, devs[0], [0.25] * N)
        (out,) = e(theta, grad)
        with pytest.raises(errors.HandleClosed, match="donated"):
            theta.to_host()
        for h in (out, theta, grad):
            h.close()          # still safe to destroy the wrapper
        e.close()


def test_no_donation_without_the_attribute(client):
    with client.devices() as devs:
        e = client.compile(MUL)
        a, b = _buf(client, devs[0], [1.0] * N), _buf(client, devs[0], [2.0] * N)
        (out,) = e(a, b)
        assert e.donate_alias_count == 0
        assert not a.is_deleted
        for h in (out, a, b):
            h.close()
        e.close()


def test_non_donatable_suppresses_donation(client):
    with client.devices() as devs:
        e = client.compile(DONATING)
        theta = _buf(client, devs[0], [1.0] * N)
        grad = _buf(client, devs[0], [0.25] * N)
        (out,) = e(theta, grad, non_donatable=(0,))
        assert e.donate_alias_count == 0
        assert not theta.is_deleted
        assert array.array("f", theta.to_host())[0] == pytest.approx(1.0)
        for h in (out, theta, grad):
            h.close()
        e.close()


def test_buffer_delete_releases_memory_but_keeps_the_handle(client):
    with client.devices() as devs:
        b = _buf(client, devs[0], [1.0] * N)
        assert not b.is_deleted
        b.delete()
        assert b.is_deleted
        b.close()


# -- session ----------------------------------------------------------------


def test_named_io_and_registry_resolution(client):
    with Session(client) as s:
        prog = s.program(MUL, ["a", "b"], outputs=["y"])
        s.set_global("b", _buf(client, s.device(), [3.0] * N))
        out = prog(a=_buf(client, s.device(), [2.0] * N))    # b comes from the registry
        assert set(out) == {"y"}
        assert array.array("f", out["y"].to_host())[0] == pytest.approx(6.0)
        out["y"].close()
        prog.close()


def test_unknown_and_missing_inputs_are_clear_errors(client):
    with Session(client) as s:
        prog = s.program(MUL, ["a", "b"])
        with pytest.raises(errors.InvalidArgument, match="unknown input"):
            prog(a=_buf(client, s.device(), [1.0] * N), nope=_buf(client, s.device(), [1.0] * N))
        with pytest.raises(errors.InvalidArgument, match="no buffer for input 'b'"):
            prog(a=_buf(client, s.device(), [1.0] * N))
        prog.close()


def test_slot_validates_dtype_and_shape(client):
    with Session(client) as s:
        prog = s.program(MUL, [Slot("a", F32, (N,)), Slot("b", F32, (N,))])
        bad = client.buffer_from_host(array.array("f", [1.0] * 8), F32, [8], s.device())
        with pytest.raises(errors.InvalidArgument, match="expected dims"):
            prog(a=bad, b=bad)
        bad.close()
        prog.close()


def test_lazy_binding_materialises_only_what_is_named(client):
    with Session(client) as s:
        calls = []

        def make(name, val):
            def loader(dev):
                calls.append(name)
                return _buf(client, dev, [val] * N)
            return loader

        s.bind_many({"a": make("a", 2.0), "b": make("b", 5.0), "unused": make("unused", 9.0)})
        assert s.resident == [] and s.pending == ["a", "b", "unused"]

        prog = s.program(MUL, ["a", "b"], outputs=["y"])
        out = prog()
        assert sorted(calls) == ["a", "b"], "an unnamed slot was materialised"
        assert s.pending == ["unused"]
        assert array.array("f", out["y"].to_host())[0] == pytest.approx(10.0)

        out["y"].close()
        prog.close()


def test_feed_back_keeps_the_buffer_on_device(client):
    """Output becomes the next input with no host roundtrip."""
    with Session(client) as s:
        prog = s.program(MUL, ["a", "b"], outputs=["a"])
        s.set_global("a", _buf(client, s.device(), [1.0] * N))
        s.set_global("b", _buf(client, s.device(), [2.0] * N))
        for _ in range(5):
            s.feed_back(prog())
        assert array.array("f", s._globals["a"].to_host())[0] == pytest.approx(32.0)
        prog.close()


# -- the gate ---------------------------------------------------------------

_WORKER = textwrap.dedent('''
    import sys, os, array, hashlib, json, pathlib
    sys.path.insert(0, {src!r})
    import pypjrt
    from pypjrt.session import Session, Slot
    F32, N = 11, 64
    STEP = """{step}"""
    state, start, stop, crash_at = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    p = pypjrt.Plugin({plugin!r})
    with pypjrt.Client.create(p) as c, Session(c) as s:
        prog = s.program(STEP, [Slot("theta", F32, (N,)), Slot("grad", F32, (N,))], outputs=["theta"])
        theta = (array.array("f", json.loads(state.read_text())["theta"])
                 if state.exists() else array.array("f", [1.0] * N))
        s.set_global("theta", c.buffer_from_host(theta, F32, [N], s.device()))
        s.set_global("grad", c.buffer_from_host(array.array("f", [0.01] * N), F32, [N], s.device()))
        for step in range(start, stop):
            s.feed_back(prog())
            if crash_at >= 0 and step == crash_at:
                h = s._globals["theta"].to_host()
                state.write_text(json.dumps({{"theta": list(array.array("f", h))}}))
                os.kill(os.getpid(), 9)
        h = s._globals["theta"].to_host()
        state.write_text(json.dumps({{"theta": list(array.array("f", h))}}))
        print(hashlib.sha256(h).hexdigest(), prog.donate_alias_count)
        prog.close()
''')


def _worker(tmp_path, cpu_plugin_path, state, start, stop, crash_at):
    src = str((__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
    script = tmp_path / f"w{start}_{stop}_{crash_at}.py"
    script.write_text(_WORKER.format(src=src, step=DONATING.replace('"""', ''),
                                     plugin=str(cpu_plugin_path)))
    return subprocess.run([sys.executable, str(script), str(state), str(start), str(stop),
                           str(crash_at)], capture_output=True, text=True)


def test_m5_gate_donating_loop_survives_kill9(tmp_path, cpu_plugin_path):
    """The gate: a donating training step, killed with SIGKILL mid-run, resumes
    in a fresh process to a byte-identical checkpoint."""
    ref = _worker(tmp_path, cpu_plugin_path, tmp_path / "ref.json", 0, 20, -1)
    assert ref.returncode == 0, ref.stderr[-800:]
    want_hash, want_donations = ref.stdout.split()
    assert int(want_donations) == 20

    crashed = _worker(tmp_path, cpu_plugin_path, tmp_path / "run.json", 0, 20, 9)
    assert crashed.returncode == -9, f"expected SIGKILL, got {crashed.returncode}"

    resumed = _worker(tmp_path, cpu_plugin_path, tmp_path / "run.json", 10, 20, -1)
    assert resumed.returncode == 0, resumed.stderr[-800:]
    got_hash, got_donations = resumed.stdout.split()

    assert got_hash == want_hash, "resumed checkpoint differs from the uninterrupted run"
    assert int(got_donations) == 10
