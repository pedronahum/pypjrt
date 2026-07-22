"""Tier 1: chunked async transfer, Layouts and Stream extensions."""
import pytest, pypjrt
from pypjrt import errors, extensions as ext
from pypjrt.transfer import ShapeSpec
from pypjrt.typing import F32

pytestmark = pytest.mark.tier1
np = pytest.importorskip("numpy")
ROWS, COLS = 64, 256
MUL = """
module @m {
  func.func public @main(%a: tensor<4x2xf32>, %b: tensor<4x2xf32>) -> tensor<4x2xf32> {
    %0 = stablehlo.multiply %a, %b : tensor<4x2xf32>
    return %0 : tensor<4x2xf32>
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


# -- async host-to-device ---------------------------------------------------


def test_chunked_transfer_reassembles_exactly(client):
    """Stream a large array in pieces without a full host staging copy --
    the overlap a blocking host copy gives up."""
    src = np.arange(ROWS * COLS, dtype=np.float32).reshape(ROWS, COLS)
    with client.devices() as devs:
        with client.async_transfer([ShapeSpec(F32.code, (ROWS, COLS))],
                                   device=devs[0]) as t:
            assert len(t) == 1
            assert t.buffer_size(0) >= src.nbytes
            chunks = 8
            step = ROWS // chunks
            for i in range(chunks):
                piece = src[i * step:(i + 1) * step]
                t.transfer(0, piece, offset=i * piece.nbytes, last=(i == chunks - 1))
            (buf,) = t.buffers()
            got = np.empty_like(src)
            buf.to_host(got)
            assert np.array_equal(got, src)
            assert buf.dimensions == (ROWS, COLS)
            buf.close()


def test_multiple_buffers_in_one_transfer(client):
    a = np.arange(16, dtype=np.float32)
    b = (np.arange(16, dtype=np.float32) * 3)
    with client.devices() as devs:
        with client.async_transfer(
                [ShapeSpec(F32.code, (16,)), ShapeSpec(F32.code, (16,))],
                device=devs[0]) as t:
            assert len(t) == 2
            t.transfer(0, a); t.transfer(1, b)
            ba, bb = t.buffers()
            ga, gb = np.empty(16, np.float32), np.empty(16, np.float32)
            ba.to_host(ga); bb.to_host(gb)
            assert np.array_equal(ga, a) and np.array_equal(gb, b)
            ba.close(); bb.close()


def test_set_error_reaches_the_consumer(client):
    """An aborted transfer must surface as an error, not as garbage."""
    with client.devices() as devs:
        with client.async_transfer([ShapeSpec(F32.code, (16,))], device=devs[0]) as t:
            t.set_error(0, 13, "deliberate")
            b = t.retrieve(0)
            with pytest.raises(errors.PjrtError, match="deliberate"):
                b.to_host(np.empty(16, np.float32))
            b.close()


def test_transfer_requires_a_destination(client):
    with pytest.raises(errors.InvalidArgument, match="memory=|device="):
        client.async_transfer([ShapeSpec(F32.code, (4,))])


def test_closed_transfer_refuses_use(client):
    with client.devices() as devs:
        t = client.async_transfer([ShapeSpec(F32.code, (4,))], device=devs[0])
        t.close()
        t.close()                       # idempotent
        with pytest.raises(errors.HandleClosed):
            t.retrieve(0)


# -- Layouts ----------------------------------------------------------------


def test_buffer_and_default_layouts_agree(client, plugin):
    if not ext.layouts_available(plugin):
        pytest.skip("plugin does not advertise Layouts")
    with client.devices() as devs:
        b = client.typed_buffer(F32, np.zeros((4, 2), np.float32), [4, 2], devs[0])
        with ext.buffer_layout(b) as bl, ext.default_layout(client, F32.code, (4, 2)) as dl:
            assert bl.serialize() == dl.serialize() == b"{1,0}"
        b.close()


def test_executable_layouts_are_not_owned_by_us(client, plugin):
    """They live inside the PJRT_Executable; destroying one corrupts the heap.
    Regression guard for a real `free(): invalid size`."""
    if not ext.layouts_available(plugin):
        pytest.skip("plugin does not advertise Layouts")
    e = client.compile(MUL)
    params = ext.executable_layouts(e, "parameters")
    outs = ext.executable_layouts(e, "outputs")
    assert len(params) == 2 and len(outs) == 1
    assert all(l.serialize() == b"{1,0}" for l in params + outs)
    assert not any(l._owned for l in params + outs)
    for l in params + outs:
        l.close()                       # must be a no-op, not a free
        assert l.serialize() == b"{1,0}"
    e.close()


# -- Stream -----------------------------------------------------------------


def test_stream_extension_is_probed_not_assumed(client, plugin):
    """Both outcomes are passes: present means it works, absent means the probe
    refuses cleanly. Skipping here would mean testing nothing in a required
    tier -- which the loud-skip guard caught on the first run."""
    with client.devices() as devs:
        if ext.stream_available(plugin):
            st = ext.device_stream(devs[0])
            b = client.typed_buffer(F32, np.zeros(4, np.float32), [4], devs[0])
            ext.wait_for_buffer(b, st)
            b.close()
        else:
            with pytest.raises(errors.UnsupportedByPlugin, match="Stream"):
                ext.device_stream(devs[0])
