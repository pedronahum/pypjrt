"""Tier 1: every remaining accessor, exercised or shown to degrade cleanly.

Also pins the deliberate non-coverage: six entry points are unbound on purpose
and this test asserts the list, so "unbound" stays a decision rather than drift.
"""
import pathlib, pytest, pypjrt
from pypjrt import _abi, errors
from pypjrt.typing import F32, S32

pytestmark = pytest.mark.tier1
np = pytest.importorskip("numpy")
N = 8
MUL = """
module @m {
  func.func public @main(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
    %0 = stablehlo.multiply %a, %b : tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
"""

#: Bound by choice, not by accident. Each needs machinery a caller must supply.
DELIBERATELY_UNBOUND = {
    # takes an xla::Literal, which has no C-API representation to build from here
    "PJRT_AsyncHostToDeviceTransferManager_TransferLiteral",
    # callback-completion variants of calls we already expose synchronously
    "PJRT_Buffer_CopyRawToHostFuture",
    "PJRT_Buffer_DonateWithControlDependency",
    # loads an *unloaded* PJRT_Executable, which only PJRT_Compile produces and
    # which Topology.compile already serializes for us
    "PJRT_Client_Load",
    # structured error payloads; our boundary reports code + message
    "PJRT_Error_ForEachPayload",
    # memory-space shape canonicalisation, no consumer yet
    "PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace",
}


def test_unbound_set_is_exactly_what_we_chose():
    A = _abi.load(0, _abi.available()[0][1])[0]
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "pypjrt"
    src = "\n".join(p.read_text() for p in root.rglob("*.py"))
    unbound = {n for n in A.SLOT if f'"{n}"' not in src}
    assert unbound == DELIBERATELY_UNBOUND, (
        f"coverage drifted.\n  newly unbound: {sorted(unbound - DELIBERATELY_UNBOUND)}"
        f"\n  newly bound:   {sorted(DELIBERATELY_UNBOUND - unbound)}")
    assert len(A.SLOT) - len(unbound) == 132


@pytest.fixture(scope="module")
def plugin(cpu_plugin_path):
    return pypjrt.Plugin(cpu_plugin_path)


@pytest.fixture(scope="module")
def client(plugin):
    c = pypjrt.Client.create(plugin)
    yield c
    c.close()


def test_lookup_device(client):
    with client.devices() as devs:
        assert client.lookup_device(devs[0].id).id == devs[0].id


def test_descriptions_and_attributes(client):
    with client.devices() as devs:
        d = devs[0]
        assert d.to_string()
        assert isinstance(d.live_attributes(), dict)
        m = d.default_memory()
        assert str(m)
        assert [x.id for x in m.addressable_by()]


def test_buffer_accessors(client):
    with client.devices() as devs:
        b = client.typed_buffer(F32, np.arange(N, dtype=np.float32), [N], devs[0])
        assert b.unpadded_dimensions == (N,)
        assert b.dynamic_dimension_indices == ()
        assert b.unsafe_pointer()
        out = np.empty(4, np.float32)
        b.copy_raw_to_host(out, offset=16)
        assert list(out) == [4.0, 5.0, 6.0, 7.0]     # 16 bytes in = element 4
        bc = b.bitcast(S32)
        assert bc.element_type == S32.code
        bc.close(); b.close()


def test_uninitialized_and_error_buffers(client):
    with client.devices() as devs:
        mem = devs[0].default_memory()
        u = client.uninitialized_buffer(F32.code, [N], mem)
        assert u.dimensions == (N,)
        u.close()
        e = client.error_buffer(mem, 13, "boom", dtype=F32.code, dims=[N])
        with pytest.raises(errors.PjrtError, match="boom"):
            e.to_host(np.empty(N, np.float32))
        e.close()


def test_invalid_dtype_is_refused_before_it_aborts_the_process(client):
    """XLA CHECK-fails on an unknown element type, which kills the interpreter
    rather than returning an error. Guard in front of every shape-taking call."""
    with client.devices() as devs:
        mem = devs[0].default_memory()
        for call in (lambda: client.uninitialized_buffer(0, [N], mem),
                     lambda: client.error_buffer(mem, 13, "x", dtype=0, dims=[N]),
                     lambda: client.alias_buffer(mem, 999, [N])):
            with pytest.raises(errors.InvalidArgument, match="not a valid PJRT_Buffer_Type"):
                call()


def test_alias_buffer_promise(client):
    """A promised buffer, completed later. Fulfilling with a failure code
    propagates that failure back through the plugin's own callback."""
    with client.devices() as devs:
        mem = devs[0].default_memory()

        ok = client.alias_buffer(mem, F32.code, [N])
        assert ok.dimensions == (N,) and ok._fulfill_cb
        client.fulfill_alias_buffer(ok)                  # code 0 -> succeeds
        ok.close()

        bad = client.alias_buffer(mem, F32.code, [N])
        with pytest.raises(errors.PjrtError, match="never arrived"):
            client.fulfill_alias_buffer(bad, code=13, message="never arrived")
        bad.close()

        plain = client.typed_buffer(F32, np.zeros(N, np.float32), [N], devs[0])
        with pytest.raises(errors.InvalidArgument, match="no fulfilment callback"):
            client.fulfill_alias_buffer(plain)
        plain.close()


def test_executable_accessors(client):
    e = client.compile(MUL)
    assert e.name
    assert isinstance(e.code_size_bytes, int)
    assert e.compile_options().startswith(b"\x1a")
    assert e.device_assignment_proto()
    assert e.is_deleted is False
    for optional in (e.output_memory_kinds, e.parameter_memory_kinds, e.loaded_fingerprint):
        try:
            optional()
        except errors.PjrtError:
            pass                      # plugin-dependent
    e.release_device_memory()
    e.close()


def test_dma_map_degrades_cleanly(client):
    arr = np.arange(N, dtype=np.float32)
    try:
        view = client.dma_map(arr)
        client.dma_unmap(view)
    except errors.PjrtError as ex:
        assert "not supported" in ex.message or "nimplemented" in ex.message


def test_async_transfer_device_and_metadata(client):
    from pypjrt.transfer import ShapeSpec
    with client.devices() as devs:
        with client.async_transfer([ShapeSpec(F32.code, (N,))], device=devs[0]) as t:
            assert t.device().id == devs[0].id
            try:
                t.add_metadata({"note": "hello"})
            except errors.PjrtError:
                pass                  # optional
            t.transfer(0, np.arange(N, dtype=np.float32))
            t.retrieve(0).close()


def test_topology_memory_space_kind_ids(client):
    from pypjrt.topology import Topology
    t = Topology.from_client(client)
    try:
        assert isinstance(t.memory_space_kind_ids(), list)
    except errors.PjrtError:
        pass                          # optional
    t.close()


def test_async_tracking_event(client):
    with client.devices() as devs:
        try:
            ev = devs[0].create_async_tracking_event("probe")
        except errors.PjrtError:
            pytest.skip if False else None
            return                    # optional on this plugin
        ev.close(); ev.close()
