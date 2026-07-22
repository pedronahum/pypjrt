"""Tier 1: topology + device descriptions (prep for TPU, useful everywhere)."""
import pytest, pypjrt
from pypjrt import errors
from pypjrt.topology import Topology

pytestmark = pytest.mark.tier1


@pytest.fixture(scope="module")
def plugin(cpu_plugin_path):
    return pypjrt.Plugin(cpu_plugin_path)


@pytest.fixture(scope="module")
def client(plugin):
    c = pypjrt.Client.create(plugin)
    yield c
    c.close()


def test_device_description_fields(client):
    with client.devices() as devs:
        d = devs[0]
        assert d.id == 0
        assert isinstance(d.kind, str) and d.kind
        assert d.process_index == 0
        assert d.local_hardware_id >= 0
        assert isinstance(d.debug_string, str)
        assert isinstance(d.attributes, dict)


def test_device_ids_are_distinct(client):
    with client.devices() as devs:
        assert len({d.id for d in devs}) == len(devs)


def test_coords_is_none_when_unreported(client):
    """Absent is None, never a crash -- TPU and CUDA report coords, CPU doesn't."""
    with client.devices() as devs:
        assert devs[0].coords is None or isinstance(devs[0].coords, tuple)


def test_topology_from_client(client):
    t = Topology.from_client(client)
    assert t.platform_name == "cpu"
    assert isinstance(t.platform_version, str)
    descs = t.device_descriptions()
    assert len(descs) == client.device_count
    assert all({"id", "process_index", "kind", "attributes"} <= set(d) for d in descs)
    t.close()


def test_topology_fingerprint_is_a_uint64(client):
    t = Topology.from_client(client)
    fp = t.fingerprint()
    assert isinstance(fp, int) and 0 <= fp < 2 ** 64
    assert Topology.from_client(client).fingerprint() == fp
    t.close()


def test_topology_serialize_roundtrips_or_says_why(client, plugin):
    t = Topology.from_client(client)
    blob = t.serialize()
    assert isinstance(blob, bytes) and len(blob) > 0
    try:
        t2 = Topology.deserialize(plugin, blob)
        assert t2.platform_name == t.platform_name
        t2.close()
    except errors.PjrtError:
        pass          # documented as plugin-dependent
    t.close()


def test_client_free_compile_is_probed_not_assumed(client, plugin):
    """PJRT_Compile with no client: works on CUDA, absent on this CPU plugin."""
    t = Topology.from_client(client)
    prog = ("module @m { func.func public @main(%a: tensor<4xf32>) -> tensor<4xf32> "
            "{ %0 = stablehlo.add %a, %a : tensor<4xf32>\n return %0 : tensor<4xf32> } }")
    try:
        blob = t.compile(prog)
        assert isinstance(blob, bytes) and len(blob) > 0
    except errors.PjrtError as e:
        assert "compiler factory" in e.message or "not implemented" in e.message.lower()
    t.close()


def test_closed_topology_refuses_use(client):
    t = Topology.from_client(client)
    t.close()
    with pytest.raises(errors.HandleClosed):
        t.platform_name


def test_platform_hint(plugin):
    assert plugin.platform_hint == "cpu"
    assert plugin.is_accelerator is False and plugin.is_gpu is False
