"""Tier 1: M4 -- serialize, cache, artifacts."""
import array, time, pytest, pypjrt
from pypjrt import errors
from pypjrt.artifact import Artifact

pytestmark = pytest.mark.tier1
F32 = 11
PROG = """
module @m {
  func.func public @main(%a: tensor<256x256xf32>, %b: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %1 = stablehlo.tanh %0 : tensor<256x256xf32>
    return %1 : tensor<256x256xf32>
  }
}
"""


@pytest.fixture(scope="module")
def client(cpu_plugin_path):
    c = pypjrt.Client.create(pypjrt.Plugin(cpu_plugin_path))
    yield c
    c.close()


def test_serialize_and_deserialize_executes(client):
    e = client.compile(PROG)
    blob = e.serialize()
    assert len(blob) > 100
    e.close()
    e2 = client.deserialize_executable(blob)
    with client.devices() as devs:
        x = array.array("f", [0.01] * (256 * 256))
        a = client.buffer_from_host(x, F32, [256, 256], devs[0])
        b = client.buffer_from_host(x, F32, [256, 256], devs[0])
        (o,) = e2(a, b)
        assert len(o.to_host()) == 256 * 256 * 4
        for h in (o, a, b):
            h.close()
    e2.close()


def test_fingerprint_is_stable_and_program_specific(client):
    e1 = client.compile(PROG)
    e2 = client.compile(PROG)
    other = client.compile(PROG.replace("tanh", "logistic"))
    f1, f2, f3 = e1.fingerprint(), e2.fingerprint(), other.fingerprint()
    for e in (e1, e2, other):
        e.close()
    if not f1:
        pytest.skip("plugin returns an empty fingerprint")
    assert f1 == f2, "same program -> same fingerprint"
    assert f1 != f3, "different program -> different fingerprint"


def test_m4_gate_cache_skips_compilation(client, tmp_path):
    """Gate (a): a warm compile must be dramatically faster than a cold one."""
    cache = pypjrt.CompileCache(tmp_path)
    t0 = time.perf_counter(); e1 = client.compile(PROG, cache=cache); cold = time.perf_counter() - t0
    e1.close()
    assert (cache.hits, cache.misses) == (0, 1)

    t0 = time.perf_counter(); e2 = client.compile(PROG, cache=cache); warm = time.perf_counter() - t0
    assert (cache.hits, cache.misses) == (1, 1)
    assert e2.num_outputs == 1
    e2.close()
    assert warm < cold / 2, f"cache gave no speedup: cold {cold*1e3:.1f}ms warm {warm*1e3:.1f}ms"


def test_cache_key_separates_distinct_programs(client, tmp_path):
    cache = pypjrt.CompileCache(tmp_path)
    a = client.compile(PROG, cache=cache); a.close()
    b = client.compile(PROG.replace("tanh", "logistic"), cache=cache); b.close()
    assert cache.misses == 2 and cache.hits == 0


def test_artifact_file_roundtrip_and_reload(client, tmp_path):
    e = client.compile(PROG)
    art = e.to_artifact(source=PROG.encode())
    e.close()
    assert art.platform == "cpu" and art.xla_version is not None
    assert art.source_sha256 and art.output_types == [F32]
    p = art.write(tmp_path / "prog.pypjrta")
    reloaded = client.load_artifact(p)
    assert reloaded.num_outputs == 1
    reloaded.close()


def test_m4_gate_mismatched_artifact_is_a_diagnostic(client, tmp_path):
    """Gate (c): a mismatched artifact fails with a reason, not a crash."""
    e = client.compile(PROG)
    art = e.to_artifact(source=PROG.encode())
    e.close()
    art.platform = "not-a-real-platform"
    art.xla_version = 999999
    with pytest.raises(errors.PjrtError) as ei:
        client.load_artifact(art)
    msg = str(ei.value)
    assert "platform" in msg and "xla_version" in msg
    # non-strict reports without raising
    assert len(art.check_compatible(client._plugin, platform="cpu", strict=False)) == 2
