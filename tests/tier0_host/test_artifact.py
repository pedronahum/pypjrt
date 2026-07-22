"""Tier 0: artifact container and cache keying. No plugin needed."""
import pytest
from pypjrt.artifact import MAGIC, VERSION, Artifact, ArtifactMismatch
from pypjrt.cache import CompileCache

pytestmark = pytest.mark.tier0


def _art():
    return Artifact(executable=b"\x01\x02\x03", platform="cpu", api_version=(0, 108),
                    xla_version=2, fingerprint="abcd", abi_proto="beef",
                    compile_options="1a0420012801", source=b"module {}",
                    source_sha256="x", output_types=[11], metadata={"note": "hi"})


def test_container_roundtrip():
    a = _art()
    b = Artifact.from_bytes(a.to_bytes())
    for f in ("executable", "platform", "api_version", "xla_version", "fingerprint",
              "abi_proto", "compile_options", "source", "output_types", "metadata"):
        assert getattr(b, f) == getattr(a, f), f


def test_file_roundtrip(tmp_path):
    p = _art().write(tmp_path / "x.pypjrta")
    assert Artifact.read(p).executable == b"\x01\x02\x03"


def test_bad_magic_is_a_diagnostic_not_a_crash():
    with pytest.raises(ArtifactMismatch, match="bad magic"):
        Artifact.from_bytes(b"NOTAPYPJRT" + b"\0" * 40)
    with pytest.raises(ArtifactMismatch, match="too short"):
        Artifact.from_bytes(b"abc")


def test_future_format_version_refused():
    blob = bytearray(_art().to_bytes())
    blob[8:12] = (VERSION + 1).to_bytes(4, "little")
    with pytest.raises(ArtifactMismatch, match="format version"):
        Artifact.from_bytes(bytes(blob))


class _FakePlugin:
    api_version = (0, 108)
    xla_version = 2


def test_guards_fire_on_each_mismatch():
    p = _FakePlugin()
    assert _art().check_compatible(p, platform="cpu") == []

    a = _art(); a.platform = "cuda"
    probs = a.check_compatible(p, platform="cpu", strict=False)
    assert any("platform" in x for x in probs)

    a = _art(); a.xla_version = 99
    probs = a.check_compatible(p, platform="cpu", strict=False)
    assert any("xla_version" in x for x in probs)

    a = _art(); a.api_version = (1, 0)
    probs = a.check_compatible(p, platform="cpu", strict=False)
    assert any("API major" in x for x in probs)


def test_strict_raises_with_every_reason():
    a = _art(); a.platform = "cuda"; a.xla_version = 99
    with pytest.raises(ArtifactMismatch) as ei:
        a.check_compatible(_FakePlugin(), platform="cpu")
    assert "platform" in str(ei.value) and "xla_version" in str(ei.value)


def test_cache_key_covers_every_component():
    k = CompileCache.key
    base = k(b"prog", b"opts", "cpu", 2, 0)
    assert base != k(b"prog2", b"opts", "cpu", 2, 0)
    assert base != k(b"prog", b"opts2", "cpu", 2, 0)
    assert base != k(b"prog", b"opts", "cuda", 2, 0)
    assert base != k(b"prog", b"opts", "cpu", 3, 0)   # xla_version matters
    assert base != k(b"prog", b"opts", "cpu", 2, 1)
    assert base == k(b"prog", b"opts", "cpu", 2, 0)


def test_cache_store_load_and_corruption(tmp_path):
    c = CompileCache(tmp_path)
    key = CompileCache.key(b"p", b"o", "cpu", 2, 0)
    assert c.load(key) is None and c.misses == 1
    path = c.store(key, _art())
    assert c.load(key).executable == b"\x01\x02\x03" and c.hits == 1
    path.write_bytes(b"garbage")
    assert c.load(key) is None            # corrupt entry is a miss, not a crash
    assert not path.exists()              # and it is evicted
