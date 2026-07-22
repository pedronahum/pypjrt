"""Tier 0: plugin discovery, including the libtpu layout. No plugin needed."""
import os, pytest
from pypjrt import _loader

pytestmark = pytest.mark.tier0


def test_libtpu_candidates_cover_the_conventional_locations():
    """libtpu ships as libtpu/libtpu.so, not jax_plugins/*/xla_*_plugin.so, so
    the GPU glob never finds it."""
    cands = [str(p) for p in _loader._libtpu_candidates()]
    assert any(c.endswith("/lib/libtpu.so") for c in cands), cands


def test_tpu_library_path_is_honoured(tmp_path, monkeypatch):
    so = tmp_path / "libtpu.so"
    so.write_bytes(b"")
    monkeypatch.delenv("PYPJRT_PLUGIN", raising=False)
    monkeypatch.setenv("TPU_LIBRARY_PATH", str(so))
    assert _loader.find_plugin() == so


def test_explicit_path_beats_environment(tmp_path, monkeypatch):
    a, b = tmp_path / "a.so", tmp_path / "b.so"
    a.write_bytes(b""); b.write_bytes(b"")
    monkeypatch.setenv("TPU_LIBRARY_PATH", str(b))
    assert _loader.find_plugin(a) == a


def test_a_bad_env_path_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("PYPJRT_PLUGIN", raising=False)
    monkeypatch.setenv("TPU_LIBRARY_PATH", str(tmp_path / "nope.so"))
    with pytest.raises(FileNotFoundError, match="TPU_LIBRARY_PATH"):
        _loader.find_plugin()
