"""Tier 0: conformance report + diff logic. No plugin needed."""
import pytest
from pypjrt.conform import CheckResult, Report, Result, diff, render

pytestmark = pytest.mark.tier0


def _rep(name, checks):
    return Report(plugin=name, api_version=(0, 108), abi_version=(0, 114),
                  struct_size=1128, slots=136,
                  extensions=[{"type": 5, "name": "FFI"}],
                  checks=[CheckResult(i, i.split(".")[0], r, d) for i, r, d in checks])


def test_counts_partition_the_checks():
    r = _rep("p.so", [("a.x", Result.PASS, ""), ("a.y", Result.UNSUPPORTED, "nope"),
                      ("b.z", Result.FAIL, "boom"), ("b.w", Result.SKIP, "later")])
    assert r.counts() == {"pass": 1, "unsupported": 1, "fail": 1, "skip": 1}
    assert sum(r.counts().values()) == len(r.checks)


def test_render_surfaces_failures():
    out = render(_rep("p.so", [("a.x", Result.FAIL, "boom")]))
    assert "FAILURES (1)" in out and "boom" in out and "0 pass" in out


def test_json_roundtrip_is_plain_data():
    import json
    r = _rep("p.so", [("a.x", Result.PASS, "ok")])
    d = json.loads(json.dumps(r.to_json()))
    assert d["checks"][0]["result"] == "pass"
    assert d["api_version"] == [0, 108]


def test_diff_reports_capability_differences():
    a = _rep("cpu.so", [("d.stats", Result.UNSUPPORTED, ""), ("c.compile", Result.PASS, "")])
    b = _rep("gpu.so", [("d.stats", Result.PASS, ""), ("c.compile", Result.PASS, "")])
    b.extensions = [{"type": 5, "name": "FFI"}, {"type": 7, "name": "Triton"}]
    out = diff(a.to_json(), b.to_json())
    assert "1 check(s) differ" in out
    assert "d.stats" in out and "unsupported" in out
    assert "extensions only in gpu.so: Triton" in out


def test_diff_of_identical_reports():
    a = _rep("x.so", [("a.x", Result.PASS, "")])
    assert "identical results" in diff(a.to_json(), a.to_json())
