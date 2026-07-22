"""Tier 2: the M2 gate -- two plugins, differing reports, no crashes."""
import pytest, pypjrt
from pypjrt.conform import Result, diff, run

pytestmark = pytest.mark.tier2
TIGHT = {"preallocate": False, "memory_fraction": 0.05}


@pytest.fixture(scope="module")
def gpu_report(gpu_plugin_path):
    return run(pypjrt.Plugin(gpu_plugin_path), TIGHT)


@pytest.fixture(scope="module")
def cpu_report(cpu_plugin_path):
    return run(pypjrt.Plugin(cpu_plugin_path))


def test_gpu_has_no_failures(gpu_report):
    bad = [c for c in gpu_report.checks if c.result is Result.FAIL]
    assert not bad, "\n".join(f"{c.id}: {c.detail}" for c in bad)


def test_gpu_advertises_more_extensions(gpu_report, cpu_report):
    g = {e["name"] for e in gpu_report.extensions}
    c = {e["name"] for e in cpu_report.extensions}
    assert "Gpu_Custom_Call" in g and "Gpu_Custom_Call" not in c
    assert g != c


def test_m2_gate_reports_differ_and_every_difference_is_a_probe(gpu_report, cpu_report):
    """The M2 gate: the two reports must differ, and every difference
    must surface as a capability probe rather than a crash."""
    a = {c.id: c.result for c in cpu_report.checks}
    b = {c.id: c.result for c in gpu_report.checks}
    differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert differing, "reports are identical -- the harness is not discriminating"
    for k in differing:
        assert Result.FAIL not in (a.get(k), b.get(k)), f"{k} differs by FAILURE, not capability"
    assert "differ" in diff(cpu_report.to_json(), gpu_report.to_json())
