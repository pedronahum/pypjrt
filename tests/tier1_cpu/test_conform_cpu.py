"""Tier 1: run the harness against a real CPU plugin."""
import pytest, pypjrt
from pypjrt.conform import Result, run

pytestmark = pytest.mark.tier1


@pytest.fixture(scope="module")
def report(cpu_plugin_path):
    return run(pypjrt.Plugin(cpu_plugin_path))


def test_no_failures(report):
    bad = [c for c in report.checks if c.result is Result.FAIL]
    assert not bad, "\n".join(f"{c.id}: {c.detail}" for c in bad)


def test_core_path_passes(report):
    by_id = {c.id: c.result for c in report.checks}
    for essential in ("abi.api_version_readable", "plugin.initialize", "client.create",
                      "client.addressable_devices", "compile.stablehlo_text",
                      "buffer.from_host", "execute.single_device"):
        assert by_id[essential] is Result.PASS, essential


def test_absent_capabilities_are_unsupported_not_failures(report):
    """The load-bearing rule: absent != broken."""
    by_id = {c.id: c.result for c in report.checks}
    assert by_id["device.memory_stats"] in (Result.PASS, Result.UNSUPPORTED)
    assert by_id["compile.cost_analysis"] in (Result.PASS, Result.UNSUPPORTED)


def test_report_records_the_negotiation(report):
    assert report.api_version[0] == 0
    assert report.abi_version >= report.api_version
    assert report.slots > 0 and report.struct_size > 0
    assert report.platform == "cpu"
