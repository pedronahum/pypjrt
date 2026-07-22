"""Test fixtures. Skips are loud by design: pyproject sets `-ra`, and
tests/test_no_silent_skips.py fails the run if a required tier skipped.
A suite reporting "0 failures, 87 skipped" can be hiding three whole suites at
100% skip -- including its primary correctness evidence.
"""
import os, pathlib, pytest, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

CPU_CANDIDATES = [pathlib.Path.home() / "lib/libpjrt_c_api_cpu_plugin.so"]
GPU_GLOBS = ["*/*/lib/python*/site-packages/jax_plugins/*/xla_*_plugin.so",
             "*/lib/python*/site-packages/jax_plugins/*/xla_*_plugin.so"]


def _first(paths):
    return next((p for p in paths if p.exists()), None)


@pytest.fixture(scope="session")
def cpu_plugin_path():
    p = _first([pathlib.Path(os.environ["PYPJRT_CPU_PLUGIN"])] if "PYPJRT_CPU_PLUGIN" in os.environ
               else CPU_CANDIDATES)
    if p is None:
        pytest.skip("no CPU PJRT plugin (set $PYPJRT_CPU_PLUGIN)")
    return p


@pytest.fixture(scope="session")
def gpu_plugin_path():
    if "PYPJRT_GPU_PLUGIN" in os.environ:
        p = pathlib.Path(os.environ["PYPJRT_GPU_PLUGIN"])
        if p.exists():
            return p
    hits = []
    for g in GPU_GLOBS:
        hits += sorted(pathlib.Path.home().glob("programming/" + g))
    if not hits:
        pytest.skip("no GPU PJRT plugin (set $PYPJRT_GPU_PLUGIN)")
    return hits[0]


# --- loud skips ------------------------------------------------------------
# Session hooks must live in conftest.py, not a test module (a plain test file
# is collected but its session-scoped hooks are never called -- which is how the
# first version of this guard silently did nothing).

REQUIRED_TIERS = {
    t.strip() for t in os.environ.get("PYPJRT_REQUIRED_TIERS", "tier0").split(",") if t.strip()
}


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    tr = session.config.pluginmanager.get_plugin("terminalreporter")
    if tr is None:
        return
    offenders = [
        (r.nodeid, (r.longrepr[-1] if r.longrepr else "?"))
        for r in tr.stats.get("skipped", [])
        for tier in REQUIRED_TIERS
        if tier in str(r.nodeid)
    ]
    if offenders:
        tr.write_line("")
        tr.write_line(f"REQUIRED TIER SKIPPED ({len(offenders)}) -- failing the run:",
                      red=True, bold=True)
        for nodeid, why in offenders:
            tr.write_line(f"  {nodeid}: {why}", red=True)
        session.exitstatus = 1
