"""Tier 0: every probe in tests/must_fail/ must be REJECTED by pyright.

A probe that starts checking cleanly means the typed facade regressed, or that
the claim it encodes was never true. Either way it is a failure, which is why
this test asserts on rejection rather than acceptance.
"""
import json, pathlib, shutil, subprocess, sys
import pytest

pytestmark = pytest.mark.tier0
ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBES = sorted((ROOT / "tests" / "must_fail").glob("*_must_fail.py"))


def _pyright():
    for cand in (ROOT / ".venv/bin/pyright", "pyright"):
        if shutil.which(str(cand)):
            return str(cand)
    return None


@pytest.fixture(scope="module")
def diagnostics():
    exe = _pyright()
    if exe is None:
        pytest.skip("pyright not installed (pip install -e '.[dev]')")
    if not PROBES:
        pytest.fail("tests/must_fail/ is empty -- the guarantee is unenforced")
    r = subprocess.run([exe, "--outputjson", *[str(p) for p in PROBES]],
                       capture_output=True, text=True, cwd=ROOT)
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        pytest.fail(f"pyright produced no JSON: {r.stdout[-500:]}{r.stderr[-500:]}")
    by_file: dict[str, int] = {}
    for d in data["generalDiagnostics"]:
        if d["severity"] == "error":
            by_file[pathlib.Path(d["file"]).name] = by_file.get(pathlib.Path(d["file"]).name, 0) + 1
    return by_file


@pytest.mark.parametrize("probe", PROBES, ids=lambda p: p.stem)
def test_probe_is_rejected(probe, diagnostics):
    n = diagnostics.get(probe.name, 0)
    assert n > 0, (
        f"{probe.name} type-checks cleanly, but it exists to be rejected. "
        f"Either the typed facade regressed or this probe no longer encodes a "
        f"real guarantee -- do not 'fix' it by making it compile.")


def test_the_real_package_still_checks_clean():
    """The probes must fail; the library must not."""
    exe = _pyright()
    if exe is None:
        pytest.skip("pyright not installed")
    r = subprocess.run([exe, "--outputjson", str(ROOT / "src" / "pypjrt")],
                       capture_output=True, text=True, cwd=ROOT)
    data = json.loads(r.stdout)
    errs = [d for d in data["generalDiagnostics"] if d["severity"] == "error"]
    assert not errs, "\n".join(
        f"{pathlib.Path(d['file']).name}:{d['range']['start']['line']+1} "
        f"{d['message'].splitlines()[0]}" for d in errs[:10])
