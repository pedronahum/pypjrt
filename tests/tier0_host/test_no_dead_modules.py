"""Tier 0: every shipped module must import, and must be reachable from a test.

A mechanism rather than an intention. It is entirely possible to carry tens of
thousands of lines of runtime that nothing links against, with nothing telling
you. Here, adding a module without a test that touches it fails the build.
"""
import importlib, pathlib, pkgutil
import pytest

pytestmark = pytest.mark.tier0
ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = ROOT / "src" / "pypjrt"


def _modules() -> list[str]:
    out = []
    for m in pkgutil.walk_packages([str(PKG)], prefix="pypjrt."):
        if "._abi.pjrt_" in m.name:
            continue          # generated ABI data, covered by test_abi.py
        out.append(m.name)
    return sorted(out)


@pytest.mark.parametrize("name", _modules())
def test_module_imports(name):
    importlib.import_module(name)


def test_every_module_is_referenced_by_a_test():
    text = "\n".join(p.read_text() for p in (ROOT / "tests").rglob("*.py"))
    text += "\n".join(p.read_text() for p in (ROOT / "spike").rglob("*.py"))
    orphans = []
    for name in _modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf.startswith("_") and leaf not in ("_abi",):
            # private modules are reached through the public surface
            continue
        if name not in text and f"pypjrt.{leaf}" not in text and f"import {leaf}" not in text \
                and f"from .{leaf}" not in text and leaf not in text:
            orphans.append(name)
    assert not orphans, (
        f"modules no test or spike mentions: {orphans}. Either exercise them or "
        f"delete them -- unreferenced code is how a 25k-LOC island happens.")
