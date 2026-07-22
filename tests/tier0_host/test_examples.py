"""Tier 0: the examples are documentation, so they are held to the same rule
as the library -- an example that no longer works must fail the build.

Actually *running* them needs a plugin, so that half lives in local-ci.sh
(`examples/run_all.py`). What tier 0 can enforce without hardware is that they
parse, that the runner sees all of them, that each is listed in the index, and
that they take the plugin argument the index promises they take.
"""
import ast
import pathlib
import pytest

pytestmark = pytest.mark.tier0
ROOT = pathlib.Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def _examples() -> list[pathlib.Path]:
    return sorted(EXAMPLES.glob("[0-9][0-9]_*.py"))


def test_there_are_examples():
    assert _examples(), "examples/ has no NN_*.py files"


@pytest.mark.parametrize("path", _examples(), ids=lambda p: p.name)
def test_example_parses_and_is_documented(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    assert ast.get_docstring(tree), f"{path.name} has no module docstring"

    # Uniform entry point: `main(plugin_path=None)` under a __main__ guard, so
    # `python examples/NN_x.py [plugin.so]` works for every one of them and the
    # runner can pass a plugin through without special-casing.
    fns = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "main" in fns, f"{path.name} defines no main()"
    src = path.read_text()
    assert "sys.argv[1] if len(sys.argv) > 1 else None" in src, (
        f"{path.name} does not accept an optional plugin path argument")


def test_every_example_is_in_the_index():
    index = (EXAMPLES / "README.md").read_text()
    missing = [p.name for p in _examples() if p.name not in index]
    assert not missing, f"examples/README.md does not mention: {missing}"


def test_index_mentions_no_example_that_does_not_exist():
    import re
    index = (EXAMPLES / "README.md").read_text()
    named = set(re.findall(r"\b(\d\d_[a-z0-9_]+\.py)\b", index))
    have = {p.name for p in _examples()}
    assert not (named - have), f"examples/README.md links dead files: {named - have}"


def test_runner_discovers_all_of_them():
    src = (EXAMPLES / "run_all.py").read_text()
    assert '[0-9][0-9]_*.py' in src, "run_all.py glob no longer matches the examples"
