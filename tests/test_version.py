"""The version lives in four files, and nothing kept them in step.

A release here is a hand edit in four places: `pyproject.toml` (what pip
resolves), `__init__.py` (what `--version` reports), `plugin.yaml` (what
hermes reads out of the installed shim), and the README's status line
(what a reader believes). Miss one and the failure is quiet in the worst
way — a shim reporting a version the package is not, or a user told they
have a fix that shipped in the release after theirs.
"""

from __future__ import annotations

import pathlib
import re

import yaml

import hermes_mnemostack

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    assert match, "pyproject.toml has no version line"
    return match.group(1)


def test_the_package_and_the_distribution_agree():
    """`--version` must not report something pip never installed."""
    assert hermes_mnemostack.__version__ == _pyproject_version()


def test_the_installed_shim_reports_the_version_it_actually_is():
    """hermes reads this file out of the plugin directory `install` writes.
    A stale value here is a plugin that misidentifies itself to the host,
    which is exactly the thing an operator would check first when a fix
    appears not to have landed."""
    manifest = yaml.safe_load(
        (ROOT / "src/hermes_mnemostack/plugin/plugin.yaml").read_text(encoding="utf-8")
    )
    assert str(manifest["version"]) == _pyproject_version()


def test_the_readme_status_line_is_not_left_a_release_behind():
    """The README states the version in prose, so it is the one that gets
    forgotten — and it is also the only one a person reads before deciding
    whether they already have the fix they are looking for."""
    first_lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()[:10]
    status = next((ln for ln in first_lines if ln.startswith("**Status:")), None)
    assert status, "README lost its status line"
    assert _pyproject_version() in status, status


def test_the_readme_states_the_number_of_tests_there_actually_are():
    """The README makes the suite size a headline claim, and a number in
    prose drifts the moment a test is added — which is exactly how it
    shipped saying 182 while the suite had grown to 185.

    Pinned so that adding a test fails here until the claim catches up.
    That friction is the point: the alternative is a number nobody
    re-checks, which is worse than no number at all.
    """
    import ast
    import re

    counted = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        counted += sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
            for node in ast.walk(tree)
        )

    text = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"covered by (\d+) tests", text)
    assert match, "README no longer states a test count"
    assert int(match.group(1)) == counted, (
        f"README says {match.group(1)} tests, the suite has {counted}"
    )
