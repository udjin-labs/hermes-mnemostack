"""`hermes-mnemostack install` — the 0.19 directory shim.

0.19 finds providers by scanning `$HERMES_HOME/plugins/<name>/`; it does
not read pip entry points. Worse, a plugin whose import raises is logged
at DEBUG and silently omitted — a broken install is invisible exactly when
it matters. So these tests care about two things above all: that hermes's
OWN loader accepts the shim, and that the command refuses to leave behind
an install that would disappear.
"""

from __future__ import annotations

import pytest

from hermes_mnemostack import cli
from hermes_mnemostack.install import (
    PLUGIN_NAME,
    SHIM_FILES,
    SHIM_MARKER,
    cmd_install,
    shim_source,
)


def _run(tmp_path, *flags):
    lines: list[str] = []
    args = cli.parse_args(["install", "--hermes-home", str(tmp_path), *flags])
    rc = cmd_install(args, out=lines.append)
    return rc, "\n".join(lines)


def _target(tmp_path):
    return tmp_path / "plugins" / PLUGIN_NAME


# ------------------------------------------------------- the shim itself


def test_the_shim_carries_the_marker_hermes_scans_for():
    """0.19 text-scans the FIRST 8 KB of `__init__.py` for
    `register_memory_provider` (or `MemoryProvider`) before importing
    anything — hiding the registration behind an imported helper would make
    the directory invisible as a memory provider."""
    source = (shim_source() / "__init__.py").read_text(encoding="utf-8")
    assert "register_memory_provider" in source[:8192]
    assert SHIM_MARKER in source


def test_the_shim_declares_its_dependency_without_a_specifier():
    """hermes 0.19's CLI setup derives an import name with
    `dep.replace("-","_").split("[")[0]` — it strips extras but NOT `>=x.y`,
    so a specifier here becomes an unimportable module name and the
    dependency reads as permanently missing (its own bundled mem0 has the
    bug). The floor is checked in Python instead."""
    import yaml

    manifest = yaml.safe_load((shim_source() / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["name"] == PLUGIN_NAME
    deps = manifest["pip_dependencies"]
    assert deps == ["hermes-mnemostack"], deps
    for dep in deps:
        import_name = dep.replace("-", "_").split("[")[0]
        __import__(import_name)  # what hermes 0.19 will actually try


def test_every_shim_file_ships_with_the_package():
    """`install` copies from the installed package; a file left out of the
    wheel turns into a broken install for everyone but the developer."""
    for name in SHIM_FILES:
        assert (shim_source() / name).is_file(), name


# ------------------------------------------------------------- the command


def test_install_puts_the_shim_where_hermes_scans(tmp_path):
    rc, out = _run(tmp_path)
    assert rc == 0, out
    for name in SHIM_FILES:
        assert (_target(tmp_path) / name).is_file(), name
    assert "hermes loads it" in out


def test_hermes_own_loader_accepts_the_installed_shim(tmp_path, monkeypatch):
    """The verdict that matters is hermes's, not ours: its discovery must
    list the provider and its loader must return our instance."""
    pytest.importorskip("plugins.memory")
    import hermes_constants
    from plugins.memory import list_memory_provider_names, load_memory_provider

    rc, out = _run(tmp_path)
    assert rc == 0, out
    token = hermes_constants.set_hermes_home_override(str(tmp_path))
    try:
        assert PLUGIN_NAME in list_memory_provider_names()
        provider = load_memory_provider(PLUGIN_NAME)
    finally:
        hermes_constants.reset_hermes_home_override(token)
    assert provider is not None
    assert provider.name == PLUGIN_NAME
    from hermes_mnemostack.provider import MnemostackProvider

    assert isinstance(provider, MnemostackProvider)


def test_install_is_idempotent(tmp_path):
    assert _run(tmp_path)[0] == 0
    rc, out = _run(tmp_path)
    assert rc == 0, out
    assert "Replaced" in out  # ...and says so, rather than pretending it was new


def test_dry_run_writes_nothing(tmp_path):
    rc, out = _run(tmp_path, "--dry-run")
    assert rc == 0, out
    assert "would create" in out
    assert not _target(tmp_path).exists()


def test_a_foreign_plugin_of_the_same_name_is_not_overwritten(tmp_path):
    """Someone else's `mnemostack` plugin is their data, not ours."""
    target = _target(tmp_path)
    target.mkdir(parents=True)
    (target / "__init__.py").write_text("# someone else's plugin\n", encoding="utf-8")
    rc, out = _run(tmp_path)
    assert rc == 2
    assert "refusing to overwrite" in out
    assert (target / "__init__.py").read_text(encoding="utf-8").startswith("# someone")
    # ...unless the operator insists.
    rc, out = _run(tmp_path, "--force")
    assert rc == 0, out
    assert SHIM_MARKER in (target / "__init__.py").read_text(encoding="utf-8")


def test_install_refuses_when_the_package_is_not_importable(tmp_path, monkeypatch):
    """The import happens during DISCOVERY, before `pip_dependencies` is
    ever read, and 0.19 logs its failure at debug level — so a shim
    installed without the package would simply never appear, with no
    diagnosis. Refuse instead of leaving that behind."""
    import hermes_mnemostack.install as inst

    monkeypatch.setattr(inst, "_package_importable", lambda: (False, "ModuleNotFoundError"))
    rc, out = _run(tmp_path)
    assert rc == 2
    assert "pip install hermes-mnemostack" in out
    assert not _target(tmp_path).exists()


def test_install_reports_when_hermes_cannot_load_what_was_written(tmp_path, monkeypatch):
    """Files in place is not the same as hermes loading them, and only the
    second one is the thing the operator wanted."""
    import hermes_mnemostack.install as inst

    monkeypatch.setattr(inst, "verify_discovery", lambda _h: (False, "hermes does not list it"))
    rc, out = _run(tmp_path)
    assert rc == 1  # written, but not working — not a success
    assert "does not list it" in out
    assert (_target(tmp_path) / "__init__.py").is_file()


def test_an_unresolvable_home_is_an_error_not_a_guess(monkeypatch):
    """Installing into `~/.hermes` when a named profile is active puts the
    plugin somewhere the running Hermes never looks."""
    import hermes_mnemostack.install as inst

    monkeypatch.setattr(inst, "resolve_hermes_home", lambda _e: (None, "hermes not importable"))
    lines: list[str] = []
    args = cli.parse_args(["install"])
    assert cmd_install(args, out=lines.append) == 2
    assert "--hermes-home" in "\n".join(lines)
