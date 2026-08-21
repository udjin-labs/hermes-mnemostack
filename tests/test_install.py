"""`hermes-mnemostack install` — the 0.19 directory shim.

0.19 finds providers by scanning `$HERMES_HOME/plugins/<name>/`; it does
not read pip entry points. Worse, a plugin whose import raises is logged
at DEBUG and silently omitted — a broken install is invisible exactly when
it matters. So these tests care about two things above all: that hermes's
OWN loader accepts the shim, and that the command refuses to leave behind
an install that would disappear.
"""

from __future__ import annotations

import pathlib

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


def test_the_follow_up_command_stays_in_the_profile_it_installed_into(tmp_path):
    """R1 (codex P2): a bare `hermes memory setup` runs against the AMBIENT
    home. Printing that after `install --hermes-home <profile>` sends the
    operator to configure a DIFFERENT profile than the one just verified —
    the profile-scoped mistake this command exists to prevent, one line
    later."""
    rc, out = _run(tmp_path)
    assert rc == 0, out
    assert f"HERMES_HOME={tmp_path} hermes memory setup {PLUGIN_NAME}" in out


def test_the_follow_up_command_is_bare_when_the_home_was_ambient(tmp_path, monkeypatch):
    """...and does NOT carry a redundant prefix when nothing was overridden."""
    import hermes_mnemostack.install as inst

    monkeypatch.setattr(inst, "resolve_hermes_home", lambda _e: (tmp_path, "ambient"))
    lines: list[str] = []
    args = cli.parse_args(["install"])
    assert inst.cmd_install(args, out=lines.append) == 0, lines
    out = "\n".join(lines)
    assert f"Next: hermes memory setup {PLUGIN_NAME}" in out
    assert "HERMES_HOME=" not in out


def test_json_output_is_machine_readable(tmp_path):
    """R1 (codex P2): `--json` was inherited by this subcommand and then
    ignored — automation selecting a documented flag got prose."""
    import json as _json

    lines: list[str] = []
    args = cli.parse_args(["install", "--hermes-home", str(tmp_path), "--json"])
    rc = cmd_install(args, out=lines.append)
    assert rc == 0
    body = _json.loads("\n".join(lines))
    assert body["status"] == "ok"
    assert body["action"] == "installed"
    assert body["verified"] is True
    assert body["target"] == str(_target(tmp_path))
    assert body["files"] == list(SHIM_FILES)
    assert body["next"].endswith(f"hermes memory setup {PLUGIN_NAME}")


def test_json_output_carries_failures_too(tmp_path, monkeypatch):
    import json as _json

    import hermes_mnemostack.install as inst

    monkeypatch.setattr(inst, "_package_importable", lambda: (False, "ModuleNotFoundError"))
    lines: list[str] = []
    args = cli.parse_args(["install", "--hermes-home", str(tmp_path), "--json"])
    assert cmd_install(args, out=lines.append) == 2
    body = _json.loads("\n".join(lines))
    assert body["status"] == "error"
    assert "not importable" in body["error"]
    assert body["action"] is None


def test_a_blank_hermes_home_is_refused_not_treated_as_unset():
    """R1 (review agent P1): every consumer tests this value for
    truthiness, so `--hermes-home ""` fell through to the AMBIENT home —
    and for `install` that means writing a plugin into a Hermes profile
    the operator never named. It cost the reviewer a real write into their
    own ~/.hermes before anyone noticed."""
    import hermes_mnemostack.install as inst

    # The CLI refuses it at parse time, for every subcommand.
    for command in ("install", "status", "doctor"):
        with pytest.raises(SystemExit) as excinfo:
            cli.parse_args([command, "--hermes-home", ""])
        assert excinfo.value.code == 2, command
    # ...and a library caller passing it directly gets no ambient fallback.
    home, how = inst.resolve_hermes_home("")
    assert home is None and "empty" in how
    home, how = inst.resolve_hermes_home("   ")
    assert home is None and "empty" in how
    lines: list[str] = []
    args = cli.parse_args(["install"])
    args.hermes_home = ""  # what a library caller could still hand over
    assert inst.cmd_install(args, out=lines.append) == 2
    assert "cannot resolve the Hermes home" in "\n".join(lines)


def test_a_tilde_home_is_expanded(tmp_path, monkeypatch):
    """`--hermes-home ~/profile` from a shell that did not expand it must
    not create a directory literally named "~"."""
    import hermes_mnemostack.install as inst

    monkeypatch.setenv("HOME", str(tmp_path))
    home, how = inst.resolve_hermes_home("~/profile")
    assert home == tmp_path / "profile", (home, how)


def test_the_scoped_setup_command_is_shell_safe(tmp_path):
    """R2 (codex P2): the line is meant to be pasted into a shell. A Hermes
    home with a space (this project's own checkout has one) would break
    into two words, and one with a metacharacter would hand the reader a
    command that runs something else."""
    import shlex

    import hermes_mnemostack.install as inst

    for raw in ("/tmp/my profile", "/tmp/x;touch pwned", "/tmp/plain"):
        line = inst._setup_command(pathlib.Path(raw), explicit=True)
        # Read it the way a shell reads it: one env assignment carrying the
        # EXACT path, then the command — nothing detached, nothing extra.
        parts = shlex.split(line)
        assert parts[0] == f"HERMES_HOME={raw}", (raw, line)
        assert parts[1:] == ["hermes", "memory", "setup", PLUGIN_NAME], (raw, line)
