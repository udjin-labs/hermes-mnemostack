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
    import shlex

    rc, out = _run(tmp_path)
    assert rc == 0, out
    line = next(ln for ln in out.splitlines() if ln.startswith("Next: "))
    # Parsed, not string-matched: the home is shell-quoted now, so a
    # tmp_path containing a space (a valid TMPDIR / --basetemp) would fail
    # a literal comparison for a reason that has nothing to do with the
    # behaviour under test.
    parts = shlex.split(line[len("Next: ") :])
    assert parts[0] == f"HERMES_HOME={tmp_path}"
    assert parts[1:] == ["hermes", "memory", "setup", PLUGIN_NAME]


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


def test_a_symlinked_target_is_never_written_through(tmp_path):
    """R1 (review agent P1): with --force the copy followed a symlink and
    overwrote whatever it pointed at — OUTSIDE the Hermes home. --force
    means "replace a foreign plugin directory", not "follow a link into
    somewhere nobody looked at"."""
    elsewhere = tmp_path / "someone-elses-data"
    elsewhere.mkdir()
    treasure = elsewhere / "__init__.py"
    treasure.write_text("IMPORTANT DATA - NOT OURS\n", encoding="utf-8")
    plugins = tmp_path / "home" / "plugins"
    plugins.mkdir(parents=True)
    (plugins / PLUGIN_NAME).symlink_to(elsewhere, target_is_directory=True)

    for flags in ((), ("--force",)):
        lines: list[str] = []
        args = cli.parse_args(
            ["install", "--hermes-home", str(tmp_path / "home"), *flags]
        )
        rc = cmd_install(args, out=lines.append)
        out = "\n".join(lines)
        assert rc == 2, (flags, out)
        assert "is a symlink" in out and "refusing to write through it" in out
        assert treasure.read_text(encoding="utf-8").startswith("IMPORTANT DATA")


def test_a_dangling_symlink_is_reported_not_crashed_into(tmp_path):
    """R1 (review agent P2): `exists()` FOLLOWS links and says False for a
    dangling one, so it read as "absent" and the run crashed on mkdir —
    on the default path, no --force needed."""
    plugins = tmp_path / "plugins"
    plugins.mkdir(parents=True)
    (plugins / PLUGIN_NAME).symlink_to(tmp_path / "nowhere")
    rc, out = _run(tmp_path)
    assert rc == 2, out
    assert "is a symlink" in out


def test_a_file_at_the_target_is_reported_not_crashed_into(tmp_path):
    plugins = tmp_path / "plugins"
    plugins.mkdir(parents=True)
    (plugins / PLUGIN_NAME).write_text("not a directory\n", encoding="utf-8")
    for flags in ((), ("--force",)):
        lines: list[str] = []
        args = cli.parse_args(["install", "--hermes-home", str(tmp_path), *flags])
        rc = cmd_install(args, out=lines.append)
        out = "\n".join(lines)
        assert rc == 2, (flags, out)
        assert "not a directory" in out


def test_the_shim_registers_through_the_entry_point_hermes_calls(tmp_path):
    """R1 (review agent P2): the end-to-end test could not fail on a broken
    `register()` — hermes falls back to scanning module attributes for a
    MemoryProvider subclass and instantiating it, and swallows a raising
    register() at DEBUG. So the documented entry point needs its own pin."""
    import importlib.util

    from hermes_mnemostack.provider import MnemostackProvider

    rc, out = _run(tmp_path)
    assert rc == 0, out
    installed = _target(tmp_path) / "__init__.py"
    spec = importlib.util.spec_from_file_location("_shim_under_test", installed)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    registered: list[object] = []

    class _Ctx:
        def register_memory_provider(self, provider):
            registered.append(provider)

    module.register(_Ctx())
    assert len(registered) == 1
    assert isinstance(registered[0], MnemostackProvider)


def test_no_link_below_the_home_is_written_through(tmp_path):
    """R2 (codex P1): the round-1 guard covered only the plugin directory.
    A symlinked `plugins/` redirects the whole install outside the home,
    and a symlinked FILE inside our own directory is written through by
    copyfile even without --force. Three positions in three rounds — so
    the rule is now stated once, for the whole chain below the home."""
    outside = tmp_path / "outside"
    outside.mkdir()
    treasure = outside / "keep.txt"
    treasure.write_text("NOT OURS\n", encoding="utf-8")

    # 1. `plugins` itself is a link.
    home = tmp_path / "home-a"
    home.mkdir()
    (home / "plugins").symlink_to(outside, target_is_directory=True)
    for flags in ((), ("--force",)):
        lines: list[str] = []
        args = cli.parse_args(["install", "--hermes-home", str(home), *flags])
        assert cmd_install(args, out=lines.append) == 2, flags
        assert "is a symlink (the plugins directory)" in "\n".join(lines)
    assert not (outside / PLUGIN_NAME).exists()
    assert treasure.read_text(encoding="utf-8") == "NOT OURS\n"

    # 2. A file INSIDE an otherwise-ours directory is a link.
    home = tmp_path / "home-b"
    rc, out = _run(home)
    assert rc == 0, out
    (_target(home) / "plugin.yaml").unlink()
    (_target(home) / "plugin.yaml").symlink_to(treasure)
    lines = []
    args = cli.parse_args(["install", "--hermes-home", str(home)])
    assert cmd_install(args, out=lines.append) == 2
    assert "plugin.yaml is a symlink" in "\n".join(lines)
    assert treasure.read_text(encoding="utf-8") == "NOT OURS\n"


def test_a_symlinked_home_itself_is_fine(tmp_path):
    """The operator named that path; a symlinked Hermes home is an
    ordinary setup and must not be refused."""
    real = tmp_path / "real-home"
    real.mkdir()
    link = tmp_path / "linked-home"
    link.symlink_to(real, target_is_directory=True)
    rc, out = _run(link)
    assert rc == 0, out
    assert (real / "plugins" / PLUGIN_NAME / "plugin.yaml").is_file()


def test_a_link_planted_after_the_check_is_still_not_followed(tmp_path, monkeypatch):
    """Defense in depth for the race the guard cannot close: if a link
    appears between the check and the copy, the write must replace the
    entry rather than open it. Simulated by passing the guard on a
    destination that IS a link."""
    import hermes_mnemostack.install as inst

    outside = tmp_path / "outside.txt"
    outside.write_text("NOT OURS\n", encoding="utf-8")
    rc, out = _run(tmp_path)
    assert rc == 0, out
    destination = _target(tmp_path) / "README.md"
    destination.unlink()
    destination.symlink_to(outside)

    monkeypatch.setattr(inst, "destination_problem", lambda _h, _t: None)  # the race
    lines: list[str] = []
    args = cli.parse_args(["install", "--hermes-home", str(tmp_path)])
    assert inst.cmd_install(args, out=lines.append) == 0, lines
    assert outside.read_text(encoding="utf-8") == "NOT OURS\n"  # untouched
    assert not destination.is_symlink()  # the link was replaced, not written through
    assert destination.read_text(encoding="utf-8").startswith("# mnemostack")


def test_the_link_check_asks_about_redirection_not_about_symlinks(tmp_path, monkeypatch):
    """R3 (codex P1): `is_symlink()` is False for an NTFS junction, which
    redirects just as well — and Windows is in this project's test matrix.
    The check reads the reparse tag too, so the question it asks is "does
    this path go somewhere else", not "is it a POSIX symlink"."""
    import os
    import stat as stat_mod

    import hermes_mnemostack.install as inst

    plain = tmp_path / "plain"
    plain.mkdir()
    assert inst.is_link_like(plain) is False
    assert inst.is_link_like(tmp_path / "absent") is False
    link = tmp_path / "link"
    link.symlink_to(plain, target_is_directory=True)
    assert inst.is_link_like(link) is True

    # A junction reports no symlink but carries a reparse tag; simulate the
    # platform difference by giving lstat the tag Windows would report.
    real_lstat = os.lstat

    class _Junction:
        st_reparse_tag = getattr(stat_mod, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)

        def __getattr__(self, item):
            return getattr(real_lstat(plain), item)

    class _WindowsStat:
        IO_REPARSE_TAG_MOUNT_POINT = _Junction.st_reparse_tag
        IO_REPARSE_TAG_SYMLINK = 0xA000000C

    monkeypatch.setattr(
        os, "lstat", lambda p, *a, **k: _Junction() if str(p) == str(plain) else real_lstat(p)
    )
    monkeypatch.setattr(inst, "stat", _WindowsStat)  # the constants Windows has
    assert inst.is_link_like(plain) is True


def test_a_link_planted_at_the_filename_cannot_be_written_through(tmp_path, monkeypatch):
    """Even having passed the check, a link at the destination filename is
    REPLACED, never opened: the content is written beside it and moved
    into place, and a rename replaces the entry rather than following it."""
    import hermes_mnemostack.install as inst

    outside = tmp_path / "outside.txt"
    outside.write_text("NOT OURS\n", encoding="utf-8")
    rc, out = _run(tmp_path)
    assert rc == 0, out

    destination = _target(tmp_path) / "README.md"
    destination.unlink()
    destination.symlink_to(outside)  # planted, and the check is bypassed

    monkeypatch.setattr(inst, "destination_problem", lambda _h, _t: None)
    lines: list[str] = []
    args = cli.parse_args(["install", "--hermes-home", str(tmp_path)])
    rc = inst.cmd_install(args, out=lines.append)
    assert rc == 0, lines
    assert outside.read_text(encoding="utf-8") == "NOT OURS\n"  # never written through
    assert not destination.is_symlink()  # the entry was replaced
    assert destination.read_text(encoding="utf-8").startswith("# mnemostack")


def test_a_plain_file_anywhere_on_the_chain_is_reported_not_crashed_into(tmp_path):
    """R3 (review agent P2): `plugins` existing as a FILE reached
    `mkdir(parents=True)`, which raises NotADirectoryError — a raw
    traceback, and in --json mode no JSON at all, breaking the contract
    that every path reports. The rule covers the whole chain now, not just
    the leaf."""
    import json as _json

    home = tmp_path / "home"
    home.mkdir()
    (home / "plugins").write_text("not a dir\n", encoding="utf-8")

    lines: list[str] = []
    args = cli.parse_args(["install", "--hermes-home", str(home)])
    assert cmd_install(args, out=lines.append) == 2
    assert "is not a directory (the plugins directory)" in "\n".join(lines)

    # ...and the machine-readable path answers too, rather than crashing.
    lines = []
    args = cli.parse_args(["install", "--hermes-home", str(home), "--json"])
    assert cmd_install(args, out=lines.append) == 2
    body = _json.loads("\n".join(lines))
    assert body["status"] == "error"
    assert "not a directory" in body["error"]


def test_an_os_error_during_the_write_is_still_a_report(tmp_path, monkeypatch):
    """Whatever the filesystem says at write time, the command answers in
    its own vocabulary — including under --json."""
    import json as _json

    import hermes_mnemostack.install as inst

    def _boom(*_a, **_k):
        raise NotADirectoryError(20, "Not a directory")

    monkeypatch.setattr(pathlib.Path, "mkdir", _boom)
    lines: list[str] = []
    args = cli.parse_args(["install", "--hermes-home", str(tmp_path), "--json"])
    assert inst.cmd_install(args, out=lines.append) == 2
    body = _json.loads("\n".join(lines))
    assert body["status"] == "error" and body["error"]


def test_an_ordinary_io_failure_is_not_diagnosed_as_a_race(tmp_path, monkeypatch):
    """R4 (codex P2): one handler told EVERY OSError that "something
    changed mid-install; re-run" — which for a full disk, a read-only
    mount or a permission error sends the operator in circles. The race
    advice belongs to the errnos O_EXCL|O_NOFOLLOW actually raises."""
    import errno as _errno

    import hermes_mnemostack.install as inst

    def _fail(errnum, message):
        def _boom(*_a, **_k):
            raise OSError(errnum, message)

        return _boom

    monkeypatch.setattr(inst.os, "open", _fail(_errno.EACCES, "Permission denied"))
    lines: list[str] = []
    args = cli.parse_args(["install", "--hermes-home", str(tmp_path)])
    assert inst.cmd_install(args, out=lines.append) == 2
    out = "\n".join(lines)
    assert "Permission denied" in out
    assert "permissions and free space" in out
    assert "mid-install" not in out

    monkeypatch.setattr(inst.os, "open", _fail(_errno.ELOOP, "Too many levels of symbolic links"))
    lines = []
    assert inst.cmd_install(args, out=lines.append) == 2
    assert "mid-install" in "\n".join(lines)


def test_a_half_written_install_can_be_retried(tmp_path, monkeypatch):
    """Found while testing the diagnosis above: a write that failed part
    way left a directory with no `__init__.py`, which the NEXT run
    classified as "foreign" and refused — stranding the operator at
    exactly the moment the retry has to work (after a full disk, a
    permission error, an interrupted run)."""
    import errno as _errno

    import hermes_mnemostack.install as inst

    def _boom(*_a, **_k):
        raise OSError(_errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(inst.os, "open", _boom)
    lines: list[str] = []
    args = cli.parse_args(["install", "--hermes-home", str(tmp_path)])
    assert inst.cmd_install(args, out=lines.append) == 2
    assert _target(tmp_path).is_dir()  # the directory is there, empty
    monkeypatch.undo()

    rc, out = _run(tmp_path)  # the retry, on a filesystem that now works
    assert rc == 0, out
    for name in SHIM_FILES:
        assert (_target(tmp_path) / name).is_file(), name


def test_matching_filenames_alone_do_not_make_a_directory_ours(tmp_path):
    """R5 (codex P2): the retry relaxation accepted a markerless directory
    whenever its entries happened to be named like ours — so a stranger's
    plugin.yaml and README.md would have been unlinked and overwritten by
    a plain install, defeating the foreign-plugin guard. Only an EMPTY
    directory is a half-written install of ours; our own partial writes
    always leave `__init__.py`, which carries the marker."""
    target = _target(tmp_path)
    target.mkdir(parents=True)
    theirs = target / "plugin.yaml"
    theirs.write_text("name: someone-else\n", encoding="utf-8")
    (target / "README.md").write_text("their docs\n", encoding="utf-8")

    rc, out = _run(tmp_path)
    assert rc == 2, out
    assert "is not this shim" in out
    assert theirs.read_text(encoding="utf-8") == "name: someone-else\n"


def test_an_unreadable_target_is_reported_not_crashed_into(tmp_path):
    """R5 (codex P2): `iterdir()` on a directory without read permission
    raised outside the command's handler — a traceback, and no JSON under
    --json. We cannot tell whose it is, so we refuse and say why."""
    import json as _json
    import os as _os

    target = _target(tmp_path)
    target.mkdir(parents=True)
    _os.chmod(target, 0o300)  # writable and searchable, NOT listable
    try:
        lines: list[str] = []
        args = cli.parse_args(["install", "--hermes-home", str(tmp_path), "--json"])
        rc = cmd_install(args, out=lines.append)
        if rc == 0:  # running as root, where the mode does not bite
            pytest.skip("this user can list a mode-300 directory")
        body = _json.loads("\n".join(lines))
        assert body["status"] == "error"
        assert "cannot inspect" in body["error"]
        # ...and --force cannot claim ownership of something unreadable.
        lines = []
        args = cli.parse_args(["install", "--hermes-home", str(tmp_path), "--force"])
        assert cmd_install(args, out=lines.append) == 2
        assert "cannot inspect" in "\n".join(lines)
    finally:
        _os.chmod(target, 0o700)


def test_a_failed_update_leaves_the_marker_in_place(tmp_path, monkeypatch):
    """R6 (codex P2): unlinking `__init__.py` before rewriting it meant a
    failure in between left a directory the next run could not recognise
    as ours — so an ordinary I/O failure made the retry need --force.
    Writing beside and moving into place means the marker is never
    absent."""
    import errno as _errno

    import hermes_mnemostack.install as inst

    rc, out = _run(tmp_path)  # a good install first
    assert rc == 0, out
    real_open = inst.os.open

    def _fail_on_first_shim_file(path, *a, **k):
        if f".{SHIM_FILES[0]}." in str(path):
            raise OSError(_errno.ENOSPC, "No space left on device")
        return real_open(path, *a, **k)

    monkeypatch.setattr(inst.os, "open", _fail_on_first_shim_file)
    lines: list[str] = []
    args = cli.parse_args(["install", "--hermes-home", str(tmp_path)])
    assert inst.cmd_install(args, out=lines.append) == 2
    monkeypatch.undo()

    # The marker survived the failed update, so the plain retry works.
    assert SHIM_MARKER in (_target(tmp_path) / "__init__.py").read_text(encoding="utf-8")
    rc, out = _run(tmp_path)
    assert rc == 0, out


def test_a_failed_write_leaves_no_scratch_file_behind(tmp_path, monkeypatch):
    import errno as _errno

    import hermes_mnemostack.install as inst

    real_open = inst.os.open

    def _fail_after_create(path, *a, **k):
        fd = real_open(path, *a, **k)
        if ".tmp" in str(path):
            os.close(fd)
            raise OSError(_errno.EIO, "I/O error")
        return fd

    import os

    monkeypatch.setattr(inst.os, "open", _fail_after_create)
    lines: list[str] = []
    args = cli.parse_args(["install", "--hermes-home", str(tmp_path)])
    assert inst.cmd_install(args, out=lines.append) == 2
    monkeypatch.undo()
    leftovers = [e.name for e in _target(tmp_path).iterdir() if e.name.endswith(".tmp")]
    assert leftovers == []
