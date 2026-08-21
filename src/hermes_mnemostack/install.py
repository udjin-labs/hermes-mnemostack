"""`hermes-mnemostack install` — put the directory shim where 0.19 looks.

hermes-agent 0.19 finds memory providers by scanning `$HERMES_HOME/plugins/
<name>/`; it does not read pip entry points (0.20 does). So installing the
package is not enough on 0.19, and this command copies the shim from the
installed package into the ACTIVE Hermes home.

Two things it deliberately does not leave to chance:

- **The active home, not `~/.hermes`.** The plugins directory is
  profile-scoped; hardcoding the default path installs into the wrong
  profile for anyone using named profiles, and the symptom — "the provider
  just isn't there" — gives no hint why.
- **A verification pass.** 0.19 logs a failed plugin import at DEBUG and
  then silently omits the provider, so a broken install is invisible at the
  moment it matters. This asks hermes's own discovery whether it can see
  and load the provider, and reports what it found.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shlex
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

#: Files the shim consists of. Anything else in the target directory is
#: someone else's and is never touched.
SHIM_FILES = ("__init__.py", "plugin.yaml", "README.md")

#: How an existing directory is recognised as OURS rather than another
#: plugin that happens to use the name. Present in the shim's `__init__.py`.
SHIM_MARKER = "hermes_mnemostack.provider"

PLUGIN_NAME = "mnemostack"


def shim_source() -> Path:
    """The shim inside the installed package."""
    return Path(__file__).resolve().parent / "plugin"


def resolve_hermes_home(explicit: str | None) -> tuple[Path | None, str]:
    """(home, how) — the ACTIVE Hermes home, resolved hermes's own way.

    A BLANK explicit value is refused rather than treated as unset: `if
    explicit:` is false for "", so it would fall through to the ambient
    home and install into a profile the caller never named. The CLI
    rejects it at parse time; this guard covers a library caller.
    """
    if explicit is not None and not str(explicit).strip():
        return None, "--hermes-home was empty"
    if explicit:
        return Path(explicit).expanduser(), "--hermes-home"
    try:
        from hermes_constants import get_hermes_home
    except Exception as exc:  # noqa: BLE001 — hermes not installed/importable
        return None, f"hermes-agent is not importable ({type(exc).__name__})"
    try:
        return Path(get_hermes_home()), "hermes-agent's own resolution"
    except Exception as exc:  # noqa: BLE001
        return None, f"hermes could not resolve its home ({type(exc).__name__})"


def is_link_like(path: Path) -> bool:
    """Whether this path redirects elsewhere — symlink OR junction.

    `Path.is_symlink()` is False for an NTFS directory junction, which
    redirects just as effectively; Windows is in this project's test
    matrix, so "is it a symlink" is the wrong question to ask there. The
    reparse tag answers the right one.
    """
    try:
        if path.is_symlink():
            return True
    except OSError:  # pragma: no cover — unreadable parent
        return True  # cannot tell: treat as unsafe
    try:
        tag = getattr(os.lstat(path), "st_reparse_tag", 0)
    except OSError:
        return False  # absent: nothing to redirect through
    return tag in (
        getattr(stat, "IO_REPARSE_TAG_SYMLINK", -1),
        getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", -1),
    )


def destination_problem(home: Path, target: Path) -> str | None:
    """Why this destination cannot be written to, or None.

    Review found a bad path component in four positions across as many
    rounds — the plugin directory, its parent, a file inside it, then the
    parent as a plain FILE. So this states one rule for the whole chain
    instead of guarding a fifth position later: below the home the
    operator named, every component on the way to a file we write must be
    a real directory (or absent), and no file we write may be a link. A
    link redirects the write outside `$HERMES_HOME`, which is the one
    thing an installer must never do quietly.

    The home itself is exempt: the operator named that path, and a
    symlinked Hermes home is a legitimate setup.
    """
    for component in (target.parent, target):
        what = "the plugins directory" if component == target.parent else "the target"
        if is_link_like(component):
            return f"{component} is a symlink ({what})"
        if component.exists() and not component.is_dir():
            # Not something to replace — something a human has to look at.
            return f"{component} is not a directory ({what})"
    if target.is_dir():
        for name in SHIM_FILES:
            entry = target / name
            if is_link_like(entry):
                return f"{entry} is a symlink"
    return None


def _target_state(target: Path) -> str:
    """What is sitting at the target path.

    "absent" | "ours" | "foreign" | "symlink" | "not-a-directory".

    A symlink is its OWN answer, checked without following: writing
    through one puts files wherever it points — outside `$HERMES_HOME`
    entirely — and `--force` means "replace a foreign plugin directory",
    not "follow a link somewhere I never looked at". A dangling symlink
    must be caught here too: `exists()` follows links and reports False
    for it, which would classify it "absent" and then crash on mkdir.
    """
    if is_link_like(target):
        return "symlink"
    if not target.exists():
        return "absent"
    if not target.is_dir():
        return "not-a-directory"
    init = target / "__init__.py"
    if init.is_file():
        marked = SHIM_MARKER in init.read_text(encoding="utf-8", errors="replace")
        return "ours" if marked else "foreign"
    # No `__init__.py`. An EMPTY directory is our own half-written install
    # — mkdir succeeded and the first write did not — and refusing that as
    # "foreign" would strand the operator at exactly the moment a retry
    # has to work. Anything else is someone else's: matching FILENAMES
    # prove nothing (a directory holding a stranger's plugin.yaml and
    # README.md would qualify on names alone, and a plain install would
    # then overwrite them), and our own partial installs always write
    # `__init__.py` first, so they carry the marker.
    try:
        empty = not any(target.iterdir())
    except OSError as exc:
        # Statted but not listable. We cannot tell whose it is, so we do
        # not touch it — and we say why, rather than letting the traceback
        # escape past the command's own error reporting.
        return f"unreadable: {exc.strerror or exc}"
    return "ours" if empty else "foreign"


def _package_importable() -> tuple[bool, str]:
    """Whether the shim's import will actually resolve at discovery time."""
    try:
        from hermes_mnemostack.provider import MnemostackProvider  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    try:
        from importlib.metadata import version

        return True, version("hermes-mnemostack")
    except Exception:  # noqa: BLE001 — importable but not installed as a dist
        return True, "unknown version (not installed as a distribution)"


@contextmanager
def _scoped_home(home: Path) -> Iterator[None]:
    """Point hermes's own home resolution at the home we installed into.

    Without this the verification scans whatever home the AMBIENT
    environment resolves to, so `install --hermes-home <profile>` reports
    "hermes does not list it" about a perfectly good install — the exact
    profile-scoped mistake this command exists to prevent, reproduced
    inside its own confirmation step.
    """
    try:
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    except Exception:  # noqa: BLE001 — older/newer layout without the hooks
        yield
        return
    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def verify_discovery(home: Path) -> tuple[bool, str]:
    """Ask hermes's OWN discovery whether it can see and load the provider.

    The point of the command is that 0.19 fails silently: a plugin whose
    import raises is logged at debug level and simply omitted. Reporting
    hermes's verdict is the only honest confirmation an install worked.
    """
    try:
        from plugins.memory import list_memory_provider_names, load_memory_provider
    except Exception as exc:  # noqa: BLE001 — layout differs across versions
        return False, f"cannot import hermes's discovery ({type(exc).__name__})"
    try:
        with _scoped_home(home):
            names = list(list_memory_provider_names())
    except Exception as exc:  # noqa: BLE001
        return False, f"discovery scan failed ({type(exc).__name__}: {exc})"
    if PLUGIN_NAME not in names:
        return False, f"hermes does not list it (found: {', '.join(names) or 'none'})"
    try:
        with _scoped_home(home):
            provider = load_memory_provider(PLUGIN_NAME)
    except Exception as exc:  # noqa: BLE001
        return False, f"hermes listed it but could not load it ({type(exc).__name__}: {exc})"
    if provider is None:
        return False, "hermes listed it but loading returned nothing"
    return True, f"hermes loads it (provider name: {getattr(provider, 'name', '?')})"


def _setup_command(home: Path, explicit: bool) -> str:
    """The follow-up command, scoped to the home we just installed into.

    A bare `hermes memory setup` runs against the AMBIENT home. Printing
    that after `install --hermes-home <profile>` sends the operator to
    configure a different profile than the one just verified — the same
    profile-scoped mistake this command exists to prevent, one line later.
    """
    base = f"hermes memory setup {PLUGIN_NAME}"
    if not explicit:
        return base
    # Quoted: this line is meant to be pasted into a shell, and a Hermes
    # home may contain spaces (this project's own checkout does) — or
    # metacharacters, where an unquoted path would hand the reader a
    # command that runs something else.
    return f"HERMES_HOME={shlex.quote(str(home))} {base}"


def cmd_install(args: argparse.Namespace, out: Any = print) -> int:
    as_json = bool(getattr(args, "json", False))
    lines: list[str] = []
    result: dict[str, Any] = {
        "action": None,
        "hermes_home": None,
        "hermes_home_source": None,
        "target": None,
        "package": None,
        "files": list(SHIM_FILES),
        "verified": None,
        "verdict": None,
        "next": None,
        "error": None,
    }

    def say(message: str) -> None:
        lines.append(message)

    def finish(rc: int) -> int:
        if as_json:
            out(json.dumps({**result, "status": "ok" if rc == 0 else "error"}, indent=2))
        else:
            for line in lines:
                out(line)
        return rc

    explicit = bool(getattr(args, "hermes_home", None))
    home, how = resolve_hermes_home(getattr(args, "hermes_home", None))
    result["hermes_home_source"] = how
    if home is None:
        result["error"] = f"cannot resolve the Hermes home — {how}"
        say(f"error: {result['error']}")
        say("       pass --hermes-home <path> to install into a known profile")
        return finish(2)
    target = home / "plugins" / PLUGIN_NAME
    result["hermes_home"] = str(home)
    result["target"] = str(target)
    say(f"Hermes home:  {home}  ({how})")
    say(f"Target:       {target}")

    ok, detail = _package_importable()
    if not ok:
        # Refuse rather than install a shim that will vanish silently: the
        # import happens during DISCOVERY, before pip_dependencies is ever
        # read, and its failure is invisible at debug level.
        result["error"] = f"hermes_mnemostack is not importable here — {detail}"
        say(f"error: {result['error']}")
        say("       install the package first: pip install hermes-mnemostack")
        return finish(2)
    result["package"] = detail
    say(f"Package:      hermes-mnemostack {detail}")

    linked = destination_problem(home, target)
    if linked is not None:
        # Not overridable by --force: through a link the write lands
        # somewhere the operator never named, and --force is a claim about
        # WHOSE plugin this is, not a licence to leave the home.
        result["error"] = linked
        say(f"error: {linked} — refusing to write through it")
        say("       remove or move it, then re-run")
        return finish(2)

    state = _target_state(target)
    if state.startswith("unreadable"):
        # Not overridable: --force says "this plugin is ours", which is a
        # claim nobody can make about a directory they cannot read.
        result["error"] = f"cannot inspect {target} ({state.split(': ', 1)[-1]})"
        say(f"error: {result['error']}")
        say("       fix its permissions, or remove it, then re-run")
        return finish(2)
    if state == "foreign" and not getattr(args, "force", False):
        result["error"] = f"{target} exists and is not this shim"
        say(f"error: {result['error']} — refusing to overwrite it")
        say("       remove it, or re-run with --force if it really is ours")
        return finish(2)

    source = shim_source()
    missing = [f for f in SHIM_FILES if not (source / f).is_file()]
    if missing:
        result["error"] = f"the installed package is missing shim file(s): {', '.join(missing)}"
        say(f"error: {result['error']}")
        return finish(2)

    if getattr(args, "dry_run", False):
        verb = "would replace" if state != "absent" else "would create"
        result["action"] = verb.replace("would ", "would-")
        result["next"] = _setup_command(home, explicit)
        say(f"dry-run: {verb} {target} ({', '.join(SHIM_FILES)})")
        return finish(0)

    for name in SHIM_FILES:
        destination = target / name
        # Written beside, then MOVED into place. Two properties come from
        # that, and neither survives an unlink-then-create:
        #
        # - the marker file is never absent. Unlinking `__init__.py` and
        #   then failing to write it leaves a directory that the NEXT run
        #   cannot recognise as ours, so an ordinary I/O failure would
        #   make the retry need --force.
        # - os.replace does not FOLLOW a link at the destination, it
        #   replaces the entry — so a link planted at this filename after
        #   the check still cannot redirect the write.
        #
        # Residual, stated rather than claimed away: a parent directory
        # swapped for a link after destination_problem() ran would still be
        # followed. Closing that needs dir_fd-relative opens, which this
        # command does not do — and an attacker who can win that race
        # already owns the plugins directory and can simply place a plugin
        # there. Not a defence this installer can honestly offer.
        scratch = target / f".{name}.{os.getpid()}.tmp"
        try:
            # mkdir is INSIDE the guard: it raises NotADirectoryError when
            # a component is a plain file, and an installer that answers a
            # misconfiguration with a traceback — and, in --json mode, with
            # no JSON at all — has failed at its one job.
            target.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(scratch, flags, 0o644)
            with open(fd, "wb") as handle:
                handle.write((source / name).read_bytes())
            os.replace(scratch, destination)
        except OSError as exc:
            scratch.unlink(missing_ok=True)
            # A refusal, not a traceback — and the RIGHT refusal. EEXIST and
            # ELOOP are what O_EXCL|O_NOFOLLOW report when something
            # appeared at the destination after the check; everything else
            # (a full disk, a read-only mount, a permission error, an
            # unreadable package file) is an ordinary I/O failure, and
            # telling that operator to "re-run" would send them in circles.
            result["error"] = f"could not write {destination}: {exc.strerror or exc}"
            say(f"error: {result['error']}")
            if exc.errno in (errno.EEXIST, errno.ELOOP):
                say("       something changed at the destination mid-install; re-run")
            else:
                say("       check permissions and free space on that filesystem")
            return finish(2)
    result["action"] = "replaced" if state != "absent" else "installed"
    say(f"{'Replaced' if state != 'absent' else 'Installed'}: {', '.join(SHIM_FILES)}")

    found, verdict = verify_discovery(home)
    result["verified"] = found
    result["verdict"] = verdict
    say(f"{'✓' if found else '✗'} {verdict}")
    if not found:
        result["error"] = verdict
        say("       the files are in place but hermes does not load them —")
        say("       check that this python is the one hermes runs")
        return finish(1)
    result["next"] = _setup_command(home, explicit)
    say(f"Next: {result['next']}")
    return finish(0)
