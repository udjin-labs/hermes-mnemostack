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
import json
import shutil
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
    """(home, how) — the ACTIVE Hermes home, resolved hermes's own way."""
    if explicit:
        return Path(explicit), "--hermes-home"
    try:
        from hermes_constants import get_hermes_home
    except Exception as exc:  # noqa: BLE001 — hermes not installed/importable
        return None, f"hermes-agent is not importable ({type(exc).__name__})"
    try:
        return Path(get_hermes_home()), "hermes-agent's own resolution"
    except Exception as exc:  # noqa: BLE001
        return None, f"hermes could not resolve its home ({type(exc).__name__})"


def _target_state(target: Path) -> str:
    """"absent" | "ours" | "foreign" — what is sitting at the target path."""
    if not target.exists():
        return "absent"
    init = target / "__init__.py"
    if init.is_file() and SHIM_MARKER in init.read_text(encoding="utf-8", errors="replace"):
        return "ours"
    return "foreign"


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
    return f"HERMES_HOME={home} {base}" if explicit else base


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

    state = _target_state(target)
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

    target.mkdir(parents=True, exist_ok=True)
    for name in SHIM_FILES:
        shutil.copyfile(source / name, target / name)
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
