"""Directory-plugin shim: how hermes-agent 0.19 finds this provider.

0.19 discovers memory providers by SCANNING two directories — the bundled
`plugins/memory/<name>/` inside hermes itself, and `$HERMES_HOME/plugins/
<name>/`. It does not read pip entry points; that landed in 0.20. So a
`pip install` alone is invisible to it, and this directory is what closes
the gap: `hermes-mnemostack install` copies it to
`$HERMES_HOME/plugins/mnemostack/`, and hermes picks it up from there.

Three properties this file must keep, each for a documented reason:

- **The literal `register_memory_provider` must appear here**, in the first
  8 KB. Before importing anything, 0.19 text-scans `__init__.py` for that
  substring (or `MemoryProvider`) to decide whether the directory is a
  memory provider at all — so hiding the registration behind an imported
  helper would make the directory invisible.
- **It stays thin.** The provider lives in the installed package; this is
  a pointer to it, not a copy. A stale copy of this file is therefore
  harmless — the behaviour comes from whatever version of the package is
  installed.
- **The package must already be installed.** `pip_dependencies` in
  plugin.yaml cannot bootstrap it: 0.19 imports this module during
  DISCOVERY and only reads the dependency list after the provider has been
  selected in the picker. Worse, a failed import is logged at debug level
  and the provider silently disappears — which is why `install` verifies
  the import itself and says so loudly.
"""

from __future__ import annotations

from hermes_mnemostack.provider import MnemostackProvider


def register(ctx) -> None:
    """hermes-agent 0.19 plugin entry point."""
    ctx.register_memory_provider(MnemostackProvider())


__all__ = ["MnemostackProvider", "register"]
