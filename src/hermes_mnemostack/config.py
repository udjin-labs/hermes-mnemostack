"""Configuration for the mnemostack memory provider.

Follows the hermes-agent provider convention (see the bundled providers):
behavioral settings live in ``$HERMES_HOME/mnemostack.json`` (written by
``hermes memory setup``), secrets live in the environment / ``.env``
(``MNEMOSTACK_API_KEY``), and environment variables provide defaults that
the JSON file overrides.

Two transports behind one ``mode`` switch:

- ``remote`` — a running mnemostack service (``mnemostack serve``); the
  tenant is resolved server-side from the service key. Requires
  mnemostack >= 2.2 on the SERVER (the remote write/lifecycle surface).
- ``local`` — mnemostack as a library against your own Qdrant; profile
  isolation is tenant-scoped through the library (same machine, same
  trust domain).

Not shipped yet (deliberately absent rather than a silent no-op knob):
capture of tool calls/results, which needs its own redaction policy —
tool output routinely carries paths, tokens and private workspace
contents.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "mnemostack.json"
API_KEY_ENV = "MNEMOSTACK_API_KEY"

MODES = ("remote", "local")

_DEFAULTS: dict[str, Any] = {
    "mode": "remote",
    # remote transport
    "base_url": "",
    "timeout": 30.0,
    # local transport
    "qdrant_url": "http://localhost:6333",
    "collection": "hermes-memory",
    "embedding_provider": "ollama",
    "embedding_model": "",
    # behavior
    "recall_limit": 5,
    "capture": True,          # store user+assistant turns
}

_ENV_MAP = {
    "mode": "MNEMOSTACK_MODE",
    "base_url": "MNEMOSTACK_BASE_URL",
    "timeout": "MNEMOSTACK_TIMEOUT",
    "qdrant_url": "MNEMOSTACK_QDRANT_URL",
    "collection": "MNEMOSTACK_COLLECTION",
    "embedding_provider": "MNEMOSTACK_EMBEDDING_PROVIDER",
    "embedding_model": "MNEMOSTACK_EMBEDDING_MODEL",
    "recall_limit": "MNEMOSTACK_RECALL_LIMIT",
    "capture": "MNEMOSTACK_CAPTURE",
}

_BOOL_KEYS = {"capture"}
_FLOAT_KEYS = {"timeout"}
_INT_KEYS = {"recall_limit"}


def _config_path(hermes_home: str | None = None) -> Path:
    if hermes_home:
        return Path(hermes_home) / CONFIG_FILENAME
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / CONFIG_FILENAME


def _coerce(key: str, value: Any) -> Any:
    """Coerce a string (env) or JSON value onto the key's expected type.

    Raises ValueError on garbage — a mistyped config must fail setup
    loudly, not silently fall back to a default mid-session.
    """
    if key in _BOOL_KEYS:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("1", "true", "yes", "on"):
            return True
        if isinstance(value, str) and value.strip().lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{key} must be a boolean, got {value!r}")
    if key in _FLOAT_KEYS:
        out = float(value)
        if not out > 0:
            raise ValueError(f"{key} must be positive, got {value!r}")
        return out
    if key in _INT_KEYS:
        out = int(value)
        if isinstance(value, float) and value != out:
            raise ValueError(f"{key} must be an integer, got {value!r}")
        if not out > 0:
            raise ValueError(f"{key} must be positive, got {value!r}")
        return out
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string, got {value!r}")
    return value


def load_config(hermes_home: str | None = None) -> dict[str, Any]:
    """Defaults <- environment <- $HERMES_HOME/mnemostack.json (file wins).

    The api_key is env-only (never written to JSON) and returned under
    ``api_key``. Unknown JSON keys are rejected loudly — a typo'd key
    silently doing nothing is the worst failure mode for memory config.
    """
    cfg = dict(_DEFAULTS)
    for key, env in _ENV_MAP.items():
        raw = os.environ.get(env)
        if raw is not None:
            cfg[key] = _coerce(key, raw)
    path = _config_path(hermes_home)
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a JSON object")
        unknown = set(data) - set(_DEFAULTS)
        if unknown:
            raise ValueError(
                f"unknown key(s) in {path.name}: {', '.join(sorted(unknown))}"
            )
        for key, value in data.items():
            cfg[key] = _coerce(key, value)
    if cfg["mode"] not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {cfg['mode']!r}")
    cfg["api_key"] = os.environ.get(API_KEY_ENV, "")
    return cfg


def is_configured(hermes_home: str | None = None) -> bool:
    """Whether the operator has explicitly configured this provider.

    Availability must never come from pure defaults: the config file
    exists, or the mode was selected via environment. No network calls
    (the ABC contract for is_available)."""
    if os.environ.get(_ENV_MAP["mode"]):
        return True
    try:
        return _config_path(hermes_home).is_file()
    except Exception:  # pragma: no cover — hermes_constants missing/broken
        return False


def save_config(values: dict[str, Any], hermes_home: str) -> None:
    """Write non-secret config for `hermes memory setup` (secrets go to .env)."""
    clean: dict[str, Any] = {}
    for key, value in values.items():
        if key not in _DEFAULTS:
            raise ValueError(f"unknown config key: {key}")
        clean[key] = _coerce(key, value)
    path = Path(hermes_home) / CONFIG_FILENAME
    path.write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8")
