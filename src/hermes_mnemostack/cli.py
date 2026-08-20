"""`hermes-mnemostack status` and `hermes-mnemostack doctor`.

Two questions an operator has that hermes itself cannot answer:

- **status** — what configuration is actually in effect, and would hermes
  activate this provider? No network, no side effects.
- **doctor** — is the thing on the other end actually working? Explicit
  probes, each reported as its own row with a remedy.

Both are deliberately read-only. In particular `doctor` never creates the
Qdrant collection: a diagnostic that provisions storage as a side effect
turns "is my deployment right?" into "my deployment is now different"
(mnemostack's own `doctor` holds the same line). Secrets are never
printed — the API key is reported as set / not set and nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from .config import API_KEY_ENV, CONFIG_FILENAME, availability_problem, load_config

OK = "ok"
WARN = "warn"
FAIL = "fail"

#: doctor's probe query. Deliberately meaningless: it must not look like a
#: real memory if it ever lands in a log the user reads back.
_PROBE_QUERY = "hermes-mnemostack doctor connectivity probe"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    remedy: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str, remedy: str = "") -> Check:
        check = Check(name, status, detail, remedy)
        self.checks.append(check)
        return check

    @property
    def failed(self) -> bool:
        return any(c.status == FAIL for c in self.checks)


_GLYPH = {OK: "✓", WARN: "!", FAIL: "✗"}


def _print_report(report: Report, *, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "status": FAIL if report.failed else OK,
                    "checks": [vars(c) for c in report.checks],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    width = max((len(c.name) for c in report.checks), default=0)
    for c in report.checks:
        print(f"{_GLYPH[c.status]} {c.name.ljust(width)}  {c.detail}")
        if c.remedy and c.status != OK:
            print(f"{' ' * (width + 4)}→ {c.remedy}")


# ------------------------------------------------------------------ config


def _config_rows(hermes_home: str | None) -> tuple[dict[str, Any] | None, str]:
    """(config, error) — the resolved config, or the reason it can't load."""
    try:
        return load_config(hermes_home), ""
    except Exception as exc:  # noqa: BLE001 — every config error is user-facing
        return None, str(exc)


def _describe_config(cfg: dict[str, Any]) -> str:
    mode = cfg["mode"]
    if mode == "remote":
        return (
            f"mode=remote base_url={cfg['base_url'] or '(unset)'} "
            f"timeout={cfg['timeout']}s api_key="
            f"{'set' if cfg.get('api_key') else 'NOT set'}"
        )
    return (
        f"mode=local qdrant_url={cfg['qdrant_url']} collection={cfg['collection']} "
        f"embedding={cfg['embedding_provider']}"
        + (f"/{cfg['embedding_model']}" if cfg["embedding_model"] else "")
    )


def _config_source(hermes_home: str | None) -> str:
    """Where the settings came from — the single most common support question."""
    from .config import _config_path  # noqa: PLC0415 — internal helper by design

    parts = []
    try:
        path = _config_path(hermes_home)
        parts.append(f"{path} ({'present' if path.is_file() else 'absent'})")
    except Exception as exc:  # noqa: BLE001 — hermes_constants missing/broken
        parts.append(f"config path unresolved ({type(exc).__name__}: {exc})")
    env = sorted(k for k in os.environ if k.startswith("MNEMOSTACK_") and os.environ[k])
    # Names only. MNEMOSTACK_API_KEY's VALUE must never reach stdout.
    parts.append("env: " + (", ".join(env) if env else "none"))
    return "; ".join(parts)


# --------------------------------------------------------------- discovery


def _discovery_check(report: Report) -> None:
    """Will hermes-agent actually find this provider in this environment?

    Asked FUNCTIONALLY — through hermes's own discovery — instead of by
    comparing version numbers: entry-point discovery landed in 0.20, but a
    0.19 install with the provider dropped into `$HERMES_HOME/plugins/` is
    equally discoverable, and only hermes's own scan knows which is true
    here.
    """
    import importlib.metadata as md

    try:
        agent_version = md.version("hermes-agent")
    except Exception:  # noqa: BLE001 — not installed at all
        report.add(
            "hermes-agent",
            FAIL,
            "not installed",
            "pip install 'hermes-agent>=0.19' (>=0.20 for entry-point discovery)",
        )
        return
    try:
        from plugins.memory import list_memory_provider_names
    except Exception as exc:  # noqa: BLE001 — layout differs across versions
        report.add(
            "discovery",
            WARN,
            f"hermes-agent {agent_version} — discovery module not importable "
            f"({type(exc).__name__}); cannot verify registration",
        )
        return
    try:
        names = list(list_memory_provider_names())
    except Exception as exc:  # noqa: BLE001
        report.add("discovery", WARN, f"discovery scan failed: {type(exc).__name__}: {exc}")
        return
    if "mnemostack" in names:
        report.add("discovery", OK, f"hermes-agent {agent_version} discovers 'mnemostack'")
        return
    eps = [
        ep
        for ep in md.entry_points(group="hermes_agent.memory_providers")
        if ep.name == "mnemostack"
    ]
    if eps:
        # The package IS installed and registered — this hermes just does
        # not read entry points yet. That is the 0.19 story exactly, and
        # it is a WARN, not a FAIL: the directory-plugin install works.
        report.add(
            "discovery",
            WARN,
            f"entry point registered, but hermes-agent {agent_version} discovers only "
            f"directory plugins (found: {', '.join(names) or 'none'})",
            "upgrade to hermes-agent >= 0.20, or install this package as a directory "
            "plugin under $HERMES_HOME/plugins/mnemostack/",
        )
        return
    report.add(
        "discovery",
        FAIL,
        f"hermes-agent {agent_version} does not see 'mnemostack' "
        f"(found: {', '.join(names) or 'none'})",
        "install hermes-mnemostack into the SAME environment as hermes-agent",
    )


# ----------------------------------------------------------------- probes


def _probe_remote(report: Report, cfg: dict[str, Any], http: Any | None = None) -> None:
    """`http` is dependency injection for tests (the same seam RemoteClient
    offers) — an injected client is used as given and never closed here."""
    base = cfg["base_url"]
    if http is not None:
        _probe_remote_with(report, cfg, http)
        return
    try:
        import httpx
    except Exception:  # noqa: BLE001 — remote mode without the http dep
        report.add("service", FAIL, "httpx is not installed", "pip install httpx")
        return
    try:
        client = httpx.Client(base_url=base, timeout=cfg["timeout"])
    except Exception as exc:  # noqa: BLE001 — client construction (bad URL)
        report.add("service", FAIL, f"cannot build an HTTP client for {base!r}: {exc}")
        return
    try:
        _probe_remote_with(report, cfg, client)
    finally:
        client.close()


def _probe_remote_with(report: Report, cfg: dict[str, Any], http: Any) -> None:
    base = cfg["base_url"]
    try:
        health = http.get("/health")
    except Exception as exc:  # noqa: BLE001
        report.add(
            "service",
            FAIL,
            f"{base} unreachable ({type(exc).__name__})",
            "check base_url, that `mnemostack serve` is running, and any proxy",
        )
        return
    if health.status_code >= 400:
        report.add("service", FAIL, f"GET /health returned {health.status_code}")
        return
    body: dict[str, Any] = {}
    try:
        body = health.json()
    except Exception:  # noqa: BLE001 — a proxy's non-JSON 200
        pass
    report.add(
        "service",
        OK,
        f"{base or 'service'} reachable"
        + (f" (mnemostack {body['version']})" if body.get("version") else ""),
    )
    # Read-scope probe. Deliberately no write probe: a doctor run must not
    # create memories, and a rejected write would be indistinguishable from
    # a quota rejection anyway.
    try:
        resp = http.post(
            "/recall",
            json={"query": _PROBE_QUERY, "limit": 1},
            headers={"X-API-Key": cfg["api_key"]} if cfg.get("api_key") else {},
        )
    except Exception as exc:  # noqa: BLE001
        report.add("recall", FAIL, f"probe failed ({type(exc).__name__})")
        return
    _report_recall_status(report, resp, authed=bool(cfg.get("api_key")))


def _report_recall_status(report: Report, resp: Any, *, authed: bool) -> None:
    if resp.status_code == 401:
        report.add(
            "recall",
            FAIL,
            "401 — the service requires a key and this one was rejected"
            if authed
            else "401 — the service requires a key and none is set",
            f"export {API_KEY_ENV}=<key issued by `mnemostack keys add`>",
        )
        return
    if resp.status_code == 403:
        report.add(
            "recall",
            FAIL,
            "403 — the key lacks the 'read' scope",
            "reissue with `mnemostack keys add --scopes read,write`",
        )
        return
    if resp.status_code >= 400:
        report.add("recall", FAIL, f"probe recall returned {resp.status_code}")
        return
    data = {}
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        report.add("recall", WARN, "probe recall returned a non-JSON body")
        return
    hits = len(data.get("results", []))
    report.add(
        "recall",
        OK,
        f"read scope confirmed ({hits} hit(s) for a probe query — 0 is normal)"
        + ("" if authed else "; NO key set: the service is running unauthenticated"),
    )
    _report_degradation(report, data.get("degraded", []), data.get("notes", []))


def _report_degradation(report: Report, degraded: list[Any], notes: list[Any]) -> None:
    """Real faults only. mnemostack 2.2's `degraded` still duplicates the
    routine `notes` tags for back-compat, so the difference — not the raw
    list — is what an operator should act on."""
    from .client import real_faults

    faults = real_faults([str(d) for d in degraded], [str(n) for n in notes])
    if faults:
        report.add(
            "retrieval",
            WARN,
            "recall reported real degradation: " + ", ".join(faults),
            "check the service log; some retrieval arm is failing, not merely idle",
        )
    else:
        report.add(
            "retrieval",
            OK,
            "no faults" + (f" (routine notes: {', '.join(str(n) for n in notes)})" if notes else ""),
        )


def _probe_local(report: Report, cfg: dict[str, Any]) -> None:
    try:
        from mnemostack.embeddings import get_provider
        from mnemostack.vector import VectorStore
    except Exception as exc:  # noqa: BLE001
        report.add("mnemostack", FAIL, f"library import failed: {exc}", "pip install 'mnemostack>=2.2'")
        return
    kwargs: dict[str, Any] = {}
    if cfg["embedding_model"]:
        kwargs["model"] = cfg["embedding_model"]
    try:
        provider = get_provider(cfg["embedding_provider"], **kwargs)
    except Exception as exc:  # noqa: BLE001
        report.add(
            "embeddings",
            FAIL,
            f"provider {cfg['embedding_provider']!r} unusable: {exc}",
            "check embedding_provider / embedding_model and the provider's own service",
        )
        return
    try:
        healthy, message = provider.health_check()
    except Exception as exc:  # noqa: BLE001
        healthy, message = False, f"{type(exc).__name__}: {exc}"
    report.add(
        "embeddings",
        OK if healthy else FAIL,
        f"{cfg['embedding_provider']} — {message}",
        "" if healthy else "start the embedding backend (e.g. `ollama serve`) or switch providers",
    )
    try:
        store = VectorStore(
            collection=cfg["collection"], dimension=provider.dimension, host=cfg["qdrant_url"]
        )
        exists = store.collection_exists()
    except Exception as exc:  # noqa: BLE001
        report.add(
            "qdrant",
            FAIL,
            f"{cfg['qdrant_url']} unreachable ({type(exc).__name__}: {exc})",
            "start Qdrant or fix qdrant_url",
        )
        return
    if exists:
        try:
            points = store.count()
        except Exception:  # noqa: BLE001 — reachable but the count failed
            points = -1
        report.add(
            "qdrant",
            OK,
            f"{cfg['qdrant_url']} reachable; collection {cfg['collection']!r} exists"
            + (f" ({points} points)" if points >= 0 else ""),
        )
    else:
        # NOT a failure and NOT created here: the first session creates it.
        report.add(
            "qdrant",
            WARN,
            f"{cfg['qdrant_url']} reachable; collection {cfg['collection']!r} does not exist yet",
            "it is created on the first session — nothing to do unless you expected data",
        )


# --------------------------------------------------------------- commands


def cmd_status(args: argparse.Namespace) -> int:
    report = Report()
    report.add("config source", OK, _config_source(args.hermes_home))
    cfg, err = _config_rows(args.hermes_home)
    if cfg is None:
        report.add("config", FAIL, err, f"fix or remove {CONFIG_FILENAME}")
    else:
        report.add("config", OK, _describe_config(cfg))
    problem = availability_problem(args.hermes_home)
    report.add(
        "availability",
        OK if problem is None else FAIL,
        "hermes would activate this provider" if problem is None else problem,
    )
    _print_report(report, as_json=args.json)
    return 1 if report.failed else 0


def cmd_doctor(args: argparse.Namespace, *, http: Any | None = None) -> int:
    report = Report()
    report.add("config source", OK, _config_source(args.hermes_home))
    cfg, err = _config_rows(args.hermes_home)
    if cfg is None:
        report.add("config", FAIL, err, f"fix or remove {CONFIG_FILENAME}")
        _print_report(report, as_json=args.json)
        return 1
    report.add("config", OK, _describe_config(cfg))
    problem = availability_problem(args.hermes_home)
    report.add(
        "availability",
        OK if problem is None else FAIL,
        "hermes would activate this provider" if problem is None else problem,
    )
    _discovery_check(report)
    # Probe regardless of the availability verdict: "unavailable AND the
    # service is down" is two remedies, and reporting only the first sends
    # the operator back for a second round.
    if cfg["mode"] == "remote":
        if cfg["base_url"] or http is not None:
            _probe_remote(report, cfg, http)
        else:
            report.add("service", FAIL, "no base_url to probe", "set MNEMOSTACK_BASE_URL")
    else:
        _probe_local(report, cfg)
    _print_report(report, as_json=args.json)
    return 1 if report.failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes-mnemostack",
        description="Inspect and diagnose the mnemostack memory provider for hermes-agent.",
    )
    parser.add_argument(
        "--hermes-home",
        default=None,
        help="Hermes home directory (default: hermes-agent's own resolution)",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="Show the effective configuration (no network)")
    sub.add_parser("doctor", help="Probe the configured transport and report remedies")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        return cmd_status(args)
    return cmd_doctor(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
