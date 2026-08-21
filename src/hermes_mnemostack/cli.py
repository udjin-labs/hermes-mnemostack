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
import re
import sys
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
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


#: Control characters have no business in a report row. A newline in one
#: lets a hostile server FORGE a row: the plain-text renderer prints one
#: line per check, so "2.2.0)\n\u2713 FORGED  all fine (" arrives as an
#: extra line that looks exactly like a genuine passing check in the
#: output an operator reads — or pastes into a support thread — to decide
#: whether their deployment is healthy.
#: Categories, not a hand-listed set of characters. Every earlier attempt
#: here was a list somebody remembered to write, and each one was missing
#: its next member (U+2028 after C0/C1, then U+061C after that). Unicode
#: already classifies exactly what must not reach a report row:
#:   Cc  control characters — a newline forges a whole extra row, because
#:       the renderer prints one line per check;
#:   Zl/Zp  line and paragraph separators — str.splitlines() breaks on
#:       them too, and so do the renderers a pasted report passes through;
#:   Cf  format characters — bidi overrides, isolates, the Arabic letter
#:       mark, zero-width joiners and the BOM. They cannot start a row,
#:       but they reorder or hide what one says, which is the same lie.
#: Ordinary text of any script is untouched; an emoji ZWJ sequence in a
#: version string is the one thing this flattens, which a diagnostic row
#: can live with.
_STRIPPED_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})
#: And a row is a row, not a document: an unbounded field would flood the
#: report even without a newline in it.
_ROW_MAX_CHARS = 300


def _row_text(value: Any) -> str:
    """One row's worth of text, whatever the value was.

    Applied at the single point where rows are created, so a field that
    reaches a report from a SERVER response — a /health version, a
    degradation tag — cannot forge structure no matter which call site it
    travelled through.
    """
    # Cut BEFORE scrubbing: the scrub is 1:1 per character, so slicing
    # first is byte-for-byte the same answer without walking a field the
    # report will never show.
    raw = str(value)
    ellipsis = len(raw) > _ROW_MAX_CHARS
    if ellipsis:
        raw = raw[: _ROW_MAX_CHARS - 1]
    cleaned = "".join(
        " " if unicodedata.category(ch) in _STRIPPED_CATEGORIES else ch for ch in raw
    )
    return cleaned + "\u2026" if ellipsis else cleaned


#: How much of ONE server-supplied value a row may spend. The row cap is a
#: backstop against total length; this is the one that keeps a server from
#: STEERING what gets cut. A row is composed as "<our words><their
#: value><our words>", so an unbounded value pushes the trailing clause —
#: the diagnostically important half — past the row cap and out of the
#: report, while the value itself survives. Bounding the value instead
#: means our own words always fit.
_FIELD_MAX_CHARS = 80


def _field_text(value: Any) -> str:
    """One server-supplied value, bounded so it cannot displace our text."""
    text = str(value)
    return text if len(text) <= _FIELD_MAX_CHARS else text[: _FIELD_MAX_CHARS - 1] + "\u2026"


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str, remedy: str = "") -> Check:
        check = Check(_row_text(name), status, _row_text(detail), _row_text(remedy))
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


@contextmanager
def _hermes_home(path: str | None) -> Iterator[None]:
    """Scope hermes's own home resolution to `--hermes-home` for the block.

    Discovery scans `$HERMES_HOME/plugins`, so a diagnostic pointed at one
    home that then reports on ANOTHER home's plugin directory can print the
    opposite verdict about the very directory being diagnosed. hermes
    exposes a context-local override for exactly this (it deliberately does
    not touch os.environ, which is process-wide); when a build lacks it we
    simply do not override, rather than mutating the environment behind the
    host's back.
    """
    if not path:
        yield
        return
    try:
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )
    except Exception:  # noqa: BLE001 — older/newer layout without the hooks
        yield
        return
    token = set_hermes_home_override(path)
    try:
        yield
    finally:
        # The PUBLIC reset, the one every hermes-agent call site uses:
        # it restores the token's prior value. Setting the override to None
        # instead would clear an OUTER override (hermes invoking this
        # programmatically inside its own profile-scoped context) rather
        # than restoring it.
        reset_hermes_home_override(token)


def _discovery_check(report: Report, hermes_home: str | None = None) -> None:
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
        with _hermes_home(hermes_home):
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


#: A hostname, positively: letters, digits, dots, hyphens and underscores.
#: Underscores are not RFC 1123, but internal DNS (compose/kubernetes
#: service names) uses them and an operator's redirect really can point at
#: one — dropping those would make a legitimate target read like an attack.
_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
#: An IPv6 literal, positively: hex digits, colons, the dots of an embedded
#: IPv4 tail, and an optional zone id.
_IPV6_RE = re.compile(r"^[0-9A-Fa-f:.]+(?:%(?:25)?[A-Za-z0-9._-]+)?$")
#: The only schemes this client can follow. Anything else is not a
#: destination we could reach even if we wanted to.
_FOLLOWABLE_SCHEMES = ("http", "https")


def _safe_location(location: str | None) -> str:
    """Where a redirect points, named ONLY by its validated authority.

    Three rounds of review found a secret in three different positions of
    this string — the query, then a path parameter, then the host — and
    the fourth found that a Location with no `//` authority makes urlsplit
    hand the ENTIRE remainder over as the "path", which sailed past every
    check. "Places a secret can sit in a URL" is not a bounded list, so
    this stops enumerating them.

    The contract: the report names the DESTINATION AUTHORITY and nothing
    else. Printed output can only ever be one of the two FOLLOWABLE
    schemes (a rejected one is described, never quoted — urlsplit's scheme
    grammar is unbounded and can carry a token), a host matching the
    hostname (or IPv6-literal) grammar, and a numeric port. The path, query, fragment and userinfo are never shown — not
    redacted char by char, simply not printed — and a Location that is not
    a followable http(s) URL with a valid authority contributes NO
    content at all, only its shape.

    What that trades away: an operator no longer sees the redirect's path.
    It buys a property worth more to a diagnostic whose output gets pasted
    into support threads — no attacker-chosen substring can reach the
    report — and the host is what actually answers "where did my base_url
    send me". Curl the endpoint for the rest.
    """
    if not location:
        return "unknown"
    try:
        from urllib.parse import urlsplit

        parts = urlsplit(location)
        scheme = parts.scheme.lower()
        try:
            port = parts.port
        except ValueError:
            # urlsplit only validates the port when it is READ; a
            # non-numeric one means the authority is not what it claims.
            return "a malformed redirect target (withheld)"
        host = parts.hostname or ""
        if host and ":" in host:
            if not _IPV6_RE.match(host):
                return "a malformed redirect target (withheld)"
            host = f"[{host}]"  # urlsplit strips an IPv6 literal's brackets
        elif host and not _HOSTNAME_RE.match(host):
            # e.g. "evil.com;jsessionid=SECRET" — urlsplit hands the whole
            # thing over as the "host".
            return "a malformed redirect target (withheld)"
        if not host:
            # A relative Location ("/login"), or an opaque one with no
            # authority at all ("data:…", a backslash-mangled target). Its
            # only content is attacker-chosen text, so none is printed.
            return "a redirect with no host (target withheld)"
        if scheme and scheme not in _FOLLOWABLE_SCHEMES:
            # The scheme is NOT echoed. urlsplit's grammar
            # ([a-zA-Z][a-zA-Z0-9+.-]*, unbounded) is wide enough to carry a
            # hex token or a UUID, so printing a rejected one would reopen
            # exactly the class this contract closed — in the branch added
            # to close it.
            return "a redirect to a scheme this client cannot follow (target withheld)"
        authority = f"{host}:{port}" if port is not None else host
        # No scheme = scheme-relative ("//host/path"), which inherits the
        # request's own — a valid and followable form.
        shown = f"{scheme}://{authority}" if scheme else f"//{authority}"
        return f"{shown} (path and query not shown)"
    except Exception:  # noqa: BLE001 — a malformed Location must not leak or crash
        return "unparseable (withheld)"


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
    if 300 <= health.status_code < 400:
        # httpx does not follow redirects by default, and the provider's own
        # client does not either — so a base_url that redirects is a service
        # this deployment cannot actually talk to. Reporting "reachable" for
        # a 302 would be the worst kind of green.
        report.add(
            "service",
            FAIL,
            f"GET /health redirected ({health.status_code} → "
            f"{_safe_location(health.headers.get('location'))})",
            "point base_url at the FINAL url — the provider's client does not "
            "follow redirects",
        )
        return
    if health.status_code >= 400 or health.status_code < 200:
        report.add("service", FAIL, f"GET /health returned {health.status_code}")
        return
    body: dict[str, Any] = {}
    try:
        body = health.json()
    except Exception:  # noqa: BLE001 — a proxy's non-JSON 200
        pass
    if not isinstance(body, dict) or "status" not in body:
        # A 200 from something that is not mnemostack (a proxy's login page,
        # an SPA index) must not read as a healthy memory service.
        report.add(
            "service",
            FAIL,
            f"{base or 'service'} answered /health, but not with a mnemostack "
            "health document",
            "check base_url — something else is serving this path",
        )
        return
    version = (
        f" (mnemostack {_field_text(body['version'])})" if body.get("version") else ""
    )
    if body.get("status") != "ok":
        # FAIL, not WARN: mnemostack reports `degraded` exactly when Qdrant
        # — its hard dependency — is unreachable. Recall is fail-soft and
        # can still answer 200 with nothing in it, so a WARN here let
        # `doctor` exit 0 and `--json` say "ok" while memory was, in fact,
        # not working. A diagnostic's whole job is to not do that.
        report.add(
            "service",
            FAIL,
            f"{base or 'service'} reachable but reports status="
            f"{_field_text(body.get('status'))!r}{version}"
            + (" — qdrant unreachable" if body.get("qdrant") is False else ""),
            "check the service's own /health and its backing stores",
        )
    else:
        report.add("service", OK, f"{base or 'service'} reachable{version}")
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
    if not 200 <= resp.status_code < 300:
        # 3xx included: an unfollowed redirect proves nothing about the read
        # scope, and treating it as anything but a failure would let doctor
        # exit 0 having confirmed nothing.
        report.add("recall", FAIL, f"probe recall returned {resp.status_code}")
        return
    data = {}
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        report.add("recall", FAIL, "probe recall returned a non-JSON body")
        return
    # TYPE, not just key presence: a null field is exactly the plausible
    # malformation this check exists for (a proxy, a non-conformant server,
    # a future version emitting null for an empty list), and iterating it
    # would hand the operator a raw traceback instead of the FAIL row a
    # diagnostic owes them.
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        report.add("recall", FAIL, "probe recall returned an unexpected document")
        return
    hits = len(data["results"])
    report.add(
        "recall",
        OK,
        f"read scope confirmed ({hits} hit(s) for a probe query — 0 is normal)"
        + ("" if authed else "; NO key set: the service is running unauthenticated"),
    )
    _report_degradation(report, data.get("degraded"), data.get("notes"))


def _report_degradation(report: Report, degraded: Any, notes: Any) -> None:
    """Real faults only. mnemostack 2.2's `degraded` still duplicates the
    routine `notes` tags for back-compat, so the difference — not the raw
    list — is what an operator should act on."""
    from .client import real_faults

    # Coerce before iterating: null/absent/scalar are all "nothing to
    # report", never a crash inside the report renderer.
    degraded_list = list(degraded) if isinstance(degraded, list) else []
    notes = list(notes) if isinstance(notes, list) else []
    faults = real_faults([str(d) for d in degraded_list], [str(n) for n in notes])
    if faults:
        report.add(
            "retrieval",
            WARN,
            "recall reported real degradation: "
            + _field_text(", ".join(faults)),
            "check the service log; some retrieval arm is failing, not merely idle",
        )
    else:
        report.add(
            "retrieval",
            OK,
            "no faults"
            + (
                f" (routine notes: {_field_text(', '.join(str(n) for n in notes))})"
                if notes
                else ""
            ),
        )


def _collection_dimension(store: Any, collection: str) -> int | None:
    """The existing collection's dense vector size, or None if it cannot be
    read (a custom store, a named-vectors config with no single size, an
    unreachable call). Read-only — never creates anything."""
    try:
        info = store.client.get_collection(collection)
        return getattr(getattr(info.config.params, "vectors", None), "size", None)
    except Exception:  # noqa: BLE001 — best effort; absence is not a failure
        return None


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
        # An existing collection built with ANOTHER embedding model has a
        # different vector size, and the provider's first session dies on it
        # (ensure_collection raises DimensionMismatchError). Reading the
        # size is the whole point of a diagnostic — reporting "exists, ok"
        # and letting initialize() discover it is a false green.
        size = _collection_dimension(store, cfg["collection"])
        if size is not None and int(size) != int(provider.dimension):
            report.add(
                "qdrant",
                FAIL,
                f"collection {cfg['collection']!r} stores {size}-dim vectors but "
                f"{cfg['embedding_provider']} produces {provider.dimension}-dim",
                "point `collection` at a collection built with this embedding "
                "model, or re-index into a new one",
            )
            return
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
    _discovery_check(report, args.hermes_home)
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
    # The shared options are registered on the root parser AND on every
    # subcommand: argparse otherwise rejects `doctor --json` (options bind to
    # the parser they were declared on), which is exactly how anyone types it
    # and exactly what the README documents.
    # SUPPRESS, not a real default: a subparser writes its defaults into the
    # SAME namespace, so a plain default would let `--hermes-home X status`
    # be silently reset to None by the subcommand. With SUPPRESS an unset
    # option contributes nothing, and main() fills the defaults once.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--hermes-home",
        default=argparse.SUPPRESS,
        help="Hermes home directory (default: hermes-agent's own resolution)",
    )
    common.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Machine-readable output",
    )
    parser = argparse.ArgumentParser(
        prog="hermes-mnemostack",
        description="Inspect and diagnose the mnemostack memory provider for hermes-agent.",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "status", parents=[common], help="Show the effective configuration (no network)"
    )
    sub.add_parser(
        "doctor",
        parents=[common],
        help="Probe the configured transport and report remedies",
    )
    p_install = sub.add_parser(
        "install",
        parents=[common],
        help="Install the directory shim hermes-agent 0.19 needs to find this provider",
    )
    p_install.add_argument(
        "--force",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Overwrite a plugin directory of the same name that is not this shim",
    )
    p_install.add_argument(
        "--dry-run",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Report what would be written, write nothing",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse, then fill the suppressed shared defaults exactly once."""
    args = build_parser().parse_args(argv)
    args.hermes_home = getattr(args, "hermes_home", None)
    args.json = getattr(args, "json", False)
    args.force = getattr(args, "force", False)
    args.dry_run = getattr(args, "dry_run", False)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "install":
        from .install import cmd_install

        return cmd_install(args)
    return cmd_doctor(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
