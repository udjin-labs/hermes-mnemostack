"""`hermes-mnemostack status` / `doctor`.

The operator questions these answer are "what config is in effect and
would hermes activate this?" and "is the other end actually working?" —
so the tests care about three things above all: no secret ever reaches
stdout, `doctor` creates nothing, and its availability verdict cannot
disagree with what hermes itself would decide.
"""

from __future__ import annotations

import json

import pytest

from hermes_mnemostack import cli
from hermes_mnemostack.provider import MnemostackProvider


def _args(command: str, hermes_home, *, as_json: bool = False):
    argv = ["--hermes-home", str(hermes_home)]
    if as_json:
        argv.append("--json")
    return cli.parse_args([*argv, command])


def _write_config(home, **values):
    (home / "mnemostack.json").write_text(json.dumps(values), encoding="utf-8")


@pytest.fixture()
def offline_local_stack(monkeypatch):
    """Local-mode probes must never reach a real Qdrant or embedding
    backend from the test suite — a doctor test that silently talks to the
    developer's own running stack passes locally and fails in CI."""

    class _Store:
        def __init__(self, **kw):
            self.kw = kw

        def ensure_collection(self, **_kw):  # pragma: no cover — must not be called
            raise AssertionError("doctor must not create the collection")

        def collection_exists(self):
            return True

        def count(self):
            return 0

    class _Provider:
        dimension = 3

        def health_check(self):
            return True, "ok"

    monkeypatch.setattr("mnemostack.vector.VectorStore", _Store)
    monkeypatch.setattr("mnemostack.embeddings.get_provider", lambda _n, **_k: _Provider())
    return _Store, _Provider


# ------------------------------------------------------------------ status


def test_status_reports_the_effective_config(tmp_path, capsys):
    _write_config(tmp_path, mode="remote", base_url="http://memory.invalid:8080")
    rc = cli.cmd_status(_args("status", tmp_path))
    out = capsys.readouterr().out
    assert rc == 0
    assert "mode=remote" in out and "http://memory.invalid:8080" in out
    assert "hermes would activate this provider" in out


def test_status_never_prints_the_api_key(tmp_path, capsys, monkeypatch):
    secret = "sk-do-not-print-me-0123456789"
    monkeypatch.setenv("MNEMOSTACK_API_KEY", secret)
    _write_config(tmp_path, mode="remote", base_url="http://memory.invalid:8080")
    cli.cmd_status(_args("status", tmp_path))
    out = capsys.readouterr().out
    assert secret not in out
    assert "api_key=set" in out
    # The env listing names variables; it must not print their values.
    assert "MNEMOSTACK_API_KEY" in out


def test_status_reports_a_missing_key_as_not_set(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("MNEMOSTACK_API_KEY", raising=False)
    _write_config(tmp_path, mode="remote", base_url="http://memory.invalid:8080")
    cli.cmd_status(_args("status", tmp_path))
    assert "api_key=NOT set" in capsys.readouterr().out


def test_status_fails_on_an_unusable_config(tmp_path, capsys):
    _write_config(tmp_path, mode="remote")  # remote without a base_url
    rc = cli.cmd_status(_args("status", tmp_path))
    out = capsys.readouterr().out
    assert rc == 1
    assert "needs a base_url" in out


def test_status_fails_loudly_on_a_broken_config_file(tmp_path, capsys):
    (tmp_path / "mnemostack.json").write_text("{ not json", encoding="utf-8")
    rc = cli.cmd_status(_args("status", tmp_path))
    assert rc == 1
    assert "config is invalid" in capsys.readouterr().out


def test_status_verdict_matches_the_provider(tmp_path, monkeypatch):
    """The CLI must never say 'available' about a provider hermes would
    refuse to activate — both read ONE rule."""
    monkeypatch.setenv("MNEMOSTACK_MODE", "remote")
    monkeypatch.delenv("MNEMOSTACK_BASE_URL", raising=False)
    from hermes_mnemostack.config import availability_problem

    monkeypatch.setattr(
        "hermes_mnemostack.config._config_path", lambda _h=None: tmp_path / "mnemostack.json"
    )
    assert (availability_problem() is None) is MnemostackProvider().is_available()
    _write_config(tmp_path, mode="remote", base_url="http://memory.invalid:8080")
    assert (availability_problem() is None) is MnemostackProvider().is_available()


def test_json_output_is_machine_readable(tmp_path, capsys):  # status: no probes
    _write_config(tmp_path, mode="local", collection="hermes-memory")
    cli.cmd_status(_args("status", tmp_path, as_json=True))
    body = json.loads(capsys.readouterr().out)
    assert body["status"] == "ok"
    assert {c["name"] for c in body["checks"]} >= {"config", "availability"}


# ------------------------------------------------------------------ doctor


def test_doctor_probes_a_real_service_and_confirms_the_read_scope(
    tmp_path, service_app, capsys, monkeypatch
):
    """Against the REAL mnemostack app: /health plus an authenticated
    recall, reported as the read scope being present."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app, _store, _emb, keys = service_app
    monkeypatch.setenv("MNEMOSTACK_API_KEY", keys["alpha"])
    _write_config(tmp_path, mode="remote", base_url="http://testserver")
    rc = cli.cmd_doctor(_args("doctor", tmp_path), http=TestClient(app))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "reachable" in out
    assert "read scope confirmed" in out


def test_doctor_reports_a_rejected_key_as_a_key_problem(tmp_path, service_app, capsys, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app, _store, _emb, _keys = service_app
    monkeypatch.setenv("MNEMOSTACK_API_KEY", "not-a-valid-key")
    _write_config(tmp_path, mode="remote", base_url="http://testserver")
    rc = cli.cmd_doctor(_args("doctor", tmp_path), http=TestClient(app))
    out = capsys.readouterr().out
    assert rc == 1
    assert "401" in out and "MNEMOSTACK_API_KEY" in out


def test_doctor_never_prints_the_api_key(tmp_path, service_app, capsys, monkeypatch):
    """Same rule as status, on both renderers: doctor sends the key to the
    service, so a key echoed into a support paste is a real leak path."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app, _store, _emb, keys = service_app
    monkeypatch.setenv("MNEMOSTACK_API_KEY", keys["alpha"])
    _write_config(tmp_path, mode="remote", base_url="http://testserver")
    cli.cmd_doctor(_args("doctor", tmp_path), http=TestClient(app))
    cli.cmd_doctor(_args("doctor", tmp_path, as_json=True), http=TestClient(app))
    assert keys["alpha"] not in capsys.readouterr().out


def test_doctor_reports_an_unreachable_service_with_a_remedy(tmp_path, capsys):
    _write_config(tmp_path, mode="remote", base_url="http://127.0.0.1:1")
    rc = cli.cmd_doctor(_args("doctor", tmp_path))
    out = capsys.readouterr().out
    assert rc == 1
    assert "unreachable" in out and "base_url" in out


def test_doctor_never_writes_to_the_service(tmp_path, service_app, monkeypatch):
    """A diagnostic must not create memories — the probe is read-only."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app, store, _emb, keys = service_app
    monkeypatch.setenv("MNEMOSTACK_API_KEY", keys["alpha"])
    _write_config(tmp_path, mode="remote", base_url="http://testserver")
    before = store.count()
    cli.cmd_doctor(_args("doctor", tmp_path), http=TestClient(app))
    assert store.count() == before


def test_doctor_does_not_create_the_local_collection(tmp_path, capsys, monkeypatch):
    """Local mode: report the collection as absent, never provision it —
    the same line mnemostack's own doctor holds."""
    created: list[str] = []

    class _Store:
        def __init__(self, **kw):
            self.kw = kw

        def ensure_collection(self, **_kw):
            created.append("ensure")

        def collection_exists(self):
            return False

    class _Provider:
        dimension = 3

        def health_check(self):
            return True, "ok"

    monkeypatch.setattr("mnemostack.vector.VectorStore", _Store)
    monkeypatch.setattr("mnemostack.embeddings.get_provider", lambda _n, **_k: _Provider())
    _write_config(tmp_path, mode="local", collection="hermes-memory")
    rc = cli.cmd_doctor(_args("doctor", tmp_path))
    out = capsys.readouterr().out
    assert created == []
    assert "does not exist yet" in out
    assert rc == 0  # an absent collection is not a failure


def test_doctor_fails_when_the_embedding_backend_is_down(tmp_path, capsys, monkeypatch):
    class _Provider:
        dimension = 3

        def health_check(self):
            return False, "connection refused"

    class _Store:
        def __init__(self, **kw):
            pass

        def collection_exists(self):
            return True

        def count(self):
            return 0

    monkeypatch.setattr("mnemostack.vector.VectorStore", _Store)
    monkeypatch.setattr("mnemostack.embeddings.get_provider", lambda _n, **_k: _Provider())
    _write_config(tmp_path, mode="local")
    rc = cli.cmd_doctor(_args("doctor", tmp_path))
    out = capsys.readouterr().out
    assert rc == 1
    assert "connection refused" in out


def test_doctor_reports_discovery(tmp_path, capsys, offline_local_stack):
    """Discovery is asked FUNCTIONALLY (hermes's own scan), so the row is
    accurate on 0.19 — where our entry point is registered but hermes
    reads only directory plugins — and on >= 0.20 alike."""
    _write_config(tmp_path, mode="local")
    cli.cmd_doctor(_args("doctor", tmp_path, as_json=True))
    body = json.loads(capsys.readouterr().out)
    row = next(c for c in body["checks"] if c["name"] in ("discovery", "hermes-agent"))
    assert row["status"] in ("ok", "warn", "fail")
    if row["status"] == "warn":
        assert "entry point registered" in row["detail"] or "not importable" in row["detail"]


def test_main_dispatches_and_requires_a_command(tmp_path, capsys):  # status: no probes
    _write_config(tmp_path, mode="local")
    assert cli.main(["--hermes-home", str(tmp_path), "status"]) == 0
    with pytest.raises(SystemExit):
        cli.main(["--hermes-home", str(tmp_path)])


# --------------------------------------------- round-1 (codex) regressions


def test_shared_options_parse_after_the_subcommand(tmp_path):
    """R1 (codex P2): `--json`/`--hermes-home` were registered only on the
    root parser, so the README's own `doctor --json` exited with
    'unrecognized arguments'."""
    args = cli.parse_args(["doctor", "--json", "--hermes-home", str(tmp_path)])
    assert args.command == "doctor" and args.json is True
    assert args.hermes_home == str(tmp_path)
    # ...and the root position still works, without the subparser's own
    # defaults clobbering it (the argparse trap this fix has to avoid).
    args = cli.parse_args(["--hermes-home", str(tmp_path), "--json", "status"])
    assert args.hermes_home == str(tmp_path) and args.json is True
    args = cli.parse_args(["status"])
    assert args.hermes_home is None and args.json is False


def test_discovery_scan_uses_the_requested_hermes_home(tmp_path, monkeypatch):
    """R1 (codex P2): discovery scans $HERMES_HOME/plugins, so a doctor run
    pointed at one home that scanned ANOTHER could print the opposite
    verdict about the very directory being diagnosed."""
    pytest.importorskip("hermes_constants")
    import hermes_constants

    seen: list[str | None] = []

    def _names():
        seen.append(hermes_constants.get_hermes_home_override())
        return []

    monkeypatch.setattr("plugins.memory.list_memory_provider_names", _names)
    report = cli.Report()
    cli._discovery_check(report, str(tmp_path))
    assert seen == [str(tmp_path)]
    # And the override does not leak past the scan.
    assert hermes_constants.get_hermes_home_override() is None


def test_doctor_fails_on_a_redirecting_service(tmp_path, capsys):
    """R1 (codex P2): neither this client nor the provider's follows
    redirects, so a 302 from a proxy is a service we cannot talk to —
    reporting it 'reachable' would be the worst kind of green."""

    class _Resp:
        status_code = 302
        headers = {"location": "https://elsewhere.invalid/health"}

        def json(self):  # pragma: no cover — must not be reached
            raise AssertionError("a redirect has no health document")

    class _Http:
        def get(self, _path):
            return _Resp()

        def post(self, *_a, **_k):  # pragma: no cover — probe stops earlier
            raise AssertionError("must not probe recall after a failed health")

    _write_config(tmp_path, mode="remote", base_url="http://proxy.invalid")
    rc = cli.cmd_doctor(_args("doctor", tmp_path), http=_Http())
    out = capsys.readouterr().out
    assert rc == 1
    assert "redirected" in out and "elsewhere.invalid" in out


def test_doctor_fails_when_something_else_serves_health(tmp_path, capsys):
    """A 200 from a proxy's login page is not a healthy memory service."""

    class _Resp:
        status_code = 200

        def json(self):
            return {"login": "please"}

    class _Http:
        def get(self, _path):
            return _Resp()

        def post(self, *_a, **_k):  # pragma: no cover
            raise AssertionError("must not probe recall after a failed health")

    _write_config(tmp_path, mode="remote", base_url="http://proxy.invalid")
    assert cli.cmd_doctor(_args("doctor", tmp_path), http=_Http()) == 1
    assert "not with a mnemostack" in capsys.readouterr().out


def test_doctor_reports_a_degraded_service_without_claiming_health(tmp_path, capsys):
    class _Health:
        status_code = 200

        def json(self):
            return {"status": "degraded", "version": "2.2.0", "qdrant": False}

    class _Recall:
        status_code = 200

        def json(self):
            return {"results": [], "degraded": [], "notes": []}

    class _Http:
        def get(self, _path):
            return _Health()

        def post(self, *_a, **_k):
            return _Recall()

    _write_config(tmp_path, mode="remote", base_url="http://svc.invalid")
    # R2 (codex P1): mnemostack reports `degraded` exactly when Qdrant is
    # unreachable, and recall stays fail-soft — so a WARN here let doctor
    # exit 0 while memory was not working.
    rc = cli.cmd_doctor(_args("doctor", tmp_path), http=_Http())
    out = capsys.readouterr().out
    assert rc == 1
    assert "status='degraded'" in out and "qdrant unreachable" in out


def test_doctor_catches_a_collection_built_with_another_model(tmp_path, capsys, monkeypatch):
    """R1 (codex P2): an existing collection whose vector size belongs to a
    different embedding model kills the provider's first session
    (ensure_collection raises). Reporting 'exists, ok' is a false green."""

    class _Info:
        class config:
            class params:
                class vectors:
                    size = 1024

    class _Client:
        def get_collection(self, _name):
            return _Info()

    class _Store:
        client = _Client()

        def __init__(self, **kw):
            pass

        def collection_exists(self):
            return True

        def count(self):  # pragma: no cover — must not be reached
            raise AssertionError("no count after a dimension mismatch")

    class _Provider:
        dimension = 3

        def health_check(self):
            return True, "ok"

    monkeypatch.setattr("mnemostack.vector.VectorStore", _Store)
    monkeypatch.setattr("mnemostack.embeddings.get_provider", lambda _n, **_k: _Provider())
    _write_config(tmp_path, mode="local", collection="wrong-model")
    rc = cli.cmd_doctor(_args("doctor", tmp_path))
    out = capsys.readouterr().out
    assert rc == 1
    assert "1024-dim" in out and "3-dim" in out


def test_a_redirect_location_is_printed_without_its_secrets(tmp_path, capsys):
    """R2 (codex P2): an SSO/proxy redirect routinely carries a token in the
    query string, and these reports get pasted into support threads. The
    operator's question is WHERE base_url sent them — nothing else."""

    class _Resp:
        status_code = 307
        headers = {
            "location": "https://user:pw@sso.invalid/authorize?access_token=SECRET-abc123#frag"
        }

        def json(self):  # pragma: no cover
            raise AssertionError("a redirect has no health document")

    class _Http:
        def get(self, _path):
            return _Resp()

        def post(self, *_a, **_k):  # pragma: no cover
            raise AssertionError("must not probe recall after a failed health")

    _write_config(tmp_path, mode="remote", base_url="http://proxy.invalid")
    assert cli.cmd_doctor(_args("doctor", tmp_path), http=_Http()) == 1
    assert cli.cmd_doctor(_args("doctor", tmp_path, as_json=True), http=_Http()) == 1
    out = capsys.readouterr().out
    assert "SECRET-abc123" not in out
    assert "user:pw" not in out and "#frag" not in out
    assert "sso.invalid" in out  # the authority — the useful half — survives
    assert "authorize" not in out  # the path is not printed at all
    assert "path and query not shown" in out


def test_doctor_reports_a_null_field_instead_of_crashing(tmp_path, capsys):
    """R4 (review agent P1): the shape check tested key PRESENCE, not type.
    A null `results` (a proxy, a non-conformant server, a future version
    emitting null for an empty list) crashed the renderer with a raw
    traceback — the one thing a diagnostic must never do."""

    class _Health:
        status_code = 200

        def json(self):
            return {"status": "ok", "version": "2.2.0"}

    def _http_with(recall_body):
        class _Recall:
            status_code = 200

            def json(self):
                return recall_body

        class _Http:
            def get(self, _path):
                return _Health()

            def post(self, *_a, **_k):
                return _Recall()

        return _Http()

    _write_config(tmp_path, mode="remote", base_url="http://svc.invalid")
    rc = cli.cmd_doctor(_args("doctor", tmp_path), http=_http_with({"results": None}))
    assert rc == 1
    assert "unexpected document" in capsys.readouterr().out
    # A null degradation list is merely "nothing to report", not a crash.
    rc = cli.cmd_doctor(
        _args("doctor", tmp_path),
        http=_http_with({"results": [], "degraded": None, "notes": None}),
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "no faults" in out


def test_hermes_home_override_restores_an_outer_scope(tmp_path):
    """R4 (review agent P2): reset(token) restores the PRIOR value; setting
    the override to None would clear an outer scope instead of restoring
    it — the nesting every hermes-agent call site relies on."""
    pytest.importorskip("hermes_constants")
    import hermes_constants

    outer = hermes_constants.set_hermes_home_override(str(tmp_path / "outer"))
    try:
        with cli._hermes_home(str(tmp_path / "inner")):
            assert hermes_constants.get_hermes_home_override() == str(tmp_path / "inner")
        assert hermes_constants.get_hermes_home_override() == str(tmp_path / "outer")
    finally:
        hermes_constants.reset_hermes_home_override(outer)


def test_the_report_names_the_authority_and_nothing_else():
    """R6 (review agent P1): four rounds found a secret in four positions
    (query, path parameter, host, and — for a Location with no `//` — the
    whole string arriving as the "path"). The contract stopped enumerating
    positions: the report names the destination AUTHORITY, so the only
    text that can ever be printed is one of two schemes, a validated host,
    and a numeric port."""
    leaky = [
        "data:text/plain,SECRETPAYLOAD",
        "javascript:alert(document.cookie)//SECRET",
        "javascript://evil.invalid/SECRET",
        "https:\\\\evil.invalid\\\\SECRET",
        "https://evil.invalid;jsessionid=SECRET/health",
        "https://user:pw@sso.invalid/authorize?access_token=SECRET#SECRET",
        "https://sso.invalid/health;jsessionid=SECRET",
        "/relative/SECRET",
        "file:///etc/SECRET",
        # R7: the SCHEME itself. urlsplit's grammar is unbounded and
        # accepts hyphens, dots and digits — wide enough for a token —
        # and the "cannot follow" branch used to quote it back.
        "SECRET-TOKEN-abc123.leaked://evil.invalid/x",
        "secret" + "0" * 200 + "://evil.invalid/x",
    ]
    for loc in leaky:
        shown = cli._safe_location(loc)
        assert "SECRET" not in shown.upper(), (loc, shown)
        assert "pw" not in shown, (loc, shown)


def test_a_followable_target_still_names_its_host():
    """The trade is the path, not the answer: an operator still learns
    WHERE base_url sent them, which is the question they asked."""
    assert cli._safe_location("https://sso.invalid/authorize?t=x") == (
        "https://sso.invalid (path and query not shown)"
    )
    assert cli._safe_location("https://ok.invalid:8443/x") == (
        "https://ok.invalid:8443 (path and query not shown)"
    )
    # Scheme-relative is a valid, followable form; it inherits our scheme.
    assert cli._safe_location("//sso.invalid/login?token=x") == (
        "//sso.invalid (path and query not shown)"
    )
    # IPv6 keeps its brackets, and a zone id survives.
    assert cli._safe_location("https://[2001:db8::1]:8443/x") == (
        "https://[2001:db8::1]:8443 (path and query not shown)"
    )
    assert "fe80::1" in cli._safe_location("http://[fe80::1%25eth0]/x")
    # Internal DNS names with an underscore are legitimate targets, not
    # attacks — dropping them would make a real host read like one.
    assert cli._safe_location("https://internal_service.corp.example/x") == (
        "https://internal_service.corp.example (path and query not shown)"
    )
    assert cli._safe_location(None) == "unknown"


def test_an_unfollowable_target_is_reported_by_shape():
    """Withheld, but not silently: the operator is told a redirect
    happened and why its target is not printed."""
    assert "cannot follow" in cli._safe_location("javascript://evil.invalid/x")
    # ...described, never quoted: the scheme is attacker-chosen text.
    assert "javascript" not in cli._safe_location("javascript://evil.invalid/x")
    # A port is part of the authority — including the one Python calls falsy.
    assert cli._safe_location("https://ok.invalid:0/x") == (
        "https://ok.invalid:0 (path and query not shown)"
    )
    assert "no host" in cli._safe_location("/relative/path")
    assert "malformed" in cli._safe_location("https://evil.invalid;x=1/health")
    assert "malformed" in cli._safe_location("https://evil.invalid:99999/x")


def _forging_http(health_body, recall_body=None):
    class _Resp:
        def __init__(self, body):
            self.status_code = 200
            self._body = body

        def json(self):
            return self._body

    class _Http:
        def get(self, _path):
            return _Resp(health_body)

        def post(self, *_a, **_k):
            return _Resp(recall_body or {"results": [], "degraded": [], "notes": []})

    return _Http()


def test_a_hostile_server_cannot_forge_a_report_row(tmp_path, capsys):
    """R8 (review agent P1): the plain-text renderer prints one line per
    check, so a newline inside a server-supplied field appends a line that
    reads exactly like a genuine passing check — in the output an operator
    uses to decide whether their deployment is healthy. Same threat model
    the redirect line spent seven rounds closing, one call site over."""
    _write_config(tmp_path, mode="remote", base_url="http://svc.invalid")
    forged = "2.2.0)\n✓ FORGED-ROW  everything is totally fine ("
    cli.cmd_doctor(
        _args("doctor", tmp_path),
        http=_forging_http({"status": "ok", "version": forged}),
    )
    out = capsys.readouterr().out
    assert "FORGED-ROW" in out  # the text is still reported, inline...
    # ...but it cannot BE a row: no line STARTS with a verdict glyph it
    # supplied, and the rendered row count still equals the check count.
    assert not any(ln.startswith("✓ FORGED-ROW") for ln in out.splitlines())
    cli.cmd_doctor(
        _args("doctor", tmp_path, as_json=True),
        http=_forging_http({"status": "ok", "version": forged}),
    )
    checks = json.loads(capsys.readouterr().out)["checks"]
    rows = [ln for ln in out.splitlines() if ln[:1] in ("✓", "!", "✗")]
    assert len(rows) == len(checks)


def test_a_forged_row_cannot_ride_in_on_a_degradation_tag(tmp_path, capsys):
    _write_config(tmp_path, mode="remote", base_url="http://svc.invalid")
    cli.cmd_doctor(
        _args("doctor", tmp_path),
        http=_forging_http(
            {"status": "ok", "version": "2.2.0"},
            {
                "results": [],
                "degraded": ["arm:failed)\n✓ FORGED-VIA-DEGRADED  fine ("],
                "notes": [],
            },
        ),
    )
    out = capsys.readouterr().out
    assert not any(
        line.lstrip().startswith("✓") and "FORGED-VIA-DEGRADED" in line for line in out.splitlines()
    )


def test_an_unbounded_field_cannot_flood_the_report(tmp_path, capsys):
    _write_config(tmp_path, mode="remote", base_url="http://svc.invalid")
    cli.cmd_doctor(
        _args("doctor", tmp_path),
        http=_forging_http({"status": "ok", "version": "v" * 10_000}),
    )
    out = capsys.readouterr().out
    assert max(len(line) for line in out.splitlines()) < 500


def test_no_line_breaking_character_survives_a_row():
    """R9 (codex P1): the sanitizer's class was C0+C1 only, but
    str.splitlines() — and the renderers a pasted report travels through —
    also break on U+2028/U+2029, so a supplied glyph after one still began
    a convincing extra row. Derived from splitlines itself rather than
    hand-listed, so the pin cannot drift from the behaviour it guards."""
    breakers = [c for c in map(chr, range(0x3000)) if len(f"a{c}b".splitlines()) > 1]
    assert " " in breakers and " " in breakers  # the sanity of the probe
    for ch in breakers:
        row = cli._row_text(f"2.2.0){ch}✓ FORGED  fine (")
        assert len(row.splitlines()) == 1, (hex(ord(ch)), row)


def test_no_invisible_control_survives_a_row():
    """R10 (codex P2): the hand-listed bidi set was missing U+061C, just as
    the hand-listed break set had been missing U+2028. It is derived from
    Unicode's own categories now, and pinned the same way — every Cc/Cf/
    Zl/Zp character is stripped, so the next one nobody thought of is
    covered without anyone thinking of it."""
    import unicodedata

    suspects = [
        ch
        for ch in map(chr, range(0x110000))
        if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp")
    ]
    assert "\u061c" in suspects and "\u2028" in suspects and "\n" in suspects
    for ch in suspects:
        assert ch not in cli._row_text(f"ok{ch}text"), hex(ord(ch))
    # Ordinary text of any script is untouched.
    assert cli._row_text(
        "\u0440\u0443\u0441\u0441\u043a\u0438\u0439 ok \U0001f600 \u4e2d\u6587"
    ) == ("\u0440\u0443\u0441\u0441\u043a\u0438\u0439 ok \U0001f600 \u4e2d\u6587")


def test_a_server_cannot_pad_a_diagnosis_out_of_the_report(tmp_path, capsys):
    """R11 (review agent P2): the row is composed as "<our words><their
    value><our words>", and the row cap cuts from the tail — so a padded
    `version` pushed the clause that names the ACTUAL fault ("qdrant
    unreachable") past the cutoff while the padding itself survived. The
    server got to choose which half of the diagnosis the operator sees."""
    _write_config(tmp_path, mode="remote", base_url="http://svc.invalid")
    rc = cli.cmd_doctor(
        _args("doctor", tmp_path),
        http=_forging_http({"status": "degraded", "version": "v" * 280, "qdrant": False}),
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "qdrant unreachable" in out
    # And the padding is bounded rather than spending the whole row.
    assert "v" * 100 not in out


def test_a_padded_degradation_tag_cannot_push_out_the_remedy(tmp_path, capsys):
    _write_config(tmp_path, mode="remote", base_url="http://svc.invalid")
    cli.cmd_doctor(
        _args("doctor", tmp_path),
        http=_forging_http(
            {"status": "ok", "version": "2.2.0"},
            {"results": [], "degraded": ["x" * 500], "notes": []},
        ),
    )
    out = capsys.readouterr().out
    assert "check the service log" in out
    assert "x" * 100 not in out
