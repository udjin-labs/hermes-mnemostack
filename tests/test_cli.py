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


def test_doctor_reports_a_rejected_key_as_a_key_problem(
    tmp_path, service_app, capsys, monkeypatch
):
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
    cli.cmd_doctor(_args("doctor", tmp_path), http=_Http())
    out = capsys.readouterr().out
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
