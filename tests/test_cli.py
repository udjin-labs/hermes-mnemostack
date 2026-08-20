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
    return cli.build_parser().parse_args([*argv, command])


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
