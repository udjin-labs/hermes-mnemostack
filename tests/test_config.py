"""Config: defaults <- env <- $HERMES_HOME/mnemostack.json, secrets env-only."""

from __future__ import annotations

import json

import pytest

from hermes_mnemostack.config import is_configured, load_config, save_config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for var in (
        "MNEMOSTACK_MODE",
        "MNEMOSTACK_BASE_URL",
        "MNEMOSTACK_API_KEY",
        "MNEMOSTACK_TIMEOUT",
        "MNEMOSTACK_RECALL_LIMIT",
        "MNEMOSTACK_CAPTURE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def test_defaults(tmp_path):
    cfg = load_config(str(tmp_path))
    assert cfg["mode"] == "remote"
    assert cfg["capture"] is True
    assert cfg["recall_limit"] == 5
    assert cfg["api_key"] == ""


def test_env_then_file_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMOSTACK_MODE", "local")
    monkeypatch.setenv("MNEMOSTACK_RECALL_LIMIT", "3")
    (tmp_path / "mnemostack.json").write_text(json.dumps({"recall_limit": 7}))
    cfg = load_config(str(tmp_path))
    assert cfg["mode"] == "local"  # env applies
    assert cfg["recall_limit"] == 7  # file wins over env


def test_api_key_is_env_only(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMOSTACK_API_KEY", "sk-test")
    cfg = load_config(str(tmp_path))
    assert cfg["api_key"] == "sk-test"
    with pytest.raises(ValueError, match="unknown config key"):
        save_config({"api_key": "sk-leak"}, str(tmp_path))


def test_unknown_json_key_fails_loud(tmp_path):
    (tmp_path / "mnemostack.json").write_text(json.dumps({"recal_limit": 7}))
    with pytest.raises(ValueError, match="unknown key"):
        load_config(str(tmp_path))


def test_type_coercion_and_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMOSTACK_CAPTURE", "false")
    cfg = load_config(str(tmp_path))
    assert cfg["capture"] is False
    (tmp_path / "mnemostack.json").write_text(json.dumps({"recall_limit": 0}))
    with pytest.raises(ValueError, match="positive"):
        load_config(str(tmp_path))
    (tmp_path / "mnemostack.json").write_text(json.dumps({"mode": "sideways"}))
    with pytest.raises(ValueError, match="mode"):
        load_config(str(tmp_path))


def test_is_configured_gates(monkeypatch, tmp_path):
    assert is_configured(str(tmp_path)) is False  # pure defaults ≠ configured
    monkeypatch.setenv("MNEMOSTACK_MODE", "remote")
    assert is_configured(str(tmp_path)) is True
    monkeypatch.delenv("MNEMOSTACK_MODE")
    (tmp_path / "mnemostack.json").write_text("{}")
    assert is_configured(str(tmp_path)) is True


def test_save_config_round_trip(tmp_path):
    save_config({"mode": "local", "collection": "hm"}, str(tmp_path))
    cfg = load_config(str(tmp_path))
    assert cfg["mode"] == "local" and cfg["collection"] == "hm"
