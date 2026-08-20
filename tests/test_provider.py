"""Provider ABC wiring over an injected fake client: prefetch/capture/
status/session lifecycle — no store, no network."""

from __future__ import annotations

import json
import time

import pytest

from hermes_mnemostack.client import MemoryItem, RecallHit, RememberOutcome
from hermes_mnemostack.provider import MnemostackProvider


class FakeClient:
    def __init__(self, hits=None):
        self.hits = hits or []
        self.remembered: list[list[MemoryItem]] = []
        self.closed = False

    def recall(self, query, *, limit=5, filters=None):
        return self.hits

    def remember(self, items):
        self.remembered.append(list(items))
        return RememberOutcome(stored=len(items), duplicates=0, failed=0)

    def invalidate(self, ids):
        return len(ids)

    def forget(self, ids):
        return len(ids)

    def close(self):
        self.closed = True


@pytest.fixture()
def provider(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("MNEMOSTACK_MODE", raising=False)
    (tmp_path / "mnemostack.json").write_text(json.dumps({"mode": "remote"}))
    fake = FakeClient()
    import hermes_mnemostack.provider as pmod

    monkeypatch.setattr(pmod, "build_client", lambda cfg, scope=None: fake)
    p = MnemostackProvider()
    assert p.is_available() is True  # config file present
    p.initialize("sess-1", hermes_home=str(tmp_path), platform="cli")
    return p, fake


def _wait_threads(p: MnemostackProvider, timeout=3.0):
    deadline = time.monotonic() + timeout
    for attr in ("_prefetch_thread", "_sync_thread"):
        t = getattr(p, attr)
        if t is not None:
            t.join(timeout=max(0.0, deadline - time.monotonic()))


def test_unconfigured_provider_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("MNEMOSTACK_MODE", raising=False)
    p = MnemostackProvider()
    assert p.is_available() is False
    assert "hermes memory setup" in p.unavailable_reason()


def test_prefetch_cycle_and_recall_status(provider):
    p, fake = provider
    fake.hits = [
        RecallHit(id="1", text="the deploy window is Friday", score=0.9),
        RecallHit(id="2", text="user prefers dark mode", score=0.8),
    ]
    p.queue_prefetch("what is our deploy window?")
    _wait_threads(p)
    block = p.prefetch("what is our deploy window?")
    assert "Friday" in block and "dark mode" in block
    status = p.recall_status()
    assert status is not None and status.count == 2
    assert status.provider_label == "mnemostack"
    # The cache is consumed: a second prefetch with no new queue is empty.
    assert p.prefetch("again") == ""
    assert p.recall_status() is None


def test_trivial_prompts_skip_prefetch(provider):
    p, fake = provider
    fake.hits = [RecallHit(id="1", text="anything", score=0.9)]
    p.queue_prefetch("ok")
    _wait_threads(p)
    assert p.prefetch("ok") == ""


def test_sync_turn_captures_verbatim_with_deterministic_ids(provider):
    p, fake = provider
    p.sync_turn("we deploy fridays", "noted — friday it is", session_id="sess-1")
    _wait_threads(p)
    (batch,) = fake.remembered
    assert [i.metadata["hermes_role"] for i in batch] == ["user", "assistant"]
    assert batch[0].source == "hermes/cli/sess-1"
    assert (batch[0].offset, batch[1].offset) == (0, 1)
    p.sync_turn("second turn", "reply", session_id="sess-1")
    _wait_threads(p)
    assert fake.remembered[1][0].offset == 2  # turn counter advances


def test_capture_gates(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "mnemostack.json").write_text(json.dumps({"capture": False}))
    fake = FakeClient()
    import hermes_mnemostack.provider as pmod

    monkeypatch.setattr(pmod, "build_client", lambda cfg, scope=None: fake)
    p = MnemostackProvider()
    p.initialize("s", hermes_home=str(tmp_path), platform="cli")
    p.sync_turn("u", "a")
    _wait_threads(p)
    assert fake.remembered == []  # capture disabled

    # Non-primary contexts (cron/subagent) never write.
    (tmp_path / "mnemostack.json").write_text("{}")
    fake2 = FakeClient()
    monkeypatch.setattr(pmod, "build_client", lambda cfg, scope=None: fake2)
    p2 = MnemostackProvider()
    p2.initialize("s", hermes_home=str(tmp_path), platform="cli", agent_context="cron")
    p2.sync_turn("u", "a")
    _wait_threads(p2)
    assert fake2.remembered == []


def test_session_switch_reset(provider):
    p, fake = provider
    p.sync_turn("u", "a")
    _wait_threads(p)
    p.on_session_switch("sess-2", reset=True)
    p.sync_turn("u2", "a2")
    _wait_threads(p)
    assert fake.remembered[-1][0].offset == 0  # counter reset
    assert fake.remembered[-1][0].source.endswith("/sess-2")


def test_scope_flows_from_identity_kwargs(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "mnemostack.json").write_text("{}")
    captured = {}
    import hermes_mnemostack.provider as pmod

    def _build(cfg, scope=None):
        captured["scope"] = scope
        return FakeClient()

    monkeypatch.setattr(pmod, "build_client", _build)
    p = MnemostackProvider()
    p.initialize(
        "s", hermes_home=str(tmp_path), platform="cli",
        agent_identity="coder", user_id="u42",
    )
    assert captured["scope"] == {"hermes_profile": "coder", "hermes_user": "u42"}


def test_shutdown_closes_client(provider):
    p, fake = provider
    p.shutdown()
    assert fake.closed is True
