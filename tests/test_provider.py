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
    (tmp_path / "mnemostack.json").write_text(
        json.dumps({"mode": "remote", "base_url": "http://svc"})
    )
    fake = FakeClient()
    import hermes_mnemostack.provider as pmod

    monkeypatch.setattr(pmod, "build_client", lambda cfg, scope=None: fake)
    p = MnemostackProvider()
    assert p.is_available() is True  # config file present
    p.initialize("sess-1", hermes_home=str(tmp_path), platform="cli")
    return p, fake


def _wait_threads(p: MnemostackProvider, timeout=3.0):
    deadline = time.monotonic() + timeout
    with p._lock:
        threads = list(p._sync_threads) + list(p._prefetch_threads.values())
    for t in threads:
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


def test_remote_mode_without_base_url_is_unavailable(monkeypatch, tmp_path):
    """R1 (codex P2): hermes must see 'unavailable', not an initialize crash."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("MNEMOSTACK_MODE", raising=False)
    (tmp_path / "mnemostack.json").write_text(json.dumps({"mode": "remote"}))
    p = MnemostackProvider()
    assert p.is_available() is False
    assert "base_url" in p.unavailable_reason()


def test_stale_prefetch_never_overwrites_fresher_result(provider):
    """R1 (both reviewers, P2): an outrun earlier recall finishing AFTER a
    newer one must not inject stale memories into the next turn."""
    import threading as _threading

    p, fake = provider
    slow_release = _threading.Event()

    class _RacingClient:
        def recall(self, query, *, limit=5, filters=None):
            if query == "slow-old-query":
                slow_release.wait(timeout=5.0)
                return [RecallHit(id="old", text="STALE memory", score=0.9)]
            return [RecallHit(id="new", text="FRESH memory", score=0.9)]

        def close(self):
            pass

    p._client = _RacingClient()
    p.queue_prefetch("slow-old-query")  # gen 1, blocked
    p.queue_prefetch("fresh query")     # gen 2, completes first
    _wait_threads(p)
    slow_release.set()  # old recall finishes AFTER the new one
    time.sleep(0.2)
    block = p.prefetch("fresh query")
    assert "FRESH" in block and "STALE" not in block


def test_batch_failure_falls_back_to_per_item(provider):
    """R1 (agent P1 tail): one bad item must not lose the other side."""
    p, fake = provider
    calls = []

    class _PickyClient:
        def remember(self, items):
            calls.append(list(items))
            if len(items) > 1:
                from hermes_mnemostack.client import MnemostackClientError

                raise MnemostackClientError("items[1]: too big", status_code=400)
            return RememberOutcome(stored=1, duplicates=0, failed=0)

        def close(self):
            pass

    p._client = _PickyClient()
    p.sync_turn("good user turn", "bad assistant turn")
    _wait_threads(p)
    # batch attempt + two per-item retries
    assert [len(c) for c in calls] == [2, 1, 1]


def test_shutdown_drains_all_inflight_captures(provider):
    """R1 (both reviewers, P1/P2): shutdown must join EVERY in-flight
    capture, not only the newest thread."""
    import threading as _threading

    p, fake = provider
    gate = _threading.Event()
    done = []

    class _SlowClient:
        def remember(self, items):
            gate.wait(timeout=5.0)
            done.append(len(items))
            return RememberOutcome(stored=len(items), duplicates=0, failed=0)

        def close(self):
            done.append("closed")

    p._client = _SlowClient()
    p.sync_turn("turn one", "reply one")
    p.sync_turn("turn two", "reply two")
    assert len(p._sync_threads) == 2
    gate.set()
    p.shutdown()
    # Both captures completed BEFORE the client closed.
    assert done == [2, 2, "closed"]


def test_sessions_do_not_share_prefetch_or_turn_state(provider):
    """R1 (agent P2 tail): per-session keying — one session's recall block
    and turn numbering must not leak into another."""
    p, fake = provider
    fake.hits = [RecallHit(id="1", text="session A memory", score=0.9)]
    p.queue_prefetch("what did we decide?", session_id="A")
    _wait_threads(p)
    # B sees nothing of A's block.
    assert p.prefetch("what did we decide?", session_id="B") == ""
    assert "session A memory" in p.prefetch("what did we decide?", session_id="A")
    p.sync_turn("u", "a", session_id="A")
    p.sync_turn("u2", "a2", session_id="B")
    _wait_threads(p)
    offsets = {batch[0].source: batch[0].offset for batch in fake.remembered}
    assert offsets["hermes/cli/A"] == 0 and offsets["hermes/cli/B"] == 0


def test_concurrent_session_prefetch_joins_own_thread(provider):
    """R2 (both reviewers): session A's prefetch must join A's OWN thread —
    joining B's (fast) thread returned A's block as empty while A's recall
    was still in flight."""
    import threading as _threading

    p, fake = provider
    a_release = _threading.Event()

    class _TwoSessionClient:
        def recall(self, query, *, limit=5, filters=None):
            if "session-a" in query:
                a_release.wait(timeout=5.0)
                return [RecallHit(id="a", text="A block", score=0.9)]
            return [RecallHit(id="b", text="B block", score=0.9)]

        def close(self):
            pass

    p._client = _TwoSessionClient()
    p.queue_prefetch("session-a slow question", session_id="A")
    p.queue_prefetch("session-b fast question", session_id="B")
    with p._lock:
        b_thread = p._prefetch_threads["B"]
    b_thread.join(timeout=3.0)  # B done; A still blocked
    a_release.set()  # A releases; its prefetch() below must join A's thread
    assert "A block" in p.prefetch("session-a slow question", session_id="A")
    assert "B block" in p.prefetch("session-b fast question", session_id="B")


def test_scope_tenant_encoding_is_injective():
    """R2 (both reviewers): crafted identity strings with '|'/'=' must not
    collide two scopes into one tenant."""
    from hermes_mnemostack.client import LocalClient

    t1 = LocalClient._scope_tenant(
        {"hermes_profile": "a", "hermes_user": "b|hermes_user=c"}
    )
    t2 = LocalClient._scope_tenant(
        {"hermes_profile": "a|hermes_user=b", "hermes_user": "c"}
    )
    assert t1 != t2
    # Canonical: order-independent.
    assert LocalClient._scope_tenant({"x": "1", "y": "2"}) == LocalClient._scope_tenant(
        {"y": "2", "x": "1"}
    )


def test_sync_after_shutdown_is_a_noop(provider):
    """R2 (agent P3): a sync_turn racing shutdown must not spawn a capture
    against a closing client."""
    p, fake = provider
    p.shutdown()
    p.sync_turn("late turn", "late reply")
    assert p._sync_threads == [] and fake.remembered == []


def test_transport_failure_skips_pointless_per_item_retry(provider):
    """R2 (agent P3): a dead service fails per item exactly like the batch —
    retry only on validation-class rejections."""
    p, fake = provider
    calls = []

    class _DeadClient:
        def remember(self, items):
            calls.append(len(items))
            from hermes_mnemostack.client import MnemostackClientError

            raise MnemostackClientError("unreachable")  # no status_code

        def close(self):
            pass

    p._client = _DeadClient()
    p.sync_turn("u", "a")
    _wait_threads(p)
    assert calls == [2]  # one batch attempt, no per-item storm
