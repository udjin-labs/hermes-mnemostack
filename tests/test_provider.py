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
        b_thread = p._prefetch_threads.get("B")
    if b_thread is not None:
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


def test_session_churn_never_evicts_inflight_prefetch(provider):
    """R3 (agent P2): filler-session churn past the LRU bound must not drop
    an active session's in-flight recall on the floor."""
    import threading as _threading

    p, fake = provider
    release = _threading.Event()

    class _MixedClient:
        def recall(self, query, *, limit=5, filters=None):
            if "active" in query:
                release.wait(timeout=5.0)
                return [RecallHit(id="a", text="active block", score=0.9)]
            return [RecallHit(id="f", text="filler block", score=0.5)]

        def close(self):
            pass

    p._client = _MixedClient()
    p._MAX_SESSION_STATES = 2  # tighten the bound for the test
    p.queue_prefetch("active slow question", session_id="active")
    for i in range(5):
        p.queue_prefetch(f"filler question {i}", session_id=f"filler-{i}")
    release.set()
    assert "active block" in p.prefetch("active slow question", session_id="active")


def test_unconsumed_block_protection_is_load_bearing(provider):
    """R5 (agent): fillers must CACHE blocks so _prefetched actually grows
    past the cap — otherwise the block-protection branch is never taken
    and the pin is vacuous."""
    p, fake = provider
    fake.hits = [RecallHit(id="v", text="victim cached block", score=0.9)]
    p._MAX_SESSION_STATES = 2
    p.queue_prefetch("victim question", session_id="victim")
    _wait_threads(p)  # completed, cached, NOT consumed
    fake.hits = [RecallHit(id="f", text="filler block", score=0.5)]
    # Three fillers: cache grows past the soft cap (2) but stays within
    # the 2x hard-cap tolerance, where block protection must hold.
    for i in range(3):
        p.queue_prefetch(f"filler question {i}", session_id=f"filler-{i}")
        _wait_threads(p)
    with p._lock:
        assert len(p._prefetched) > 2  # cache genuinely grew past the cap
    assert "victim cached block" in p.prefetch("victim question", session_id="victim")


def test_generations_survive_reset_and_eviction(provider):
    """R3 (codex P2): a reset (or eviction) must not recycle generation
    numbers — an old in-flight worker finishing last would pass the gate
    and inject stale memories."""
    import threading as _threading

    p, fake = provider
    release = _threading.Event()

    class _SlowThenFast:
        def recall(self, query, *, limit=5, filters=None):
            if "old" in query:
                release.wait(timeout=5.0)
                return [RecallHit(id="o", text="STALE from before reset", score=0.9)]
            return [RecallHit(id="n", text="FRESH after reset", score=0.9)]

        def close(self):
            pass

    p._client = _SlowThenFast()
    p.queue_prefetch("old question", session_id="S")
    p.on_session_switch("S", reset=True)  # in-flight worker must be orphaned
    p.queue_prefetch("new question", session_id="S")
    _wait_threads(p)
    release.set()
    time.sleep(0.2)
    block = p.prefetch("new question", session_id="S")
    assert "FRESH" in block and "STALE" not in block


def test_5xx_skips_per_item_retry(provider):
    """R3 (both reviewers): an overloaded service answering 503 fails per
    item exactly like the batch — no retry storm."""
    p, fake = provider
    calls = []

    class _OutageClient:
        def remember(self, items):
            calls.append(len(items))
            from hermes_mnemostack.client import MnemostackClientError

            raise MnemostackClientError("boom", status_code=503)

        def close(self):
            pass

    p._client = _OutageClient()
    p.sync_turn("u", "a")
    _wait_threads(p)
    assert calls == [2]


def test_unconsumed_block_survives_session_churn(provider):
    """R4 (agent P1): recall usually finishes BEFORE the host consumes it —
    a completed-but-undelivered block must survive filler churn just like
    an in-flight thread."""
    p, fake = provider
    fake.hits = [RecallHit(id="v", text="victim cached block", score=0.9)]
    p._MAX_SESSION_STATES = 2
    p.queue_prefetch("victim question", session_id="victim")
    _wait_threads(p)  # completed, cached, NOT consumed
    fake.hits = []
    for i in range(5):
        p.queue_prefetch(f"filler question {i}", session_id=f"filler-{i}")
    _wait_threads(p)
    assert "victim cached block" in p.prefetch("victim question", session_id="victim")


def test_completed_workers_self_clean_thread_entries(provider):
    """R4 (codex P2): a drained burst must shrink back — completed workers
    drop their own thread references and re-prune."""
    p, fake = provider
    p.queue_prefetch("some question", session_id="solo")
    _wait_threads(p)
    with p._lock:
        assert "solo" not in p._prefetch_threads  # self-cleaned on completion


def test_fresh_queue_survives_all_protected_prune(provider):
    """R4 (codex P2): with every slot protected, the freshly queued key must
    not become its own victim (thread not registered yet at prune time)."""
    import threading as _threading

    p, fake = provider
    release = _threading.Event()

    class _Slow:
        def recall(self, query, *, limit=5, filters=None):
            release.wait(timeout=5.0)
            return [RecallHit(id="x", text=f"answer to {query}", score=0.9)]

        def close(self):
            pass

    p._client = _Slow()
    p._MAX_SESSION_STATES = 1
    p.queue_prefetch("first question", session_id="one")   # live, protected
    p.queue_prefetch("second question", session_id="two")  # must survive queueing
    release.set()
    assert "second question" in p.prefetch("second question", session_id="two")


def test_empty_followup_clears_stale_block(provider):
    """R5 (both reviewers, P1): a later query that legitimately finds
    nothing must not leave the prior turn's block to be injected as if
    fresh."""
    p, fake = provider
    fake.hits = [RecallHit(id="1", text="SECRET turn-1 memory", score=0.9)]
    p.queue_prefetch("first question", session_id="s")
    _wait_threads(p)
    fake.hits = []  # second query finds nothing
    p.queue_prefetch("unrelated second question", session_id="s")
    _wait_threads(p)
    assert p.prefetch("unrelated second question", session_id="s") == ""
    assert p.recall_status() is None


def test_eviction_removes_a_session_from_every_dict(provider):
    """R5 (codex P2): the victim leaves ALL state dicts at once —
    per-dict eviction desynced turn counters from blocks."""
    p, fake = provider
    p._MAX_SESSION_STATES = 1
    for i in range(4):
        p.sync_turn(f"u{i}", f"a{i}", session_id=f"sess-{i}")
    _wait_threads(p)
    with p._lock:
        keys = set(p._turn_index)
        assert len(keys) <= 1
        for d in (p._prefetched, p._prefetch_gen, p._prefetch_threads):
            assert set(d) <= keys | set()
