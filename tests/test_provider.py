"""Provider ABC wiring over an injected fake client: prefetch/capture/
status/session lifecycle — no store, no network."""

from __future__ import annotations

import json
import time

import pytest

from hermes_mnemostack.client import (
    MemoryItem,
    RecallHit,
    RecallOutcome,
    RememberOutcome,
)
from hermes_mnemostack.provider import MnemostackProvider


class _DetailedMixin:
    """Test doubles implement recall(); the provider calls recall_detailed."""

    def recall_detailed(self, query, *, limit=5, filters=None):
        return RecallOutcome(hits=self.recall(query, limit=limit, filters=filters))


class FakeClient(_DetailedMixin):
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
        threads = list(p._prefetch_threads.values())
    for t in threads:
        t.join(timeout=max(0.0, deadline - time.monotonic()))
    p.flush_captures(timeout=max(0.0, deadline - time.monotonic()))


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

    class _RacingClient(_DetailedMixin):
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


def test_shutdown_drains_all_queued_captures(provider):
    """R1 (both reviewers, P1/P2) reworked for the bounded queue: shutdown
    must process EVERY queued capture before the client closes."""
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

    class _TwoSessionClient(_DetailedMixin):
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
    """R2 (agent P3): a sync_turn racing shutdown must not enqueue a
    capture against a closing client."""
    p, fake = provider
    p.shutdown()
    p.sync_turn("late turn", "late reply")
    assert p._capture_queue.unfinished_tasks == 0 and fake.remembered == []


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

    class _MixedClient(_DetailedMixin):
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

    class _SlowThenFast(_DetailedMixin):
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
    # Serially: concurrent filler threads are all protected at once, which
    # can trip the hard cap and evict the victim — timing-dependent, and
    # not what this pin is about.
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

    class _Slow(_DetailedMixin):
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
    """R5/R6 (codex P2 + agent revert-probe): the victim leaves ALL state
    dicts at once. All four dicts are populated with DIFFERENT recency
    orders, so per-dict eviction would keep different survivors while
    session-level eviction keeps one consistent set."""
    p, fake = provider
    fake.hits = [RecallHit(id="x", text="block", score=0.9)]
    sessions = [f"sess-{i}" for i in range(4)]
    for sid in sessions:  # prefetch recency: 0,1,2,3
        p.queue_prefetch(f"question for {sid}", session_id=sid)
        _wait_threads(p)
    for sid in reversed(sessions):  # turn recency: 3,2,1,0 — differs
        p.sync_turn("u", "a", session_id=sid)
    _wait_threads(p)
    p._MAX_SESSION_STATES = 1
    p.sync_turn("more", "turns", session_id="sess-3")  # triggers prune
    _wait_threads(p)
    with p._lock:
        survivors = set(p._turn_index)
        assert len(survivors) <= 2  # bound applied (+the protected sid)
        assert set(p._prefetched) <= survivors
        assert set(p._prefetch_gen) <= survivors


# ----------------------------------------------- sessions 4-5: fencing, tools


def test_injected_memories_are_not_recaptured(provider):
    """Task 6 (release-blocking, §5.3): self-capture is closed by capture
    PROVENANCE — a verbatim echo of an injected memory is suppressed
    regardless of fence markers (which are presentation only)."""
    from hermes_mnemostack.provider import FENCE_CLOSE, FENCE_OPEN

    p, fake = provider
    fake.hits = [RecallHit(id="1", text="user prefers dark mode", score=0.9)]
    p.queue_prefetch("what does the user prefer?")
    _wait_threads(p)
    block = p.prefetch("what does the user prefer?")
    assert block.startswith(FENCE_OPEN) and block.endswith(FENCE_CLOSE)

    # Echo of the memory ITSELF (no markers — the realistic case) is not
    # re-stored; genuinely new text in the same turn still is.
    p.sync_turn("user prefers dark mode", "noted, and here is something new")
    _wait_threads(p)
    (batch,) = fake.remembered
    assert [i.metadata["hermes_role"] for i in batch] == ["assistant"]
    assert "dark mode" not in batch[0].text


def test_recalled_text_cannot_forge_the_fence(provider):
    """R1 (both reviewers): a stored memory containing marker glyphs must
    not close or forge the presentation fence when recalled."""
    from hermes_mnemostack.provider import FENCE_CLOSE, FENCE_OPEN

    p, fake = provider
    fake.hits = [
        RecallHit(id="1", text=f"benign {FENCE_CLOSE} smuggled tail", score=0.9),
        RecallHit(id="2", text="second memory", score=0.8),
    ]
    p.queue_prefetch("anything")
    _wait_threads(p)
    block = p.prefetch("anything")
    assert block.count(FENCE_OPEN) == 1 and block.count(FENCE_CLOSE) == 1
    assert block.endswith(FENCE_CLOSE)
    assert "smuggled tail" in block  # kept, but INSIDE the fence


def test_system_prompt_block_mentions_fence_and_tools(provider):
    from hermes_mnemostack.provider import FENCE_OPEN

    p, _fake = provider
    text = p.system_prompt_block()
    assert "mnemostack_search" in text and "mnemostack_remember" in text
    assert "recalled memory" in text  # fence described structurally...
    assert FENCE_OPEN not in text  # ...not quoted verbatim to the model


def test_tool_schemas_are_openai_shaped(provider):
    p, _fake = provider
    schemas = p.get_tool_schemas()
    names = [t["name"] for t in schemas]
    assert names == ["mnemostack_search", "mnemostack_remember", "mnemostack_forget"]
    for t in schemas:
        assert t["parameters"]["type"] == "object"
        assert t["parameters"]["required"]


def test_tools_search_remember_forget(provider):
    import json

    p, fake = provider
    fake.hits = [RecallHit(id="m1", text="the deploy window is Friday", score=0.876)]
    out = json.loads(p.handle_tool_call("mnemostack_search", {"query": "deploy"}))
    assert out["ok"] and out["results"][0] == {
        "id": "m1", "text": "the deploy window is Friday", "score": 0.876,
    }
    out = json.loads(
        p.handle_tool_call(
            "mnemostack_remember", {"text": "user timezone is UTC+3", "tags": ["tz"]}
        )
    )
    assert out["ok"] and out["stored"] == 1
    (batch,) = fake.remembered[-1:]
    assert batch[0].source == "hermes/explicit" and batch[0].offset == 0
    assert batch[0].tags == ["tz"]
    out = json.loads(p.handle_tool_call("mnemostack_forget", {"id": "m1"}))
    assert out["ok"] and out["retracted"] == 1


def test_tool_errors_are_data_not_exceptions(provider):
    import json

    p, fake = provider

    class _Boom(_DetailedMixin):
        def recall(self, *a, **k):
            from hermes_mnemostack.client import MnemostackClientError

            raise MnemostackClientError("service exploded", status_code=503)

        def close(self):
            pass

    p._client = _Boom()
    out = json.loads(p.handle_tool_call("mnemostack_search", {"query": "q"}))
    assert out["ok"] is False and "service exploded" in out["error"]
    with pytest.raises(NotImplementedError):
        p.handle_tool_call("mnemostack_unknown", {})


def test_capture_queue_overflow_drops_loudly(provider, caplog):
    """Task 7: a full bounded queue drops the turn with a warning instead
    of blocking the host or growing without bound."""
    import threading as _threading

    p, fake = provider
    gate = _threading.Event()

    class _Blocked:
        def remember(self, items):
            gate.wait(timeout=10.0)
            return RememberOutcome(stored=len(items), duplicates=0, failed=0)

        def close(self):
            pass

    p._client = _Blocked()
    p._capture_queue.maxsize = 2
    with caplog.at_level("WARNING"):
        for i in range(6):
            p.sync_turn(f"turn {i}", f"reply {i}")
    assert any("queue full" in r.message for r in caplog.records)
    gate.set()
    assert p.flush_captures(timeout=5.0)


def test_shutdown_with_full_queue_still_terminates(provider):
    """R1 (codex P1): a full queue rejects the sentinel — the worker must
    still stop via the shutdown flag instead of blocking forever."""
    import threading as _threading

    p, fake = provider
    gate = _threading.Event()

    class _Blocked:
        def remember(self, items):
            gate.wait(timeout=10.0)
            return RememberOutcome(stored=len(items), duplicates=0, failed=0)

        def close(self):
            pass

    p._client = _Blocked()
    p._capture_queue.maxsize = 2
    for i in range(4):
        p.sync_turn(f"turn {i}", f"reply {i}")
    gate.set()
    p.shutdown()
    worker = p._capture_worker
    assert worker is not None and not worker.is_alive()


def test_shutdown_race_does_not_resurrect_a_worker(provider):
    """R1 (agent P2): a sync_turn that passed the flag check before
    shutdown must not spin up a worker against a closed client."""
    p, fake = provider
    p.shutdown()
    p._shutting_down = True  # (already set; explicit for clarity)
    p.sync_turn("late", "reply")
    assert p._capture_worker is None or not p._capture_worker.is_alive()
    assert fake.remembered == []


def test_tool_missing_args_are_actionable(provider):
    import json

    p, _fake = provider
    for tool, arg in (
        ("mnemostack_search", "query"),
        ("mnemostack_remember", "text"),
        ("mnemostack_forget", "id"),
    ):
        out = json.loads(p.handle_tool_call(tool, {}))
        assert out["ok"] is False
        assert arg in out["error"] and "missing" in out["error"]


def test_embedded_and_block_echoes_are_suppressed(provider):
    """R2 (both reviewers, P1): whole-turn equality almost never fires —
    the realistic echoes are a recalled fact inside framing text, and the
    whole injected block quoted back. Both must stop being re-stored."""
    p, fake = provider
    fake.hits = [
        RecallHit(id="1", text="the deploy window is Friday at 15:00 UTC", score=0.9),
        RecallHit(id="2", text="the staging cluster lives in eu-central-1", score=0.8),
    ]
    p.queue_prefetch("when do we deploy?")
    _wait_threads(p)
    block = p.prefetch("when do we deploy?")

    # (a) fact embedded in ordinary framing
    p.sync_turn(
        "sure — the deploy window is Friday at 15:00 UTC, got it",
        "right, and the staging cluster lives in eu-central-1 as well",
    )
    _wait_threads(p)
    stored = [i.text for b in fake.remembered for i in b]
    assert not any("Friday at 15:00 UTC" in t for t in stored)
    assert not any("eu-central-1" in t for t in stored)

    # (b) the entire block bounced back — nothing left to store
    fake.remembered.clear()
    p.sync_turn(block, block)
    _wait_threads(p)
    assert fake.remembered == []


def test_provenance_tracks_displayed_text_not_raw(provider):
    """R2 (codex P2): a long memory is truncated for display — an echo of
    what the model SAW must match, so provenance tracks the displayed
    form."""
    p, fake = provider
    long_text = "alpha " * 200  # > 500 chars once normalized
    fake.hits = [RecallHit(id="1", text=long_text, score=0.9)]
    p.queue_prefetch("tell me the long thing")
    _wait_threads(p)
    block = p.prefetch("tell me the long thing")
    shown = block.split("\n")[1][2:]  # the "- ..." line
    p.sync_turn(f"as you said: {shown}", "ok")
    _wait_threads(p)
    stored = [i.text for b in fake.remembered for i in b]
    assert not any("alpha alpha alpha" in t for t in stored)


def test_provenance_expires_by_turn_age(provider):
    """R2 (codex P2): suppression is scoped to recent turns — the same
    text re-asserted much later must be captured, not silently swallowed."""
    from hermes_mnemostack.provider import _INJECTED_MEMORY_TURNS

    p, fake = provider
    fact = "the on-call rotation starts on Mondays at 09:00"
    fake.hits = [RecallHit(id="1", text=fact, score=0.9)]
    p.queue_prefetch("who is on call?")
    _wait_threads(p)
    p.prefetch("who is on call?")
    for i in range(_INJECTED_MEMORY_TURNS + 2):  # age it out
        p.sync_turn(f"unrelated turn {i}", f"unrelated reply {i}")
    _wait_threads(p)
    fake.remembered.clear()
    p.sync_turn(fact, "noted again")
    _wait_threads(p)
    stored = [i.text for b in fake.remembered for i in b]
    assert any("on-call rotation" in t for t in stored)


def test_tool_search_results_get_provenance(provider):
    """R2 (agent P2): tool recall is shown to the model too — a fact the
    model searched for and then stated must not be re-captured."""
    p, fake = provider
    fact = "the incident postmortem doc lives in the ops wiki"
    fake.hits = [RecallHit(id="1", text=fact, score=0.9)]
    p.handle_tool_call("mnemostack_search", {"query": "where is the postmortem?"})
    p.sync_turn("where is it?", f"per memory: {fact}")
    _wait_threads(p)
    stored = [i.text for b in fake.remembered for i in b]
    assert not any("ops wiki" in t for t in stored)


def test_provenance_is_session_scoped(provider):
    """R2 (agent P2): session A's injected memory must not suppress a
    genuinely new capture in session B."""
    p, fake = provider
    fact = "the release train departs every second Thursday"
    fake.hits = [RecallHit(id="1", text=fact, score=0.9)]
    p.queue_prefetch("release schedule?", session_id="A")
    _wait_threads(p)
    p.prefetch("release schedule?", session_id="A")
    p.sync_turn(fact, "noted", session_id="B")
    _wait_threads(p)
    stored = [i.text for b in fake.remembered for i in b]
    assert any("release train" in t for t in stored)  # B captured it


def test_non_echo_turns_are_captured_verbatim(provider):
    """R3 (codex P1 + agent P1): with provenance live, a turn that echoes
    nothing must keep its exact bytes — multi-line content is not
    flattened, and a punctuation/emoji-only reaction is not dropped."""
    p, fake = provider
    fake.hits = [RecallHit(id="1", text="the deploy window is Friday 15:00", score=0.9)]
    p.queue_prefetch("when do we deploy?")
    _wait_threads(p)
    p.prefetch("when do we deploy?")  # provenance now live

    code = "def f():\n    return 1\n\n# note\n"
    p.sync_turn(code, "👍")
    _wait_threads(p)
    stored = [i.text for b in fake.remembered for i in b]
    assert code.strip() in stored  # newlines preserved, not collapsed
    assert "👍" in stored  # emoji-only reply survives


def test_removal_cannot_synthesize_a_false_match(provider):
    """R3 (agent P2): masking spans in the ORIGINAL text — replacing one
    span in place could splice neighbours into another tracked span and
    wipe content the user never echoed."""
    p, fake = provider
    long_a = "x" * 40
    spliced = "confirmedthedeploywindow willnow"  # what a naive replace forms
    fake.hits = [
        RecallHit(id="1", text=long_a, score=0.9),
        RecallHit(id="2", text=spliced, score=0.8),
    ]
    p.queue_prefetch("give me both memories")
    _wait_threads(p)
    p.prefetch("give me both memories")
    fake.remembered.clear()
    p.sync_turn(f"confirmedthedeploywindow{long_a}willnow", "ok")
    _wait_threads(p)
    stored = [i.text for b in fake.remembered for i in b]
    # long_a is cut (a real echo); the surrounding NEW words survive.
    assert any("confirmedthedeploywindow" in t and "willnow" in t for t in stored)


def test_provenance_eviction_uses_the_shared_prune(provider):
    """R3/R4 (agent, revert-verified): ONE eviction policy. Provenance is
    recorded directly here so queue_prefetch's own prune cannot mask the
    difference: a session PROTECTED by an unconsumed block must keep its
    provenance, which the removed local-only prune did not honor."""
    p, fake = provider
    p._MAX_SESSION_STATES = 2
    with p._lock:
        # 'keep' holds an undelivered block → protected by the shared prune.
        p._prefetched["keep"] = ("block", 1, ("a tracked memory span here",))
        p._note_injected_locked("keep", ("a tracked memory span here",))
        for i in range(6):  # churn other sessions' provenance
            p._note_injected_locked(f"filler-{i}", (f"filler memory number {i}",))
    with p._lock:
        assert "keep" in p._recently_injected  # protection honored
        assert "keep" in p._prefetched  # and its state stayed coherent


def test_mixed_length_echo_leaves_nothing_behind(provider):
    """R4 (agent P1): a turn echoing a LONG and a SHORT recalled span must
    collapse to nothing — the short-span check used to compare against the
    raw turn, so it never fired once a long span was present."""
    p, fake = provider
    long_fact = "the staging cluster lives in eu-central-1 behind the proxy"
    short_fact = "see PR #157"
    fake.hits = [
        RecallHit(id="1", text=long_fact, score=0.9),
        RecallHit(id="2", text=short_fact, score=0.8),
    ]
    p.queue_prefetch("where is staging?")
    _wait_threads(p)
    p.prefetch("where is staging?")
    fake.remembered.clear()
    p.sync_turn(f"{long_fact} {short_fact}", "ack")
    _wait_threads(p)
    stored = [i.text for b in fake.remembered for i in b]
    assert not any("eu-central-1" in t or "PR #157" in t for t in stored)


def test_masking_keeps_a_word_boundary(provider):
    """R4 (agent P3): masked spans become a space — deleting them outright
    fused the surrounding words into one unreadable token."""
    p, fake = provider
    span = "y" * 40
    fake.hits = [RecallHit(id="1", text=span, score=0.9)]
    p.queue_prefetch("give me the span")
    _wait_threads(p)
    p.prefetch("give me the span")
    fake.remembered.clear()
    p.sync_turn(f"before{span}after", "ok")
    _wait_threads(p)
    stored = [i.text for b in fake.remembered for i in b]
    assert any("before after" in t for t in stored)


def test_marker_only_memory_does_not_hang_capture(provider):
    """R5 (codex P1): a memory consisting only of fence glyphs sanitizes
    to an empty displayed span — an empty span matches everywhere and
    advances no scan, hanging every later sync_turn."""
    from hermes_mnemostack.provider import FENCE_CLOSE, FENCE_OPEN

    p, fake = provider
    fake.hits = [RecallHit(id="1", text=f"{FENCE_OPEN}{FENCE_CLOSE}", score=0.9)]
    p.queue_prefetch("give me the marker memory")
    _wait_threads(p)
    p.prefetch("give me the marker memory")
    with p._lock:
        assert "" not in p._recently_injected.get(p._session_key(), {})
    fake.remembered.clear()
    p.sync_turn("an ordinary follow-up turn", "an ordinary reply")
    assert p.flush_captures(timeout=5.0)  # would hang before the fix
    stored = [i.text for b in fake.remembered for i in b]
    assert "an ordinary follow-up turn" in stored


def test_wordless_turns_survive_any_live_provenance(provider):
    """R5 (agent P1): a punctuation/emoji-only turn must be captured even
    when the session has fresh SHORT provenance (which it never contained)
    and even next to a genuinely echoed long span."""
    p, fake = provider
    long_fact = "the release checklist lives in the ops handbook appendix"
    fake.hits = [
        RecallHit(id="1", text=long_fact, score=0.9),
        RecallHit(id="2", text="see PR #157", score=0.8),  # short span
    ]
    p.queue_prefetch("where is the checklist?")
    _wait_threads(p)
    p.prefetch("where is the checklist?")

    # (a) neither span appears in the turn
    fake.remembered.clear()
    p.sync_turn("unrelated question about nothing in particular", "👍")
    _wait_threads(p)
    stored = [i.text for b in fake.remembered for i in b]
    assert "👍" in stored

    # (b) an emoji reaction alongside a genuinely echoed long span
    fake.remembered.clear()
    p.sync_turn(f"{long_fact}", "🙂")
    _wait_threads(p)
    stored = [i.text for b in fake.remembered for i in b]
    assert "🙂" in stored  # the reaction is the user's own content
    assert not any("ops handbook" in t for t in stored)  # the echo is gone


def test_wordless_memory_echo_is_still_suppressed(provider):
    """R7 (codex P1): the wordless-turn guard must not shelter an ACTUAL
    echo — a memory that is itself wordless ("👍") re-sent verbatim is
    still an echo and must not be re-stored."""
    p, fake = provider
    fake.hits = [RecallHit(id="1", text="👍", score=0.9)]
    p.queue_prefetch("what did they react with?")
    _wait_threads(p)
    p.prefetch("what did they react with?")
    fake.remembered.clear()
    p.sync_turn("👍", "acknowledged that reaction")
    _wait_threads(p)
    roles = [i.metadata["hermes_role"] for b in fake.remembered for i in b]
    assert roles == ["assistant"]  # the echoed 👍 is suppressed


def test_wordless_addition_to_a_complete_echo_is_dropped(provider):
    """R9 contract: coverage by WORDS decides. A turn contributing no
    words of its own is an echo however it was framed — including when a
    wordless reaction rides along. Documented residual: that reaction is
    lost, which is bounded and preferable to re-storing recalled text
    (which compounds). Contrast with the mixed-content pin below."""
    p, fake = provider
    fact = "the release checklist lives in the ops handbook appendix"
    fake.hits = [RecallHit(id="1", text=fact, score=0.9)]
    p.queue_prefetch("where is the checklist?")
    _wait_threads(p)
    p.prefetch("where is the checklist?")
    fake.remembered.clear()
    p.sync_turn(f"{fact} 👍", f"👍 {fact}")
    _wait_threads(p)
    stored = [i.text for b in fake.remembered for i in b]
    assert stored == []  # no words of their own → pure echoes
    # But a turn with ANY word of its own keeps that word.
    fake.remembered.clear()
    p.sync_turn(f"{fact} — thanks", "ok")
    _wait_threads(p)
    stored = [i.text for b in fake.remembered for i in b]
    assert any("thanks" in t for t in stored)
    assert not any("release checklist" in t for t in stored)


def test_block_echo_leaves_no_bullet_residue(provider):
    """The bullets of an echoed BLOCK are formatting, not content — they
    must not be stored as a memory of '- -'."""
    p, fake = provider
    fake.hits = [
        RecallHit(id="1", text="first recalled fact about deploys", score=0.9),
        RecallHit(id="2", text="second recalled fact about staging", score=0.8),
    ]
    p.queue_prefetch("give me the facts")
    _wait_threads(p)
    block = p.prefetch("give me the facts")
    fake.remembered.clear()
    p.sync_turn(block, block)
    _wait_threads(p)
    assert fake.remembered == []


def test_formatted_short_echo_still_collapses(provider):
    """R8 (codex P1): a short echo wrapped in quotes or a bullet is still
    a pure echo — presentation must not defeat suppression."""
    p, fake = provider
    fake.hits = [RecallHit(id="1", text="see PR #157", score=0.9)]
    p.queue_prefetch("which PR was that?")
    _wait_threads(p)
    p.prefetch("which PR was that?")
    for shape in (
        '"see PR #157"',
        "- see PR #157",
        "`see PR #157`",
        "_see PR #157_",
        "~~see PR #157~~",
        "{see PR #157}",
    ):
        fake.remembered.clear()
        p.sync_turn(shape, "ack")
        _wait_threads(p)
        stored = [i.text for b in fake.remembered for i in b]
        assert not any("PR #157" in t for t in stored), shape


def test_block_echo_with_a_wordless_addition_collapses(provider):
    """R9 contract: an echoed block plus a wordless reaction contributes
    no words of its own — dropped as a pure echo (documented residual).
    A block echo plus real words keeps the words."""
    p, fake = provider
    fake.hits = [
        RecallHit(id="1", text="first recalled fact about deploys", score=0.9),
        RecallHit(id="2", text="second recalled fact about staging", score=0.8),
    ]
    p.queue_prefetch("give me the facts")
    _wait_threads(p)
    block = p.prefetch("give me the facts")
    fake.remembered.clear()
    p.sync_turn(f"{block} 👍", "ack")
    _wait_threads(p)
    stored = [i.text for b in fake.remembered for i in b]
    assert not any("recalled fact" in t for t in stored)
    assert "👍" not in stored  # wordless addition rides along with the echo
    fake.remembered.clear()
    p.sync_turn(f"{block} noted for later", "ack")
    _wait_threads(p)
    stored = [i.text for b in fake.remembered for i in b]
    assert any("noted for later" in t for t in stored)
    assert not any("recalled fact" in t for t in stored)


def test_block_of_short_memories_echoes_through_the_probe_path(provider):
    """R8 (agent P3): a block whose memories are ALL below the short-span
    threshold exercises the probe path, not long-span masking — that
    branch was correct but untested."""
    p, fake = provider
    fake.hits = [
        RecallHit(id="1", text="see PR #157", score=0.9),
        RecallHit(id="2", text="ping @ops", score=0.8),
    ]
    p.queue_prefetch("what were the two notes?")
    _wait_threads(p)
    block = p.prefetch("what were the two notes?")
    assert all(len(h.text) < 24 for h in fake.hits)  # probe path, by construction
    fake.remembered.clear()
    p.sync_turn(block, "ack")
    _wait_threads(p)
    stored = [i.text for b in fake.remembered for i in b]
    assert stored == ["ack"]  # the block echo contributed no words of its own
