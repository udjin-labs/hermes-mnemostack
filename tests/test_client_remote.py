"""RemoteClient against a REAL mnemostack service app (starlette TestClient
injected as the httpx client): full lifecycle plus tenant isolation."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from hermes_mnemostack.client import (
    MemoryItem,
    MnemostackClientError,
    RemoteClient,
)


def _client(app, key):
    return RemoteClient("http://testserver", api_key=key, http=TestClient(app))


def test_full_lifecycle_round_trip(service_app):
    app, store, _emb, keys = service_app
    c = _client(app, keys["alpha"])
    out = c.remember([MemoryItem(text="the deploy window is Friday 15:00", source="chat/1")])
    assert (out.stored, out.duplicates, out.failed) == (1, 0, 0)
    # Duplicate re-send: zero-cost dedup.
    again = c.remember([MemoryItem(text="the deploy window is Friday 15:00", source="chat/1")])
    assert again.duplicates == 1 and again.stored == 0

    hits = c.recall("deploy window", limit=5)
    assert hits and any("Friday" in h.text for h in hits)
    pid = next(h.id for h in hits if "Friday" in h.text)

    assert c.invalidate([pid]) == 1
    assert all(h.id != pid for h in c.recall("deploy window", limit=5))

    # Re-remember reactivates; then hard-delete erases.
    c.remember([MemoryItem(text="the deploy window is Friday 15:00", source="chat/1")])
    assert c.forget([pid]) == 1
    assert c.forget([pid]) == 0  # idempotent retry
    assert store.client.retrieve(store.collection, ids=[pid], with_payload=False) == []


def test_tenant_isolation_read_and_lifecycle(service_app):
    app, store, _emb, keys = service_app
    alpha = _client(app, keys["alpha"])
    beta = _client(app, keys["beta"])
    alpha.remember([MemoryItem(text="alpha private fact", source="s")])
    alpha_id = alpha.recall("alpha private fact")[0].id

    # Read isolation: beta never sees alpha's memory.
    assert all("alpha private" not in h.text for h in beta.recall("alpha private fact"))
    # Lifecycle isolation: beta's invalidate/forget silently no-op on
    # alpha's ids (anti-oracle counts), and the point survives.
    assert beta.invalidate([alpha_id]) == 0
    assert beta.forget([alpha_id]) == 0
    assert len(store.client.retrieve(store.collection, ids=[alpha_id], with_payload=False)) == 1
    assert alpha.recall("alpha private fact")[0].id == alpha_id


def test_service_errors_are_clean(service_app):
    app, _store, _emb, keys = service_app
    c = _client(app, keys["alpha"])
    with pytest.raises(MnemostackClientError, match="400"):
        c.invalidate(["not-a-uuid"])
    unauth = RemoteClient("http://testserver", api_key="bogus", http=TestClient(app))
    with pytest.raises(MnemostackClientError, match="401"):
        unauth.recall("q")


def test_base_url_required_without_injection():
    with pytest.raises(MnemostackClientError, match="base_url"):
        RemoteClient("")


def test_oversized_item_rides_chunking_not_a_400(service_app):
    """R1 (agent P1): 40k-char assistant turn previously 400-failed the
    whole batch, losing BOTH sides. chunk:true must kick in."""
    app, _store, _emb, keys = service_app
    c = _client(app, keys["alpha"])
    big = "long assistant answer " * 2000  # > 32768 chars
    out = c.remember(
        [
            MemoryItem(text="short user question", source="chat/2", offset=0),
            MemoryItem(text=big, source="chat/2", offset=1),
        ]
    )
    assert out.failed == 0 and out.stored > 2
    assert any("short user question" in h.text for h in c.recall("short user question"))


def test_client_error_carries_status_code(service_app):
    app, _store, _emb, keys = service_app
    c = _client(app, keys["alpha"])
    try:
        c.invalidate(["not-a-uuid"])
        raise AssertionError("expected MnemostackClientError")
    except MnemostackClientError as e:
        assert e.status_code == 400


def test_routine_notes_are_not_reported_as_faults():
    """R4 (agent P2): pin the CONTRACT against a controlled response, not
    against whatever a live deployment happens to emit — mnemostack 2.2
    duplicates routine `notes` tags into `degraded` for back-compat, so a
    real fault is exactly degraded-minus-notes."""
    from hermes_mnemostack.client import real_faults

    class _FakeResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class _FakeHTTP:
        def __init__(self, payload):
            self._payload = payload
            self.calls = []

        def request(self, method, path, json=None, headers=None):
            self.calls.append((method, path))
            return _FakeResponse(self._payload)

    # Healthy recall carrying a routine signal in BOTH lists.
    http = _FakeHTTP(
        {
            "results": [{"id": "1", "text": "a memory", "score": 0.5}],
            "notes": ["temporal:no_parse"],
            "degraded": ["temporal:no_parse"],
        }
    )
    out = RemoteClient("http://svc", http=http).recall_detailed("q")
    assert out.notes == ["temporal:no_parse"]
    assert out.faults == []  # routine duplicate is NOT a fault
    assert len(out.hits) == 1

    # A genuine fault: present in degraded, absent from notes.
    http = _FakeHTTP(
        {
            "results": [],
            "notes": ["temporal:no_parse"],
            "degraded": ["temporal:no_parse", "vector:error"],
        }
    )
    out = RemoteClient("http://svc", http=http).recall_detailed("q")
    assert out.faults == ["vector:error"]

    # Unit contract, independent of transport.
    assert real_faults(["temporal:no_parse"], ["temporal:no_parse"]) == []
    assert real_faults(["vector:error"], []) == ["vector:error"]


def test_the_event_time_reaches_the_store(service_app):
    """The field was plumbed through this transport from the start and
    nothing ever filled it, which is exactly how "supported" and "working"
    came apart. Asserted against the STORE, not the request body: a payload
    key that leaves here but is dropped on arrival is the same outage, and
    only one of those two checks would notice.
    """
    app, store, _emb, keys = service_app
    c = _client(app, keys["alpha"])
    stamp = "2026-03-04T05:06:07+00:00"
    out = c.remember(
        [MemoryItem(text="quarterly review moved to March", source="chat/ts", timestamp=stamp)]
    )
    assert out.stored == 1, out

    points, _ = store.client.scroll(collection_name=store.collection, limit=100, with_payload=True)
    stamps = [
        p.payload.get("timestamp")
        for p in points
        if str(p.payload.get("source", "")).startswith("chat/ts")
    ]
    assert stamps == [stamp], (stamps, [p.payload for p in points])
