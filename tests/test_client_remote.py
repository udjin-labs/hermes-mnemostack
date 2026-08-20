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
    out = c.remember(
        [MemoryItem(text="the deploy window is Friday 15:00", source="chat/1")]
    )
    assert (out.stored, out.duplicates, out.failed) == (1, 0, 0)
    # Duplicate re-send: zero-cost dedup.
    again = c.remember(
        [MemoryItem(text="the deploy window is Friday 15:00", source="chat/1")]
    )
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
        [MemoryItem(text="short user question", source="chat/2", offset=0),
         MemoryItem(text=big, source="chat/2", offset=1)]
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


def test_routine_notes_are_not_reported_as_faults(service_app):
    """mnemostack 2.2 duplicates routine `notes` tags into `degraded` for
    back-compat until the next major — a real fault is exactly
    degraded-minus-notes. Reporting all of `degraded` would flag healthy
    recalls as broken."""
    from hermes_mnemostack.client import real_faults

    app, _store, _emb, keys = service_app
    c = _client(app, keys["alpha"])
    c.remember([MemoryItem(text="a fact about deploys", source="s")])
    # A query with no parsable time expression emits the routine
    # temporal:no_parse signal on a real service.
    out = c.recall_detailed("deploys", limit=5)
    assert out.faults == [] or all(f not in out.notes for f in out.faults)
    for note in out.notes:
        assert note not in out.faults

    # Unit contract, independent of what this deployment happens to emit.
    assert real_faults(["temporal:no_parse"], ["temporal:no_parse"]) == []
    assert real_faults(["vector:error", "temporal:no_parse"], ["temporal:no_parse"]) == [
        "vector:error"
    ]
