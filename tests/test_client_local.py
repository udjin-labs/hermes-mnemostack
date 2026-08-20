"""LocalClient: library stack over in-memory Qdrant, metadata scoping."""

from __future__ import annotations

import pytest

# `conftest`, not `tests.conftest`: tests/ is not a package, so on a
# clean checkout (CI) there is no `tests` module to import from —
# pytest puts the test directory itself on sys.path.
from conftest import FakeEmbedding, make_mem_store

from hermes_mnemostack.client import LocalClient, MemoryItem


@pytest.fixture()
def local_pair(monkeypatch):
    """Two LocalClients over ONE collection, scoped to different profiles."""
    import hermes_mnemostack.client as cl

    emb = FakeEmbedding()
    store = make_mem_store("local")

    import mnemostack.embeddings as emods
    import mnemostack.vector as vmod

    monkeypatch.setattr(emods, "get_provider", lambda _n, **_k: emb)
    monkeypatch.setattr(vmod, "VectorStore", lambda **_: store)
    # client.py imports inside methods — patch the modules it imports FROM.
    del cl

    coder = LocalClient(
        collection="local",
        qdrant_url="http://unused",
        embedding_provider="fake",
        scope={"hermes_profile": "coder"},
    )
    writer = LocalClient(
        collection="local",
        qdrant_url="http://unused",
        embedding_provider="fake",
        scope={"hermes_profile": "writer"},
    )
    return coder, writer, store


def test_round_trip_and_scope_stamping(local_pair):
    coder, _writer, store = local_pair
    out = coder.remember([MemoryItem(text="prefers tabs over spaces", source="s")])
    assert out.stored == 1
    hits = coder.recall("tabs or spaces")
    assert hits and hits[0].payload.get("hermes_profile") == "coder"
    # Duplicate: dedup against the store.
    assert coder.remember(
        [MemoryItem(text="prefers tabs over spaces", source="s")]
    ).duplicates == 1


def test_two_profiles_do_not_see_each_other(local_pair):
    coder, writer, store = local_pair
    coder.remember([MemoryItem(text="coder-only secret preference", source="s")])
    writer.remember([MemoryItem(text="writer-only style guide", source="s")])
    assert all("coder-only" not in h.text for h in writer.recall("coder-only secret"))
    assert all("writer-only" not in h.text for h in coder.recall("writer-only style"))


def test_lifecycle_is_scope_guarded(local_pair):
    coder, writer, store = local_pair
    coder.remember([MemoryItem(text="ephemeral fact", source="s")])
    pid = coder.recall("ephemeral fact")[0].id
    # Foreign scope: silent no-op, the point survives (anti-oracle counts).
    assert writer.invalidate([pid]) == 0
    assert writer.forget([pid]) == 0
    assert len(store.client.retrieve(store.collection, ids=[pid], with_payload=False)) == 1
    # Own scope: works.
    assert coder.invalidate([pid]) == 1
    assert all(h.id != pid for h in coder.recall("ephemeral fact"))
    assert coder.forget([pid]) == 1
    assert store.client.retrieve(store.collection, ids=[pid], with_payload=False) == []


def test_caller_filters_cannot_widen_scope(local_pair):
    coder, writer, _store = local_pair
    coder.remember([MemoryItem(text="scoped datum", source="s")])
    # A caller-supplied filter must never override the client's own scope.
    hits = writer.recall("scoped datum", filters={"hermes_profile": "coder"})
    assert all("scoped datum" not in h.text for h in hits)


def test_same_turn_text_across_profiles_is_not_a_cross_scope_duplicate(local_pair):
    """R1 (both reviewers, P1): two profiles capturing the same
    (source, offset, text) — trivially likely for short turns like "hi"
    under one session id — must yield TWO points, not a silent duplicate
    that leaves the second profile's memory invisible."""
    coder, writer, _store = local_pair
    item = MemoryItem(text="hi", source="hermes/cli/sess-1", offset=0)
    assert coder.remember([item]).stored == 1
    out = writer.remember([item])
    assert (out.stored, out.duplicates) == (1, 0)  # not a duplicate!
    assert any(h.text == "hi" for h in coder.recall("hi"))
    assert any(h.text == "hi" for h in writer.recall("hi"))


def test_oversized_item_chunks_instead_of_failing(local_pair):
    """R1 (agent P1): an item past the per-item cap rides the chunker —
    the batch must not be lost."""
    coder, _writer, _store = local_pair
    big = "deploy notes " * 4000  # > 32768 chars
    out = coder.remember(
        [MemoryItem(text=big, source="hermes/cli/s", offset=0),
         MemoryItem(text="small survivor", source="hermes/cli/s", offset=1)]
    )
    assert out.failed == 0 and out.stored > 2  # chunk expansion happened
    assert any("survivor" in h.text for h in coder.recall("small survivor"))
