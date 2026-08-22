"""Shared fixtures: a REAL mnemostack service app (auth, real recall path,
in-memory Qdrant, deterministic fake embedder) and a local library stack."""

from __future__ import annotations

import hashlib

import pytest


class FakeEmbedding:
    """A 3-dim embedder that is deterministic ACROSS PROCESSES.

    It used to build the vector from `hash(text)`, which Python salts per
    process for str: the same text embedded in two runs produced two
    different vectors, so the ranking of any fixture corpus was random
    from run to run. Every test that asserts something about ORDER — that
    a particular memory comes back for a particular query — was therefore
    flaky by construction, and failed on roughly whichever job drew a bad
    salt. `blake2s` gives the same bytes everywhere, which is what
    "deterministic" was always claiming.

    Pinning PYTHONHASHSEED in CI would have hidden it in CI alone and left
    every contributor's local run rolling dice.
    """

    dimension = 3

    def __init__(self):
        self.embedded: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.embedded.append(text)
        h = int.from_bytes(hashlib.blake2s(text.encode("utf-8")).digest()[:8], "big")
        return [(h % 97) / 97.0, (h % 89) / 89.0, 1.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def health_check(self):
        return True, "ok"


def make_mem_store(collection: str = "hermes"):
    from mnemostack.vector import VectorStore
    from qdrant_client import QdrantClient

    store = VectorStore(collection=collection, dimension=3)
    store.client = QdrantClient(":memory:")
    store.ensure_collection()
    return store


@pytest.fixture()
def service_app(monkeypatch, tmp_path):
    """Real mnemostack build_app: auth with two tenants, REAL recall path
    (Recaller + VectorRetriever over in-memory Qdrant), no graph/LLM."""
    pytest.importorskip("fastapi")
    import mnemostack.server as srv
    from mnemostack.auth import FileKeyStore
    from mnemostack.server import ServerConfig, build_app

    emb = FakeEmbedding()
    store = make_mem_store("svc")
    monkeypatch.setattr(srv, "VectorStore", lambda **_: store)
    monkeypatch.setattr(srv, "get_provider", lambda _n, **_k: emb)

    class _Probe:
        def get_collections(self):
            return object()

    monkeypatch.setattr(srv, "_make_probe_client", lambda *_a, **_k: _Probe())

    def _no_llm(*_a, **_k):
        raise RuntimeError("no llm in tests")

    monkeypatch.setattr(srv, "get_llm", _no_llm)

    ks = FileKeyStore(tmp_path / "keys.json")
    keys = {}
    _, keys["alpha"] = ks.issue("alpha", ["read", "write"])
    _, keys["beta"] = ks.issue("beta", ["read", "write"])

    cfg = ServerConfig(
        provider_name="fake",
        llm_name="fake",
        graph_uri=None,
        auth_enabled=True,
        keys_file=str(tmp_path / "keys.json"),
    )
    app = build_app(cfg)
    return app, store, emb, keys
