"""One client boundary, two transports.

The provider only ever talks to :class:`MnemoStackClient`; whether that is
a remote mnemostack service or an in-process mnemostack library stack is a
configuration detail. Both cover the full lifecycle: recall, remember,
invalidate (soft retraction), forget (hard delete).

Isolation models differ by design and are documented per client:

- Remote: the TENANT is resolved server-side from the service key — a
  client cannot read or write another tenant, enforced by the service.
- Local: profile/user scoping is metadata-based within one collection —
  a same-machine trust domain, not a security boundary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class MnemostackClientError(RuntimeError):
    """A transport-level or service-reported failure, with a caller-safe
    message (no stack traces or secrets)."""


@dataclass(frozen=True)
class RecallHit:
    id: str
    text: str
    score: float
    sources: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryItem:
    text: str
    source: str = ""
    offset: int = 0
    timestamp: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RememberOutcome:
    stored: int
    duplicates: int
    failed: int


class MnemoStackClient(Protocol):
    def recall(
        self, query: str, *, limit: int = 5, filters: dict[str, Any] | None = None
    ) -> list[RecallHit]: ...

    def remember(self, items: list[MemoryItem]) -> RememberOutcome: ...

    def invalidate(self, ids: list[str]) -> int: ...

    def forget(self, ids: list[str]) -> int: ...

    def close(self) -> None: ...


# --------------------------------------------------------------- remote


class RemoteClient:
    """HTTP transport against a running mnemostack service (>= 2.2).

    Uses `POST /recall`, `POST /memories`, `POST /invalidate`, and
    `DELETE /memories`. The service key (if any) rides in X-API-Key; the
    tenant comes from the key, never from the client.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        timeout: float = 30.0,
        http: Any | None = None,
    ) -> None:
        if http is not None:
            # Dependency injection for tests (e.g. starlette TestClient,
            # itself an httpx.Client wired to an in-process app).
            self._http = http
            self._owns_http = False
        else:
            import httpx

            if not base_url:
                raise MnemostackClientError("remote mode requires base_url")
            self._http = httpx.Client(base_url=base_url, timeout=timeout)
            self._owns_http = True
        self._headers = {"X-API-Key": api_key} if api_key else {}

    def _request(self, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self._http.request(method, path, json=payload, headers=self._headers)
        except Exception as exc:
            raise MnemostackClientError(
                f"mnemostack service unreachable: {type(exc).__name__}"
            ) from exc
        if resp.status_code >= 400:
            detail = ""
            try:
                detail = str(resp.json().get("detail", ""))[:300]
            except Exception:  # noqa: BLE001 — non-JSON error body
                pass
            raise MnemostackClientError(
                f"mnemostack service returned {resp.status_code} for {path}"
                + (f": {detail}" if detail else "")
            )
        return resp.json()

    def recall(
        self, query: str, *, limit: int = 5, filters: dict[str, Any] | None = None
    ) -> list[RecallHit]:
        payload: dict[str, Any] = {"query": query, "limit": limit}
        if filters:
            payload["filters"] = filters
        data = self._request("POST", "/recall", payload)
        return [
            RecallHit(
                id=str(r.get("id", "")),
                text=str(r.get("text", "")),
                score=float(r.get("score", 0.0)),
                sources=list(r.get("retrievers", [])),
                payload=dict(r.get("metadata", {})),
            )
            for r in data.get("results", [])
        ]

    def remember(self, items: list[MemoryItem]) -> RememberOutcome:
        payload = {
            "items": [
                {
                    "text": it.text,
                    "source": it.source,
                    "offset": it.offset,
                    **({"timestamp": it.timestamp} if it.timestamp else {}),
                    **({"tags": it.tags} if it.tags else {}),
                    **({"metadata": it.metadata} if it.metadata else {}),
                }
                for it in items
            ]
        }
        data = self._request("POST", "/memories", payload)
        return RememberOutcome(
            stored=int(data.get("stored", 0)),
            duplicates=int(data.get("duplicates", 0)),
            failed=int(data.get("failed", 0)),
        )

    def invalidate(self, ids: list[str]) -> int:
        data = self._request("POST", "/invalidate", {"ids": ids})
        return int(data.get("invalidated", 0))

    def forget(self, ids: list[str]) -> int:
        data = self._request("DELETE", "/memories", {"ids": ids})
        return int(data.get("deleted", 0))

    def close(self) -> None:
        if self._owns_http:
            try:
                self._http.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------- local


class LocalClient:
    """mnemostack as a library against the agent's own Qdrant.

    ``scope`` (e.g. ``{"hermes_profile": "coder", "hermes_user": "u1"}``)
    is stamped into every write's metadata and applied as a filter on
    every recall, so profiles/users share one collection without seeing
    each other. This is METADATA scoping in one trust domain — the id-
    addressed lifecycle calls (invalidate/forget) are scope-checked here
    in the client, but anyone with library access to the same Qdrant can
    bypass it; use remote mode with service keys for a real boundary.
    """

    def __init__(
        self,
        *,
        collection: str,
        qdrant_url: str,
        embedding_provider: str,
        embedding_model: str = "",
        scope: dict[str, str] | None = None,
        recall_limit_overfetch: int = 20,
    ) -> None:
        from mnemostack.embeddings import get_provider
        from mnemostack.recall import Recaller
        from mnemostack.vector import VectorStore

        kwargs: dict[str, Any] = {}
        if embedding_model:
            kwargs["model"] = embedding_model
        self._provider = get_provider(embedding_provider, **kwargs)
        self._store = VectorStore(
            collection=collection,
            dimension=self._provider.dimension,
            host=qdrant_url,
        )
        self._store.ensure_collection()
        self._recaller = Recaller(
            embedding_provider=self._provider, vector_store=self._store
        )
        self._scope = dict(scope or {})
        self._overfetch = recall_limit_overfetch

    def recall(
        self, query: str, *, limit: int = 5, filters: dict[str, Any] | None = None
    ) -> list[RecallHit]:
        merged = dict(filters or {})
        merged.update(self._scope)  # scope always wins — never widen it
        results = self._recaller.recall(
            query,
            limit=limit,
            vector_limit=max(limit, self._overfetch),
            filters=merged or None,
        )
        return [
            RecallHit(
                id=str(r.id),
                text=r.text,
                score=float(r.score),
                sources=list(getattr(r, "sources", [])),
                payload=dict(getattr(r, "payload", {}) or {}),
            )
            for r in results
        ]

    def remember(self, items: list[MemoryItem]) -> RememberOutcome:
        from mnemostack.ingest import IngestItem, ingest_remote_items

        ingest_items = []
        for it in items:
            metadata = dict(it.metadata)
            metadata.update(self._scope)  # scope always wins
            ingest_items.append(
                IngestItem(
                    text=it.text,
                    source=it.source,
                    offset=it.offset,
                    timestamp=it.timestamp,
                    tags=list(it.tags),
                    metadata=metadata,
                )
            )
        results = ingest_remote_items(self._provider, self._store, ingest_items)
        return RememberOutcome(
            stored=sum(r.status == "stored" for r in results),
            duplicates=sum(r.status == "duplicate" for r in results),
            failed=sum(r.status == "failed" for r in results),
        )

    def _owned(self, ids: list[str]) -> list[str]:
        """Ids from this client's scope only — mirrors the remote surface's
        silent-skip semantics (counts are not an existence oracle)."""
        if not self._scope:
            return list(ids)
        points = self._store.client.retrieve(
            self._store.collection, ids=list(ids), with_payload=True
        )
        owned = []
        for pt in points:
            payload = getattr(pt, "payload", None) or {}
            if all(payload.get(k) == v for k, v in self._scope.items()):
                owned.append(str(pt.id))
        return owned

    def invalidate(self, ids: list[str]) -> int:
        owned = self._owned(ids)
        if not owned:
            return 0
        return int(self._store.invalidate(owned))

    def forget(self, ids: list[str]) -> int:
        owned = self._owned(ids)
        if not owned:
            return 0
        return int(self._store.delete_points(list(owned)))

    def close(self) -> None:
        close = getattr(getattr(self._store, "client", None), "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass


def build_client(cfg: dict[str, Any], *, scope: dict[str, str] | None = None) -> MnemoStackClient:
    """Construct the configured transport from a load_config() dict."""
    if cfg["mode"] == "remote":
        return RemoteClient(
            cfg["base_url"], api_key=cfg.get("api_key", ""), timeout=cfg["timeout"]
        )
    return LocalClient(
        collection=cfg["collection"],
        qdrant_url=cfg["qdrant_url"],
        embedding_provider=cfg["embedding_provider"],
        embedding_model=cfg["embedding_model"],
        scope=scope,
    )
