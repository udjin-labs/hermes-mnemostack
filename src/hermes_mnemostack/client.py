"""One client boundary, two transports.

The provider only ever talks to :class:`MnemoStackClient`; whether that is
a remote mnemostack service or an in-process mnemostack library stack is a
configuration detail. Both cover the full lifecycle: recall, remember,
invalidate (soft retraction), forget (hard delete — a client-API
capability; the model-facing tool exposes retraction only).

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
    message (no stack traces or secrets). ``status_code`` carries the
    HTTP status when the service answered (None for transport errors) so
    callers can distinguish permanent conditions (401 revoked key, 507
    quota) from transient ones (429/503) without parsing the message."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


#: Mirror of the service's per-item text cap (REMOTE_MAX_TEXT_CHARS):
#: longer items must ride chunk=true or the whole batch 400s.
REMOTE_TEXT_CAP = 32768


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
class RecallOutcome:
    """Hits plus REAL faults for this recall.

    mnemostack 2.2 splits recall signals: ``notes`` is the authoritative
    list of routine ones (e.g. ``temporal:no_parse`` — a query with no
    parsable time expression), while ``degraded`` still duplicates those
    same routine tags for back-compat until the next major. So a genuine
    fault is exactly ``degraded - notes``; treating all of ``degraded``
    as breakage would report perfectly healthy calls as degraded.
    """

    hits: list[RecallHit] = field(default_factory=list)
    faults: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def real_faults(degraded: list[str], notes: list[str]) -> list[str]:
    """Entries in ``degraded`` that are not routine ``notes`` signals."""
    routine = set(notes or ())
    return [d for d in (degraded or ()) if d not in routine]


@dataclass(frozen=True)
class RememberOutcome:
    stored: int
    duplicates: int
    failed: int


class MnemoStackClient(Protocol):
    def recall(
        self, query: str, *, limit: int = 5, filters: dict[str, Any] | None = None
    ) -> list[RecallHit]: ...

    def recall_detailed(
        self, query: str, *, limit: int = 5, filters: dict[str, Any] | None = None
    ) -> RecallOutcome: ...

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
                + (f": {detail}" if detail else ""),
                status_code=resp.status_code,
            )
        return resp.json()

    def recall(
        self, query: str, *, limit: int = 5, filters: dict[str, Any] | None = None
    ) -> list[RecallHit]:
        return self.recall_detailed(query, limit=limit, filters=filters).hits

    def recall_detailed(
        self, query: str, *, limit: int = 5, filters: dict[str, Any] | None = None
    ) -> RecallOutcome:
        payload: dict[str, Any] = {"query": query, "limit": limit}
        if filters:
            payload["filters"] = filters
        data = self._request("POST", "/recall", payload)
        notes = [str(n) for n in data.get("notes", [])]
        return RecallOutcome(
            hits=[
                RecallHit(
                    id=str(r.get("id", "")),
                    text=str(r.get("text", "")),
                    score=float(r.get("score", 0.0)),
                    sources=list(r.get("retrievers", [])),
                    payload=dict(r.get("metadata", {})),
                )
                for r in data.get("results", [])
            ],
            faults=real_faults([str(d) for d in data.get("degraded", [])], notes),
            notes=notes,
        )

    @staticmethod
    def _chunkable(it: MemoryItem) -> MemoryItem:
        """The chunk contract requires offset 0 (chunk offsets are computed
        server-side) — for a long item at a non-zero offset, fold the
        offset into the source so the deterministic id stays unique.
        Assumption: hermes sources are "hermes/{platform}/{session}" and
        never naturally contain "#o<digits>"; ids also hash the full text,
        so a crafted collision additionally needs identical content."""
        if len(it.text) <= REMOTE_TEXT_CAP or it.offset == 0:
            return it
        from dataclasses import replace

        return replace(it, source=f"{it.source}#o{it.offset}", offset=0)

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
                    # Long items ride the server-side chunker instead of
                    # 400-failing the whole batch on the per-item cap.
                    **({"chunk": True} if len(it.text) > REMOTE_TEXT_CAP else {}),
                }
                for it in (self._chunkable(i) for i in items)
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
    is folded into a canonical mnemostack TENANT string, which rides the
    library's native tenant mechanism end to end: deterministic ids are
    tenant-prefixed (two profiles ingesting the same (source, offset,
    text) get DIFFERENT points, never a cross-profile duplicate), every
    point is stamped with ``tenant_id``, recall filters on it, and the
    id-addressed lifecycle calls use the store's own tenant ownership
    guards with the remote surface's silent-skip anti-oracle semantics.
    Still one trust domain — anyone with library access to the same
    Qdrant can pass a different tenant; use remote mode with service
    keys for a real boundary. Scope keys are additionally stamped as
    plain metadata for inspectability.
    """

    @staticmethod
    def _scope_tenant(scope: dict[str, str] | None) -> str | None:
        """Canonical, order-independent, INJECTIVE tenant string.

        Canonical JSON, not a delimiter join: scope values come from
        host-supplied identity strings with no character restrictions, and
        a bare "k=v|k=v" encoding lets crafted values collide two
        different scopes into one tenant."""
        if not scope:
            return None
        import json

        return json.dumps(scope, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

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
        self._recaller = Recaller(embedding_provider=self._provider, vector_store=self._store)
        self._scope = dict(scope or {})
        self._tenant = self._scope_tenant(scope)
        self._overfetch = recall_limit_overfetch

    def recall(
        self, query: str, *, limit: int = 5, filters: dict[str, Any] | None = None
    ) -> list[RecallHit]:
        # No trace: a RecallTrace makes the recaller build per-retriever
        # ranked lists and a fused list this caller would discard.
        return self._recall(query, limit=limit, filters=filters, trace=False).hits

    def recall_detailed(
        self, query: str, *, limit: int = 5, filters: dict[str, Any] | None = None
    ) -> RecallOutcome:
        return self._recall(query, limit=limit, filters=filters, trace=True)

    def _recall(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        trace: bool = True,
    ) -> RecallOutcome:
        merged = dict(filters or {})
        merged.update(self._scope)  # scope always wins — never widen it
        # The NATIVE tenant parameter, not a tenant_id filter: it rides
        # every tenant-aware retriever, gates non-aware ones, and keeps
        # the recaller's filter_by_tenant backstop — a future bm25/graph
        # arm stays isolated automatically.
        tkw: dict[str, Any] = {"tenant": self._tenant} if self._tenant is not None else {}
        rt = None
        if trace:
            from mnemostack.recall import RecallTrace

            rt = RecallTrace()
            tkw["trace"] = rt
        results = self._recaller.recall(
            query,
            limit=limit,
            vector_limit=max(limit, self._overfetch),
            filters=merged or None,
            **tkw,
        )
        notes = [str(n) for n in getattr(rt, "notes", [])] if rt is not None else []
        return RecallOutcome(
            hits=[
                RecallHit(
                    id=str(r.id),
                    text=r.text,
                    score=float(r.score),
                    sources=list(getattr(r, "sources", [])),
                    payload=dict(getattr(r, "payload", {}) or {}),
                )
                for r in results
            ],
            faults=real_faults(
                [str(d) for d in getattr(rt, "degraded", [])] if rt is not None else [],
                notes,
            ),
            notes=notes,
        )

    def remember(self, items: list[MemoryItem]) -> RememberOutcome:
        from mnemostack.ingest import (
            REMOTE_MAX_TEXT_CHARS,
            IngestItem,
            expand_remote_items,
            ingest_remote_items,
        )

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
        # Same chunking escape hatch as the remote surface: an oversized
        # item splits into fixed windows instead of failing the batch.
        # Chunked items must sit at offset 0 (the chunker computes window
        # offsets) — fold a non-zero offset into the source, mirroring
        # RemoteClient._chunkable for cross-transport id parity.
        prepared = []
        for item in ingest_items:
            if len(item.text) > REMOTE_MAX_TEXT_CHARS and item.offset != 0:
                item = IngestItem(
                    text=item.text,
                    source=f"{item.source}#o{item.offset}",
                    offset=0,
                    timestamp=item.timestamp,
                    tags=item.tags,
                    metadata=item.metadata,
                )
            prepared.append(item)
        entries = [(item, len(item.text) > REMOTE_MAX_TEXT_CHARS) for item in prepared]
        flat_items, _origins = expand_remote_items(entries)
        results = ingest_remote_items(self._provider, self._store, flat_items, tenant=self._tenant)
        return RememberOutcome(
            stored=sum(r.status == "stored" for r in results),
            duplicates=sum(r.status == "duplicate" for r in results),
            failed=sum(r.status == "failed" for r in results),
        )

    def invalidate(self, ids: list[str]) -> int:
        from mnemostack.ingest import coerce_point_ids

        tkw = {"tenant": self._tenant} if self._tenant is not None else {}
        return int(self._store.invalidate(coerce_point_ids(list(ids)), **tkw))

    def forget(self, ids: list[str]) -> int:
        from mnemostack.ingest import coerce_point_ids

        tkw = {"tenant": self._tenant} if self._tenant is not None else {}
        return int(self._store.delete_points(coerce_point_ids(list(ids)), **tkw))

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
        return RemoteClient(cfg["base_url"], api_key=cfg.get("api_key", ""), timeout=cfg["timeout"])
    return LocalClient(
        collection=cfg["collection"],
        qdrant_url=cfg["qdrant_url"],
        embedding_provider=cfg["embedding_provider"],
        embedding_model=cfg["embedding_model"],
        scope=scope,
    )
