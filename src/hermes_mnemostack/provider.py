"""mnemostack memory provider for hermes-agent.

Lifecycle wiring over the transport-agnostic client boundary
(:mod:`hermes_mnemostack.client`): background prefetch feeding the next
turn's context injection, verbatim turn capture, deterministic recall
indicator, session-switch bookkeeping. Tools and the fenced system-prompt
block arrive in a later task — this layer is recall+capture only.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from agent.memory_provider import MemoryProvider

# hermes-agent 0.20 additions, absent on the 0.19 floor: the recall
# indicator dataclass and the shared trivial-prompt gate. On 0.19 the host
# never calls recall_status(), so a minimal stand-in dataclass is enough,
# and the prefetch gate falls back to a conservative local check.
try:  # pragma: no cover — exercised implicitly by whichever host is installed
    from agent.memory_provider import RecallStatus
except ImportError:  # hermes-agent 0.19
    from dataclasses import dataclass as _dataclass

    @_dataclass(frozen=True)
    class RecallStatus:  # type: ignore[no-redef]
        provider_label: str
        count: int
        glyph: str = "🧠"


try:  # pragma: no cover
    from agent.memory_provider import is_trivial_prompt
except ImportError:  # hermes-agent 0.19

    def is_trivial_prompt(text: str | None) -> bool:  # type: ignore[misc]
        # Conservative GUESS, not a verified match of the 0.20 gate (which
        # is regex-based): may skip prefetch for short real queries like
        # "k8s"; capture is unaffected either way.
        if not text or not text.strip():
            return True
        stripped = text.strip()
        return stripped.startswith("/") or len(stripped) < 3

from .client import MemoryItem, MnemoStackClient, build_client
from .config import is_configured, load_config
from .config import save_config as _save_config_file

logger = logging.getLogger(__name__)

PROVIDER_NAME = "mnemostack"
GLYPH = "🗿"

#: Payload keys the provider stamps for local-mode scoping. The remote
#: transport does not send them — the tenant comes from the service key.
SCOPE_PROFILE_KEY = "hermes_profile"
SCOPE_USER_KEY = "hermes_user"

_SYNC_JOIN_TIMEOUT = 5.0


class MnemostackProvider(MemoryProvider):
    """Persistent agent memory backed by a mnemostack deployment."""

    def __init__(self) -> None:
        self._client: MnemoStackClient | None = None
        self._cfg: dict[str, Any] = {}
        self._session_id = ""
        self._platform = ""
        self._agent_context = ""
        self._lock = threading.Lock()
        # Per-session state: the ABC threads session_id through
        # prefetch/queue_prefetch/sync_turn for hosts serving concurrent
        # sessions — keying by it keeps one session's recall block and
        # turn numbering from leaking into another. Key "" = the
        # provider's own session (single-session CLI).
        self._turn_index: dict[str, int] = {}
        self._prefetched: dict[str, tuple[str, int]] = {}
        self._prefetch_gen: dict[str, int] = {}
        # Single field by ABC design: recall_status() takes no session_id
        # and the host calls it "right after prefetch, on the same
        # (single) turn thread" — that serialization is the ABC's own
        # contract, not an assumption of ours.
        self._last_injected_count: int | None = None
        self._prefetch_threads: dict[str, threading.Thread] = {}
        self._sync_threads: list[threading.Thread] = []
        self._shutting_down = False

    # -- Required ABC surface -------------------------------------------------

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        # Contract: config and deps only, no network. Mode-specific
        # required fields are validated HERE so hermes reports an
        # unavailable provider instead of activating one that dies in
        # initialize() (remote mode without a base_url).
        if not is_configured():
            return False
        try:
            cfg = load_config()
        except Exception:  # noqa: BLE001 — malformed config = unavailable
            return False
        if cfg["mode"] == "remote" and not cfg["base_url"]:
            return False
        return True

    def unavailable_reason(self) -> str:
        if is_configured():
            try:
                cfg = load_config()
            except Exception as exc:  # noqa: BLE001
                return f"mnemostack config is invalid: {exc}"
            if cfg["mode"] == "remote" and not cfg["base_url"]:
                return (
                    "mnemostack remote mode needs a base_url — set it in "
                    "mnemostack.json or MNEMOSTACK_BASE_URL"
                )
        return (
            "mnemostack is not configured — run `hermes memory setup` or set "
            "MNEMOSTACK_MODE (remote: MNEMOSTACK_BASE_URL + MNEMOSTACK_API_KEY)"
        )

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._session_id = session_id
        self._platform = str(kwargs.get("platform", ""))
        self._agent_context = str(kwargs.get("agent_context", "") or "primary")
        hermes_home = str(kwargs.get("hermes_home", "")) or None
        self._cfg = load_config(hermes_home)
        scope: dict[str, str] = {}
        identity = str(kwargs.get("agent_identity", "") or "")
        user_id = str(kwargs.get("user_id", "") or "")
        if identity:
            scope[SCOPE_PROFILE_KEY] = identity
        if user_id:
            scope[SCOPE_USER_KEY] = user_id
        self._client = build_client(self._cfg, scope=scope or None)
        logger.info(
            "mnemostack provider initialized (mode=%s, session=%s, platform=%s)",
            self._cfg.get("mode"),
            session_id,
            self._platform,
        )

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return []  # tools land in a later task

    # -- Recall path ----------------------------------------------------------

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        client = self._client
        if client is None or is_trivial_prompt(query):
            return
        key = session_id or ""
        with self._lock:
            # Generation gate: an outrun earlier recall finishing AFTER a
            # newer one must not overwrite the fresher block with stale
            # memories for a previous query.
            gen = self._prefetch_gen.get(key, 0) + 1
            self._prefetch_gen[key] = gen
            self._prune_session_dicts_locked()

        def _run() -> None:
            try:
                hits = client.recall(query, limit=int(self._cfg["recall_limit"]))
            except Exception as exc:  # noqa: BLE001 — recall must never break a turn
                logger.warning("mnemostack prefetch failed: %s", exc)
                hits = []
            block = self._format_hits(hits)
            with self._lock:
                if self._prefetch_gen.get(key, 0) == gen:
                    self._prefetched[key] = (block, len(hits))

        t = threading.Thread(target=_run, daemon=True, name="mnemostack-prefetch")
        with self._lock:
            # Per-SESSION thread reference: joining some other session's
            # thread would return this session's block as empty while its
            # own recall is still in flight.
            self._prefetch_threads[key] = t
        t.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        # Serve the cached background result; if the thread is still in
        # flight give it a short grace window rather than blocking a turn.
        key = session_id or ""
        with self._lock:
            t = self._prefetch_threads.get(key)
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        with self._lock:
            block, count = self._prefetched.pop(key, ("", 0))
        self._last_injected_count = count if block else None
        return block

    def recall_status(self) -> RecallStatus | None:
        if self._last_injected_count is None:
            return None
        return RecallStatus(
            provider_label="mnemostack",
            count=self._last_injected_count,
            glyph=GLYPH,
        )

    @staticmethod
    def _format_hits(hits: list[Any]) -> str:
        if not hits:
            return ""
        lines = ["Relevant memories (mnemostack recall):"]
        for h in hits:
            text = " ".join(h.text.split())
            if len(text) > 500:
                text = text[:500] + "…"
            lines.append(f"- {text}")
        return "\n".join(lines)

    # -- Capture path ---------------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        if self._client is None or not self._cfg.get("capture", True):
            return
        if self._shutting_down:
            return  # racing a shutdown: the client is about to close
        if self._agent_context != "primary":
            # Cron/subagent/flush contexts must not pollute user memory.
            return
        sid = session_id or self._session_id
        with self._lock:
            turn = self._turn_index.get(sid, 0)
            self._turn_index[sid] = turn + 1
            self._prune_session_dicts_locked()
        items = []
        for role, content in (("user", user_content), ("assistant", assistant_content)):
            content = (content or "").strip()
            if not content:
                continue
            items.append(
                MemoryItem(
                    text=content,
                    # Deterministic (source, offset, text) id: replaying a
                    # turn is a zero-cost duplicate, never a second copy.
                    source=f"hermes/{self._platform or 'cli'}/{sid}",
                    offset=turn * 2 + (0 if role == "user" else 1),
                    metadata={"hermes_role": role, "hermes_turn": turn},
                )
            )
        if not items:
            return

        client = self._client

        def _sync() -> None:
            # Capture must never break a turn — but it also must not lose
            # BOTH sides to one bad item: on a service-rejected batch,
            # retry per item so the valid side still lands. Permanent
            # conditions (401 revoked key, 507 quota) log like transient
            # ones here — a deliberate v1 tradeoff, see MnemostackClientError.
            try:
                out = client.remember(items)
                if out.failed:
                    logger.warning(
                        "mnemostack capture: %d item(s) failed to embed", out.failed
                    )
                return
            except Exception as exc:  # noqa: BLE001
                # Retry per item only when it can help: a service-side
                # VALIDATION rejection (4xx other than auth) means one bad
                # item poisoned the batch. Transport errors (no status) and
                # permanent conditions (revoked key, exhausted quota) fail
                # identically per item — pure overhead during an outage.
                status = getattr(exc, "status_code", None)
                retryable = status is not None and status not in (401, 403, 429, 507)
                if len(items) < 2 or not retryable:
                    logger.warning("mnemostack sync_turn failed: %s", exc)
                    return
                logger.warning(
                    "mnemostack sync_turn batch failed (%s) — retrying per item", exc
                )
            for item in items:
                try:
                    client.remember([item])
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "mnemostack capture lost a %s item: %s",
                        item.metadata.get("hermes_role", "turn"),
                        exc,
                    )

        t = threading.Thread(target=_sync, daemon=True, name="mnemostack-sync")
        with self._lock:
            # Prune BEFORE appending and START inside the lock: an
            # unstarted thread reports is_alive() False, so a concurrent
            # sync_turn's prune could otherwise drop it from the drain
            # list before it ever ran.
            self._sync_threads = [x for x in self._sync_threads if x.is_alive()]
            self._sync_threads.append(t)
            t.start()

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        # v1: the evicted turns were already captured by sync_turn (same
        # deterministic ids — re-ingest would be a no-cost duplicate), so
        # compression needs no extra extraction pass yet.
        return ""

    # -- Session bookkeeping --------------------------------------------------

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        old = self._session_id
        self._session_id = new_session_id
        if reset:
            with self._lock:
                for sid in (old, new_session_id, ""):
                    self._turn_index.pop(sid, None)
                    self._prefetched.pop(sid, None)
                    self._prefetch_gen.pop(sid, None)
                    self._prefetch_threads.pop(sid, None)
            self._last_injected_count = None

    _MAX_SESSION_STATES = 64

    def _prune_session_dicts_locked(self) -> None:
        """Bound the per-session dicts on long-running multi-session hosts
        (insertion-ordered: oldest sessions evict first). Caller holds
        the lock."""
        for d in (
            self._turn_index,
            self._prefetched,
            self._prefetch_gen,
            self._prefetch_threads,
        ):
            while len(d) > self._MAX_SESSION_STATES:
                d.pop(next(iter(d)))

    def shutdown(self) -> None:
        # Drain EVERY in-flight capture (not just the newest) before the
        # shared client closes underneath them; one overall time budget.
        # The flag closes the race with a sync_turn arriving mid-drain.
        import time as _time

        self._shutting_down = True
        with self._lock:
            pending = [t for t in self._sync_threads if t.is_alive()]
            self._sync_threads = []
        deadline = _time.monotonic() + _SYNC_JOIN_TIMEOUT
        for t in pending:
            t.join(timeout=max(0.0, deadline - _time.monotonic()))
        if self._client is not None:
            self._client.close()
            self._client = None

    # -- Setup flow -----------------------------------------------------------

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "mode",
                "description": "Transport: 'remote' (mnemostack service) or 'local' (library + your Qdrant)",
                "required": True,
                "default": "remote",
                "choices": ["remote", "local"],
            },
            {
                "key": "base_url",
                "description": "Remote mode: base URL of the mnemostack service (e.g. https://memory.example.com)",
            },
            {
                "key": "api_key",
                "description": "Remote mode: service key (tenant + scopes are resolved from it)",
                "secret": True,
                "env_var": "MNEMOSTACK_API_KEY",
            },
            {
                "key": "qdrant_url",
                "description": "Local mode: Qdrant URL",
                "default": "http://localhost:6333",
            },
            {
                "key": "collection",
                "description": "Local mode: Qdrant collection name",
                "default": "hermes-memory",
            },
            {
                "key": "embedding_provider",
                "description": "Local mode: mnemostack embedding provider (ollama, gemini, huggingface)",
                "default": "ollama",
            },
            {
                "key": "embedding_model",
                "description": "Local mode: embedding model override (provider default when empty)",
            },
            {
                "key": "recall_limit",
                "description": "Memories injected per turn",
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
            {
                "key": "capture",
                "description": "Store user/assistant turns automatically",
                "type": "boolean",
                "default": True,
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        # Secrets (api_key) are routed to .env by the setup flow; only
        # non-secret keys reach the JSON file.
        clean = {k: v for k, v in values.items() if k != "api_key"}
        _save_config_file(clean, hermes_home)
