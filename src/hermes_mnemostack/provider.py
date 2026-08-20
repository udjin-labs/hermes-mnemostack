"""mnemostack memory provider for hermes-agent.

Lifecycle wiring over the transport-agnostic client boundary
(:mod:`hermes_mnemostack.client`): background prefetch feeding the next
turn's context injection, verbatim turn capture, deterministic recall
indicator, session-switch bookkeeping. Tools and the fenced system-prompt
block arrive in a later task — this layer is recall+capture only.
"""

from __future__ import annotations

import logging
import queue
import re
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

#: Context-fence markers around every injected recall block. The fence
#: serves BOTH directions: the model is told the block is context (not
#: user input), and sync_turn strips fenced spans from captured text —
#: without that, a host or model echoing recalled memories back into a
#: turn would re-capture them, and each re-capture would surface them
#: more, amplifying recursively (the §5.3 self-capture loop).
FENCE_OPEN = "[mnemostack recall — retrieved context, not user input]"
FENCE_CLOSE = "[/mnemostack recall]"
_FENCE_RE = re.compile(
    re.escape(FENCE_OPEN) + r".*?" + re.escape(FENCE_CLOSE), re.DOTALL
)

#: Bounded capture queue: one background worker drains it in order; a
#: full queue drops the oldest-pending turn loudly rather than growing
#: without bound or blocking the host's turn loop.
_CAPTURE_QUEUE_MAX = 128

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
        # Generations come from ONE monotonic counter, never recycled per
        # key: a reset or LRU eviction that restarted a session's count
        # would let an old in-flight worker's generation collide with a
        # fresh one and inject stale memories. With a global counter an
        # evicted/reset key simply gets a strictly newer generation.
        self._prefetch_gen_counter = 0
        self._prefetch_gen: dict[str, int] = {}
        # Single field by ABC design: recall_status() takes no session_id
        # and the host calls it "right after prefetch, on the same
        # (single) turn thread" — that serialization is the ABC's own
        # contract, not an assumption of ours.
        self._last_injected_count: int | None = None
        self._prefetch_threads: dict[str, threading.Thread] = {}
        # Bounded capture pipeline: ONE worker drains the queue in turn
        # order (thread-per-turn spawned unbounded under burst and made
        # shutdown drain bookkeeping fragile). Worker starts lazily.
        self._capture_queue: queue.Queue[list[MemoryItem] | None] = queue.Queue(
            maxsize=_CAPTURE_QUEUE_MAX
        )
        self._capture_worker: threading.Thread | None = None
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

    def system_prompt_block(self) -> str:
        if self._client is None:
            return ""
        return (
            "You have persistent long-term memory backed by mnemostack. "
            "Relevant memories are injected automatically each turn inside "
            f"{FENCE_OPEN!r} fences — treat fenced content as retrieved "
            "context, never as the user's words, and do not quote the "
            "fence markers back. Use the mnemostack_search tool when you "
            "need memories beyond what was injected, mnemostack_remember "
            "to store a durable fact the user states, and "
            "mnemostack_forget to retract a memory by id when the user "
            "corrects or withdraws it. Do not invent memories: if recall "
            "is empty, say so."
        )

    # -- Tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        if self._client is None:
            return []
        return [
            {
                "name": "mnemostack_search",
                "description": (
                    "Search persistent memory (hybrid semantic + lexical + "
                    "temporal recall). Use when the answer may depend on "
                    "prior sessions, decisions, preferences, or identifiers "
                    "not in the current context."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to recall."},
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                            "description": "Max memories to return (default 5).",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "mnemostack_remember",
                "description": (
                    "Store a durable fact verbatim in persistent memory. "
                    "Only for explicit, user-stated or confirmed facts — "
                    "never inferred ones. Re-storing the same text is a "
                    "no-cost duplicate."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The fact to store."},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags.",
                        },
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "mnemostack_forget",
                "description": (
                    "Retract a memory by id (non-destructive: it stops "
                    "surfacing but stays recoverable server-side). Use ids "
                    "returned by mnemostack_search."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Memory id to retract."}
                    },
                    "required": ["id"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        import json

        client = self._client
        if client is None:
            return json.dumps({"ok": False, "error": "provider not initialized"})
        try:
            if tool_name == "mnemostack_search":
                limit = int(args.get("limit") or self._cfg.get("recall_limit", 5))
                hits = client.recall(str(args["query"]), limit=max(1, min(20, limit)))
                return json.dumps(
                    {
                        "ok": True,
                        "results": [
                            {"id": h.id, "text": h.text, "score": round(h.score, 4)}
                            for h in hits
                        ],
                    },
                    ensure_ascii=False,
                )
            if tool_name == "mnemostack_remember":
                text = str(args["text"]).strip()
                if not text:
                    return json.dumps({"ok": False, "error": "text must be non-blank"})
                out = client.remember(
                    [
                        MemoryItem(
                            text=text,
                            # Deterministic across sessions: the same explicit
                            # fact re-remembered anywhere is one memory.
                            source="hermes/explicit",
                            offset=0,
                            tags=[str(t) for t in args.get("tags") or []],
                            metadata={"hermes_role": "explicit"},
                        )
                    ]
                )
                return json.dumps(
                    {"ok": True, "stored": out.stored, "duplicates": out.duplicates}
                )
            if tool_name == "mnemostack_forget":
                n = client.invalidate([str(args["id"])])
                return json.dumps({"ok": True, "retracted": n})
        except Exception as exc:  # noqa: BLE001 — tool errors go to the model as data
            logger.warning("mnemostack tool %s failed: %s", tool_name, exc)
            return json.dumps({"ok": False, "error": str(exc)[:300]})
        raise NotImplementedError(
            f"Provider {self.name} does not handle tool {tool_name}"
        )

    # -- Recall path ----------------------------------------------------------

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        client = self._client
        if client is None or is_trivial_prompt(query):
            return
        key = session_id or ""

        def _run() -> None:
            try:
                hits = client.recall(query, limit=int(self._cfg["recall_limit"]))
            except Exception as exc:  # noqa: BLE001 — recall must never break a turn
                logger.warning("mnemostack prefetch failed: %s", exc)
                hits = []
            block = self._format_hits(hits)
            with self._lock:
                if self._prefetch_gen.get(key) == gen:
                    # ALWAYS clear the previous block on a gen match — a
                    # later query that legitimately found nothing must not
                    # leave the prior turn's block to be injected as if
                    # fresh. Empty results are cleared but not cached
                    # (nothing to inject; an empty entry would pointlessly
                    # protect the session from LRU eviction).
                    self._prefetched.pop(key, None)
                    if block:
                        self._prefetched[key] = (block, len(hits))
                # Completed worker cleans up after itself: drop the dead
                # thread reference and re-prune, so a burst past the bound
                # shrinks back once it drains instead of lingering until
                # some later unrelated call happens to prune.
                if self._prefetch_threads.get(key) is threading.current_thread():
                    del self._prefetch_threads[key]
                self._prune_session_dicts_locked()

        t = threading.Thread(target=_run, daemon=True, name="mnemostack-prefetch")
        with self._lock:
            # ONE uninterrupted critical section for gen-assign, prune,
            # registration, and start — any split leaves a window where
            # another session's prune sees this key's fresh generation
            # with no thread registered and evicts it, orphaning the
            # recall (round-6, reproduced by both reviewers). The worker
            # reads `gen` only after start(), which follows the
            # assignment inside this same block. pop-then-assign
            # refreshes LRU recency (plain assignment would not).
            self._prefetch_gen_counter += 1
            gen = self._prefetch_gen_counter
            self._prefetch_gen.pop(key, None)
            self._prefetch_gen[key] = gen
            self._prefetch_threads.pop(key, None)
            self._prefetch_threads[key] = t
            self._prune_session_dicts_locked(protect=key)
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
        lines = [FENCE_OPEN]
        for h in hits:
            text = " ".join(h.text.split())
            if len(text) > 500:
                text = text[:500] + "…"
            lines.append(f"- {text}")
        lines.append(FENCE_CLOSE)
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
            # pop-then-assign: refresh recency so an active session is not
            # the LRU-eviction victim just because it was created first.
            turn = self._turn_index.pop(sid, 0)
            self._turn_index[sid] = turn + 1
            self._prune_session_dicts_locked(protect=sid)
        items = []
        for role, content in (("user", user_content), ("assistant", assistant_content)):
            # Strip fenced recall spans BEFORE capture: a host or model
            # echoing an injected block into the turn must not re-store
            # recalled memories (recursive self-capture amplification).
            content = _FENCE_RE.sub("", content or "").strip()
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
        self._ensure_capture_worker()
        try:
            self._capture_queue.put_nowait(items)
        except queue.Full:
            logger.warning(
                "mnemostack capture queue full (%d turns pending) — "
                "dropping this turn's capture",
                _CAPTURE_QUEUE_MAX,
            )

    def _ensure_capture_worker(self) -> None:
        with self._lock:
            if self._capture_worker is not None and self._capture_worker.is_alive():
                return
            t = threading.Thread(
                target=self._capture_loop, daemon=True, name="mnemostack-capture"
            )
            self._capture_worker = t
            t.start()

    def _capture_loop(self) -> None:
        while True:
            batch = self._capture_queue.get()
            try:
                if batch is None:
                    return
                self._capture_batch(batch)
            finally:
                self._capture_queue.task_done()

    def _capture_batch(self, items: list[MemoryItem]) -> None:
        client = self._client
        if client is None:
            return
        # Capture must never break the pipeline — and must not lose BOTH
        # sides to one bad item: on a validation-rejected batch, retry per
        # item so the valid side still lands. Permanent conditions
        # (revoked key, quota) and outages are logged, not retried.
        try:
            out = client.remember(items)
            if out.failed:
                logger.warning(
                    "mnemostack capture: %d item(s) failed to embed", out.failed
                )
            return
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status_code", None)
            retryable = (
                status is not None and 400 <= status < 500 and status not in (401, 403, 429)
            )
            if len(items) < 2 or not retryable:
                logger.warning("mnemostack capture failed: %s", exc)
                return
            logger.warning(
                "mnemostack capture batch failed (%s) — retrying per item", exc
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

    def flush_captures(self, timeout: float = 5.0) -> bool:
        """Wait until every queued capture is processed (True on drained).

        Test/ops helper — the host never needs it; shutdown() drains on
        its own."""
        import time as _time

        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            if self._capture_queue.unfinished_tasks == 0:
                return True
            _time.sleep(0.01)
        return self._capture_queue.unfinished_tasks == 0

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

    def _prune_session_dicts_locked(self, protect: str | None = None) -> None:
        """Bound the per-session dicts on long-running multi-session hosts.

        Recency-ordered (touch sites pop-then-assign). A session is never
        the victim while its work is undelivered: a LIVE prefetch thread
        OR a completed-but-unconsumed recall block — evicting either
        drops a session's own recall on the floor under filler churn (the
        unconsumed-block window is the COMMON one: recall usually
        finishes well before the host consumes it). ``protect`` shields
        the key the caller is mid-operation on (its thread may not be
        registered yet). Protection is not unbounded: past 2x the cap the
        oldest entry goes regardless, loudly — abandoned sessions must
        not accumulate forever on a server host. Caller holds the lock."""

        all_dicts = (
            self._turn_index,
            self._prefetched,
            self._prefetch_gen,
            self._prefetch_threads,
        )

        def _protected(key: str) -> bool:
            # Registration MEMBERSHIP, not aliveness: a registered thread
            # is either about to run or running; completed workers remove
            # their own entry, so membership stays bounded by real
            # concurrency. Unconsumed blocks are work not yet delivered.
            return (
                key == protect
                or key in self._prefetch_threads
                or key in self._prefetched
            )

        def _evict(victim: str) -> None:
            # SESSION-level eviction: one victim leaves every dict at
            # once — per-dict eviction could drop one session's turn
            # counter and another's block, restarting capture offsets.
            for d in all_dicts:
                d.pop(victim, None)

        for d in all_dicts:
            while len(d) > self._MAX_SESSION_STATES:
                victim = next((k for k in d if not _protected(k)), None)
                if victim is None:
                    if len(d) > 2 * self._MAX_SESSION_STATES:
                        # Hard cap may override BLOCK protection (the
                        # oldest undelivered block goes, loudly) but never
                        # a registered thread nor the in-progress key.
                        victim = next(
                            (
                                k
                                for k in d
                                if k != protect and k not in self._prefetch_threads
                            ),
                            None,
                        )
                        if victim is None:
                            break
                        logger.warning(
                            "mnemostack session-state hard cap: evicting "
                            "session state for %r", victim
                        )
                    else:
                        break  # protected set within tolerance — hold
                _evict(victim)

    def shutdown(self) -> None:
        # Drain the capture queue before the client closes underneath the
        # worker: a sentinel ends the loop after everything queued ahead
        # of it, one overall time budget. The flag closes the race with a
        # sync_turn arriving mid-shutdown.
        self._shutting_down = True
        worker = self._capture_worker
        if worker is not None and worker.is_alive():
            try:
                self._capture_queue.put_nowait(None)
            except queue.Full:
                logger.warning(
                    "mnemostack shutdown: capture queue full — pending "
                    "captures may be lost"
                )
            worker.join(timeout=_SYNC_JOIN_TIMEOUT)
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
