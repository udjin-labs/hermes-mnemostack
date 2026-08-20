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

#: Context-fence markers around every injected recall block. Their job is
#: to TELL THE MODEL the block is retrieved context, not the user's
#: words — a presentation boundary, not a security or dedup mechanism.
#: The self-capture loop (§5.3) is closed by capture-side provenance
#: (see _recently_injected below), NOT by parsing these markers back out
#: of turn text: exact-fence stripping is defeated by paraphrase and by a
#: stray marker in stored content, so it is deliberately not relied on.
FENCE_OPEN = "\u23a2 recalled memory (context, not user input) \u23a5"
FENCE_CLOSE = "\u23a3 end recalled memory \u23a6"

#: How many TURNS an injected memory stays eligible for capture
#: suppression (age, not a count of distinct memories: an entry must not
#: linger for months and silently swallow a genuine re-assertion of the
#: same fact). Also caps the per-session set size.
_INJECTED_MEMORY_TURNS = 8
_INJECTED_MEMORY_MAX = 128
#: The bullet _format_hits renders before each memory. Echoed back, it is
#: OUR artifact, not the caller's content — removing exactly this token
#: (not a general punctuation taxonomy) keeps mixed echoes clean.
_BLOCK_BULLET = "-"


def _is_pure_echo(content: str, mask: bytearray) -> bool:
    """Whether every WORD in this turn came from recalled content.

    The contract is word coverage, deliberately NOT a taxonomy of
    "presentation" characters: any such list is endlessly incomplete in
    both directions (markdown wrappers keep appearing, and braces or
    brackets can be real content). If a turn contributes no words of its
    own beyond what was recalled, it is an echo however it was framed —
    quoted, bulleted, italicized, or bare.

    Documented residual: a WORDLESS addition (an emoji or punctuation)
    sent together with an otherwise complete echo is dropped with it.
    That is a bounded, one-turn loss of a low-signal reaction, and it is
    preferable to re-storing recalled content, which compounds.
    """
    words = [i for i, ch in enumerate(content) if ch.isalnum()]
    if not words:
        # No words at all: an echo only if literally everything non-space
        # was recalled (a stored "👍" re-sent verbatim).
        return all(
            mask[i] for i, ch in enumerate(content) if not ch.isspace()
        ) and any(not ch.isspace() for ch in content)
    return all(mask[i] for i in words)


#: A recalled span is only worth suppressing if it carries content; a
#: two-word fragment appearing inside a sentence is coincidence, not an
#: echo, and cutting it would mangle a legitimately new memory.
_MIN_SUPPRESSED_SPAN = 24

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
        self._prefetched: dict[str, tuple[str, int, tuple[str, ...]]] = {}
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
        # Capture-side provenance, PER SESSION (like every other prefetch
        # structure here): {session: {displayed_text: injected_at_turn}}.
        # An echo of these spans — whole-turn or embedded in framing text
        # — is removed before capture.
        self._recently_injected: dict[str, dict[str, int]] = {}
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
            "a bracketed 'recalled memory' header/footer — treat anything "
            "between those markers as retrieved context, never as the "
            "user's words, and do not repeat the markers back. Use the "
            "mnemostack_search tool when you need memories beyond what was "
            "injected, mnemostack_remember to store a durable fact the "
            "user states, and mnemostack_forget to retract a memory by id "
            "when the user corrects or withdraws it. Do not invent "
            "memories: if recall is empty, say so."
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
                query = str(args.get("query") or "").strip()
                if not query:
                    return json.dumps(
                        {"ok": False, "error": "missing required argument: query"}
                    )
                limit = int(args.get("limit") or self._cfg.get("recall_limit", 5))
                hits = client.recall(query, limit=max(1, min(20, limit)))
                # Tool results are shown to the model just like injected
                # blocks — same provenance treatment, or a fact the model
                # searched for and then stated gets re-captured.
                # The tool response must carry EXACTLY the text provenance
                # records, or an echo of what the model saw is not matched.
                shown = [(h, self._display_text(h.text)) for h in hits]
                if shown:
                    with self._lock:
                        self._note_injected_locked(
                            self._session_key(), tuple(t for _h, t in shown)
                        )
                return json.dumps(
                    {
                        "ok": True,
                        "results": [
                            {"id": h.id, "text": text, "score": round(h.score, 4)}
                            for h, text in shown
                        ],
                    },
                    ensure_ascii=False,
                )
            if tool_name == "mnemostack_remember":
                text = str(args.get("text") or "").strip()
                if not text:
                    return json.dumps(
                        {"ok": False, "error": "missing required argument: text"}
                    )
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
                mem_id = str(args.get("id") or "").strip()
                if not mem_id:
                    return json.dumps(
                        {"ok": False, "error": "missing required argument: id"}
                    )
                n = client.invalidate([mem_id])
                return json.dumps({"ok": True, "retracted": n})
        except Exception as exc:  # noqa: BLE001 — tool errors go to the model as data
            logger.warning("mnemostack tool %s failed: %s", tool_name, exc)
            return json.dumps({"ok": False, "error": str(exc)[:300]})
        raise NotImplementedError(
            f"Provider {self.name} does not handle tool {tool_name}"
        )

    # -- Recall path ----------------------------------------------------------

    def _session_key(self, session_id: str = "") -> str:
        """One session identity for ALL per-session state. prefetch() and
        sync_turn() must resolve the same key or provenance, turn
        numbering, and caches silently belong to different sessions."""
        return session_id or self._session_id or ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        client = self._client
        if client is None or is_trivial_prompt(query):
            return
        key = self._session_key(session_id)

        def _run() -> None:
            try:
                outcome = client.recall_detailed(
                    query, limit=int(self._cfg["recall_limit"])
                )
                hits = outcome.hits
                if outcome.faults:
                    # `degraded` still duplicates routine `notes` tags for
                    # back-compat until mnemostack 3.0 — only entries
                    # ABSENT from notes are real breakage. Logging the raw
                    # degraded list would flag healthy recalls.
                    logger.warning(
                        "mnemostack recall degraded: %s", ", ".join(outcome.faults)
                    )
            except Exception as exc:  # noqa: BLE001 — recall must never break a turn
                logger.warning("mnemostack prefetch failed: %s", exc)
                hits = []
            block = self._format_hits(hits)
            norm_texts = tuple(self._display_text(h.text) for h in hits)
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
                        self._prefetched[key] = (block, len(hits), norm_texts)
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
        key = self._session_key(session_id)
        with self._lock:
            t = self._prefetch_threads.get(key)
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        with self._lock:
            block, count, texts = self._prefetched.pop(key, ("", 0, ()))
            if texts:
                self._note_injected_locked(key, texts)
        self._last_injected_count = count if block else None
        return block

    def _note_injected_locked(self, key: str, texts: tuple[str, ...]) -> None:
        """Record spans shown to the model, stamped with the session's
        current turn so they expire by AGE. Caller holds the lock."""
        seen = self._recently_injected.setdefault(key, {})
        turn = self._turn_index.get(key, 0)
        for text in texts:
            if not text:
                # A memory made only of fence glyphs sanitizes to "" — an
                # empty span matches everywhere and advances no scan.
                continue
            seen.pop(text, None)
            seen[text] = turn
        while len(seen) > _INJECTED_MEMORY_MAX:
            seen.pop(next(iter(seen)))
        # ONE eviction policy for session state: the shared prune, which
        # protects sessions with undelivered work and evicts a victim from
        # every dict at once. A second local loop here would evict
        # protected sessions and desync them across dicts.
        self._prune_session_dicts_locked(protect=key)

    def _strip_injected(self, content: str, key: str) -> str:
        """Remove spans this session was recently shown from turn text.

        Containment, not whole-turn equality: the realistic echo is a
        recalled fact embedded in ordinary framing ("sure — <fact>, got
        it"), or the whole injected block quoted back. Both must stop
        being re-stored, or each cycle re-surfaces the memory and
        amplifies it. Residual (documented): a PARAPHRASE is not matched
        — verbatim v1 capture has no semantic dedup, and that cost is
        linear, not recursive."""
        with self._lock:
            seen = dict(self._recently_injected.get(key, {}))
            turn = self._turn_index.get(key, 0)
        if not seen:
            return content
        fresh = [
            text
            for text, injected_turn in seen.items()
            if text and turn - injected_turn <= _INJECTED_MEMORY_TURNS
        ]
        if not fresh:
            return content
        # Spans are located in the ORIGINAL text and masked — never by
        # rewriting the output in place: replacing one span can splice its
        # neighbours into a byte sequence that equals ANOTHER tracked
        # span, which would then be cut although the user never wrote it.
        mask = bytearray(len(content))

        def _mask(span: str) -> bool:
            if not span:
                return False  # an empty span never advances the scan
            hit = False
            start_at = 0
            while True:
                i = content.find(span, start_at)
                if i < 0:
                    return hit
                for j in range(i, i + len(span)):
                    mask[j] = 1
                hit = True
                start_at = i + len(span)

        def _residual(m: bytearray) -> str:
            # Masked runs become a SPACE, never nothing: deleting them
            # outright would fuse the surrounding words into one token.
            out_parts: list[str] = []
            prev_masked = False
            for ch, masked in zip(content, m, strict=True):
                if masked:
                    prev_masked = True
                    continue
                if prev_masked:
                    out_parts.append(" ")
                    prev_masked = False
                out_parts.append(ch)
            return "".join(out_parts)

        long_spans = sorted(
            (t for t in fresh if len(t) >= _MIN_SUPPRESSED_SPAN), key=len, reverse=True
        )
        short_spans = sorted(
            (t for t in fresh if len(t) < _MIN_SUPPRESSED_SPAN), key=len, reverse=True
        )
        removed = False
        for text in long_spans:
            removed |= _mask(text)

        # Short spans are NOT cut out of real content (a few words
        # appearing mid-sentence is coincidence, not an echo) — but a turn
        # whose ENTIRE remainder is short recalled spans is a pure echo,
        # so they are evaluated against the post-mask residual, not the
        # raw text (a long span alongside a short one used to leak).
        # Fence markers are presentation-only; if a whole block bounced
        # back, its markers are noise — and their presence is what
        # distinguishes BLOCK artifacts (list bullets) from a user's own
        # wordless reaction.
        fence_echoed = False
        for marker in (FENCE_OPEN, FENCE_CLOSE):
            fence_echoed |= _mask(marker)
        removed |= fence_echoed

        # Short spans are NOT cut out of real content (a few words
        # appearing mid-sentence is coincidence, not an echo) — but a turn
        # whose ENTIRE remainder is short recalled spans is a pure echo,
        # so they are evaluated against the post-mask residual.
        if short_spans:
            probe = bytearray(mask)
            matched_short = False
            for text in short_spans:
                if not text:
                    continue
                start_at = 0
                while True:
                    i = content.find(text, start_at)
                    if i < 0:
                        break
                    for j in range(i, i + len(text)):
                        probe[j] = 1
                    matched_short = True
                    start_at = i + len(text)
            if (removed or matched_short) and _is_pure_echo(content, probe):
                return ""

        if not removed:
            # Nothing was an echo — capture stays VERBATIM. (Normalizing
            # unconditionally would flatten every multi-line message, code
            # block and table, and would drop punctuation/emoji-only turns
            # for 8 turns after any recall.)
            return content
        if _is_pure_echo(content, mask):
            return ""
        tokens = _residual(mask).split()
        if fence_echoed:
            # A block was echoed: its bullets are ours, and only they are
            # dropped — any other token is the caller's own content.
            tokens = [t for t in tokens if t != _BLOCK_BULLET]
        out = " ".join(tokens)
        if not out:
            return ""
        logger.debug(
            "mnemostack capture: removed recalled span(s) from a %s-char turn",
            len(content),
        )
        return out

    def recall_status(self) -> RecallStatus | None:
        if self._last_injected_count is None:
            return None
        return RecallStatus(
            provider_label="mnemostack",
            count=self._last_injected_count,
            glyph=GLYPH,
        )

    @staticmethod
    def _display_text(raw: str) -> str:
        """Exactly what the model sees for one memory — the same string
        provenance tracks, so an echo of the DISPLAYED text is matched
        (tracking the raw text would miss sanitized/truncated ones)."""
        text = " ".join(raw.split())
        # A stored memory could itself contain the marker glyphs (via
        # capture or remember) — neutralize them so recalled content
        # can't forge or prematurely close the presentation fence.
        text = text.replace(FENCE_OPEN, "").replace(FENCE_CLOSE, "")
        if len(text) > 500:
            text = text[:500] + "…"
        return text

    @classmethod
    def _format_hits(cls, hits: list[Any]) -> str:
        if not hits:
            return ""
        lines = [FENCE_OPEN]
        for h in hits:
            lines.append(f"{_BLOCK_BULLET} {cls._display_text(h.text)}")
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
        sid = self._session_key(session_id)
        with self._lock:
            # pop-then-assign: refresh recency so an active session is not
            # the LRU-eviction victim just because it was created first.
            turn = self._turn_index.pop(sid, 0)
            self._turn_index[sid] = turn + 1
            self._prune_session_dicts_locked(protect=sid)
        items = []
        for role, content in (("user", user_content), ("assistant", assistant_content)):
            content = (content or "").strip()
            if not content:
                continue
            # Provenance-based self-capture suppression (§5.3).
            content = self._strip_injected(content, sid).strip()
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
        with self._lock:
            if self._shutting_down:
                return  # shutdown won the race — do not enqueue
            try:
                self._capture_queue.put_nowait(items)
            except queue.Full:
                logger.warning(
                    "mnemostack capture queue full (%d turns pending) — "
                    "dropping this turn's capture",
                    self._capture_queue.maxsize,
                )

    def _ensure_capture_worker(self) -> None:
        with self._lock:
            if self._shutting_down:
                return  # never resurrect a worker after shutdown started
            if self._capture_worker is not None and self._capture_worker.is_alive():
                return
            t = threading.Thread(
                target=self._capture_loop, daemon=True, name="mnemostack-capture"
            )
            self._capture_worker = t
            t.start()

    def _capture_loop(self) -> None:
        while True:
            try:
                batch = self._capture_queue.get(timeout=0.2)
            except queue.Empty:
                # Woke with nothing pending: honor a shutdown that could
                # not deliver a sentinel into a (transiently) full queue.
                if self._shutting_down:
                    return
                continue
            try:
                if batch is None:
                    return
                try:
                    self._capture_batch(batch)
                except Exception as exc:  # noqa: BLE001 — worker must survive
                    logger.warning("mnemostack capture batch crashed: %s", exc)
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
                    self._recently_injected.pop(sid, None)
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
            self._recently_injected,
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
        # Drain queued captures before the client closes under the worker.
        # The flag (set under the lock, so it strictly orders against
        # sync_turn's own locked enqueue check) stops new work; the worker
        # is told to stop by the flag AND a sentinel — but a FULL queue
        # can reject the sentinel, so the worker also checks the flag when
        # it wakes with an empty queue, and we bound the whole wait.
        import time as _time

        with self._lock:
            self._shutting_down = True
            worker = self._capture_worker
        if worker is not None and worker.is_alive():
            deadline = _time.monotonic() + _SYNC_JOIN_TIMEOUT
            # Best-effort sentinel: if the queue has room it ends the loop
            # promptly after pending items; if it's full, the flag-poll
            # below is the fallback.
            try:
                self._capture_queue.put_nowait(None)
            except queue.Full:
                pass
            worker.join(timeout=max(0.0, deadline - _time.monotonic()))
            if worker.is_alive():
                logger.warning(
                    "mnemostack shutdown: capture worker did not drain within "
                    "%.0fs — %d turn(s) may be unsaved",
                    _SYNC_JOIN_TIMEOUT,
                    self._capture_queue.unfinished_tasks,
                )
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
