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
        self._turn_index = 0
        self._lock = threading.Lock()
        self._prefetched: str = ""
        self._prefetched_count = 0
        self._last_injected_count: int | None = None
        self._prefetch_thread: threading.Thread | None = None
        self._sync_thread: threading.Thread | None = None

    # -- Required ABC surface -------------------------------------------------

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        # Contract: config and deps only, no network.
        return is_configured()

    def unavailable_reason(self) -> str:
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

        def _run() -> None:
            try:
                hits = client.recall(query, limit=int(self._cfg["recall_limit"]))
            except Exception as exc:  # noqa: BLE001 — recall must never break a turn
                logger.warning("mnemostack prefetch failed: %s", exc)
                hits = []
            block = self._format_hits(hits)
            with self._lock:
                self._prefetched = block
                self._prefetched_count = len(hits)

        t = threading.Thread(target=_run, daemon=True, name="mnemostack-prefetch")
        with self._lock:
            self._prefetch_thread = t
        t.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        # Serve the cached background result; if the thread is still in
        # flight give it a short grace window rather than blocking a turn.
        t = self._prefetch_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        with self._lock:
            block, count = self._prefetched, self._prefetched_count
            self._prefetched, self._prefetched_count = "", 0
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
        if self._agent_context != "primary":
            # Cron/subagent/flush contexts must not pollute user memory.
            return
        turn = self._turn_index
        self._turn_index += 1
        sid = session_id or self._session_id
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
            try:
                client.remember(items)
            except Exception as exc:  # noqa: BLE001 — capture must never break a turn
                logger.warning("mnemostack sync_turn failed: %s", exc)

        t = threading.Thread(target=_sync, daemon=True, name="mnemostack-sync")
        self._sync_thread = t
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
        self._session_id = new_session_id
        if reset:
            self._turn_index = 0
            with self._lock:
                self._prefetched, self._prefetched_count = "", 0
            self._last_injected_count = None

    def shutdown(self) -> None:
        t = self._sync_thread
        if t is not None and t.is_alive():
            t.join(timeout=_SYNC_JOIN_TIMEOUT)
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
