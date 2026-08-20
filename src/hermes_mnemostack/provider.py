"""mnemostack memory provider for hermes-agent.

Task-1 skeleton: the provider class exists, registers through the
``hermes_agent.memory_providers`` entry point, and reports itself
unavailable until configuration lands in a later task. No recall, no
capture, no tools yet — discovery and lifecycle wiring only.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

PROVIDER_NAME = "mnemostack"


class MnemostackProvider(MemoryProvider):
    """Persistent agent memory backed by a mnemostack deployment.

    Two transports behind one boundary (selected by configuration):
    local SDK (mnemostack as a library against the agent's own Qdrant)
    or remote HTTP (a shared mnemostack service; the tenant is resolved
    from the service key). Both cover the full read+write lifecycle:
    recall, remember, invalidate.
    """

    def __init__(self) -> None:
        self._session_id: str = ""
        self._hermes_home: str = ""
        self._platform: str = ""

    # -- Required ABC surface -------------------------------------------------

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        # Contract: no network calls here — config and deps only.
        # Configuration loading arrives in a later task; until then the
        # provider is discoverable but honestly unavailable.
        return False

    def unavailable_reason(self) -> str:
        return (
            "hermes-mnemostack is a pre-alpha skeleton — configuration "
            "support has not landed yet"
        )

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._session_id = session_id
        self._hermes_home = str(kwargs.get("hermes_home", ""))
        self._platform = str(kwargs.get("platform", ""))
        logger.debug(
            "mnemostack provider initialized (session=%s, platform=%s)",
            session_id,
            self._platform,
        )

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return []

    def shutdown(self) -> None:
        pass
