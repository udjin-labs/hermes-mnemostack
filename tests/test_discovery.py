"""Task-1 pin: the provider is discoverable exactly the way hermes-agent
loads pip providers — entry point group, ABC compliance, zero-arg
construction."""

from __future__ import annotations

import importlib.metadata

from agent.memory_provider import MemoryProvider

ENTRY_POINTS_GROUP = "hermes_agent.memory_providers"


def _our_entry_point():
    eps = importlib.metadata.entry_points()
    group = eps.select(group=ENTRY_POINTS_GROUP)
    matches = [ep for ep in group if ep.name == "mnemostack"]
    assert len(matches) == 1, f"expected exactly one 'mnemostack' entry point, got {matches}"
    return matches[0]


def test_entry_point_registered():
    ep = _our_entry_point()
    assert ep.value == "hermes_mnemostack.provider:MnemostackProvider"


def test_entry_point_loads_an_abc_compliant_class():
    """hermes-agent's loader accepts a MemoryProvider subclass constructible
    with zero args — pin exactly that shape."""
    loaded = _our_entry_point().load()
    assert isinstance(loaded, type) and issubclass(loaded, MemoryProvider)
    provider = loaded()
    assert provider.name == "mnemostack"
    # Contract: is_available makes no network calls; skeleton is honest
    # about being unconfigured and says why.
    assert provider.is_available() is False
    assert provider.unavailable_reason()


def test_lifecycle_skeleton_is_callable():
    from hermes_mnemostack.provider import MnemostackProvider

    p = MnemostackProvider()
    p.initialize("sess-1", hermes_home="/tmp/hh", platform="cli")
    assert p.get_tool_schemas() == []
    assert p.system_prompt_block() == ""
    assert p.prefetch("query") == ""
    p.shutdown()
