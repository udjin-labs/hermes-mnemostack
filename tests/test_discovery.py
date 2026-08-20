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


def test_entry_point_loads_an_abc_compliant_class(monkeypatch, tmp_path):
    """hermes-agent's loader accepts a MemoryProvider subclass constructible
    with zero args — pin exactly that shape."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("MNEMOSTACK_MODE", raising=False)
    loaded = _our_entry_point().load()
    assert isinstance(loaded, type) and issubclass(loaded, MemoryProvider)
    provider = loaded()
    assert provider.name == "mnemostack"
    # Contract: is_available makes no network calls; an unconfigured
    # provider is honestly unavailable and says why.
    assert provider.is_available() is False
    assert provider.unavailable_reason()


def test_lifecycle_is_callable(monkeypatch, tmp_path):
    import hermes_mnemostack.provider as pmod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "mnemostack.json").write_text('{"mode": "local"}')

    class _NullClient:
        def recall(self, query, *, limit=5, filters=None):
            return []

        def recall_detailed(self, query, *, limit=5, filters=None):
            from hermes_mnemostack.client import RecallOutcome

            return RecallOutcome()

        def remember(self, items):
            raise AssertionError("no capture expected in this test")

        def invalidate(self, ids):
            return 0

        def forget(self, ids):
            return 0

        def close(self):
            pass

    monkeypatch.setattr(pmod, "build_client", lambda cfg, scope=None: _NullClient())
    p = pmod.MnemostackProvider()
    assert p.is_available() is True
    p.initialize("sess-1", hermes_home=str(tmp_path), platform="cli")
    assert [t["name"] for t in p.get_tool_schemas()] == [
        "mnemostack_search", "mnemostack_remember", "mnemostack_forget",
    ]
    assert "mnemostack" in p.system_prompt_block()
    assert p.prefetch("query") == ""
    p.shutdown()
