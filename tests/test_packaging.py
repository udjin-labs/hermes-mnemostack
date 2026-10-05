"""The host is supplied by Hermes, or requested explicitly for a pip install."""

from __future__ import annotations

import importlib.metadata

from packaging.requirements import Requirement


def test_host_dependency_is_opt_in():
    requirements = [
        Requirement(spec)
        for spec in importlib.metadata.requires("hermes-mnemostack") or []
    ]
    default = [
        req.name for req in requirements if req.marker is None or req.marker.evaluate({"extra": ""})
    ]
    assert "mnemostack" in default
    assert "hermes-agent" not in default
    standalone = [
        req for req in requirements if req.marker is None or req.marker.evaluate({"extra": "hermes"})
    ]
    host = next(req for req in standalone if req.name == "hermes-agent")
    assert "0.19" in host.specifier
    assert "0.18" not in host.specifier
