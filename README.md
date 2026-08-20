# hermes-mnemostack

**Status: pre-alpha — skeleton only, not yet usable.**

[mnemostack](https://github.com/udjin-labs/mnemostack) memory provider for
[hermes-agent](https://github.com/NousResearch/hermes-agent): hybrid
semantic + BM25 + temporal + graph recall as the agent's persistent memory,
over either transport:

- **local SDK** — mnemostack as a library against your own Qdrant/Memgraph;
- **remote HTTP** — a shared multi-tenant mnemostack service (the tenant is
  resolved from the service key), full read+write lifecycle: recall,
  remember, invalidate, erase.

## Install (once released)

```bash
pip install hermes-mnemostack
hermes memory setup   # select "mnemostack"
```

Requires mnemostack >= 2.2 (on PyPI) and hermes-agent >= 0.19.

The provider registers through the `hermes_agent.memory_providers` entry
point; hermes-agent discovers it automatically once the package is installed
in the same environment. Note: pip entry-point discovery requires
hermes-agent ≥ 0.20 (not yet on PyPI); on 0.19 the provider can instead be
dropped into `$HERMES_HOME/plugins/mnemostack/` as a directory plugin.

## License

MIT.
