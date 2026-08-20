# hermes-mnemostack

**Status: pre-alpha.** The provider is wired end to end (recall injection, turn capture, tools, configuration) and covered by tests, but it has not been run against a live hermes-agent session yet and the entry-point discovery it relies on needs hermes-agent >= 0.20 (see below).

[mnemostack](https://github.com/udjin-labs/mnemostack) memory provider for
[hermes-agent](https://github.com/NousResearch/hermes-agent): persistent
agent memory over either transport:

- **remote HTTP** — a shared mnemostack service (the tenant is resolved
  from the service key). Recall uses whatever the deployment configures —
  vector, lexical, temporal, graph — and the full write lifecycle is
  available: remember, invalidate (soft retraction), delete.
- **local SDK** — mnemostack as a library against your own Qdrant. Recall
  is vector-only in this mode today; profile/user scoping rides the
  library's tenant mechanism.

The tools exposed to the model are `mnemostack_search`,
`mnemostack_remember` and `mnemostack_forget` (soft retraction —
recoverable server-side). Hard deletion is available through the client
API, deliberately not as a model-callable tool.

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
