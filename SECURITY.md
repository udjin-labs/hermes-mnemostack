# Security

`hermes-mnemostack` moves conversation content into durable storage. That is the whole
point of it, and it is also the whole risk: what would otherwise vanish with the
session is written down, indexed, and read back into a later prompt. This document is
what an operator needs in order to decide whether that trade is acceptable in their
deployment.

## What leaves the machine

| Mode | What is sent, and where |
| --- | --- |
| `remote` | Every captured turn (user + assistant text, verbatim), every recall query, and any text passed to `mnemostack_remember` go to the configured `base_url` over HTTP. Embedding happens on the SERVER — the client needs no provider key. |
| `local` | Nothing leaves the host except the calls mnemostack itself makes: text to the configured embedding provider (a local Ollama by default, a vendor API if you configure one) and vectors/payloads to your Qdrant. |

In remote mode, `base_url` should be `https://` on anything but a loopback address:
the payload is the conversation.

**Not** sent in either mode: tool calls and tool results, and anything from a cron,
subagent or flush context. Turn capture can be switched off entirely with
`capture: false`.

## Secrets

`MNEMOSTACK_API_KEY` is read from the environment only. It is never written to
`$HERMES_HOME/mnemostack.json` — `save_config()` rejects any key outside the non-secret
set — and it is never printed by `hermes-mnemostack status` / `doctor`, which report
only whether a key is set (a pin in the test suite asserts the literal key value never
reaches stdout). `doctor --json` follows the same rule.

The key is a mnemostack service key. Scope it to what the agent actually needs
(`--scopes read,write`) and rotate it with `mnemostack keys revoke` / `keys add`;
revocation takes effect on the next call with the file keystore.

## Trust boundaries

- **remote mode is an authorization boundary.** The tenant is resolved server-side from
  the service key; this client cannot assert a tenant, and a compromised agent host can
  reach only its own tenant's memories.
- **local mode is NOT.** Profile/user scoping rides mnemostack's tenant mechanism —
  deterministic per-scope ids, `tenant_id` stamps, filtered recall — which cleanly
  isolates profiles from each other in normal operation. But anything with library
  access to the same Qdrant can pass a different tenant. Treat local mode as one trust
  domain: use it for a single user's own machine, and use remote mode with service keys
  when isolation has to hold against an adversary.

## Prompt-injection posture

Recalled memories are injected inside a fenced block labelled as retrieved context.
The fence is **presentation for the model, not a security control**: stored text can
contain the marker strings, and no parsing of them is relied on anywhere. Memory
content is model input, and model input is untrusted — a memory written in an earlier
session can carry instructions. Two consequences worth stating plainly:

- Anything that can write to the memory store can influence future sessions of every
  agent reading it. In remote mode that set is "whoever holds a `write`-scoped key for
  the tenant"; in local mode it is "whoever can reach the Qdrant".
- The provider does not filter or sanitize memory text. If your deployment needs that,
  it belongs in front of the store, not here.

## Self-capture

A memory provider that captures its own injected memories compounds them. This one
closes that loop with capture-side provenance — the exact spans it displayed in recent
turns — and a word-coverage rule: a turn whose every word came from recalled content is
an echo and is not stored. Two residuals are known and documented in the README: a
paraphrase of a recalled memory is new text and IS captured, and a wordless addition
(an emoji) sent alongside a complete echo is dropped with it. Neither is a data leak;
the first is a slow-growth risk worth watching in long-lived deployments.

## Erasure

- `mnemostack_forget` (the model-facing tool) is a **soft** retraction: the memory drops
  out of default recall but stays recoverable server-side. This is deliberate — a model
  should not be able to destroy data from a conversation.
- Hard deletion is available through the client API (`forget()` → `DELETE /memories`),
  and through mnemostack's own operator surfaces. For a right-to-erasure request, follow
  mnemostack's erasure notes in its `docs/deployment.md` — in particular, an in-process
  BM25 corpus is a startup snapshot and can keep serving deleted text until the service
  restarts.
- Captured turns carry a stable source (`hermes/{platform}/{session}`), so an entire
  session can be retracted or erased by source through mnemostack's source-scoped
  lifecycle endpoints.

## Reporting a vulnerability

Open a GitHub security advisory on
[udjin-labs/hermes-mnemostack](https://github.com/udjin-labs/hermes-mnemostack), or an
issue if the problem is not sensitive. Please do not include real conversation content,
keys, or internal hostnames in a public report.
