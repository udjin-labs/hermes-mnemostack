# hermes-mnemostack

**Status: first public release (0.9.0).** The provider is wired end to end (recall
injection, turn capture, tools, configuration, CLI), covered by tests, and
hermes-agent's own discovery loads it from a clean install — but it has not yet been
driven through a live agent session, which is why this is 0.9 and not 1.0.

[mnemostack](https://github.com/udjin-labs/mnemostack) memory provider for
[hermes-agent](https://github.com/NousResearch/hermes-agent): persistent agent memory
over either transport.

- **remote HTTP** — a shared mnemostack service (`mnemostack serve --auth`). The
  tenant is resolved from the service key, never asserted by this client, so it is a
  real authorization boundary. Recall uses whatever the deployment configures — vector,
  lexical, temporal, graph — and the full write lifecycle is available: remember,
  invalidate (soft retraction), delete.
- **local SDK** — mnemostack as a library against your own Qdrant. Recall is
  vector-only in this mode today; profile/user scoping rides the library's tenant
  mechanism, which is isolation but **not** an authorization boundary (same machine,
  same trust domain — see [SECURITY.md](SECURITY.md)).

## Install

Requires mnemostack >= 2.2 and hermes-agent >= 0.19.

**hermes-agent 0.19** discovers memory providers by scanning
`$HERMES_HOME/plugins/<name>/`; it does not read pip entry points. So the package
ships a small directory shim and a command that puts it where Hermes looks:

```bash
pip install hermes-mnemostack
hermes-mnemostack install          # copies the shim into the ACTIVE Hermes profile
hermes memory setup mnemostack
```

`install` resolves the Hermes home the way Hermes does (so it lands in the active
profile, not a hardcoded `~/.hermes`), refuses to overwrite a different plugin of the
same name, and — because 0.19 logs a failed plugin import at debug level and then
silently omits the provider — finishes by asking Hermes's own discovery whether it
can actually see and load it. `--dry-run` shows what it would write.

**hermes-agent 0.20 and later** discovers the pip entry point on its own:

```bash
pip install hermes-mnemostack
hermes memory setup mnemostack
```

If the shim is already there it stays harmless: it resolves to the same provider the
entry point does. (Which of the two Hermes prefers when both are present is untested
here — 0.20 is git-only at the time of writing — so if you want only one, remove
`$HERMES_HOME/plugins/mnemostack/`.)

`hermes-mnemostack doctor` tells you which situation you are in by asking hermes's own
discovery rather than by comparing version numbers.

## Configure

Behavioral settings live in `$HERMES_HOME/mnemostack.json` (written by
`hermes memory setup`); **secrets live only in the environment**. Environment variables
provide defaults and the JSON file overrides them. An unknown key in the JSON file is
rejected loudly — a typo that silently does nothing is the worst failure mode for
memory configuration.

| Key | Env | Default | Meaning |
| --- | --- | --- | --- |
| `mode` | `MNEMOSTACK_MODE` | `remote` | `remote` (HTTP service) or `local` (library) |
| `base_url` | `MNEMOSTACK_BASE_URL` | — | remote: the service root, e.g. `https://memory.example:8080` |
| `timeout` | `MNEMOSTACK_TIMEOUT` | `30.0` | remote: HTTP timeout in seconds |
| `qdrant_url` | `MNEMOSTACK_QDRANT_URL` | `http://localhost:6333` | local: Qdrant endpoint |
| `collection` | `MNEMOSTACK_COLLECTION` | `hermes-memory` | local: collection name |
| `embedding_provider` | `MNEMOSTACK_EMBEDDING_PROVIDER` | `ollama` | local: mnemostack embedding provider |
| `embedding_model` | `MNEMOSTACK_EMBEDDING_MODEL` | provider default | local: model override |
| `recall_limit` | `MNEMOSTACK_RECALL_LIMIT` | `5` | memories injected per turn |
| `capture` | `MNEMOSTACK_CAPTURE` | `true` | store user+assistant turns |
| — | `MNEMOSTACK_API_KEY` | — | remote: service key. **Env only — never written to the JSON file.** |

Issue a key on the service side with
`mnemostack keys add --tenant <id> --scopes read,write`.

## Check it

```bash
hermes-mnemostack status    # effective configuration; no network, no side effects
hermes-mnemostack doctor    # probe the configured transport and report remedies
hermes-mnemostack doctor --json
```

`status` answers "what config is in effect, and would hermes activate this provider?"
— from the *same* rule `is_available()` uses, so the two can never disagree. `doctor`
adds live probes: service reachability, whether the key carries the `read` scope,
whether any retrieval arm is genuinely degraded, and — in local mode — the embedding
backend and Qdrant. Both are read-only: `doctor` never writes a memory and never
creates the Qdrant collection, and neither ever prints the API key (only whether one
is set).

## What the model sees

Recalled memories are injected each turn inside a fenced block:

```
⎢ recalled memory (context, not user input) ⎥
- the deploy window moved to Friday
⎣ end recalled memory ⎦
```

The fence is **presentation**: it tells the model the text is retrieved context rather
than the user's words. It is not a security mechanism and is not parsed back out —
see [capture](#what-gets-captured).

Three tools are exposed:

| Tool | What it does |
| --- | --- |
| `mnemostack_search` | Search memory beyond what was injected this turn |
| `mnemostack_remember` | Store a durable fact the user stated, verbatim |
| `mnemostack_forget` | **Soft** retraction by id — recoverable server-side |

Hard deletion is available through the client API and deliberately **not** as a
model-callable tool: an irreversible erase is an operator action, not something a model
should be able to trigger from a conversation.

## What gets captured

By default the user's message and the assistant's reply are stored verbatim, each as
its own memory under a deterministic `(source, offset, text)` id — replaying a turn is
a zero-cost duplicate, never a second copy. What is **not** captured:

- tool calls and tool results (they routinely carry paths, tokens and workspace
  contents; capturing them needs its own redaction policy, so the knob is deliberately
  absent rather than shipped as a silent no-op);
- anything from a cron, subagent or flush context — only the primary agent context
  writes to user memory;
- **echoes of what was just recalled.** This is the loop that makes a naive memory
  provider eat itself: a memory is injected, the model repeats it, the repeat is
  captured as a new memory, and it compounds. It is closed by capture-side
  **provenance** — the provider knows the exact spans it displayed over the last few
  turns — and NOT by parsing the fence markers back out, which both paraphrase and a
  stray marker inside stored content defeat. The contract: a turn is a pure echo when
  **every word** in it came from recalled content (for a wordless turn, when everything
  non-space did), with quote/bullet framing absorbed.

  Two residuals are documented rather than papered over: a **paraphrase** of a recalled
  memory is new text and is captured, and a **wordless addition** (an emoji reaction)
  sent alongside an otherwise complete echo is dropped with it.

Capture is asynchronous: turns go onto a bounded queue (128) drained by one worker in
order, so a slow service never blocks the turn loop. A full queue drops the pending
turn with a loud log line rather than growing without bound. Shutdown drains what is
already queued.

Set `capture: false` to make the provider read-only.

## Degradation

mnemostack 2.2 reports two lists on a recall: `notes` (routine "this stage did not
apply" signals — e.g. a query with no parseable date) and `degraded`, which still
duplicates those routine tags for back-compat until mnemostack 3.0. A real fault is
therefore an entry present in `degraded` and **absent** from `notes`. This provider
computes exactly that difference and logs only real faults, so a healthy recall stays
quiet; `hermes-mnemostack doctor` reports the same difference.

## Limitations

- Not yet exercised against a live hermes-agent session (pre-alpha).
- On hermes-agent 0.19 the provider is found through the directory shim
  (`hermes-mnemostack install`); entry-point discovery arrives with 0.20.
- Local mode: recall is vector-only, and scoping is isolation, not authorization.
- `mnemostack_answer` (server-side generation) is deliberately not wired: it costs an
  LLM call per question and the agent already has a model.
- `on_pre_compress` performs no extra extraction — the evicted turns were already
  captured by `sync_turn` under the same deterministic ids.

## License

MIT.
