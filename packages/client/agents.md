# Agent Guide — `launchdarkly-ai-server` (Core Client)

This document describes what the core client package owns, what it exports, and what invariants agents must respect when modifying it or reading its contracts to implement handler packages.

---

## Role

This is **Tier 0** — the foundation. It owns:
- The LaunchDarkly client singleton and lifecycle
- The telemetry pipeline (OTel via `opentelemetry-sdk` + OTLP HTTP exporter)
- All shared Python types (`AiConfigRep`, `ProviderHandler`, etc.)
- The primary runtime entry point: `config()`
- Utility helpers: `parse_template`, `parse_json_with_possible_fences`

No other `launchdarkly-ai-*` package may define or duplicate these. They import from here.

---

## File Map

| File | Responsibility |
|---|---|
| `src/launchdarkly_ai_server/conversation.py` | `conversation_id`, `ConversationIdSpanProcessor` — stamps `gen_ai.conversation.id` |
| `src/launchdarkly_ai_server/sdk_info.py` | `$ld:ai:sdk:info` package registry and flush |
| `src/launchdarkly_ai_server/lifecycle.py` | `init_client`, `get_client`, `shutdown`, `extract_variation` |
| `src/launchdarkly_ai_server/client.py` | `config()`, `ConfigInstance` |
| `src/launchdarkly_ai_server/tracking.py` | `execute_and_track`, `execute_and_stream`, `wrap_tool_handlers`, `parse_usage` |
| `src/launchdarkly_ai_server/graph.py` | `graph()`, `resolve_graph()`, `GraphInstance` |
| `src/launchdarkly_ai_server/types.py` | All shared Python types — `AiConfigRep`, `ProviderHandler`, `LDContext`, `NativeTool`, etc. |
| `src/launchdarkly_ai_server/types_validation.py` | `parse_ai_config` — validates flag variation shape; `is_valid_skill_key` / `is_valid_skill_version` / `skill_key_rejection_reason` (the canonical key-grammar explanation every layer quotes) |
| `src/launchdarkly_ai_server/skills.py` | Agent Skills, retrieval half — `skill_refs`, `get_skill`/`get_skills`/`all_skills`, `InMemorySkillStore`, and the store/telemetry injection points `_set_store` / `_set_emitter_for_testing` |
| `src/launchdarkly_ai_server/skills_core.py` | Shared skills internals — the `SkillStore` seam, module state, the telemetry seam and its three recorders, integrity verification, and store resolution. Imported by both `skills.py` and the materialization layer; imports neither |
| `src/launchdarkly_ai_server/skills_fdv2.py` | Agent Skills, delivery transport — the FDv2 protocol, the wire-key/`version` translation, the held object set, and `FDv2SkillStore`. Sits **below** the store interface; imports `skills_core` only, and nothing imports it |
| `src/launchdarkly_ai_server/skills_watch.py` | Agent Skills, eager re-reconcile — `watch_skills` / `SkillWatcher`, wiring the store's change listener to `write_skills`. Sits **above** `skills_fs` and modifies none of it |
| `src/launchdarkly_ai_server/skills_fs.py` | Agent Skills, materialization half — `write_skills`, request resolution, the manifest format and on-disk filenames, per-skill reconcile, and pruning |
| `src/launchdarkly_ai_server/safe_fs.py` | Descriptor-pinned filesystem primitives — `atomic_write`, `unlink_file`, `pinned_directory`, `open_directory_nofollow`, `open_or_create_directory`, `SymlinkRefused`, `DirectoryMissing`, and the `*at()` capability probe. Owns the descriptor-vs-path platform split; knows nothing about skills |
| `src/launchdarkly_ai_server/utils.py` | `parse_template`, `parse_json_with_possible_fences`, `create_handler`, `parse_usage`, `make_track_data`, `to_ld_context` |
| `src/launchdarkly_ai_server/registry.py` | `Registry`, `global_registry`, `compose`, `resolve_handlers`, `resolve_tools` |
| `src/launchdarkly_ai_server/judges.py` | `run_judges`, `build_judge_tasks`, `run_judge` |
| `src/launchdarkly_ai_server/evaluations/` | `init_evaluations`, the private management API operations, and generation-only `EvaluationsModule.run()` orchestration |
| `src/launchdarkly_ai_server/__init__.py` | Public barrel — the only surface handler packages import from |

---

## Public Exports

Key symbols exported from `launchdarkly_ai_server`:

```python
# Lifecycle
from launchdarkly_ai_server import init_client, get_client, shutdown, extract_variation, register_ai_sdk_package
from launchdarkly_ai_server import conversation_id, set_conversation_id_if_absent, ConversationIdSpanProcessor

# Types
from launchdarkly_ai_server import (
    AiConfigRep, ProviderHandler, ProviderResponse, ProviderGraphResponse,
    LDContext, LDClientInterface,
    NativeTool, NATIVE_TOOL_KEY,
    GraphDefinition, GraphNode, GraphEdge, GraphTopology, GraphOptions, GraphArgs,
    TrackData, UsageDict, HandlerResult, HandlerStreamEvent,
    StreamEvent, StreamChunkEvent, StreamDoneEvent, ExecuteStreamEvent, ExecuteStreamDoneEvent,
    VariationMeta, InitClientOptions, JudgeResult, ParseResult, ParseSuccess, ParseFailure,
)

# Utilities
from launchdarkly_ai_server import (
    parse_template, parse_json_with_possible_fences, create_handler,
    parse_usage, make_track_data, normalize_mode, to_ld_context, parse_ai_config,
)

# Registry
from launchdarkly_ai_server import Registry, global_registry, compose, resolve_handlers, resolve_tools

# Tracking
from launchdarkly_ai_server import execute_and_track, execute_and_stream, wrap_tool_handlers

# Entry points
from launchdarkly_ai_server import config, graph, resolve_graph, init_evaluations

# Agent Skills (experimental: none of these is exported from the package root)
from launchdarkly_ai_server.experimental.skills import (
    set_skill_store, SkillStore, InMemorySkillStore, FDv2SkillStore, StoreDiagnostics,
    skill_refs, get_skill, get_skill_result, get_skills, all_skills, write_skills,
    watch_skills, SkillWatcher,
    Skill, SkillReference, SkillOutcome, ReconcileAction, ReconcileReport,
    SKILL_FILENAME, MANIFEST_FILENAME, MANIFEST_VERSION,
    ReconcileActionKind, OnUnavailable, SkillOutcomeReason,  # the three closed-set unions
)
```

An experimental feature is exported only from its module under
`launchdarkly_ai_server.experimental`, and its names may change in a minor release. No core
public type, function or `init_client` option may name it. Core reaches it only through an
internal hook (as `shutdown` clears the skill store), and an error there is logged, never
raised into the core call. The experimental part is the SDK's API, not the LaunchDarkly data
model: `AiConfigRep` still documents the `skills` references a config carries.

`MAX_SKILL_CONTENT_BYTES` and `SKILL_OBJECT_KIND` are deliberately **not** exported; both stay
internal to `skills_core`:

- The size cap is a local bound on content the platform produces. Exporting it would
  semver-lock a number this SDK does not own.
- The kind is the SDK-side value handed to a store, not the wire contract, and publishing it
  would imply otherwise. An adapter that needs it imports it from
  `launchdarkly_ai_server.skills_core`.

When adding a new core export, add it to `__init__.py`'s imports and `__all__`. An Agent Skills export goes in `experimental/skills.py` instead, and in `EXPERIMENTAL_SKILLS_SURFACE` in `tests/test_skills.py`. Handler packages must never import from sub-paths (e.g. `launchdarkly_ai_server.client`).

---

## Key Types

### `ProviderHandler`

The callable type every handler package must produce. In Python it is created via `create_handler`:

```python
from launchdarkly_ai_server import create_handler

handler = create_handler(
    provides_for=("Anthropic", "messages"),   # (provider, mode) routing tuple
    call_impl=_call_impl,                      # async (config, user_input, tool_handlers, variables) → dict
    stream_impl=_stream_impl,                  # (config, user_input, tool_handlers, variables) → AsyncGenerator
)
```

- `provides_for` is the routing key for `config()`. It must match `config.provider.name` and `meta.mode` exactly.
- The callable signature is `(config, user_input, tool_handlers, variables) → Awaitable[dict]`.

### `AiConfigRep`

Validated by `parse_ai_config` in `extract_variation`. At least one of `instructions` or a non-empty `messages` list must be present. Do not relax this constraint.

### Token usage normalization

`execute_and_track` calls `parse_usage(response.usage)` which accepts any of these key variants:
- `input_tokens` / `output_tokens`
- `inputTokens` / `outputTokens`
- `input` / `output`

Handlers may return any of these — the client normalizes them before emitting LD telemetry events.

---

## `config()` Behavior

1. Accepts a `ProviderHandler` or list of `ProviderHandler`s plus a `key` and optional `tool_handlers`.
2. On `.invoke(user_input, context, variables?)`:
   a. Calls `extract_variation(key, context)` → validates the flag is enabled and parses `AiConfigRep`.
   b. Finds the handler whose `provides_for[0] == provider` and `provides_for[1] == normalized_mode`. Throws if no handler matches.
   c. Calls `execute_and_track(...)` which:
      - Records wall-clock duration, emits `$ld:ai:duration:total`
      - Calls `handler(config, user_input, tool_handlers, variables)`
      - On success: emits `$ld:ai:generation:success` + token tracks
      - On error: emits `$ld:ai:generation:error` then re-raises
3. If `judge_configuration.judges` is present, runs each judge handler (sampled by `sampling_rate`) against the primary response, tracks `evaluation_metric_key`, and emits a `gen_ai.evaluation.result` span event on the judge's `invoke_agent` span (`gen_ai.evaluation.name` / `.score.value` / `.explanation`).
4. Returns `ProviderResponse`: `{ response: str, usage: UsageDict, track_data: TrackData, judge_results?: dict[str, JudgeResult], judge_tasks?: list[JudgeTask] }`. `judge_results` is populated when `skip_judges=False` (default) and judges ran; `judge_tasks` is populated when `skip_judges=True`.

---

## SDK-run evaluations

`init_evaluations()` creates an evaluations harness using `LD_API_TOKEN` and the management API host `LD_API_BASE_URI`. Do not reuse `LD_BASE_URI`: that variable configures SDK delivery and may point at a relay proxy. Evaluation-run links use the separate `ui_base_uri` option, then `LD_UI_BASE_URI`, then `https://app.launchdarkly.com`; do not derive their host from `LD_API_BASE_URI`. An event transport is resolved in `init_evaluations()`, which raises before any network I/O when it finds neither an SDK key (`sdk_key` or `LD_SDK_KEY`) nor an already-initialized event-capable client: generation events are the only ingest path for row results, so a run without a transport could never complete. The lifecycle module's bring-your-own-client path (`init_client(client=...)`) therefore satisfies the check on its own, and `run()` reuses that singleton through `_resolve_client`; `run()` raises if the client disappears before it emits. Both polling arguments reject NaN, which would otherwise never compare past a deadline and hang the run. The harness always queues one `$ld:ai:offline-evals:generation` custom event per row through the standard SDK event transport and flushes before returning. No feature flag gates event emission. The harness polls the run summary endpoint until a nonzero `total_rows` has `pending_rows == 0` and `passed + failed + error` rows accounting for the total, polling every `poll_interval_seconds` (default 2s) until `poll_timeout_seconds` (default 180s); both are `run()` arguments so large datasets can widen them. The summary endpoint does not return run state, so `RunSummary` exposes row counts only.

`await EvaluationsModule.run(...)` takes `project_key` per call. Dataset lookup/row pagination, evaluation creation, and run creation are private helpers; only `run()` is public. Each call creates a new evaluation with `POST` and a run with `source="api"`, so its key must be unique. The harness directly invokes the supplied handler once per row and never retries it — event delivery is never a reason to rerun a handler because that would repeat tool side effects; retries apply only to management API requests. A 429 is replayed for any method, but 5xx responses and transport failures are replayed only for `GET`/`HEAD`, so an evaluation or run `POST` that the server may already have applied is never duplicated. Management API calls run in a worker thread (`asyncio.to_thread`) because the client is synchronous; the caller's event loop stays free. Generation events go through the already-initialized SDK client when the application has one — `init_client` is idempotent, so an existing singleton wins and the evaluations SDK key is ignored with a warning. Dataset-owned `input`, `expected_output`, `metadata`, and `variables` are deliberately excluded from the event payload. The harness flushes events, polls the run summary endpoint until row accounting is complete (`total_rows > 0`, `pending_rows == 0`, and `passed + failed + error == total_rows`), and raises a timeout once `poll_timeout_seconds` elapses if the backend never reaches one. `RunSummary` includes row counts only, and `EvalRunResult.passed` is true only when error and pending row counts are both zero.

---

## Conversation grouping

LaunchDarkly's conversation view groups spans on `gen_ai.conversation.id`. Bind a caller-supplied id around any `invoke()` / `stream()` / `graph().invoke()` call:

```python
from launchdarkly_ai_server import conversation_id, config

with conversation_id("thread-123"):
    await config(key=key, handler=handler).invoke(user_input, ctx)
```

`stream()` binds at call time rather than on first `__anext__`, so building the generator inside
the block and iterating it later — the normal shape for a chat app — keeps the id:

```python
with conversation_id("thread-123"):
    gen = config(key=key, handler=handler).stream(user_input, ctx)
async for event in gen:  # spans opened here still carry thread-123
    ...
```

Only the id is re-applied per step; the ambient context at iteration time is otherwise untouched,
so streaming span parenting is the same as it is with no id bound.

`init_client()` registers a span processor that stamps the id write-if-absent on every SDK span (root, chat, execute_tool, graph). The processor is registered on the *global* tracer provider, so it is scoped to spans from `@launchdarkly/ai-*` tracers only — a caller-supplied id must not land on third-party instrumentation spans (HTTP, Postgres, the outbound provider call). No id is invented when the caller supplies none — a UUID, a trace id, or a content hash would violate the semantic conventions.

This is an OTel context value, not W3C baggage, so the id does not leak onto outbound provider HTTP calls. A multi-tenant process must bind a different id per request; do not put it on the tracer resource.

---

## Agent Skills

Versioned `SKILL.md` documents attached to AI Config variations by reference, retrieved
through an injectable store, and materialized onto disk for agent runtimes to discover.
Three layers, in increasing order of blast radius:

1. **Reference discovery** — `skill_refs(config)` projects the config's `skills` array into
   typed `SkillReference` values. Pure: no network, no client, no store, no telemetry.
   Validation of the array itself lives in `parse_ai_config` and is **fail closed** — one
   malformed reference fails the whole config parse.
2. **Content accessors** — `get_skill`, `get_skill_result`, `get_skills`, `all_skills` read
   through the `SkillStore` seam. Configure a store with `set_skill_store(store)`; with
   none configured the accessors raise
   an actionable `RuntimeError`. A delivery transport can be added behind the seam
   without touching the public API.
3. **Materialization** — `write_skills(skills, root)` writes `<root>/<key>/SKILL.md` and
   reconciles against a manifest at `<root>/.launchdarkly-skills.json`.

### The store seam, and why version is part of the lookup

`SkillStore` is `get_object(kind, key, version=None)`, `all_objects(kind)`, an optional
`is_initialized()`, and an optional `add_listener(kind, fn)` / `remove_listener(kind, fn)` pair.

- **Version is part of the lookup identity**, not a filter on the answer. A delivery payload
  carries the newest version of every skill *plus* every version a variation pins, so two
  versions of one key routinely coexist. A key-only store would answer a pin with the newest
  object and turn a version-pinned attachment into a missing skill. `version=None` asks for
  the newest held.
- **The equality check in `resolve_from_store` is a defense**, not the selection mechanism:
  the store is untrusted, so an answer that is not the version asked for is withheld.
- **`all_objects` keys are opaque.** It returns one entry per `(key, version)`; do not parse
  the keys or assume one per skill key. Identity comes from each object's own `key` and
  `version`, which are revalidated.
- **`newest_by_key` is the one place that collapses to one object per key**, for both
  whole-store consumers: `all_skills`, and the `"*"` reconcile (`<root>/<key>/SKILL.md` is a
  single path). It keeps an object too malformed to carry a usable key and version, so
  verification withholds it and its key stays in the requested set, out of prune's reach —
  unless another version resolved that key.

### `is_initialized()` is what stands between a slow boot and deleting a customer's files

Through the store interface, "this environment holds no skills" and "delivery has not
answered yet" are the same empty answer, and `write_skills("*")` would read the first as
every skill revoked. So `_available_store` — the single gate that sets `unavailable` and
therefore suppresses pruning — consults `store_is_initialized`:

- Not yet initialized: retrieval is blocked instead of authorizing a prune.
- Probe not implemented: treated as initialized (right for a store populated by hand).
- Probe raises: treated as not initialized.

Keep this check in that one gate. A second copy that drifts from the first deletes the
user's files.

### The delivery transport, and the one field that will bite you

`FDv2SkillStore` speaks LaunchDarkly's SDK-facing FDv2 channel (`GET /sdk/poll`,
`GET /sdk/stream`, server-side SDK key in `Authorization`, `kinds` + `basis` params,
`If-None-Match`/304). It sits below the store interface and produces raw objects in the
shape `skills_core.SkillStore` documents; **nothing above that interface knows it exists**.
If a transport change seems to require editing an accessor, verification, or
`write_skills`, the adapter boundary is wrong.

**Every request declares `kinds=agent-skill`.** Delivery defaults to flags, so without it
the store is served the flag payload and holds no skills while reporting healthy. It also
narrows the connection to the one payload `_ProtocolReader` is built for; otherwise a
skill-enabled environment assigns two, and the reader warns and reads only the first.
`FDV2_PAYLOAD_KIND` (the payload) and `FDV2_OBJECT_KIND` (the objects inside it) are
different strings; keep them apart. No `mv`: it selects the *flag* data model, and delivery
ignores it for non-flag payloads.

**HTTP 422 is fatal.** Delivery answers 422 when a connection's declared kinds exclude every
payload assigned to it, and chose a non-400 4xx because LD SDKs treat those as terminal.
`_classify_status` returns a `_FatalTransportError` and the normal give-up path runs:
`failed` and `last_error` are set, `wait_for_skills` returns `False` at once, and
`connection_failures` is left untouched (it counts consecutive *recoverable* failures; an
existing count is kept, not zeroed, until `start()` runs delivery again and `_rearm_waiters`
resets it).

**The 422 message matches the TypeScript SDK's word for word, and stays short.** It names the
one cause a customer can fix — a view-scoped SDK key — and refers every other case to
support. Do not list the other causes. `test_the_422_message_names_its_one_actionable_cause`
asserts both the cause and the absences.

**A fatal stops the run, not the store, so every surface says `start()`, not "restart the
process".** `_give_up` does not close; only `close` sets `_closed`, the one thing `start`
refuses. A store that gave up — on a 401, 403, 404, 422, or another fatal status —
resumes in place once the cause is fixed, clearing the terminal reason through
`_rearm_waiters`. Asserted by `test_the_give_up_line_points_at_start_not_a_process_restart`,
`test_a_store_that_gave_up_on_a_422_resumes_on_start`, and
`test_a_restarted_store_does_not_report_the_old_failure`.

**The key travels over TLS only, and only to the base URI.**

- `_require_https_base_uri` rejects a plain `http://` base URI in the constructor, except for
  loopback (`localhost`, `127.0.0.1`, `::1`), where the test fakes listen.
- The opener uses `_RefuseRedirects`, because `urllib`'s standard redirect handler copies
  `Authorization` onto the redirected request. Any 3xx except 304 (a poll's not-modified
  answer) surfaces as an `HTTPError` that `_classify_status` maps to a fatal, non-retried
  failure.

**Reads are memory-bounded.** `_read_bounded` (poll bodies) and
`_iter_stream_lines`/`_iter_sse` (each line and each event) enforce `MAX_RESPONSE_BYTES`
(64 MiB). Crossing it raises `_ResponseTooLargeError`, a fatal error: nothing from that
body or event is applied, the in-flight payload is abandoned, `failed` and `last_error` are
set, and the committed set stays served. It is fatal because the size belongs to the
environment, not the connection: retried, it would re-download up to 64 MiB on every backoff
step forever. The requester wrappers re-raise fatal errors unchanged; do not let a generic
`except Exception` turn one back into a recoverable error. This bound is independent of
`skills_core.MAX_SKILL_CONTENT_BYTES` (one skill's content, at verification); do not derive
one from the other.

**The skill's version is in the object's `key`. `version` is the payload's.** Each version
of a skill is its own object on the wire, identified as `<key>:<version>`:

```json
{"key":"pdf-extraction:3","kind":"skill","version":42,
 "object":{"contentType":"text/markdown","content":"…","contentHash":"…","name":"…"}}
```

The `3` after the delimiter is what a `{key, version}` reference pins; it becomes the stored
`version`, under the stored key `pdf-extraction`. `version` (42) is the version of the
*payload*, and moves when anything in the environment moves, including an unrelated flag.
Reading it as the skill's version fails **silently**: the object verifies, the hash matches,
and the caller gets content under a meaningless version. Objects carry only `key`, `kind`,
`version` and `object`, like a flag, so there is no other field to read.

- `_split_wire_key` is the only place the wire key is read. `_store_object_from_put` and
  `_tombstone_from_delete` both go through it, and `TestVersionTranslation` asserts both
  directions.
- A wire key that will not split cleanly is *held*, not dropped (version-less, or with the
  offending text as its version), so verification withholds it as `invalid_version` under a
  key the caller recognises. Only a key with nothing before the delimiter is dropped.

**Skills are identified by `kind == "skill"`; other kinds are ignored, not rejected.** Object
kinds are open strings, and a skill arrives under the bare kind its producer registered.
With the `kinds` declaration, flag and segment objects should not arrive at all, but the skip
stays and stays tested: erroring on an unknown kind would turn a payload that gained a new
object kind into a permanent reconnect loop — a flag-delivery outage caused by a skills
rollout.

**Changes commit at `payload-transferred`, not as objects arrive.** A payload version is the
unit of consistency. A half-applied full transfer would publish a state the server never
described and briefly empty the store, which with pruning on deletes skill files. An
interrupted transfer keeps last known good, and listeners fire once per commit.

**A commit is the only thing that publishes a first payload, and a 304 is not one.**
`is_initialized()` — the fact `write_skills("*")` authorizes a prune on — goes true when a
payload *commits*, so every other answer has to stop short of claiming one. A
`payload-transferred` that applied nothing reports neither a commit nor an up-to-date
answer, and does not adopt the selector of a payload it never applied: a `none` intent
builds no pending set, nor does an `intentCode` this SDK does not recognise, and a foreign
payload's contents are declined. A poll adopts the response `ETag` only from a body that
completed an exchange — a commit, or a `none` intent, which is the server saying the
content held is what the etag describes. And a 304 *confirms* the payload held rather than
establishing one, because the exchange it stands in for cannot establish one either.
Loosen any of the three and the other two carry a store that received nothing into a prune
of every managed `SKILL.md` on disk: an empty committed set reads as an environment that
revoked every skill, and a 304 carries nothing to notice it on. There is no cached basis
to make it safe — `_basis` and `_etag` both start as `None` with no injection point, so a
304 reaching a store that holds nothing takes a server answering a request that carried no
etag at all.

**The first payload intent is read, and assumed to be the skill payload.** Delivery sends one
payload per credential and the protocol says to ignore all but the first intent, so
`payloads[0]` is read. The risk: an `xfer-full` for *another* payload would start an empty
pending set, and the next `payload-transferred` would publish it — every skill revoked, and
with pruning on, files deleted.

- `_ProtocolReader` therefore learns the skills payload (from the intent's `id` or the
  `(p:<id>:<version>)` selector) and declines a transfer of any other: one WARNING, counted
  in `diagnostics.payloads_ignored`, last known good kept.
- A transfer that names no payload is applied, since one-payload delivery is the common case.
- The first transfer on a connection has nothing to compare against; the separate WARNING on
  a multi-payload intent covers it.

**A hashless object is held, not dropped.** Verification withholds it as
`missing_content_hash`, and the transport makes that loud: an error per object, a summary
per wholly-hashless payload, and `diagnostics.hashless_objects`. Dropping it would report
`absent` — indistinguishable from "no such skill" — and let a prune delete the last
known-good copy on disk. Never synthesize a hash from the delivered content; that verifies
nothing.

**There is one network timeout.** `urllib`'s `timeout` covers connect, headers and each read,
and the standard library cannot bound connect separately. `read_timeout` is the only knob,
with a per-mode default: `DEFAULT_POLL_TIMEOUT` for a whole poll request,
`DEFAULT_STREAM_READ_TIMEOUT` for the gap between reads on a stream. Do not add a parameter
the standard library cannot honour. `TestTimeouts` measures the bound against a socket that
accepts and never answers.

**`close` interrupts the socket, not just a flag.** The delivery thread is blocked in a read
no flag can reach, and closing the response from another thread does not unblock CPython's
buffered reader. `_interrupt_read` shuts the socket down; without it, every shutdown of a
healthy stream waits out the full join timeout.

### The reported outcome vocabulary, and the `Resolution` mapping

`get_skill` returns `Skill | None`; `get_skill_result` returns a frozen `SkillOutcome`
(`skill`, `reason`, `detail`) naming which outcome happened. Both run `resolve_from_store` —
one retrieval, one verification, one telemetry pass — and differ only in what they report.

- **`get_skill`'s contract is frozen**: `None` for every failure, never raises for one
  (documented in its docstring and the README). Callers treat `None` as "no skill"; changing
  it breaks them silently.
- **`SkillOutcomeReason` is five tokens**, listed alphabetically like `IntegrityReasonCode`.
  The type name, accessor name and tokens are identical in the Python and TypeScript SDKs;
  do not rename one side.

Internal `Resolution.reason` maps 1:1 onto it, set explicitly at every construction site:

| `resolve_from_store` outcome | `reason` | Detection surface |
|---|---|---|
| the store raised (`unavailable=True`) | `store_unavailable` | an ERROR log line, no integrity record |
| `raw` is not a dict | `absent` | none — a pinned non-dict tells us nothing about the key, and a tampering signal from a broken adapter would be a false positive |
| `verify_raw_skill` returned `None` | `integrity_failure` | both: the log record **and** the product signal |
| `skill.key != key` | `integrity_failure` | the log record only, `reason_code: key_mismatch` |
| `skill.version != wanted_version` | `wrong_version` | the log record only, `reason_code: version_mismatch` |
| success | `ok` | none |

**The two record-only rows are intentional; do not "fix" them.** Both are decided after
`verify_raw_skill` passes, and the usual cause is a broken custom store adapter rather than
tampered content, so the product signal is left out to keep adapter bugs out of
LaunchDarkly's counter. Tests pin both directions for each (record present, signal absent).
The version row's spellings differ on purpose: `wrong_version` names what the caller got,
`version_mismatch` which check failed, so a detection rule is never ambiguous about which
surface it matches.

**A new internal outcome must choose its public token.** `Resolution.reason` has no default,
so the type checker asks; do not answer `absent`, which claims the store lacks the skill. If
none of the five fits, grow the token set in both SDKs in the same change.

The reason is deliberately *not*:

- **Derived from `Resolution.error`.** That string is prose for a human. `detail` passes it
  through: safe to surface (key and failure mode only, never content or a path), but not for
  matching on.
- **The same as `Resolution.unavailable`.** That flag answers "may prune run?" and is `True`
  exactly for `store_unavailable`. It must stay distinct from `absent`: only a raising store
  suppresses pruning, so an outage does not become data loss.

`get_skill_result` emits nothing of its own; the integrity record and signal already fired
during verification, and emitting again would double-count. There is no `get_skills_result`
or `all_skills_result`: the batch accessors omit unresolved entries and log a run-level WARN
count.

### Security posture — do not relax any of this

Store data is **untrusted input**; the transport is not part of the trust boundary.

- **Skill content is an opaque byte buffer.** `Skill.content` is the verified verbatim
  `bytes` that were hashed. The wire string is UTF-8 encoded once, during verification; after
  that the SDK never parses, decodes, or interprets it. Consumers parse frontmatter
  themselves.
- **Integrity is mandatory and checked twice, by one implementation.** Every raw object is
  verified at the accessor boundary (key pattern and length, integer version >= 1, content at
  most 10 MiB, lowercase-hex sha256 of the verbatim bytes against `contentHash`), and the
  hash is re-verified just before a write. Both go through `skills_core.verified_bytes`, so
  the integrity signal is the same whichever layer caught the defect. A `Skill` is only built
  from content that passed.
- **`contentHash` is required.** An object without one is withheld. A payload without hashes
  therefore yields nothing, so a withholding run logs a run-level count at WARN rather than
  looking like "this project has no skills".
- **No unencodable string reaches an encode.** `json.loads` turns a `\ud800` escape into an
  unpaired surrogate; every `.encode("utf-8")` site treats that as a verification failure.
  Never use `errors="surrogatepass"` — fabricated bytes could satisfy the hash comparison.
- **Wire strings are never echoed into telemetry.** A store could put the skill body in
  `contentHash` or `key`, so both are shape-checked and redacted before reaching a signal or
  a log line.
- **`write_skills` re-validates the key** before any filesystem call, regardless of upstream
  validation — a key becomes a directory name.
- **Never write through a symlink** — skill directory or target file, on both the write and
  prune paths.
- **Destructive operations only on manifest-listed paths whose `key` matches.** A file at a
  managed path with no matching manifest entry is reported `error` and left alone — unless its
  bytes already are the resolved content, in which case it is adopted (entry recorded,
  reported `skipped_current`). That lets a reconcile killed before its final manifest rewrite
  recover instead of wedging. Do not widen the exception:
  - Compare the verbatim bytes against the resolved `contentHash` — never a prefix, a length,
    an mtime, or the manifest's own `sha256`, which is untrusted and never a decision input.
  - A read that fails is a refusal, never an overwrite.
  - The read is bounded at `len(content) + 1` bytes, so a file that merely *begins* with the
    resolved content is refused.

  `skipped_current` is reused rather than adding an `adopted` kind because
  `ReconcileActionKind` is a public closed set.
- **Temp files are swept, within tight bounds.** `atomic_write` removes its temp file on any
  exception, but a `SIGKILL` can leave one behind, and one orphan makes `_prune_one`'s
  `rmdir` fail forever. The sweep is the only place the SDK removes a file the manifest does
  not list, and it is bounded on every axis:
  - only inside `<root>/<key>/`, for a key that passes `_key_rejection_reason`;
  - only names `safe_fs.is_temp_name` recognizes (ask `safe_fs`; a copied pattern would
    drift from the writer);
  - only regular files, typed off the descriptor;
  - listed with `os.listdir(fd)` and unlinked through that same pinned descriptor.

  It never raises and never aborts a run.
- **A corrupt manifest fails closed.** Unreadable, unparseable, not an object, malformed
  `entries`, larger than `_MAX_MANIFEST_BYTES`, or a `manifestVersion` outside
  `1 <= v <= MANIFEST_VERSION` means: no overwrites, no prunes, brand-new paths may still be
  written, an `error` action names the manifest, and the manifest itself is not rewritten.
  - The version is bounded on both sides: this SDK never wrote a version below 1.
  - The size bound exists because the manifest's length is unpredictable and it lives in a
    directory the SDK does not own exclusively. Every read in `skills_fs` is bounded;
    `_read_regular_file` requires `max_bytes` so a new call site cannot skip it.
- **An incomplete retrieval suppresses pruning**, so a transport outage is not read as
  "everything was revoked".
- **Writes are atomic**: temp file created exclusively in the target's *own* directory, mode
  `0644` set explicitly (never from the umask, never executable), write, fsync,
  `os.replace`, fsync the directory. `os.replace` is the single rename call site; do not swap
  in `os.rename`.
- **Every operation under the root goes through a pinned descriptor, not a path — including
  the reads that decide an action.** See "Descriptor-pinned filesystem access" below.
  Re-resolving `<root>/<key>` by path reopens a swap window. For the decision reads, a swap
  chooses the *branch*: a prune whose existence probe is answered "absent" by a swapped
  directory skips its unlink, drops the manifest entry, and reports `removed` while the
  revoked skill stays on disk.
- **Every numeric option rejects non-finite values, not just negative ones.**
  `write_skills`'s `timeout`, `watch_skills`'s `debounce`, and `FDv2SkillStore`'s
  `poll_interval` and `read_timeout` share one rule. `nan` and `inf` both pass a `< 0` check
  and then mean opposite things downstream: an unbounded vs. an already-expired deadline, a
  zero vs. a never-closing coalescing window, a poll loop that spins vs. one that stops while
  still reporting healthy. Do not narrow any of them back to a sign check.
- **A key valid to the data model may still be unrepresentable on disk.** The model allows
  256 characters; `NAME_MAX` is 255 bytes. Windows also reserves 22 device names: `con`,
  `prn`, `aux`, `nul`, `com1`–`com9`, `lpt1`–`lpt9` (`com0` and `lpt0` are *not* reserved;
  do not add them). `write_skills` rejects both before any filesystem call. Every per-skill
  filesystem failure is caught in the loop and becomes an `error` action; aborting would skip
  the manifest rewrite and orphan files already written that run.
- **Those two bounds live in `_key_rejection_reason`, not in the key grammar, and must not
  move.** `is_valid_skill_key` / `skill_key_rejection_reason` deliberately admit an over-long
  or reserved key, because:
  - `parse_ai_config` fails closed on a bad `skills` entry, so a grammar rejection would
    invalidate the *entire* AI Config for a Linux customer over a Windows-only constraint;
  - it would also shrink `skill_refs`, which authorizes a prune, turning "fails to write on
    Windows" into "deleted on Linux".

  `_key_rejection_reason` is shared by the write and prune paths. The reserved-name check is
  not gated on `os.name == "nt"`: a root written from a Linux container is routinely read
  from Windows, and with no Windows CI runner a gated branch would be untested. No suffix
  stripping or case folding is needed, since the grammar is lowercase-only and admits no `.`
  or `$` (so `con.txt` and `CONIN$` are unreachable). Total path length is not checked: the
  255-byte bound is per component and the root is the customer's, so `MAX_PATH` overflow is a
  README note.
- **A key is untrusted input everywhere it appears.** `skill_key_rejection_reason` is the
  single canonical explanation, so the config parser, the reference projection, and any later
  layer reject a key for the same stated reason. A projection never shrinks silently: every
  dropped entry is logged.

### Telemetry seam

Skills telemetry goes through a private emitter with one method, `record(signal,
properties)`, whose default is a **no-op** — nothing leaves the process in this release. Do
not use `client.track()`: it needs an LD context, spends the customer's event volume, lands
in their data export, and is silenced by offline mode. No LD context is involved anywhere in
this feature.

Exactly three signals exist, and the list is an **allowlist, not a floor**:

| Signal | When | Properties |
|---|---|---|
| `AgentControl Skill Integrity Failure` | any hash/size/shape verification failure | `skill_key`, `version?`, `expected_hash?`, `observed_hash?`, `language` |
| `AgentControl Skill Materialized` | each `written` / `updated` / `skipped_current` | `skill_key`, `content_bytes`, `content_hash`, `reconcile_action`, `language` |
| `AgentControl Skill Revoked Received` | prune removes a formerly managed skill | `skill_key`, `version`, `removed_from_disk`, `language` |

### The integrity-failure log record

The signal above is product telemetry; the **log record** beside it is the customer-owned
detection path, and the more important of the two. It is the only integrity surface when
telemetry is off, so it is a documented README contract, not a debugging aid.
`record_integrity_failure` writes both and is the only place either is constructed.

One ERROR record per withheld skill. The message text is `INTEGRITY_FAILURE_EVENT` + a space
+ `json.dumps(record, sort_keys=True, separators=(",", ":"))`, and the same mapping is
attached as `extra={"ld_skills": record}`.

- Fields: `event`, `action` (always `withheld`), `skill_key`, `version?`, `expected_hash?`,
  `observed_hash?`, `reason_code`, `reason`, `language`.
- Plus one record-only field on the boundary codes: `served_key?` on a `key_mismatch`, or
  `served_version?` on a `version_mismatch`, never both.
- Both served fields are store-controlled and shape-checked before writing (`<invalid-key>`,
  `<invalid-version>`), even though verification has already accepted them. The guards keep
  a future reordering from publishing a body, and keep `served_version` an integer for the
  byte-comparable JSON.

Do not undo any of these as a simplification:

- **The event name is in the message text**, not only in `extra`. Severity cannot
  discriminate (`resolve_from_store` and `list_raw_objects` also log ERROR for a raising
  store), and the stdlib's default formatter drops `extra`, so an `extra`-only record is
  invisible under a plain `logging.basicConfig()`.
- **`ld.skills.integrity_failure` is documented for customers to match on.** Never rename it.
- **`sort_keys=True` makes the line byte-identical across SDKs** (modulo `language`), since
  the other implementations build the object in alphabetical key order.
- **Optional fields are omitted, never nulled**, so a SIEM field-existence check means
  something.
- **The record spreads the signal's properties** rather than rebuilding them, so the two
  cannot drift, especially on redaction. Any new wire-sourced field needs the same
  shape-check-then-redact treatment.
- **`reason_code` is in the record only.** The signal's property set is the allowlist above.

`reason_code` is a **closed vocabulary of exactly ten tokens** — `IntegrityReasonCode`, a
`Literal`, so a typo at a call site is a type error — identical in every SDK. It is finer
than `SkillOutcomeReason` (five tokens); widening one does not widen the other. Eight map to
`record_integrity_failure` call sites; the other two come from `record_key_mismatch` and
`record_version_mismatch` at the retrieval boundary and write the log record **without** the
product signal:

| `reason_code` | Call site |
|---|---|
| `not_an_object` | `verify_raw_skill` — raw object is not a dict |
| `invalid_key` | `verify_raw_skill` — fails `is_valid_skill_key` |
| `invalid_version` | `verify_raw_skill` — fails `is_valid_skill_version` |
| `missing_content` | `verify_raw_skill` — `content` absent or not a string |
| `missing_content_hash` | `verify_raw_skill` — `contentHash` absent or not a string |
| `not_utf8` | `verified_bytes` — `UnicodeEncodeError` on encode (wire-`str` path only; a `Skill` already holds bytes) |
| `over_size_cap` | `verified_bytes` — over `MAX_SKILL_CONTENT_BYTES` |
| `hash_mismatch` | `verified_bytes` — observed sha256 != `contentHash` |
| `key_mismatch` | `resolve_from_store` — the served object's own `key` is not the key requested. **Log record only, no signal**, and carries a `served_key` field no other record has |
| `version_mismatch` | `resolve_from_store` — the served object's `version` is not the pinned version requested. **Log record only, no signal**, and carries a `served_version` field no other record has, beside a `version` that means the version *requested* |

**Testing the boundary codes.** Neither can join `REASON_CODE_CASES`, which drives every case
through `all_skills`: a listing has no requested key or version to mismatch. They are added
to the exhaustiveness assertion instead and covered by
`test_key_mismatch_records_the_log_but_not_the_signal` and
`test_version_mismatch_records_the_log_but_not_the_signal`. Do not "fix" the missing signal
(see the outcome table above).

**Both codes reach all four callers of `resolve_from_store`** — `get_skill`, `get_skills`,
`get_skill_result`, and `write_skills` — so cover the write path too. There, `wrong_version`
is per-skill and not `unavailable`: the key stays in the requested set with an `error`
action, its on-disk copy is not pruned, and the run is **not** marked incomplete, so a
genuinely revoked skill in the same run is still removed.

**An eleventh failure mode** means widening `IntegrityReasonCode`, adding a case to
`REASON_CODE_CASES` in `test_skills.py` (its exhaustiveness assertion fails otherwise),
documenting it in the README table, **and** doing the same in the other SDKs. A token added
on one side only silently breaks a customer's detection rule.

**Excluded signals.** `AgentControl Skill SDK Reference Returned` and `AgentControl Skill
Content Retrieved` are **deliberately not emitted** — both are observable server-side. Do
not add them. The skill body never appears in a signal, log line, or error message, and no
signal carries a filesystem path (paths belong in the user-facing `ReconcileReport`). An
emitter that raises is caught and logged; it never fails the operation.

**Where state lives.** Module state is in `skills_core.py`, shared by `skills.py` and
`skills_fs.py`, so there is one store and one emitter however the feature is entered. All
three signals are emitted by the `record_*` functions there; nothing else calls `emit`, so
the allowlist is enforced in one place.

**Injection goes through `skills.py`.** `_set_store`, `_set_emitter_for_testing` and
`_clear_state` delegate to `skills_core`. `set_skill_store`, `shutdown` and tests use
those; none should reach into `skills_core` directly.

### Descriptor-pinned filesystem access

A path check is only as good as the last path resolution after it. `lstat`, `realpath` and
containment checks validate an *inode*, but a following
`os.replace(tmp, root / key / "SKILL.md")` re-resolves `<root>/<key>` by *name*. Anyone with
write permission on a managed directory can move it aside, leave a symlink in its place, and
redirect the write or an unlink. Narrowing the window is not a fix; the race is winnable at
any width.

So the checks hand off to a descriptor, and no path is re-resolved afterwards. The
primitives live in `safe_fs.py`, which knows nothing about skills:

- **`open_directory_nofollow`** opens with `O_RDONLY | O_DIRECTORY | O_NOFOLLOW` and
  confirms `S_ISDIR` on the `fstat` (covering platforms with no `O_DIRECTORY`). It raises
  `DirectoryMissing`, a `ValueError` subclass, when nothing at the path is a directory
  (`ENOENT`, `ENOTDIR`). Callers learn a skill directory is absent from the pin itself, not
  from a separate `exists()` a swap could answer differently; a prune of an already-gone file
  and a sweep of a never-created directory both take that branch.
- **`open_or_create_directory`** adds `os.mkdir` plus an `lstat` on `FileExistsError`.
  `Path.mkdir(exist_ok=True)` would accept a symlink-to-directory as "already there".
- **`pinned_directory`** holds either for a block, so a caller branches once on
  `if dir_fd is not None` and cannot forget the `os.close`.
- **`atomic_write`** creates the temp file with `O_CREAT | O_EXCL | O_NOFOLLOW` at the
  descriptor (`_mkstemp_at`, since `tempfile` has no `dir_fd` form), `fchmod`s the descriptor
  (probed: Windows has no `os.fchmod` before 3.13, and 3.12 is supported), writes, fsyncs,
  renames with `os.replace(tmp, name, src_dir_fd=fd, dst_dir_fd=fd)`, then fsyncs the
  directory. `atomic_write_in` does the same for a directory the caller has not opened.
  `os.replace` is the single rename call site, looked up by attribute so tests can intercept
  it. Do not substitute `os.rename`; it also lacks defined overwrite semantics on Windows.
- **`unlink_file`** probes and unlinks descriptor-relative too: `unlink` never follows a
  *trailing* symlink but does resolve the directory above it. A symlink where the SDK expects
  its own file raises `SymlinkRefused` for the caller to report, rather than being tidied
  away. `_prune_one` uses it; its `rmdir` is relative to the root descriptor, which is safe
  because `rmdir` fails `ENOTDIR` on a symlink and only removes an empty directory.

**`skills_fs` pins the reads that decide an action.** `_write_one` and `_prune_one` pin
`<root>/<key>` relative to the root descriptor *before* deciding anything and hold it through
the action:

- existence probe: `os.stat(SKILL_FILENAME, dir_fd=skill_fd, follow_symlinks=False)`;
- compare read: `_read_regular_file(SKILL_FILENAME, dir_fd=skill_fd)`;
- orphan sweep: `os.listdir(skill_fd)`.

A later swap cannot make a prune skip its unlink and still report `removed`, make a compare
read adopt or refuse over a file outside the root, or feed the sweep names from elsewhere.
Where `dir_fd` is `None` (the `lstat` floor) these stay path-based — the documented Windows
bound. Nothing else under the root is read by path.

**Path checks stay as defense in depth.** Every `lstat`, `realpath` and containment check
lives in one shared `_unsafe_path_reason`, so the write and prune paths agree on what is
unsafe. They run before the pin; they are not the boundary, but do not remove them.

**`safe_fs.SUPPORTS_DIR_FD` gates all of it, and probes the advertised twins.**
`os.supports_dir_fd` is populated per syscall, and CPython registers `renameat` under
`os.rename` only and `fstatat` under `os.stat` only, even though `os.replace` and `os.lstat`
use the same syscalls. Probing the names actually called would report "unsupported" on every
POSIX platform and silently disable the defense. So:

- the probe names `{os.rename, os.open, os.unlink, os.stat}`;
- symlink checks are spelled `os.stat(..., follow_symlinks=False)`, not `os.lstat`;
- where the family is absent (Windows), `open_directory_nofollow` returns `None` after an
  `lstat` check (`os.open` cannot open a directory there), and every caller falls back to the
  per-component `lstat` floor. The residual window there is documented, not closed;
- the TOCTOU tests skip off this same flag, so a probe that wrongly reports "unsupported"
  cannot also silently skip the tests that would catch it.

**Test seam.** `os.replace` stays the single interceptable rename. In the descriptor-relative
form `dst` is the bare `"SKILL.md"`, so an `endswith("SKILL.md")` spy filter still matches,
and same-directory is proved by descriptor identity (`src_dir_fd == dst_dir_fd`, resolving to
the skill directory's `(st_dev, st_ino)`) rather than path strings. A spy must `fstat` the
descriptor **inside** the intercepted call; it is closed as soon as the write returns.

**The platform bound is POSIX-only by decision — do not quietly "fix" it.** Windows
reparse-point checks (`GetFileAttributesW`, `FILE_FLAG_OPEN_REPARSE_POINT`) are not
implemented: Windows is not a supported or tested platform for this release, there is no
Windows CI runner to verify them, and the TypeScript SDK has none to compare against.

- The TypeScript SDK closes the swap window on Linux by addressing children as
  `/proc/self/fd/<fd>/<name>` (behind `SUPPORTS_PROC_FD`), so its `lstat` floor applies on
  macOS and Windows only — the same shape as here. The decision rests on the missing CI
  runner, not on parity.
- On Windows, write permission on the managed root is the only boundary, which is why
  privilege separation is documented as the mitigation.
- Keep the reserved-device-name check, but do not read it as Windows hardening. If Windows
  becomes supported, add the CI runner first, then revisit both together.

**Privilege separation is the deployment-side half, and `ReconcileReport` must not grow a
writability field.** Run the reconcile as a different identity than the agent, so the
`0644`/`0755` modes actually deny something: a prompt-injected agent can read its skills but
cannot rewrite them or the manifest. The SDK cannot report whether the root is writable *by
the agent* — it knows only its own identity, which just wrote there — so any such field would
create false confidence. The operator's verification steps are in the README.

### Deferred: bounded retries

`timeout` is a monotonic deadline, checked before each retrieval, each write, and each prune;
only the final manifest rewrite runs past it, so files already written are never orphaned.
Bounded retries are **not** implemented at this layer; retry policy belongs to the delivery
transport (`FDv2SkillStore`'s capped backoff). Why:

1. **There is nothing transient to retry.** `SkillStore.get_object` is a synchronous
   in-process read of already-delivered data, modelled on the LaunchDarkly data-store API. A
   retry re-invokes customer code and gets the same answer.
2. **The seam cannot classify a failure** — it only sees "this raised". Retrying a
   `PermissionError` or a malformed payload just spends the caller's `timeout`.
3. **Backoff has nowhere to sleep.** The retrieval path (`_resolve_requests`,
   `_resolve_reference`, `_resolve_all`) is synchronous inside an async `write_skills`;
   `time.sleep` would block the caller's event loop.

There is no retry test and no assumable attempt count here. Keep both SDKs retry-free at
this layer, since how often a throwing store is called is observable.

---

## OTel Setup

The core client owns all OTel initialization. `init_client()` configures a `TracerProvider` with `ConversationIdSpanProcessor` and a `BatchSpanProcessor` plus an OTLP HTTP exporter when the optional OTel packages are installed.

**Required packages:**

```sh
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http \
  opentelemetry-propagator-b3
# or via the extras:
pip install "launchdarkly-ai-python[otel]"
```

**OTLP endpoint configuration** — the exporter uses the standard `OTEL_EXPORTER_OTLP_ENDPOINT` env var. The default (when not set) points to LaunchDarkly's hosted OTel collector.

**Other env vars / options read by `init_client()`:**
- `LD_SERVICE_NAME` / `options["serviceName"]` — sets `service.name` resource attribute (default: `'python-sdk'`)
- `LD_ENVIRONMENT` / `options["environment"]` — sets `deployment.environment` resource attribute

**Graceful degradation:** if any OTel package is missing, telemetry is silently skipped and a `logger.warning` is emitted. The LD client still initializes and all AI API calls work normally.

**Handler spans:** handler packages (e.g. `launchdarkly-ai-claude-agents`) create spans using the `opentelemetry` API. Those spans are picked up by the tracer provider registered here — no additional setup is required in the handler packages themselves.

---

## `inspect_config(key, context)`

Reads an AI Config variation **without invoking the model**. Use for health checks, logging, feature-gate probes, or any case where you need to know the current config state without spending AI API quota.

```python
result = await inspect_config("my-flag", context)
# result: {"enabled": bool, "config": dict | None, "meta": dict | None}
```

**Key guarantees:**
- Never raises — returns `{"enabled": False, "config": None, "meta": None}` on any error (network, bad key, unparseable config).
- Does not emit LD telemetry events.
- Does not call any AI provider.
- Lazily initializes the LD client when `LD_SDK_KEY` is set (same as other lifecycle functions).

When `enabled` is `False`, `config` is always `None`. When `enabled` is `True` but `config` is `None`, the flag variation failed schema validation.

---

## `init_client()` — When to Call It

**You do not need to call `init_client()` explicitly.** Every entry point (`config().invoke()`, `graph()`, etc.) lazily initializes the LD client on the first call, as long as `LD_SDK_KEY` is set in the environment.

**Call `init_client()` explicitly when you need to:**

- **Pass custom options** — `serviceName`, `environment`, or OTel configuration:
  ```python
  await init_client({"serviceName": "my-service", "environment": "production"})
  ```
- **Use a custom or edge runtime (BYOC path)** — pass any pre-initialized client that satisfies `LDClientInterface`:
  ```python
  ld_client = create_your_custom_client(os.environ["LD_SDK_KEY"])
  await init_client(ld_client)
  ```
- **Pre-warm the connection** — call at startup to eliminate cold-start latency on the first request.

`init_client()` is idempotent — calling it twice is a no-op. See full invariants below.

---

## Lifecycle Invariants

- **Lazy initialization.** Importing the package does not initialize the LD client. The first API call that needs LaunchDarkly calls `init_client()` internally when `LD_SDK_KEY` is set.
- **Explicit initialization — SDK path.** `await init_client(options?)` dynamically imports `launchdarkly-server-sdk` at runtime (optional peer dep). If the package is not installed it raises with a clear message.
- **Explicit initialization — BYOC path.** `await init_client(client)` accepts any pre-initialized object that satisfies `LDClientInterface` — this is the path for custom or edge environments whose SDK has different init semantics.
- `get_client()` raises `RuntimeError` if `init_client()` has not resolved.
- `await shutdown()` must be called before process exit. It flushes OTel spans, flushes LD events, and closes the LD client. It also releases the process-global OTel tracer provider, so a later `init_client()` can install its own — `trace.set_tracer_provider` is once-guarded, and leaving it set would route every later span to the provider just torn down. Only released when this SDK's own `set_tracer_provider` actually took effect — the set is once-guarded, so when another library registered first ours is refused (a warning is emitted) and that provider is left alone rather than torn down.

---

## Dependencies

Tier 0, so the runtime surface is deliberately tiny: **one** hard dependency, and everything else either an optional extra, resolved dynamically at runtime, or dev-only. Nothing here may grow without a reason recorded in this table.

### Runtime (`[project] dependencies`)

| Package | Why |
|---|---|
| `opentelemetry-api>=1.25` | The tracer/span API used on every instrumented path (`tracking.py`, `graph.py`, `content.py`, `conversation.py`, `utils.py`). API-only — the *SDK* half is an optional extra, so a consumer that never configures OTel gets no-op spans rather than an `ImportError`. `conversation.py` imports `opentelemetry.sdk.trace.SpanProcessor` under `TYPE_CHECKING` only, for exactly this reason. |

There is deliberately **no** `python-dotenv` here: `lifecycle.py` reads `os.environ` directly, so loading a `.env` file is the application's job rather than the SDK's. `python-dotenv` is in the workspace dev group for the examples only.

### Optional extra (`[project.optional-dependencies] otel`)

| Package | Why |
|---|---|
| `opentelemetry-sdk>=1.25` | Tracer provider, resources, and the batch span processor, imported inside `_setup_telemetry()` in `lifecycle.py`. Optional so telemetry is opt-in; absent ⇒ a `logger.warning` and no spans, never a raise. |
| `opentelemetry-exporter-otlp-proto-http>=1.25` | OTLP/HTTP span export and its compression enum. Same optionality, same loader. |

Install with `pip install "launchdarkly-ai-server[otel]"`; see [OTel Setup](#otel-setup) for the endpoint variables.

### Resolved dynamically, declared nowhere

| Package | Why |
|---|---|
| `launchdarkly-server-sdk` | The LaunchDarkly server SDK, reached by `importlib.import_module("ldclient")` (falling back to `launchdarkly_server_sdk`) inside `init_client()`'s options path. Undeclared because the BYOC path (`init_client(client=...)`) supplies its own client and should not have to install an unused SDK. Absent ⇒ a `RuntimeError` naming the `pip install`, only on the path that needs it. |

### Dev-only (workspace root `[dependency-groups] dev`) — the ones with a contract attached

| Package | Why |
|---|---|
| `launchdarkly-server-sdk>=9.0`, and the `otel` extra mirrored (`opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http`) | Each dynamically-resolved or optional package is repeated in the dev group so the test suite can import it. Something that is *only* optional would not be installed in this workspace and the tests covering its present-and-working path could not run. |
| `pytest>=8`, `pytest-asyncio>=0.24` | Test runner and the async support the whole suite relies on. `asyncio_mode = "auto"` is set at the workspace root, which is why no test in this package carries an `@pytest.mark.asyncio`. |
| `mypy>=1.10` (`strict`), `ruff>=0.15` | Type checker and linter/formatter. `mypy` strict mode is the only thing enforcing the `Literal[...]` closed set on `ReconcileAction.action` — unlike `write_skills`'s `on_unavailable`, which is also checked at runtime because the value can arrive from untyped code. |

---

## Common Pitfalls

### 1. Calling `get_client()` before `init_client()` resolves

`get_client()` raises `RuntimeError` if no client has been initialized. Handler packages that emit LD tracking events call `get_client()` — this is safe only inside a handler call because by then `config().invoke()` has already validated the flag variation, which requires an initialized client. Never call `get_client()` at module load time or in a package constructor.

### 2. Returning `dict` not a dataclass from handlers

`execute_and_track` expects the handler to return a plain `dict` with at least `output` and `usage` keys. Do not return a custom class — `parse_usage` and the telemetry pipeline both access dict keys.

### 3. Interpreting skill content anywhere

`Skill.content` is opaque `bytes`: the verified verbatim bytes that were hashed. Do not add a
parser, a decoder, or a convenience accessor — no YAML/frontmatter parsing, no "decode as
UTF-8 for display". Consumers who want structure parse the bytes themselves. Same contract
as the TypeScript SDK.

### 4. Assuming `write_skills` prunes on every run

Pruning is suppressed when the manifest is corrupt or any retrieval was incomplete: the SDK
cannot tell what it owns or what is current, and deleting anyway is data loss. A run whose
report has a manifest `error` pruned nothing, so "no `removed` actions" does not mean
"nothing is stale".

### 5. Treating "absent from the resolved set" as always meaning revoked

Revocation is pruning, but only for a skill the store no longer serves. An object that is
*present and unverifiable* is different: `_resolve_all` must emit a failed `_PendingWrite`
for it, not filter it out. Dropping it leaves its key out of the requested set, so prune
deletes the last known-good copy and reports a routine `removed` with `report.ok` still
true. Tampered content must never trigger deletion.

### 6. Expecting revocation to reach a boot-only `write_skills` deployment

Even with `watch_skills`, only the `"*"` form removes a revoked skill from disk. With an
explicit list such as `skill_refs(config)`, the list is fixed: a skill the store answers
`absent` for is reported as an `error` and its files are kept, and the watcher listens only to
the skill store, so unpinning a skill or moving it to a new version is not seen until the refs
are read again.

Without `watch_skills`, the revocation bound is process lifetime: a skill revoked after boot
stays on disk until the process reconciles again, so a restart (or an explicit re-run of
`write_skills`) is the incident-response action — and content an agent has already read into
a conversation is out of reach at this layer either way.

---

## Adding a New Export

1. Implement the function/type in the appropriate `src/launchdarkly_ai_server/*.py` file.
2. Add a named import to `__init__.py` and add the name to `__all__`.
3. All handler packages pick up the change automatically via the local path dependency.

## Invariants to Preserve

- Do not add dependencies on any `launchdarkly-ai-*` handler package. This package has no upward dependencies.
- Do not add a hard dependency on `launchdarkly-server-sdk`. It must remain an optional peer, discovered via dynamic `importlib.import_module`.
- Handler packages must import `LDContext` from `launchdarkly-ai-server` — not directly from any LD SDK.
- Do not weaken the `parse_ai_config` validation — handler packages rely on `config` being valid when they receive it.
- `parse_usage` must continue to accept `input_tokens/output_tokens`, `inputTokens/outputTokens`, and `input/output` as all existing handlers return one of these variants.
- `Skill.content` is opaque `bytes`. Do not add anything that parses or interprets it — no YAML library in this package's dependencies at any tier, and no accessor that decodes content.
- Do not route skills telemetry through `client.track()`, and do not introduce an LD context anywhere in the skills path. Signals go through the `skills_core.py` emitter seam, whose default is a no-op, and only via its `record_*` functions.
- Do not add a signal name outside the three in the Agent Skills table above — the list is an allowlist. `AgentControl Skill SDK Reference Returned` and `AgentControl Skill Content Retrieved` were considered and deliberately excluded from SDK emission.
- Do not rename `ld.skills.integrity_failure`, and do not add an eleventh `reason_code` in one language only — both are documented compatibility surfaces. See "The integrity-failure log record" above.
- Do not relax any of the `write_skills` filesystem defenses (local key re-validation, symlink refusal, manifest-authorized destruction, corrupt-manifest fail-closed, atomic `0644` writes). Each is a deliberate security property with abuse-case tests attached.
- Do not make `SkillStore` lookups key-only. Version is part of the lookup identity because a payload holds several versions of one key; a key-only seam cannot express a version-pinned reference.
