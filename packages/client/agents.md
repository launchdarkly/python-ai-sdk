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
| `src/launchdarkly_ai_server/lifecycle.py` | `init_client`, `get_client`, `shutdown`, `extract_variation` |
| `src/launchdarkly_ai_server/client.py` | `config()`, `ConfigInstance` |
| `src/launchdarkly_ai_server/tracking.py` | `execute_and_track`, `execute_and_stream`, `wrap_tool_handlers`, `parse_usage` |
| `src/launchdarkly_ai_server/graph.py` | `graph()`, `resolve_graph()`, `GraphInstance` |
| `src/launchdarkly_ai_server/types.py` | All shared Python types — `AiConfigRep`, `ProviderHandler`, `LDContext`, `NativeTool`, etc. |
| `src/launchdarkly_ai_server/types_validation.py` | `parse_ai_config` — validates flag variation shape; `is_valid_skill_key` / `is_valid_skill_version` / `skill_key_rejection_reason` (the canonical key-grammar explanation every layer quotes) |
| `src/launchdarkly_ai_server/skills.py` | Agent Skills, retrieval half — `skill_refs`, `get_skill`/`get_skills`/`all_skills`, `InMemorySkillStore`, and the store/telemetry injection points `_set_store` / `_set_emitter_for_testing` |
| `src/launchdarkly_ai_server/skills_core.py` | Shared skills internals — the `SkillStore` seam, module state, the telemetry seam and its three recorders, integrity verification, and store resolution. Imported by both `skills.py` and the materialization layer; imports neither |
| `src/launchdarkly_ai_server/skills_fs.py` | Agent Skills, materialization half — `write_skills`, request resolution, the manifest format and on-disk filenames, per-skill reconcile, and pruning |
| `src/launchdarkly_ai_server/safe_fs.py` | Descriptor-pinned filesystem primitives — `atomic_write`, `unlink_file`, `pinned_directory`, `open_directory_nofollow`, `open_or_create_directory`, `SymlinkRefused`, and the `*at()` capability probe. Owns the descriptor-vs-path platform split; knows nothing about skills |
| `src/launchdarkly_ai_server/utils.py` | `parse_template`, `parse_json_with_possible_fences`, `create_handler`, `parse_usage`, `make_track_data`, `to_ld_context` |
| `src/launchdarkly_ai_server/registry.py` | `Registry`, `global_registry`, `compose`, `resolve_handlers`, `resolve_tools` |
| `src/launchdarkly_ai_server/judges.py` | `run_judges`, `build_judge_tasks`, `run_judge` |
| `src/launchdarkly_ai_server/__init__.py` | Public barrel — the only surface handler packages import from |

---

## Public Exports

Key symbols exported from `launchdarkly_ai_server`:

```python
# Lifecycle
from launchdarkly_ai_server import init_client, get_client, shutdown, extract_variation
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
    Skill, SkillReference, ReconcileAction, ReconcileReport,
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
from launchdarkly_ai_server import config, graph, resolve_graph

# Agent Skills
from launchdarkly_ai_server import (
    skill_refs, get_skill, get_skills, all_skills, write_skills,
    SkillStore, InMemorySkillStore,
    SKILL_FILENAME, MANIFEST_FILENAME, MANIFEST_VERSION,
    ReconcileActionKind, OnUnavailable,   # the two closed-set unions
)
```

`MAX_SKILL_CONTENT_BYTES` is deliberately *not* among them: it is a local enforcement
bound on content the platform produces, not a value this SDK defines, so exporting it
would semver-lock a number this side does not own. Keep it internal to `skills_core`.

`SKILL_OBJECT_KIND` is not exported either, for a different reason: it is the string this
SDK hands a store, and a store adapter maps whatever the transport underneath calls a skill
onto it. Publishing it would advertise an SDK-side seam value as the wire contract — a claim
this side cannot make, and hard to walk back once a caller depends on it. An adapter that
needs to agree with it reaches it through `launchdarkly_ai_server.skills_core`.

When adding a new export, add it to `__init__.py`'s imports and `__all__`. Handler packages must never import from sub-paths (e.g. `launchdarkly_ai_server.client`).

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
3. If `judge_configuration.judges` is present, runs each judge handler (sampled by `sampling_rate`) against the primary response and tracks `evaluation_metric_key`.
4. Returns `ProviderResponse`: `{ response: str, usage: UsageDict, track_data: TrackData, judge_results?: dict[str, JudgeResult], judge_tasks?: list[JudgeTask] }`. `judge_results` is populated when `skip_judges=False` (default) and judges ran; `judge_tasks` is populated when `skip_judges=True`.

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
2. **Content accessors** — `get_skill`, `get_skills`, `all_skills` read through the
   `SkillStore` seam. Configure a store with
   `init_client(options={"skillStore": store})`; with none configured the accessors raise
   an actionable `RuntimeError`. A delivery transport can be added behind the seam
   without touching the public API.
3. **Materialization** — `write_skills(skills, root)` writes `<root>/<key>/SKILL.md` and
   reconciles against a manifest at `<root>/.launchdarkly-skills.json`.

### The store seam, and why version is part of the lookup

`SkillStore` is `get_object(kind, key, version=None)`, `all_objects(kind)`, and an optional
`add_listener(kind, fn)`. Version is part of the **lookup identity**, not a filter applied
to the answer, and that is load-bearing: a delivery payload carries the newest version of
every skill *plus* every version any variation currently pins, so two versions of one key
coexist routinely. A seam keyed by key alone would answer a pinned reference with the newest
object, and the caller would then have to reject it — turning the primary use case, a
version-pinned attachment, into a missing skill. `version=None` asks for the newest held.

The equality check in `resolve_from_store` stays, now as a **defense** rather than as the
selection mechanism: the store is untrusted, so an answer that is not the version asked for
is withheld.

`all_objects` returns one entry per `(key, version)` under keys that are **opaque** to this
SDK. Do not parse them and do not assume one per skill key; identity is read off each
object's own `key` and `version`, which are revalidated anyway. `newest_by_key` is the
one place that collapses the result to one object per key, because both whole-store
consumers need it — `all_skills`, since a list holding two versions of one key is not a set
of skills, and the `"*"` reconcile, since `<root>/<key>/SKILL.md` is a single path.

### Security posture — do not relax any of this

Store data is **untrusted input**; the transport is not part of the trust boundary.

- **Skill content is an opaque byte buffer.** `Skill.content` is `bytes` — the verified
  verbatim bytes, exactly what was hashed. The wire object delivers content as a JSON
  string; the UTF-8 encode happens once, during verification, and from then on the SDK
  never parses, decodes, or interprets the bytes anywhere: not in the integrity path, not
  in an accessor, not during materialization. Consumers who want frontmatter parse it
  themselves.
- **Integrity is mandatory and doubled, through one implementation.** Every raw object is
  verified at the accessor boundary (key pattern and length, integer version >= 1, content
  at most 64 KiB, sha256 lowercase hex over the verbatim bytes against `contentHash`)
  and the hash is re-verified immediately before a write, both through
  `skills_core.verified_bytes`, so the integrity signal's property set cannot depend on
  which layer caught the defect. A `Skill` is only ever constructed from content that
  passed. Nothing unverified reaches user code.
- **`contentHash` is required.** An object without one is withheld, not accepted on trust.
  A payload built before the field is populated therefore yields nothing, which is why a
  withholding run logs a run-level count at WARN — an empty result would otherwise be
  indistinguishable from "this project has no skills".
- **No unencodable string ever reaches an encode.** `json.loads` turns a `\ud800` escape
  into an unpaired surrogate with no UTF-8 representation; every `.encode("utf-8")` site
  treats that as a verification failure. Never reach for `errors="surrogatepass"` —
  fabricating bytes could satisfy the hash comparison.
- **Attacker-controlled strings are never echoed into telemetry.** `contentHash` and `key`
  come off the wire, so a store could put the skill body in either; both are shape-checked
  and redacted before they reach a signal or a log line.
- **The key is re-validated inside `write_skills`**, regardless of upstream validation — a
  key becomes a directory name. Rejection happens before any filesystem call.
- **Never write through a symlink**, in either the skill directory or the target file, on
  the write path *and* the prune path.
- **Destructive operations only on manifest-listed paths whose `key` matches.** A file at a
  managed path with no matching manifest entry is reported as `error` and left alone —
  *unless its bytes already are the resolved content*, in which case it is adopted (manifest
  entry recorded, reported `skipped_current`). That single exception is what makes a
  reconcile killed between the content writes and the final manifest rewrite recoverable
  instead of permanently wedged, and it cannot be widened: the comparison is over the
  verbatim bytes against the resolved `contentHash`, a read that fails is a refusal and
  never an overwrite, and the read is bounded at `len(content) + 1` bytes so a file that
  merely *begins* with the resolved content is refused too. Do not relax it to a prefix, a
  length, an mtime, or the manifest's own recorded `sha256` — that field is untrusted and is
  never a decision input. `skipped_current` is reused deliberately rather than adding an
  `adopted` action kind; `ReconcileActionKind` is a public closed set.
- **Temp files are swept, within the same bounds as everything else.** `atomic_write` unlinks
  its own temp file on any exception, but a `SIGKILL` leaves one behind that no manifest
  entry records, and a non-empty directory defeats `_prune_one`'s `rmdir` — so one orphan
  pins a skill directory forever. The sweep is the only place this SDK removes a file the
  manifest does not list, and it is bounded on every axis: inside `<root>/<key>/` only, for a
  key that passes `_key_rejection_reason`; only names `safe_fs.is_temp_name` recognizes,
  anchored at both ends and asked of `safe_fs` rather than re-spelled (a copy would drift
  from the writer); only regular files, with the type read off the descriptor; unlinked
  through the pinned descriptor. It never raises and never aborts a run.
- **A corrupt manifest fails closed**: unreadable, unparseable, not an object, malformed
  `entries`, or a `manifestVersion` this release cannot read means no overwrites and no
  prunes, brand-new paths may still be written, an `error` action names the manifest, and
  the manifest file itself is not rewritten.
- **An incomplete retrieval suppresses pruning.** Otherwise a transport outage would read
  as "everything was revoked" and delete the customer's managed files.
- **Writes are atomic**: temp file created exclusively in the target's *own* directory,
  mode `0644` set explicitly (never inherited from the umask, never executable), write,
  fsync, `os.replace`, fsync the directory. `os.replace` is the single rename call site
  and must not be swapped for `os.rename`.
- **Every operation under the root goes through a pinned descriptor, not a path.** See
  "Descriptor-pinned filesystem access" below. Re-resolving `<root>/<key>` from its path at
  write or unlink time reopens a swap window that the checks above cannot cover.
- **A key valid to the data model may still be unrepresentable on disk.** The model allows
  256 characters; `NAME_MAX` is 255 bytes. Windows additionally reserves 22 MS-DOS device
  names, none of which can be a directory name there: `con`, `prn`, `aux`, `nul`,
  `com1`–`com9`, `lpt1`–`lpt9` (`com0` and `lpt0` are *not* reserved; do not add them).
  `write_skills` rejects both before any filesystem call, and every per-skill filesystem
  failure is caught at the loop so it becomes an `error` action — aborting the loop would
  skip the manifest rewrite and orphan files already written in that run.
- **Those two bounds live in `_key_rejection_reason`, not in the key grammar, and must not
  move.** `is_valid_skill_key` / `skill_key_rejection_reason` keep admitting an over-long or
  reserved key on purpose. `parse_ai_config` fails closed on a bad `skills` entry, so a
  grammar-level rejection would invalidate the *entire* AI Config — model, provider,
  instructions, tools — for a Linux customer over a Windows-only constraint; and it would
  silently shrink `skill_refs`, which is what authorizes a prune, converting "this skill
  fails to write on Windows" into "this skill gets deleted on Linux". `_key_rejection_reason`
  is shared by the write and prune paths, so one edit covers both destructive paths.
  The reserved-name check is unconditional rather than `os.name == "nt"`-gated: a root
  written from a Linux container is routinely read from a Windows host, and neither
  repository has a Windows CI runner (every matrix job is `ubuntu-latest`), so a gated branch
  would be untestable — the exact condition that produced the gap. No suffix stripping and no
  case folding are needed, because the grammar admits no `.` and no `$` (so `con.txt` and
  `CONIN$` are unreachable) and is lowercase-only. The residual the SDK cannot check is total
  path length: the 255-byte bound is per *component*, and the root belongs to the customer,
  so `MAX_PATH` overflow is a README note rather than a check.
- **A key is untrusted input everywhere it appears.** `skill_key_rejection_reason` is the
  single canonical explanation, so the config parser and the reference projection reject a
  key for the same stated reason — and so does every layer added later. A silently
  shortened projection is not acceptable: every dropped entry is logged.

### Telemetry seam

Skills telemetry goes through a private emitter with one method,
`record(signal, properties)`, whose default implementation is a **no-op** — nothing leaves
the process in this release. `client.track()` is deliberately *not* used: it needs an LD
context, spends the customer's event volume, lands in their data export, and is silenced by
offline mode. No LD context is involved anywhere in this feature.

Exactly three signals exist, and the list is an **allowlist, not a floor**:

| Signal | When | Properties |
|---|---|---|
| `AgentControl Skill Integrity Failure` | any hash/size/shape verification failure | `skill_key`, `version?`, `expected_hash?`, `observed_hash?`, `language` |
| `AgentControl Skill Materialized` | each `written` / `updated` / `skipped_current` | `skill_key`, `content_bytes`, `content_hash`, `reconcile_action`, `language` |
| `AgentControl Skill Revoked Received` | prune removes a formerly managed skill | `skill_key`, `version`, `removed_from_disk`, `language` |

### The integrity-failure log record

The signal above is product telemetry; the **log record** beside it is the customer-owned
detection path, and the more load-bearing of the two. It is the only integrity surface that
works when telemetry is off, and the only one that exists at all in an instance with no
telemetry destination, so it is a documented contract in the README rather than a debugging
aid. `record_integrity_failure` writes both, and is the only place either is constructed.

One ERROR record per withheld skill, message text = `INTEGRITY_FAILURE_EVENT` + a space +
`json.dumps(record, sort_keys=True, separators=(",", ":"))`, plus the same mapping under
`extra={"ld_skills": record}`. Fields: `event`, `action` (always `withheld`), `skill_key`,
`version?`, `expected_hash?`, `observed_hash?`, `reason_code`, `reason`, `language`.

Each of those choices is load-bearing; do not undo one as a simplification.

- **The event name is in the message text**, not only in `extra`. Severity cannot
  discriminate — `resolve_from_store` and `list_raw_objects` in the same module also log
  ERROR for a raising store — and the stdlib's default formatter drops `extra` entirely, so
  an `extra`-only record is invisible under a plain `logging.basicConfig()`.
- **`ld.skills.integrity_failure` is documented for customers to match on**, which makes it
  a compatibility surface. It must never be renamed.
- **`sort_keys=True` is not cosmetic.** The other language implementations build the object
  in alphabetical key order, so sorting makes the serialized line byte-identical across
  SDKs for the same input, modulo `language`.
- **Optional fields are omitted, never nulled**, so a SIEM field-existence check means
  something.
- **The record spreads the signal's properties** rather than rebuilding them, so the two
  cannot drift on the fields they share — in particular on which are redacted. Anything
  added later that comes off the wire needs the same shape-check-then-redact treatment.
- **`reason_code` is in the record only.** The signal's property set is the allowlist above
  and does not grow; the local record is where the detection vocabulary lives.

`reason_code` is a **closed vocabulary of exactly eight tokens** — `IntegrityReasonCode`, a
`Literal`, so a typo at a call site is a type error — one per `record_integrity_failure`
call site, and the same eight in every language implementation:

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

Adding a ninth failure mode means widening `IntegrityReasonCode`, adding a case to
`REASON_CODE_CASES` in `test_skills.py` (whose exhaustiveness assertion fails otherwise),
documenting it in the README table, **and** doing the same in the other language SDKs. A
token added on one side only is a drift bug: a customer's detection rule stops matching
where they cannot see it.

`AgentControl Skill SDK Reference Returned` and `AgentControl Skill Content Retrieved`
were considered and **deliberately excluded from SDK emission** — both are observable
server-side. Do not add them. The skill body never appears in a signal, a log line, or
an error message, and no signal carries a filesystem path (paths belong in the returned
`ReconcileReport`, which is user-facing API). An emitter that raises is caught and logged;
it never fails the operation.

Module state lives in `skills_core.py`, the module `skills.py` and `skills_fs.py` share,
so there is exactly one store and one emitter however the feature is entered. All three signals are emitted from the `record_*` functions
next to the seam there — nothing outside that module calls `emit`, so the allowlist is
enforced in one place.

The injection path is deliberately narrower than the state's location: `skills.py` owns
`_set_store`, `_set_emitter_for_testing` and `_clear_state`, which delegate to
`skills_core`. `init_client` and `shutdown` use those, tests inject through those
(`skills._set_store(store)` is the same setter `init_client` uses), and neither should
reach into `skills_core` directly.

### Descriptor-pinned filesystem access

A path check is only as good as the last path resolution after it. Every `lstat`, `realpath`
and containment check validates an *inode*, but a following
`os.replace(tmp, root / key / "SKILL.md")` re-resolves `<root>/<key>` from its *name* — so
anything holding write permission on a managed directory can move the validated directory
aside, leave a symlink in its place, and redirect the write (or an unlink) somewhere else.
Narrowing that window is not a fix; the race is winnable at any width.

So the checks hand off to a descriptor and nothing re-resolves a path afterwards. The
primitives live in `safe_fs.py`, which knows nothing about skills:

- `open_directory_nofollow` opens the directory with `O_RDONLY | O_DIRECTORY | O_NOFOLLOW`
  and confirms `S_ISDIR` on the `fstat` (the explicit check is what covers platforms with no
  `O_DIRECTORY`). `open_or_create_directory` wraps it with `os.mkdir` plus an `lstat` on the
  `FileExistsError` path — `Path.mkdir(exist_ok=True)` accepts a symlink-to-directory as
  "already there", which would reopen the hole the caller's check just closed.
  `pinned_directory` holds either for the duration of a block, so a caller states the
  platform split once as `if dir_fd is not None` and cannot forget the `os.close`.
- `atomic_write` creates the temp file with `O_CREAT | O_EXCL | O_NOFOLLOW` **at** that
  descriptor (`_mkstemp_at`, since `tempfile` has no `dir_fd` form), `fchmod`s the
  descriptor rather than `chmod`ing a path, writes, fsyncs, and renames with
  `os.replace(tmp, name, src_dir_fd=fd, dst_dir_fd=fd)`, then fsyncs the directory so the
  rename survives a crash. `atomic_write_in` is the same against a directory the caller does
  not already hold open. `os.replace` is the single rename call site, reached by attribute
  lookup so tests can intercept it, and `os.rename` must not be substituted for it — it is
  also the only one with defined overwrite semantics on Windows.
- `unlink_file` probes and unlinks descriptor-relative too. `unlink` never follows a
  *trailing* symlink, but it does resolve the directory above it, so the same swap turns a
  removal into a delete of an attacker-chosen file. A symlink found where this SDK expects
  its own file raises `SymlinkRefused` rather than being tidied away: the state on disk is
  not what the caller believes, and that is the caller's to report. `_prune_one` goes
  through it; `rmdir` stays path-based and is safe that way, since it fails `ENOTDIR` on a
  symlink and only ever succeeds on an empty directory.

Every `lstat`, `realpath` and containment check on the skills side lives in one shared
`_unsafe_path_reason`, so the write and prune paths cannot drift apart on what counts as
unsafe. `skills_fs._prune_one` spells its symlink check `os.stat(..., follow_symlinks=False)`
rather than `os.lstat`, matching the name the capability probe advertises.

`safe_fs.SUPPORTS_DIR_FD` gates all of it, and the probe is not the obvious one.
`os.supports_dir_fd` is populated per underlying syscall, and CPython registers `renameat`
under `os.rename` only and `fstatat` under `os.stat` only — even though `os.replace` is the
same `renameat`-backed function and `os.lstat` is `fstatat` with `AT_SYMLINK_NOFOLLOW`.
Probing the names this module actually calls reports "unsupported" on every POSIX platform
and silently turns the defense off, so the probe names the advertised twins
(`{os.rename, os.open, os.unlink, os.stat}`) and a caller's symlink check is spelled
`os.stat(..., follow_symlinks=False)` rather than `os.lstat`. Where the family is absent
(Windows) `open_directory_nofollow` returns `None` after an `lstat` check instead of
attempting the descriptor open — `os.open` cannot open a directory there — and every caller
falls back to the identical full-path sequence, the per-component `lstat` floor. The
residual window on those platforms is documented rather than closed; the TOCTOU tests skip
off this same flag, deliberately, so a probe that wrongly reports "unsupported" cannot also
silently skip the tests that would have caught it.

Both call shapes are admitted by the test seam. `os.replace` remains the single
interceptable rename call site; under the descriptor-relative shape `dst` is the bare string
`"SKILL.md"`, so an `endswith("SKILL.md")` spy filter still matches, and the
same-directory requirement is proved by descriptor identity (`src_dir_fd == dst_dir_fd`,
resolving to the skill directory's `(st_dev, st_ino)`) instead of by comparing path strings.
A spy must `fstat` the descriptor **inside** the intercepted call — the implementation closes
it as soon as the write returns.

### Deferred: bounded retries

`timeout` is implemented — a monotonic deadline, checked before each retrieval, before
each write, and before each prune; only the final manifest rewrite runs past it, so files
already written are never orphaned. Bounded retries inside that deadline are **not**
implemented, and belong to the delivery transport, not to this layer. Three structural
reasons, all of which the transport changes:

1. **There is nothing transient to retry.** `SkillStore.get_object` is a synchronous
   in-process read against already-delivered data, modelled on the LaunchDarkly
   data-store API. `InMemorySkillStore` reads a dict. A retry re-invokes customer code and
   returns the same answer.
2. **The seam cannot classify a failure.** All it surfaces is "this raised". Retrying a
   `PermissionError` or a malformed payload spends the caller's `timeout` on a certainty.
   The transient/permanent taxonomy a retry policy needs is the transport's to define.
3. **Backoff has nowhere to sleep.** The retrieval path (`_resolve_requests`,
   `_resolve_reference`, `_resolve_all`) is synchronous, called from an async
   `write_skills`. Backoff would mean either `time.sleep` — blocking the event loop of every
   caller — or async-ifying the whole path for a store that cannot benefit.

Picking a bound and a backoff now would fix numbers in a cross-language contract with no
transport to calibrate them against, so there is **no** retry test and no assumable attempt
count. When the transport lands it owns the policy; keep both languages retry-free until
then, since the number of times a throwing store is invoked is observable and the two would
otherwise diverge.

---

## OTel Setup

The core client owns all OTel initialization. `init_client()` configures a `TracerProvider` with `ConversationIdSpanProcessor` and a `BatchSpanProcessor` plus an OTLP HTTP exporter when the optional OTel packages are installed.

**Required packages:**

```sh
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http \
  opentelemetry-propagator-b3
# or via the extras:
pip install "launchdarkly-ai[otel]"
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
- `await shutdown()` must be called before process exit. It flushes OTel spans, flushes LD events, and closes the LD client.

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
| `launchdarkly-server-sdk` | The LaunchDarkly server SDK, reached by `importlib.import_module("ldclient")` (falling back to `launchdarkly_server_sdk`) inside `init_client()`'s options path. Undeclared on purpose: the BYOC path (`init_client(client=...)`) targets environments that supply their own client, and a hard dependency would force an unused SDK into every such install. So it is imported late and raises actionably when missing — absent ⇒ a `RuntimeError` naming the `pip install`, and only on the path that needs it. |

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

`Skill.content` is opaque `bytes` by construction. Do not add a parser, a decoder, or a
convenience accessor that reads meaning into it — no YAML/frontmatter parsing, no
"decode as UTF-8 for display", nothing. The SDK's whole contract is that content is the
verified verbatim byte buffer and nothing more; a consumer who wants structure parses it
on their side of the boundary.

### 4. Assuming `write_skills` prunes on every run

Pruning is suppressed when the manifest is corrupt or any retrieval was incomplete — both
mean the SDK cannot tell what it owns or what is still current, and deleting under that
uncertainty is data loss. A run whose report contains a manifest `error` will not have
pruned anything, so do not read "no `removed` actions" as "nothing is stale".

### 5. Treating "absent from the resolved set" as always meaning revoked

Revocation is pruning, but only for a skill the store genuinely no longer serves. An object
that is *present and unverifiable* is a different thing, and `_resolve_all` must emit a
failed `_PendingWrite` for it rather than filtering it out: dropping it silently leaves its
key out of the requested set, so prune deletes the last known-good copy on disk and reports
a routine `removed` with `report.ok` still true. Tampered content must never be able to
trigger deletion.

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
- Do not rename `ld.skills.integrity_failure`, and do not add a ninth `reason_code` in one language only — both are documented compatibility surfaces. See "The integrity-failure log record" above.
- Do not relax any of the `write_skills` filesystem defenses (local key re-validation, symlink refusal, manifest-authorized destruction, corrupt-manifest fail-closed, atomic `0644` writes). Each is a deliberate security property with abuse-case tests attached.
- Do not make `SkillStore` lookups key-only. Version is part of the lookup identity because a payload holds several versions of one key; a key-only seam cannot express a version-pinned reference.
