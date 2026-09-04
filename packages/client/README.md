# `launchdarkly-ai-server` — Core Client

The core package for the LaunchDarkly AI Python SDK. It owns the LaunchDarkly client lifecycle, telemetry pipeline, all shared types, and the primary entry points that handler packages depend on.

All handler packages (`launchdarkly-ai-*`) depend on this package.

> **Tip:** for the simplest install, use [`launchdarkly-ai`](../ai/README.md) instead. It re-exports this package's full API and is the recommended default for most applications.

## Installation

### Without telemetry

```bash
pip install launchdarkly-ai-server
```

`launchdarkly-server-sdk` is an optional dependency — include it for standard usage, or pass a pre-initialized client to `init_client(client=...)` if you bring your own.

The SDK works fully without the OpenTelemetry packages — feature flags evaluate, handlers run, and LaunchDarkly AI events are tracked. Spans are created as no-ops. If you call `init_client()` without the OTel packages installed, the SDK logs a single warning and continues normally.

### With telemetry (recommended for production)

To export traces to the LaunchDarkly Observability dashboard (or any OTLP-compatible backend), install the `otel` extras group:

```bash
pip install "launchdarkly-ai-server[otel]"
```

No code changes are required — `init_client()` detects the packages at runtime and sets up the tracer provider automatically.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `LD_SDK_KEY` | Yes | LaunchDarkly server-side SDK key |
| `LD_BASE_URI` | No | Override the LaunchDarkly polling base URI (e.g. for staging) |
| `LD_STREAM_URI` | No | Override the streaming URI |
| `LD_EVENTS_URI` | No | Override the events URI |
| `LD_SERVICE_NAME` | No | OTel `service.name` resource attribute (default: `python-sdk`) |
| `LD_ENVIRONMENT` | No | `deployment.environment` resource attribute attached to telemetry |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | OTLP endpoint override (default: LaunchDarkly Observability backend) |

The client uses **lazy initialization**: importing the package does not connect to LaunchDarkly. The singleton is created automatically on the first API call that needs it (`config().invoke()`, `graph().invoke()`, `resolve_graph()`, etc.), as long as `LD_SDK_KEY` is set in the environment.

Call `init_client()` explicitly when you want to:
- Pass SDK or telemetry options programmatically (overriding env vars)
- Initialize at startup before the first AI call (e.g. to avoid latency on the first request)
- Fail fast at boot if `LD_SDK_KEY` is missing

```python
import asyncio
from launchdarkly_ai_server import init_client, shutdown

async def main():
    # Standard path — auto-discovers launchdarkly-server-sdk.
    client = await init_client({
        "sdkKey": "sdk-...",
        "serviceName": "my-service",
        "environment": "production",
    })

    # Or skip init_client() and let the first model/graph call initialize lazily.

    # Flush telemetry, flush LD events, and close the client.
    await shutdown()

asyncio.run(main())
```

| Export | Description |
|---|---|
| `init_client(options?)` | Auto-discover and initialize `launchdarkly-server-sdk`. Optional — the first AI API call triggers lazy init when `LD_SDK_KEY` is set. Returns `Awaitable[LDClientInterface]`. |
| `init_client(client=...)` | **BYOC overload** — accept a pre-initialized `LDClientInterface`. Skips SDK auto-discovery. |
| `get_client()` | Return the initialized `LDClientInterface`. Raises if `init_client` has not completed. |
| `shutdown()` | Flush all events and telemetry, then close the client. Await before process exit. |
| `inspect_config(key, context)` | Read an AI Config variation without invoking the model. Never raises. Returns `{"enabled", "config", "meta"}`. |

### `config(**args)`

The primary entry point for AI config invocations. Accepts either a single handler or a list of handlers and routes to the correct one at invoke-time based on the flag variation's provider and mode.

```python
import asyncio
from launchdarkly_ai_server import config, shutdown
from launchdarkly_ai_openai_messages import create_openai_messages_handler
from launchdarkly_ai_openai_agents import create_openai_agent_handler
from launchdarkly_ai_claude_agents import create_claude_agents_handler

# Single handler — must match the flag variation's provider+mode, or raises.
caller = config(
    key="my-ai-config-flag",
    handler=create_openai_messages_handler(),
    tool_handlers={"my_tool": my_tool_fn},  # optional: tool implementations
)

async def main():
    result = await caller.invoke(
        "What is feature flagging?",
        {"kind": "user", "key": "user-123"},
        {"user_name": "Alice"},             # optional: template substitutions
    )
    print(result.response)  # str
    print(result.usage)     # {"input": ..., "output": ..., "total": ...}

    # Multiple handlers — routing selects the match by provider + mode.
    router = config(
        key="my-ai-config-flag",
        tool_handlers={"search": search_fn},
        handler=[
            create_openai_messages_handler(),  # provides_for: ["OpenAI", "messages"]
            create_openai_agent_handler(),     # provides_for: ["OpenAI", "agent"]
            create_claude_agents_handler(),    # provides_for: ["Anthropic", "agent"]
        ],
    )
    result2 = await router.invoke("Summarize this document", {"kind": "user", "key": "user-123"})
    print(result2.judge_results)  # judge evaluation results when skip_judges=False (default)
    print(result2.track_data)     # run ID, config key, model name, etc.

    # Multi-turn conversation — pass prior turns as history (4th arg after variables).
    history = [
        {"role": "user", "content": "What is feature flagging?"},
        {"role": "assistant", "content": "Feature flagging is a technique for safely releasing features..."},
    ]
    result3 = await caller.invoke("Can you give me an example?", {"kind": "user", "key": "user-123"}, None, history)
    await shutdown()

asyncio.run(main())
```

### `graph(key, **options)`

Runs a multi-agent workflow defined in a LaunchDarkly agent graph flag. The SDK uses a **model-driven router**: it starts at the root node, presents outgoing edges as handoff choices to the model, and follows whichever edge the model selects. The loop terminates when the model produces a final answer, a leaf is reached, a cycle is detected, or the step cap is hit.

Each node runs through the same tracked path as `config().invoke()`, so every node emits its own telemetry and judges. Graph-level `$ld:ai:graph:*` events wrap the full run.

```python
import asyncio
from launchdarkly_ai_server import graph, shutdown
from launchdarkly_ai_claude_agents import create_claude_agents_handler

async def main():
    g = graph(
        "support-graph",
        handlers=[create_claude_agents_handler()],
        tool_handlers={"search": search_fn},
    )

    result = await g.invoke(
        "I was double charged",
        {"kind": "user", "key": "user-123"},
        {"account_tier": "pro"},  # optional variables
    )

    print(result.response)  # final output
    print(result.usage)     # UsageDict with .input, .output, .total
    await shutdown()

asyncio.run(main())
```

`resolve_graph(key, *, context, **options)` returns a `GraphDefinition` without executing it. The definition carries `enabled` so you can branch on a disabled graph before traversing. `graph(...).invoke()` raises if the graph is disabled.

### `Registry` / `global_registry` / `compose`

A `Registry` bundles handlers and tool implementations that can be shared across `config()`, `graph()`, and `resolve_graph()` calls. Pass it as `registry=...`; local `handler`/`tool_handlers` always take precedence.

```python
from launchdarkly_ai_server import Registry, global_registry, compose, config
from launchdarkly_ai_claude_agents import create_claude_agents_handler

# Build a reusable registry
my_registry = Registry(
    handlers=[create_claude_agents_handler()],
    tools={"my_tool": my_tool_fn},
)

# Or register incrementally
my_registry.register(tools={"another_tool": another_fn})

# Use global_registry as a process-wide default
global_registry.register(handlers=[create_claude_agents_handler()])

# Combine two registries — b wins over a on conflict, neither is mutated
combined = compose(my_registry, another_registry)

router = config(key="my-flag", registry=my_registry)
```

### `inspect_config(key, context)`

Reads an AI Config flag variation **without invoking any AI provider**. Use this for health checks, logging, feature-gate probes, or any situation where you need to know whether a config is enabled or what model it points to — without spending API quota.

```python
import asyncio
from launchdarkly_ai_server import inspect_config

async def main():
    result = await inspect_config("my-ai-config-flag", {"kind": "user", "key": "user-123"})

    if not result["enabled"]:
        print("Flag is off — skipping AI call")
    else:
        print(result["config"]["model"]["name"])  # e.g. "claude-opus-4-5"
        print(result["meta"]["variationKey"])

asyncio.run(main())
```

**Guarantees:**
- Never raises — returns `{"enabled": False, "config": None, "meta": None}` on any error (network failure, bad key, schema mismatch, etc.)
- Does not emit LD telemetry events
- Does not call any AI provider
- Lazily initializes the LD client (same as all other entry points)

| Return key | Type | Description |
|---|---|---|
| `enabled` | `bool` | Whether the flag variation is active |
| `config` | `dict \| None` | The parsed AI config, or `None` when disabled or invalid |
| `meta` | `dict \| None` | Variation metadata (key, version, mode), or `None` when unreachable |

---

### Agent Skills

Skills are versioned `SKILL.md` documents managed in LaunchDarkly and attached to AI Config
variations by reference. The SDK surfaces which skills a config references, retrieves their
content, and materializes them onto disk where agent runtimes (Claude Agent SDK, and
anything else following the `<root>/<key>/SKILL.md` convention) discover them.

```python
import asyncio
import hashlib
from pathlib import Path

from launchdarkly_ai_server import (
    init_client, inspect_config, skill_refs, get_skill, write_skills,
    InMemorySkillStore,
)

SKILL_MD = "---\nname: PDF Extraction\n---\nExtract text from PDFs.\n"

async def main():
    # A store supplies skill content. InMemorySkillStore is the dict-backed
    # store for local development, testing, and bring-your-own-content use.
    store = InMemorySkillStore()
    store.put({
        "key": "pdf-extraction",
        "version": 2,
        "content": SKILL_MD,
        # sha256, lowercase hex, over the verbatim utf-8 bytes. Content whose hash
        # does not match is withheld, so this is not optional.
        "contentHash": hashlib.sha256(SKILL_MD.encode("utf-8")).hexdigest(),
    })
    await init_client(options={"skillStore": store})

    # 1. Which skills does this config reference? Pure projection — no I/O.
    info = await inspect_config("doc-agent", {"kind": "user", "key": "user-123"})
    refs = skill_refs(info["config"])          # [SkillReference(key='pdf-extraction', version=2)]

    # 2. Fetch content. Returns None rather than raising when a skill is unavailable.
    skill = await get_skill("pdf-extraction")
    if skill is not None:
        print(skill.content)

    # 3. Write them where the agent runtime will look. Only the leaf directory is
    #    created, so the parent must already exist.
    Path(".claude").mkdir(exist_ok=True)
    report = await write_skills(refs, ".claude/skills")
    for action in report.errors:
        print(f"skill {action.key or '<run>'}: {action.error}")

asyncio.run(main())
```

Pass `"*"` instead of a reference list to materialize every skill the store holds — but know
what you are asking for. `"*"` materializes the **whole project library**, which puts every
skill's `description` into the agent's context, including skills no AI Config references and
skills belonging to other teams. `write_skills(skill_refs(...), root)` is the form used above
because it materializes only what the resolved variation actually asked for; reach for `"*"`
when you genuinely want the whole library on disk.

**`skills` is now a validated field.** Config parsing fails closed on a `skills` value that
is not a list of `{key, version}` objects (key matching `^[a-z0-9][a-z0-9-]*$`, version an
integer ≥ 1): the whole variation is rejected, `inspect_config` returns `config: None`, and
`extract_variation` raises. A variation that previously carried its own custom `skills`
field of a different shape must rename it before upgrading.

**Integrity is not optional.** Content is only returned after its sha256 (lowercase hex,
over the verbatim UTF-8 bytes) matches the delivered `contentHash`, its key and version
revalidate, and its size is within 64 KiB. Anything that fails is withheld and treated as
missing — no unverified content ever reaches your code. A retrieval that withheld anything
logs a count at WARN, so a run that resolved nothing is not silent.

#### Detecting integrity failures

Every withheld skill emits one structured **ERROR** log record on the SDK's own logger
(`launchdarkly_ai_server.skills_core`), designed to be ingested by a SIEM and alerted on.
It is emitted **regardless of how telemetry is configured** — it is not conditional on any
opt-in, and it is the detection path that works when nothing leaves your process.

The message text is the stable event name followed by compact JSON, so it is greppable and
`jq`-able under any handler configuration, and the same mapping is attached as
`extra["ld_skills"]` for a structured handler:

```
ERROR ld.skills.integrity_failure {"action":"withheld","event":"ld.skills.integrity_failure","expected_hash":"0000…0000","language":"python","observed_hash":"5fc8…6ec0","reason":"content hash mismatch","reason_code":"hash_mismatch","skill_key":"pdf-extraction","version":2}
```

**`ld.skills.integrity_failure` is a stability commitment.** It is the string to match on,
it will not be renamed, and the JSON keys are sorted so the line is byte-identical across
LaunchDarkly's AI SDKs for the same input.

| Field | Description |
|---|---|
| `event` | Always `ld.skills.integrity_failure`. |
| `action` | Always `withheld` — the content was not returned to your code. |
| `skill_key` | The skill key, or `<invalid-key>` when the delivered key was itself malformed. |
| `version` | The delivered version. Omitted when it was not a valid version. |
| `expected_hash` | The delivered `contentHash`, or `<not-a-sha256-digest>` when it was not one. Omitted when none was delivered. |
| `observed_hash` | The sha256 the SDK computed. Omitted when the failure happened before anything was hashed. |
| `reason_code` | A stable token naming the failure mode — see below. |
| `reason` | Human-readable detail, including byte counts where relevant. |
| `language` | Always `python`. |

Absent optional fields are **omitted entirely** rather than emitted as `null`, so a field
existence check is meaningful. The skill body, and any attacker-controllable string that
could carry it, never appears in the record; neither does any filesystem path.

| `reason_code` | Meaning |
|---|---|
| `not_an_object` | The delivered object was not a JSON object. |
| `invalid_key` | The key did not match `^[a-z0-9][a-z0-9-]*$` or exceeded 256 characters. |
| `invalid_version` | The version was not an integer ≥ 1. |
| `missing_content` | `content` was absent or not a string. |
| `missing_content_hash` | `contentHash` was absent or not a string. |
| `not_utf8` | The content string had no UTF-8 encoding, so there are no bytes that could have been hashed. |
| `over_size_cap` | The content exceeded the SDK's local size cap. |
| `hash_mismatch` | The computed sha256 did not match the delivered `contentHash`. |

**`hash_mismatch` is the one worth paging on.** The other seven describe a malformed or
truncated payload; a mismatch means content was delivered whose bytes are not the bytes
LaunchDarkly hashed, which is a possible **active-tampering** signal. Alert on it, and
treat `expected_hash` / `observed_hash` as the evidence pair.

#### Failing closed on tampering

The log record above is the operator's surface. `get_skill_result` is the application's:
same retrieval, same verification, same telemetry as `get_skill`, and it reports which of
five outcomes happened instead of collapsing all of them to `None`.

```python
from launchdarkly_ai_server import get_skill_result

outcome = await get_skill_result("pdf-extraction")

if outcome.reason == "integrity_failure":
    # Content was delivered and did not verify. Do not degrade quietly.
    raise SystemExit(f"refusing to start: {outcome.detail}")

if outcome.reason == "store_unavailable":
    # The store could not answer at all. Retry, alert, or carry on with what
    # you already have — but this is an outage, not a revocation.
    print(f"skill retrieval unavailable: {outcome.detail}")
elif outcome.reason in ("absent", "wrong_version"):
    # Nothing was tampered with — this skill is simply not available to you.
    print(f"continuing without a skill: {outcome.detail}")
elif outcome.skill is not None:
    print(outcome.skill.content)
```

| `reason` | Meaning |
|---|---|
| `ok` | A verified skill was returned; `.skill` is set and `.detail` is `None`. |
| `absent` | The store answered, and does not hold that key. |
| `integrity_failure` | Content was delivered and failed verification, so it was withheld. **The one to fail closed on.** |
| `store_unavailable` | The store itself could not answer — it raised. An outage, not a deletion. |
| `wrong_version` | The store answered with a version other than the one asked for, so the answer was withheld. |

`.detail` is human-readable and safe to log or show an operator — it names the key and the
failure mode, and never carries skill content or a filesystem path. Branch on `.reason`,
not on `.detail`. `.skill` is populated only when `.reason == "ok"`. `SkillOutcome` is
frozen, like every other value type here.

**`get_skill` is unchanged.** It still returns `None` for all four failures and still never
raises for one, so no existing caller has to move. The two accessors run the same code path
and differ only in what they report — `get_skill_result` adds no second log record and no
second signal for a failure that already emitted one, so a caller can switch to it without
double-counting anything.

`get_skills` and `all_skills` have no reported form: they still omit entries that could not
be resolved, and a run that omitted anything logs a count at WARN. Retrieve individually
with `get_skill_result` when you need the reason per key.

**Versions are selected, not filtered.** A store may hold several versions of one key at
once, because a delivery payload does: the newest version of every skill, plus every
version a variation currently pins. `get_skill("k", version=1)` asks the store for version
1 and gets it even when a newer one is also held. `all_skills()` and `write_skills("*")`
collapse to one skill per key at its newest version, since `<root>/<key>/SKILL.md` is a
single path.

**The root's parent must exist.** `write_skills` creates the root itself but never its
ancestors, so a typo cannot scatter a directory tree across your project. An absent parent,
a root that is an existing file, and a root that is a symlink each raise `ValueError` —
these are caller errors, distinct from the per-skill `error` actions in the report.

**`write_skills` is deliberately conservative** about your filesystem. It writes only
`<root>/<key>/SKILL.md`, tracks what it owns in a manifest at
`<root>/.launchdarkly-skills.json`, and will overwrite or delete **only** paths that
manifest records. A file you placed yourself is reported as an error and left untouched; it
never writes through a symlink; writes are atomic (temp file, `fsync`, rename) at mode
`0644`; and if the manifest is unreadable it performs no destructive action at all. Removing
a skill from a variation is how revocation works — the next reconcile prunes it.

**Platform bound: the descriptor-pinned guarantee is POSIX-only.** On POSIX every destructive
step — the open, the rename, the unlink — runs relative to a directory descriptor opened
`O_RDONLY|O_DIRECTORY|O_NOFOLLOW` and held for the whole reconcile, so a directory swapped for
a symlink *after* its checks cannot redirect a write or a delete: the descriptor names the
inode that was checked, which closes the swap window rather than narrowing it. Windows has no
`*at()` syscall family, so there `write_skills` falls back to a per-component `lstat` check
taken immediately before each step. That floor is a check-then-use race rather than a closed
window: an attacker who already holds **write permission on the managed root** can still win
it. Windows reparse-point checks (`GetFileAttributesW` / `FILE_FLAG_OPEN_REPARSE_POINT`) are
deliberately not implemented in this release, and Windows is not a tested platform for it —
neither SDK repository has a Windows CI runner. Treat write permission on the managed root as
the security boundary on every platform, and on Windows as the *only* one.

**One exception, and it is what makes a crashed reconcile recoverable.** A file at a managed
path whose bytes are *already byte-identical* to the content LaunchDarkly resolved is
adopted — recorded in the manifest and reported `skipped_current` — rather than refused.
Without that, a process killed after a skill file lands but before the manifest is rewritten
leaves that file managed-but-unrecorded, which is indistinguishable from a file you wrote
yourself, so every later reconcile would refuse it and the skill would stay wedged until
someone intervened. Adoption cannot weaken the guarantee above, because bytes that differ in
any way are still refused and left untouched. Note that an adopted file becomes prunable
like any other managed file — which is the same outcome the crash pre-empted.

**A few keys are legal to an AI Config but not to a filesystem.** A key becomes a single
directory name, so `write_skills` applies bounds of its own on top of the key grammar: no
mainstream filesystem allows a 256-byte path component, and Windows reserves 22 MS-DOS
device names (`con`, `prn`, `aux`, `nul`, `com1`–`com9`, `lpt1`–`lpt9`) that cannot be
directory names there. Either one is a reported `error` action for that skill, and the
rejection is unconditional rather than platform-gated — a managed root written from a Linux
container is routinely read from a Windows host, so the on-disk result must not depend on
which OS ran the write. The keys stay valid everywhere else: an AI Config referencing a skill
named `aux` parses, and its other fields are unaffected. If you have a skill named for a
device, rename it.

#### Receiving skills from LaunchDarkly

`InMemorySkillStore` is for tests and bring-your-own-content. In production, skill content
arrives through `FDv2SkillStore`, which speaks LaunchDarkly's SDK-facing FDv2 delivery
channel — the same `GET /sdk/poll` and `GET /sdk/stream` endpoints the base SDK's FDv2 data
source uses, authenticated with the environment's server-side SDK key.

```python
from launchdarkly_ai_server import FDv2SkillStore, init_client, watch_skills

store = FDv2SkillStore(os.environ["LD_SDK_KEY"]).start()
store.wait_for_skills(timeout=10)
await init_client(options={"skillStore": store})

# Materialize now, and re-materialize whenever delivery changes.
report, watcher = await watch_skills("*", ".claude/skills")
try:
    ...
finally:
    watcher.close()
    store.close()
```

**Nothing above the store changes.** The accessors, integrity verification, and
`write_skills` are transport-agnostic: they see raw objects through the `SkillStore` seam and
cannot tell which store produced them. Everything documented above about verification and
reconcile semantics applies unchanged.

**Server-side only.** Skills are for server-side agent runtimes and skill content is
customer-confidential. A mobile key (`mob-…`) or a client-side environment ID raises from the
constructor.

**Streaming is the default, and it is what makes revocation fast.** A `delete-object` reaches
a live stream in seconds; with `mode="poll"` it arrives within one `poll_interval`. Paired
with `watch_skills`, a revoked skill's `SKILL.md` leaves the disk without a restart. During an
outage the store keeps serving the last content it received and `write_skills`' default
`on_unavailable="keep"` leaves managed files alone — an outage must not read as "everything
was revoked".

**The connection also carries your flags.** A client cannot request only the skill payload,
so a skills-enabled environment delivers flag and segment objects on the same connection.
They are skipped, not evaluated — this store does no evaluation of any kind — and
`diagnostics.objects_ignored` counts them.

> **Beta caveats, worth knowing before you deploy.** Payload signing does not exist on this
> channel yet, so delivery is TLS-only and the content hash establishes self-consistency, not
> origin authenticity. FDv2 is opt-in per account: without it the endpoints return HTTP 403,
> which the store reports as a fatal error naming the setting. `ld-relay` does not speak the
> FDv2 endpoints, so relay-only deployments cannot receive skills.

**If every skill comes back empty, check `diagnostics.hashless_objects`.** Verification
requires `contentHash` on the delivered object and withholds anything without one, so a
nonzero count there means skills are being withheld rather than that the environment has
none. The store logs an error per hashless object and one summary per wholly-hashless
payload, both naming the reason. There is deliberately no fallback that skips verification: a
hash the SDK computed from the content it was handed would certify the content against
itself.

**Total path length is yours to bound, not the SDK's.** The 255-byte bound above is per
*component*; the root is your path, so `<root>` + `<key>` + `/SKILL.md` can still exceed
Windows' 260-character `MAX_PATH` with a perfectly legal key. Choose a short managed root on
Windows.

| Export | Description |
|---|---|
| `skill_refs(config)` | Project a config's `skills` array into `list[SkillReference]`. Pure — no client, store, or network needed. Returns `[]` when absent. |
| `get_skill(key, *, version=None)` | One verified skill, or `None`. `version=None` means newest available; a specific `version` matches exactly. Raises only when no store is configured. |
| `get_skill_result(key, *, version=None)` | The same retrieval, reporting **why**: a frozen `SkillOutcome` with `.skill`, `.reason` (`ok` / `absent` / `integrity_failure` / `store_unavailable` / `wrong_version`), and `.detail`. Use it to fail closed on tampering — see *Failing closed on tampering* above. Raises only when no store is configured. |
| `get_skills(refs)` | Batch form. Accepts `SkillReference` values and bare key strings (string = latest). Results follow input order; missing or unverifiable entries are omitted. |
| `all_skills()` | Every verified skill the store holds, one per key at its newest version. |
| `write_skills(skills, root, *, prune=True, timeout=10.0, on_unavailable="keep")` | Materialize skills under `root`, returning a `ReconcileReport`. `prune` removes formerly-managed skills no longer requested. `on_unavailable="raise"` raises instead of reporting when content cannot be retrieved. Raises `ValueError` for an unusable root, a negative `timeout`, or an unrecognised `on_unavailable`. **Performs synchronous filesystem I/O — see the note below.** |
| `SkillStore` | The structural interface content arrives through: `get_object(kind, key, version=None)`, `all_objects(kind)`, optional `add_listener(kind, fn)`. |
| `InMemorySkillStore(objects=None)` | A dict-backed store with `put(raw)`, for local development and testing. Holds several versions of a key. |
| `FDv2SkillStore(sdk_key, *, base_uri=…, mode="stream", …)` | The delivery transport: a store fed by LaunchDarkly over the SDK-facing FDv2 channel. `start()`, `wait_for_skills(timeout)`, `close()`, `diagnostics`, `failed`; also a context manager. **Server-side only** — a mobile key or client-side environment ID raises. See *Receiving skills from LaunchDarkly* above. |
| `watch_skills(skills, root, …)` | `write_skills` plus a re-reconcile on every delivery change. Returns `(initial report, SkillWatcher)`; close the watcher when done. Revocation then takes effect within `debounce` of arriving rather than at the next restart. |
| `StoreDiagnostics` | What the transport has seen: `payloads_transferred`, `skill_objects_received`, `objects_ignored`, `objects_revoked`, `hashless_objects`, `connection_failures`, `last_error`. |

Configure the store with `init_client(options={"skillStore": store})`. With none configured,
the accessors raise `RuntimeError` explaining what to do and `write_skills` reports the
failure in its report (or raises, with `on_unavailable="raise"`). `shutdown()` clears it.

`ReconcileReport.actions` holds one `ReconcileAction` per outcome — `written`, `updated`,
`skipped_current`, `removed`, or `error` — each carrying `key`, `version`, the resolved
`path`, and `error`. `report.ok` is `True` when no action is an `error`, and
`report.errors` is just the `error` actions, so you rarely need to filter `actions`
yourself. A failure that belongs to the whole run rather than to one skill — an unreadable
manifest, for instance — carries the empty string as its `key`.

The fixed on-disk values are exported too, so you do not have to hardcode them:
`MANIFEST_FILENAME` (`.launchdarkly-skills.json`, handy for a `.gitignore`),
`SKILL_FILENAME`, and `MANIFEST_VERSION`. So are the three closed-set types, for annotating
your own helpers: `ReconcileActionKind` (`written` / `updated` / `skipped_current` /
`removed` / `error`), `OnUnavailable` (`keep` / `raise`), and `SkillOutcomeReason`
(`absent` / `integrity_failure` / `ok` / `store_unavailable` / `wrong_version`).

**`write_skills` blocks.** It is `async` for parity with the other accessors and with the
TypeScript SDK, but it awaits nothing: every read, write, `fsync` and rename runs inline,
so a large reconcile holds the event loop for its duration. Wrap it in
`asyncio.to_thread` if that matters. For the same reason `timeout` is checked between
steps rather than interrupting one already in progress. Reconcile one root at a time,
though: because nothing yields today, a run is atomic against the rest of your loop, and
wrapping it to run concurrently makes two runs against the same root race on the manifest.

`all_objects` returns one entry per `(key, version)` under keys that are **opaque** to the
SDK — identity is read from each object's own `key` and `version` fields, so a store is free
to key its own map however the transport underneath does.

> `Skill.content` is `bytes` — the verified verbatim bytes LaunchDarkly delivered, exactly
> what was hashed. The SDK never parses or interprets them; if you want the frontmatter,
> decode and parse the content on your side.

#### Privilege separation: the agent must not be able to rewrite its own skills

**The recommended deployment runs `write_skills` as a different identity than the agent.**
Reconcile as one user, run the agent as another. Everything the reconcile puts on disk is
owner-write-only, and set explicitly rather than inherited from your umask: skill files and
the manifest at `0644` (via `fchmod` on the descriptor, so it cannot be redirected), the
per-skill `<root>/<key>/` directories at `0755`, and the execute bit never set on anything.
Those modes are only a defense if the two identities actually differ — under a single identity
they describe a directory the agent can freely rewrite.

**What to verify, as the identity that will run the agent.** The SDK cannot check this for you
(see below), so make it a deployment step: confirm the agent's identity has no write access to

- the managed root itself,
- the per-skill directories `<root>/<key>/` and the files `<root>/<key>/SKILL.md`,
- the manifest at `<root>/.launchdarkly-skills.json`.

```bash
# Run as the agent's user. Every line should print DENIED.
root=.claude/skills
for target in "$root" "$root/.launchdarkly-skills.json" "$root"/*/ "$root"/*/SKILL.md; do
  [ -e "$target" ] || continue
  if [ -w "$target" ]; then echo "WRITABLE — fix this: $target"; else echo "DENIED: $target"; fi
done
```

Note that the managed root's own mode is **yours, not the SDK's**: `write_skills` creates only
that one leaf directory and does so with your umask, precisely because the root is a path you
chose. Own it — `chown reconcile-user:agent-group` and `chmod 0755` on the root is the shape
that makes the rest of the tree's modes mean something.

**Why this is the mitigation that matters.** A `SKILL.md` is agent *instructions*. An agent
that can write its own skills directory can rewrite its own instructions, and an agent
processing untrusted input is exactly the thing that might be induced to do so. Write access
to the manifest is worse than write access to a skill, because the manifest is what tells the
*next* reconcile which paths the SDK owns and may delete: an agent that can edit it can keep a
skill LaunchDarkly has revoked, or aim the SDK's own delete path at something it should not
touch. `write_skills` re-validates every manifest entry from scratch for exactly that reason —
it treats that file as untrusted input, never as authorization — but an agent that cannot edit
it at all is the stronger position, and only your deployment can provide that.

**The SDK deliberately does not report whether the root is writable.** There is no such field
on `ReconcileReport`, and its absence is a decision rather than an oversight. The SDK knows
only its own identity, which trivially has write access — it just wrote there. It cannot know
which identity will later run the agent, so any check it could make would answer a different
question than the one that matters, and would read as reassurance exactly where caution is
wanted. You know both identities; the SDK knows one.

---

### Utility Helpers

```python
from launchdarkly_ai_server import parse_template, parse_json_with_possible_fences

# Replaces {{variable}} placeholders, supports dot-notation ({{user.name}})
prompt = parse_template("Hello, {{name}}!", {"name": "Alice"})

# Parses JSON that may be wrapped in ```json fences
data = parse_json_with_possible_fences(model_output)
```

## Shared Types

All types are exported from this package. Handler packages import them from here and never redefine them.

| Type | Description |
|---|---|
| `AiConfigRep` | The AI configuration object fetched from a LaunchDarkly flag variation |
| `Tool` | A tool definition (name, description, JSON Schema parameters) |
| `ProviderHandler` | The callable type that all handler packages produce |
| `ProviderResponse` | The value returned to callers: `response`, `usage`, `track_data`, `judge_results?`, `judge_tasks?`. `judge_results` is populated when `skip_judges=False`; `judge_tasks` (a `list[JudgeTask]`) is populated when `skip_judges=True`. |
| `ConfigArgs` | Arguments accepted by `config()` (key, handler, tool_handlers, registry) |
| `NativeTool` | Marker class for provider built-in tools |
| `LDContext` | Standard LaunchDarkly context dict. Import from `launchdarkly_ai_server`. |
| `GraphOptions` | Options accepted by `graph()` (handlers, tool_handlers, graph_judge — no context) |
| `GraphDefinition` | A resolved agent graph: topology accessors, `run_node`, and the traverse primitives (attribute access, e.g. `gd.enabled`, `gd.get_node(key)`) |
| `GraphNode` / `GraphEdge` | A dataclass node (`.key`, `.config`, `.meta`, `.edges`, `.is_terminal`) and a dataclass directed edge (`.key`, `.source_key`, `.target_key`, `.handoff`) |
| `ProviderGraphResponse` | A dataclass returned by `graph(...).invoke()`: `.response`, `.usage`, `.judge_results` |
| `GraphTopology` | The parsed graph flag shape (`root` + `edges`) |
| `Skill` | A frozen skill document: `.key`, `.version`, `.content` (verified verbatim `bytes`), `.content_hash`, `.name?`, `.description?` |
| `SkillReference` | A frozen version-pinned pointer to a skill: `.key`, `.version` |
| `SkillOutcome` | A frozen retrieval outcome: `.skill`, `.reason` (`SkillOutcomeReason`), `.detail` |
| `ReconcileAction` | One `write_skills` outcome: `.key`, `.action`, `.version?`, `.path?`, `.error?` |
| `ReconcileReport` | The `write_skills` result: `.actions`, `.ok`, and `.errors` |
