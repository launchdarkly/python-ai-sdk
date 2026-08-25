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

Pass `"*"` instead of a reference list to materialize every skill the store holds.

**`skills` is now a validated field.** Config parsing fails closed on a `skills` value that
is not a list of `{key, version}` objects (key matching `^[a-z0-9][a-z0-9-]*$`, version an
integer ≥ 1): the whole variation is rejected, `inspect_config` returns `config: None`, and
`extract_variation` raises. A variation that previously carried its own custom `skills`
field of a different shape must rename it before upgrading.

**The root's parent must exist.** `write_skills` creates the root itself but never its
ancestors, so a typo cannot scatter a directory tree across your project. An absent parent,
a root that is an existing file, and a root that is a symlink each raise `ValueError` —
these are caller errors, distinct from the per-skill `error` actions in the report.

**Integrity is not optional.** Content is only returned after its sha256 (lowercase hex,
over the verbatim UTF-8 bytes) matches the delivered `contentHash`, its key and version
revalidate, and its size is within 64 KiB. Anything that fails is withheld and treated as
missing — no unverified content ever reaches your code.

**`write_skills` is deliberately conservative** about your filesystem. It writes only
`<root>/<key>/SKILL.md`, tracks what it owns in a manifest at
`<root>/.launchdarkly-skills.json`, and will overwrite or delete **only** paths that
manifest records. A file you placed yourself is reported as an error and left untouched; it
never writes through a symlink; writes are atomic (temp file, `fsync`, rename) at mode
`0644`; and if the manifest is unreadable it performs no destructive action at all. Removing
a skill from a variation is how revocation works — the next reconcile prunes it.

| Export | Description |
|---|---|
| `skill_refs(config)` | Project a config's `skills` array into `list[SkillReference]`. Pure — no client, store, or network needed. Returns `[]` when absent. |
| `get_skill(key, *, version=None)` | One verified skill, or `None`. `version=None` means newest available; a specific `version` matches exactly. Raises only when no store is configured. |
| `get_skills(refs)` | Batch form. Accepts `SkillReference` values and bare key strings (string = latest). Results follow input order; missing or unverifiable entries are omitted. |
| `all_skills()` | Every verified skill the store holds. |
| `write_skills(skills, root, *, prune=True, timeout=10.0, on_unavailable="keep")` | Materialize skills under `root`, returning a `ReconcileReport`. `prune` removes formerly-managed skills no longer requested. `on_unavailable="raise"` raises instead of reporting when content cannot be retrieved. Raises `ValueError` for an unusable root, a negative `timeout`, or an unrecognised `on_unavailable`. **Performs synchronous filesystem I/O — see the note below.** |
| `SkillStore` | The structural interface content arrives through: `get_object(kind, key)`, `all_objects(kind)`, optional `add_listener(kind, fn)`. |
| `InMemorySkillStore(objects=None)` | A dict-backed store with `put(raw)`, for local development and testing. |

Configure the store with `init_client(options={"skillStore": store})`. With none configured,
the three accessors raise `RuntimeError` explaining what to do; `write_skills` reports the
failure in its report (or raises, with `on_unavailable="raise"`).

`ReconcileReport.actions` holds one `ReconcileAction` per outcome — `written`, `updated`,
`skipped_current`, `removed`, or `error` — each carrying `key`, `version`, the resolved
`path`, and `error`. `report.ok` is `True` when no action is an `error`, and
`report.errors` is just the `error` actions, so you rarely need to filter `actions`
yourself. A failure that belongs to the whole run rather than to one skill — an unreadable
manifest, for instance — carries the empty string as its `key`.

The fixed on-disk and on-the-wire values are exported too, so you do not have to hardcode
them: `MANIFEST_FILENAME` (`.launchdarkly-skills.json`, handy for a `.gitignore`),
`SKILL_FILENAME`, `MANIFEST_VERSION`, and `SKILL_OBJECT_KIND`. So are the two closed-set
types, for annotating your own helpers: `ReconcileActionKind` (`written` / `updated` /
`skipped_current` / `removed` / `error`) and `OnUnavailable` (`keep` / `raise`).

**`write_skills` blocks.** It is `async` for parity with the other accessors and with the
TypeScript SDK, but it awaits nothing: every read, write, `fsync` and rename runs inline,
so a large reconcile holds the event loop for its duration. Wrap it in
`asyncio.to_thread` if that matters. For the same reason `timeout` is checked between
steps rather than interrupting one already in progress. Reconcile one root at a time,
though: because nothing yields today, a run is atomic against the rest of your loop, and
wrapping it to run concurrently makes two runs against the same root race on the manifest.

> `Skill.frontmatter()` parses the leading YAML block as a convenience. It needs `pyyaml`,
> which is **not** a dependency of this package — install it yourself if you want that
> method to return anything. It returns `None` rather than raising when the library is
> missing, the block is absent or oversize, or the YAML is hostile.

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
| `Skill` | A frozen `SKILL.md` document: `.key`, `.version`, `.content`, `.content_hash`, `.name?`, `.description?`, and `.frontmatter()` |
| `SkillReference` | A frozen version-pinned pointer to a skill: `.key`, `.version` |
| `ReconcileAction` | One `write_skills` outcome: `.key`, `.action`, `.version?`, `.path?`, `.error?` |
| `ReconcileReport` | The `write_skills` result: `.actions`, `.ok`, and `.errors` |
