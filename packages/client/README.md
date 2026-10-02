# `launchdarkly-ai-server` — Core Client

The core package for the LaunchDarkly AI Python SDK. It owns the LaunchDarkly client lifecycle, telemetry pipeline, all shared types, and the primary entry points that handler packages depend on.

All handler packages (`launchdarkly-ai-*`) depend on this package.

> **Tip:** for the simplest install, use [`launchdarkly-ai-python`](../ai/README.md) instead. It re-exports this package's full API and is the recommended default for most applications.

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
| `LD_API_TOKEN` | For evaluations | API access token used by the evaluations management API |
| `LD_SDK_KEY` | For evaluations | SDK key whose event transport carries generation results to LaunchDarkly |
| `LD_API_BASE_URI` | No | Evaluations management API host override; intentionally separate from `LD_BASE_URI` |
| `LD_UI_BASE_URI` | No | LaunchDarkly application host for evaluation-run links (default: `https://app.launchdarkly.com`). Set it for a non-production project, or its runs still link to the production app |

### Run an evaluation from code

The evaluations harness reads an LD-hosted dataset, creates a new evaluation and API-source run, and invokes your handler once per row. Rows can then be scored by LaunchDarkly judges and local scorer functions; see [Score rows with judges and scorers](#score-rows-with-judges-and-scorers). Each success or error queues a `$ld:ai:offline-evals:generation` custom event containing the evaluation, run, dataset, and row identifiers plus output or error (`errorMessage` is included for `ERROR` rows), nested `usage.inputTokens`/`usage.outputTokens`, timing, and stable hashes. Dataset-owned input, expected output, metadata, and variables are not duplicated in the event. Each queued event is logged at `INFO` on the `launchdarkly_ai_server.evaluations.runner` logger with its RFC3339 UTC `emittedAt` timestamp and stable `eventId`, making it possible to compare SDK emission time with ClickHouse arrival time once that logger is enabled. The same `emittedAt` value is included in the event payload. Events are flushed before the summary is fetched and the call returns; handlers are never rerun to retry event delivery. Pass/fail is derived from LaunchDarkly's run summary.

Result links use `ui_base_uri`, then `LD_UI_BASE_URI`, then `https://app.launchdarkly.com`; this is independent of `LD_API_BASE_URI`. After flushing generation events, the harness polls the run summary endpoint until passed + failed + error rows fully account for a nonzero total with no pending rows, polling every `poll_interval_seconds` (default 2s) up to `poll_timeout_seconds` (default 180s); pass either to `run()` to widen both for large datasets. The summary endpoint does not return run state, so `RunSummary` exposes row counts only. A run passes only when the completed summary has no failed, error, or pending rows. `failed_rows` counts rows whose criteria were scored and did not meet their threshold, so a gate that ignored it would exit 0 on a run where every row failed its judge. Evaluation keys must be unique because every call creates a new evaluation with `POST`.

```python
import asyncio
import sys

from launchdarkly_ai_openai_messages import create_openai_messages_handler
from launchdarkly_ai_server import init_evaluations


async def main() -> int:
    evals = init_evaluations()  # LD_API_TOKEN required; LD_SDK_KEY unless a client is already initialized
    result = await evals.run(
        project_key="my-project",
        key="support-qa-2026-08-20",
        dataset="support-golden",
        handler=create_openai_messages_handler(),
        generation={
            "provider": "OpenAI",
            "model": "gpt-4o",
            "instructions": "You are a support agent.",
        },
    )
    print(result.url, result.summary)
    return 0 if result.passed else 1


sys.exit(asyncio.run(main()))
```

`project_key` is supplied per run rather than during initialization. `generation.instructions` is shorthand for one system message; use `generation.messages` instead for a full message list, but do not supply both. The harness never retries a handler invocation because doing so could repeat tool side effects. Its retries apply only to LaunchDarkly management API requests.

Generation and criterion events are the only path by which row results reach LaunchDarkly, so `init_evaluations()` raises rather than creating a run that can never complete unless it can resolve an event transport: either an SDK key (`sdk_key` or `LD_SDK_KEY`) or a client already initialized through `init_client(client=...)`. Bringing your own client lets a process emit evaluation events without an SDK key in scope. Every generated row is emitted and flushed unconditionally; no feature flag gates event publishing. The harness then polls the summary endpoint until row accounting shows processing is complete.

### Score rows with judges and scorers

Pass `criteria` to `run()` to score every generated row. A `Judge` references an AI Judge config that already exists in LaunchDarkly — the SDK creates no judges and ships none of its own — and a `Scorer` wraps a local function, so a run can mix model-graded and deterministic checks. Each criterion runs once per generated row, bounded by the same `concurrency` as generation, and emits one `$ld:ai:offline-evals:criterion` event per `(row, criterion)` carrying the criterion identity, the judge's variation key and version, the validated score, its reason, usage, and timings.

```python
from launchdarkly_ai_claude_messages import create_claude_messages_handler
from launchdarkly_ai_openai_messages import create_openai_messages_handler
from launchdarkly_ai_server import DatasetRow, Judge, Scorer, init_evaluations


def mentions_policy(row: DatasetRow, output: str | None) -> bool:
    return "refund policy" in (output or "").lower()


result = await init_evaluations().run(
    project_key="my-project",
    key="support-qa-2026-08-20",
    dataset="support-golden",
    handler=create_openai_messages_handler(),
    generation={"provider": "OpenAI", "model": "gpt-4o"},
    criteria=[
        Judge(key="accuracy-judge", threshold=0.8),
        Scorer(name="mentions-policy", fn=mentions_policy),
    ],
    # Needed only because this judge is served by a different provider than
    # the generation config above.
    judge_handlers=[create_claude_messages_handler()],
)
```

`Scorer.fn` receives the `DatasetRow` the output was generated from plus the generated output, may be sync or async, and must return a bool or a number from 0 to 1; booleans become 1.0 or 0.0. `Judge.threshold` defaults to 0.5 and `Scorer.threshold` to 1.0 — a perfect score, which is what a boolean scorer wants — and both accept an optional `pass_rate_threshold`. Judge keys and scorer names share one `criterionType` namespace and must be unique within a run, case-insensitively, because that name is part of each result's deterministic event identity. `Judge.ground_truth_context` overrides what the judge is graded against when the dataset row's expected output is not it.

**The SDK reports scores and never rules on them.** LaunchDarkly derives each row's verdict at ingest by comparing the score against the criterion's stored threshold and success direction, so pass/fail policy is one server-side implementation that applies to every SDK version and to runs already recorded. A judge's direction lives on its AI Config and is injected server-side, keeping the one input a verdict turns on server-attested; a `Scorer` has no LaunchDarkly-side config to read, so it declares its own `success_direction` (default `"higher_is_better"` — set `"lower_is_better"` for a scorer that counts something unwanted, like a regex hit count).

#### Judge the tool trajectory

A judge is shown the tool calls the row made on the way to its output, so a rubric can grade *how* the agent answered and not only *what* it answered — whether it called the right tool, in the right order, with the right arguments, and how it handled a tool that failed.

The harness records this itself: it wraps your tool implementations once per row before handing them to the handler, so every handler package is covered without changes and your tools still return and raise exactly what they did before.

**This is not specific to offline evaluations.** Online judges — both the inline ones sampled by `config().invoke()` and the deferred ones you run from a `JudgeTask` on a background thread — are shown the same trajectory, built by the same function. See [Judges see one conversation](#judges-see-one-conversation).

The trajectory is rendered into **`{{message_history}}`** — the row input, then the trajectory, then the generated output, then the formatting instructions, in that order. There is no separate trajectory variable: `message_history` is already the transcript variable every judge reads, and judges built from the AI Library's default templates reference it, so a trajectory rubric can be written against an existing judge template with no new placeholder.

```
Tools available: lookup_order, issue_refund
Tool calls made while producing the response, in order:
1. lookup_order
   arguments: {"id":"A1"}
   result: order A1 shipped 2026-08-02
2. issue_refund
   arguments: {"id":"A1","amount":19.99}
   error: refund window closed
```

A row with tools that called none of them says so explicitly, which is the finding a tool-selection rubric most needs. A run with no tools adds no block at all, so judges written before trajectories existed read exactly the history they read before.

#### Judges see one conversation

All three judge paths build `{{message_history}}` through a single function, `judge_scoring.build_message_history`:

| Path | Entry point |
| --- | --- |
| Online, inline | `config().invoke()` → `run_judges` |
| Online, deferred | `config(skip_judges=True).invoke()` → `run_judge(task, handlers)` on your own thread |
| Offline | `init_evaluations().run(criteria=[Judge(...)])` |

Each one is the input, then the tool trajectory, then the output, then the `{score, reasoning}` format block, with empty parts skipped. A judge therefore grades the same conversation wherever it runs, which is what makes a rubric portable between a production sample and a dataset replay.

They did not always agree, and that is why this is a single function now: each path used to join its own history. The offline one carried the row input, the inline one carried the user input, and the deferred one carried **neither** — so a deferred judge graded a response with no request beside it. `JudgeTask` gained `user_input` and `trajectory` to close that.

For the deferred path those two fields travel on the task, which stays picklable — the trajectory crosses as the rendered string, not the structured record.

A **graph-level** judge (`graph_judge`) gets no trajectory: it grades a final answer produced across several nodes, and splicing their trajectories together would describe a conversation that never happened. Per-node judges inside a graph do get their own node's.

Two limits keep a trajectory from spending the judge's context window: at most 50 recorded calls per row and 2000 characters per rendered argument bag or result, with anything beyond either reported as a count or marked truncated. Calls past the limit still execute — truncation drops the record, never the work. A `NativeTool` runs inside the provider, so no local wrapper sees it; such a tool is left out of the trajectory and out of the "Tools available" line, since naming a tool whose use cannot be shown would invite a judge to conclude the model ignored it.

A tool result is now judge-prompt input. It stays literal for the same reason the generated output does: the judge config is handed to the handler unrendered and the handler makes exactly one template pass, so a `{{...}}` sequence coming back from a tool is never expanded into the judge prompt.

**Judges are independent AI Configs, so handlers are routed per judge.** A judge may resolve to a different provider or mode than `generation`, and a handler built for one provider cannot execute another's config. `handler` runs a judge when it provides for that judge's provider; pass handlers for any other providers in `judge_handlers`. Selection prefers a handler naming the judge's provider outright over a wildcard multi-provider adapter, and an agent-mode handler can serve a messages-mode judge with its messages collapsed into one instructions block. A plain callable that declares no `provides_for` routes itself, exactly as it already does for the generation config.

Judges are resolved through flag delivery, and handlers are matched to them, **before** any evaluation records are created — a missing judge or one no handler covers fails the run up front rather than after the generation spend. After that point a criterion failure never aborts the run: an unparseable judge response, an out-of-range score, a raising handler or scorer, and a row whose generation errored each become a per-criterion `ERROR` event with a cause code (`invalid_judge_output`, `invalid_score`, `handler_raised`, `scorer_raised`, `generation_incomplete`) and a top-level `errorMessage`. Event *delivery* is different: the backend needs one result per `(row, criterion)` to finish row accounting, so if tracking a criterion event fails, every remaining result is still attempted and flushed and then `run()` raises — rather than polling to its timeout with the cause hidden.

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
variations by reference. The SDK tells you which skills a config references, retrieves their
content, and writes them to `<root>/<key>/SKILL.md`, where agent runtimes such as the Claude
Agent SDK discover them.

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
        # sha256, lowercase hex, over the verbatim utf-8 bytes. Content whose
        # hash does not match is withheld.
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

**Know what `"*"` asks for.** Passing `"*"` instead of a reference list writes the **whole
project library**, which puts every skill's `description` into the agent's context, including
skills no AI Config references and skills belonging to other teams. `write_skills(skill_refs(...), root)`,
as above, writes only what the resolved variation asked for.

**`skills` is a validated field.** Config parsing fails closed on a `skills` value that is not
a list of `{key, version}` objects (key matching `^[a-z0-9][a-z0-9-]*$`, version an integer
≥ 1): the whole variation is rejected, `inspect_config` returns `config: None`, and
`extract_variation` raises. If your variations carry a custom `skills` field of a different
shape, rename it before upgrading.

**Integrity is not optional.** Content is returned only when its sha256 (lowercase hex, over
the verbatim UTF-8 bytes) matches the delivered `contentHash`, its key and version revalidate,
and it is at most 10 MiB. Anything else is withheld and treated as missing. A retrieval that
withheld anything logs a count at WARN.

**Versions are selected, not filtered.** A store can hold several versions of one key: the
newest of every skill, plus every version a variation pins. `get_skill("k", version=1)` returns
version 1 even when a newer one is held. `all_skills()` and `write_skills("*")` take the newest
version per key, since `<root>/<key>/SKILL.md` is a single path.

**The root's parent must exist.** `write_skills` creates the root but never its ancestors, so a
typo cannot scatter directories across your project. A missing parent, a root that is a file,
or a root that is a symlink raises `ValueError`. These are caller errors, distinct from the
per-skill `error` actions in the report.

**`write_skills` is a reconcile, not a copy.** It writes only `<root>/<key>/SKILL.md`, records
what it owns in a manifest at `<root>/.launchdarkly-skills.json`, and overwrites or deletes
**only** paths that manifest records.

- A file you placed yourself is reported as an error and left untouched.
- It never writes through a symlink.
- Writes are atomic (temp file, `fsync`, rename) at mode `0644`.
- An unreadable manifest suppresses every destructive action.
- Revocation is pruning: a skill removed from a variation is deleted on the next reconcile.

**One exception: byte-identical files are adopted.** A file at a managed path whose bytes
already equal the resolved content is recorded in the manifest and reported `skipped_current`
instead of refused. This lets a reconcile that crashed after writing a skill file, but before
rewriting the manifest, recover on the next run. A file that differs in any way is still
refused. An adopted file is prunable like any other managed file.

**Platform bound: the descriptor-pinned guarantee is POSIX-only.**

- **POSIX: the swap window is closed.** The managed root is opened once per reconcile
  (`O_RDONLY|O_DIRECTORY|O_NOFOLLOW`, checked with `fstat`) and held until the call returns.
  Every read and write under it (per-skill directories, the manifest, comparisons against an
  existing `SKILL.md`, temp-file cleanup, unlink and `rmdir`) runs relative to that
  descriptor, so a directory swapped for a symlink after its checks cannot redirect or alter a
  write or delete. Ancestors are covered only once the root is open, which is why the
  [checklist below](#privilege-separation-the-agent-must-not-be-able-to-rewrite-its-own-skills)
  denies the agent write access to every ancestor.
- **Windows: the window is narrowed, not closed.** There is no `*at()` syscall family, so
  `write_skills` falls back to a per-component `lstat` check immediately before each step — a
  check-then-use race that an attacker with **write permission on the managed root** can win.
  Reparse-point checks (`GetFileAttributesW` / `FILE_FLAG_OPEN_REPARSE_POINT`) are not
  implemented, and Windows is not a supported or tested platform for this feature.

Treat write permission on the managed root **or any directory above it** as the security
boundary on every platform, and on Windows as the *only* one.

**Some valid keys cannot be directory names.** Each key becomes one directory name, so
`write_skills` rejects, as a per-skill `error` action, a key over 255 bytes and the 22 Windows
reserved device names (`con`, `prn`, `aux`, `nul`, `com1`–`com9`, `lpt1`–`lpt9`). The check
runs on every OS, because a root written from a Linux container is often read from a Windows
host. The key stays valid everywhere else: an AI Config referencing `aux` still parses. If you
have a skill named for a device, rename it.

**Total path length is yours to bound.** The 255-byte limit is per path component, so
`<root>/<key>/SKILL.md` can still exceed Windows' 260-character `MAX_PATH`. Choose a short
managed root on Windows.

#### Detecting integrity failures

Every withheld skill emits one **ERROR** log record on the SDK's logger
(`launchdarkly_ai_server.skills_core`), suitable for SIEM ingestion and alerting. It is
emitted **regardless of telemetry configuration**, so it works even when nothing leaves your
process.

The message is the event name followed by compact JSON, so it is greppable and `jq`-able under
any handler. The same mapping is attached as `extra["ld_skills"]` for structured handlers:

```
ERROR ld.skills.integrity_failure {"action":"withheld","event":"ld.skills.integrity_failure","expected_hash":"0000…0000","language":"python","observed_hash":"5fc8…6ec0","reason":"content hash mismatch","reason_code":"hash_mismatch","skill_key":"pdf-extraction","version":2}
```

**`ld.skills.integrity_failure` is a stability commitment.** Match on it; it will not be
renamed. JSON keys are sorted, so the line is byte-identical across LaunchDarkly's AI SDKs for
the same input.

| Field | Description |
|---|---|
| `event` | Always `ld.skills.integrity_failure`. |
| `action` | Always `withheld` — the content was not returned to your code. |
| `skill_key` | The skill key **requested**, or `<invalid-key>` when the key was itself malformed. |
| `served_key` | Only on `key_mismatch`: the key the store answered under, redacted like `skill_key`. |
| `served_version` | Only on `version_mismatch`: the version the store answered with, as an integer, or `<invalid-version>`. Never on the same record as `served_key`. |
| `version` | The delivered version, or on `version_mismatch` the version **requested**. Omitted when not a valid version, and on `key_mismatch`. |
| `expected_hash` | The delivered `contentHash`, or `<not-a-sha256-digest>` when it was not one. Omitted when none was delivered. |
| `observed_hash` | The sha256 the SDK computed. Omitted when the failure happened before hashing. |
| `reason_code` | A stable token naming the failure mode — see below. |
| `reason` | Human-readable detail, including byte counts where relevant. |
| `language` | Always `python`. |

Optional fields are **omitted, never `null`**, so an existence check is meaningful. The record
never contains the skill body, any attacker-controllable string that could carry it, or a
filesystem path.

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
| `key_mismatch` | The store answered under a different key than requested. Adds `served_key`; records **no** `AgentControl Skill Integrity Failure` signal. |
| `version_mismatch` | The store answered a version pin with a different version. Adds `served_version` (`version` is the one requested); records **no** `AgentControl Skill Integrity Failure` signal. |

**Page on `hash_mismatch`.** It means the delivered bytes are not the bytes LaunchDarkly
hashed, a possible sign of **active tampering**. Treat `expected_hash` / `observed_hash` as
the evidence pair. Most other codes indicate a malformed or truncated payload. If all your
content comes from LaunchDarkly, `over_size_cap` and `not_utf8` should never occur and are
worth alerting on too.

**`key_mismatch` and `version_mismatch` skip the product signal.** Both are detected after
verification passes, and the usual cause is a bug in a custom `SkillStore` adapter (a stale
cache entry, a colliding key, a wrong index lookup) rather than tampering, so neither inflates
LaunchDarkly's integrity counter. Both still write this record, so a rule on
`ld.skills.integrity_failure` catches them. If `FDv2SkillStore` is your only store, treat them
like `hash_mismatch`; behind a custom adapter, suspect the adapter first.

`get_skill` returns `None` for a `version_mismatch` like any other failure, so this record is
the only place it is visible unless you use `get_skill_result`, which reports it as the
`wrong_version` outcome (below).

#### Failing closed on tampering

The log record is for operators; `get_skill_result` is for your application. It runs the same
retrieval, verification, and telemetry as `get_skill`, but reports which of five outcomes
happened instead of collapsing them all to `None`.

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
| `integrity_failure` | Content was delivered and did not verify, or the store answered under a different key (`reason_code: key_mismatch`), so it was withheld. **The one to fail closed on.** |
| `store_unavailable` | The store itself could not answer — it raised. An outage, not a deletion. |
| `wrong_version` | The store answered with a version other than the one asked for, so the answer was withheld. Also logged, as `reason_code: version_mismatch`. |

`.detail` is human-readable and safe to log or show an operator: it names the key and the
failure mode, never skill content or a filesystem path. Branch on `.reason`, not `.detail`.
`.skill` is set only when `.reason == "ok"`. `SkillOutcome` is frozen.

**`get_skill` is unchanged.** It still returns `None` for all four failures and never raises
for one. Both accessors run the same code path; `get_skill_result` adds no second log record
or signal, so switching to it does not double-count anything.

`get_skills` and `all_skills` have no outcome-reporting form: they omit entries that could not
be resolved and log a count at WARN. Use `get_skill_result` per key when you need the reason.

#### Receiving skills from LaunchDarkly

`InMemorySkillStore` is for tests and bring-your-own-content. In production, skill content
arrives through `FDv2SkillStore`, which uses LaunchDarkly's SDK-facing FDv2 delivery channel
(the `GET /sdk/poll` and `GET /sdk/stream` endpoints the base SDK's FDv2 data source uses),
authenticated with the environment's server-side SDK key.

```python
import os

from launchdarkly_ai_server import FDv2SkillStore, init_client, watch_skills

store = FDv2SkillStore(os.environ["LD_SDK_KEY"]).start()
if not store.wait_for_skills(timeout=10):
    # No payload arrived. Reconciling now would find an empty store; see below.
    print(f"skill delivery has not answered yet: {store.failed or 'still waiting'}")
await init_client(options={"skillStore": store})

# Materialize now, and re-materialize whenever delivery changes.
report, watcher = await watch_skills("*", ".claude/skills")
try:
    ...
finally:
    watcher.close()
    store.close()
```

**A reconcile that runs before delivery answers does not prune.** A store still waiting for its
first payload looks the same as an environment with no skills, and `write_skills("*")` would
otherwise treat that as every skill revoked and delete the files from a previous run.
`FDv2SkillStore` reports readiness through the optional `is_initialized()`; until it is true, a
reconcile reports the retrieval unavailable (`report.ok` is `False`, and the error says why)
and leaves the disk alone.

**A 422 means this connection will never be assigned a skill payload, and delivery stops.**
Every request declares the payload it wants (`kinds=agent-skill`), and LaunchDarkly answers
HTTP 422 when it will not serve one.

- **Causes:** a view-scoped SDK key, which cannot be assigned a skill payload (use a key that
  is not view-scoped), or Agent Skills delivery not being enabled for your account (contact
  LaunchDarkly support).
- **What the store does:** retrying cannot help, so it gives up. `failed` carries the reason,
  `last_error` is set, and `wait_for_skills` returns `False` immediately instead of at your
  timeout. The 422 does not count toward `connection_failures`, which tracks recoverable
  failures only.
- **Not the empty case:** an environment with zero skills is served an empty payload that
  commits normally.
- **Recovery:** once the cause is fixed, call `start()` on the same store. Only `close()` is
  final: a restart clears `failed`, resets the backoff, and held content stays readable
  throughout. Restarting the process also works.

**Nothing above the store changes.** The accessors, verification, and `write_skills` see raw
objects through the `SkillStore` interface and cannot tell which store produced them.

**Server-side only.** Skills are for server-side agent runtimes and skill content is
customer-confidential. A mobile key (`mob-…`) or a client-side environment ID raises from the
constructor.

**The SDK key goes only where you pointed it.** `base_uri` and `stream_uri` must be `https://`
(plain `http://` is allowed only to a loopback host, for a local test double). Redirects are
never followed, so a 3xx stops delivery instead of forwarding the key to the `Location` host.

**Polling and streaming have separate hosts.** By default `/sdk/poll` goes to
`https://sdk.launchdarkly.com` and `/sdk/stream` to `https://stream.launchdarkly.com`. Pass
`base_uri` alone to use one host for both, or `stream_uri` as well to set them independently.

**`close()` is final.** A closed store still answers from the content it received, but
`start()` afterwards raises, so delivery is guaranteed stopped even if the join times out.
Construct a new store to resume.

**Reads are memory-bounded.** A poll body or streamed event larger than `MAX_RESPONSE_BYTES`
(64 MiB) is dropped without being applied, and delivery stops: the payload's size belongs to the
environment, so a retry would download it again and be refused the same way. `failed` carries
the reason, the store keeps serving what it last held, and `start()` resumes delivery once the
payload is back under the bound.

**Streaming is the default, and it is what makes revocation fast.** A `delete-object` reaches a
live stream in seconds; with `mode="poll"` it arrives within one `poll_interval`. With
`watch_skills("*", ...)`, a revoked skill's `SKILL.md` leaves the disk without a restart. During
an outage the store keeps serving its last content and retries for as long as it runs, and
`write_skills`' default `on_unavailable="keep"` leaves managed files alone, so an outage does
not read as "everything was revoked".

**With an explicit skill list, revocation does not reach the disk.** Given a list such as
`skill_refs(config)`, a skill deleted in LaunchDarkly is reported as an `error` action under
its key, and its `SKILL.md` is kept rather than pruned. The watcher also listens only to the
skill store, not to flag changes, so unpinning a skill from a variation is not seen either. Use
`"*"` when revocation must reach the disk.

**Without the watcher, the revocation bound is process lifetime.** If you call `write_skills`
once at boot and never run `watch_skills`, a skill revoked after boot stays on disk, and in the
agent's context, until the process reconciles again. To pull a skill immediately, restart or
re-run `write_skills`. Neither recalls content an agent has already read into a conversation.

**One network timeout, and its default depends on the mode.** `read_timeout` bounds every socket
operation of a request, connecting included. In `mode="poll"` it bounds the whole request
(default 10 seconds); in `mode="stream"` it bounds each wait for the next bytes (default 300
seconds, well beyond LaunchDarkly's heartbeat interval).

**The connection carries only skills.** Every request declares the skill payload, so flag and
segment objects do not arrive on it. Any other object kind is skipped, not rejected, and
counted in `diagnostics.objects_ignored`; a nonzero count means the payload has a kind this
version does not recognise, not that something failed.

> **Beta caveats, worth knowing before you deploy.** Payload signing does not exist on this
> channel yet, so delivery is TLS-only and the content hash establishes self-consistency, not
> origin authenticity. The FDv2 protocol is opt-in per account: without it the endpoints
> return HTTP 403, which the store reports as a fatal error explaining what to do. `ld-relay`
> does not speak the FDv2 endpoints, so relay-only deployments cannot receive skills.

**If every skill comes back empty, check `diagnostics.hashless_objects`.** Objects without a
`contentHash` are withheld, so a nonzero count means skills are being withheld, not that the
environment has none. The store also logs an error per hashless object. There is no fallback
that skips verification.

| Export | Description |
|---|---|
| `skill_refs(config)` | Project a config's `skills` array into `list[SkillReference]`. Pure — no client, store, or network needed. Returns `[]` when the field is absent. A `skills` field that is present but not a list (including `null`) fails the config parse instead, so an unreadable field never reaches a pruning reconcile as "no skills". |
| `get_skill(key, *, version=None)` | One verified skill, or `None`. `version=None` means newest available; a specific `version` matches exactly. Raises only when no store is configured. |
| `get_skill_result(key, *, version=None)` | The same retrieval, reporting **why**: a frozen `SkillOutcome` with `.skill`, `.reason` (`ok` / `absent` / `integrity_failure` / `store_unavailable` / `wrong_version`), and `.detail`. See *Failing closed on tampering* above. Raises only when no store is configured. |
| `get_skills(refs)` | Batch form. Accepts `SkillReference` values and bare key strings (string = latest). Results follow input order; missing or unverifiable entries are omitted. |
| `all_skills()` | Every verified skill the store holds, one per key at its newest version. |
| `write_skills(skills, root, *, prune=True, timeout=10.0, on_unavailable="keep")` | Materialize skills under `root`, returning a `ReconcileReport`. `prune` removes formerly-managed skills no longer requested. `on_unavailable="raise"` raises instead of reporting when content cannot be retrieved. Raises `ValueError` for an unusable root, a negative or non-finite `timeout`, or an unrecognised `on_unavailable`. `timeout=0` is valid and makes every skill report an error. **Performs synchronous filesystem I/O — see the note below.** |
| `SkillStore` | The structural interface content arrives through: `get_object(kind, key, version=None)`, `all_objects(kind)`, optional `is_initialized()`, `add_listener(kind, fn)` / `remove_listener(kind, fn)`. A store without `is_initialized()` is treated as initialized. Both shipped stores deliver only the skill kind, so `add_listener` on any other kind raises. |
| `InMemorySkillStore(objects=None)` | A dict-backed store with `put(raw)`, for local development and testing. Holds several versions of a key. |
| `FDv2SkillStore(sdk_key, *, base_uri=…, stream_uri=…, mode="stream", …)` | The delivery transport: a store fed by LaunchDarkly over the SDK-facing FDv2 channel. `start()`, `wait_for_skills(timeout)`, `is_initialized()`, `close()`, `diagnostics`, `failed`; also a context manager. `close()` is **final** — `start()` afterwards raises. `poll_interval`, `read_timeout`, `initial_backoff` and `max_backoff` must be positive and finite, and `initial_backoff` may not exceed `max_backoff`. **Server-side only.** See *Receiving skills from LaunchDarkly* above. |
| `watch_skills(skills, root, *, debounce=0.5, on_reconcile=None, …)` | `write_skills` plus a re-reconcile on every delivery change, so revocation takes effect within `debounce` rather than at the next restart. Returns `(initial report, SkillWatcher)`; close the watcher when done. `debounce` is in **seconds**, non-negative and finite. `on_reconcile` receives each *subsequent* report. One watcher per root. |
| `StoreDiagnostics` | What the transport has seen: `payloads_transferred`, `skill_objects_received`, `objects_ignored`, `objects_revoked`, `payloads_ignored`, `hashless_objects`, `connection_failures`, `last_error`. |

Configure the store with `init_client(options={"skillStore": store})`. With none configured,
the accessors raise `RuntimeError` explaining what to do, and `write_skills` reports the
failure (or raises, with `on_unavailable="raise"`). `shutdown()` clears it.

`ReconcileReport.actions` holds one `ReconcileAction` per outcome (`written`, `updated`,
`skipped_current`, `removed`, or `error`), each with `key`, `version`, the resolved `path`, and
`error`. `report.ok` is `True` when no action is an `error`, and `report.errors` lists the
`error` actions. A failure that belongs to the whole run, such as an unreadable manifest, has
the empty string as its `key`.

The fixed on-disk values are exported, so you do not have to hardcode them:
`MANIFEST_FILENAME` (`.launchdarkly-skills.json`, handy for a `.gitignore`), `SKILL_FILENAME`,
and `MANIFEST_VERSION`. So are the closed-set types, for annotating your own helpers:
`ReconcileActionKind` (`written` / `updated` / `skipped_current` / `removed` / `error`),
`OnUnavailable` (`keep` / `raise`), and `SkillOutcomeReason` (`absent` / `integrity_failure` /
`ok` / `store_unavailable` / `wrong_version`).

**`write_skills` blocks.** It is `async` for parity with the other accessors, but awaits
nothing: all file I/O runs inline and holds the event loop for the duration. Wrap it in
`asyncio.to_thread` if that matters; `timeout` is checked between steps, not mid-step. Do not
run two reconciles against the same root concurrently — they race on the manifest.

`all_objects` returns one entry per `(key, version)` under keys that are **opaque** to the SDK.
Identity comes from each object's own `key` and `version` fields, so a store can key its map
however its transport does.

> `Skill.content` is `bytes` — the verified verbatim bytes LaunchDarkly delivered, exactly
> what was hashed. The SDK never parses them; to read the frontmatter, decode and parse the
> content yourself.

#### Privilege separation: the agent must not be able to rewrite its own skills

**Run `write_skills` as a different identity than the agent.** Reconcile as one user, run the
agent as another. The reconcile sets modes explicitly rather than from your umask: skill files
and the manifest at `0644` (via `fchmod` on the descriptor, so it cannot be redirected),
per-skill `<root>/<key>/` directories at `0755`, and never the execute bit. Those modes only
protect anything if the two identities differ.

**What to verify, as the identity that will run the agent.** The SDK cannot check this for you
(see below), so make it a deployment step. The agent's identity must have no write access to:

- the managed root itself,
- the per-skill directories `<root>/<key>/` and the files `<root>/<key>/SKILL.md`,
- the manifest at `<root>/.launchdarkly-skills.json`,
- **the root's parent, and every ancestor directory above it.** Write access there lets the
  root be renamed aside and replaced with a symlink, redirecting the reconcile, and the agent's
  own skill lookups, to a directory the agent controls. In the layout `<app>/.claude/skills`
  the parent is `.claude`, which an agent identity is likely to own.

```bash
# Run as the agent's user. Every line should print DENIED.
root=.claude/skills
for target in "$root" "$root/.launchdarkly-skills.json" "$root"/*/ "$root"/*/SKILL.md; do
  [ -e "$target" ] || continue
  if [ -w "$target" ]; then echo "WRITABLE — fix this: $target"; else echo "DENIED: $target"; fi
done

# The root's ancestors, up to /. A writable one is enough to rename the root
# aside and put a symlink where it was, so these matter as much as the root.
ancestor=$(cd "$(dirname "$root")" && pwd)
while :; do
  if [ -w "$ancestor" ]; then echo "WRITABLE ANCESTOR — fix this: $ancestor"; else echo "DENIED: $ancestor"; fi
  [ "$ancestor" = / ] && break
  ancestor=$(dirname "$ancestor")
done
```

The managed root's own mode is **yours, not the SDK's**: `write_skills` creates only that leaf
directory, with your umask, because you chose the path. `chown reconcile-user:agent-group` and
`chmod 0755` on the root is what makes the rest of the tree's modes meaningful.

**Why this is the mitigation that matters.** A `SKILL.md` is agent *instructions*: an agent
that can write its skills directory can rewrite its own instructions, and an agent handling
untrusted input may be induced to. The manifest is more sensitive still, because it tells the
next reconcile which paths the SDK owns and may delete; editing it can keep a revoked skill or
aim the SDK's delete at something else. `write_skills` re-validates every manifest entry as
untrusted input, but an agent that cannot edit it at all is the stronger position.

Ancestors matter for the same reason. An identity that can rename a directory above the root
can substitute the whole tree, with no race involved, and the agent then loads skills it wrote
itself. Descriptor pinning inside `write_skills` cannot prevent that, because the substituted
tree is what the agent reads.

**The SDK does not report whether the root is writable.** It knows only its own identity, which
just wrote there. It cannot know which identity will run the agent, so any check would answer
the wrong question and look like reassurance where caution is needed.

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
