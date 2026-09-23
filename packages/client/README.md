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


def mentions_policy(row: DatasetRow, output: str | None) -> float:
    return 1.0 if "refund policy" in (output or "").lower() else 0.0


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

`Scorer.fn` receives the `DatasetRow` the output was generated from plus the generated output, may be sync or async, and must return a **finite number from 0 to 1** — that is the whole contract, and nothing is coerced on your behalf. A scorer answering a yes/no question returns `1.0` or `0.0` itself; a `bool`, a `NaN`, or an out-of-range number is an `invalid_score` result. Returning one score type keeps a threshold comparison meaning the same thing for binary and graded scorers, and keeps a scorer that accidentally returns a non-score from passing silently — every non-empty value is truthy, so `return "high"` would otherwise score 1.0. `Judge.threshold` defaults to 0.5 and `Scorer.threshold` to 1.0 — a perfect score, which is what a yes/no scorer wants — and both accept an optional `pass_rate_threshold`. Judge keys and scorer names share one `criterionType` namespace and must be unique within a run, case-insensitively, because that name is part of each result's deterministic event identity. `Judge.ground_truth_context` overrides what the judge is graded against when the dataset row's expected output is not it.

**The SDK reports scores and never rules on them.** LaunchDarkly derives each row's verdict at ingest by comparing the score against the criterion's stored threshold and success direction, so pass/fail policy is one server-side implementation that applies to every SDK version and to runs already recorded. A judge's direction lives on its AI Config and is injected server-side, keeping the one input a verdict turns on server-attested; a `Scorer` has no LaunchDarkly-side config to read, so it declares its own `success_direction` (default `"higher_is_better"` — set `"lower_is_better"` for a scorer that counts something unwanted, like a regex hit count).

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
