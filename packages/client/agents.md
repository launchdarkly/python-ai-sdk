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
| `src/launchdarkly_ai_server/types_validation.py` | `parse_ai_config` — validates flag variation shape |
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
```

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

## SDK-run evaluations

`init_evaluations()` creates an evaluations harness using `LD_API_TOKEN` and the management API host `LD_API_BASE_URI`. Do not reuse `LD_BASE_URI`: that variable configures SDK flag delivery and may point at a relay proxy. Evaluation-run links use the separate `ui_base_uri` option, then `LD_UI_BASE_URI`, then `https://app.launchdarkly.com`; do not derive their host from `LD_API_BASE_URI`. `LD_SDK_KEY` is optional for generation-only runs; when set it enables the normal handler observability path and gates generation-result ingest on the `enable-batch-ingest-in-evals-from-code` flag (only a strictly `true` variation publishes; false, default, malformed, or evaluation-error results skip publish safely). Without an SDK key the gate cannot be evaluated and ingest runs unconditionally.

`await EvaluationsModule.run(...)` takes `project_key` per call. Dataset lookup/row pagination, evaluation creation, and run creation are private helpers; only `run()` is public. Each call creates a new evaluation with `POST`, so its key must be unique. The harness directly invokes the supplied handler once per row, never retries it, and batches generation ingest when the gate permits. When ingest is enabled, it polls `GET .../runs/{id}` on lifecycle status until a terminal state and then fetches the run summary. When ingest is disabled, it skips polling and fetches the current summary once so the call returns promptly. `RunSummary` includes pending rows, and `EvalRunResult.passed` is true only when failed, error, and pending row counts are all zero.

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

## Common Pitfalls

### 1. Calling `get_client()` before `init_client()` resolves

`get_client()` raises `RuntimeError` if no client has been initialized. Handler packages that emit LD tracking events call `get_client()` — this is safe only inside a handler call because by then `config().invoke()` has already validated the flag variation, which requires an initialized client. Never call `get_client()` at module load time or in a package constructor.

### 2. Returning `dict` not a dataclass from handlers

`execute_and_track` expects the handler to return a plain `dict` with at least `output` and `usage` keys. Do not return a custom class — `parse_usage` and the telemetry pipeline both access dict keys.

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
