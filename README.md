# LaunchDarkly AI SDK — Python

- [Repository Layout](#repository-layout)
- [How It Works](#how-it-works)
- [Package Structure](#package-structure)
- [Quick Start](#quick-start)
  - [1. Install](#1-install)
  - [2. Configure environment](#2-configure-environment)
  - [3. Call a model](#3-call-a-model)
    - [3a. Convenience functions](#3a-convenience-functions)
    - [3b. `config()`](#3b-config)
    - [3c. `graph(key, **options)`](#3c-graphkey-options)
    - [3d. `resolve_graph(key, *, context, ...)`](#3d-resolve_graphkey--context-)
    - [3e. Framework-native graph runners](#3e-framework-native-graph-runners)
    - [3f. Built-in / native tools (`NativeTool`)](#3f-built-in--native-tools-nativetool)
- [Managing configuration](#managing-configuration)
  - [Global registry](#global-registry)
  - [Scoping configuration](#scoping-configuration)
  - [Ad-hoc options](#ad-hoc-options)
- [Telemetry](#telemetry)
- [Development](#development)
  - [Running the examples](#running-the-examples)

---

A Python monorepo for integrating LaunchDarkly AgentControl with multiple AI providers. LaunchDarkly manages which model, provider, prompt, and tools are used at runtime via feature flags — your code just calls the right handler.

## Repository Layout

```
python-ai-sdk/
├── main.py              # Entry point — brokers to an example based on CLI args
├── examples/            # Runnable examples (not part of any published package)
│   ├── agent.py         # config() with the global registry
│   ├── graph_example.py # graph() multi-agent workflow
│   ├── openai_only.py   # config() with an OpenAI-only registry
│   ├── register.py      # Global registry setup (handlers + tools)
│   ├── streaming.py     # config().stream() — token-by-token output
│   ├── tools.py         # Tool implementations (get_preferences, web_search, etc.)
│   └── utils.py         # Shared helpers (new_context, write_output)
├── packages/
│   ├── client/          # launchdarkly-ai-server       — core client (Tier 0)
│   ├── ai/              # launchdarkly-ai-python        — convenience barrel re-export
│   ├── claude-agents/   # launchdarkly-ai-claude-agents
│   ├── claude-messages/ # launchdarkly-ai-claude-messages
│   ├── openai-agents/   # launchdarkly-ai-openai-agents
│   ├── openai-messages/ # launchdarkly-ai-openai-messages
│   ├── langchain-agents/   # launchdarkly-ai-langchain-agents
│   └── langchain-messages/ # launchdarkly-ai-langchain-messages
├── .env.example         # Template — copy to .env and fill in your values
└── agents.md            # Architecture reference for AI agents and contributors
```

The `examples/` directory is a **sample implementation** showing how a consumer application wires the packages together. These files are not published and are not part of any package.

## How It Works

1. You define an AI config in LaunchDarkly (model, provider, system prompt, tools).
2. Your application fetches the variation for a user context.
3. The SDK routes to the correct provider handler, executes the call, and emits telemetry.
4. You can change providers, models, or prompts in LaunchDarkly without deploying code.

## Package Structure

This monorepo follows a three-tier architecture. Dependencies only flow downward.

```
Tier 2 — Consumer Application  (main.py, your app)
         │
Tier 1 — Handler Packages      (launchdarkly-ai-*)
         │
Tier 0 — Core Client           (launchdarkly-ai-server)
```

### Core

| Package | Description |
| --- | --- |
| [`launchdarkly-ai-server`](packages/client/README.md) | Core client — LaunchDarkly lifecycle, telemetry, shared types, `config()`, `graph()` |
| [`launchdarkly-ai-python`](packages/ai/README.md) | Convenience barrel — re-exports all of `launchdarkly-ai-server`. Install this for the simplest setup. |

### Handler Packages

| Package | Provider | Mode | Description |
| --- | --- | --- | --- |
| [`launchdarkly-ai-openai-messages`](packages/openai-messages/README.md) | OpenAI | `messages` | OpenAI Responses API with manual tool-call loop |
| [`launchdarkly-ai-openai-agents`](packages/openai-agents/README.md) | OpenAI | `agent` | OpenAI Agents SDK — fully managed agentic loop |
| [`launchdarkly-ai-claude-messages`](packages/claude-messages/README.md) | Anthropic | `messages` | Anthropic Messages API with manual tool-use loop |
| [`launchdarkly-ai-claude-agents`](packages/claude-agents/README.md) | Anthropic | `agent` | Claude Agent SDK — agentic loop with MCP tool support |
| [`launchdarkly-ai-langchain-messages`](packages/langchain-messages/README.md) | `*` (any) | `messages` | Any `BaseChatModel` via LangChain `bind_tools` loop |
| [`launchdarkly-ai-langchain-agents`](packages/langchain-agents/README.md) | `*` (any) | `agent` | LangGraph `StateGraph` — managed ReAct loop |

## Quick Start

### 1. Install

```bash
pip install launchdarkly-ai-python launchdarkly-ai-openai-messages
```

`launchdarkly-ai-python` is a thin barrel that re-exports all of `launchdarkly-ai-server`. `init_client()` auto-discovers `launchdarkly-server-sdk` at runtime — no extra setup required.

**With telemetry** (recommended for production) — traces export to the LaunchDarkly Observability dashboard:

```bash
pip install "launchdarkly-ai-python[otel]" launchdarkly-ai-openai-messages
```

No code changes are needed — `init_client()` detects whether the OTel packages are present at runtime and configures the tracer provider automatically. If they are absent, the SDK logs a one-time warning and continues normally.

### 2. Configure environment

```bash
cp .env.example .env
# Fill in LD_SDK_KEY and the API key for your provider
```

### 3. Call a model

#### 3a. Convenience functions

Each handler package exports a convenience function — the shortest path to a working call.

| Argument | Type | Required | Description |
| --- | --- | --- | --- |
| `user_input` | `str \| None` | Yes | The user's message |
| `context` | `LDContext` | Yes | User/context for flag evaluation |
| `options["key"]` | `str` | Yes | LaunchDarkly flag key |
| `options["tool_handlers"]` | `dict[str, Callable \| NativeTool]` | No | Tool name → implementation or built-in sentinel |
| `options["variables"]` | `dict[str, Any]` | No | Template variables passed to `invoke()` (e.g. `{"user_input": user_input}`) |

```python
import asyncio
from launchdarkly_ai_openai_messages import openai_messages

async def main():
    result = await openai_messages(
        "What is feature flagging?",
        {"kind": "user", "key": "user-123"},
        {"key": "my-ai-config-flag"},
        variables={"user_input": "What is feature flagging?"},
    )
    print(result.response)

asyncio.run(main())
```

| Function | Package | Underlying SDK | API |
| --- | --- | --- | --- |
| `openai_messages` | `launchdarkly-ai-openai-messages` | `openai` | OpenAI Responses API |
| `openai_agents` | `launchdarkly-ai-openai-agents` | `openai-agents` | OpenAI Agents SDK |
| `claude_messages` | `launchdarkly-ai-claude-messages` | `anthropic` | Anthropic Messages API |
| `claude_agents` | `launchdarkly-ai-claude-agents` | `claude-agent-sdk` | Claude Agent SDK (MCP) |
| `langchain_messages` | `launchdarkly-ai-langchain-messages` | `langchain-core` | LangChain `bind_tools` loop |
| `langchain_agents` | `launchdarkly-ai-langchain-agents` | `langgraph` | LangGraph `StateGraph` |

---

#### 3b. `config()`

`config()` accepts either a single handler or a list of handlers and routes to the correct one at invoke-time. The handler can be one of the pre-built `create_*_handler()` factories, a list of them, or any function you write.

Use `create_handler(provides_for, fn)` to build a custom handler. It attaches routing metadata (`provides_for`) so the handler works with `config()` routing and registries.

| Argument | Type | Required | Description |
| --- | --- | --- | --- |
| `key` | `str` | Yes | LaunchDarkly flag key |
| `handler` | `ProviderHandler \| list[ProviderHandler]` | No | Single handler or pool to route between |
| `tool_handlers` | `dict[str, Callable \| NativeTool]` | No | Tool name → implementation or built-in sentinel |
| `registry` | `Registry` | No | Registry to source handlers and tools from |

Returns a `ConfigInstance` with `.invoke(user_input, context, variables?)` and `.stream(user_input, context, variables?)`.

```python
import asyncio
from launchdarkly_ai_server import config, create_handler, shutdown
from launchdarkly_ai_openai_messages import create_openai_messages_handler
from launchdarkly_ai_openai_agents import create_openai_agent_handler
from launchdarkly_ai_claude_agents import create_claude_agents_handler
from launchdarkly_ai_claude_messages import create_claude_messages_handler

# Custom handler for an internal or self-hosted model.
async def _call_internal(cfg, user_input, tool_handlers, variables):
    import httpx
    resp = await httpx.AsyncClient().post(
        "https://models.internal.example.com/generate",
        json={"model": cfg["model"]["name"], "prompt": user_input},
    )
    data = resp.json()
    return {"output": data["text"], "usage": data.get("usage", {})}

internal_handler = create_handler(["InternalProvider", "messages"], _call_internal)

# Single handler — must match the flag variation's provider+mode, or raises.
single_caller = config(key="my-ai-config-flag", handler=internal_handler)

# Multiple handlers — routing selects the match by provider + mode.
router = config(
    key="my-ai-config-flag",
    tool_handlers={"search": lambda q: "..."},
    handler=[
        create_openai_messages_handler(),
        create_openai_agent_handler(),
        create_claude_agents_handler(),
        create_claude_messages_handler(),
    ],
)

async def main():
    result = await router.invoke(
        "What is feature flagging?",
        {"kind": "user", "key": "user-123"},
        {"user_name": "Ada"},
    )
    print(result.response)
    await shutdown()

asyncio.run(main())
```

---

#### 3c. `graph(key, **options)`

Orchestrates a multi-agent workflow defined in a LaunchDarkly agent graph flag. The SDK uses a **model-driven router**: starts at the root node, presents outgoing edges as handoff choices to the model, and follows whichever edge the model selects. The loop terminates when the model produces a final answer, a leaf is reached, a cycle is detected, or the step cap is hit.

| Argument | Type | Required | Description |
| --- | --- | --- | --- |
| `key` | `str` | Yes | LaunchDarkly agent graph flag key |
| `handlers` | `list[ProviderHandler]` | Yes | Handler pool; each node is routed by its provider + mode |
| `tool_handlers` | `dict[str, Callable \| NativeTool]` | No | Tool name → implementation or built-in sentinel |
| `graph_judge` | `str` | No | Optional judge config key evaluated against the final output |

Returns a `GraphInstance` with `.invoke(user_input, context, variables?)`.

```python
import asyncio
from launchdarkly_ai_server import graph, shutdown
from launchdarkly_ai_claude_agents import create_claude_agents_handler

async def main():
    result = await graph(
        "support-graph",
        handlers=[create_claude_agents_handler()],
    ).invoke(
        "I was double charged",
        {"kind": "user", "key": "user-123"},
        {"account_tier": "pro"},
    )
    print(result["response"])  # final output
    print(result["usage"])     # aggregate {"input": ..., "output": ..., "total": ...}
    await shutdown()

asyncio.run(main())
```

Provider packages also export single-provider conveniences (`claude_graph`, `openai_graph`, `langchain_graph`) that pre-bind their handler.

---

#### 3d. `resolve_graph(key, *, context, ...)`

For framework packages that need to walk the topology and build their own execution structure, use `resolve_graph` instead of `graph`. It returns a `GraphDefinition` dict without executing anything.

```python
from launchdarkly_ai_server import resolve_graph

def_obj = await resolve_graph(
    "support-graph",
    context={"kind": "user", "key": "user-123"},
)

if def_obj["enabled"]:
    # Walk root → leaves
    async def visitor(node, ctx):
        print(node["key"], node["config"])

    await def_obj["traverse"](visitor)

    # Or leaves → root
    await def_obj["reverse_traverse"](visitor)
```

`GraphDefinition` also exposes `get_node`, `get_child_nodes`, `get_parent_nodes`, `terminal_nodes`, `edges_from`, `run_node`, and `route` for fine-grained control.

> **Note:** `handlers` is optional in `resolve_graph`. Omit it entirely when passing the result to a framework-native runner — the runners below handle execution without going through `run_node`.

---

#### 3e. Framework-native graph runners

Each handler package ships a native runner that converts `resolve_graph` output into the provider's own multi-agent orchestration primitives. Native runners **bypass the SDK's model-driven router** and let the provider's SDK manage handoffs, tool loops, and conversation state.

All three runners share the same signature:

```python
to_xxx(
    def_promise,       # coroutine or GraphDefinition
    opts={"tool_handlers": ..., "context": ...}
).invoke(input_text, variables?)
```

##### `to_openai_agents` — OpenAI Agents SDK

Uses a post-order traversal (leaves → root) to build an `Agent` tree, wires children as handoffs, then runs the root with `Runner.run`.

```python
from launchdarkly_ai_server import resolve_graph
from launchdarkly_ai_openai_agents import to_openai_agents

ctx = {"kind": "user", "key": "user-123"}
result = await to_openai_agents(
    resolve_graph("support-graph", context=ctx),
    {"tool_handlers": registry.tools, "context": ctx},
).invoke("I was double charged")
```

##### `to_lang_graph` — LangGraph `StateGraph`

Uses a pre-order traversal (root → leaves) to build a compiled `StateGraph`. Single-child edges become direct edges; multi-child edges use `Command`-returning handoff tools so the model picks exactly one target.

```python
from launchdarkly_ai_server import resolve_graph
from launchdarkly_ai_langchain_agents import to_lang_graph

ctx = {"kind": "user", "key": "user-123"}
result = await to_lang_graph(
    resolve_graph("support-graph", context=ctx),
    {
        "tool_handlers": registry.tools,
        "context": ctx,
        # optional: supply your own model per node
        "model_factory": lambda node: ChatOpenAI(model=node["config"]["model"]["name"]),
    },
).invoke("I was double charged")
```

##### `to_claude_agents` — Claude Agent SDK

Uses a post-order traversal to wrap each child node as an MCP tool whose implementation runs `query()` in-process. The root node then runs a single top-level `query()` with its children accessible as sub-agent tools.

```python
from launchdarkly_ai_server import resolve_graph
from launchdarkly_ai_claude_agents import to_claude_agents

ctx = {"kind": "user", "key": "user-123"}
result = await to_claude_agents(
    resolve_graph("support-graph", context=ctx),
    {"tool_handlers": registry.tools, "context": ctx},
).invoke("I was double charged")
```

All three runners:

- Accept the same `tool_handlers` dict as `graph()` (including `NativeTool` sentinels)
- Emit the same `$ld:ai:*` tracking events as `graph()` — per-node duration, token counts, handoff events, and graph-level summary
- Wrap execution in a parent OTel span for hierarchical traces

---

#### 3f. Built-in / native tools (`NativeTool`)

Some provider SDKs expose built-in capabilities (e.g. `WebSearch` in the Claude Agent SDK) that are handled natively by the provider. Use a `NativeTool` sentinel in `tool_handlers` to opt in:

```python
from launchdarkly_ai_server import config
from launchdarkly_ai_claude_agents import ClaudeWebSearch, create_claude_agents_handler

result = await config(
    key="my-ai-config-flag",
    tool_handlers={
        "web-search": ClaudeWebSearch,          # NativeTool sentinel — no function needed
        "get-prefs": lambda id: {"theme": "dark"},
    },
    handler=[create_claude_agents_handler()],
).invoke(
    "What are the latest LD release notes?",
    {"kind": "user", "key": "user-123"},
)
```

The handler package recognises the sentinel, enables the provider's built-in tool, and still emits `$ld:ai:tool_call` tracking when the model invokes it.

Available Claude built-ins (from `launchdarkly-ai-claude-agents`):

| Export | Provider tool |
| --- | --- |
| `ClaudeBash` | `Bash` |
| `ClaudeRead` | `Read` |
| `ClaudeEdit` | `Edit` |
| `ClaudeWrite` | `Write` |
| `ClaudeGlob` | `Glob` |
| `ClaudeGrep` | `Grep` |
| `ClaudeWebFetch` | `WebFetch` |
| `ClaudeWebSearch` | `WebSearch` |
| `ClaudeTodoWrite` | `TodoWrite` |
| `ClaudeNotebookEdit` | `NotebookEdit` |

## Managing configuration

Handlers and tools can be registered once and reused across every call in your application. The `Registry` class is the vehicle for this — it holds a list of `ProviderHandler` instances and a dict of tool implementations.

### Global registry

> ⚠️ It is generally recommended to use scoped registries over global to satisfy the [principle of least privilege](https://en.wikipedia.org/wiki/Principle_of_least_privilege).

`global_registry` is a process-wide singleton exported from `launchdarkly_ai_server`. Populate it once at startup (typically in an initialisation module), then pass it as `registry` to any call site.

```python
from launchdarkly_ai_server import config, global_registry, graph
from launchdarkly_ai_claude_agents import create_claude_agents_handler, ClaudeWebSearch
from launchdarkly_ai_openai_messages import create_openai_messages_handler

# Called once at app startup
global_registry.register(
    handlers=[create_claude_agents_handler(), create_openai_messages_handler()],
    tools={
        "web-search": ClaudeWebSearch,
        "get-prefs": get_preferences_fn,
    },
)

# Any call site can reference it directly
result = await config(key="my-flag", registry=global_registry).invoke(user_input, context)

graph_result = await graph("support-graph", registry=global_registry).invoke(user_input, context)
```

You can call `register()` more than once to add handlers or tools incrementally. If two registrations target the same handler key (`provider:mode`) or tool name, the last one wins and a warning is logged.

### Scoping configuration

```python
from launchdarkly_ai_server import Registry
from launchdarkly_ai_openai_messages import create_openai_messages_handler
from launchdarkly_ai_openai_agents import create_openai_agent_handler

openai_registry = Registry(
    handlers=[create_openai_messages_handler(), create_openai_agent_handler()],
    tools={"get-prefs": get_preferences_fn},
)
```

Use `compose(a, b)` to merge two registries without mutating either. `b` takes precedence over `a` on any conflict.

```python
from launchdarkly_ai_server import compose

base_registry = Registry(
    handlers=[create_openai_messages_handler()],
    tools={"get-prefs": get_preferences_fn},
)
premium_registry = Registry(
    handlers=[create_claude_agents_handler()],
    tools={"web-search": ClaudeWebSearch},
)

# Neither registry is mutated; premium_registry wins on any conflict
combined = compose(base_registry, premium_registry)
```

### Ad-hoc options

```python
from launchdarkly_ai_server import config
from launchdarkly_ai_claude_messages import create_claude_messages_handler

# No registry — handlers and tools supplied inline
result = await config(
    key="my-flag",
    handler=[create_claude_messages_handler()],
    tool_handlers={"my-tool": my_tool_fn},
).invoke(user_input, context)

# Registry provides defaults; inline tool_handlers override for this call only
result2 = await config(
    key="my-flag",
    registry=global_registry,
    tool_handlers={"get-prefs": overridden_prefs_fn},
).invoke(user_input, context)
```

---

## Telemetry

Every handler wraps its provider call in an OpenTelemetry span following [Gen AI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). The core client also emits LaunchDarkly AI telemetry events (duration, token counts, generation success/failure) automatically on every `config().invoke()` call. No extra instrumentation code is required.

When running inside `graph()`, every node's events carry the graph key, tool invocations emit `$ld:ai:tool_call`, and the graph run itself emits graph-level events (`$ld:ai:graph:invocation_success`/`invocation_failure`, `duration:total`, `total_tokens`, `path`, `handoff_success`/`handoff_failure`).

### Optional dependencies

The OpenTelemetry SDK packages are **optional** — detected at runtime via `importlib`. The LaunchDarkly server SDK (`launchdarkly-server-sdk`) is also an optional dependency; pass a pre-initialized client to `init_client(client=...)` if you bring your own.

**OTel packages** (installed via `pip install "launchdarkly-ai-python[otel]"` or `pip install "launchdarkly-ai-server[otel]"`):
- **If installed:** `init_client()` sets up a `TracerProvider` with a GZIP-compressed OTLP HTTP exporter and W3C trace-context/baggage propagators — no code changes needed.
- **If not installed:** `init_client()` logs a warning and continues. Feature flags and AI calls work normally; spans become no-ops.

### Environment variables

| Variable | Description |
| --- | --- |
| `LD_SDK_KEY` | LaunchDarkly server-side SDK key |
| `LD_SERVICE_NAME` | OTel `service.name` resource attribute (default: `python-sdk`) |
| `LD_ENVIRONMENT` | `deployment.environment` resource attribute (e.g. `production`, `staging`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint override (default: LaunchDarkly Observability backend) |

See `.env.example` for a complete template.

## Development

After cloning, create a virtual environment and install the workspace in editable mode using `uv`:

```bash
# Install uv if needed
pip install uv

# Sync the workspace (installs all packages in editable mode)
uv sync
```

### Available `make` commands

A `Makefile` is provided as a consistent interface alongside the `uv` commands.

| Command | Equivalent | Description |
|---|---|---|
| `make start` | `uv run python main.py` | Run `main.py` |
| `make test` | `uv run pytest` | Run all tests |
| `make typecheck` | `uv run mypy .` | Type-check all packages |

### Running the examples

`main.py` selects an example based on the first CLI argument. The second and third arguments are the flag key and user input, both of which have sensible defaults.

```bash
uv run python main.py [example] [flag-key] [user-input]
```

| Example | Command | What it demonstrates |
| --- | --- | --- |
| `agent` *(default)* | `uv run python main.py agent` | `config()` via the global registry — switches providers without code changes |
| `graph` | `uv run python main.py graph` | `graph()` multi-agent workflow driven by a LaunchDarkly agent graph flag |
| `openai-only` | `uv run python main.py openai-only` | `config()` with a custom `Registry` restricted to OpenAI handlers |
| `streaming` | `uv run python main.py streaming` | `config().stream()` — token-by-token output |

**Examples:**

```bash
# Default: agent example with the built-in flag key and question
uv run python main.py

# Graph example with a custom flag key
uv run python main.py graph my-graph-flag "Summarise the latest release notes"

# OpenAI-only registry, custom question
uv run python main.py openai-only my-flag-key "What is feature flagging?"
```

Output from each run is written as a timestamped JSON file to the `output/` directory.

See [`agents.md`](agents.md) for the full architecture reference.
