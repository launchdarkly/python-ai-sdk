# Agent Guide — `launchdarkly-ai-claude-agents`

> **Span shape.** This package emits `invoke_agent` → `chat {model}` → `execute_tool {name}`, with
> tool spans as siblings of `chat`. Span construction lives in `spans.py` beside the handler.
> Conversation content is off unless the caller passes `capture_content=True`.
> `TELEMETRY-CONTRACT.md` at the repo root is the authority; read it before changing span code.


This document tells an agent exactly how this package is implemented so it can be correctly modified, debugged, or used as a reference when building a new handler.

---

## Role and Routing

This is a **Tier 1 handler package**. It wraps the `claude-agent-sdk` Python package and exposes a `ProviderHandler` that routes to flag variations where:

```
provides_for = ('Anthropic', 'agent')
```

That means the LaunchDarkly flag variation must have `provider.name == "Anthropic"` and `meta.mode == "agent"`.

---

## File Map

| File | Responsibility |
|---|---|
| `src/launchdarkly_ai_claude_agents/handler.py` | All implementation — prompt building, MCP tool wiring, agentic loop, telemetry |
| `src/launchdarkly_ai_claude_agents/graph.py` | `claude_graph()` convenience wrapper around `graph()` |
| `src/launchdarkly_ai_claude_agents/native_graph.py` | `to_claude_agents()` native graph adapter |
| `src/launchdarkly_ai_claude_agents/builtins.py` | Pre-constructed `NativeTool` sentinels for Claude built-in tools |
| `src/launchdarkly_ai_claude_agents/__init__.py` | Package exports |

---

## Exports

```python
# Factory — returns a ProviderHandler with provides_for attached
def create_claude_agents_handler() -> ProviderHandler: ...

# Convenience wrapper — equivalent to config(key=config_key, handler=create_claude_agents_handler()).invoke(user_input, context)
def claude_agents(config_key: str, user_input: str, context: dict, **kwargs) -> ProviderResponse: ...

# Graph convenience wrapper — equivalent to graph(key, options, handlers=[create_claude_agents_handler()])
def claude_graph(key: str, options: dict | None = None): ...

# Native graph adapter — builds a Claude code-agents graph from a resolved GraphDefinition
def to_claude_agents(def_promise: Awaitable[dict], opts: dict | None = None): ...

# NativeTool sentinels for Claude Code built-in capabilities
ClaudeBash: NativeTool
ClaudeRead: NativeTool
ClaudeEdit: NativeTool
ClaudeWrite: NativeTool
ClaudeGlob: NativeTool
ClaudeGrep: NativeTool
ClaudeWebFetch: NativeTool
ClaudeWebSearch: NativeTool
ClaudeTodoWrite: NativeTool
ClaudeNotebookEdit: NativeTool
```

---

## Implementation Details

### 1. Prompt Construction (`build_prompt`)

The handler uses a flat `prompt` + optional `system_prompt` shape that the `claude_agent_sdk.query()` call accepts:

```
config.instructions present?
  → system_prompt = parse_template(config.instructions, variables)
  → prompt = user_input

config.messages present?
  → system-role messages → system_prompt (joined with \n)
  → non-system messages → joined as plain text → prepended to user_input
  → prompt = conversation_history + "\n\n" + user_input

neither?
  → prompt = user_input, no system_prompt
```

Note: the `messages` path collapses conversation history into a single flat string — roles are not individually structured. This is a limitation of the agent SDK's `query()` interface.

### 2. Tool Wiring (`build_tool_mcp`)

Tools are delivered via an in-process MCP server, not as raw JSON schema defs. The pipeline:

1. Each `Tool` in `config.tools` is converted to a `claude_agent_sdk.tool()` call.
2. The tool's executor calls `tool_handlers[tool_name](args)` and returns `{ "content": [{ "type": "text", "text": str(result) }] }`.
3. All tools are registered in a single `create_sdk_mcp_server(name="tool-mcp", ...)`.
4. The MCP server is passed to `query()` via `ClaudeAgentOptions(mcp_servers={"tool-mcp": mcp_server})`.
5. Allowed tools are prefixed: `mcp__tool-mcp__<tool_name>`.

If `config.tools` is absent, the MCP server is not created and `allowed_tools` is an empty list.

### 3. Agentic Loop

The `claude_agent_sdk` handles the loop internally. The handler iterates the `query()` async generator, holding an explicit reference so it can be explicitly closed on early exit:

```python
gen = query_fn(prompt=prompt, options=options)
try:
    async for message in gen:
        if isinstance(message, ResultMessage):
            # done — extract output and usage
            break
finally:
    await gen.aclose()
```

**Important:** a bare `return` inside `async for` abandons the generator. Python's asyncio finalizer will try to `aclose()` it later and may raise `RuntimeError: aclose(): asynchronous generator is already running` if the generator is still suspended inside a real SDK `await`. Always use `break` + explicit `aclose()`. See **Appendix A.4** in `TESTING.md`.

### 4. Telemetry

Span name: `'claude.query'`
Span attributes set before the call:
- `gen_ai.operation.name` = `'chat'`
- `gen_ai.system` = `'anthropic'`
- `gen_ai.request.model` = `config.model.name`

Span event before the call:
- `gen_ai.content.prompt` with attribute `gen_ai.prompt` = the raw prompt string

Span attributes set after the call (inside the result branch):
- `gen_ai.usage.input_tokens`
- `gen_ai.usage.output_tokens`
- `gen_ai.usage.total_tokens`

Span event after the call:
- `gen_ai.content.completion` with attribute `gen_ai.completion` = the result string

On error: `span.record_exception(exc)`, status set to ERROR, span ended, error re-raised.

---

## OTel Setup

This package emits one span per invocation using `opentelemetry-api`. **No OTel configuration is needed in this package** — the tracer provider is registered by `init_client()` in `launchdarkly-ai-server` (or `launchdarkly-ai`).

To receive spans, install the OTel SDK in your application:
```sh
pip install "launchdarkly-ai[otel]"
# or:
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

Span names and attributes are described in [Implementation Details → Telemetry](#4-telemetry) above.

---

## `init_client()` — When to Call It

**You do not need to call `init_client()` from this package.** Every entry point (`claude_agents()`, `config().invoke()`) lazily initializes the LaunchDarkly client on the first call, as long as `LD_SDK_KEY` is set in the environment.

**Call `init_client()` explicitly in your application startup code when you need to:**

- **Pass custom options** — `serviceName`, `environment`, or OTel configuration:
  ```python
  from launchdarkly_ai_python import init_client  # or launchdarkly_ai_server
  await init_client({"serviceName": "my-service", "environment": "production"})
  ```
- **Use a custom or edge runtime (BYOC path)** — pass a pre-initialized client that satisfies `LDClientInterface`:
  ```python
  from launchdarkly_ai_server import init_client
  ld_client = create_your_custom_client(os.environ["LD_SDK_KEY"])
  await init_client(ld_client)
  ```
- **Pre-warm the connection** — call `init_client()` at startup to avoid cold-start latency on the first user request.

`init_client()` is idempotent — calling it twice is a no-op. Never call `init_client()` inside this handler package; initialization belongs in application startup code. Full details in the [`launchdarkly-ai-server` agents.md](../client/agents.md#lifecycle-invariants).

---

## Dependencies

| Package | Why |
|---|---|
| `claude-agent-sdk` | `query()`, `tool()`, `create_sdk_mcp_server()`, `ResultMessage` |
| `launchdarkly-ai-server` | `AiConfigRep`, `ProviderHandler`, `parse_template`, `get_client`, `make_track_data` |
| `opentelemetry-api` | `StatusCode`, `trace.get_tracer().start_span()` for span creation |

---

## Common Pitfalls

- **MCP tool name prefix**: tools registered in the MCP server are accessible as `mcp__tool-mcp__<name>`. The `allowed_tools` list must use this prefix; omitting it will cause the agent to not invoke any tools.
- **Async generator teardown**: always use `break` inside `async for` (not `return`) and wrap the loop in `try/finally: await gen.aclose()`. See TESTING.md Appendix A.4.
- **`message.usage` shape**: the raw usage dict from the `claude_agent_sdk` is passed through directly to `parse_usage`. It accepts `input_tokens`/`output_tokens` which the SDK provides.
- **Native tools** (`ClaudeWebSearch`, etc.) are registered in `tool_handlers` as `NativeTool` sentinel instances (not callables). The handler's `partition_tools` function separates them from user-defined tools — native tools go into `native_tool_names` for the agent's built-in access, user tools go through the MCP server.
