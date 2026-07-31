# Agent Guide — `launchdarkly-ai-openai-agents`

This document tells an agent exactly how this package is implemented so it can be correctly modified, debugged, or used as a reference when building a new handler.

---

## Role and Routing

This is a **Tier 1 handler package**. It wraps the OpenAI Agents SDK (`agents` Python package) and exposes a `ProviderHandler` that routes to flag variations where:

```
provides_for = ('OpenAI', 'agent')
```

That means the LaunchDarkly flag variation must have `provider.name == "OpenAI"` and `meta.mode == "agent"`.

---

## File Map

| File | Responsibility |
|---|---|
| `src/launchdarkly_ai_openai_agents/handler.py` | All implementation — tool wiring, agent construction, run invocation, telemetry |
| `src/launchdarkly_ai_openai_agents/graph.py` | `openai_graph()` convenience wrapper around `graph()` |
| `src/launchdarkly_ai_openai_agents/native_graph.py` | `to_openai_agents()` native graph adapter |
| `src/launchdarkly_ai_openai_agents/utils.py` | Shared utility helpers (e.g. `await_coroutine_or_run`) |
| `src/launchdarkly_ai_openai_agents/__init__.py` | Package exports |

---

## Exports

```python
# Factory — returns a ProviderHandler with provides_for attached
def create_openai_agent_handler() -> ProviderHandler: ...

# Convenience wrapper — equivalent to config(key=config_key, handler=create_openai_agent_handler()).invoke(user_input, context)
def openai_agents(config_key: str, user_input: str, context: dict, **kwargs) -> ProviderResponse: ...

# Graph convenience wrapper
def openai_graph(key: str, options: dict | None = None): ...

# Native graph adapter
def to_openai_agents(def_promise: Awaitable[dict], opts: dict | None = None): ...
```

---

## Implementation Details

### 1. Prompt / Instructions

`config.instructions` is used as the system prompt when present. If absent, `config.messages` is consulted: `system`-role messages are combined into the system prompt, and `user`/`assistant` messages form conversation history prepended to `user_input`.

```python
if config.get("instructions"):
    instructions = parse_template(config["instructions"], variables)
elif config.get("messages"):
    sys_msgs = [m for m in config["messages"] if m.get("role") == "system"]
    conv_msgs = [m for m in config["messages"] if m.get("role") != "system"]
    if sys_msgs:
        instructions = parse_template("\n".join(m["content"] for m in sys_msgs), variables)
    history = "\n".join(parse_template(m["content"], variables) for m in conv_msgs)
    prompt = f"{history}\n\n{user_input}" if history else user_input
```

### 2. Tool Wiring (`_build_agent_tools`)

Each `Tool` in `config.tools` is converted to an agents SDK `FunctionTool`:

```python
FunctionTool(
    name=name,
    description=tool_cfg.get("description", ""),
    params_json_schema=tool_cfg.get("parameters") or {"type": "object", "properties": {}},
    on_invoke_tool=_execute,   # async (ctx, args_str) → str
)
```

The `on_invoke_tool` callback receives `(ToolContext, json_args_str)`. Arguments arrive as a **JSON string** and must be parsed with `json.loads`. The result is returned as `str(result)`.

If `config.tools` is absent, the tools list is empty.

### 3. Agent Construction and Run

```python
agent = Agent(
    name="assistant",
    model=config["model"]["name"],
    instructions=instructions,   # omitted if None
    tools=tools,                 # omitted if empty
)
result = await Runner.run(agent, prompt)
```

The Agents SDK manages the full agentic loop — tool calls, retries, and re-prompting — internally. The handler does not implement any loop.

### 4. Reading Results

```python
output = result.final_output or ""
usage = result.input_usage  # has input_tokens, output_tokens, total_tokens (snake_case)
```

Token counts are on `result.input_usage` using snake_case (`input_tokens`, `output_tokens`, `total_tokens`). These map directly to `parse_usage`'s expected keys.

### 5. Telemetry

Span name: `'openai.agent.run'`
Span attributes set before the call:
- `gen_ai.operation.name` = `'chat'`
- `gen_ai.system` = `'openai'`
- `gen_ai.request.model` = `config.model.name`

Prompt event: `gen_ai.content.prompt` = `instructions + '\n\nQuery:\n' + user_input`.

Span attributes set after the run:
- `gen_ai.response.model` = `config.model.name`
- `gen_ai.usage.input_tokens`
- `gen_ai.usage.output_tokens`
- `gen_ai.usage.total_tokens`

On error: `span.record_exception(exc)`, status ERROR, span ended, error re-raised.

---

## OTel Setup

This package emits one span per invocation using `opentelemetry-api`. **No OTel configuration is needed in this package** — the tracer provider is registered by `init_client()` in `launchdarkly-ai-server` (or `launchdarkly-ai`).

To receive spans, install the OTel SDK in your application:
```sh
pip install "launchdarkly-ai[otel]"
# or:
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

Span names and attributes are described in [Implementation Details → Telemetry](#5-telemetry) above.

---

## `init_client()` — When to Call It

**You do not need to call `init_client()` from this package.** Every entry point (`openai_agents()`, `config().invoke()`) lazily initializes the LaunchDarkly client on the first call, as long as `LD_SDK_KEY` is set in the environment.

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
| `agents` (openai-agents) | `Agent`, `Runner.run()`, `FunctionTool` |
| `launchdarkly-ai-server` | `AiConfigRep`, `ProviderHandler`, `parse_template`, `create_handler` |
| `opentelemetry-api` | `StatusCode`, `trace.get_tracer().start_span()` for span creation |

---

## Common Pitfalls

- **`on_invoke_tool` receives a JSON string**: unlike some other frameworks, arguments arrive as `args_str: str` and must be decoded with `json.loads(args_str)`. An empty string should be treated as `{}`.
- **`callable(handler)` guard**: the executor checks `callable(handler)` before calling it. Passing a `NativeTool` sentinel as a `tool_handlers` value will produce a clear `"No handler registered"` error rather than an opaque TypeError.
- **No manual loop**: do not add a tool-call loop. The Agents SDK's `Runner.run()` handles everything internally. Adding a manual loop on top would double-execute tools.
- **`result.final_output` may be `None`**: if the agent completes without producing a final text output (e.g. it only executed tools), `final_output` is `None`. The `or ""` guard is required.
