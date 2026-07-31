# Agent Guide — `launchdarkly-ai-langchain-agents`

This document tells an agent exactly how this package is implemented so it can be correctly modified, debugged, or used as a reference when building a new handler.

---

## Role and Routing

This is a **Tier 1 handler package**. It wraps LangGraph's `create_react_agent` (`langgraph.prebuilt`) and exposes a `ProviderHandler` that routes to flag variations where:

```
provides_for = ['*', 'agent']
```

The `'*'` wildcard means this handler acts as a fallback for any `meta.mode == "agent"` variation that has no more-specific (exact-provider-name) handler registered. LangChain is a framework adapter — not a provider itself — so it routes through `langchain-anthropic`, `langchain-openai`, or other `BaseChatModel` implementations at runtime by inspecting `config.provider.name`. Using `'*'` lets users keep their flag variations configured with their real provider name (`"Anthropic"`, `"OpenAI"`, etc.) without needing a native handler for each.

> **Priority rule:** if the caller also registers an explicit provider handler (e.g. `['OpenAI', 'agent']`), that handler takes precedence over the wildcard for matching variations.

The handler is model-agnostic — it accepts any `BaseChatModel`. The default is `ChatOpenAI`.

---

## File Map

| File | Responsibility |
|---|---|
| `src/launchdarkly_ai_langchain_agents/handler.py` | All implementation — message building, LangGraph tool wiring, agent invocation, telemetry |
| `src/launchdarkly_ai_langchain_agents/graph.py` | `langchain_graph()` convenience wrapper around `graph()` |
| `src/launchdarkly_ai_langchain_agents/native_graph.py` | `to_lang_graph()` native graph adapter |
| `src/launchdarkly_ai_langchain_agents/__init__.py` | Package exports |

---

## Exports

```python
# Factory — accepts an optional BaseChatModel; defaults to ChatOpenAI()
def create_langchain_agents_handler(llm: BaseChatModel | None = None) -> ProviderHandler: ...

# Convenience wrapper — equivalent to config(key=config_key, handler=create_langchain_agents_handler()).invoke(user_input, context)
def langchain_agents(config_key: str, user_input: str, context: dict, **kwargs) -> ProviderResponse: ...

# Graph convenience wrapper — equivalent to graph(key, options, handlers=[create_langchain_agents_handler()])
def langchain_graph(key: str, options: dict | None = None, llm: BaseChatModel | None = None): ...

# Native graph adapter — builds a LangGraph StateGraph from a resolved GraphDefinition
def to_lang_graph(def_promise: Awaitable[dict], opts: dict | None = None): ...
```

---

## Implementation Details

### 1. Message Construction

Returns `BaseMessage[]` using LangChain message types:

```
config.instructions present?
  → [SystemMessage(parse_template(instructions, variables)), HumanMessage(user_input)]

config.messages present?
  → system-role messages → SystemMessage (joined with \n)
  → user messages → HumanMessage
  → assistant messages → AIMessage
  → append HumanMessage(user_input)

neither?
  → [HumanMessage(user_input)]
```

`parse_template` is applied to every message's content.

### 2. Tool Wiring

Each `Tool` in `config.tools` is converted using LangGraph's `tool()` decorator from `langchain_core.tools`:

```python
@tool(name=name, description=tool_config.description, ...)
async def executor(args):
    result = await tool_handlers[name](args)
    return str(result)
```

The `schema` field accepts a JSON Schema dict. Tool executor receives parsed args — do not call `json.loads` inside the executor.

### 3. Agent Construction and Invocation

```python
agent = create_react_agent(model=base_model, tools=tools, prompt=system_prompt)
result = await agent.ainvoke({"messages": initial_messages})
```

`create_react_agent` from `langgraph.prebuilt` builds a production-ready ReAct-style agent. The handler passes the initial messages and lets LangGraph manage all tool calls, retries, and re-prompting internally.

### 4. Extracting Output

```python
last_message = result["messages"][-1]
output = last_message.content if isinstance(last_message.content, str) else ""
```

LangGraph returns the full message history in `result["messages"]`. The final response is always the last message.

### 5. Token Accumulation

Token usage is summed from `usage_metadata` on every message in `result["messages"]`:

```python
for msg in result["messages"]:
    meta = getattr(msg, "usage_metadata", None)
    if meta:
        total_input += meta.get("input_tokens", 0)
        total_output += meta.get("output_tokens", 0)
```

### 6. Telemetry

Span name: `'langchain.agent'`  
Span attributes set before the call:
- `gen_ai.operation.name` = `'chat'`
- `gen_ai.system` = `'langchain'`
- `gen_ai.request.model` = `config.model.name`

Span attributes set after the agent run:
- `gen_ai.usage.input_tokens` — summed from all messages
- `gen_ai.usage.output_tokens` — summed from all messages
- `gen_ai.usage.total_tokens`

On error: `span.record_exception(err)`, status ERROR, span ended, error re-thrown.

### 7. `to_lang_graph()` — native graph adapter

Converts a `resolve_graph()` result into a compiled LangGraph `StateGraph`. Pre-order traversal (root → leaves) registers each node. Single-child edges become direct edges after a tool loop; multi-child edges use `Command`-returning handoff tools (bound with `parallel_tool_calls=False`) so the model picks exactly one target.

Key implementation detail: `WorkflowState` is a `TypedDict` with an `Annotated[list[Any], add_messages]` field. Because `from __future__ import annotations` defers annotation evaluation, LangGraph's `StateGraph(WorkflowState)` calls `get_type_hints(WorkflowState)` which resolves `add_messages` in the **module's global namespace**. Therefore `add_messages` must be imported at module level — see Common Pitfalls below.

---

## OTel Setup

This package emits one span per invocation using `opentelemetry-api`. **No OTel configuration is needed in this package** — the tracer provider is registered by `init_client()` in `launchdarkly-ai-server` (or `launchdarkly-ai`).

To receive spans, install the OTel SDK in your application:
```sh
pip install "launchdarkly-ai[otel]"
# or:
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

Span names and attributes are described in [Implementation Details → Telemetry](#6-telemetry) above.

---

## `init_client()` — When to Call It

**You do not need to call `init_client()` from this package.** Every entry point (`langchain_agents()`, `config().invoke()`) lazily initializes the LaunchDarkly client on the first call, as long as `LD_SDK_KEY` is set in the environment.

**Call `init_client()` explicitly in your application startup code when you need to:**

- **Pass custom options** — `serviceName`, `environment`, or OTel configuration:
  ```python
  from launchdarkly_ai import init_client  # or launchdarkly_ai_server
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
| `langchain-core` | `BaseChatModel`, `HumanMessage`, `SystemMessage`, `AIMessage`, `BaseMessage`, `tool()` |
| `langchain-openai` | `ChatOpenAI` (default model) |
| `langgraph` | `StateGraph`, `ToolNode`, `tools_condition`, `create_react_agent` |
| `launchdarkly-ai-server` | `AiConfigRep`, `Tool`, `ProviderHandler`, `parse_template` |
| `opentelemetry-api` | `StatusCode`, `trace.get_tracer().start_span()` for span creation |

---

## Common Pitfalls

- **No manual tool loop**: `create_react_agent` manages the entire reasoning loop internally. Do not add a manual tool-call loop on top — it would be redundant and would interfere with LangGraph's state graph.
- **`result["messages"]` is the full history**: the agent appends every intermediate AI message, tool call, and tool result. Always take the **last** message as the final output.
- **`usage_metadata` may be sparse**: not every message carries usage. The accumulation loop silently skips messages where `usage_metadata` is absent. If total tokens are `0`, the underlying model doesn't report per-message usage.
- **`add_messages` and all annotation-reducer symbols must be module-level imports.** `from __future__ import annotations` (present at the top of `native_graph.py`) causes Python to store all annotations as strings. When LangGraph calls `get_type_hints(WorkflowState)`, Python evaluates those strings in the **module's global namespace** — not in the local scope of the `invoke()` function. Any symbol that is only a local variable (e.g. imported inside `invoke()`) will raise `NameError: name 'add_messages' is not defined` at `StateGraph(WorkflowState)` time. See Appendix A.3 in `TESTING.md`.
- **`tool()` executor receives parsed args**: LangGraph parses the model's tool call arguments before calling the executor. Do not call `json.loads` inside the tool executor.
- **`last_message.content` may not be a string**: if the final message is a tool call or has a complex content list, the `isinstance(..., str)` check fails and `output` will be `''`. This should not occur in a well-behaved ReAct agent but is guarded defensively.
