# Agent Guide — `launchdarkly-ai-langchain-messages`

This document tells an agent exactly how this package is implemented so it can be correctly modified, debugged, or used as a reference when building a new handler.

---

## Role and Routing

This is a **Tier 1 handler package**. It wraps LangChain chat models (`langchain_core`) and exposes a `ProviderHandler` that routes to flag variations where:

```
provides_for = ('*', 'messages')
```

The `'*'` wildcard means this handler acts as a fallback for any `meta.mode == "messages"` variation that has no more-specific (exact-provider-name) handler registered. LangChain is a framework adapter — not a provider itself — so it routes through `langchain_anthropic`, `langchain_openai`, or other `BaseChatModel` implementations at runtime. Using `'*'` lets users keep their flag variations configured with their real provider name (`"Anthropic"`, `"OpenAI"`, etc.) without needing a native handler for each.

> **Priority rule:** if the caller also registers an explicit provider handler (e.g. `('OpenAI', 'messages')`), that handler takes precedence over the wildcard for matching variations.

The handler is model-agnostic — it accepts any `BaseChatModel`. The default is `ChatOpenAI`.

---

## File Map

| File | Responsibility |
|---|---|
| `src/launchdarkly_ai_langchain_messages/handler.py` | All implementation — message building, tool binding, invoke loop, telemetry |
| `src/launchdarkly_ai_langchain_messages/__init__.py` | Package exports |

---

## Exports

```python
# Factory — accepts an optional BaseChatModel; defaults to ChatOpenAI()
def create_lang_chain_handler(llm=None) -> ProviderHandler: ...

# Convenience wrapper — equivalent to config(key=config_key, handler=create_langchain_messages_handler()).invoke(user_input, context)
def langchain_messages(config_key: str, user_input: str, context: dict, llm=None, **kwargs) -> ProviderResponse: ...
```

---

## Implementation Details

### 1. Message Construction (`_build_messages`)

Returns `list[BaseMessage]` using LangChain message types (imported lazily from `langchain_core.messages`):

```
config.messages present?
  → system-role messages → SystemMessage (joined with \n)
  → user messages → HumanMessage
  → assistant messages → AIMessage
  → append HumanMessage(user_input)

config.instructions present? (fallback)
  → [SystemMessage(parse_template(instructions, variables)), HumanMessage(user_input)]

neither?
  → [HumanMessage(user_input)]
```

`parse_template` is applied to every message's content.

### 2. Tool Schema Conversion (`_build_tools`)

Each `Tool` in `config.tools` is converted to the OpenAI function-call format that LangChain's `bind_tools` understands:

```python
{
    "type": "function",
    "function": {
        "name": name,
        "description": tool_config.get("description", ""),
        "parameters": tool_config.get("parameters", {}),  # JSON Schema passed through
    },
}
```

### 3. Model Binding and Invoke Loop

```python
active_model = base_model.bind_tools(tool_defs) if tool_defs else base_model
```

`bind_tools` is called on the concrete model instance. If no tools are configured the call is skipped entirely.

The loop:

```
1. response = await active_model.ainvoke(conversation_messages)
2. Accumulate response.usage_metadata?.input_tokens + output_tokens
3. Push response (AIMessage) onto conversation_messages
4. tool_calls = response.tool_calls or []
5. If none: output = response.content; break
6. For each tool call:
     - call tool_handlers[tc["name"]](tc["args"])   # tc["args"] is already parsed
     - build ToolMessage(tool_call_id=tc["id"] or tc["name"], content=str(result))
7. Push all ToolMessages onto conversation_messages
8. Repeat from step 1
```

### 4. Reading Tool Call Arguments

`tc["args"]` on a LangChain tool call is already a parsed dict (not a JSON string). Pass it directly to `tool_handlers[tc["name"]]`.

### 5. Telemetry

Span name: `'langchain.invoke'`
Span attributes set before the call:
- `gen_ai.operation.name` = `'chat'`
- `gen_ai.system` = `'langchain'`
- `gen_ai.request.model` = `config.model.name`

Prompt event: `gen_ai.content.prompt` — each message formatted as `"<type>: <content>"` joined with `\n`.

Span attributes set after the loop:
- `gen_ai.usage.input_tokens` — **total across all iterations**
- `gen_ai.usage.output_tokens` — **total across all iterations**
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

Span names and attributes are described in [Implementation Details → Telemetry](#4-telemetry) above.

---

## `init_client()` — When to Call It

**You do not need to call `init_client()` from this package.** Every entry point (`langchain_messages()`, `config().invoke()`) lazily initializes the LaunchDarkly client on the first call, as long as `LD_SDK_KEY` is set in the environment.

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
| `langchain-core` | `BaseChatModel`, `HumanMessage`, `SystemMessage`, `AIMessage`, `ToolMessage` |
| `langchain-openai` | `ChatOpenAI` (default model, imported lazily) |
| `launchdarkly-ai-server` | `AiConfigRep`, `ProviderHandler`, `parse_template`, `create_handler` |
| `opentelemetry-api` | `StatusCode`, `trace.get_tracer().start_span()` for span creation |

---

## Common Pitfalls

- **`usage_metadata` may be `None`**: not all `BaseChatModel` implementations return token usage. The `or 0` guards on `input_tokens` and `output_tokens` are required. If usage is consistently `0`, the underlying model doesn't report it.
- **`bind_tools` availability**: `bind_tools` exists on concrete subclasses of `BaseChatModel`. Not all models implement it. If you pass a model without `bind_tools` and tools are configured, the handler will raise `AttributeError`.
- **`tc["id"]` may be `None`**: some LangChain model wrappers do not populate the tool call `id`. The `tc.get("id") or tc["name"]` fallback ensures `ToolMessage` always has a non-empty `tool_call_id`.
- **`response.content` type**: if the model returns a complex content array (not a string), `output` will be `""`. Extend the content extraction logic if you need to handle array content.
- **Async handler dispatch**: tool handlers may be async (`coroutinefunction`) or sync. The loop checks `asyncio.iscoroutinefunction` to decide whether to `await` them.
- **Custom models**: when passing a non-OpenAI `BaseChatModel`, ensure the model's `bind_tools` accepts the OpenAI function-call format used here. Models from `langchain_anthropic`, `langchain_google_genai`, etc. use the same format via LangChain's abstraction.
