# Agent Guide — `launchdarkly-ai-openai-messages`

This document tells an agent exactly how this package is implemented so it can be correctly modified, debugged, or used as a reference when building a new handler.

---

## Role and Routing

This is a **Tier 1 handler package**. It wraps the OpenAI Responses API (`openai` Python SDK) and exposes a `ProviderHandler` that routes to flag variations where:

```
provides_for = ('OpenAI', 'messages')
```

That means the LaunchDarkly flag variation must have `provider.name == "OpenAI"` and `meta.mode == "messages"`.

---

## File Map

| File | Responsibility |
|---|---|
| `src/launchdarkly_ai_openai_messages/handler.py` | All implementation — message building, tool schema conversion, function-call loop, telemetry |
| `src/launchdarkly_ai_openai_messages/__init__.py` | Package exports |

---

## Exports

```python
# Factory — returns a ProviderHandler with provides_for attached
def create_openai_messages_handler() -> ProviderHandler: ...

# Convenience wrapper — equivalent to config(key=config_key, handler=create_openai_messages_handler()).invoke(user_input, context)
def openai_messages(config_key: str, user_input: str, context: dict, **kwargs) -> ProviderResponse: ...
```

---

## Implementation Details

### 1. Message Construction (`_build_input_messages`)

The handler uses the OpenAI Responses API's list-of-dicts input format:

```
config.messages present and non-empty?
  → messages = [{"role": m["role"], "content": parse_template(m["content"], variables)} for m in config.messages]
  → if user_input: append {"role": "user", "content": user_input}

config.instructions present? (fallback)
  → instructions = parse_template(config.instructions, variables)
  → result = [{"role": "system", "content": instructions}]  # only if non-empty
  → result.append({"role": "user", "content": user_input})
```

`config.messages` takes priority when present and non-empty. `config.instructions` is the fallback used only when `messages` is absent or empty.

### 2. Tool Schema Conversion (`_build_tools`)

Each `Tool` in `config.tools` is converted to a `FunctionTool` dict:

```python
{
    "type": "function",
    "name": name,
    "description": tool_config.get("description", ""),
    "parameters": tool_config.get("parameters", {}),   # JSON Schema passed through as-is
    "strict": False,
}
```

### 3. Function-Call Loop

The handler drives the tool loop manually via `openai.AsyncOpenAI().responses.create()` using the `previous_response_id` chaining mechanism:

```
1. responses.create(model=..., input=input_messages, tools=tools)
2. Accumulate usage.input_tokens + usage.output_tokens
3. Filter response.output for items where item.type == "function_call"
4. If none: output = response.output_text; break
5. For each function_call:
     - args = json.loads(tc.arguments)
     - result = await tool_handlers[tc.name](args)
     - build {"type": "function_call_output", "call_id": tc.call_id, "output": str(result)}
6. responses.create(model=..., previous_response_id=response.id, input=tool_outputs)
7. Accumulate tokens from new response
8. Repeat from step 3
```

Tool argument deserialization: `json.loads(tc.arguments)` — the Responses API delivers arguments as a JSON string.

### 4. Telemetry

Span name: `'openai.response'`
Span attributes set before the call:
- `gen_ai.operation.name` = `'chat'`
- `gen_ai.system` = `'openai'`
- `gen_ai.request.model` = `config.model.name`

Prompt event: `gen_ai.content.prompt` = `instructions + '\n\nQuery:\n' + user_input`.

Additional attribute set after the first response:
- `gen_ai.response.model` = `response.model`

Span attributes set after the loop:
- `gen_ai.usage.input_tokens` — **total across all iterations**
- `gen_ai.usage.output_tokens` — **total across all iterations**
- `gen_ai.usage.total_tokens`

On error: `span.record_exception(exc)`, status ERROR, span ended, error re-raised.

---

## OTel Setup

This package emits one span per invocation using `opentelemetry-api`. **No OTel configuration is needed in this package** — the tracer provider is registered by `init_client()` in `launchdarkly-ai-server` (or `launchdarkly-ai-python`).

To receive spans, install the OTel SDK in your application:
```sh
pip install "launchdarkly-ai-python[otel]"
# or:
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

Span names and attributes are described in [Implementation Details → Telemetry](#4-telemetry) above.

---

## `init_client()` — When to Call It

**You do not need to call `init_client()` from this package.** Every entry point (`openai_messages()`, `config().invoke()`) lazily initializes the LaunchDarkly client on the first call, as long as `LD_SDK_KEY` is set in the environment.

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
| `openai` | `AsyncOpenAI` client, `Response` type, `ResponseFunctionToolCall` |
| `launchdarkly-ai-server` | `AiConfigRep`, `ProviderHandler`, `parse_template`, `create_handler` |
| `opentelemetry-api` | `StatusCode`, `trace.get_tracer().start_span()` for span creation |

---

## Common Pitfalls

- **`previous_response_id` chaining**: subsequent tool-result calls must pass `previous_response_id=response.id` and **not** re-send the original `input_messages`. The Responses API uses stateful response chaining.
- **`tc.arguments` is a JSON string**: it must be decoded with `json.loads` before passing to `tool_handlers`. Do not pass the raw string.
- **`config.messages` takes priority over `config.instructions`**: when `config.messages` is present and non-empty, it is used as the full input (with `user_input` appended as the final turn). `config.instructions` is only used as a fallback when `messages` is absent or empty.
- **`output_text` may be `None`**: the final text response is `response.output_text` (a convenience accessor). If the model produces no text (e.g. all output items are tool calls), `output_text` is `None` — the `or ""` guard is required.
- **Async handler dispatch**: tool handlers may be async or sync. The loop uses `asyncio.iscoroutinefunction` to decide whether to `await` them.
