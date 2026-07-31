# Agent Guide — `launchdarkly-ai-claude-messages`

This document tells an agent exactly how this package is implemented so it can be correctly modified, debugged, or used as a reference when building a new handler.

---

## Role and Routing

This is a **Tier 1 handler package**. It wraps the Anthropic Messages API (`anthropic` Python SDK) and exposes a `ProviderHandler` that routes to flag variations where:

```
provides_for = ('Anthropic', 'messages')
```

That means the LaunchDarkly flag variation must have `provider.name == "Anthropic"` and `meta.mode == "messages"`.

---

## File Map

| File | Responsibility |
|---|---|
| `src/launchdarkly_ai_claude_messages/handler.py` | All implementation — message building, tool schema conversion, tool-use loop, telemetry |
| `src/launchdarkly_ai_claude_messages/__init__.py` | Package exports |

---

## Exports

```python
# Factory — returns a ProviderHandler with provides_for attached
def create_claude_messages_handler() -> ProviderHandler: ...

# Convenience wrapper — equivalent to config(key=config_key, handler=create_claude_messages_handler()).invoke(user_input, context)
def claude_messages(config_key: str, user_input: str, context: dict, **kwargs) -> ProviderResponse: ...
```

---

## Implementation Details

### 1. Message Construction (`_build_messages`)

Returns `(messages: list[dict], system: str | None)` for the Anthropic Messages API:

```
config.messages present?
  → system-role messages → system (joined with \n)
  → user/assistant messages → {"role": ..., "content": ...} with parse_template applied
  → append {"role": "user", "content": user_input}

config.instructions present? (fallback)
  → system = parse_template(config.instructions, variables)
  → messages = [{"role": "user", "content": user_input}]

neither?
  → messages = [{"role": "user", "content": user_input}], no system
```

If `config.outputFormat` is present, a JSON schema instruction is appended to the system prompt.

### 2. Tool Schema Conversion (`_build_tools`)

Each `Tool` in `config.tools` is converted to an Anthropic tool dict:

```python
{
    "name": name,
    "description": tool_config.get("description", ""),
    "input_schema": tool_config.get("parameters", {}),
}
```

The `parameters` field (JSON Schema) is passed directly as `input_schema` — no conversion needed.

### 3. Tool-Use Loop (`_run_tool_loop`)

The handler drives the loop manually against `anthropic.AsyncAnthropic().messages.create()`:

```
1. Call messages.create(model=..., max_tokens=..., system=..., messages=..., tools=...)
2. Accumulate input_tokens + output_tokens from response.usage
3. If stop_reason != 'tool_use': extract text blocks → output; break
4. Append {"role": "assistant", "content": response.content} to conversation
5. For each tool_use block:
     - call tool_handlers[block.name](block.input)
     - build {"type": "tool_result", "tool_use_id": block.id, "content": str(result)}
6. Append {"role": "user", "content": tool_results} to conversation
7. Repeat from step 1
```

`max_tokens` is read from `config.model.parameters.max_tokens`, defaulting to `1024`.

Output is all `text`-type blocks joined: `"".join(b.text for b in response.content if b.type == "text")`.

### 4. Telemetry

Span name: `'claude.messages'`
Span attributes set before the call:
- `gen_ai.operation.name` = `'chat'`
- `gen_ai.system` = `'anthropic'`
- `gen_ai.request.model` = `config.model.name`

Prompt event: `gen_ai.content.prompt` — a formatted string of `system: ...` + each message's role and content.

Span attributes set after the loop:
- `gen_ai.usage.input_tokens` — **total across all loop iterations**
- `gen_ai.usage.output_tokens` — **total across all loop iterations**
- `gen_ai.usage.total_tokens`

Completion event: `gen_ai.content.completion`.

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

**You do not need to call `init_client()` from this package.** Every entry point (`claude_messages()`, `config().invoke()`) lazily initializes the LaunchDarkly client on the first call, as long as `LD_SDK_KEY` is set in the environment.

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
| `anthropic` | `AsyncAnthropic` client, `Message`, `ToolUseBlock`, `TextBlock` |
| `launchdarkly-ai-server` | `AiConfigRep`, `ProviderHandler`, `parse_template`, `create_handler` |
| `opentelemetry-api` | `StatusCode`, `trace.get_tracer().start_span()` for span creation |

---

## Common Pitfalls

- **Conversation must alternate `user` / `assistant`**: The Anthropic Messages API requires that messages strictly alternate roles. The `_build_messages` function ensures user/assistant messages from `config.messages` are appended as-is, then the final user input is appended. If your `config.messages` has two consecutive `user` messages, the API will reject the request.
- **Tool results go in a `user` turn**: After executing `tool_use` blocks, the results are appended as `{"role": "user", "content": tool_results}`. This is correct for the Anthropic API — do not change to `"assistant"`.
- **`input_schema` not `parameters`**: The Anthropic SDK uses `input_schema` (not `parameters` or `schema`). The JSON Schema from `Tool.parameters` maps directly to `input_schema`.
- **Token accumulation**: Tokens are summed across every loop iteration. The `usage` dict returned by this handler contains the **total** for the entire multi-turn exchange, which is what the telemetry tracking expects.
- **Async handler dispatch**: Tool handlers may be async (`coroutinefunction`) or sync. The loop uses `asyncio.iscoroutinefunction` to decide whether to `await` them.
