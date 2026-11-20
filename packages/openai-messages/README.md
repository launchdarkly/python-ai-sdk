# `launchdarkly-ai-openai-messages`

OpenAI handler for `launchdarkly-ai-server` using the **OpenAI Responses API** (`openai`). Runs a manual function-call loop directly against the Responses API.

**`provides_for`:** `['OpenAI', 'messages']` — matches flag variations where `provider.name` is `"OpenAI"` and `meta.mode` is `"messages"`.

## Installation

```bash
pip install "launchdarkly-ai-server[ld]" launchdarkly-ai-openai-messages
```

Set `OPENAI_API_KEY` in your environment (the OpenAI SDK reads it automatically).

## Usage

### With `config()`

```python
import asyncio
from launchdarkly_ai_server import config, shutdown
from launchdarkly_ai_openai_messages import create_openai_messages_handler

async def main():
    result = await config(
        key="my-ai-config-flag",
        handler=create_openai_messages_handler(),
        tool_handlers={"search": lambda q: "..."},
    ).invoke("What is feature flagging?", {"kind": "user", "key": "user-123"})

    print(result.response)
    await shutdown()

asyncio.run(main())
```

### Convenience wrapper

```python
import asyncio
from launchdarkly_ai_openai_messages import openai_messages

async def main():
    user_input = "What is feature flagging?"
    result = await openai_messages(
        "my-ai-config-flag",
        user_input,
        {"kind": "user", "key": "user-123"},
    )
    print(result.response)

asyncio.run(main())
```

## How It Works

- Uses the system prompt defined in your LaunchDarkly flag config.
- Template placeholders (`{{variable}}`) in the prompt are substituted using `variables` before the call.
- If tools are defined in the flag config, executes them as the model requests and feeds results back until the model produces a final response.
- Only the tools attached to the variation are offered to the model, and only those with a function in `tool_handlers`; a `NativeTool` is never offered. If the model names a tool it was not offered, the call raises `ValueError: No handler registered for tool "<name>"`. A tool implementation receives the model's arguments as one `dict`.
- Emits an OTel span and LaunchDarkly telemetry for every call.

## Retries

`create_openai_messages_handler(max_retries=...)` sets how many times the `AsyncOpenAI` client retries a failed request (a rate limit, a timeout, a 5xx) before the error reaches the handler. `None`, the default, keeps the client's own default of 2. Raise it when many calls share one rate-limit window, as an evaluation run with many concurrent workers does. A negative or non-integer value raises when the handler is built.

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (read automatically by the OpenAI SDK) |
| `LD_SDK_KEY` | LaunchDarkly server-side SDK key |
| `LD_SERVICE_NAME` | OTel `service.name` resource attribute (default: `python-sdk`) |
| `LD_ENVIRONMENT` | `deployment.environment` attribute attached to telemetry |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint override (default: LaunchDarkly Observability backend) |
