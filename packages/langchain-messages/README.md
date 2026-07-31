# `launchdarkly-ai-langchain-messages`

LangChain handler for `launchdarkly-ai-server` using **LangChain chat models** (`langchain-core`). Works with any `BaseChatModel` — defaults to `ChatOpenAI`. Runs a manual tool-call loop using LangChain's `bind_tools` API.

**`provides_for`:** `['*', 'messages']` — matches any flag variation where `meta.mode` is `"messages"` and no more-specific handler is registered. LangChain is a framework adapter, not a provider: it routes through `langchain-anthropic`, `langchain-openai`, and others at runtime based on `config.provider.name`. Use `'*'` so that flags configured with `provider.name = "Anthropic"` or `"OpenAI"` are automatically handled without requiring a separate native handler.

## Installation

```bash
pip install launchdarkly-ai-server launchdarkly-ai-langchain-messages
```

The default model is `ChatOpenAI`, so set `OPENAI_API_KEY` unless you pass a custom `BaseChatModel`.

## Usage

### With the default model (`ChatOpenAI`)

```python
import asyncio
from launchdarkly_ai_server import config, shutdown
from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

async def main():
    result = await config(
        key="my-ai-config-flag",
        handler=create_langchain_messages_handler(),
    ).invoke("What is feature flagging?", {"kind": "user", "key": "user-123"})

    print(result.response)
    await shutdown()

asyncio.run(main())
```

### With a custom `BaseChatModel`

```python
from langchain_anthropic import ChatAnthropic
from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

handler = create_langchain_messages_handler(ChatAnthropic(model="claude-opus-4-5"))
```

### Convenience wrapper

```python
import asyncio
from launchdarkly_ai_langchain_messages import langchain_messages

async def main():
    user_input = "What is feature flagging?"
    result = await langchain_messages(
        user_input,
        {"kind": "user", "key": "user-123"},
        {"key": "my-ai-config-flag"},
        variables={"user_input": user_input},
    )
    print(result.response)

asyncio.run(main())
```

## How It Works

- Uses the system prompt and conversation history defined in your LaunchDarkly flag config.
- Template placeholders (`{{variable}}`) in the prompt are substituted using `variables` before the call.
- If tools are defined in the flag config, binds them to the model and executes them as requested, feeding results back until the model produces a final response.
- Emits an OTel span and LaunchDarkly telemetry for every call.

## Choosing Between `langchain-agents` and `langchain-messages`

| | `langchain-agents` | `langchain-messages` |
|---|---|---|
| Orchestration | LangGraph `StateGraph` | Manual tool loop |
| Reasoning style | ReAct (reason + act cycles) | Single invoke per tool round-trip |
| Best for | Complex multi-step reasoning | Straightforward tool calls |

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Required when using the default `ChatOpenAI` model |
| `LD_SDK_KEY` | LaunchDarkly server-side SDK key |
| `LD_SERVICE_NAME` | OTel `service.name` resource attribute (default: `python-sdk`) |
| `LD_ENVIRONMENT` | `deployment.environment` attribute attached to telemetry |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint override (default: LaunchDarkly Observability backend) |
