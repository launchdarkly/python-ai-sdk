# `launchdarkly-ai-claude-messages`

Anthropic Claude handler for `launchdarkly-ai-server` using the **Anthropic Messages API** (`anthropic`). Runs a manual tool-use loop without the Claude Agent SDK layer.

**`provides_for`:** `['Anthropic', 'messages']` — matches flag variations where `provider.name` is `"Anthropic"` and `meta.mode` is `"messages"`.

## Installation

```bash
pip install launchdarkly-ai-server launchdarkly-ai-claude-messages
```

Set `ANTHROPIC_API_KEY` in your environment (the Anthropic SDK reads it automatically).

## Usage

### With `config()`

```python
import asyncio
from launchdarkly_ai_server import config, shutdown
from launchdarkly_ai_claude_messages import create_claude_messages_handler

async def main():
    result = await config(
        key="my-ai-config-flag",
        handler=create_claude_messages_handler(),
        tool_handlers={"search": lambda q: "..."},
    ).invoke("What is feature flagging?", {"kind": "user", "key": "user-123"})

    print(result.response)
    await shutdown()

asyncio.run(main())
```

### Convenience wrapper

```python
import asyncio
from launchdarkly_ai_claude_messages import claude_messages

async def main():
    user_input = "What is feature flagging?"
    result = await claude_messages(
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
- If tools are defined in the flag config, executes them as the model requests and feeds results back until the model produces a final response.
- Emits an OTel span and LaunchDarkly telemetry for every call.

## Choosing Between `claude-agents` and `claude-messages`

| | `claude-agents` | `claude-messages` |
|---|---|---|
| Underlying SDK | `claude-agent-sdk` | `anthropic` |
| Tool loop | Managed by the Claude Agent SDK | Executed and fed back manually |
| Complexity | Lower (SDK manages loop) | More explicit control |

## Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key (read automatically by the Anthropic SDK) |
| `LD_SDK_KEY` | LaunchDarkly server-side SDK key |
| `LD_SERVICE_NAME` | OTel `service.name` resource attribute (default: `python-sdk`) |
| `LD_ENVIRONMENT` | `deployment.environment` attribute attached to telemetry |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint override (default: LaunchDarkly Observability backend) |
