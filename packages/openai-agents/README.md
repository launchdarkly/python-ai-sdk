# `launchdarkly-ai-openai-agents`

OpenAI handler for `launchdarkly-ai-server` using the **OpenAI Agents SDK** (`openai-agents`). Delegates the full agentic loop — tool calls, retries, and orchestration — to the Agents SDK.

**`provides_for`:** `['OpenAI', 'agent']` — matches flag variations where `provider.name` is `"OpenAI"` and `meta.mode` is `"agent"`.

## Installation

```bash
pip install launchdarkly-ai-server launchdarkly-ai-openai-agents
```

Set `OPENAI_API_KEY` in your environment (the OpenAI SDK reads it automatically).

## Usage

### With `config()`

```python
import asyncio
from launchdarkly_ai_server import config, shutdown
from launchdarkly_ai_openai_agents import create_openai_agent_handler

async def main():
    result = await config(
        key="my-ai-config-flag",
        handler=create_openai_agent_handler(),
        tool_handlers={"search": lambda q: "..."},
    ).invoke("Summarize today's changelog", {"kind": "user", "key": "user-123"})

    print(result.response)
    await shutdown()

asyncio.run(main())
```

### Convenience wrapper

```python
import asyncio
from launchdarkly_ai_openai_agents import openai_agents

async def main():
    user_input = "Summarize today's changelog"
    result = await openai_agents(
        user_input,
        {"kind": "user", "key": "user-123"},
        {"key": "my-ai-config-flag"},
        variables={"user_input": user_input},
    )
    print(result.response)

asyncio.run(main())
```

### Agent graphs — `openai_graph()`

Runs a LaunchDarkly agent graph with the OpenAI agent handler pre-bound. Equivalent to calling the base `graph()` with `handlers=[create_openai_agent_handler()]`. See the [core client docs](../client/README.md#graphkey-options) for the full `graph()` API.

```python
import asyncio
from launchdarkly_ai_openai_agents import openai_graph

async def main():
    result = await openai_graph("support-graph").invoke(
        "I was double charged",
        {"kind": "user", "key": "user-123"},
    )
    print(result["response"])

asyncio.run(main())
```

### Native graph adapter — `to_openai_agents()`

Converts a `resolve_graph()` result into a framework-native OpenAI Agents swarm (post-order traversal: leaves → root). Children are wired as handoffs; the root is run with `Runner.run`.

```python
import asyncio
from launchdarkly_ai_server import resolve_graph
from launchdarkly_ai_openai_agents import to_openai_agents

async def main():
    ctx = {"kind": "user", "key": "user-123"}
    result = await to_openai_agents(
        resolve_graph("support-graph", context=ctx),
        {"tool_handlers": registry.tools, "context": ctx},
    ).call("I was double charged")
    print(result["response"])

asyncio.run(main())
```

## How It Works

- Uses the system prompt and tools defined in your LaunchDarkly flag config.
- Template placeholders (`{{variable}}`) in the prompt are substituted using `variables` before the call.
- The OpenAI Agents SDK manages the full agentic loop — tool dispatch, re-prompting, and termination — automatically.
- Emits an OTel span and LaunchDarkly telemetry for every call.

## Choosing Between `openai-agents` and `openai-messages`

| | `openai-agents` | `openai-messages` |
|---|---|---|
| Underlying SDK | `openai-agents` | `openai` (Responses API) |
| Tool loop | Managed by Agents SDK | Executed and fed back manually |
| Complexity | Lower (SDK manages loop) | More explicit control |

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (read automatically by the OpenAI SDK) |
| `LD_SDK_KEY` | LaunchDarkly server-side SDK key |
| `LD_SERVICE_NAME` | OTel `service.name` resource attribute (default: `python-sdk`) |
| `LD_ENVIRONMENT` | `deployment.environment` attribute attached to telemetry |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint override (default: LaunchDarkly Observability backend) |
