# `launchdarkly-ai-langchain-agents`

LangChain handler for `launchdarkly-ai-server` using **LangGraph's `StateGraph`** (`langgraph`). Delegates the full agentic loop to the LangGraph ReAct agent. Works with any `BaseChatModel` — defaults to `ChatOpenAI`.

**`provides_for`:** `['*', 'agent']` — matches any flag variation where `meta.mode` is `"agent"` and no more-specific handler is registered. LangChain is a framework adapter, not a provider: it routes through `langchain-anthropic`, `langchain-openai`, and others at runtime based on `config.provider.name`. Use `'*'` so that flags configured with `provider.name = "Anthropic"` or `"OpenAI"` are automatically handled without requiring a separate native handler.

## Installation

```bash
pip install launchdarkly-ai-server launchdarkly-ai-langchain-agents
```

The default model is `ChatOpenAI`, so set `OPENAI_API_KEY` unless you pass a custom `BaseChatModel`.

## Usage

### With the default model (`ChatOpenAI`)

```python
import asyncio
from launchdarkly_ai_server import config, shutdown
from launchdarkly_ai_langchain_agents import create_langchain_agents_handler

async def main():
    result = await config(
        key="my-ai-config-flag",
        handler=create_langchain_agents_handler(),
        tool_handlers={"search": lambda q: "..."},
    ).invoke(
        "Research and summarize feature flagging best practices",
        {"kind": "user", "key": "user-123"},
    )

    print(result.response)
    await shutdown()

asyncio.run(main())
```

### With a custom `BaseChatModel`

```python
from langchain_anthropic import ChatAnthropic
from launchdarkly_ai_langchain_agents import create_langchain_agents_handler

handler = create_langchain_agents_handler(ChatAnthropic(model="claude-opus-4-5"))
```

### Convenience wrapper

```python
import asyncio
from launchdarkly_ai_langchain_agents import langchain_agents

async def main():
    user_input = "Research feature flagging best practices"
    result = await langchain_agents(
        user_input,
        {"kind": "user", "key": "user-123"},
        {"key": "my-ai-config-flag"},
        variables={"user_input": user_input},
    )
    print(result.response)

asyncio.run(main())
```

### Agent graphs — `langchain_graph()`

Runs a LaunchDarkly agent graph with the LangChain agent handler pre-bound. Equivalent to calling the base `graph()` with `handlers=[create_langchain_agents_handler()]`. See the [core client docs](../client/README.md#graphkey-options) for the full `graph()` API.

```python
import asyncio
from launchdarkly_ai_langchain_agents import langchain_graph

async def main():
    result = await langchain_graph("support-graph").invoke(
        "I was double charged",
        {"kind": "user", "key": "user-123"},
    )
    print(result["response"])

asyncio.run(main())
```

### Native graph adapter — `to_lang_graph()`

Converts a `resolve_graph()` result into a framework-native LangGraph `StateGraph`. Pre-order traversal (root → leaves) builds a compiled `StateGraph`. Single-child edges become direct edges after a tool loop; multi-child edges use `Command`-returning handoff tools (bound with `parallel_tool_calls=False`) so the model picks exactly one target.

```python
import asyncio
from launchdarkly_ai_server import resolve_graph
from launchdarkly_ai_langchain_agents import to_lang_graph

async def main():
    ctx = {"kind": "user", "key": "user-123"}
    result = await to_lang_graph(
        resolve_graph("support-graph", context=ctx),
        {
            "tool_handlers": registry.tools,
            "context": ctx,
            # optional: supply your own model per node
            "model_factory": lambda node: ChatOpenAI(model=node["config"]["model"]["name"]),
        },
    ).invoke("I was double charged")
    print(result["response"])

asyncio.run(main())
```

## How It Works

- Uses the system prompt and conversation history defined in your LaunchDarkly flag config.
- Template placeholders (`{{variable}}`) in the prompt are substituted using `variables` before the call.
- The LangGraph ReAct agent manages the full reasoning and tool-call loop autonomously — reasoning through steps, calling tools, and deciding when to stop.
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
