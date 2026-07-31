# `launchdarkly-ai-claude-agents`

Anthropic Claude handler for `launchdarkly-ai-server` using the **Claude Agent SDK** (`claude-agent-sdk`). Runs an agentic query loop with native MCP tool support.

**`provides_for`:** `['Anthropic', 'agent']` — matches flag variations where `provider.name` is `"Anthropic"` and `meta.mode` is `"agent"`.

## Installation

```bash
pip install launchdarkly-ai-server launchdarkly-ai-claude-agents
```

Set `ANTHROPIC_API_KEY` in your environment (the Anthropic SDK reads it automatically).

## Usage

### With `config()`

```python
import asyncio
from launchdarkly_ai_server import config, shutdown
from launchdarkly_ai_claude_agents import create_claude_agents_handler

async def main():
    result = await config(
        key="my-ai-config-flag",
        handler=create_claude_agents_handler(),
        tool_handlers={"search": lambda q: "..."},
    ).invoke("What is feature flagging?", {"kind": "user", "key": "user-123"})

    print(result.response)
    await shutdown()

asyncio.run(main())
```

### Convenience wrapper

```python
import asyncio
from launchdarkly_ai_claude_agents import claude_agents

async def main():
    user_input = "What is feature flagging?"
    result = await claude_agents(
        user_input,
        {"kind": "user", "key": "user-123"},
        {"key": "my-ai-config-flag"},
        variables={"user_input": user_input},
    )
    print(result.response)

asyncio.run(main())
```

### Agent graphs — `claude_graph()`

Runs a LaunchDarkly agent graph with the Claude agent handler pre-bound. Equivalent to calling the base `graph()` with `handlers=[create_claude_agents_handler()]`. See the [core client docs](../client/README.md#graphkey-options) for the full `graph()` API.

```python
import asyncio
from launchdarkly_ai_claude_agents import claude_graph

async def main():
    result = await claude_graph("support-graph").invoke(
        "I was double charged",
        {"kind": "user", "key": "user-123"},
    )
    print(result["response"])

asyncio.run(main())
```

### Native graph adapter — `to_claude_agents()`

Converts a `resolve_graph()` result into a framework-native Claude sub-agent tree (post-order traversal: leaves → root). Each child node is wrapped as an in-process MCP tool; the root runs a single `query()` with its children accessible as sub-agent tools. No cloud-registered agents required.

```python
import asyncio
from launchdarkly_ai_server import resolve_graph
from launchdarkly_ai_claude_agents import to_claude_agents

async def main():
    ctx = {"kind": "user", "key": "user-123"}
    result = await to_claude_agents(
        resolve_graph("support-graph", context=ctx),
        {"tool_handlers": registry.tools, "context": ctx},
    ).invoke("I was double charged")
    print(result["response"])

asyncio.run(main())
```

## How It Works

- Uses the system prompt and conversation history defined in your LaunchDarkly flag config.
- Template placeholders (`{{variable}}`) in the prompt are substituted using `variables` before the call.
- If tools are defined in the flag config, the Claude Agent SDK handles all tool dispatch and the agentic loop automatically — no extra wiring needed.
- Emits an OTel span and LaunchDarkly telemetry for every call.

## Built-in Claude Tools

This package exports `NativeTool` sentinels for Claude Code built-in capabilities. Place them as values in `tool_handlers` to enable the corresponding native Claude tool without writing a handler function:

| Export | Claude SDK tool name |
|---|---|
| `ClaudeBash` | `Bash` |
| `ClaudeRead` | `Read` |
| `ClaudeEdit` | `Edit` |
| `ClaudeWrite` | `Write` |
| `ClaudeGlob` | `Glob` |
| `ClaudeGrep` | `Grep` |
| `ClaudeWebFetch` | `WebFetch` |
| `ClaudeWebSearch` | `WebSearch` |
| `ClaudeTodoWrite` | `TodoWrite` |
| `ClaudeNotebookEdit` | `NotebookEdit` |

```python
from launchdarkly_ai_claude_agents import ClaudeWebSearch, create_claude_agents_handler
from launchdarkly_ai_server import config

result = await config(
    key="my-ai-config-flag",
    tool_handlers={"web-search": ClaudeWebSearch},
    handler=[create_claude_agents_handler()],
).invoke("What are the latest LD release notes?", {"kind": "user", "key": "user-123"})
```

## Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key (read automatically by the Anthropic SDK) |
| `LD_SDK_KEY` | LaunchDarkly server-side SDK key |
| `LD_SERVICE_NAME` | OTel `service.name` resource attribute (default: `python-sdk`) |
| `LD_ENVIRONMENT` | `deployment.environment` attribute attached to telemetry |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint override (default: LaunchDarkly Observability backend) |
