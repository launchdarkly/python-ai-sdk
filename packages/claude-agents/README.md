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

### Handler options

`create_claude_agents_handler()` takes keyword arguments:

| Option | Default | Description |
|---|---|---|
| `capture_content` | `None` | Put prompts, model output, and tool arguments and results on the emitted spans. `None` follows `configure()`, which is off by default. |
| `cwd` | the process's working directory | Working directory for the Claude Code process the Agent SDK starts. A `str` or path-like. |
| `setting_sources` | `[]` (none) | Which Claude Code settings sources to load: `"user"` (`~/.claude/settings.json`), `"project"` (`.claude/settings.json`, and `CLAUDE.md`), `"local"` (`.claude/settings.local.json`). |

These configure the host process, not the model, so only your application can set them. A flag's model parameters can't set or override either one.

> **A run loads none of the host's Claude Code settings unless you opt in.** The Claude Agent SDK's own default loads the user, project and local settings of the machine it runs on, including their plugins, skills, hooks, permission rules and `CLAUDE.md`. This handler sends `setting_sources=[]` instead, so an agent runs from its LaunchDarkly config alone. To load host settings, name the sources:
>
> ```python
> create_claude_agents_handler(setting_sources=["user", "project", "local"])
> ```
>
> The Agent SDK still reads the managed-settings policy tier (managed settings files and MDM) whatever you pass.

`claude_agents()`, `claude_graph()` and `to_claude_agents()` don't take `cwd` or `setting_sources`, and load no host settings. To set either, build the handler yourself and pass it to `config()` or `graph()`.

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

Runs a LaunchDarkly agent graph with the Claude agent handler pre-bound. Equivalent to calling the base `graph()` with `handlers=[create_claude_agents_handler()]`. See the [core client docs](https://github.com/launchdarkly/python-ai-sdk/blob/main/packages/client/README.md#graphkey-options) for the full `graph()` API.

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

A run has no Claude Code built-in tools (`Read`, `Bash`, and the rest) unless its config attaches them. Tools you define in the flag config are always available to the model through the handler's MCP server, whatever built-ins are attached.

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
