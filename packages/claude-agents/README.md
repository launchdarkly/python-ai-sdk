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

These configure the host process, not the model, so only your application can set them. A flag's model parameters can't set or override either one. `claude_agents()`, `claude_graph()` and `to_claude_agents()` take `cwd` and `setting_sources` too, with the same defaults.

> **A run loads none of the host's Claude Code settings unless you opt in.** The Claude Agent SDK's own default loads the user, project and local settings of the machine it runs on, including their plugins, skills, hooks, permission rules and `CLAUDE.md`. This handler sends `setting_sources=[]` instead, so an agent runs from its LaunchDarkly config alone. To load host settings, name the sources:
>
> ```python
> create_claude_agents_handler(setting_sources=["user", "project", "local"])
> ```
>
> The Agent SDK still reads the managed-settings policy tier (managed settings files and MDM) whatever you pass.

#### Migrating from a version that loaded host settings

Earlier versions left `setting_sources` unset, so the Agent SDK loaded every settings source on the host. A run now loads none of them. Anything a deployment configured through those files stops applying, including:

- plugins, skills, hooks, permission rules and `CLAUDE.md`
- the `env` block, for example `ANTHROPIC_BASE_URL`, `CLAUDE_CODE_USE_BEDROCK` or `CLAUDE_CODE_USE_VERTEX`, and proxy variables
- `apiKeyHelper` and `model`

A deployment that reaches Bedrock, Vertex or a gateway through `~/.claude/settings.json` fails to authenticate after the upgrade. To fix it, either set those variables in the process environment, or load the source that holds them:

```python
create_claude_agents_handler(setting_sources=["user"])
claude_agents("my-ai-config-flag", user_input, context, setting_sources=["user"])
claude_graph("support-graph", setting_sources=["user"])
to_claude_agents(
    resolve_graph("support-graph", context=context),
    {"context": context, "setting_sources": ["user"]},
)
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

## Agent Skills

`create_claude_agents_handler()` doesn't give the model a variation's skills. `create_claude_agents_skills_handler()` does. It takes the same options, plus `skills_temp_dir`.

```python
import os
from launchdarkly_ai_server import FDv2SkillStore, config, set_skill_store
from launchdarkly_ai_claude_agents import create_claude_agents_skills_handler

store = FDv2SkillStore(os.environ["LD_SDK_KEY"]).start()
store.wait_for_skills(timeout=10)
set_skill_store(store)

triage = config(
    key="triage-agent",
    handler=create_claude_agents_skills_handler(setting_sources=[]),
)
```

On every invoke and stream it:

1. Writes the variation's `skills` into a new directory under `skills_temp_dir` (default: `tempfile.gettempdir()`), through the store you configured with `set_skill_store`. Pinned versions are honored.
2. Hands that directory to the Claude Agent SDK as a local plugin named `launchdarkly`, enables only those skills (`launchdarkly:<key>`), and turns on the `Skill` tool.
3. Removes the directory when the run ends, whether it succeeded, failed, or a stream was abandoned.

Each run has its own directory, so concurrent runs and different variations never share files, and nothing needs to watch for changes. Detaching a skill from a variation takes effect on the next run.

`setting_sources` behaves as on the core factory and is always sent, `[]` unless you give sources. Without it, `claude-agent-sdk` would load the `user` and `project` settings sources whenever skills are enabled.

**A run whose skills can't be written doesn't start.** The handler raises before the model is called if no store is configured, the store hasn't received its first payload, a skill fails integrity verification, or a pinned version isn't in the store. The error names each skill. A variation with no skills runs exactly as it would with `create_claude_agents_handler()`, and needs no store.

A `Skill` call shows up in traces as an `execute_tool Skill` span.

**Not covered yet:** `claude_agents()`, `claude_graph()` and `to_claude_agents()` use the core handler, so a variation run through them doesn't get its skills.

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
