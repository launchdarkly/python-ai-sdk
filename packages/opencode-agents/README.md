# `launchdarkly-ai-opencode-agents`

OpenCode handler for `launchdarkly-ai-server`. `async with handler` starts `opencode serve` and waits until it is healthy. The process stops when that context exits. It does not depend on an OpenCode SDK.

**`provides_for`:** `['*', 'agent']` — matches an agent-mode variation for any provider.

```bash
pip install launchdarkly-ai-server launchdarkly-server-sdk launchdarkly-ai-opencode-agents
```

```python
import asyncio
import os

from launchdarkly_ai_opencode_agents import create_opencode_handler
from launchdarkly_ai_server import config, init_client, shutdown


async def main() -> None:
    await init_client()
    handler = create_opencode_handler(
        cwd="/work",
        config="/home/app/.vega/opencode.json",
        env=os.environ.copy(),
    )
    async with handler:
        async for event in config(key="my-agent", handler=handler).stream(
            "Hello", {"kind": "user", "key": "user-1"}
        ):
            print(event)
    await shutdown()


asyncio.run(main())
```

`cwd` is the working directory, `config` is the OpenCode config file, and `env` is the child environment. The handler sets `OPENCODE_CONFIG` from `config`, replaces any `OPENCODE_SERVER_USERNAME` in `env` with `opencode`, replaces any `OPENCODE_SERVER_PASSWORD` with a generated secret, and binds `opencode serve` to loopback. Requests to that server use that username and secret as basic auth. A caller does not pass a URL, an HTTP client, or a transport. `command` overrides the `opencode` executable; a test passes a fake executable there.

Entering the handler starts `opencode serve` and returns once it is healthy. The process stops when the context exits, after `config().stream()` returns. Put a wall-clock limit around `config().stream()`, not around the context. The handler yields AG-UI text and tool events. `config().stream()` yields `chunk` then `done`.

## Tools

The AI Config's `tools`, filtered to the keys also present in `tool_handlers`, is the only catalog the model sees for one call — every other tool on the running `opencode serve`, including every built-in, is denied. A variation with no tools (and every judge, which always calls with `tool_handlers=None`) gets a model with no tools at all.

```python
from launchdarkly_ai_opencode_agents import OpenCodeBash, OpenCodeRead

response = await config(
    key="my-agent",
    handler=handler,
    tool_handlers={
        "run-shell": OpenCodeBash,       # an OpenCode built-in, allowed by name
        "read-file": OpenCodeRead,
        "get-weather": get_weather,      # a plain callable, served from a per-call bridge
    },
).invoke(user_input, context)
```

A `NativeTool` (see `OpenCodeBash`, `OpenCodeRead`, and the rest of this package's sentinels) names an OpenCode built-in or a server-configured MCP tool directly — OpenCode already has it, so the handler only allows it for this call's session. A plain callable has no route into OpenCode otherwise: `POST /session/{id}/message`'s own `tools` field is a deprecated boolean permission map, not a place to describe one. Each call with at least one callable starts its own loopback MCP server, registers it on the running `opencode serve` with `POST /mcp`, and disconnects it when the call ends. This is unit-tested directly (`test_tool_bridge.py`) and through the handler against a fake `opencode serve` (`test_tool_contract.py`); it has not been exercised against a real OpenCode MCP client.

`chat` and `execute_tool` spans are opened and closed from the same bus events the AG-UI stream is derived from, named after the AI Config key (not OpenCode's own tool id or the bridge's composed MCP name).
