# `launchdarkly-ai-opencode-agents`

OpenCode handler for `launchdarkly-ai-server`. `async with handler` starts `opencode serve` and waits until it is healthy. The process stops when that context exits. It does not depend on an OpenCode SDK.

**`provides_for`:** `['*', 'agent']` — matches an agent-mode variation for any provider.

```bash
pip install "launchdarkly-ai-server[ld]" launchdarkly-ai-opencode-agents
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

Entering the handler starts `opencode serve` and returns once it is healthy. The process stops when the context exits, after `config().stream()` returns. Put a wall-clock limit around `config().stream()`, not around the context. A judge routed here runs as a full OpenCode agent: the server's tools, in `cwd`, apply to the judge prompt. There is no tool-less judge mode. The handler yields AG-UI text and tool events. `config().stream()` yields `chunk` then `done`.
