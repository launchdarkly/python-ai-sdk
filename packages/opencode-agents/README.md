# `launchdarkly-ai-opencode-agents`

OpenCode handler for `launchdarkly-ai-server`. It reduces caller-supplied OpenCode session events into assistant text. It does not start OpenCode and it does not depend on an OpenCode SDK.

**`provides_for`:** `['*', 'agent']` — matches an agent-mode variation for any provider.

```bash
pip install launchdarkly-ai-server launchdarkly-server-sdk launchdarkly-ai-opencode-agents
```

```python
import asyncio

from launchdarkly_ai_opencode_agents import create_opencode_handler
from launchdarkly_ai_server import config, init_client, shutdown


async def main() -> None:
    await init_client()
    handler = create_opencode_handler(base_url="http://127.0.0.1:4096")
    async for event in config(key="my-agent", handler=handler).stream(
        "Hello", {"kind": "user", "key": "user-1"}
    ):
        print(event)
    await shutdown()


asyncio.run(main())
```

`base_url` is a running `opencode serve`. The handler owns every HTTP call to that server. A caller does not pass an HTTP client or transport. Each call opens its own session, including a judge routed to this handler. A judge runs as a full OpenCode agent: the server's tools, in the server's working directory, apply to the judge prompt. There is no tool-less judge mode. The handler yields AG-UI text and tool events. `config().stream()` yields `chunk` then `done`.
