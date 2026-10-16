# `launchdarkly-ai-opencode-agents`

OpenCode handler for `launchdarkly-ai-server`. It reduces caller-supplied OpenCode session events into assistant text. It does not start OpenCode and it does not depend on an OpenCode SDK.

**`provides_for`:** `['*', 'agent']` — matches an agent-mode variation for any provider.

```bash
pip install launchdarkly-ai-server launchdarkly-server-sdk launchdarkly-ai-opencode-agents
```

```python
import asyncio
from collections.abc import AsyncIterator

from launchdarkly_ai_opencode_agents import create_opencode_handler
from launchdarkly_ai_server import config, init_client, shutdown


async def session_events(config, user_input, variables, history, session_id):
    """Subscribe to session_id, send user_input, and stop at session.idle."""
    del config, variables, history
    yield {"type": "message.updated", "properties": {"info": {"id": "m1", "role": "assistant", "time": {"created": 1}}}}
    yield {
        "type": "message.part.delta",
        "properties": {
            "sessionID": session_id,
            "messageID": "m1",
            "partID": "p1",
            "field": "text",
            "delta": "Hello",
        },
    }
    yield {"type": "session.idle", "properties": {"sessionID": session_id}}


async def main() -> None:
    await init_client()
    handler = create_opencode_handler(events=session_events, session_id="ses")
    async for event in config(key="my-agent", handler=handler).stream(
        "Hello", {"kind": "user", "key": "user-1"}
    ):
        print(event)
    await shutdown()


asyncio.run(main())
```

`events` is required. Each invocation calls `events(config, user_input, variables, history, session_id)` and the return value must be a fresh async iterable of that session only. The iterable must end: return it after `session.idle`, or close it yourself. The handler yields AG-UI text and tool events. `config().stream()` yields `chunk` then `done`.

This handler matches every agent-mode provider, so judge selection can land here. It does not run the judge prompt: that call raises without calling `events`, and the judge does not score.
