# `launchdarkly-ai-opencode-agents`

OpenCode handler for `launchdarkly-ai-server`. It reduces caller-supplied OpenCode session events into assistant text and tool events. It does not start OpenCode and it does not depend on an OpenCode SDK.

**`provides_for`:** `['*', 'agent']` — matches an agent-mode variation for any provider.

```bash
pip install launchdarkly-ai-server launchdarkly-server-sdk launchdarkly-ai-opencode-agents
```

```python
from launchdarkly_ai_opencode_agents import create_opencode_handler
from launchdarkly_ai_server import config

handler = create_opencode_handler(events=session_events)
config(key="my-agent", handler=handler).stream("Hello", {"kind": "user", "key": "user-1"})
```

`events` is called on every invocation and must return a fresh async iterable of OpenCode session events. `config().stream()` yields `chunk` then `done`. Tool events are available from `launchdarkly_ai_server.rich_stream`.
