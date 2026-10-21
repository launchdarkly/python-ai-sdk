# Agent Guide — `launchdarkly-ai-opencode-agents`

```python
provides_for = ('*', 'agent')
```

The handler reduces OpenCode session events the caller supplies. It does not import or construct `anomalyco/opencode-sdk-python`, and it does not start an OpenCode process.

This package is Python-only. There is no JS counterpart in this slice. The spec section, API surface list, and `usage-ids.txt` row live in `launchdarkly/ai-sdks-monorepo`, which is not this repository. The usage helper ids are `opencode-agents.createOpenCodeHandler` and `opencode-agents.opencodeAgents`. The package is a normal pre-v1 package, same as the other handlers.

## File map

| File | Role |
| --- | --- |
| `handler.py` | Reducer, `create_opencode_handler`, `opencode_agents` |
| `spans.py` | `invoke_agent` root, and the `chat` / `execute_tool` constructors the parity checks call |
| `_version.py` | Package name and version sent with usage events |

## Dependencies

`launchdarkly-ai-server` and `opentelemetry-api`. `launchdarkly-server-sdk` is what `init_client()` loads for a normal process. No OpenCode SDK.

## init_client

Call `init_client()` before `config()`, the same way as the other handlers. The example in `README.md` does.

## OTel

A run opens one `invoke_agent` span and sets `launchdarkly.operation.type`. The handler stream is the AG-UI text and tool subset. `config().stream()` is `chunk` then `done`.

## Call shape

`create_opencode_handler(*, base_url, capture_content=None, model=None, agent=None)` returns a `ProviderHandler`. `base_url` is a running `opencode serve`. The handler owns the HTTP client. Each call opens its own session.

Assistant text is emitted only after `message.updated` marks that message `role == "assistant"`. A part snapshot is the full text of that part so far; a later delta appends. A text delta without `partID` is dropped.

## Common pitfalls

The handler posts the prompt for the session it just opened. A caller that only watches a pre-existing session never sees that prompt.

`provides_for` is `('*', 'agent')`, so a judge is routed here too. Each call opens its own session, so the judge prompt does not enter the agent's session and does not need `skip_judges=True`. The judge still runs as a full OpenCode agent, with the server's tools, in the server's working directory. This package does not take a tool-less judge agent.

The reducer does not adopt the first session on the bus. Other sessions are dropped, including their `session.error`. `session.idle` for this session ends the run.
