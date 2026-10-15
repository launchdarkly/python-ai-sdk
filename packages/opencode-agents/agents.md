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

`create_opencode_handler(*, events, session_id, capture_content=None)` returns a `ProviderHandler`. `events` and `session_id` are required. Each invocation calls `events(config, user_input, variables, history, session_id)` and expects a fresh async iterable.

Assistant text is emitted only after `message.updated` marks that message `role == "assistant"`. A part snapshot is the full text of that part so far; a later delta appends. A text delta without `partID` is dropped.

## Common pitfalls

The AI Config, user input, variables, and history are arguments to `events`. A callable that ignores them never sees the prompt.

`provides_for` is `('*', 'agent')`, so a judge is routed here too. Pass `skip_judges=True` until this handler runs the prompt it is given. Otherwise `events` runs again on the judge prompt and the score is wrong.

`session_id` is required. The reducer does not adopt the first session on a shared stream. Other sessions are dropped, including their `session.error`. `session.idle` for this session ends the run. If the iterable is a server-wide stream and you neither send `session.idle` nor close it, `done` never arrives.
