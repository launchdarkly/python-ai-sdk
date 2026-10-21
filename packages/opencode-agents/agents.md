# Agent Guide — `launchdarkly-ai-opencode-agents`

```python
provides_for = ('*', 'agent')
```

Each call starts `opencode serve`, speaks the stable HTTP API (`/session`, `/session/{id}/message`, `/event`, `/session/{id}/abort`), and stops the process when the stream finishes or the caller closes it. The package does not import or construct `anomalyco/opencode-sdk-python`.

This package is Python-only. There is no JS counterpart in this slice. The spec section, API surface list, and `usage-ids.txt` row live in `launchdarkly/ai-sdks-monorepo`, which is not this repository. The usage helper ids are `opencode-agents.createOpenCodeHandler` and `opencode-agents.opencodeAgents`. The package is a normal pre-v1 package, same as the other handlers.

## File map

| File | Role |
| --- | --- |
| `handler.py` | Reducer, `create_opencode_handler`, `opencode_agents` |
| `process.py` | Starts and stops `opencode serve` for one call |
| `server.py` | The HTTP session this process just started |
| `spans.py` | `invoke_agent` root, and the `chat` / `execute_tool` constructors the parity checks call |
| `_version.py` | Package name and version sent with usage events |

## Dependencies

`launchdarkly-ai-server`, `httpx`, and `opentelemetry-api`. `launchdarkly-server-sdk` is what `init_client()` loads for a normal process. No OpenCode SDK. The `opencode` executable is the default command.

## init_client

Call `init_client()` before `config()`, the same way as the other handlers. The example in `README.md` does.

## OTel

A run opens one `invoke_agent` span and sets `launchdarkly.operation.type`. The handler stream is the AG-UI text and tool subset. `config().stream()` is `chunk` then `done`.

## Call shape

`create_opencode_handler(*, cwd, config, env, capture_content=None, model=None, agent=None, command=None)` returns a `ProviderHandler`. `cwd` is the working directory. `config` is the path of the OpenCode config file. `env` replaces the child environment, and the handler sets `OPENCODE_CONFIG` from `config`. `command` defaults to `opencode`. A test passes a fake executable instead of a URL. There is no `base_url`, client, or transport argument.

The process starts on the call and stops when the stream finishes or the caller closes it. Closing the stream still aborts the session before the process is stopped.

Assistant text is emitted only after `message.updated` marks that message `role == "assistant"`. A part snapshot is the full text of that part so far; a later delta appends. A text delta without `partID` is dropped.

## Common pitfalls

`provides_for` is `('*', 'agent')`, so a judge is routed here too. Each call starts its own process, so the judge prompt does not enter the agent's session and does not need `skip_judges=True`. The judge still runs as a full OpenCode agent, with the server's tools, in `cwd`. This package does not take a tool-less judge agent.

The reducer does not adopt the first session on the bus. Other sessions are dropped, including their `session.error`. `session.idle` for this session ends the run.

`env` is not merged with the parent environment. A bare `opencode` command needs `PATH` in `env`.
