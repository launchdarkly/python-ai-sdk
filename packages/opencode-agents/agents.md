# Agent Guide — `launchdarkly-ai-opencode-agents`

```python
provides_for = ('*', 'agent')
```

Each call starts `opencode serve`, speaks the stable HTTP API (`/session`, `/session/{id}/message`, `/event`, `/session/{id}/abort`), and stops the process when the stream finishes or the caller closes it. The package does not import or construct `anomalyco/opencode-sdk-python`.

This package is Python-only. There is no JS counterpart in this slice. The spec section, API surface list, and `usage-ids.txt` row live in `launchdarkly/ai-sdks-monorepo`, which is not this repository. The usage helper ids are `opencode-agents.createOpenCodeHandler` and `opencode-agents.opencodeAgents`. The package is a normal pre-v1 package, same as the other handlers.

## File map

| File | Role |
| --- | --- |
| `handler.py` | Reducer, `create_opencode_handler`, `opencode_agents`, tool planning (`_plan_tools`, `_build_tool_ruleset`) |
| `process.py` | Starts and stops `opencode serve` for one call |
| `server.py` | The HTTP session this process just started; `register_mcp_bridge`/`disconnect_mcp_bridge` |
| `spans.py` | `invoke_agent` root, the `chat` / `execute_tool` constructors, and `close_open_spans`/`abandon_open_spans` |
| `tool_bridge.py` | The per-call loopback MCP server a plain callable tool is served from |
| `builtins.py` | `NativeTool` sentinels for OpenCode's own built-ins (`OpenCodeBash`, etc.) |
| `_version.py` | Package name and version sent with usage events |

## Dependencies

`launchdarkly-ai-server`, `httpx`, and `opentelemetry-api`. `launchdarkly-server-sdk` is what `init_client()` loads for a normal process. No OpenCode SDK. The `opencode` executable is the default command.

## init_client

Call `init_client()` before `config()`, the same way as the other handlers. The example in `README.md` does.

## OTel

A run opens one `invoke_agent` span and sets `launchdarkly.operation.type`. The handler stream is the AG-UI text and tool subset. `config().stream()` is `chunk` then `done`.

## Call shape

`create_opencode_handler(*, cwd, config, env, capture_content=None, model=None, agent=None, command=None)` returns a `ProviderHandler`. `cwd` is the working directory. `config` is the path of the OpenCode config file. `env` replaces the child environment, and the handler sets `OPENCODE_CONFIG` from `config`. Each start also sets `OPENCODE_SERVER_USERNAME` to `opencode` and `OPENCODE_SERVER_PASSWORD` to a generated secret, replacing either value inherited from `env`, and sends that pair as basic auth. `command` defaults to `opencode`. A test passes a fake executable instead of a URL. There is no `base_url`, client, or transport argument.

`async with handler` starts the process and returns once `/global/health` returns 200 and `healthy: true`. Concurrent entries share that process. The process stops when the last context exits, after `config().stream()` returns, including when the caller times out or closes the stream. A wall-clock budget goes around `config().stream()`, not around the context. Closing the stream still aborts the session. Without the context, one call starts and stops its own process.

Assistant text is emitted only after `message.updated` marks that message `role == "assistant"`. A part snapshot is the full text of that part so far; a later delta appends. A text delta without `partID` is dropped.

## Tool contract (Option C)

`_plan_tools(config, tool_handlers)` is the single place that decides what one call offers: the intersection of `config["tools"]` and the keys present in `tool_handlers` (already wrapped by `tracking.py`, so a `NativeTool` entry carries `NATIVE_TOOL_KEY`). A `NativeTool` is allowed by its own `tool_name`. A plain callable is queued as a `_CallableEntry` and only gets a model-visible name once `_stream_impl` has started that call's `ToolBridge` and knows its generated MCP server name (`ld_<run id>`); `bridge_tool_name` composes the name OpenCode will actually report, matching `mcp/catalog.ts`'s own `sanitize` + `toolName`.

`_build_tool_ruleset(offered_names)` is sent as `POST /session`'s `permission`: a leading `{"*": deny}` rule, then one `{name: allow}` rule per offered name. `Permission.merge` (OpenCode's own code) appends a session's `permission` after the agent's static `opencode.json` permission, and `Permission.disabled` resolves the *last* matching rule — so this ruleset always wins, and an empty `offered_names` still denies everything rather than falling back to the host's static catalog.

The reducer (`_Reducer`) opens and closes `chat`/`execute_tool` spans from the same tool-part and message-info transitions the AG-UI events already come from, independent of what was actually offered — a span reports what OpenCode said happened, not what was requested. `tool_display_names` maps the name OpenCode reports (a native id or a bridge's composed name) back to the AI Config key for the span name; an unmapped name falls back to itself. A `NativeTool`'s own `$ld:ai:tool_call` tracking stub fires once, when its span opens (mirroring `claude-agents`'s `PreToolUse` hook); a bridged callable's tracked wrapper already fires it when the bridge invokes it, so the reducer does not fire it again.

`open_tool_spans`/`open_model_spans` are owned by `_stream_impl`, not the reducer, specifically so `close_open_spans`/`abandon_open_spans` (registered in `test_cross_handler_parity.py`'s `HOOK_BASED_TOOL_HANDLERS`, this package's closest analogue to a vendor hook object) can find and end whatever is still open from the streaming `finally`, even if the reducer was never constructed (no raw bus event ever arrived). Keep both span-teardown signatures on one physical line — the parity suite's body extractor is indentation-based and mis-reads a wrapped signature whose closing `) -> None:` lands back at column 0, which is exactly what `ruff format` produces once a signature exceeds the line-length limit.

## Common pitfalls

`provides_for` is `('*', 'agent')`, so a judge is routed here too. Each call starts its own process, so the judge prompt does not enter the agent's session and does not need `skip_judges=True`. Judges always call with `tool_handlers=None` (`judges.py`), so the offered set is empty and the judge gets a tool-less OpenCode agent — unlike `claude-agents`, which leaves Claude Code's built-ins on for a tool-less judge today; that inconsistency is open, not fixed here.

The bridge (`tool_bridge.py`) answers in MCP streamable-HTTP "JSON response" mode only: no SSE, because this bridge never needs to push an unsolicited message to OpenCode. It runs on a background thread (`http.server`), not an asyncio server, to avoid taking on an ASGI dependency for one loopback endpoint; a `tools/call` crosses back onto the caller's event loop with `asyncio.run_coroutine_threadsafe`. None of this has been exercised against OpenCode's real MCP client (`@modelcontextprotocol/sdk`); `fake_opencode.py` only records the `POST /mcp` body, it never actually calls back into a registered bridge, except the one test in `test_tool_contract.py` that does so directly with a raw `httpx` request built from that recorded body.

The reducer does not adopt the first session on the bus. Other sessions are dropped, including their `session.error`. `session.idle` for this session ends the run.

`env` is not merged with the parent environment. A bare `opencode` command needs `PATH` in `env`.
