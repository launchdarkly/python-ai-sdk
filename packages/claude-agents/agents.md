# Agent Guide — `launchdarkly-ai-claude-agents`

> **Span shape.** This package emits `invoke_agent` → `chat {model}` → `execute_tool {name}`, with
> tool spans as siblings of `chat`. Span construction lives in `spans.py` beside the handler.
> Conversation content is on unless the caller passes `capture_content=False`, or the process turns it off. See [Process-wide defaults](../client/README.md#process-wide-defaults).
> `TELEMETRY-CONTRACT.md` at the repo root is the authority; read it before changing span code.


This document tells an agent exactly how this package is implemented so it can be correctly modified, debugged, or used as a reference when building a new handler.

---

## Role and Routing

This is a **Tier 1 handler package**. It wraps the `claude-agent-sdk` Python package and exposes a `ProviderHandler` that routes to flag variations where:

```
provides_for = ('Anthropic', 'agent')
```

That means the LaunchDarkly flag variation must have `provider.name == "Anthropic"` and `meta.mode == "agent"`.

---

## File Map

| File | Responsibility |
|---|---|
| `src/launchdarkly_ai_claude_agents/handler.py` | All implementation — prompt building, MCP tool wiring, agentic loop, telemetry |
| `src/launchdarkly_ai_claude_agents/graph.py` | `claude_graph()` convenience wrapper around `graph()` |
| `src/launchdarkly_ai_claude_agents/native_graph.py` | `to_claude_agents()` native graph adapter |
| `src/launchdarkly_ai_claude_agents/builtins.py` | Pre-constructed `NativeTool` sentinels for Claude built-in tools |
| `src/launchdarkly_ai_claude_agents/skills.py` | `create_claude_agents_skills_handler()`, exported from the package root |
| `src/launchdarkly_ai_claude_agents/__init__.py` | Package exports |
| `src/launchdarkly_ai_claude_agents/_version.py` | `PACKAGE_NAME` and `__version__`, sent with `$ld:ai:sdk:info` and `$ld:ai:sdk:usage` |

---

## Exports

```python
# Factory — returns a ProviderHandler with provides_for attached
def create_claude_agents_handler(
    *,
    capture_content: bool | None = None,
    cwd: str | os.PathLike[str] | None = None,
    setting_sources: list[SettingSource] | None = None,
) -> ProviderHandler: ...

# Convenience wrapper — equivalent to config(key=config_key, handler=create_claude_agents_handler(cwd=cwd, setting_sources=setting_sources)).invoke(user_input, context)
def claude_agents(config_key: str, user_input: str, context: dict, *, cwd=None, setting_sources=None, **kwargs) -> ProviderResponse: ...

# Graph convenience wrapper — equivalent to graph(key, handlers=[create_claude_agents_handler(cwd=cwd, setting_sources=setting_sources)], **options)
def claude_graph(key: str, *, cwd=None, setting_sources=None, **options): ...

# Native graph adapter — builds a Claude code-agents graph from a resolved GraphDefinition.
# opts: context, tool_handlers, and the host options cwd and setting_sources
def to_claude_agents(def_promise: Awaitable[GraphDefinition], opts: ClaudeNativeGraphOptions | None = None) -> NativeGraphRunner: ...

# NativeTool sentinels for Claude Code built-in capabilities
ClaudeBash: NativeTool
ClaudeRead: NativeTool
ClaudeEdit: NativeTool
ClaudeWrite: NativeTool
ClaudeGlob: NativeTool
ClaudeGrep: NativeTool
ClaudeWebFetch: NativeTool
ClaudeWebSearch: NativeTool
ClaudeTodoWrite: NativeTool
ClaudeNotebookEdit: NativeTool
```

---

## Implementation Details

### 1. Prompt Construction (`_build_prompt`)

The handler uses a flat `prompt` + optional `system_prompt` shape that the `claude_agent_sdk.query()` call accepts:

```
config.instructions present?
  → system_prompt = parse_template(config.instructions, variables)
  → prompt = user_input

config.messages present?
  → system-role messages → system_prompt (joined with \n)
  → non-system messages → joined as plain text → prepended to user_input
  → prompt = conversation_history + "\n\n" + user_input

neither?
  → prompt = user_input, no system_prompt
```

Note: the `messages` path collapses conversation history into a single flat string — roles are not individually structured. This is a limitation of the agent SDK's `query()` interface.

### 2. Tool Wiring (`_build_tool_mcp`)

Tools are delivered via an in-process MCP server, not as raw JSON schema defs. The pipeline:

1. Each `Tool` in `config.tools` is converted to a `claude_agent_sdk.tool()` call.
2. The tool's executor calls `tool_handlers[tool_name](args)` and returns `{ "content": [{ "type": "text", "text": str(result) }] }`.
3. All tools are registered in a single `create_sdk_mcp_server(name="tool-mcp", ...)`.
4. The MCP server is passed to `query()` via `ClaudeAgentOptions(mcp_servers={"tool-mcp": mcp_server})`.
5. Allowed tools are prefixed: `mcp__tool-mcp__<tool_name>`.

If `config.tools` is absent, the MCP server is not created and `allowed_tools` is an empty list.

**`tools` is always sent, and it is `[]` when the config attaches no `NativeTool`.** The explicit empty list switches every Claude Code built-in off. Omitting the key would leave the Agent SDK default, the full built-in set, on every run, and the read-only tools in it need no permission. A config turns a built-in on by attaching its sentinel (`ClaudeRead`, `ClaudeBash`, …). User-defined tools go through the MCP server and are not governed by `tools`. `to_claude_agents` follows the same rule.

### 2a. Host options (`cwd`, `setting_sources`)

`create_claude_agents_handler(cwd=..., setting_sources=...)` takes two trusted, app-supplied options and passes them to every `query()` call, blocking and streaming. `_options_parameters` forwards only the allowlisted model and run settings from `model.parameters`, and both keys are in `_CLAUDE_AGENT_OPTIONS_EXCLUDED_KEYS`, so a config can neither set nor override them. **`setting_sources` is always sent, `[]` unless the application gives sources**: the Agent SDK reads `None` as "load every source", which put the host's plugins, skills, hooks, permission rules and `CLAUDE.md` into every run (AIC-3584). `[]` still lets the SDK read the managed-settings policy tier. `cwd` is sent only when given. Settings-file `env` (`ANTHROPIC_BASE_URL`, `CLAUDE_CODE_USE_BEDROCK` / `_VERTEX`, proxy variables), `apiKeyHelper` and `model` come from those sources too, so `[]` drops them; the README's migration note covers this.

`claude_agents()` and `claude_graph()` take `cwd` and `setting_sources` keyword-only and build the handler with them, so neither reaches `config()` or `graph()`. `to_claude_agents()` reads them from `opts` and passes `_host_query_options(...)` to every node's `query()`, root and sub-agents. All four entry points share `_host_query_options`, so all four default to `[]`.

### 3. Agentic Loop

The `claude_agent_sdk` handles the loop internally. The handler iterates the `query()` async generator, holding an explicit reference so it can be explicitly closed on early exit:

```python
gen = query_fn(prompt=prompt, options=options)
try:
    async for message in gen:
        if isinstance(message, ResultMessage):
            _raise_if_run_failed(message)  # raises on a failed run
            # done — extract output and usage
            break
finally:
    await gen.aclose()
```

A `success` subtype is not enough to call the run a success. An error subtype (`error_max_turns`,
`error_during_execution`, ...) fails, and so does a `success` result with `is_error` set. The second
is how the CLI reports a failed Anthropic API call (bad key, no credits, rate limit after retries):
`api_error_status` is set and `result` holds the error text. `_raise_if_run_failed` raises on either
shape, so the caller tracks `$ld:ai:generation:error` instead of being handed the error text as the
model's answer. The native graph adapter applies the same check to every node's run.

The CLI also writes its own `AssistantMessage`, with `model` set to the literal `<synthetic>`, when
an API call fails and after a response stops at `max_output_tokens`. It is not a model response:
it gets no `chat` span and never joins the conversation a later span reports.

**Important:** a bare `return` inside `async for` abandons the generator. Python's asyncio finalizer will try to `aclose()` it later and may raise `RuntimeError: aclose(): asynchronous generator is already running` if the generator is still suspended inside a real SDK `await`. Always use `break` + explicit `aclose()`. See **Appendix A.4** in `TESTING.md`.

### 4. Telemetry

Span name: `'claude.query'`
Span attributes set before the call:
- `gen_ai.operation.name` = `'chat'`
- `gen_ai.system` = `'anthropic'`
- `gen_ai.request.model` = `config.model.name`

Span event before the call:
- `gen_ai.content.prompt` with attribute `gen_ai.prompt` = the raw prompt string

Span attributes set after the call (inside the result branch):
- `gen_ai.usage.input_tokens`
- `gen_ai.usage.output_tokens`
- `gen_ai.usage.total_tokens`

Span event after the call:
- `gen_ai.content.completion` with attribute `gen_ai.completion` = the result string

On error: `span.record_exception(exc)`, status set to ERROR, span ended, error re-raised.

`gen_ai.conversation.id` is a caller-supplied id from `conversation_id(...)`, or the CLI
`session_id` from the `init` message when the caller supplied none. Write-if-absent: the caller
id wins. An app that opens a fresh CLI session per turn and re-feeds history must pass its own
conversation id, or each turn becomes its own conversation.

### 5. Agent Skills (`skills.py`)

`create_claude_agents_skills_handler(...)` is exported from the package root. It reports `claude-agents.createClaudeAgentsSkillsHandler` and builds the handler through `_create_claude_agents_handler(..., _prepare=...)`.

`_prepare` is a package-internal hook, a private keyword argument of the internal factory only, so `create_claude_agents_handler`'s signature stays unchanged. It runs at the start of every invoke and stream, inside the `invoke_agent` span, and returns `None` (the run is exactly the core handler's) or a `QueryPreparation`. That holds extra `query()` options, extra built-ins appended to `tools` (never to `allowed_tools`), and a `cleanup` that both paths await last in their `finally`.

For a variation with skills, `_prepare_skills`:

1. `tempfile.mkdtemp(prefix="ld-skills-", dir=skills_temp_dir)`, then `os.path.realpath`, then `os.mkdir(<dir>/launchdarkly)`.
2. `write_skills(skill_refs(config), <dir>/launchdarkly/skills, prune=False)`. A report that is not `ok` raises, naming each failed key, and `query()` is never called.
3. Returns `plugins=[{"type": "local", "path": <dir>/launchdarkly}]`, `skills=["launchdarkly:<key>", ...]` and `extra_builtins=["Skill"]`.

The plugin's name comes from its directory's base name, so no manifest is written. The skill names are qualified because a bare name would also match a host plugin's skill. `claude-agent-sdk`'s `SdkPluginConfig` has no `skipMcpDiscovery` key, so unlike TypeScript the plugin entry carries only `type` and `path`. The directory holds no `.mcp.json`, so there is nothing to discover. `setting_sources` is never `None` here, because the core handler always sends it. With `skills` set and `setting_sources=None`, `claude-agent-sdk` sends `--setting-sources=user,project`. The tool hooks are installed whenever `Skill` is in `tools`, so a `Skill` call gets an `execute_tool Skill` span. The directory is removed with a synchronous `shutil.rmtree`, so a cancellation during teardown cannot interrupt it. There's no shared root, no watcher and no prune: every run writes exactly its own variation's references, which is also how detaching a skill takes effect.

---

## OTel Setup

This package emits one span per invocation using `opentelemetry-api`. **No OTel configuration is needed in this package** — the tracer provider is registered by `init_client()` in `launchdarkly-ai-server` (or `launchdarkly-ai-python`).

To receive spans, install the OTel SDK in your application:
```sh
pip install "launchdarkly-ai-python[otel]"
# or:
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

Span names and attributes are described in [Implementation Details → Telemetry](#4-telemetry) above.

---

## `init_client()` — When to Call It

**You do not need to call `init_client()` from this package.** Every entry point (`claude_agents()`, `config().invoke()`) lazily initializes the LaunchDarkly client on the first call, as long as `LD_SDK_KEY` is set in the environment.

**Call `init_client()` explicitly in your application startup code when you need to:**

- **Pass custom options** — `serviceName`, `environment`, or OTel configuration:
  ```python
  from launchdarkly_ai_python import init_client  # or launchdarkly_ai_server
  await init_client({"serviceName": "my-service", "environment": "production"})
  ```
- **Use a custom or edge runtime (BYOC path)** — pass a pre-initialized client that satisfies `LDClientInterface`:
  ```python
  from launchdarkly_ai_server import init_client
  ld_client = create_your_custom_client(os.environ["LD_SDK_KEY"])
  await init_client(ld_client)
  ```
- **Pre-warm the connection** — call `init_client()` at startup to avoid cold-start latency on the first user request.

`init_client()` is idempotent — calling it twice is a no-op. Never call `init_client()` inside this handler package; initialization belongs in application startup code. Full details in the [`launchdarkly-ai-server` agents.md](../client/agents.md#lifecycle-invariants).

---

## Dependencies

| Package | Why |
|---|---|
| `claude-agent-sdk` | `query()`, `tool()`, `create_sdk_mcp_server()`, `ResultMessage` |
| `launchdarkly-ai-server` | `AiConfigRep`, `ProviderHandler`, `parse_template`, `get_client`, `make_track_data` |
| `opentelemetry-api` | `StatusCode`, `trace.get_tracer().start_span()` for span creation |

---

## Common Pitfalls

- **MCP tool name prefix**: tools registered in the MCP server are accessible as `mcp__tool-mcp__<name>`. The `allowed_tools` list must use this prefix; omitting it will cause the agent to not invoke any tools.
- **Async generator teardown**: always use `break` inside `async for` (not `return`) and wrap the loop in `try/finally: await gen.aclose()`. See TESTING.md Appendix A.4.
- **`message.usage` shape**: the raw usage dict from the `claude_agent_sdk` is passed through directly to `parse_usage`. It accepts `input_tokens`/`output_tokens` which the SDK provides.
- **Native tools** (`ClaudeWebSearch`, etc.) are registered in `tool_handlers` as `NativeTool` sentinel instances (not callables). The handler's `_partition_tools` function separates them from user-defined tools — native tools go into `native_tool_names` for the agent's built-in access, user tools go through the MCP server.
