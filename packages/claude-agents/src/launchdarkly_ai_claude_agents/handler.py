"""
Claude Agents handler: uses the claude-agent-sdk ``query()`` plus MCP tools.

Span shape: ``invoke_agent`` root, one ``chat`` child per model response, ``execute_tool`` children
per tool call, the same three-span vocabulary the other five handlers emit.

``query()`` reports no request boundaries, so this handler derives them from the message stream: an
``AssistantMessage`` *is* one model response, carrying its own ``message_id`` (the Anthropic response
id), ``usage`` and ``model``. See :mod:`spans` for how those are folded into ``chat`` spans.

Deliberately NOT done here: enabling the CLI's own OTel exporter. Its spans are named outside the
semantic conventions and duplicate what this handler already emits.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    HookMatcher,
    ResultMessage,
    SettingSource,
    StreamEvent,
    ToolResultBlock,
    UserMessage,
    query,
)

from launchdarkly_ai_server import (
    NATIVE_TOOL_KEY,
    AiConfigRep,
    LDContext,
    NativeTool,
    ProviderHandler,
    SpanMessage,
    SpanMessagePart,
    _config,
    _create_handler,
    compose_history,
    content_to_text,
    end_span_once,
    end_unfinished_spans,
    parse_template,
    report_usage,
    set_conversation_id_if_absent,
    set_input_content_attributes,
    set_output_content_attributes,
    set_tool_call_content_attributes,
    tool_result_string,
)
from launchdarkly_ai_server.agui import TurnEmitter
from launchdarkly_ai_server.lifecycle import effective_capture_content
from launchdarkly_ai_server.parameter_forwarding import select_forwarded_parameters
from launchdarkly_ai_server.utils import model_parameters

from ._version import PACKAGE_NAME, __version__
from .spans import (
    MCP_TOOL_PREFIX,
    TOOL_MCP_NAME,
    InferenceSpans,
    Opening,
    ToolCatalog,
    fail_span,
    finish_root_span,
    mark_ok,
    parent_context_of,
    record_conversation_id,
    record_native_tools,
    start_root_span,
    start_tool_span,
    succeed_span,
    tool_display_name,
)

#: Every field ``ClaudeAgentOptions`` declares, classified by hand into exactly one of: forwarded
#: (below), handler-owned (``model``, ``allowed_tools``, ``mcp_servers``, ``hooks``, ``tools``,
#: ``system_prompt``, set by each call site itself), or excluded (below).
#: ``TestClaudeAgentOptionsAcceptsExactlyTheseFields`` in this package's tests asserts this
#: classification stays exhaustive as the SDK's own dataclass changes.
#:
#: Only model and run settings are forwarded: how the model thinks, how long the run may go, and
#: what it may spend. Everything that configures the host process the SDK launches (its binary,
#: environment, working directory, file access, permissions, settings files, plugins, sandbox,
#: session state) stays under the application's control, never a config's.
#:
#: This is the cross-SDK list for Claude Agents (TESTING.md section 1.12), shared with the native
#: graph through :func:`_options_parameters`.
#:
#: The SDK offers no ``temperature``/``top_p``/``top_k``/``max_tokens``/``stop_sequences``/
#: ``tool_choice``/``metadata``, all of which the LaunchDarkly UI's model parameters panel offers
#: for other providers; they are dropped like any other key not listed here.
_CLAUDE_AGENT_OPTIONS_FORWARDED_KEYS = frozenset(
    {
        "betas",
        "effort",
        "fallback_model",
        "max_budget_usd",
        "max_thinking_tokens",
        "max_turns",
        "thinking",
    }
)

#: Forwarded keys whose value must be an object. A config value of any other type is malformed
#: and dropped rather than passed to ``ClaudeAgentOptions``.
_CLAUDE_AGENT_OPTIONS_MAPPING_KEYS = frozenset({"thinking"})

#: Accepted by ``ClaudeAgentOptions`` but never forwarded, and why:
#: * ``cli_path``, ``env``, ``cwd``, ``add_dirs``, ``settings``, ``setting_sources``, ``plugins``,
#:   ``skills``, ``sandbox``, ``user``, ``extra_args``: which binary runs, with what environment,
#:   as which user, with what files, settings, plugins, and CLI arguments. A config that could set
#:   these could run code on, or read files from, the host.
#: * ``permission_mode``, ``permission_prompt_tool_name``, ``can_use_tool``, ``disallowed_tools``,
#:   ``strict_mcp_config``, ``agents``: what the agent is allowed to do and which tools or
#:   subagents it gets. The handler wires tools and permissions itself.
#: * ``resume``, ``session_id``, ``fork_session``, ``continue_conversation``, ``session_store``,
#:   ``session_store_flush``, ``enable_file_checkpointing``: session state on the host.
#: * ``stderr``, ``debug_stderr``, ``include_partial_messages``, ``include_hook_events``,
#:   ``max_buffer_size``, ``load_timeout_ms``: process I/O and transport plumbing, including the
#:   streamed message shape the handler reads.
#: * ``output_format``: structured output, which the handler owns through the config's
#:   ``outputFormat`` by adding its schema to the system prompt. Forwarding a second schema here
#:   would have the reply follow one schema while the prompt asks for another.
#: * ``task_budget``: not one of the agreed run settings yet; ``max_turns`` and ``max_budget_usd``
#:   cover run limits.
#:
#: Named for the drift test and for review, not read at runtime: the forwarded list above already
#: leaves these out, so nothing needs to subtract them again.
_CLAUDE_AGENT_OPTIONS_EXCLUDED_KEYS = frozenset(
    {
        "add_dirs",
        "agents",
        "can_use_tool",
        "cli_path",
        "continue_conversation",
        "cwd",
        "debug_stderr",
        "disallowed_tools",
        "enable_file_checkpointing",
        "env",
        "extra_args",
        "fork_session",
        "include_hook_events",
        "include_partial_messages",
        "load_timeout_ms",
        "max_buffer_size",
        "output_format",
        "permission_mode",
        "permission_prompt_tool_name",
        "plugins",
        "resume",
        "sandbox",
        "session_id",
        "session_store",
        "session_store_flush",
        "setting_sources",
        "settings",
        "skills",
        "stderr",
        "strict_mcp_config",
        "task_budget",
        "user",
    }
)

# ---------------------------------------------------------------------------
# Tool wiring
# ---------------------------------------------------------------------------


async def _build_tool_mcp(
    config_tools: dict[str, Any],
    handlers: dict[str, Any],
) -> Any:
    """Build an in-process SDK MCP server from LD config tools + handler functions.

    Imports the SDK lazily rather than off the module-level ``claude_agent_sdk`` import that
    ``query``/``ClaudeAgentOptions``/etc. use: ``native_graph.py`` (out of scope for this telemetry
    pass) calls this function too, and its own tests mock the SDK by patching
    ``importlib.import_module`` rather than this module's names.
    """
    import importlib

    sdk = importlib.import_module("claude_agent_sdk")
    tool_fn = sdk.tool
    create_server = sdk.create_sdk_mcp_server

    tool_objs = []
    for tool_name, tool_cfg in config_tools.items():

        async def _execute(args: Any, _name: str = tool_name) -> dict[str, Any]:
            handler = handlers.get(_name)
            if not handler:
                raise ValueError(f'No handler registered for tool "{_name}"')
            result = await handler(args) if _is_coroutine(handler) else handler(args)
            return {"content": [{"type": "text", "text": tool_result_string(result)}]}

        mcp_tool = tool_fn(
            tool_name,
            tool_cfg.get("description", ""),
            tool_cfg.get("parameters") or {},
        )(_execute)
        tool_objs.append(mcp_tool)

    return create_server(name=TOOL_MCP_NAME, version="1.0.0", tools=tool_objs)


def _partition_tools(
    config_tools: dict[str, Any] | None,
    tool_handlers: dict[str, Callable[..., Any] | NativeTool],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """
    Returns (native_tool_map, user_config_tools, native_tool_names).

    native_tool_map  : provider tool name → tracking stub (for PreToolUse/PostToolUse hooks)
    user_config_tools: LD tool definitions for user-defined tools (sent via MCP)
    native_tool_names: provider-facing names for query(options.tools=[...])

    Kept at a 3-tuple return for backward compatibility: ``native_graph.py`` (out of scope for this
    telemetry pass) unpacks this directly. See :func:`_native_tool_aliases` for the fourth mapping
    the span work needs.
    """
    native_tool_map: dict[str, Any] = {}
    user_config_tools: dict[str, Any] = {}

    for ld_name, stub in tool_handlers.items():
        native = getattr(stub, NATIVE_TOOL_KEY, None)
        if isinstance(native, NativeTool):
            native_tool_map[native.tool_name] = stub
        elif config_tools and ld_name in config_tools:
            user_config_tools[ld_name] = config_tools[ld_name]

    return native_tool_map, user_config_tools, list(native_tool_map.keys())


def _native_tool_aliases(
    tool_handlers: dict[str, Callable[..., Any] | NativeTool],
) -> dict[str, str]:
    """AI Config key → provider tool name, for the config's native tools only.

    ``_partition_tools``'s ``native_tool_map`` is keyed the other way round, by provider name, which
    loses the AI Config key, and the key is what a config's declared schema is filed under.
    :class:`~.spans.ToolCatalog` needs both ends to report one tool once, under the name the model
    saw, with the schema the config gave it.
    """
    aliases: dict[str, str] = {}
    for ld_name, stub in tool_handlers.items():
        native = getattr(stub, NATIVE_TOOL_KEY, None)
        if isinstance(native, NativeTool):
            aliases[ld_name] = native.tool_name
    return aliases


def _is_coroutine(fn: Any) -> bool:
    return asyncio.iscoroutinefunction(fn)


def _build_hooks(native_tool_map: dict[str, Any]) -> dict[str, Any] | None:
    """The pre-span-work hook set: tracks a native tool call for telemetry purposes only.

    Kept for ``native_graph.py`` (out of scope for this telemetry pass), which imports this name
    directly and does not build ``execute_tool`` spans of its own. See :func:`build_tool_hooks` for
    the span-aware hook set this handler's own ``_call_impl``/``_stream_gen`` use.
    """
    if not native_tool_map:
        return None

    async def _pre_tool_hook(
        input_data: Any, tool_use_id: str | None, hook_context: Any
    ) -> dict[str, Any]:
        tool_name = (
            input_data.get("tool_name") if isinstance(input_data, dict) else None
        )
        stub = native_tool_map.get(tool_name) if tool_name is not None else None
        if stub and callable(stub):
            stub()
        return {}

    return {"PreToolUse": [HookMatcher(hooks=[_pre_tool_hook])]}  # type: ignore[list-item]


ToolTelemetry = Callable[[BaseException], None]
#: Ends every open tool span without failing it, for the abandonment path.
#: Takes the ended-tracker and whether the unwind was a cancellation rather than a consumer
#: stopping early, so a cancelled run marks launchdarkly.run.cancelled on its tool spans too.
AbandonTelemetry = Callable[..., None]
#: Ends every open tool span without failing it, for a task.cancel() the blocking path cannot
#: see coming. Distinct from AbandonTelemetry: that one marks a span abandoned mid-stream, this
#: one marks it launchdarkly.run.cancelled, via the same shared helper the root's own finally uses.
CancelTelemetry = Callable[[], None]


def build_tool_hooks(
    native_tool_map: dict[str, Any],
    parent_context: Any,
    capture_content: bool,
    *,
    emitter: TurnEmitter | None = None,
    pending: list[dict[str, Any]] | None = None,
) -> tuple[
    dict[str, list[HookMatcher]], ToolTelemetry, AbandonTelemetry, CancelTelemetry
]:
    """Builds the PreToolUse/PostToolUse/PostToolUseFailure hooks that open and close
    ``execute_tool`` spans around the Agent SDK's own tool dispatch.

    Returns ``(hooks, close_open_spans, abandon_open_spans, cancel_open_spans)``.
    ``close_open_spans`` fails every span this run still has open, for the path where the SDK
    throws mid-tool-call and no ``PostToolUse*`` hook ever fires. ``cancel_open_spans`` is the same
    idea for a ``task.cancel()``: nothing failed, so it ends each span still open through
    :func:`~launchdarkly_ai_server.end_unfinished_spans` instead of failing it.
    """
    tool_spans: dict[str, Any] = {}

    def _finish(
        tool_use_id: str | None, *, error: BaseException | None, result: Any = None
    ) -> None:
        if tool_use_id is None:
            return
        if emitter is not None and pending is not None:
            if error is None:
                pending.extend(emitter.tool_result(tool_use_id, result))
            else:
                pending.extend(emitter.tool_result(tool_use_id, failed=True))
        span = tool_spans.pop(tool_use_id, None)
        if span is None:
            return
        # Once popped, ending it is this function's job alone: nothing else knows the span exists. A
        # tool result comes from the caller's own function, so it can be anything, including something
        # json.dumps refuses.
        try:
            if result is not None:
                set_tool_call_content_attributes(span, capture_content, result=result)
        except Exception as exc:
            # `error or exc`, so a tool that already failed keeps the reason it failed. The
            # serialisation problem is this span's second-worst fact, not its first.
            fail_span(span, error or exc)
            raise
        if error is None:
            succeed_span(span)
        else:
            fail_span(span, error)

    async def _pre_tool_use(
        input_data: Any, tool_use_id: str | None, hook_context: Any
    ) -> dict[str, Any]:
        if input_data.get("hook_event_name") != "PreToolUse":
            return {}
        provider_name = input_data.get("tool_name", "")
        stub = native_tool_map.get(provider_name)
        if stub and callable(stub):
            stub()

        display_name = tool_display_name(provider_name)
        use_id = input_data.get("tool_use_id") or tool_use_id or ""
        span = start_tool_span(display_name, use_id, parent_context)
        session_id = input_data.get("session_id")
        # Same grouping key as the root and as the CLI's own spans; the hook input is where this
        # side sees it without waiting for a message. See TELEMETRY-CONTRACT.md section 4.
        if span is not None and session_id:
            set_conversation_id_if_absent(span, session_id)
        # Filed before the content write, not after. Serialising the arguments can raise, and a raise
        # out of an unfiled span leaves it open with nothing tracking it: close_open_spans and the
        # teardown both walk this dict, so a span missing from it is a span that never exports.
        tool_spans[use_id] = span
        if emitter is not None and pending is not None and use_id:
            pending.extend(emitter.tool_start(use_id, display_name))
            pending.extend(
                emitter.tool_end(use_id, arguments=input_data.get("tool_input"))
            )
        set_tool_call_content_attributes(
            span, capture_content, arguments=input_data.get("tool_input")
        )
        return {}

    async def _post_tool_use(
        input_data: Any, tool_use_id: str | None, hook_context: Any
    ) -> dict[str, Any]:
        if input_data.get("hook_event_name") != "PostToolUse":
            return {}
        use_id = input_data.get("tool_use_id") or tool_use_id
        _finish(use_id, error=None, result=input_data.get("tool_response"))
        return {}

    async def _post_tool_use_failure(
        input_data: Any, tool_use_id: str | None, hook_context: Any
    ) -> dict[str, Any]:
        if input_data.get("hook_event_name") != "PostToolUseFailure":
            return {}
        use_id = input_data.get("tool_use_id") or tool_use_id
        _finish(
            use_id,
            error=RuntimeError(str(input_data.get("error") or "tool call failed")),
        )
        return {}

    def close_open_spans(error: BaseException) -> None:
        for span in list(tool_spans.values()):
            fail_span(span, error)
        tool_spans.clear()

    def abandon_open_spans(ended: set[int], cancelled: bool = False) -> None:
        """Ends every tool span still open, for stream abandonment.

        Unlike :func:`close_open_spans`, nothing failed: a consumer stopping early is normal, so
        each span is left UNSET and marked ``launchdarkly.stream.abandoned``. Matches what the
        openai-agents and langchain-agents handlers do on the same path, so one abandoned run does
        not read as an error in one SDK and a clean stop in another.
        """
        for span in list(tool_spans.values()):
            end_span_once(span, ended, abandoned=True, cancelled=cancelled)
        tool_spans.clear()

    def cancel_open_spans() -> None:
        """Ends every tool span still open, for the blocking path's unwind on a task.cancel().

        ``asyncio.CancelledError`` is a ``BaseException``, so it walks past ``close_open_spans``'s
        caller too. Nothing failed here either: the caller went away, the same reasoning
        :func:`abandon_open_spans` applies, using the shared helper so a cancelled run's tool spans
        agree with its root about ``launchdarkly.run.cancelled``.
        """
        end_unfinished_spans(*tool_spans.values())
        tool_spans.clear()

    hooks: dict[str, list[HookMatcher]] = {
        "PreToolUse": [HookMatcher(hooks=[_pre_tool_use])],  # type: ignore[list-item]
        "PostToolUse": [HookMatcher(hooks=[_post_tool_use])],  # type: ignore[list-item]
        "PostToolUseFailure": [HookMatcher(hooks=[_post_tool_use_failure])],  # type: ignore[list-item]
    }
    return hooks, close_open_spans, abandon_open_spans, cancel_open_spans


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_prompt(
    config: AiConfigRep,
    user_input: str | None,
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> tuple[str, str | None]:
    """Returns (prompt, system_prompt).

    One user message, not one per configured role: ``query()`` takes a single prompt string, so this
    really does flatten a configured history into one turn before the model sees it.
    """
    safe_input = user_input or ""
    system_prompt: str | None = None

    if config.get("instructions"):
        system_prompt = parse_template(config["instructions"], variables)
    elif config.get("messages"):
        system_msgs = [m for m in config["messages"] if m.get("role") == "system"]
        non_system = [m for m in config["messages"] if m.get("role") != "system"]
        system_prompt = (
            parse_template("\n".join(m["content"] for m in system_msgs), variables)
            if system_msgs
            else None
        )
        config_history = "\n".join(
            parse_template(m["content"], variables) for m in non_system
        )
        safe_input = (
            f"{config_history}\n\n{safe_input}" if config_history else safe_input
        )

    return safe_input, system_prompt


def _parse_message_content(content: Any, variables: dict[str, Any]) -> Any:
    return parse_template(content, variables) if isinstance(content, str) else content


def _config_conversation_turns(
    config: AiConfigRep, variables: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "role": message.get("role"),
            "content": _parse_message_content(message.get("content", ""), variables),
        }
        for message in (config.get("messages") or [])
        if message.get("role") != "system"
    ]


def _to_anthropic_user_content(content: Any) -> Any:
    if isinstance(content, str):
        return content

    blocks: list[dict[str, Any]] = []
    for block in content:
        if block.get("type") == "text":
            blocks.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image":
            source = block.get("source", {})
            if source.get("type") == "url":
                mapped_source = {"type": "url", "url": source.get("url", "")}
            else:
                mapped_source = {
                    "type": "base64",
                    "media_type": source.get("media_type", ""),
                    "data": source.get("data", ""),
                }
            blocks.append({"type": "image", "source": mapped_source})
    return blocks


async def _to_streamed_prompt(
    turns: list[dict[str, Any]],
) -> AsyncGenerator[dict[str, Any], None]:
    # The envelope ``type`` has to agree with the message role. The CLI reading this stream
    # accepts an "assistant" envelope as a replayed turn, but every other envelope type is
    # required to carry role "user" — an assistant turn sent as ``type: "user"`` is rejected
    # outright with "Expected message role 'user', got 'assistant'".
    for turn in turns:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "assistant":
            yield {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": content_to_text(content)}],
                },
                "parent_tool_use_id": None,
            }
        else:
            yield {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": _to_anthropic_user_content(content),
                },
                "parent_tool_use_id": None,
            }


def build_query_prompt(
    config: AiConfigRep,
    user_input: str | None,
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None,
    fallback_prompt: str,
) -> str | AsyncGenerator[dict[str, Any], None]:
    if not history:
        return fallback_prompt

    turns = compose_history(
        history=history,
        user_input=user_input,
        config_messages=(
            []
            if config.get("instructions")
            else _config_conversation_turns(config, variables)
        ),
    )
    return _to_streamed_prompt(turns)


def _opening_of(prompt: str, system_prompt: str | None) -> Opening:
    return Opening(
        system_instructions=system_prompt,
        messages=[
            SpanMessage(
                role="user", parts=[SpanMessagePart(type="text", content=prompt)]
            )
        ],
    )


def _raise_if_run_failed(message: Any) -> None:
    """Raises when a result message reports a failed run.

    Two shapes fail. A non-``success`` subtype carries no ``result`` field, so a run that hit
    ``error_max_turns`` or ``error_max_budget_usd`` genuinely failed and has to surface as a failure
    rather than being reported OK with zeroed usage.

    The other is a ``success`` subtype with ``is_error`` set, which is how the CLI reports a failed
    Anthropic API call. Measured against Agent SDK 0.3.220 with a key that had no credits: the run
    ends ``success``, ``is_error`` true, ``api_error_status`` set, and the error text in ``result`` —
    so a caller was handed "Credit balance is too low" as the model's answer.
    """
    if message.subtype != "success":
        detail = f": {'; '.join(message.errors)}" if message.errors else ""
        raise RuntimeError(f"Claude agent run ended with {message.subtype}{detail}")
    if message.is_error is not True:
        return
    # ``getattr``: the field arrived with CLI v2.1.110, and older SDKs' dataclass lacks it.
    status = getattr(message, "api_error_status", None)
    status_text = f" (HTTP {status})" if isinstance(status, int) else ""
    detail = f": {message.result}" if message.result else ""
    raise RuntimeError(
        f"Claude agent run ended with an error result{status_text}{detail}"
    )


#: The ``thinking`` types ``claude_agent_sdk`` knows (``ThinkingConfig``). It reads
#: ``thinking["type"]`` when it builds the CLI command, so a ``thinking`` without one raises.
_THINKING_TYPES = frozenset({"adaptive", "enabled", "disabled"})

#: The ``thinking.display`` values the SDK passes to the CLI (``ThinkingDisplay``).
_THINKING_DISPLAYS = frozenset({"summarized", "omitted"})


def _rebuild_thinking(value: Mapping[str, Any]) -> dict[str, Any] | None:
    """``thinking`` in the shape ``claude_agent_sdk`` reads, or ``None`` when it is malformed.

    Keeps ``type``, ``budget_tokens`` and ``display`` only, the same sub-keys the JS SDK keeps.
    The type must be one the SDK knows, and an ``enabled`` thinking must carry an integer
    ``budget_tokens``: the SDK reads ``thinking["budget_tokens"]`` for it straight away, so one
    without a budget would raise. ``budgetTokens`` is read as well, as in the JS SDK. ``display``
    is kept only for a type that takes it and only with a value the SDK knows.
    """
    kind = value.get("type")
    if kind not in _THINKING_TYPES:
        return None
    thinking: dict[str, Any] = {"type": kind}
    if kind == "enabled":
        budget = value.get("budget_tokens", value.get("budgetTokens"))
        if not isinstance(budget, int) or isinstance(budget, bool):
            return None
        thinking["budget_tokens"] = budget
    if kind != "disabled" and value.get("display") in _THINKING_DISPLAYS:
        thinking["display"] = value["display"]
    return thinking


def _options_parameters(config: AiConfigRep) -> dict[str, Any]:
    """The config's ``model.parameters`` that become ``ClaudeAgentOptions`` fields: only
    :data:`_CLAUDE_AGENT_OPTIONS_FORWARDED_KEYS`, with malformed object values dropped and
    ``thinking`` rebuilt by :func:`_rebuild_thinking`. The handler and the native graph both call
    this, so they forward the same keys.
    """
    params = select_forwarded_parameters(
        model_parameters(config),
        _CLAUDE_AGENT_OPTIONS_FORWARDED_KEYS,
        mapping_keys=_CLAUDE_AGENT_OPTIONS_MAPPING_KEYS,
    )
    if "thinking" in params:
        thinking = _rebuild_thinking(params.pop("thinking"))
        if thinking is not None:
            params["thinking"] = thinking
    return params


def _build_query_options(
    config: AiConfigRep,
    system_prompt: str | None,
    native_tool_names: list[str],
    mcp_allowed_tools: list[str],
    tool_mcp: Any,
    hooks: dict[str, list[HookMatcher]] | None,
    extra_builtins: list[str] | None = None,
    **extra: Any,
) -> ClaudeAgentOptions:
    """*extra_builtins* are built-ins the handler itself turns on, after the config's native tools,
    because a per-invoke preparation asked for them (``Skill``, for example). They go in ``tools``
    only, never ``allowed_tools``: a built-in the Agent SDK enables through its own option needs no
    permission entry here.
    """
    all_allowed = [*mcp_allowed_tools, *native_tool_names]
    params = _options_parameters(config)
    kwargs: dict[str, Any] = {
        **params,
        "model": config["model"]["name"],
        "allowed_tools": all_allowed if all_allowed else [],
        "mcp_servers": {TOOL_MCP_NAME: tool_mcp} if tool_mcp else {},
        "hooks": hooks or {},
        # Always sent, `[]` when the config attaches no NativeTool. The empty list switches every
        # Claude Code built-in off; omitting the key leaves the SDK default, which gives every run
        # the full built-in set, and its read-only tools need no permission. A config turns a
        # built-in on by attaching its sentinel. User-defined tools go through the MCP server and
        # are not governed by `tools`.
        "tools": [*native_tool_names, *(extra_builtins or [])],
        **extra,
    }
    if system_prompt:
        kwargs["system_prompt"] = system_prompt
    return ClaudeAgentOptions(**kwargs)


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryPreparation:
    """What a per-invoke preparation adds to one run.

    Package-internal and generic: it names no feature, so code at a later stage can register
    through it without this module importing or naming that code.
    """

    options: dict[str, Any]
    """Extra ``query()`` options, set with the handler's own host options."""
    extra_builtins: list[str]
    """Built-in tool names appended to the config's native tools."""
    cleanup: Callable[[], Awaitable[None]]
    """Runs once when the run ends, however it ends. Must not raise."""


PrepareQuery = Callable[[AiConfigRep], Awaitable[QueryPreparation | None]]
"""Called at the start of every invoke and stream, inside the run's span, so a raise fails the run
like any other error. ``None`` means nothing to add, and the run is exactly the core handler's."""


def _host_query_options(
    cwd: str | os.PathLike[str] | None,
    setting_sources: list[SettingSource] | None,
) -> dict[str, Any]:
    """The host options for ``query()``: ``setting_sources`` always, ``[]`` unless given; ``cwd``
    only if given.

    The Agent SDK reads ``setting_sources=None`` as "load every source", which put the host's
    plugins, skills, hooks, permission rules and ``CLAUDE.md`` into every run (AIC-3584).
    """
    options: dict[str, Any] = {
        "setting_sources": list(setting_sources) if setting_sources is not None else []
    }
    if cwd is not None:
        options["cwd"] = cwd
    return options


def create_claude_agents_handler(
    *,
    capture_content: bool | None = None,
    cwd: str | os.PathLike[str] | None = None,
    setting_sources: list[SettingSource] | None = None,
) -> ProviderHandler:
    """Creates a ``ProviderHandler`` for Anthropic's Claude via the claude-agent-sdk.

    Set *capture_content* to put prompts, model output, tool arguments and tool results on the
    emitted spans. ``None`` (the default) reads the process-wide default at invocation,
    which is off until ``configure`` sets it. ``True`` or ``False`` overrides that default
    for this handler. See TELEMETRY-CONTRACT.md section 7.

    *cwd* and *setting_sources* configure the Claude Code process the Agent SDK starts, not the
    model, so they come from the application and never from a config. Each one given is passed to
    every ``query()`` call unchanged. *cwd* not given is not sent, which leaves the process's
    working directory.

    Host settings load only when the application opts in. *setting_sources* not given sends
    ``[]``, because the Agent SDK's own default loads every source (``user``, ``project``,
    ``local``), including that host's plugins, skills, hooks, permission rules and ``CLAUDE.md``.
    Pass the sources to load them, e.g. ``["user", "project", "local"]`` for all. The Agent SDK
    still reads the managed-settings policy tier, which no option turns off.
    """
    report_usage("claude-agents.createClaudeAgentsHandler", PACKAGE_NAME, __version__)
    return _create_claude_agents_handler(
        capture_content=capture_content, cwd=cwd, setting_sources=setting_sources
    )


def _create_claude_agents_handler(
    *,
    capture_content: bool | None = None,
    cwd: str | os.PathLike[str] | None = None,
    setting_sources: list[SettingSource] | None = None,
    _prepare: PrepareQuery | None = None,
) -> ProviderHandler:
    """Non-reporting :func:`create_claude_agents_handler`, used by this package's wrappers.

    *_prepare* is a package-internal hook, never an option of the public factory: see
    :data:`PrepareQuery`.
    """
    content_override = capture_content
    # Resolved once, so a caller mutating its list afterwards does not change later runs.
    host_options = _host_query_options(cwd, setting_sources)

    async def _call_impl(
        config: AiConfigRep,
        user_input: str = "",
        tool_handlers: dict[str, Callable[..., Any] | NativeTool] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        # None follows configure(); an explicit bool overrides it. Read when the call runs.
        capture_content = effective_capture_content(content_override)
        th = tool_handlers or {}
        vs = variables or {}

        span = start_root_span(config, vs)
        parent = parent_context_of(span)
        # Cleared by whichever path ends the root, so the finally below can tell an open root
        # from a closed one without asking the span. A mock span answers `is_recording()`
        # truthily, and this handler's own suite is built on mock spans in places.
        open_root_span: Any = span

        prompt, system_prompt = _build_prompt(config, user_input, vs, history)
        query_prompt = build_query_prompt(config, user_input, vs, history, prompt)
        if config.get("outputFormat"):
            schema_instr = f"Respond with valid JSON matching this schema:\n{json.dumps(config['outputFormat'])}"
            system_prompt = (
                f"{system_prompt}\n\n{schema_instr}" if system_prompt else schema_instr
            )
        opening = _opening_of(prompt, system_prompt)

        native_tool_map, user_config_tools, native_tool_names = _partition_tools(
            config.get("tools"), th
        )
        catalog = ToolCatalog(config.get("tools"), _native_tool_aliases(th))

        # Declared out here so the except clause can end a chat span the throw left open.
        inference = InferenceSpans(config, parent, capture_content, catalog, opening)
        tool_telemetry: ToolTelemetry | None = None
        cancel_tool_spans: CancelTelemetry | None = None
        # Set wherever the root's usage is written, so the failure path can tell whether the CLI
        # already reported an authoritative run-level total and must not overwrite it.
        root_usage_written = False
        gen: AsyncIterator[Any] | None = None
        prepared: QueryPreparation | None = None
        try:
            # Inside the guard, because serialising the prompt raises on anything that is not
            # JSON-serialisable and a raise out here would leave the root open: never ended, never
            # exported, and the run gone from AI Config Monitoring with the feature_flag event on it.
            set_input_content_attributes(
                span,
                capture_content,
                system_instructions=opening.system_instructions,
                messages=opening.messages,
                tool_definitions=catalog.current,
            )
            prepared = await _prepare(config) if _prepare is not None else None
            extra_builtins = prepared.extra_builtins if prepared is not None else []
            tool_mcp = (
                await _build_tool_mcp(user_config_tools, th)
                if user_config_tools
                else None
            )
            mcp_allowed_tools = [MCP_TOOL_PREFIX + n for n in user_config_tools]
            hooks = None
            if native_tool_names or mcp_allowed_tools or extra_builtins:
                # The blocking path cannot be abandoned, so it takes no abandon hook, but it can
                # still be cancelled, so it keeps cancel_open_spans.
                hooks, tool_telemetry, _, cancel_tool_spans = build_tool_hooks(
                    native_tool_map, parent, capture_content
                )

            options = _build_query_options(
                config,
                system_prompt,
                native_tool_names,
                mcp_allowed_tools,
                tool_mcp,
                hooks,
                extra_builtins,
                # Merged first, so a preparation's key replaces a host option's rather than
                # colliding with it.
                **{
                    **host_options,
                    **(prepared.options if prepared is not None else {}),
                },
            )

            output = ""

            # Held in a variable so the finally below can aclose() it. A bare `return` inside
            # `async for` abandons the generator, and asyncio's finalizer then raises RuntimeError
            # when the generator is suspended inside a real await in the SDK.
            gen = query(prompt=query_prompt, options=options)
            try:
                async for message in gen:
                    record_conversation_id(span, message)
                    record_native_tools(span, message, capture_content, catalog)
                    inference.record(message)

                    if isinstance(message, ResultMessage):
                        # Before the root, so the children it parents are already closed.
                        inference.finish()
                        raw_usage: dict[str, Any] = dict(message.usage or {})
                        finish_root_span(span, config, raw_usage)
                        # The CLI's own run-level total is authoritative, so the except clause below
                        # must not overwrite it with the summed-per-response figure.
                        root_usage_written = True
                        _raise_if_run_failed(message)
                        result_text = (
                            message.result
                            if isinstance(message.result, str)
                            else json.dumps(message.result)
                        )
                        set_output_content_attributes(
                            span,
                            capture_content,
                            [
                                SpanMessage(
                                    role="assistant",
                                    parts=[
                                        SpanMessagePart(
                                            type="text", content=result_text
                                        )
                                    ],
                                )
                            ],
                        )
                        succeed_span(span)
                        open_root_span = None
                        return {"output": message.result, "usage": raw_usage}

                # The stream ended without a result message, so no message closed the last
                # response, and no message carried a run-level total either. The per-response sum
                # is the only record of what the run spent.
                inference.finish()
                streamed_usage = inference.run_usage
                # Only when something reported. All-zero attributes claim the run cost nothing, and a
                # stream that ended without a result message and without absorbing a single turn
                # cannot make that claim: absent usage means unknown, which is the honest answer. The
                # error and abandonment paths already guard the same way.
                if streamed_usage["reported"]:
                    finish_root_span(span, config, streamed_usage["total"])
                    root_usage_written = True
                succeed_span(span)
                open_root_span = None
                return {"output": output, "usage": streamed_usage["total"]}
            finally:
                if gen is not None:
                    await gen.aclose()  # type: ignore[attr-defined]

        except Exception as exc:
            # Before the root: the exporter only receives ended spans, so a chat span left open by
            # the throw would never reach the trace.
            inference.finish()
            if tool_telemetry is not None:
                tool_telemetry(exc)
            if not root_usage_written and inference.run_usage["reported"]:
                finish_root_span(span, config, inference.run_usage["total"])
            fail_span(span, exc)
            open_root_span = None
            raise
        finally:
            # Not an except: asyncio.CancelledError is a BaseException, so a timeout or a
            # task.cancel() never reaches the clause above. Without this the root is stranded, and
            # the root is the only span carrying the feature_flag event and the launchdarkly.*
            # attributes, so the whole run would vanish from AI Config Monitoring rather than show
            # as incomplete.
            try:
                inference.finish()
                if cancel_tool_spans is not None:
                    cancel_tool_spans()
                if (
                    open_root_span is not None
                    and not root_usage_written
                    and inference.run_usage["reported"]
                ):
                    # The turns that completed were billed, the same reason the failure path
                    # reports them.
                    finish_root_span(
                        open_root_span, config, inference.run_usage["total"]
                    )
                end_unfinished_spans(open_root_span)
            finally:
                # Last, after the span tree is closed, and in its own finally so a raise above
                # still removes the preparation's temp dir.
                if prepared is not None:
                    await prepared.cleanup()

    def _stream_impl(
        config: AiConfigRep,
        user_input: str = "",
        tool_handlers: dict[str, Callable[..., Any] | NativeTool] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        return _stream_gen(
            config,
            user_input,
            tool_handlers or {},
            variables or {},
            history,
            capture_content=effective_capture_content(content_override),
            host_options=host_options,
            prepare=_prepare,
        )

    return _create_handler(
        ("Anthropic", "agent"),
        _call_impl,  # type: ignore[arg-type]
        _stream_impl,  # type: ignore[arg-type]
        capture_content=content_override,
    )


def _map_claude_stream_event(
    event: dict[str, Any],
    emitter: TurnEmitter,
    block_ids: dict[int, str],
    saw_args: set[str],
) -> list[dict[str, Any]]:
    """Map one Claude Agent SDK stream event onto the AG-UI subset."""
    kind = event.get("type")
    if kind == "content_block_start":
        block = event.get("content_block") or {}
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            return []
        index = event.get("index")
        call_id = str(block.get("id") or "")
        if isinstance(index, int):
            block_ids[index] = call_id
        return emitter.tool_start(
            call_id, tool_display_name(str(block.get("name") or ""))
        )
    if kind == "content_block_delta":
        delta = event.get("delta") or {}
        if not isinstance(delta, dict):
            return []
        if delta.get("type") == "text_delta":
            text = delta.get("text") or ""
            if not isinstance(text, str) or not text:
                return []
            return emitter.text(text)
        if delta.get("type") == "input_json_delta":
            index = event.get("index")
            call_id = block_ids.get(index, "") if isinstance(index, int) else ""
            partial = delta.get("partial_json") or ""
            if isinstance(partial, str) and partial and call_id:
                saw_args.add(call_id)
            return emitter.tool_args(
                call_id, partial if isinstance(partial, str) else ""
            )
    if kind == "content_block_stop":
        index = event.get("index")
        call_id = block_ids.get(index, "") if isinstance(index, int) else ""
        if call_id and call_id in saw_args:
            return emitter.tool_end(call_id)
    return []


async def _stream_gen(
    config: AiConfigRep,
    user_input: str,
    tool_handlers: dict[str, Callable[..., Any] | NativeTool],
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    *,
    capture_content: bool = False,
    host_options: dict[str, Any] | None = None,
    prepare: PrepareQuery | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Streams the run, emitting the same span tree as the blocking path.

    A consumer that breaks out of ``async for``, or raises inside the loop body, makes this
    generator run its ``finally`` without ever entering ``except``: ``GeneratorExit`` inherits from
    ``BaseException``, so ``except Exception`` does not see it. The same ``finally`` also closes the
    vendor's own generator, for the same reason the blocking path holds one in a variable: a bare
    exit abandons it, and asyncio's finalizer later raises ``RuntimeError`` when it is suspended
    inside a real await in the SDK.
    """
    span = start_root_span(config, variables)
    parent = parent_context_of(span)

    prompt, system_prompt = _build_prompt(config, user_input, variables, history)
    query_prompt = build_query_prompt(config, user_input, variables, history, prompt)
    opening = _opening_of(prompt, system_prompt)

    native_tool_map, user_config_tools, native_tool_names = _partition_tools(
        config.get("tools"), tool_handlers
    )
    catalog = ToolCatalog(config.get("tools"), _native_tool_aliases(tool_handlers))

    inference = InferenceSpans(config, parent, capture_content, catalog, opening)
    tool_telemetry: ToolTelemetry | None = None
    abandon_tool_spans: AbandonTelemetry | None = None
    root_usage_written = False
    ended: set[int] = set()
    gen: AsyncIterator[Any] | None = None

    # Distinguishes the two teardown reasons. A consumer that stops reading abandoned the
    # stream; a CancelledError means something cancelled the run, usually a timeout, and the
    # consumer chose nothing. The blocking path already tells these apart.
    cancelled = False
    emitter = TurnEmitter(capture_content=capture_content)
    pending: list[dict[str, Any]] = []
    block_ids: dict[int, str] = {}
    saw_args: set[str] = set()
    prepared: QueryPreparation | None = None
    try:
        # Inside the guard, because serialising the prompt raises on anything that is not
        # JSON-serialisable. A raise out here would leave the root open with the `finally` never
        # entered, so the run would vanish from AI Config Monitoring with its feature_flag event.
        set_input_content_attributes(
            span,
            capture_content,
            system_instructions=opening.system_instructions,
            messages=opening.messages,
            tool_definitions=catalog.current,
        )
        prepared = await prepare(config) if prepare is not None else None
        extra_builtins = prepared.extra_builtins if prepared is not None else []
        tool_mcp = (
            await _build_tool_mcp(user_config_tools, tool_handlers)
            if user_config_tools
            else None
        )
        mcp_allowed_tools = [MCP_TOOL_PREFIX + n for n in user_config_tools]
        hooks = None
        if native_tool_names or mcp_allowed_tools or extra_builtins:
            hooks, tool_telemetry, abandon_tool_spans, _ = build_tool_hooks(
                native_tool_map,
                parent,
                capture_content,
                emitter=emitter,
                pending=pending,
            )

        options = _build_query_options(
            config,
            system_prompt,
            native_tool_names,
            mcp_allowed_tools,
            tool_mcp,
            hooks,
            extra_builtins,
            **{
                **(host_options or _host_query_options(None, None)),
                **(prepared.options if prepared is not None else {}),
            },
            include_partial_messages=True,
        )

        full_output = ""
        gen = query(prompt=query_prompt, options=options)
        async for message in gen:
            while pending:
                yield pending.pop(0)
            record_conversation_id(span, message)
            record_native_tools(span, message, capture_content, catalog)
            inference.record(message)

            if isinstance(message, StreamEvent):
                event = message.event
                if event.get("type") == "message_start":
                    for rich in emitter.end_text():
                        yield rich
                for rich in _map_claude_stream_event(
                    event, emitter, block_ids, saw_args
                ):
                    if rich.get("type") == "TEXT_MESSAGE_CONTENT":
                        full_output += rich.get("delta") or ""
                    yield rich
            elif isinstance(message, UserMessage):
                blocks = message.content if isinstance(message.content, list) else []
                for block in blocks:
                    if not isinstance(block, ToolResultBlock):
                        continue
                    # Built-in tools have no hook when the config lists none.
                    # The SDK still yields the result, and that is what closes
                    # the call instead of fail_open marking it failed.
                    for rich in emitter.tool_result(
                        str(block.tool_use_id),
                        block.content,
                        failed=bool(getattr(block, "is_error", False)),
                    ):
                        yield rich
            elif isinstance(message, ResultMessage):
                # Before the root, so the children it parents are already closed.
                inference.finish()
                raw_usage: dict[str, Any] = dict(message.usage or {})
                finish_root_span(span, config, raw_usage)
                root_usage_written = True
                _raise_if_run_failed(message)
                final_output = (
                    message.result if message.result is not None else full_output
                )
                result_text = (
                    final_output
                    if isinstance(final_output, str)
                    else json.dumps(final_output)
                )
                set_output_content_attributes(
                    span,
                    capture_content,
                    [
                        SpanMessage(
                            role="assistant",
                            parts=[SpanMessagePart(type="text", content=result_text)],
                        )
                    ],
                )
                mark_ok(span)
                end_span_once(span, ended)
                while pending:
                    yield pending.pop(0)
                for rich in emitter.end_text():
                    yield rich
                for rich in emitter.fail_open():
                    yield rich
                yield {"type": "done", "output": final_output, "usage": raw_usage}
                return

        # The stream ended without a result message, so nothing carried a run-level total. The
        # per-response sum is the only record of the spend.
        inference.finish()
        streamed_usage = inference.run_usage
        # Same guard as the blocking path: zeros would assert the run cost nothing.
        if streamed_usage["reported"]:
            finish_root_span(span, config, streamed_usage["total"])
            root_usage_written = True
        set_output_content_attributes(
            span,
            capture_content,
            [
                SpanMessage(
                    role="assistant",
                    parts=[SpanMessagePart(type="text", content=full_output)],
                )
            ],
        )
        mark_ok(span)
        end_span_once(span, ended)
        while pending:
            yield pending.pop(0)
        for rich in emitter.end_text():
            yield rich
        for rich in emitter.fail_open():
            yield rich
        yield {"type": "done", "output": full_output, "usage": streamed_usage["total"]}

    except asyncio.CancelledError:
        cancelled = True
        raise
    except Exception as exc:
        if tool_telemetry is not None:
            tool_telemetry(exc)
        if not root_usage_written and inference.run_usage["reported"]:
            finish_root_span(span, config, inference.run_usage["total"])
            root_usage_written = True
        fail_span(span, exc, ended)
        while pending:
            yield pending.pop(0)
        for rich in emitter.fail_open():
            yield rich
        raise
    finally:
        try:
            # A no-op on the success path, where the result message already closed the last
            # response. On the error and abandonment paths this is the only thing that ends a chat
            # span still open, and the exporter only receives ended spans.
            inference.finish()
            # A no-op on the success and failure paths; on abandonment it is the only chance to
            # close the tree, including any tool span whose PostToolUse hook never fired, and the
            # only chance to report what the responses that did arrive cost.
            if id(span) not in ended:
                # Abandonment, not failure: UNSET plus the abandoned marker, the same as the model
                # span and the root get just below.
                if abandon_tool_spans is not None:
                    abandon_tool_spans(ended, cancelled=cancelled)
                if not root_usage_written and inference.run_usage["reported"]:
                    finish_root_span(span, config, inference.run_usage["total"])
            end_span_once(span, ended, abandoned=True, cancelled=cancelled)
        finally:
            # In their own finally, so a raise closing the spans still closes the vendor's
            # generator and removes the preparation's temp dir. Same reasoning as the blocking
            # path: a bare exit through this generator's boundary abandons the vendor's own
            # generator if it is not closed explicitly.
            try:
                if gen is not None:
                    await gen.aclose()  # type: ignore[attr-defined]
            finally:
                # Last, after the span tree is closed: the run is over however it ended.
                if prepared is not None:
                    await prepared.cleanup()


def claude_agents(
    config_key: str,
    user_input: str,
    context: LDContext,
    *,
    cwd: str | os.PathLike[str] | None = None,
    setting_sources: list[SettingSource] | None = None,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper: creates a handler and calls config(...).invoke().

    *cwd* and *setting_sources* build the handler, as on :func:`create_claude_agents_handler`,
    with the same defaults, and are not passed on to ``config()``.
    """
    report_usage("claude-agents.claudeAgents", PACKAGE_NAME, __version__)
    variables = kwargs.pop("variables", None)
    capture_content = kwargs.pop("capture_content", None)
    return _config(
        key=config_key,
        handler=_create_claude_agents_handler(
            capture_content=capture_content,
            cwd=cwd,
            setting_sources=setting_sources,
        ),
        **kwargs,
    ).invoke(user_input, context, variables=variables)
