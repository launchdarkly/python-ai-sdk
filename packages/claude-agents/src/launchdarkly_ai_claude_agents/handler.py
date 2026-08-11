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
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    HookMatcher,
    ResultMessage,
    StreamEvent,
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
    config,
    create_handler,
    end_span_once,
    parse_template,
    set_input_content_attributes,
    set_output_content_attributes,
    set_tool_call_content_attributes,
)

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

# ---------------------------------------------------------------------------
# Tool wiring
# ---------------------------------------------------------------------------


async def build_tool_mcp(
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
            return {"content": [{"type": "text", "text": str(result)}]}

        mcp_tool = tool_fn(
            tool_name,
            tool_cfg.get("description", ""),
            tool_cfg.get("parameters") or {},
        )(_execute)
        tool_objs.append(mcp_tool)

    return create_server(name=TOOL_MCP_NAME, version="1.0.0", tools=tool_objs)


def partition_tools(
    config_tools: dict[str, Any] | None,
    tool_handlers: dict[str, Any],
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


def _native_tool_aliases(tool_handlers: dict[str, Any]) -> dict[str, str]:
    """AI Config key → provider tool name, for the config's native tools only.

    ``partition_tools``'s ``native_tool_map`` is keyed the other way round, by provider name, which
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


def build_tool_hooks(
    native_tool_map: dict[str, Any],
    parent_context: Any,
    capture_content: bool,
) -> tuple[dict[str, list[HookMatcher]], ToolTelemetry]:
    """Builds the PreToolUse/PostToolUse/PostToolUseFailure hooks that open and close
    ``execute_tool`` spans around the Agent SDK's own tool dispatch.

    Returns ``(hooks, close_open_spans)``. ``close_open_spans`` fails every span this run still has
    open, for the path where the SDK throws mid-tool-call and no ``PostToolUse*`` hook ever fires.
    """
    tool_spans: dict[str, Any] = {}

    def _finish(
        tool_use_id: str | None, *, error: BaseException | None, result: Any = None
    ) -> None:
        if tool_use_id is None:
            return
        span = tool_spans.pop(tool_use_id, None)
        if span is None:
            return
        if result is not None:
            set_tool_call_content_attributes(span, capture_content, result=result)
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
            span.set_attribute("gen_ai.conversation.id", session_id)
        set_tool_call_content_attributes(
            span, capture_content, arguments=input_data.get("tool_input")
        )
        tool_spans[use_id] = span
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

    hooks: dict[str, list[HookMatcher]] = {
        "PreToolUse": [HookMatcher(hooks=[_pre_tool_use])],  # type: ignore[list-item]
        "PostToolUse": [HookMatcher(hooks=[_post_tool_use])],  # type: ignore[list-item]
        "PostToolUseFailure": [HookMatcher(hooks=[_post_tool_use_failure])],  # type: ignore[list-item]
    }
    return hooks, close_open_spans


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _format_history(history: list[dict[str, Any]] | None) -> str | None:
    if not history:
        return None
    lines = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "Conversation History:\n\n" + "\n".join(lines)


def build_prompt(
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

    history_text = _format_history(history)
    if history_text:
        system_prompt = (
            f"{system_prompt}\n\n{history_text}" if system_prompt else history_text
        )

    return safe_input, system_prompt


def _opening_of(prompt: str, system_prompt: str | None) -> Opening:
    return Opening(
        system_instructions=system_prompt,
        messages=[
            SpanMessage(
                role="user", parts=[SpanMessagePart(type="text", content=prompt)]
            )
        ],
    )


def _result_error(subtype: str, errors: list[str] | None) -> str:
    """Builds the error for a non-success result message.

    ``SDKResultError`` carries no ``result`` field, so a run that hit ``error_max_turns`` or
    ``error_max_budget_usd`` genuinely failed and has to surface as a failure rather than being
    reported OK with zeroed usage.
    """
    detail = f": {'; '.join(errors)}" if errors else ""
    return f"Claude agent run ended with {subtype}{detail}"


def _build_query_options(
    config: AiConfigRep,
    system_prompt: str | None,
    native_tool_names: list[str],
    mcp_allowed_tools: list[str],
    tool_mcp: Any,
    hooks: dict[str, list[HookMatcher]] | None,
    **extra: Any,
) -> ClaudeAgentOptions:
    all_allowed = [*mcp_allowed_tools, *native_tool_names]
    kwargs: dict[str, Any] = {
        "model": config["model"]["name"],
        "allowed_tools": all_allowed if all_allowed else [],
        "mcp_servers": {TOOL_MCP_NAME: tool_mcp} if tool_mcp else {},
        "hooks": hooks or {},
        **extra,
    }
    # Omitted rather than passed empty. `tools=[]` is an explicit "no tools", which switches off the
    # Claude Code built-ins; leaving the key out keeps the SDK default, which is what a run with only
    # MCP tools, or none, has always had. A config with no native tools would otherwise silently lose
    # Read, Bash and the rest.
    if native_tool_names:
        kwargs["tools"] = native_tool_names
    if system_prompt:
        kwargs["system_prompt"] = system_prompt
    return ClaudeAgentOptions(**kwargs)


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------


def create_claude_agents_handler(*, capture_content: bool = False) -> ProviderHandler:
    """Creates a ``ProviderHandler`` for Anthropic's Claude via the claude-agent-sdk.

    Set *capture_content* to put prompts, model output, tool arguments and tool results on the
    emitted spans. It defaults to off. See TELEMETRY-CONTRACT.md section 7.
    """

    async def _call_impl(
        config: AiConfigRep,
        user_input: str = "",
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        th = tool_handlers or {}
        vs = variables or {}

        span = start_root_span(config, vs)
        parent = parent_context_of(span)

        prompt, system_prompt = build_prompt(config, user_input, vs, history)
        if config.get("outputFormat"):
            schema_instr = f"Respond with valid JSON matching this schema:\n{json.dumps(config['outputFormat'])}"
            system_prompt = (
                f"{system_prompt}\n\n{schema_instr}" if system_prompt else schema_instr
            )
        opening = _opening_of(prompt, system_prompt)

        native_tool_map, user_config_tools, native_tool_names = partition_tools(
            config.get("tools"), th
        )
        catalog = ToolCatalog(config.get("tools"), _native_tool_aliases(th))
        set_input_content_attributes(
            span,
            capture_content,
            system_instructions=opening.system_instructions,
            messages=opening.messages,
            tool_definitions=catalog.current,
        )

        # Declared out here so the except clause can end a chat span the throw left open.
        inference = InferenceSpans(config, parent, capture_content, catalog, opening)
        tool_telemetry: ToolTelemetry | None = None
        # Set wherever the root's usage is written, so the failure path can tell whether the CLI
        # already reported an authoritative run-level total and must not overwrite it.
        root_usage_written = False
        gen: AsyncIterator[Any] | None = None
        try:
            tool_mcp = (
                await build_tool_mcp(user_config_tools, th)
                if user_config_tools
                else None
            )
            mcp_allowed_tools = [MCP_TOOL_PREFIX + n for n in user_config_tools]
            hooks = None
            if native_tool_names or mcp_allowed_tools:
                hooks, tool_telemetry = build_tool_hooks(
                    native_tool_map, parent, capture_content
                )

            options = _build_query_options(
                config,
                system_prompt,
                native_tool_names,
                mcp_allowed_tools,
                tool_mcp,
                hooks,
            )

            output = ""

            # Held in a variable so the finally below can aclose() it. A bare `return` inside
            # `async for` abandons the generator, and asyncio's finalizer then raises RuntimeError
            # when the generator is suspended inside a real await in the SDK.
            gen = query(prompt=prompt, options=options)
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
                        if message.subtype != "success":
                            raise RuntimeError(
                                _result_error(message.subtype, message.errors)
                            )
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
                        return {"output": message.result, "usage": raw_usage}

                # The stream ended without a result message, so no message closed the last
                # response, and no message carried a run-level total either. The per-response sum
                # is the only record of what the run spent.
                inference.finish()
                streamed_usage = inference.run_usage
                finish_root_span(span, config, streamed_usage["total"])
                root_usage_written = True
                succeed_span(span)
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
            raise

    def _stream_impl(
        config: AiConfigRep,
        user_input: str = "",
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        return _stream_gen(
            config,
            user_input,
            tool_handlers or {},
            variables or {},
            history,
            capture_content=capture_content,
        )

    return create_handler(("Anthropic", "agent"), _call_impl, _stream_impl)  # type: ignore[arg-type]


async def _stream_gen(
    config: AiConfigRep,
    user_input: str,
    tool_handlers: dict[str, Any],
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    *,
    capture_content: bool = False,
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

    prompt, system_prompt = build_prompt(config, user_input, variables, history)
    opening = _opening_of(prompt, system_prompt)

    native_tool_map, user_config_tools, native_tool_names = partition_tools(
        config.get("tools"), tool_handlers
    )
    catalog = ToolCatalog(config.get("tools"), _native_tool_aliases(tool_handlers))
    set_input_content_attributes(
        span,
        capture_content,
        system_instructions=opening.system_instructions,
        messages=opening.messages,
        tool_definitions=catalog.current,
    )

    inference = InferenceSpans(config, parent, capture_content, catalog, opening)
    tool_telemetry: ToolTelemetry | None = None
    root_usage_written = False
    ended: set[int] = set()
    gen: AsyncIterator[Any] | None = None

    try:
        tool_mcp = (
            await build_tool_mcp(user_config_tools, tool_handlers)
            if user_config_tools
            else None
        )
        mcp_allowed_tools = [MCP_TOOL_PREFIX + n for n in user_config_tools]
        hooks = None
        if native_tool_names or mcp_allowed_tools:
            hooks, tool_telemetry = build_tool_hooks(
                native_tool_map, parent, capture_content
            )

        options = _build_query_options(
            config,
            system_prompt,
            native_tool_names,
            mcp_allowed_tools,
            tool_mcp,
            hooks,
            include_partial_messages=True,
        )

        full_output = ""
        gen = query(prompt=prompt, options=options)
        async for message in gen:
            record_conversation_id(span, message)
            record_native_tools(span, message, capture_content, catalog)
            inference.record(message)

            if isinstance(message, StreamEvent):
                event = message.event
                if (
                    event.get("type") == "content_block_delta"
                    and event.get("delta", {}).get("type") == "text_delta"
                ):
                    text = event["delta"].get("text", "")
                    if text:
                        yield {"type": "chunk", "text": text}
                        full_output += text
            elif isinstance(message, ResultMessage):
                # Before the root, so the children it parents are already closed.
                inference.finish()
                raw_usage: dict[str, Any] = dict(message.usage or {})
                finish_root_span(span, config, raw_usage)
                root_usage_written = True
                if message.subtype != "success":
                    raise RuntimeError(_result_error(message.subtype, message.errors))
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
                yield {"type": "done", "output": final_output, "usage": raw_usage}
                return

        # The stream ended without a result message, so nothing carried a run-level total. The
        # per-response sum is the only record of the spend.
        inference.finish()
        streamed_usage = inference.run_usage
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
        yield {"type": "done", "output": full_output, "usage": streamed_usage["total"]}

    except Exception as exc:
        if tool_telemetry is not None:
            tool_telemetry(exc)
        if not root_usage_written and inference.run_usage["reported"]:
            finish_root_span(span, config, inference.run_usage["total"])
            root_usage_written = True
        fail_span(span, exc, ended)
        raise
    finally:
        # A no-op on the success path, where the result message already closed the last response.
        # On the error and abandonment paths this is the only thing that ends a chat span still
        # open, and the exporter only receives ended spans.
        inference.finish()
        # A no-op on the success and failure paths; on abandonment it is the only chance to close
        # the tree, including any tool span whose PostToolUse hook never fired, and the only chance
        # to report what the responses that did arrive cost.
        if id(span) not in ended:
            if tool_telemetry is not None:
                tool_telemetry(RuntimeError("stream abandoned before completion"))
            if not root_usage_written and inference.run_usage["reported"]:
                finish_root_span(span, config, inference.run_usage["total"])
        end_span_once(span, ended, abandoned=True)
        # Same reasoning as the blocking path: a bare exit through this generator's boundary
        # abandons the vendor's own generator if it is not closed explicitly.
        if gen is not None:
            await gen.aclose()  # type: ignore[attr-defined]


def claude_agents(
    config_key: str,
    user_input: str,
    context: LDContext,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper: creates a handler and calls config(...).invoke()."""
    variables = kwargs.pop("variables", None)
    capture_content = kwargs.pop("capture_content", False)
    return config(
        key=config_key,
        handler=create_claude_agents_handler(capture_content=capture_content),
        **kwargs,
    ).invoke(user_input, context, variables=variables)
