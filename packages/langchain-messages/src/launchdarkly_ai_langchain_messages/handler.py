from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

from launchdarkly_ai_server import (
    AiConfigRep,
    LDContext,
    ProviderHandler,
    SpanMessage,
    SpanMessagePart,
    SpanUsage,
    config,
    create_handler,
    create_run_usage,
    end_span_once,
    end_unfinished_spans,
    lang_chain_finish_reasons,
    lang_chain_span_messages,
    lang_chain_span_usage,
    number_or_zero,
    parse_template,
    set_input_content_attributes,
    set_output_content_attributes,
    set_tool_call_content_attributes,
)

from .spans import (
    fail_span,
    finish_model_span,
    finish_root_span,
    mark_ok,
    parent_context_of,
    start_model_span,
    start_root_span,
    start_tool_span,
    succeed_span,
    to_tool_definitions,
)


def _build_tools(config_tools: dict[str, Any]) -> list[dict[str, Any]]:
    # Not filtered to the tools that have a registered handler, unlike the TypeScript SDK's
    # `buildTools`. That difference predates this span work and changes what the model is offered,
    # not what the span reports, so it stays as it is: the catalog recorded below (via
    # `to_tool_definitions`) is the catalog actually sent.
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            },
        }
        for name, tool in config_tools.items()
    ]


def _build_messages(
    config: AiConfigRep,
    user_input: str,
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> list[Any]:
    """Builds a list of LangChain message objects."""
    import importlib

    msgs_mod = importlib.import_module("langchain_core.messages")
    SystemMessage = msgs_mod.SystemMessage
    HumanMessage = msgs_mod.HumanMessage
    AIMessage = msgs_mod.AIMessage

    messages: list[Any] = []
    last_role: str | None = None

    if config.get("messages"):
        system_msgs = [m for m in config["messages"] if m.get("role") == "system"]
        conv_msgs = [m for m in config["messages"] if m.get("role") != "system"]
        if system_msgs:
            messages.append(
                SystemMessage(
                    parse_template(
                        "\n".join(m["content"] for m in system_msgs), variables
                    )
                )
            )
        for msg in conv_msgs:
            content = parse_template(msg["content"], variables)
            if msg["role"] == "user":
                messages.append(HumanMessage(content))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content))
            last_role = msg["role"]
    elif config.get("instructions"):
        messages.append(
            SystemMessage(parse_template(config["instructions"], variables))
        )

    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content))
            elif role == "assistant":
                messages.append(AIMessage(content))
            last_role = role

    if last_role != "user":
        messages.append(HumanMessage(user_input or ""))
    return messages


def _assistant_output_messages(
    content: Any, tool_calls: list[Any] | None
) -> list[SpanMessage]:
    """One assistant reply as a canonical output message, tool calls included.

    Built as a duck-typed stand-in rather than a real ``AIMessage``, matching the TypeScript
    handler's ``assistantOutput``: :func:`lang_chain_span_messages` only reads ``_get_type()``,
    ``content`` and ``tool_calls``, so constructing a real message here would be extra ceremony for
    fields nothing downstream reads.
    """
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    msg._get_type = lambda: "ai"
    _, messages = lang_chain_span_messages([msg])
    return messages


def _with_get_type(msg: Any) -> Any:
    """Adapts one LangChain message to the interface the client's ``lang_chain_span_messages``
    narrows on.

    Works around a version-skew bug in the shared client helper rather than fixing it there:
    ``lang_chain_span_messages`` reads a message's role off ``_get_type()``, which older LangChain
    releases exposed as the canonical accessor. The ``langchain-core`` release this package
    actually depends on replaced it with a plain ``type`` field and dropped the method entirely, so
    every real ``SystemMessage``/``HumanMessage``/``AIMessage`` reaching the client helper
    unmodified is misclassified as role ``user`` with no error raised: ``getattr(raw,
    '_get_type', None)`` returns ``None`` for a missing attribute rather than raising, and the
    caller has no way to tell "the method is absent" from "this message really has no type".
    Reported in this package's TELEMETRY-CONTRACT.md report rather than patched in
    ``packages/client``, which is out of scope for this change.
    """
    if callable(getattr(msg, "_get_type", None)):
        return msg
    msg_type = getattr(msg, "type", None)
    shim = SimpleNamespace(
        content=getattr(msg, "content", None),
        tool_calls=getattr(msg, "tool_calls", None),
        tool_call_id=getattr(msg, "tool_call_id", None),
    )
    shim._get_type = lambda: msg_type
    return shim


def _span_messages(messages: list[Any]) -> tuple[str | None, list[SpanMessage]]:
    """``lang_chain_span_messages``, after adapting each message. See :func:`_with_get_type`."""
    return lang_chain_span_messages([_with_get_type(m) for m in messages])


def _is_coroutine(fn: Any) -> bool:
    return asyncio.iscoroutinefunction(fn)


_MAX_STEPS = 10


def _make_default_chat_model(config: AiConfigRep, importlib: Any) -> Any:
    """
    Instantiate the appropriate LangChain chat model based on ``config.provider.name``.
    Falls back to ``ChatOpenAI`` when the provider is not recognised.
    Requires the matching ``langchain-<provider>`` integration package to be installed.
    """
    provider = config.get("provider", {}).get("name", "openai").lower()
    model_name = config.get("model", {}).get("name", "")
    if provider == "anthropic":
        lc_anthropic = importlib.import_module("langchain_anthropic")
        return lc_anthropic.ChatAnthropic(
            model=model_name or "claude-3-5-sonnet-20241022"
        )
    lc_openai = importlib.import_module("langchain_openai")
    return lc_openai.ChatOpenAI(model=model_name or "gpt-4o")


async def _run_structured_turn(
    base_model: Any,
    config: AiConfigRep,
    messages: list[Any],
    output_format: dict[str, Any],
    parent: Any,
    *,
    capture_content: bool,
    run_usage: Any,
) -> Any:
    """Runs one structured-output turn under its own ``chat`` child span. Returns the parsed value.

    Used both for the outputFormat-only path and for the final turn of a tool loop that also has an
    outputFormat, mirroring the TypeScript handler's ``runStructuredTurn``.
    """
    model_span = start_model_span(config, parent)
    # Held so the `finally` below can end it. `except Exception` never sees an asyncio.CancelledError,
    # which is a BaseException, so a timeout or a task.cancel() would otherwise strand this span.
    open_model_span: Any = model_span
    # Everything that touches this span stays inside one guard, the input write included. Serialising
    # conversation content raises on anything that is not JSON-serialisable, and a raise out here
    # would leave the chat span open: this path has no `finally` that could recover it.
    try:
        if capture_content:
            system_instructions, span_messages = _span_messages(messages)
            set_input_content_attributes(
                model_span,
                capture_content,
                system_instructions=system_instructions,
                messages=span_messages,
            )
        structured_model = base_model.with_structured_output(
            output_format, include_raw=True
        )
        result = await structured_model.ainvoke(messages)

        raw: Any = (
            result.get("raw")
            if isinstance(result, dict)
            else getattr(result, "raw", None)
        )
        raw_usage = getattr(raw, "usage_metadata", None) or {}
        # Accumulated before anything that can raise. The provider has already billed this turn, so a
        # later content failure must not report the run as having spent less than it did.
        run_usage.add(lang_chain_span_usage(raw_usage))

        parsed: Any = (
            result.get("parsed")
            if isinstance(result, dict)
            else getattr(result, "parsed", None)
        )
        if capture_content:
            set_output_content_attributes(
                model_span,
                capture_content,
                [
                    SpanMessage(
                        role="assistant",
                        parts=[
                            SpanMessagePart(
                                type="text",
                                content=parsed
                                if isinstance(parsed, str)
                                else json.dumps(parsed),
                            )
                        ],
                    )
                ],
            )
        finish_model_span(
            model_span,
            config,
            lang_chain_span_usage(raw_usage) or SpanUsage(),
            lang_chain_finish_reasons(raw),
        )
        open_model_span = None
    except Exception as exc:
        fail_span(model_span, exc)
        open_model_span = None
        raise
    finally:
        # Not an `except`: the whole point is the unwind an `except Exception` cannot see. A
        # cancelled turn still leaves its span exportable, marked and left at UNSET, because
        # nothing failed. The caller went away.
        end_unfinished_spans(open_model_span)
    return parsed


def create_langchain_messages_handler(
    llm: Any = None, *, capture_content: bool = False
) -> ProviderHandler:
    """
    Creates a ``ProviderHandler`` for LangChain (chat models).
    Requires ``langchain-openai`` or another LangChain integration to be installed.
    Pass *llm* to use a specific chat model; omit to default to
    ``ChatOpenAI(model=<config model name>)`` resolved at call time.

    Set *capture_content* to put prompts, model output, tool arguments and tool results on the
    emitted spans. It defaults to off. Conversation content is PII, so a run emits only metadata,
    meaning models, token counts, timings and tool names, until a caller asks for more.
    """

    async def _call_impl(
        config: AiConfigRep,
        user_input: str = "",
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        import importlib

        th = tool_handlers or {}
        vs = variables or {}

        span = start_root_span(config, vs)
        parent = parent_context_of(span)
        # Cleared by whichever path ends the root, so the `finally` can tell an open root from a
        # closed one without asking the span. A mock span answers `is_recording()` truthily, and the
        # test suites are built on mock spans.
        open_root_span: Any = span

        initial_messages = _build_messages(config, user_input, vs, history)

        # Outside the try, so the failure path can still report the spend of the turns that
        # completed before it.
        run_usage = create_run_usage()
        try:
            # Inside the guard, because serialising the prompt raises on anything that is not
            # JSON-serialisable and a raise out here would leave the root open: never ended, never
            # exported, and the run gone from AI Config Monitoring with the feature_flag event on it.
            if capture_content:
                system_instructions, span_messages = _span_messages(initial_messages)
                set_input_content_attributes(
                    span,
                    capture_content,
                    system_instructions=system_instructions,
                    messages=span_messages,
                )
            base_model = (
                llm if llm is not None else _make_default_chat_model(config, importlib)
            )

            tool_defs = _build_tools(config.get("tools") or {})
            output_format = config.get("outputFormat")

            # CASE 1: outputFormat only, no tools -> withStructuredOutput (all providers).
            # LangChain cannot apply with_structured_output and bind_tools to the same model.
            if output_format and not tool_defs:
                parsed = await _run_structured_turn(
                    base_model,
                    config,
                    initial_messages,
                    output_format,
                    parent,
                    capture_content=capture_content,
                    run_usage=run_usage,
                )
                # Behind the flag, because set_output_content_attributes is a no-op without it and
                # json.dumps is not: serialising an object it refuses turned a successful run into a
                # raised TypeError for a caller who had asked for no content at all.
                if capture_content:
                    output_str = (
                        parsed if isinstance(parsed, str) else json.dumps(parsed)
                    )
                    set_output_content_attributes(
                        span,
                        capture_content,
                        [
                            SpanMessage(
                                role="assistant",
                                parts=[
                                    SpanMessagePart(type="text", content=output_str)
                                ],
                            )
                        ],
                    )
                finish_root_span(span, config, run_usage.total)
                succeed_span(span)
                open_root_span = None
                return {
                    "output": parsed,
                    "usage": {
                        "input_tokens": run_usage.total.input,
                        "output_tokens": run_usage.total.output,
                    },
                }

            # CASE 2: tools present -> agentic loop, then withStructuredOutput for the final
            # response when outputFormat is set.
            provider_name = str(
                (config.get("provider") or {}).get("name") or "openai"
            ).lower()
            is_openai = provider_name == "openai"

            # For OpenAI models with both outputFormat and tools: bind response_format so the
            # final text response is structured JSON. The client layer parses the returned string.
            #
            # When this binding applies, the tool loop's own final reply is already structured, so
            # the structured follow-up turn below must not run: it would discard that reply, bill a
            # second turn, and leave the two strategies fighting over the same output.
            format_is_bound = False
            bound_model = base_model
            if output_format and tool_defs and is_openai:
                format_is_bound = True
                bound_model = base_model.bind(
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "output",
                            "schema": output_format,
                            "strict": False,
                        },
                    }
                )

            tool_model = bound_model.bind_tools(tool_defs) if tool_defs else bound_model
            tool_definitions = to_tool_definitions(tool_defs)
            conversation_messages = list(initial_messages)
            output: Any = ""
            steps = 0

            msgs_mod = importlib.import_module("langchain_core.messages")
            ToolMessage = msgs_mod.ToolMessage

            # Held so the `finally` below can end whatever is still open. `except Exception` never
            # sees an asyncio.CancelledError, which is a BaseException, so a timeout or a
            # task.cancel() would otherwise strand every span this loop opened.
            open_model_span: Any = None
            open_tool_span: Any = None
            try:
                while True:
                    model_span = start_model_span(config, parent)
                    open_model_span = model_span
                    # Every write to this span lives inside the guard, the input one included.
                    # Serialising conversation content raises on anything that is not
                    # JSON-serialisable, and a raise out here would leave the chat span open: the
                    # blocking path has no `finally` that could recover it.
                    try:
                        if capture_content:
                            system_instructions, span_messages = _span_messages(
                                conversation_messages
                            )
                            set_input_content_attributes(
                                model_span,
                                capture_content,
                                system_instructions=system_instructions,
                                messages=span_messages,
                                tool_definitions=tool_definitions,
                            )
                        response = await tool_model.ainvoke(conversation_messages)

                        usage = getattr(response, "usage_metadata", None) or {}
                        # Accumulated before anything that can raise, the same way the structured
                        # turn does it. The provider has already billed this turn, so a later
                        # content failure must not report the run as having spent less than it did.
                        run_usage.add(lang_chain_span_usage(usage))
                        tool_calls = getattr(response, "tool_calls", None) or []
                        if capture_content:
                            set_output_content_attributes(
                                model_span,
                                capture_content,
                                _assistant_output_messages(
                                    response.content, tool_calls
                                ),
                            )
                        finish_model_span(
                            model_span,
                            config,
                            lang_chain_span_usage(usage) or SpanUsage(),
                            lang_chain_finish_reasons(response),
                        )
                        open_model_span = None
                    except Exception as exc:
                        fail_span(model_span, exc)
                        open_model_span = None
                        raise

                    if not tool_calls:
                        # Only when response_format was not already bound above. Anthropic and any
                        # other non-OpenAI provider reach the model through this second turn, because
                        # binding response_format is an OpenAI-only mechanism.
                        if output_format and not format_is_bound:
                            output = await _run_structured_turn(
                                base_model,
                                config,
                                conversation_messages,
                                output_format,
                                parent,
                                capture_content=capture_content,
                                run_usage=run_usage,
                            )
                        else:
                            output = (
                                response.content
                                if isinstance(response.content, str)
                                else ""
                            )
                        break

                    if steps >= _MAX_STEPS:
                        raise RuntimeError(
                            f"Tool loop exceeded the maximum number of steps ({_MAX_STEPS})"
                        )
                    steps += 1

                    conversation_messages.append(response)

                    tool_results: list[Any] = []
                    for tc in tool_calls:
                        tool_span = start_tool_span(
                            tc["name"], tc.get("id") or tc["name"], parent
                        )
                        open_tool_span = tool_span
                        set_tool_call_content_attributes(
                            tool_span, capture_content, arguments=tc.get("args")
                        )
                        try:
                            handler_fn = th.get(tc["name"])
                            if not handler_fn or not callable(handler_fn):
                                raise ValueError(
                                    f'No handler registered for tool "{tc["name"]}"'
                                )
                            result_val = (
                                await handler_fn(tc["args"])
                                if _is_coroutine(handler_fn)
                                else handler_fn(tc["args"])
                            )
                            # Inside the try on purpose. Serialising a tool result can raise, most
                            # easily when capture_content is on and the result is not
                            # JSON-serialisable, and a raise out here would leave this span open:
                            # nothing else knows it exists.
                            set_tool_call_content_attributes(
                                tool_span, capture_content, result=result_val
                            )
                            succeed_span(tool_span)
                            open_tool_span = None
                        except Exception as exc:
                            fail_span(tool_span, exc)
                            open_tool_span = None
                            raise
                        tool_results.append(
                            ToolMessage(
                                tool_call_id=tc.get("id") or tc["name"],
                                content=str(result_val),
                            )
                        )
                    conversation_messages.extend(tool_results)
            finally:
                # Not an `except`: the whole point is the unwind an `except Exception` cannot see. A
                # cancelled turn still leaves its spans exportable, marked and left at UNSET,
                # because nothing failed. The caller went away.
                end_unfinished_spans(open_tool_span, open_model_span)

            # Behind the flag, because set_output_content_attributes is a no-op without it and
            # json.dumps is not. With tools and an outputFormat on a non-OpenAI provider the output
            # here is the parsed object, so serialising one json.dumps refuses turned a successful
            # run into a raised TypeError for a caller who had asked for no content at all. The
            # outputFormat-only path above already guards the same work the same way.
            if capture_content:
                output_str = output if isinstance(output, str) else json.dumps(output)
                set_output_content_attributes(
                    span,
                    capture_content,
                    [
                        SpanMessage(
                            role="assistant",
                            parts=[SpanMessagePart(type="text", content=output_str)],
                        )
                    ],
                )
            finish_root_span(span, config, run_usage.total)
            succeed_span(span)
            open_root_span = None
            return {
                "output": output,
                "usage": {
                    "input_tokens": run_usage.total.input,
                    "output_tokens": run_usage.total.output,
                },
            }
        except Exception as exc:
            # The turns that completed were billed, and the root is the only span a config-scoped
            # cost query can find them on.
            if run_usage.reported:
                finish_root_span(span, config, run_usage.total)
            fail_span(span, exc)
            open_root_span = None
            raise
        finally:
            # Not an `except`: asyncio.CancelledError is a BaseException, so a timeout or a
            # task.cancel() never reaches the clause above. Without this the root is stranded, and
            # the root is the only span carrying the feature_flag event and the launchdarkly.*
            # attributes, so the whole run would vanish from AI Config Monitoring rather than show as
            # incomplete.
            if open_root_span is not None and run_usage.reported:
                # The turns that completed were billed, the same reason the failure path reports
                # them.
                finish_root_span(open_root_span, config, run_usage.total)
            end_unfinished_spans(open_root_span)

    def _stream_impl(
        config: AiConfigRep,
        user_input: str = "",
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        return _stream_gen(
            llm,
            config,
            user_input,
            tool_handlers or {},
            variables or {},
            history,
            capture_content=capture_content,
        )

    return create_handler(
        ("*", "messages"),
        _call_impl,  # type: ignore[arg-type]
        _stream_impl,  # type: ignore[arg-type]
        capture_content=capture_content,
    )


async def _stream_gen(
    llm: Any,
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
    ``BaseException``, so ``except Exception`` does not see it. Without the cleanup in ``finally``
    the root span is never ended, so it is never exported, and the whole run disappears from AI
    Config Monitoring along with the ``feature_flag`` event it carries.

    ``ended`` stops the success, failure and abandonment paths from ending the same span twice.

    ``open_chunk_stream`` holds LangChain's own ``astream`` generator, closed by hand in the same
    ``finally``. LangChain's ``BaseChatModel.astream`` is itself an async generator that awaits
    ``run_manager.on_llm_error(...)`` inside a ``except BaseException`` block, the same shape that
    makes ``claude_agent_sdk``'s generator unsafe to abandon bare: a consumer breaking out of this
    generator leaves that inner one suspended mid-await with nothing to resume it until Python's
    garbage collector finalizes it, which can raise rather than clean up quietly. Calling
    ``aclose()`` on it here, on the same object the loop was iterating, drives that cleanup
    immediately instead of leaving it to GC.
    """
    import importlib

    base_model = llm if llm is not None else _make_default_chat_model(config, importlib)

    span = start_root_span(config, variables)
    parent = parent_context_of(span)

    initial_messages = _build_messages(config, user_input, variables, history)

    tool_defs = _build_tools(config.get("tools") or {})
    tool_model = base_model.bind_tools(tool_defs) if tool_defs else base_model
    tool_definitions = to_tool_definitions(tool_defs)
    conversation_messages = list(initial_messages)
    full_output: Any = ""
    steps = 0

    ended: set[int] = set()
    open_model_span: Any = None
    # Tracked for the same reason as the model span: a BaseException raised while a tool runs skips
    # `except Exception` entirely, and `finally` is then the only code that can close this span.
    open_tool_span: Any = None
    open_chunk_stream: Any = None
    # Outside the try, so the failure and abandonment paths can still report the spend.
    run_usage = create_run_usage()

    msgs_mod = importlib.import_module("langchain_core.messages")
    AIMessage = msgs_mod.AIMessage
    ToolMessage = msgs_mod.ToolMessage

    # Distinguishes the two teardown reasons. A consumer that stops reading abandoned the
    # stream; a CancelledError means something cancelled the run, usually a timeout, and the
    # consumer chose nothing. The blocking path already tells these apart.
    cancelled = False
    try:
        # Inside the guard, because serialising the prompt raises on anything that is not
        # JSON-serialisable. A raise out here would leave the root open with the `finally` never
        # entered, so the run would vanish from AI Config Monitoring along with its feature_flag
        # event.
        if capture_content:
            system_instructions, span_messages = _span_messages(initial_messages)
            set_input_content_attributes(
                span,
                capture_content,
                system_instructions=system_instructions,
                messages=span_messages,
            )
        while True:
            model_span = start_model_span(config, parent)
            open_model_span = model_span

            accumulated_content = ""
            accumulated_tool_calls: list[Any] = []
            # The cache breakdown has to be accumulated alongside the scalars. LangChain reports it
            # per chunk in `usage_metadata.input_token_details`, and synthesizing a usage bag
            # without it would make the streaming span emit `cache_read = 0` where the blocking path
            # emits the real figure: a zero that reads as "no cached tokens" rather than "not
            # reported".
            turn_usage = SpanUsage()
            # Carried forward chunk by chunk for the same reason as the cache breakdown above: only
            # the terminal chunk reports it, and dropping it would make the streaming span omit a
            # finish reason where the blocking path emits the real one.
            finish_reasons: list[str] | None = None
            usage_reported = False

            try:
                # Inside the guard, so a serialisation failure fails this chat span. Outside it, the
                # raise reached the outer `finally` with `open_model_span` still set and the span was
                # ended as abandoned, which reads as a consumer walking away rather than a failure.
                if capture_content:
                    system_instructions, span_messages = _span_messages(
                        conversation_messages
                    )
                    set_input_content_attributes(
                        model_span,
                        capture_content,
                        system_instructions=system_instructions,
                        messages=span_messages,
                        tool_definitions=tool_definitions,
                    )
                chunk_stream = tool_model.astream(conversation_messages)
                open_chunk_stream = chunk_stream
                async for chunk in chunk_stream:
                    text = chunk.content if isinstance(chunk.content, str) else ""
                    if text:
                        yield {"type": "chunk", "text": text}
                        accumulated_content += text
                    usage = getattr(chunk, "usage_metadata", None)
                    if usage:
                        usage_reported = True
                        details = usage.get("input_token_details") or {}
                        turn_usage.input += number_or_zero(usage.get("input_tokens"))
                        turn_usage.output += number_or_zero(usage.get("output_tokens"))
                        turn_usage.cache_read += number_or_zero(
                            details.get("cache_read")
                        )
                        turn_usage.cache_creation += number_or_zero(
                            details.get("cache_creation")
                        )
                    chunk_tools = getattr(chunk, "tool_calls", None) or []
                    if chunk_tools:
                        accumulated_tool_calls = chunk_tools
                    finish_reasons = lang_chain_finish_reasons(chunk) or finish_reasons
                open_chunk_stream = None
            except Exception as exc:
                # The tracker matters here: the outer `except` also fails `open_model_span`, which
                # still points at this span because the line that clears it is unreachable on this
                # path.
                fail_span(model_span, exc, ended)
                open_model_span = None
                raise

            # The already-mapped figures, not a bag rebuilt from them: this path summed the chunks
            # into a SpanUsage to begin with, so re-parsing its own output would be a round trip
            # whose only effect is another chance to disagree with itself.
            #
            # Only when a chunk actually carried usage. Adding unconditionally marks the run as
            # having reported, so a later failure or abandonment writes all-zero totals on the root
            # and claims the run cost nothing, which is the one thing `reported` exists to prevent.
            # The blocking path gets this for free, because lang_chain_span_usage returns None for a
            # bag the provider never filled.
            #
            # Before the content write, and before the span finish, because the provider has already
            # billed these tokens: a content failure is our problem and must not report the run as
            # having spent less than it did.
            run_usage.add(turn_usage if usage_reported else None)

            # Inside a guard for the same reason as the blocking path: a raise while serialising
            # completion content would otherwise leave this span for `finally` to end as abandoned,
            # which reads as a consumer who walked away rather than as the failure it is.
            try:
                if capture_content:
                    set_output_content_attributes(
                        model_span,
                        capture_content,
                        _assistant_output_messages(
                            accumulated_content, accumulated_tool_calls
                        ),
                    )
                # finish_model_span ends the span. Clearing open_model_span is what stops the
                # `finally` from ending it a second time.
                finish_model_span(model_span, config, turn_usage, finish_reasons)
                open_model_span = None
            except Exception as exc:
                fail_span(model_span, exc, ended)
                open_model_span = None
                raise

            if not accumulated_tool_calls:
                full_output = (full_output or "") + accumulated_content
                break

            if steps >= _MAX_STEPS:
                raise RuntimeError(
                    f"Tool loop exceeded the maximum number of steps ({_MAX_STEPS})"
                )
            steps += 1

            full_output = (full_output or "") + accumulated_content
            assistant_msg = AIMessage(
                content=accumulated_content, tool_calls=accumulated_tool_calls
            )
            conversation_messages.append(assistant_msg)

            tool_results: list[Any] = []
            for tc in accumulated_tool_calls:
                tool_span = start_tool_span(
                    tc["name"], tc.get("id") or tc["name"], parent
                )
                open_tool_span = tool_span
                set_tool_call_content_attributes(
                    tool_span, capture_content, arguments=tc.get("args")
                )
                try:
                    handler_fn = tool_handlers.get(tc["name"])
                    if not handler_fn or not callable(handler_fn):
                        raise ValueError(
                            f'No handler registered for tool "{tc["name"]}"'
                        )
                    result_val = (
                        await handler_fn(tc["args"])
                        if _is_coroutine(handler_fn)
                        else handler_fn(tc["args"])
                    )
                    # Inside the try on purpose. Serialising a tool result can raise, most easily
                    # when capture_content is on and the result is not JSON-serialisable, and a
                    # raise out here would leave this span open: nothing else knows it exists.
                    set_tool_call_content_attributes(
                        tool_span, capture_content, result=result_val
                    )
                    succeed_span(tool_span)
                    open_tool_span = None
                except Exception as exc:
                    fail_span(tool_span, exc, ended)
                    open_tool_span = None
                    raise
                # Cleared on both paths that end the span, and deliberately not in a `finally`:
                # a `finally` would also clear it for a BaseException, which is the one case where
                # the span is still open and the outer `finally` is the only thing left to close it.
                tool_results.append(
                    ToolMessage(
                        tool_call_id=tc.get("id") or tc["name"], content=str(result_val)
                    )
                )
            conversation_messages.extend(tool_results)

        full_output_str = (
            full_output if isinstance(full_output, str) else json.dumps(full_output)
        )
        # Guarded like every equivalent site on the blocking path. The helper is a no-op without
        # capture, but building the SpanMessage is not, and a caller who turned content capture off
        # should not pay for a transcript nobody will read. `full_output_str` itself stays outside
        # the guard: the done event below returns it whatever the capture setting is.
        if capture_content:
            set_output_content_attributes(
                span,
                capture_content,
                [
                    SpanMessage(
                        role="assistant",
                        parts=[SpanMessagePart(type="text", content=full_output_str)],
                    )
                ],
            )
        finish_root_span(span, config, run_usage.total)
        mark_ok(span)
        end_span_once(span, ended)

        yield {
            "type": "done",
            "output": full_output_str,
            "usage": {
                "input_tokens": run_usage.total.input,
                "output_tokens": run_usage.total.output,
            },
        }

    except asyncio.CancelledError:
        cancelled = True
        raise
    except Exception as exc:
        if run_usage.reported:
            finish_root_span(span, config, run_usage.total)
        fail_span(span, exc, ended)
        raise
    finally:
        # A no-op on the success and failure paths, because both already ended their spans through
        # `ended` and already drove `chunk_stream` to completion. On abandonment this is the only
        # chance to close the tree, to close LangChain's own generator, and to report what the
        # completed turns already cost. An abandoned span is left UNSET rather than ERROR: stopping
        # early is a normal thing for a consumer to do, and LaunchDarkly's own metrics record
        # neither a success nor an error for it, so ERROR would put two dashboards in disagreement.
        # Spans first, and the vendor generator after. aclose() can raise, and doing it first would
        # take the whole teardown with it: the root would never end, never export, and the run would
        # vanish from AI Config Monitoring along with the feature_flag event this block exists to
        # protect. Its own failure is not worth losing the trace over, so it is contained.
        # Tool span first: it is a child, and a reader following the tree should not meet a closed
        # parent above an open child.
        if open_tool_span is not None:
            end_span_once(open_tool_span, ended, abandoned=True, cancelled=cancelled)
        if open_model_span is not None:
            end_span_once(open_model_span, ended, abandoned=True, cancelled=cancelled)
        if span is not None and id(span) not in ended and run_usage.reported:
            finish_root_span(span, config, run_usage.total)
        end_span_once(span, ended, abandoned=True, cancelled=cancelled)
        if open_chunk_stream is not None:
            try:
                await open_chunk_stream.aclose()
            except Exception:  # pragma: no cover - best-effort vendor teardown
                pass


def langchain_messages(
    config_key: str,
    user_input: str,
    context: LDContext,
    llm: Any = None,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper: creates a handler and calls config(...).invoke()."""
    # Both are lifted out of kwargs: capture_content configures the handler, variables belong to
    # the invocation. Leaving either in would pass it to config(), which takes neither, so a caller
    # asking for content on spans got a TypeError instead of content.
    variables = kwargs.pop("variables", None)
    capture_content = kwargs.pop("capture_content", False)
    return config(
        key=config_key,
        handler=create_langchain_messages_handler(
            llm=llm, capture_content=capture_content
        ),
        **kwargs,
    ).invoke(user_input, context, variables=variables)
