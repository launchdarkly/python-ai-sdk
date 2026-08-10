from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

from launchdarkly_ai_server import (
    AiConfigRep,
    LDContext,
    ProviderHandler,
    compose_history,
    config,
    create_handler,
    image_block_to_url,
    is_content_blocks,
    parse_template,
    set_ld_span_attributes,
    set_openllmetry_completion,
    set_openllmetry_prompt,
)

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode as SpanStatusCode

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


def _build_tools(config_tools: dict[str, Any]) -> list[dict[str, Any]]:
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


def _to_langchain_content(content: Any) -> Any:
    """Maps LD-canonical content blocks to LangChain multimodal content parts.
    String content passes through so text-only callers keep plain strings."""
    if not is_content_blocks(content):
        return content if content is not None else ""

    parts: list[dict[str, Any]] = []
    for block in content:
        if block.get("type") == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image":
            parts.append(
                {"type": "image_url", "image_url": {"url": image_block_to_url(block)}}
            )
    return parts


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
    config_messages: list[dict[str, Any]] = []

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
            content = msg.get("content", "")
            if isinstance(content, str):
                content = parse_template(content, variables)
            if msg.get("role") in ("user", "assistant"):
                config_messages.append({"role": msg["role"], "content": content})
    elif config.get("instructions"):
        messages.append(
            SystemMessage(parse_template(config["instructions"], variables))
        )

    def append_turn(turn: dict[str, Any]) -> None:
        content = _to_langchain_content(turn.get("content"))
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))

    if history:
        for turn in compose_history(
            history=history, user_input=user_input, config_messages=config_messages
        ):
            append_turn(turn)
        return messages

    for turn in config_messages:
        append_turn(turn)

    # Preserve the no-history behaviour: an empty input still produces a human
    # message when the config carries no trailing user turn, so an empty history
    # array stays identical to omitting history entirely.
    last_non_system = next(
        (m for m in reversed(messages) if getattr(m, "type", "") != "system"), None
    )
    if getattr(last_non_system, "type", "") != "human":
        messages.append(HumanMessage(user_input or ""))
    return messages


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


def create_langchain_messages_handler(llm: Any = None) -> ProviderHandler:
    """
    Creates a ``ProviderHandler`` for LangChain (chat models).
    Requires ``langchain-openai`` or another LangChain integration to be installed.
    Pass *llm* to use a specific chat model; omit to default to
    ``ChatOpenAI(model=<config model name>)`` resolved at call time.
    """
    tracer_name = "@launchdarkly/ai-langchain-messages"

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

        base_model = llm
        if base_model is None:
            base_model = _make_default_chat_model(config, importlib)

        if _HAS_OTEL:
            span = trace.get_tracer(tracer_name).start_span("langchain.invoke")
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute(
                "gen_ai.system",
                config.get("provider", {}).get("name", "langchain").lower(),
            )
            span.set_attribute(
                "gen_ai.request.model", config.get("model", {}).get("name", "")
            )
            set_ld_span_attributes(span, vs)
        else:
            span = None

        initial_messages = _build_messages(config, user_input, vs, history)

        if span:
            prompt_text = "\n".join(
                f"{getattr(m, 'type', type(m).__name__)}: {m.content if isinstance(m.content, str) else json.dumps(m.content)}"
                for m in initial_messages
            )
            span.add_event("gen_ai.content.prompt", {"gen_ai.prompt": prompt_text})
            set_openllmetry_prompt(
                span,
                [
                    {
                        "role": getattr(m, "type", type(m).__name__),
                        "content": m.content
                        if isinstance(m.content, str)
                        else json.dumps(m.content),
                    }
                    for m in initial_messages
                ],
            )

        try:
            tool_defs = _build_tools(config.get("tools") or {})
            output_format = config.get("outputFormat")
            provider_name = config.get("provider", {}).get("name", "openai").lower()
            is_openai = provider_name == "openai"

            # Structured output path — only when no tools are present.
            # LangChain cannot apply with_structured_output and bind_tools to the same model.
            if output_format and not tool_defs:
                structured_model = base_model.with_structured_output(
                    output_format, include_raw=True
                )
                result = await structured_model.ainvoke(initial_messages)
                raw_usage = getattr(result.get("raw"), "usage_metadata", None) or {}
                input_tokens = raw_usage.get("input_tokens", 0)
                output_tokens = raw_usage.get("output_tokens", 0)
                if span:
                    span.set_attribute(
                        "gen_ai.response.model", config.get("model", {}).get("name", "")
                    )
                    span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                    span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
                    span.set_attribute(
                        "gen_ai.usage.total_tokens", input_tokens + output_tokens
                    )
                    span.add_event(
                        "gen_ai.content.completion",
                        {"gen_ai.completion": json.dumps(result.get("parsed"))},
                    )
                    set_openllmetry_completion(
                        span,
                        json.dumps(result.get("parsed")),
                        {"input_tokens": input_tokens, "output_tokens": output_tokens},
                    )
                    span.set_status(SpanStatusCode.OK)
                    span.end()
                return {
                    "output": result.get("parsed"),
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                }

            # For OpenAI models with both outputFormat and tools: bind response_format so the
            # final text response is structured JSON. The client layer parses the returned string.
            bound_model = base_model
            if output_format and tool_defs and is_openai:
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

            active_model = (
                bound_model.bind_tools(tool_defs) if tool_defs else bound_model
            )
            conversation_messages = list(initial_messages)
            total_input = 0
            total_output = 0
            output = ""
            steps = 0

            while True:
                response = await active_model.ainvoke(conversation_messages)
                usage = getattr(response, "usage_metadata", None) or {}
                total_input += usage.get("input_tokens", 0)
                total_output += usage.get("output_tokens", 0)
                conversation_messages.append(response)

                tool_calls = getattr(response, "tool_calls", []) or []
                if not tool_calls:
                    output = (
                        response.content if isinstance(response.content, str) else ""
                    )
                    break

                if steps >= _MAX_STEPS:
                    raise RuntimeError(
                        f"Tool loop exceeded the maximum number of steps ({_MAX_STEPS})"
                    )
                steps += 1

                import importlib

                msgs_mod = importlib.import_module("langchain_core.messages")
                ToolMessage = msgs_mod.ToolMessage

                tool_results: list[Any] = []
                for tc in tool_calls:
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
                    tool_results.append(
                        ToolMessage(
                            tool_call_id=tc.get("id") or tc["name"],
                            content=str(result_val),
                        )
                    )
                conversation_messages.extend(tool_results)

            if span:
                span.set_attribute(
                    "gen_ai.response.model", config.get("model", {}).get("name", "")
                )
                span.set_attribute("gen_ai.usage.input_tokens", total_input)
                span.set_attribute("gen_ai.usage.output_tokens", total_output)
                span.set_attribute(
                    "gen_ai.usage.total_tokens", total_input + total_output
                )
                span.add_event(
                    "gen_ai.content.completion",
                    {
                        "gen_ai.completion": output
                        if isinstance(output, str)
                        else json.dumps(output)
                    },
                )
                set_openllmetry_completion(
                    span,
                    output if isinstance(output, str) else json.dumps(output),
                    {"input_tokens": total_input, "output_tokens": total_output},
                )
                span.set_status(SpanStatusCode.OK)
                span.end()

            return {
                "output": output,
                "usage": {"input_tokens": total_input, "output_tokens": total_output},
            }

        except Exception as exc:
            if span:
                span.record_exception(exc)
                span.set_status(SpanStatusCode.ERROR, str(exc))
                span.end()
            raise

    def _stream_impl(
        config: AiConfigRep,
        user_input: str = "",
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        return _stream_gen(
            llm, config, user_input, tool_handlers or {}, variables or {}, history
        )

    return create_handler(("*", "messages"), _call_impl, _stream_impl)  # type: ignore[arg-type]


async def _stream_gen(
    llm: Any,
    config: AiConfigRep,
    user_input: str,
    tool_handlers: dict[str, Any],
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    import importlib

    base_model = llm
    if base_model is None:
        base_model = _make_default_chat_model(config, importlib)

    tracer_name = "@launchdarkly/ai-langchain-messages"
    if _HAS_OTEL:
        span = trace.get_tracer(tracer_name).start_span("langchain.stream")
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute(
            "gen_ai.system", config.get("provider", {}).get("name", "langchain").lower()
        )
        span.set_attribute(
            "gen_ai.request.model", config.get("model", {}).get("name", "")
        )
        set_ld_span_attributes(span, variables)
    else:
        span = None

    initial_messages = _build_messages(config, user_input, variables, history)
    if span:
        prompt_text = "\n".join(
            f"{getattr(m, 'type', type(m).__name__)}: {m.content if isinstance(m.content, str) else json.dumps(m.content)}"
            for m in initial_messages
        )
        span.add_event("gen_ai.content.prompt", {"gen_ai.prompt": prompt_text})
        set_openllmetry_prompt(
            span,
            [
                {
                    "role": getattr(m, "type", type(m).__name__),
                    "content": m.content
                    if isinstance(m.content, str)
                    else json.dumps(m.content),
                }
                for m in initial_messages
            ],
        )

    tool_defs = _build_tools(config.get("tools") or {})
    active_model = base_model.bind_tools(tool_defs) if tool_defs else base_model
    conversation_messages = list(initial_messages)
    total_input = 0
    total_output = 0
    full_output = ""
    steps = 0

    try:
        import importlib

        msgs_mod = importlib.import_module("langchain_core.messages")
        AIMessage = msgs_mod.AIMessage
        ToolMessage = msgs_mod.ToolMessage

        while True:
            chunk_stream = active_model.astream(conversation_messages)
            accumulated_content = ""
            accumulated_tool_calls: list[Any] = []
            turn_input = 0
            turn_output = 0

            async for chunk in chunk_stream:
                text = chunk.content if isinstance(chunk.content, str) else ""
                if text:
                    yield {"type": "chunk", "text": text}
                    accumulated_content += text
                usage = getattr(chunk, "usage_metadata", None)
                if usage:
                    turn_input += usage.get("input_tokens", 0)
                    turn_output += usage.get("output_tokens", 0)
                chunk_tools = getattr(chunk, "tool_calls", []) or []
                if chunk_tools:
                    accumulated_tool_calls = chunk_tools

            total_input += turn_input
            total_output += turn_output

            if not accumulated_tool_calls:
                full_output += accumulated_content
                break

            if steps >= _MAX_STEPS:
                raise RuntimeError(
                    f"Tool loop exceeded the maximum number of steps ({_MAX_STEPS})"
                )
            steps += 1

            full_output += accumulated_content
            assistant_msg = AIMessage(
                content=accumulated_content, tool_calls=accumulated_tool_calls
            )
            conversation_messages.append(assistant_msg)

            tool_results: list[Any] = []
            for tc in accumulated_tool_calls:
                handler_fn = tool_handlers.get(tc["name"])
                if not handler_fn or not callable(handler_fn):
                    raise ValueError(f'No handler registered for tool "{tc["name"]}"')
                result_val = (
                    await handler_fn(tc["args"])
                    if _is_coroutine(handler_fn)
                    else handler_fn(tc["args"])
                )
                tool_results.append(
                    ToolMessage(
                        tool_call_id=tc.get("id") or tc["name"], content=str(result_val)
                    )
                )
            conversation_messages.extend(tool_results)

        if span:
            span.set_attribute(
                "gen_ai.response.model", config.get("model", {}).get("name", "")
            )
            span.set_attribute("gen_ai.usage.input_tokens", total_input)
            span.set_attribute("gen_ai.usage.output_tokens", total_output)
            span.set_attribute("gen_ai.usage.total_tokens", total_input + total_output)
            span.add_event(
                "gen_ai.content.completion",
                {
                    "gen_ai.completion": full_output
                    if isinstance(full_output, str)
                    else json.dumps(full_output)
                },
            )
            set_openllmetry_completion(
                span,
                full_output
                if isinstance(full_output, str)
                else json.dumps(full_output),
                {"input_tokens": total_input, "output_tokens": total_output},
            )
            span.set_status(SpanStatusCode.OK)
            span.end()

        yield {
            "type": "done",
            "output": full_output,
            "usage": {"input_tokens": total_input, "output_tokens": total_output},
        }

    except Exception as exc:
        if span:
            span.record_exception(exc)
            span.set_status(SpanStatusCode.ERROR, str(exc))
            span.end()
        raise


def langchain_messages(
    config_key: str,
    user_input: str,
    context: LDContext,
    llm: Any = None,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper: creates a handler and calls config(...).invoke()."""
    variables = kwargs.pop("variables", None)
    return config(
        key=config_key, handler=create_langchain_messages_handler(llm=llm), **kwargs
    ).invoke(user_input, context, variables=variables)
