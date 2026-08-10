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
    content_to_text,
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
            "name": name,
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
            "strict": False,
        }
        for name, tool in config_tools.items()
    ]


def _build_input_messages(
    config: AiConfigRep,
    user_input: str | None,
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    system_messages: list[dict[str, Any]] = []
    config_messages: list[dict[str, Any]] = []

    if config.get("messages"):
        for message in config["messages"]:
            content = message.get("content", "")
            if isinstance(content, str):
                content = parse_template(content, variables)
            mapped = {"role": message["role"], "content": content}
            if message["role"] == "system":
                system_messages.append(mapped)
            else:
                config_messages.append(mapped)
    else:
        instructions = parse_template(config.get("instructions") or "", variables)
        if instructions:
            system_messages.append({"role": "system", "content": instructions})

    if history:
        turns = compose_history(
            history=history,
            user_input=user_input,
            config_messages=config_messages,
        )
    else:
        # No history: preserve the pre-history behaviour so an empty history is
        # identical to passing none (TESTING.md §1.11). compose_history only
        # appends user_input when truthy, which would drop the trailing user
        # turn an instructions-only config still needs.
        turns = list(config_messages)
        if config.get("messages"):
            if user_input and (not turns or turns[-1].get("role") != "user"):
                turns.append({"role": "user", "content": user_input})
        else:
            turns.append({"role": "user", "content": user_input or ""})

    return system_messages + [
        {"role": turn["role"], "content": _map_message_content(turn)}
        for turn in turns
        if turn.get("role") in ("user", "assistant")
    ]


def _map_message_content(message: dict[str, Any]) -> Any:
    raw_content = message.get("content")
    content: str | list[dict[str, Any]] = (
        raw_content if isinstance(raw_content, (str, list)) else ""
    )
    if not is_content_blocks(content):
        return content
    assert isinstance(content, list)

    if message.get("role") != "user":
        return content_to_text(content)

    parts: list[dict[str, Any]] = []
    for block in content:
        if block.get("type") == "text":
            parts.append({"type": "input_text", "text": block.get("text", "")})
        elif block.get("type") == "image":
            parts.append(
                {"type": "input_image", "image_url": image_block_to_url(block)}
            )
    return parts


def _telemetry_messages(
    input_messages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "role": message["role"],
            "content": message["content"]
            if isinstance(message["content"], str)
            else json.dumps(message["content"]),
        }
        for message in input_messages
    ]


def _is_coroutine(fn: Any) -> bool:
    return asyncio.iscoroutinefunction(fn)


_MAX_STEPS = 10


def create_openai_messages_handler() -> ProviderHandler:
    """
    Creates a ``ProviderHandler`` for OpenAI (responses API).
    Requires ``openai`` to be installed as a peer dependency.
    """
    import importlib

    openai_mod = importlib.import_module("openai")
    client = openai_mod.AsyncOpenAI()

    tracer_name = "@launchdarkly/ai-openai-messages"

    async def _call_impl(
        config: AiConfigRep,
        user_input: str = "",
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        th = tool_handlers or {}
        vs = variables or {}

        if _HAS_OTEL:
            span = trace.get_tracer(tracer_name).start_span("openai.response")
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.system", "openai")
            span.set_attribute(
                "gen_ai.request.model", config.get("model", {}).get("name", "")
            )
            set_ld_span_attributes(span, vs)
        else:
            span = None

        tools = _build_tools(config.get("tools") or {})
        input_messages = _build_input_messages(config, user_input, vs, history)

        if span:
            span.add_event(
                "gen_ai.content.prompt", {"gen_ai.prompt": json.dumps(input_messages)}
            )
            set_openllmetry_prompt(
                span,
                _telemetry_messages(input_messages),
            )

        try:
            kwargs: dict[str, Any] = {
                "model": config["model"]["name"],
                "input": input_messages,
            }
            if tools:
                kwargs["tools"] = tools
            if config.get("outputFormat"):
                kwargs["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "output",
                        "schema": config["outputFormat"],
                        "strict": False,
                    }
                }

            response = await client.responses.create(**kwargs)
            total_input = getattr(response.usage, "input_tokens", 0) or 0
            total_output = getattr(response.usage, "output_tokens", 0) or 0
            steps = 0

            while True:
                tool_calls = [
                    item
                    for item in (response.output or [])
                    if getattr(item, "type", None) == "function_call"
                ]
                if not tool_calls:
                    break

                if steps >= _MAX_STEPS:
                    raise RuntimeError(
                        f"Tool loop exceeded the maximum number of steps ({_MAX_STEPS})"
                    )
                steps += 1

                tool_outputs = []
                for tc in tool_calls:
                    args = json.loads(tc.arguments)
                    handler_fn = th.get(tc.name)
                    if not handler_fn:
                        raise ValueError(f'No handler registered for tool "{tc.name}"')
                    result = (
                        await handler_fn(args)
                        if _is_coroutine(handler_fn)
                        else handler_fn(args)
                    )
                    tool_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": tc.call_id,
                            "output": str(result),
                        }
                    )

                response = await client.responses.create(
                    model=config["model"]["name"],
                    previous_response_id=response.id,
                    input=tool_outputs,
                )
                total_input += getattr(response.usage, "input_tokens", 0) or 0
                total_output += getattr(response.usage, "output_tokens", 0) or 0

            output = getattr(response, "output_text", None) or ""

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
            client, config, user_input, tool_handlers or {}, variables or {}, history
        )

    return create_handler(("OpenAI", "messages"), _call_impl, _stream_impl)  # type: ignore[arg-type]


async def _stream_gen(
    client: Any,
    config: AiConfigRep,
    user_input: str,
    tool_handlers: dict[str, Any],
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    tracer_name = "@launchdarkly/ai-openai-messages"
    if _HAS_OTEL:
        span = trace.get_tracer(tracer_name).start_span("openai.response.stream")
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.system", "openai")
        span.set_attribute(
            "gen_ai.request.model", config.get("model", {}).get("name", "")
        )
        set_ld_span_attributes(span, variables)
    else:
        span = None

    tools = _build_tools(config.get("tools") or {})
    input_messages = _build_input_messages(config, user_input, variables, history)

    if span:
        span.add_event(
            "gen_ai.content.prompt", {"gen_ai.prompt": json.dumps(input_messages)}
        )
        set_openllmetry_prompt(
            span,
            _telemetry_messages(input_messages),
        )

    total_input = 0
    total_output = 0
    full_output = ""
    previous_response_id: str | None = None
    current_input: Any = input_messages
    steps = 0

    try:
        while True:
            stream_params: dict[str, Any] = {
                "model": config["model"]["name"],
                "input": current_input,
            }
            if previous_response_id:
                stream_params["previous_response_id"] = previous_response_id
            if tools:
                stream_params["tools"] = tools

            stream = client.responses.stream(**stream_params)
            async with stream as s:
                async for event in s:
                    if getattr(event, "type", None) == "response.output_text.delta":
                        text = getattr(event, "delta", "")
                        full_output += text
                        yield {"type": "chunk", "text": text}

                final_resp = await s.get_final_response()

            total_input += (
                getattr(getattr(final_resp, "usage", None), "input_tokens", 0) or 0
            )
            total_output += (
                getattr(getattr(final_resp, "usage", None), "output_tokens", 0) or 0
            )

            tool_calls = [
                item
                for item in (getattr(final_resp, "output", []) or [])
                if getattr(item, "type", None) == "function_call"
            ]
            if not tool_calls:
                break

            if steps >= _MAX_STEPS:
                raise RuntimeError(
                    f"Tool loop exceeded the maximum number of steps ({_MAX_STEPS})"
                )
            steps += 1

            previous_response_id = getattr(final_resp, "id", None)
            tool_outputs = []
            for tc in tool_calls:
                args = json.loads(tc.arguments)
                handler_fn = tool_handlers.get(tc.name)
                if not handler_fn:
                    raise ValueError(f'No handler registered for tool "{tc.name}"')
                result = (
                    await handler_fn(args)
                    if _is_coroutine(handler_fn)
                    else handler_fn(args)
                )
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": tc.call_id,
                        "output": str(result),
                    }
                )
            current_input = tool_outputs

        if span:
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


def openai_messages(
    config_key: str,
    user_input: str,
    context: LDContext,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper: creates a handler and calls config(...).invoke()."""
    variables = kwargs.pop("variables", None)
    return config(
        key=config_key, handler=create_openai_messages_handler(), **kwargs
    ).invoke(user_input, context, variables=variables)
