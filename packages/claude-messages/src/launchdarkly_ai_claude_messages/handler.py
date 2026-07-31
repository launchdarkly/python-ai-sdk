from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

from launchdarkly_ai_server import (
    AiConfigRep,
    LDContext,
    ProviderHandler,
    config,
    create_handler,
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

try:
    import anthropic as _anthropic_mod  # noqa: F401

    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


def _build_tools(config_tools: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters", {}),
        }
        for name, tool in config_tools.items()
    ]


def _build_messages(
    config: AiConfigRep,
    user_input: str,
    variables: dict[str, Any],
    *,
    include_output_format: bool = True,
    history: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Returns (messages, system_prompt)."""
    system: str | None = None
    messages: list[dict[str, Any]] = []

    if config.get("messages"):
        system_msgs = [m for m in config["messages"] if m.get("role") == "system"]
        conv_msgs = [m for m in config["messages"] if m.get("role") != "system"]
        if system_msgs:
            system = parse_template(
                "\n".join(m["content"] for m in system_msgs), variables
            )
        for msg in conv_msgs:
            messages.append(
                {
                    "role": msg["role"],
                    "content": parse_template(msg["content"], variables),
                }
            )
    elif config.get("instructions"):
        system = parse_template(config["instructions"], variables)

    if history:
        for msg in history:
            role = msg.get("role", "user")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": msg.get("content", "")})

    if not messages or messages[-1].get("role") != "user":
        messages.append({"role": "user", "content": user_input or ""})

    if include_output_format and config.get("outputFormat"):
        schema_instruction = f"Respond with valid JSON matching this schema:\n{json.dumps(config['outputFormat'])}"
        system = f"{system}\n\n{schema_instruction}" if system else schema_instruction

    return messages, system


async def _run_tool_loop(
    client: Any,
    config: AiConfigRep,
    messages: list[dict[str, Any]],
    system: str | None,
    tool_handlers: dict[str, Any],
) -> tuple[str, int, int]:
    """Runs the Anthropic messages loop, handling tool calls.  Returns (output, input_tokens, output_tokens)."""
    tools = _build_tools(config.get("tools") or {})
    max_tokens = (config.get("model", {}).get("parameters") or {}).get(
        "max_tokens", 1024
    )
    conversation = list(messages)
    total_input = 0
    total_output = 0
    output = ""
    steps = 0

    while True:
        kwargs: dict[str, Any] = {
            "model": config["model"]["name"],
            "max_tokens": max_tokens,
            "messages": conversation,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        resp = await client.messages.create(**kwargs)
        total_input += resp.usage.input_tokens
        total_output += resp.usage.output_tokens

        if resp.stop_reason != "tool_use":
            output = "".join(
                block.text for block in resp.content if block.type == "text"
            )
            break

        if steps >= _MAX_STEPS:
            raise RuntimeError(
                f"Tool loop exceeded the maximum number of steps ({_MAX_STEPS})"
            )
        steps += 1

        conversation.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            handler_fn = tool_handlers.get(block.name)
            if not handler_fn or not callable(handler_fn):
                raise ValueError(f'No handler registered for tool "{block.name}"')
            result = (
                await handler_fn(block.input)
                if _is_coroutine(handler_fn)
                else handler_fn(block.input)
            )
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": str(result)}
            )
        conversation.append({"role": "user", "content": tool_results})

    return output, total_input, total_output


def _is_coroutine(fn: Any) -> bool:
    return asyncio.iscoroutinefunction(fn)


_MAX_STEPS = 10


def create_claude_messages_handler() -> ProviderHandler:
    """
    Creates a ``ProviderHandler`` for Anthropic Claude (messages API).
    Requires ``anthropic`` to be installed as a peer dependency.
    """
    import importlib

    anthropic_mod = importlib.import_module("anthropic")
    client = anthropic_mod.AsyncAnthropic()

    tracer_name = "@launchdarkly/ai-claude-messages"

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
            tracer = trace.get_tracer(tracer_name)
            span = tracer.start_span("claude.messages")
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.system", "anthropic")
            span.set_attribute(
                "gen_ai.request.model", config.get("model", {}).get("name", "")
            )
            set_ld_span_attributes(span, vs)
        else:
            span = None

        messages, system = _build_messages(config, user_input, vs, history=history)
        if span:
            prompt_text = (f"system: {system}\n" if system else "") + "\n".join(
                f"{m['role']}: {m['content']}" for m in messages
            )
            span.add_event("gen_ai.content.prompt", {"gen_ai.prompt": prompt_text})
            prompt_msgs = (
                [{"role": "system", "content": system}] if system else []
            ) + [{"role": m["role"], "content": m["content"]} for m in messages]
            set_openllmetry_prompt(span, prompt_msgs)

        try:
            output, inp, out = await _run_tool_loop(
                client, config, messages, system, th
            )
            if span:
                span.set_attribute(
                    "gen_ai.response.model", config.get("model", {}).get("name", "")
                )
                span.set_attribute("gen_ai.usage.input_tokens", inp)
                span.set_attribute("gen_ai.usage.output_tokens", out)
                span.set_attribute("gen_ai.usage.total_tokens", inp + out)
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
                    {"input_tokens": inp, "output_tokens": out},
                )
                span.set_status(SpanStatusCode.OK)
                span.end()
            return {
                "output": output,
                "usage": {"input_tokens": inp, "output_tokens": out},
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

    return create_handler(("Anthropic", "messages"), _call_impl, _stream_impl)  # type: ignore[arg-type]


async def _stream_gen(
    client: Any,
    config: AiConfigRep,
    user_input: str,
    tool_handlers: dict[str, Any],
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    tracer_name = "@launchdarkly/ai-claude-messages"
    if _HAS_OTEL:
        span = trace.get_tracer(tracer_name).start_span("claude.messages.stream")
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.system", "anthropic")
        span.set_attribute(
            "gen_ai.request.model", config.get("model", {}).get("name", "")
        )
        set_ld_span_attributes(span, variables)
    else:
        span = None

    messages, system = _build_messages(
        config, user_input, variables, include_output_format=False, history=history
    )
    if span:
        prompt_text = (f"system: {system}\n" if system else "") + "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        span.add_event("gen_ai.content.prompt", {"gen_ai.prompt": prompt_text})
        prompt_msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        set_openllmetry_prompt(span, prompt_msgs)

    tools = _build_tools(config.get("tools") or {})
    max_tokens = (config.get("model", {}).get("parameters") or {}).get(
        "max_tokens", 1024
    )
    conversation = list(messages)
    total_input = 0
    total_output = 0
    full_output = ""
    steps = 0

    try:
        while True:
            kwargs: dict[str, Any] = {
                "model": config["model"]["name"],
                "max_tokens": max_tokens,
                "messages": conversation,
            }
            if system:
                kwargs["system"] = system
            if tools:
                kwargs["tools"] = tools

            stream = client.messages.stream(**kwargs)
            async with stream as s:
                async for event in s:
                    if (
                        hasattr(event, "type")
                        and event.type == "content_block_delta"
                        and hasattr(event, "delta")
                        and getattr(event.delta, "type", None) == "text_delta"
                    ):
                        text = event.delta.text
                        full_output += text
                        yield {"type": "chunk", "text": text}

                final_msg = await s.get_final_message()

            total_input += final_msg.usage.input_tokens
            total_output += final_msg.usage.output_tokens

            if final_msg.stop_reason != "tool_use":
                break

            if steps >= _MAX_STEPS:
                raise RuntimeError(
                    f"Tool loop exceeded the maximum number of steps ({_MAX_STEPS})"
                )
            steps += 1

            conversation.append({"role": "assistant", "content": final_msg.content})
            tool_results = []
            for block in final_msg.content:
                if block.type != "tool_use":
                    continue
                handler_fn = tool_handlers.get(block.name)
                if not handler_fn or not callable(handler_fn):
                    raise ValueError(f'No handler registered for tool "{block.name}"')
                result = (
                    await handler_fn(block.input)
                    if _is_coroutine(handler_fn)
                    else handler_fn(block.input)
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    }
                )
            conversation.append({"role": "user", "content": tool_results})

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


def claude_messages(
    config_key: str,
    user_input: str,
    context: LDContext,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper: creates a handler and calls config(...).invoke()."""
    variables = kwargs.pop("variables", None)
    return config(
        key=config_key, handler=create_claude_messages_handler(), **kwargs
    ).invoke(user_input, context, variables=variables)
