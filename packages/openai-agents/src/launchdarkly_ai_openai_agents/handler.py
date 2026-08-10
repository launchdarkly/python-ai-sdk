"""
OpenAI Agents handler — uses the openai-agents SDK (``agents`` package).
Mirrors the TypeScript @launchdarkly/ai-openai-agents handler.
"""

from __future__ import annotations

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


def _build_agent_tools(
    config_tools: dict[str, Any],
    tool_handlers: dict[str, Any],
) -> list[Any]:
    import importlib

    agents_mod = importlib.import_module("agents")
    FunctionTool = agents_mod.FunctionTool

    result = []
    for name, tool_cfg in config_tools.items():
        # on_invoke_tool receives (ToolContext, json_args_str); decode and dispatch.
        async def _execute(_ctx: Any, args_str: str, _name: str = name) -> str:
            handler = tool_handlers.get(_name)
            if not handler or not callable(handler):
                raise ValueError(f'No handler registered for tool "{_name}"')
            try:
                args = json.loads(args_str) if args_str else {}
            except (json.JSONDecodeError, ValueError):
                args = {}
            res = await handler(args)
            return str(res)

        t = FunctionTool(
            name=name,
            description=tool_cfg.get("description", "") or "",
            params_json_schema=tool_cfg.get("parameters")
            or {"type": "object", "properties": {}},
            on_invoke_tool=_execute,
        )
        result.append(t)
    return result


def _parse_message_content(content: Any, variables: dict[str, Any]) -> Any:
    """Apply templates to text content while preserving structured blocks."""
    return parse_template(content, variables) if isinstance(content, str) else content


def _to_openai_agent_items(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map LaunchDarkly canonical turns to OpenAI Agents input items."""
    items: list[dict[str, Any]] = []
    for turn in turns:
        role = turn["role"]
        content = turn["content"]
        if role == "assistant":
            items.append({"role": "assistant", "content": content_to_text(content)})
            continue

        blocks = (
            content
            if isinstance(content, list)
            else [{"type": "text", "text": content}]
        )
        parts: list[dict[str, Any]] = []
        for block in blocks:
            if block.get("type") == "image":
                parts.append(
                    {"type": "input_image", "image_url": image_block_to_url(block)}
                )
            elif block.get("type") == "text":
                parts.append({"type": "input_text", "text": block.get("text", "")})
        items.append({"role": "user", "content": parts})
    return items


def _prompt_to_text(prompt: str | list[dict[str, Any]]) -> str:
    return prompt if isinstance(prompt, str) else json.dumps(prompt)


def _build_agent_and_prompt(
    config: AiConfigRep,
    user_input: str | None,
    tool_handlers: dict[str, Any],
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> tuple[Any, str | list[dict[str, Any]], str | None]:
    import importlib

    agents_mod = importlib.import_module("agents")
    Agent = agents_mod.Agent

    safe_input = user_input or ""
    instructions: str | None = None
    prompt: str | list[dict[str, Any]] = safe_input

    config_messages = config.get("messages") or []
    parsed_messages = [
        {
            **message,
            "content": _parse_message_content(message.get("content", ""), variables),
        }
        for message in config_messages
    ]

    if config.get("instructions"):
        instructions = parse_template(config["instructions"], variables)
    elif parsed_messages:
        system_msgs = [m for m in parsed_messages if m.get("role") == "system"]
        conv_msgs = [m for m in parsed_messages if m.get("role") != "system"]
        if system_msgs:
            instructions = "\n".join(content_to_text(m["content"]) for m in system_msgs)
        conv_history = "\n".join(content_to_text(m["content"]) for m in conv_msgs)
        prompt = f"{conv_history}\n\n{safe_input}" if conv_history else safe_input

    if history:
        turns = compose_history(
            history=history,
            user_input=user_input,
            config_messages=[
                message
                for message in parsed_messages
                if message.get("role") != "system"
            ],
        )
        prompt = _to_openai_agent_items(turns)

    tools = _build_agent_tools(config.get("tools") or {}, tool_handlers)

    agent = Agent(
        name="assistant",
        model=config.get("model", {}).get("name", "gpt-4o"),
        **({"instructions": instructions} if instructions else {}),
        **({"tools": tools} if tools else {}),
    )
    return agent, prompt, instructions


def _sum_usage(raw_responses: list[Any]) -> tuple[int, int, int]:
    input_tokens = sum(
        getattr(r.usage, "input_tokens", 0)
        for r in raw_responses
        if hasattr(r, "usage")
    )
    output_tokens = sum(
        getattr(r.usage, "output_tokens", 0)
        for r in raw_responses
        if hasattr(r, "usage")
    )
    total_tokens = sum(
        getattr(r.usage, "total_tokens", 0)
        for r in raw_responses
        if hasattr(r, "usage")
    )
    return input_tokens, output_tokens, total_tokens


def create_openai_agent_handler() -> ProviderHandler:
    """Creates a ``ProviderHandler`` for OpenAI via the openai-agents SDK."""
    tracer_name = "@launchdarkly/ai-openai-agents"

    async def _call_impl(
        config: AiConfigRep,
        user_input: str = "",
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        import importlib

        agents_mod = importlib.import_module("agents")
        Runner = agents_mod.Runner

        th = tool_handlers or {}
        vs = variables or {}

        if _HAS_OTEL:
            span = trace.get_tracer(tracer_name).start_span("openai.agent.run")
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.system", "openai")
            span.set_attribute(
                "gen_ai.request.model", config.get("model", {}).get("name", "")
            )
            set_ld_span_attributes(span, vs)
        else:
            span = None

        agent, prompt, instructions = _build_agent_and_prompt(
            config, user_input, th, vs, history
        )

        if span:
            serialized_prompt = _prompt_to_text(prompt)
            prompt_text = (
                f"system: {instructions}\n\n" if instructions else ""
            ) + serialized_prompt
            span.add_event("gen_ai.content.prompt", {"gen_ai.prompt": prompt_text})
            prompt_msgs: list[dict[str, str]] = []
            if instructions:
                prompt_msgs.append({"role": "system", "content": instructions})
            prompt_msgs.append({"role": "user", "content": serialized_prompt})
            set_openllmetry_prompt(span, prompt_msgs)

        try:
            result = await Runner.run(agent, prompt)
            final_output = result.final_output
            input_tokens, output_tokens, total_tokens = _sum_usage(result.raw_responses)

            if span:
                span.set_attribute(
                    "gen_ai.response.model", config.get("model", {}).get("name", "")
                )
                span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
                span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
                span.add_event(
                    "gen_ai.content.completion",
                    {
                        "gen_ai.completion": final_output
                        if isinstance(final_output, str)
                        else json.dumps(final_output)
                    },
                )
                set_openllmetry_completion(
                    span,
                    final_output
                    if isinstance(final_output, str)
                    else json.dumps(final_output),
                    {"input_tokens": input_tokens, "output_tokens": output_tokens},
                )
                span.set_status(SpanStatusCode.OK)
                span.end()

            output = (
                final_output if config.get("outputFormat") else str(final_output or "")
            )
            return {
                "output": output,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
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
            config, user_input, tool_handlers or {}, variables or {}, history
        )

    return create_handler(("OpenAI", "agent"), _call_impl, _stream_impl)  # type: ignore[arg-type]


async def _stream_gen(
    config: AiConfigRep,
    user_input: str,
    tool_handlers: dict[str, Any],
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    import importlib

    agents_mod = importlib.import_module("agents")
    Runner = agents_mod.Runner

    tracer_name = "@launchdarkly/ai-openai-agents"
    if _HAS_OTEL:
        span = trace.get_tracer(tracer_name).start_span("openai.agent.run.stream")
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.system", "openai")
        span.set_attribute(
            "gen_ai.request.model", config.get("model", {}).get("name", "")
        )
        set_ld_span_attributes(span, variables)
    else:
        span = None

    agent, prompt, instructions = _build_agent_and_prompt(
        config, user_input, tool_handlers, variables, history
    )

    if span:
        serialized_prompt = _prompt_to_text(prompt)
        prompt_text = (
            f"system: {instructions}\n\n" if instructions else ""
        ) + serialized_prompt
        span.add_event("gen_ai.content.prompt", {"gen_ai.prompt": prompt_text})
        prompt_msgs: list[dict[str, str]] = []
        if instructions:
            prompt_msgs.append({"role": "system", "content": instructions})
        prompt_msgs.append({"role": "user", "content": serialized_prompt})
        set_openllmetry_prompt(span, prompt_msgs)

    try:
        streamed = Runner.run_streamed(agent, prompt)
        full_output = ""

        async for event in streamed.stream_events():
            if event.type == "raw_response_event":
                raw = getattr(event, "data", None)
                if (
                    raw is not None
                    and getattr(raw, "type", None) == "response.output_text.delta"
                ):
                    delta = getattr(raw, "delta", "")
                    if isinstance(delta, str) and delta:
                        yield {"type": "chunk", "text": delta}
                        full_output += delta

        input_tokens, output_tokens, total_tokens = _sum_usage(streamed.raw_responses)
        final_output = streamed.final_output or full_output

        if span:
            span.set_attribute(
                "gen_ai.response.model", config.get("model", {}).get("name", "")
            )
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
            span.add_event(
                "gen_ai.content.completion",
                {
                    "gen_ai.completion": str(final_output)
                    if isinstance(final_output, str)
                    else json.dumps(final_output)
                },
            )
            set_openllmetry_completion(
                span,
                str(final_output)
                if isinstance(final_output, str)
                else json.dumps(final_output),
                {"input_tokens": input_tokens, "output_tokens": output_tokens},
            )
            span.set_status(SpanStatusCode.OK)
            span.end()

        yield {
            "type": "done",
            "output": str(final_output),
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }

    except Exception as exc:
        if span:
            span.record_exception(exc)
            span.set_status(SpanStatusCode.ERROR, str(exc))
            span.end()
        raise


def openai_agents(
    config_key: str,
    user_input: str,
    context: LDContext,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper: creates a handler and calls config(...).invoke()."""
    variables = kwargs.pop("variables", None)
    return config(
        key=config_key, handler=create_openai_agent_handler(), **kwargs
    ).invoke(user_input, context, variables=variables)
