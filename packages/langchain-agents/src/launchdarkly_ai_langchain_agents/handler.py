"""
LangChain Agents handler — uses LangGraph's create_react_agent / StateGraph.
Mirrors the TypeScript @launchdarkly/ai-langchain-agents handler.
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
    parse_template,
    set_ld_span_attributes,
    set_openllmetry_completion,
    set_openllmetry_prompt,
)

from .messages import to_lang_chain_messages

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

    lc_tools = importlib.import_module("langchain_core.tools")
    tool_fn = lc_tools.tool

    result = []
    for name, tool_cfg in config_tools.items():
        schema = tool_cfg.get("parameters") or {}

        async def _handler(_name: str = name, **kwargs: Any) -> str:
            fn = tool_handlers.get(_name)
            if not fn:
                raise ValueError(f'No handler registered for tool "{_name}"')
            res = await fn(kwargs)
            return str(res)

        t = tool_fn(
            name,
            _handler,
            description=tool_cfg.get("description", ""),
            args_schema=schema,
        )
        result.append(t)
    return result


def _extract_system_prompt(
    config: AiConfigRep,
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> str | None:
    """The system prompt, from ``instructions`` or the system-role config messages.

    ``history`` is accepted so callers can pass it uniformly, but it never
    reaches the system prompt: history is conversation, and it goes through the
    structured message path in ``_build_initial_messages`` (TESTING.md §1.11).
    """
    if config.get("instructions"):
        return parse_template(config["instructions"], variables)
    if config.get("messages"):
        sys_msgs = [m for m in config["messages"] if m.get("role") == "system"]
        if sys_msgs:
            return parse_template("\n".join(m["content"] for m in sys_msgs), variables)
    return None


def _config_conversation_turns(
    config: AiConfigRep,
    variables: dict[str, Any],
) -> list[dict[str, Any]]:
    """Non-system config conversation messages, template-applied, in canonical form."""
    return [
        {"role": m["role"], "content": parse_template(m["content"], variables)}
        for m in (config.get("messages") or [])
        if m.get("role") != "system"
    ]


def _build_initial_messages(
    config: AiConfigRep,
    user_input: str | None,
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> list[Any]:
    import importlib

    # With history, the whole conversation is composed as LangChain messages —
    # the framework's native input path. `config.instructions` / system messages
    # stay on the system prompt; history never becomes system-prompt text.
    if history:
        return to_lang_chain_messages(
            compose_history(
                history=history,
                user_input=user_input,
                config_messages=(
                    []
                    if config.get("instructions")
                    else _config_conversation_turns(config, variables)
                ),
            )
        )

    msgs_mod = importlib.import_module("langchain_core.messages")
    HumanMessage = msgs_mod.HumanMessage
    AIMessage = msgs_mod.AIMessage

    messages: list[Any] = []
    last_role: str | None = None
    for msg in _config_conversation_turns(config, variables):
        if msg["role"] == "user":
            messages.append(HumanMessage(msg["content"]))
        else:
            messages.append(AIMessage(msg["content"]))
        last_role = msg["role"]
    if last_role != "user":
        messages.append(HumanMessage(user_input or ""))
    return messages


def _message_text(message: Any) -> str:
    """Span-safe text for a message. Multimodal content contributes only its text
    parts, so an image never lands in a span attribute as a base64 payload."""
    content = getattr(message, "content", "")
    if isinstance(content, str | list):
        return content_to_text(content)
    return str(content)


def _record_prompt(
    span: Any,
    system_prompt: str | None,
    messages: list[Any],
) -> None:
    prompt_text = "\n".join(
        [
            *(["system: " + system_prompt] if system_prompt else []),
            *[
                f"{getattr(m, 'type', type(m).__name__)}: {_message_text(m)}"
                for m in messages
            ],
        ]
    )
    span.add_event("gen_ai.content.prompt", {"gen_ai.prompt": prompt_text})
    prompt_msgs: list[dict[str, str]] = []
    if system_prompt:
        prompt_msgs.append({"role": "system", "content": system_prompt})
    prompt_msgs.extend(
        [
            {
                "role": getattr(m, "type", type(m).__name__),
                "content": _message_text(m),
            }
            for m in messages
        ]
    )
    set_openllmetry_prompt(span, prompt_msgs)


def _make_default_chat_model(config: AiConfigRep) -> Any:
    """
    Instantiate the appropriate LangChain chat model based on ``config.provider.name``.
    Falls back to ``ChatOpenAI`` when the provider is not recognised.
    Requires the matching ``langchain-<provider>`` integration package to be installed.
    """
    import importlib

    provider = ((config.get("provider") or {}).get("name") or "openai").lower()
    model_name = (config.get("model") or {}).get("name", "")
    if provider == "anthropic":
        lc_anthropic = importlib.import_module("langchain_anthropic")
        return lc_anthropic.ChatAnthropic(
            model=model_name or "claude-3-5-sonnet-20241022"
        )
    lc_openai = importlib.import_module("langchain_openai")
    return lc_openai.ChatOpenAI(model=model_name or "gpt-4o")


def create_langchain_agents_handler(llm: Any = None) -> ProviderHandler:
    """Creates a ``ProviderHandler`` for LangChain via ``create_react_agent``."""
    tracer_name = "@launchdarkly/ai-langchain-agents"

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

        if _HAS_OTEL:
            span = trace.get_tracer(tracer_name).start_span("langchain.agent")
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

        system_prompt = _extract_system_prompt(config, vs)
        if config.get("outputFormat"):
            schema_instr = f"Respond with valid JSON matching this schema:\n{json.dumps(config['outputFormat'])}"
            system_prompt = (
                f"{system_prompt}\n\n{schema_instr}" if system_prompt else schema_instr
            )

        initial_messages = _build_initial_messages(config, user_input, vs, history)

        if span:
            _record_prompt(span, system_prompt, initial_messages)

        try:
            base_model = llm
            if base_model is None:
                base_model = _make_default_chat_model(config)

            langgraph_prebuilt = importlib.import_module("langgraph.prebuilt")
            create_react_agent = langgraph_prebuilt.create_react_agent
            tools = _build_agent_tools(config.get("tools") or {}, th)
            agent = create_react_agent(
                base_model,
                tools,
                **({"prompt": system_prompt} if system_prompt else {}),
            )
            result = await agent.ainvoke({"messages": initial_messages})

            msgs = (
                result.get("messages", [])
                if isinstance(result, dict)
                else getattr(result, "messages", [])
            )
            last_msg = msgs[-1] if msgs else None
            output = (
                (last_msg.content if isinstance(last_msg.content, str) else "")
                if last_msg
                else ""
            )

            total_input = sum(
                (getattr(m, "usage_metadata", None) or {}).get("input_tokens", 0)
                for m in msgs
            )
            total_output = sum(
                (getattr(m, "usage_metadata", None) or {}).get("output_tokens", 0)
                for m in msgs
            )

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

    return create_handler(("*", "agent"), _call_impl, _stream_impl)  # type: ignore[arg-type]


async def _stream_gen(
    llm: Any,
    config: AiConfigRep,
    user_input: str,
    tool_handlers: dict[str, Any],
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    import importlib

    tracer_name = "@launchdarkly/ai-langchain-agents"
    if _HAS_OTEL:
        span = trace.get_tracer(tracer_name).start_span("langchain.agent.stream")
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

    system_prompt = _extract_system_prompt(config, variables)
    initial_messages = _build_initial_messages(config, user_input, variables, history)

    if span:
        _record_prompt(span, system_prompt, initial_messages)

    try:
        base_model = llm
        if base_model is None:
            base_model = _make_default_chat_model(config)

        langgraph_prebuilt = importlib.import_module("langgraph.prebuilt")
        create_react_agent = langgraph_prebuilt.create_react_agent
        tools = _build_agent_tools(config.get("tools") or {}, tool_handlers)
        agent = create_react_agent(
            base_model,
            tools,
            **({"prompt": system_prompt} if system_prompt else {}),
        )

        total_input = 0
        total_output = 0
        full_output = ""

        async for step_state in agent.astream({"messages": initial_messages}):
            for step_messages in (
                step_state.values() if isinstance(step_state, dict) else []
            ):
                msgs = getattr(step_messages, "messages", None) or (
                    step_messages.get("messages", [])
                    if isinstance(step_messages, dict)
                    else []
                )
                for msg in msgs:
                    usage = getattr(msg, "usage_metadata", None) or {}
                    total_input += usage.get("input_tokens", 0)
                    total_output += usage.get("output_tokens", 0)
                    if getattr(msg, "type", None) == "ai":
                        text = msg.content if isinstance(msg.content, str) else ""
                        if text:
                            yield {"type": "chunk", "text": text}
                            full_output = text

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


def langchain_agents(
    config_key: str,
    user_input: str,
    context: LDContext,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper: creates a handler and calls config(...).invoke()."""
    variables = kwargs.pop("variables", None)
    return config(
        key=config_key, handler=create_langchain_agents_handler(), **kwargs
    ).invoke(user_input, context, variables=variables)
