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
    SpanMessage,
    SpanMessagePart,
    config,
    create_handler,
    create_run_usage,
    end_span_once,
    lang_chain_span_messages,
    lang_chain_span_usage,
    parse_template,
    set_input_content_attributes,
    set_output_content_attributes,
)

from .spans import (
    build_span_callbacks,
    fail_span,
    finish_root_span,
    mark_ok,
    parent_context_of,
    start_root_span,
    succeed_span,
    to_tool_definitions,
)

try:
    from opentelemetry import trace  # noqa: F401
    from opentelemetry.trace import StatusCode as SpanStatusCode  # noqa: F401

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


def _format_history(history: list[dict[str, Any]] | None) -> str | None:
    if not history:
        return None
    lines = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "Conversation History:\n\n" + "\n".join(lines)


def _extract_system_prompt(
    config: AiConfigRep,
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> str | None:
    system_prompt: str | None = None
    if config.get("instructions"):
        system_prompt = parse_template(config["instructions"], variables)
    elif config.get("messages"):
        sys_msgs = [m for m in config["messages"] if m.get("role") == "system"]
        if sys_msgs:
            system_prompt = parse_template(
                "\n".join(m["content"] for m in sys_msgs), variables
            )

    history_text = _format_history(history)
    if history_text:
        system_prompt = (
            f"{system_prompt}\n\n{history_text}" if system_prompt else history_text
        )

    return system_prompt


def _build_initial_messages(
    config: AiConfigRep,
    user_input: str,
    variables: dict[str, Any],
) -> list[Any]:
    import importlib

    msgs_mod = importlib.import_module("langchain_core.messages")
    HumanMessage = msgs_mod.HumanMessage
    AIMessage = msgs_mod.AIMessage

    messages: list[Any] = []
    last_role: str | None = None
    if config.get("messages"):
        for msg in config["messages"]:
            if msg.get("role") == "system":
                continue
            content = parse_template(msg["content"], variables)
            if msg["role"] == "user":
                messages.append(HumanMessage(content))
            else:
                messages.append(AIMessage(content))
            last_role = msg["role"]
    if last_role != "user":
        messages.append(HumanMessage(user_input or ""))
    return messages


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


def _run_usage_from_messages(messages: list[Any]) -> Any:
    """Sums ``usage_metadata`` over a run's messages, the same set of numbers the callbacks see
    from the other side.

    Only ``AIMessage`` carries usage. Summing here rather than trusting the callbacks' own total
    matters when there is a real ``result``/stepped state to read: it is the same path the
    TypeScript handler takes, and it keeps the two run-usage totals (this one, and the callbacks'
    fallback used only on the failure path) computed the same way.
    """
    run_usage = create_run_usage()
    for msg in messages:
        usage = getattr(msg, "usage_metadata", None)
        if usage:
            run_usage.add(lang_chain_span_usage(usage))
    return run_usage


def create_langchain_agents_handler(
    llm: Any = None, *, capture_content: bool = False
) -> ProviderHandler:
    """Creates a ``ProviderHandler`` for LangChain via ``create_react_agent``.

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

        system_prompt = _extract_system_prompt(config, vs, history)
        if config.get("outputFormat"):
            schema_instr = f"Respond with valid JSON matching this schema:\n{json.dumps(config['outputFormat'])}"
            system_prompt = (
                f"{system_prompt}\n\n{schema_instr}" if system_prompt else schema_instr
            )

        initial_messages = _build_initial_messages(config, user_input, vs)
        if capture_content:
            set_input_content_attributes(
                span,
                capture_content,
                system_instructions=system_prompt,
                messages=lang_chain_span_messages(initial_messages)[1],
            )

        span_callbacks = build_span_callbacks(
            config,
            parent,
            capture_content,
            to_tool_definitions(config.get("tools") or {}),
        )

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
            result = await agent.ainvoke(
                {"messages": initial_messages},
                config={"callbacks": span_callbacks.callbacks},
            )

            msgs = (
                result.get("messages", [])
                if isinstance(result, dict)
                else getattr(result, "messages", [])
            )
            run_usage = _run_usage_from_messages(msgs)

            last_msg = msgs[-1] if msgs else None
            output = (
                (last_msg.content if isinstance(last_msg.content, str) else "")
                if last_msg
                else ""
            )

            set_output_content_attributes(
                span,
                capture_content,
                [
                    SpanMessage(
                        role="assistant",
                        parts=[SpanMessagePart(type="text", content=output)],
                    )
                ],
            )
            finish_root_span(span, config, run_usage.total)
            succeed_span(span)

            return {
                "output": output,
                "usage": {
                    "input_tokens": run_usage.total.input,
                    "output_tokens": run_usage.total.output,
                },
            }

        except Exception as exc:
            span_callbacks.close_open_spans(exc)
            # There is no `result` to sum, so the run total comes from the callbacks, which saw
            # every turn that did complete. Those tokens were billed and the root is the only span
            # a config-scoped cost query can find them on.
            if span_callbacks.run_usage.reported:
                finish_root_span(span, config, span_callbacks.run_usage.total)
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
            llm,
            config,
            user_input,
            tool_handlers or {},
            variables or {},
            history,
            capture_content=capture_content,
        )

    return create_handler(("*", "agent"), _call_impl, _stream_impl)  # type: ignore[arg-type]


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
    the root span, and any ``chat``/``execute_tool`` span LangChain's end callback never fired for,
    is never ended, so it is never exported.

    ``ended`` stops the success, failure and abandonment paths from ending the same span twice.
    """
    import importlib

    span = start_root_span(config, variables)
    parent = parent_context_of(span)

    system_prompt = _extract_system_prompt(config, variables, history)
    initial_messages = _build_initial_messages(config, user_input, variables)
    if capture_content:
        set_input_content_attributes(
            span,
            capture_content,
            system_instructions=system_prompt,
            messages=lang_chain_span_messages(initial_messages)[1],
        )

    span_callbacks = build_span_callbacks(
        config, parent, capture_content, to_tool_definitions(config.get("tools") or {})
    )
    ended: set[int] = set()

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

        run_usage = create_run_usage()
        full_output = ""

        # agent.astream() yields state updates per graph step: { [node_name]: { messages: [...] } }
        async for step_state in agent.astream(
            {"messages": initial_messages},
            config={"callbacks": span_callbacks.callbacks},
        ):
            for step_messages in (
                step_state.values() if isinstance(step_state, dict) else []
            ):
                msgs = getattr(step_messages, "messages", None) or (
                    step_messages.get("messages", [])
                    if isinstance(step_messages, dict)
                    else []
                )
                for msg in msgs:
                    usage = getattr(msg, "usage_metadata", None)
                    if usage:
                        run_usage.add(lang_chain_span_usage(usage))
                    if getattr(msg, "type", None) == "ai":
                        text = msg.content if isinstance(msg.content, str) else ""
                        if text:
                            yield {"type": "chunk", "text": text}
                            full_output = text

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
        finish_root_span(span, config, run_usage.total)
        mark_ok(span)
        end_span_once(span, ended)

        yield {
            "type": "done",
            "output": full_output,
            "usage": {
                "input_tokens": run_usage.total.input,
                "output_tokens": run_usage.total.output,
            },
        }

    except Exception as exc:
        span_callbacks.close_open_spans(exc)
        if span_callbacks.run_usage.reported:
            finish_root_span(span, config, span_callbacks.run_usage.total)
        fail_span(span, exc, ended)
        raise
    finally:
        # A no-op on the success and failure paths, because both already ended their spans through
        # `ended`. On abandonment it is the only chance to close the tree, including any chat or
        # tool span whose LangChain end callback never fired, and to report what the completed
        # turns already cost. An abandoned span is left UNSET rather than ERROR: stopping early is
        # a normal thing for a consumer to do, and LaunchDarkly's own metrics record neither a
        # success nor an error for it.
        if span is not None and id(span) not in ended:
            span_callbacks.abandon_open_spans(ended)
            if span_callbacks.run_usage.reported:
                finish_root_span(span, config, span_callbacks.run_usage.total)
        end_span_once(span, ended, abandoned=True)


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
