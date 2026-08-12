from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

from launchdarkly_ai_server import (
    AiConfigRep,
    LDContext,
    ProviderHandler,
    RunUsage,
    SpanMessage,
    SpanMessagePart,
    config,
    create_handler,
    create_run_usage,
    end_span_once,
    parse_template,
    set_input_content_attributes,
    set_output_content_attributes,
    set_tool_call_content_attributes,
)

from .spans import (
    fail_span,
    finish_model_span,
    finish_reason_of,
    finish_root_span,
    mark_ok,
    model_name,
    parent_context_of,
    set_response_output_content,
    split_input_messages,
    start_model_span,
    start_root_span,
    start_tool_span,
    succeed_span,
    to_span_usage,
    to_tool_definitions,
)


def _build_tools(config_tools: dict[str, Any]) -> list[dict[str, Any]]:
    # Not filtered to the tools that have a registered handler, unlike the TypeScript SDK. That
    # difference predates this span work and changes what the model is offered, not what the span
    # reports, so it stays as it is: the catalog recorded on the span is the catalog actually sent.
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
    user_input: str,
    variables: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if config.get("messages"):
        msgs = [
            {"role": m["role"], "content": parse_template(m["content"], variables)}
            for m in config["messages"]
        ]
        if history:
            for msg in history:
                role = msg.get("role", "user")
                if role in ("user", "assistant"):
                    msgs.append({"role": role, "content": msg.get("content", "")})
        if user_input and (not msgs or msgs[-1].get("role") != "user"):
            msgs.append({"role": "user", "content": user_input})
        return msgs
    instructions = parse_template(config.get("instructions") or "", variables)
    result: list[dict[str, Any]] = []
    if instructions:
        result.append({"role": "system", "content": instructions})
    if history:
        for msg in history:
            role = msg.get("role", "user")
            if role in ("user", "assistant"):
                result.append({"role": role, "content": msg.get("content", "")})
    result.append({"role": "user", "content": user_input or ""})
    return result


def _json_schema_format(schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "json_schema", "name": "output", "schema": schema, "strict": False}


def _is_coroutine(fn: Any) -> bool:
    return asyncio.iscoroutinefunction(fn)


_MAX_STEPS = 10


async def _run_model_turn(
    client: Any,
    config: AiConfigRep,
    params: dict[str, Any],
    tool_definitions: list[Any],
    *,
    capture_content: bool,
    parent: Any,
    run_usage: RunUsage,
) -> Any:
    """Runs one provider turn under its own ``chat`` child span.

    Written before the call, so an in-flight or failed turn still shows what it was asked. Returns
    the raw provider response so the caller can inspect its output items.
    """
    model_span = start_model_span(config, parent)
    # Everything that touches this span sits inside the try, including the content writes on both
    # sides of the call. Serialising conversation content raises on anything that is not
    # JSON-serialisable, and a raise outside the guard would leave this span open forever: only the
    # root gets failed, and nothing else knows the chat span exists.
    try:
        if capture_content:
            system_instructions, messages = split_input_messages(params["input"])
            set_input_content_attributes(
                model_span,
                capture_content,
                system_instructions=system_instructions,
                messages=messages,
                tool_definitions=tool_definitions,
            )

        response = await client.responses.create(**params)

        # Accounting before anything that can raise. The provider has already billed this turn, so a
        # later failure while serialising content must not lose the tokens: the root is the only span
        # a config-scoped cost query can read them from. `to_span_usage` of an absent bag is still a
        # real object, so a turn that completed without reported usage counts as reported: the call
        # happened, whatever the provider said.
        usage = to_span_usage(getattr(response, "usage", None))
        run_usage.add(usage)

        set_response_output_content(model_span, capture_content, response)
        finish_reason = finish_reason_of(response)
        response_model = getattr(response, "model", None) or model_name(config)
        finish_model_span(model_span, response_model, usage, finish_reason)
    except Exception as exc:
        fail_span(model_span, exc)
        raise
    return response


def create_openai_messages_handler(*, capture_content: bool = False) -> ProviderHandler:
    """
    Creates a ``ProviderHandler`` for OpenAI (responses API).
    Requires ``openai`` to be installed as a peer dependency.

    Set *capture_content* to put prompts, model output, tool arguments and tool results on the
    emitted spans. It defaults to off. Conversation content is PII, so a run emits only metadata,
    meaning models, token counts, timings and tool names, until a caller asks for more.
    """
    import importlib

    openai_mod = importlib.import_module("openai")
    client = openai_mod.AsyncOpenAI()

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

        # Declared out here, not inside the `try`, so the failure path can still report the tokens
        # the run had already spent.
        run_usage = create_run_usage()

        try:
            tools = _build_tools(config.get("tools") or {})
            input_messages = _build_input_messages(config, user_input, vs, history)
            tool_definitions = to_tool_definitions(tools)

            root_system, root_messages = split_input_messages(input_messages)
            set_input_content_attributes(
                span,
                capture_content,
                system_instructions=root_system,
                messages=root_messages,
            )

            params: dict[str, Any] = {
                "model": config["model"]["name"],
                "input": input_messages,
            }
            if tools:
                params["tools"] = tools
            if config.get("outputFormat"):
                params["text"] = {"format": _json_schema_format(config["outputFormat"])}

            response = await _run_model_turn(
                client,
                config,
                params,
                tool_definitions,
                capture_content=capture_content,
                parent=parent,
                run_usage=run_usage,
            )

            steps = 0
            while True:
                tool_calls = [
                    item
                    for item in (getattr(response, "output", None) or [])
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
                    tool_span = start_tool_span(tc.name, tc.call_id, parent)
                    set_tool_call_content_attributes(
                        tool_span, capture_content, arguments=tc.arguments
                    )
                    try:
                        args = json.loads(tc.arguments)
                        handler_fn = th.get(tc.name)
                        if not handler_fn or not callable(handler_fn):
                            raise ValueError(
                                f'No handler registered for tool "{tc.name}"'
                            )
                        result = (
                            await handler_fn(args)
                            if _is_coroutine(handler_fn)
                            else handler_fn(args)
                        )
                        # Inside the try on purpose. Serialising a tool result can raise, most easily
                        # when capture_content is on and the result is not JSON-serialisable, and a
                        # raise out here would leave this span open: nothing else knows it exists.
                        set_tool_call_content_attributes(
                            tool_span, capture_content, result=result
                        )
                        succeed_span(tool_span)
                    except Exception as exc:
                        fail_span(tool_span, exc)
                        raise
                    tool_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": tc.call_id,
                            "output": str(result),
                        }
                    )

                response = await _run_model_turn(
                    client,
                    config,
                    {
                        "model": config["model"]["name"],
                        "previous_response_id": response.id,
                        "input": tool_outputs,
                    },
                    tool_definitions,
                    capture_content=capture_content,
                    parent=parent,
                    run_usage=run_usage,
                )

            output = getattr(response, "output_text", None) or ""
            set_output_content_attributes(
                span, capture_content, _final_output_messages(output)
            )
            response_model = getattr(response, "model", None) or model_name(config)
            finish_root_span(span, response_model, run_usage.total)
            succeed_span(span)
            # Cache keys are deliberately omitted: OpenAI's input already includes them, and
            # `parse_usage` would otherwise fold them in a second time.
            return {
                "output": output,
                "usage": {
                    "input_tokens": run_usage.total.input,
                    "output_tokens": run_usage.total.output,
                },
            }
        except Exception as exc:
            # Report what the turns that did complete already cost. Falls back to the requested
            # model name rather than tracking the last answering model, matching the TypeScript
            # SDK's blocking failure path.
            if run_usage.reported:
                finish_root_span(span, model_name(config), run_usage.total)
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
            client,
            config,
            user_input,
            tool_handlers or {},
            variables or {},
            history,
            capture_content=capture_content,
        )

    return create_handler(("OpenAI", "messages"), _call_impl, _stream_impl)  # type: ignore[arg-type]


def _final_output_messages(output: str) -> list[SpanMessage]:
    return [
        SpanMessage(
            role="assistant", parts=[SpanMessagePart(type="text", content=output)]
        )
    ]


async def _stream_gen(
    client: Any,
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
    """
    span = start_root_span(config, variables)
    parent = parent_context_of(span)

    ended: set[int] = set()
    open_model_span: Any = None
    # Tracked for the same reason as the model span: a BaseException raised while a tool runs skips
    # `except Exception` entirely, and `finally` is then the only code that can close this span.
    open_tool_span: Any = None
    # Outside the try, so the failure and abandonment paths can still report the spend and the
    # model that answered.
    run_usage = create_run_usage()
    last_response_model = model_name(config)

    try:
        tools = _build_tools(config.get("tools") or {})
        input_messages = _build_input_messages(config, user_input, variables, history)
        tool_definitions = to_tool_definitions(tools)

        root_system, root_messages = split_input_messages(input_messages)
        set_input_content_attributes(
            span,
            capture_content,
            system_instructions=root_system,
            messages=root_messages,
        )

        full_output = ""
        previous_response_id: str | None = None
        current_input: Any = input_messages
        steps = 0

        while True:
            model_span = start_model_span(config, parent)
            open_model_span = model_span
            if capture_content:
                turn_system, turn_messages = split_input_messages(current_input)
                set_input_content_attributes(
                    model_span,
                    capture_content,
                    system_instructions=turn_system,
                    messages=turn_messages,
                    tool_definitions=tool_definitions,
                )

            stream_params: dict[str, Any] = {
                "model": config["model"]["name"],
                "input": current_input,
            }
            if previous_response_id:
                stream_params["previous_response_id"] = previous_response_id
            # Tools are forwarded on every streaming turn, not only the first, unlike the blocking
            # path and unlike the TypeScript SDK. That difference predates this span work and
            # changes what the model is offered, not what the span reports, so it stays as it is.
            if tools:
                stream_params["tools"] = tools

            try:
                stream = client.responses.stream(**stream_params)
                async with stream as s:
                    async for event in s:
                        if getattr(event, "type", None) == "response.output_text.delta":
                            text = getattr(event, "delta", "")
                            full_output += text
                            yield {"type": "chunk", "text": text}

                    final_resp = await s.get_final_response()

                last_response_model = getattr(final_resp, "model", None) or model_name(
                    config
                )
                # Accumulated before anything that can raise. The provider has already billed this
                # turn, so a later content failure must not report the run as having spent less than
                # it did.
                usage = to_span_usage(getattr(final_resp, "usage", None))
                run_usage.add(usage)
                # Inside the guard for the same reason as the blocking path: a raise out here would
                # leave this span for `finally` to end as abandoned, which reads as a consumer who
                # walked away rather than as the failure it is.
                set_response_output_content(model_span, capture_content, final_resp)
                finish_reason = finish_reason_of(final_resp)
                finish_model_span(model_span, last_response_model, usage, finish_reason)
                open_model_span = None
            except Exception as exc:
                fail_span(model_span, exc, ended)
                open_model_span = None
                raise

            tool_calls = [
                item
                for item in (getattr(final_resp, "output", None) or [])
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
                tool_span = start_tool_span(tc.name, tc.call_id, parent)
                open_tool_span = tool_span
                set_tool_call_content_attributes(
                    tool_span, capture_content, arguments=tc.arguments
                )
                try:
                    args = json.loads(tc.arguments)
                    handler_fn = tool_handlers.get(tc.name)
                    if not handler_fn or not callable(handler_fn):
                        raise ValueError(f'No handler registered for tool "{tc.name}"')
                    result = (
                        await handler_fn(args)
                        if _is_coroutine(handler_fn)
                        else handler_fn(args)
                    )
                    # Inside the try on purpose. Serialising a tool result can raise, most easily
                    # when capture_content is on and the result is not JSON-serialisable, and a
                    # raise out here would leave this span open: nothing else knows it exists.
                    set_tool_call_content_attributes(
                        tool_span, capture_content, result=result
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
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": tc.call_id,
                        "output": str(result),
                    }
                )
            current_input = tool_outputs

        set_output_content_attributes(
            span, capture_content, _final_output_messages(full_output)
        )
        finish_root_span(span, last_response_model, run_usage.total)
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
        if open_model_span is not None:
            fail_span(open_model_span, exc, ended)
        if run_usage.reported:
            finish_root_span(span, last_response_model, run_usage.total)
        fail_span(span, exc, ended)
        raise
    finally:
        # A no-op on the success and failure paths, because both already ended their spans through
        # `ended`. On abandonment it is the only chance to close the tree, and to report what the
        # completed turns already cost. An abandoned span is left UNSET rather than ERROR: stopping
        # early is a normal thing for a consumer to do, and LaunchDarkly's own metrics record
        # neither a success nor an error for it, so ERROR would put two dashboards in disagreement.
        # Tool span first: it is a child, and a reader following the tree should not meet a closed
        # parent above an open child.
        if open_tool_span is not None:
            end_span_once(open_tool_span, ended, abandoned=True)
        if open_model_span is not None:
            end_span_once(open_model_span, ended, abandoned=True)
        if span is not None and id(span) not in ended and run_usage.reported:
            finish_root_span(span, last_response_model, run_usage.total)
        end_span_once(span, ended, abandoned=True)


def openai_messages(
    config_key: str,
    user_input: str,
    context: LDContext,
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
        handler=create_openai_messages_handler(capture_content=capture_content),
        **kwargs,
    ).invoke(user_input, context, variables=variables)
