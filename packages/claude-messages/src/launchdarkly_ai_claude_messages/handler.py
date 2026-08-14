from __future__ import annotations

import asyncio
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
    end_span_once,
    end_unfinished_spans,
    parse_template,
    set_input_content_attributes,
    set_output_content_attributes,
    set_tool_call_content_attributes,
    to_semconv_finish_reason,
)

from .spans import (
    RawRunUsage,
    fail_span,
    finish_model_span,
    finish_root_span,
    mark_ok,
    parent_context_of,
    raw_usage_of,
    start_model_span,
    start_root_span,
    start_tool_span,
    succeed_span,
    to_span_messages,
    to_span_parts,
    to_tool_definitions,
)

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
    *,
    capture_content: bool = False,
    parent: Any = None,
    run_usage: RawRunUsage,
) -> tuple[str, dict[str, Any]]:
    """Runs the model, then its tools, until the model stops asking for tools.

    Emits one ``chat`` span per model turn and one ``execute_tool`` span per tool call. Both are
    parented to *parent*, which is the root's context, so tool spans are siblings of the ``chat``
    span rather than children of it.

    *run_usage* is owned by the caller rather than created here, so a turn that raises does not take
    the run's spend with it.
    """
    # Not filtered to the tools that have a registered handler, unlike the TypeScript SDK. That
    # difference predates this span work and changes what the model is offered, not what the span
    # reports, so it stays as it is: the catalog recorded below is the catalog actually sent.
    tools = _build_tools(config.get("tools") or {})
    max_tokens = (config.get("model", {}).get("parameters") or {}).get(
        "max_tokens", 1024
    )
    conversation = list(messages)
    output = ""
    steps = 0

    tool_definitions = to_tool_definitions(tools)

    # Held so the `finally` below can end whatever is still open. `except Exception` never sees an
    # asyncio.CancelledError, which is a BaseException, so a timeout or a task.cancel() would
    # otherwise strand every span this loop opened.
    open_model_span: Any = None
    open_tool_span: Any = None

    try:
        while True:
            model_span = start_model_span(config, parent)
            open_model_span = model_span

            kwargs: dict[str, Any] = {
                "model": config["model"]["name"],
                "max_tokens": max_tokens,
                "messages": conversation,
            }
            if system:
                kwargs["system"] = system
            if tools:
                kwargs["tools"] = tools

            # Everything that touches this span stays inside one guard. Serialising conversation content
            # raises on anything that is not JSON-serialisable, and a raise out here would leave the chat
            # span open: the blocking path has no `finally` that could recover it, so the turn would never
            # be exported and the run would show a model call that left no trace.
            try:
                # Written before the call, so an in-flight or failed turn still shows what it was asked.
                # `conversation` grows with each turn, which is what makes a `chat` span self-contained.
                # Inside the guard, because serialising it raises on anything that is not
                # JSON-serialisable and a raise out here would leave this span open.
                if capture_content:
                    set_input_content_attributes(
                        model_span,
                        capture_content,
                        system_instructions=system,
                        messages=to_span_messages(conversation),
                        tool_definitions=tool_definitions,
                    )
                resp = await client.messages.create(**kwargs)

                raw_usage = raw_usage_of(getattr(resp, "usage", None))
                # Accumulated before anything that can raise. Anthropic has already billed this turn, so
                # a later content failure must not report the run as having spent less than it did.
                run_usage.add_turn(raw_usage)
                # Mapped once, into a local, so the span attribute and the output message cannot disagree:
                # Anthropic's `end_turn` is semconv's `stop`, and this handler is not the only one whose
                # spans a consumer groups by that value.
                finish_reason = to_semconv_finish_reason(
                    getattr(resp, "stop_reason", None)
                )
                if capture_content:
                    set_output_content_attributes(
                        model_span,
                        capture_content,
                        [
                            SpanMessage(
                                role="assistant",
                                parts=to_span_parts(resp.content),
                                finish_reason=finish_reason,
                            )
                        ],
                    )
                finish_model_span(model_span, config, raw_usage, finish_reason)
                open_model_span = None
            except Exception as exc:
                fail_span(model_span, exc)
                open_model_span = None
                raise

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
                tool_span = start_tool_span(block.name, block.id, parent)
                open_tool_span = tool_span
                set_tool_call_content_attributes(
                    tool_span, capture_content, arguments=block.input
                )
                try:
                    handler_fn = tool_handlers.get(block.name)
                    if not handler_fn or not callable(handler_fn):
                        raise ValueError(
                            f'No handler registered for tool "{block.name}"'
                        )
                    result = (
                        await handler_fn(block.input)
                        if _is_coroutine(handler_fn)
                        else handler_fn(block.input)
                    )
                    # Inside the try on purpose. Serialising a tool result can raise, most easily when
                    # capture_content is on and the result is not JSON-serialisable, and a raise out
                    # here would leave this span open forever: nothing else knows it exists.
                    set_tool_call_content_attributes(
                        tool_span, capture_content, result=result
                    )
                    succeed_span(tool_span)
                    open_tool_span = None
                except Exception as exc:
                    fail_span(tool_span, exc)
                    open_tool_span = None
                    raise
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    }
                )
            conversation.append({"role": "user", "content": tool_results})

    finally:
        # Not an `except`: the whole point is the unwind an `except Exception` cannot see. A
        # cancelled turn still leaves its spans exportable, marked and left at UNSET, because
        # nothing failed. The caller went away.
        end_unfinished_spans(open_tool_span, open_model_span)

    return output, run_usage.total


def _is_coroutine(fn: Any) -> bool:
    return asyncio.iscoroutinefunction(fn)


_MAX_STEPS = 10


def create_claude_messages_handler(*, capture_content: bool = False) -> ProviderHandler:
    """Creates a ``ProviderHandler`` for Anthropic Claude (messages API).

    Requires ``anthropic`` to be installed as a peer dependency.

    Set *capture_content* to put prompts, model output, tool arguments and tool results on the
    emitted spans. It defaults to off. Conversation content is PII, so a run emits only metadata,
    meaning models, token counts, timings and tool names, until a caller asks for more. Turning this
    on sends the text of every request and response to whatever collector the SDK is pointed at.
    """
    import importlib

    anthropic_mod = importlib.import_module("anthropic")
    client = anthropic_mod.AsyncAnthropic()

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
        # Cleared by whichever path ends the root, so the `finally` can tell an open root from a
        # closed one without asking the span. A mock span answers `is_recording()` truthily, and the
        # test suites are built on mock spans.
        open_root_span: Any = span

        messages, system = _build_messages(config, user_input, vs, history=history)

        # Outside the try, so the failure path can still report the spend of the turns that
        # completed before it.
        run_usage = RawRunUsage()
        try:
            # Inside the guard, because serialising the prompt raises on anything that is not
            # JSON-serialisable and a raise out here would leave the root open: never ended, never
            # exported, and the run gone from AI Config Monitoring with the feature_flag event on it.
            set_input_content_attributes(
                span,
                capture_content,
                system_instructions=system,
                messages=to_span_messages(messages),
            )
            output, usage = await _run_tool_loop(
                client,
                config,
                messages,
                system,
                th,
                capture_content=capture_content,
                parent=parent,
                run_usage=run_usage,
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
            finish_root_span(span, config, usage)
            succeed_span(span)
            open_root_span = None
            # Raw usage, cache fields intact. `parse_usage` folds them exactly once; handing back a
            # pre-folded figure alongside the fields would count the cache twice.
            return {"output": output, "usage": usage}
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
            # task.cancel() never reaches the clause above. Without this the root is stranded, and the
            # root is the only span carrying the feature_flag event and the launchdarkly.* attributes,
            # so the whole run would vanish from AI Config Monitoring rather than show as incomplete.
            if open_root_span is not None and run_usage.reported:
                # The turns that completed were billed, the same reason the failure path reports them.
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
            client,
            config,
            user_input,
            tool_handlers or {},
            variables or {},
            history,
            capture_content=capture_content,
        )

    return create_handler(("Anthropic", "messages"), _call_impl, _stream_impl)  # type: ignore[arg-type]


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

    ``ended`` stops the success, failure and abandonment paths from ending the same span twice.
    """
    span = start_root_span(config, variables)
    parent = parent_context_of(span)

    messages, system = _build_messages(
        config, user_input, variables, include_output_format=False, history=history
    )

    tools = _build_tools(config.get("tools") or {})
    tool_definitions = to_tool_definitions(tools)
    max_tokens = (config.get("model", {}).get("parameters") or {}).get(
        "max_tokens", 1024
    )
    conversation = list(messages)
    full_output = ""
    steps = 0

    ended: set[int] = set()
    open_model_span: Any = None
    # Tracked for the same reason as the model span: a BaseException raised while a tool runs skips
    # `except Exception` entirely, and `finally` is then the only code that can close this span.
    open_tool_span: Any = None
    # Outside the try, so the failure and abandonment paths can still report the spend.
    run_usage = RawRunUsage()

    # Distinguishes the two teardown reasons. A consumer that stops reading abandoned the
    # stream; a CancelledError means something cancelled the run, usually a timeout, and the
    # consumer chose nothing. The blocking path already tells these apart.
    cancelled = False
    try:
        # Inside the guard, because serialising the prompt raises on anything that is not
        # JSON-serialisable. A raise out here would leave the root open with the `finally` never
        # entered, so the run would vanish from AI Config Monitoring with its feature_flag event.
        set_input_content_attributes(
            span,
            capture_content,
            system_instructions=system,
            messages=to_span_messages(messages),
        )
        while True:
            model_span = start_model_span(config, parent)
            open_model_span = model_span

            kwargs: dict[str, Any] = {
                "model": config["model"]["name"],
                "max_tokens": max_tokens,
                "messages": conversation,
            }
            if system:
                kwargs["system"] = system
            if tools:
                kwargs["tools"] = tools

            try:
                # Inside the guard that calls fail_span on this span, the same as the blocking path
                # and the same as the output write below. Serialising the conversation raises on
                # anything that is not JSON-serialisable, and a raise out here reached the outer
                # handler: the root was marked ERROR while this chat span was left open, so the
                # `finally` ended it as abandoned and the trace showed a failed run whose model call
                # merely stopped.
                if capture_content:
                    set_input_content_attributes(
                        model_span,
                        capture_content,
                        system_instructions=system,
                        messages=to_span_messages(conversation),
                        tool_definitions=tool_definitions,
                    )
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

                raw_usage = raw_usage_of(getattr(final_msg, "usage", None))
                finish_reason = to_semconv_finish_reason(
                    getattr(final_msg, "stop_reason", None)
                )
                # Accumulated before anything that can raise, the same way the blocking path does
                # it. Anthropic has already billed this turn, so a later content failure must not
                # report the run as having spent less than it did.
                run_usage.add_turn(raw_usage)
                # The content write and the finish stay inside this guard, so a serialisation failure
                # fails the chat span the way the blocking path does. Outside it, the raise reached
                # the outer `finally` with `open_model_span` still set, and the chat span was ended as
                # abandoned and left UNSET while the root was marked ERROR: one turn described as a
                # consumer walking away and a failure at the same time.
                if capture_content:
                    set_output_content_attributes(
                        model_span,
                        capture_content,
                        [
                            SpanMessage(
                                role="assistant",
                                parts=to_span_parts(final_msg.content),
                                finish_reason=finish_reason,
                            )
                        ],
                    )
                # finish_model_span ends the span. Clearing open_model_span is what stops the
                # `finally` from ending it a second time.
                finish_model_span(model_span, config, raw_usage, finish_reason)
                open_model_span = None
            except Exception as exc:
                fail_span(model_span, exc, ended)
                open_model_span = None
                raise

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
                tool_span = start_tool_span(block.name, block.id, parent)
                open_tool_span = tool_span
                set_tool_call_content_attributes(
                    tool_span, capture_content, arguments=block.input
                )
                try:
                    handler_fn = tool_handlers.get(block.name)
                    if not handler_fn or not callable(handler_fn):
                        raise ValueError(
                            f'No handler registered for tool "{block.name}"'
                        )
                    result = (
                        await handler_fn(block.input)
                        if _is_coroutine(handler_fn)
                        else handler_fn(block.input)
                    )
                    # Inside the try, for the same reason as the blocking path above.
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
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    }
                )
            conversation.append({"role": "user", "content": tool_results})

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
            "usage": run_usage.total,
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
        # `ended`. On abandonment it is the only chance to close the tree, and to report what the
        # completed turns already cost. An abandoned span is left UNSET rather than ERROR: stopping
        # early is a normal thing for a consumer to do, and LaunchDarkly's own metrics record neither
        # a success nor an error for it, so ERROR would put two dashboards in disagreement.
        # Tool span first: it is a child, and a reader following the tree should not meet a closed
        # parent above an open child.
        if open_tool_span is not None:
            end_span_once(open_tool_span, ended, abandoned=True, cancelled=cancelled)
        if open_model_span is not None:
            end_span_once(open_model_span, ended, abandoned=True, cancelled=cancelled)
        if span is not None and id(span) not in ended and run_usage.reported:
            finish_root_span(span, config, run_usage.total)
        end_span_once(span, ended, abandoned=True, cancelled=cancelled)


def claude_messages(
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
        handler=create_claude_messages_handler(capture_content=capture_content),
        **kwargs,
    ).invoke(user_input, context, variables=variables)
