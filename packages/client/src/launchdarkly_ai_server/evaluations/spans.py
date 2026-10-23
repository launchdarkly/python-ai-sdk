"""Spans the offline runner writes for one dataset row.

Every row gets a ``launchdarkly.dataset_row`` root, so its generation, tool
calls, and criteria share one trace. The root names no ``gen_ai`` operation and
no evaluation, so the conversation view treats it as an envelope and keeps the
generation in the transcript.

A provider handler traces its own call. A plain function does not, so the
runner describes what it handed the function and what came back: one ``chat``
span, an ``execute_tool`` span per recorded tool call, and, under each judge's
``evaluate`` span, a ``chat`` span for the judge call. These are written only
when the handler opened no SDK span of its own, so a call is never traced twice.

These spans export to the customer's project on the SDK key. They follow the
same content gate as every other span: the handler's explicit override when it
has one, otherwise the process default, which is off. An explicit ``True``
exports the transcript on these spans too. The ``$ld:ai:offline-evals:*``
events never carry the transcript.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from ..content import (
    SpanMessage,
    SpanMessagePart,
    ToolDefinitionInput,
    set_input_content_attributes,
    set_output_content_attributes,
    set_tool_call_content_attributes,
    text_message,
)
from ..conversation import (
    GEN_AI_CONVERSATION_ID,
    _record_evaluation,
    conversation_id,
    stamp_span_attributes,
)
from ..lifecycle import effective_capture_content
from ..trajectory import ToolInvocation
from ..utils import (
    SpanUsage,
    parse_template,
    parse_usage,
    set_model_identity_attributes,
    set_usage_span_attributes,
)

TRACER_NAME = "@launchdarkly/ai-server"
ROW_SPAN_NAME = "launchdarkly.dataset_row"

LD_EVALUATION_RUN_ID = "launchdarkly.evaluation_run.id"
LD_EVALUATION_RUN_ROW_INDEX = "launchdarkly.evaluation_run.row_index"

# Same namespace ai-evaluator uses for conversationId on a result row.
# Changing it re-keys every offline evaluation conversation.
_EVALUATION_CONVERSATION_NAMESPACE = uuid.UUID("6f0c2a4e-3b6d-4d0e-9c1a-2e6b7f8a9d01")


def conversation_id_for_evaluation_row(run_id: str, row_index: int | None) -> str:
    """UUID5 of ``{run_id}:{row_index}`` in the ai-evaluator namespace.

    Row index ``0`` is the suffix ``:0``. A missing index is row 0, matching
    the workers that persist a missing index that way.
    """
    index = 0 if row_index is None else int(row_index)
    return str(uuid.uuid5(_EVALUATION_CONVERSATION_NAMESPACE, f"{run_id}:{index}"))


@contextmanager
def bind_evaluation_row(run_id: str, row_index: int | None) -> Iterator[None]:
    """Bind one dataset row onto SDK spans started in this block.

    Sets ``gen_ai.conversation.id`` to :func:`conversation_id_for_evaluation_row`
    and ``launchdarkly.evaluation_run.id`` / ``.row_index``. Does not set
    ``launchdarkly.run.id``; that key is the AI Config tracking id.

    An outer conversation id is shadowed until the block exits. The evaluation
    row id is the join key for this row.
    """
    index = 0 if row_index is None else int(row_index)
    with (
        conversation_id(conversation_id_for_evaluation_row(run_id, index)),
        stamp_span_attributes(
            {LD_EVALUATION_RUN_ID: str(run_id), LD_EVALUATION_RUN_ROW_INDEX: index}
        ),
    ):
        yield


def _resolved_capture(capture: bool | None) -> bool:
    """An explicit bool wins. ``None`` is the process default."""
    return effective_capture_content(None) if capture is None else capture


def capture_for_call(handler: Any) -> bool:
    """Content capture for one call.

    A handler exposes ``capture_content`` as its effective decision. A plain
    function does not, so it follows the process default.
    """
    flag = getattr(handler, "capture_content", None)
    if isinstance(flag, bool):
        return flag
    return effective_capture_content(None)


def record_evaluation_result(
    span: Any,
    name: str,
    score: float,
    explanation: str | None = None,
    *,
    capture: bool | None = None,
) -> None:
    """Write ``gen_ai.evaluation.result`` with no verdict and no score label.

    ``explanation`` is the criterion reason. It is written only when capture is
    on for this call, the same gate as the online judge. The reason still
    travels on the criterion event either way. This does not add a prompt, a
    completion, or a tool payload.
    """
    if not _resolved_capture(capture):
        explanation = None
    _record_evaluation(span, name, score, explanation)


def _tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


def set_row_attributes(span: Span, run_id: str, row_index: int) -> None:
    span.set_attribute(
        GEN_AI_CONVERSATION_ID, conversation_id_for_evaluation_row(run_id, row_index)
    )
    span.set_attribute(LD_EVALUATION_RUN_ID, run_id)
    span.set_attribute(LD_EVALUATION_RUN_ROW_INDEX, row_index)


def start_row_span(run_id: str, row_index: int) -> Span:
    """Open the row's root span. The caller ends it after the row's criteria."""
    span = _tracer().start_span(
        ROW_SPAN_NAME, context=trace.set_span_in_context(trace.INVALID_SPAN)
    )
    set_row_attributes(span, run_id, row_index)
    return span


def _provider_name(config: Mapping[str, Any]) -> str:
    provider = config.get("provider")
    name = provider.get("name") if isinstance(provider, Mapping) else None
    return name.lower() if isinstance(name, str) else ""


def _model_name(config: Mapping[str, Any]) -> str:
    model = config.get("model")
    name = model.get("name") if isinstance(model, Mapping) else None
    return name if isinstance(name, str) else ""


def _call_id(index: int) -> str:
    """Row-local id joining a ``tool_call`` part to its ``execute_tool`` span.

    A plain function reports no provider call id, so the runner numbers the
    calls it recorded, in start order.
    """
    return f"call_{index + 1}"


def set_config_model_identity(span: Span, config: Mapping[str, Any]) -> None:
    """Write the provider and model a config names onto ``span``."""
    set_model_identity_attributes(span, _provider_name(config), _model_name(config))


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _prompt(
    config: Mapping[str, Any], user_input: Any, variables: Mapping[str, Any]
) -> tuple[str | None, list[SpanMessage]]:
    """The system instructions and messages the handler was given, rendered.

    Renders the config with the row's variables, the same single
    ``parse_template`` pass a provider handler applies. ``user_input`` is the
    final user turn, as it is for every handler.
    """
    values = dict(variables)
    system_parts: list[str] = []
    messages: list[SpanMessage] = []
    instructions = config.get("instructions")
    if isinstance(instructions, str) and instructions:
        system_parts.append(parse_template(instructions, values))
    for message in config.get("messages") or []:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        rendered = parse_template(content, values)
        role = message.get("role")
        if role == "system":
            system_parts.append(rendered)
        elif isinstance(role, str):
            messages.append(text_message(role, rendered))
    if user_input is not None:
        messages.append(text_message("user", _text(user_input)))
    return ("\n\n".join(system_parts) or None), messages


def _tool_definitions(config: Mapping[str, Any]) -> list[ToolDefinitionInput]:
    tools = config.get("tools")
    if not isinstance(tools, Mapping):
        return []
    definitions: list[ToolDefinitionInput] = []
    for key, tool in tools.items():
        if not isinstance(tool, Mapping):
            continue
        name = tool.get("name", key)
        if not isinstance(name, str) or not name:
            continue
        definitions.append(
            ToolDefinitionInput(
                name=name,
                description=tool.get("description") or None,
                parameters=tool.get("parameters"),
            )
        )
    return definitions


def _span_usage(usage: Any) -> SpanUsage | None:
    if not isinstance(usage, Mapping):
        return None
    parsed = parse_usage(dict(usage))
    details = parsed.get("input_details") or {}
    return SpanUsage(
        input=parsed["input"],
        output=parsed["output"],
        cache_read=details.get("cache_read", 0),
        cache_creation=details.get("cache_creation", 0),
    )


def record_chat(
    parent: Span,
    *,
    config: Mapping[str, Any],
    user_input: Any,
    variables: Mapping[str, Any],
    started_ns: int,
    ended_ns: int,
    result: Mapping[str, Any] | None,
    error: BaseException | None = None,
    tool_calls: Sequence[ToolInvocation] = (),
    capture: bool | None = None,
) -> None:
    """Write one ``chat {model}`` span for a call the handler did not trace.

    ``result`` is the handler's return value. ``error`` is what it raised; the
    span then has ERROR status and no output. ``tool_calls`` are the calls the
    handler made, written as ``tool_call`` parts ahead of the answer and tied
    to their ``execute_tool`` spans by call id, so the transcript keeps their
    order even when every step lands in the same millisecond.
    """
    model = _model_name(config)
    span = _tracer().start_span(
        f"chat {model}".strip(),
        context=trace.set_span_in_context(parent),
        start_time=started_ns,
    )
    try:
        span.set_attribute("gen_ai.operation.name", "chat")
        set_config_model_identity(span, config)
        resolved = _resolved_capture(capture)
        if resolved:
            system_instructions, messages = _prompt(config, user_input, variables)
            set_input_content_attributes(
                span,
                resolved,
                system_instructions=system_instructions,
                messages=messages,
                tool_definitions=_tool_definitions(config),
            )
        if error is not None:
            span.record_exception(error)
            span.set_attribute("error.type", type(error).__name__)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            return
        result = result or {}
        usage = _span_usage(result.get("usage"))
        if usage is not None:
            set_usage_span_attributes(span, usage)
        parts = [
            SpanMessagePart(
                type="tool_call",
                id=_call_id(index),
                name=call.name,
                arguments=call.arguments,
            )
            for index, call in enumerate(tool_calls)
        ]
        output = result.get("output", result.get("response"))
        if output is not None:
            parts.append(SpanMessagePart(type="text", content=_text(output)))
        if parts and resolved:
            set_output_content_attributes(
                span, resolved, [SpanMessage(role="assistant", parts=parts)]
            )
        span.set_status(Status(StatusCode.OK))
    finally:
        span.end(end_time=ended_ns)


def record_tool_calls(
    parent: Span,
    invocations: Sequence[ToolInvocation],
    *,
    capture: bool | None = None,
) -> None:
    """Write an ``execute_tool {name}`` span for each recorded call.

    Arguments and results are the recorder's rendered text, the same bounded
    values a judge reads. A call that never completed has no end time and is
    skipped rather than given one.
    """
    context = trace.set_span_in_context(parent)
    for index, invocation in enumerate(invocations):
        if invocation.started_ns is None or invocation.ended_ns is None:
            continue
        span = _tracer().start_span(
            f"execute_tool {invocation.name}",
            context=context,
            start_time=invocation.started_ns,
        )
        try:
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            span.set_attribute("gen_ai.tool.name", invocation.name)
            span.set_attribute("gen_ai.tool.call.id", _call_id(index))
            span.set_attribute("gen_ai.tool.type", "function")
            resolved = _resolved_capture(capture)
            if resolved:
                set_tool_call_content_attributes(
                    span,
                    resolved,
                    arguments=invocation.arguments,
                    result=invocation.result,
                )
            if invocation.error is not None:
                span.set_attribute("error.type", "tool_error")
                span.set_status(Status(StatusCode.ERROR, invocation.error))
            else:
                span.set_status(Status(StatusCode.OK))
        finally:
            span.end(end_time=invocation.ended_ns)
