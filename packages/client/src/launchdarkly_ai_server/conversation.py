"""Caller-supplied ``gen_ai.conversation.id`` and judge evaluation span events.

A dedicated OTel context key, not W3C baggage: the id must not leak onto outbound
provider HTTP calls. A multi-tenant process binds a different id per request.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.trace import Span

GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"

_CONV_KEY = otel_context.create_key("launchdarkly.gen_ai.conversation.id")
_EVAL_KEY = otel_context.create_key("launchdarkly.judge.evaluation")


@dataclass
class _JudgeEvalCapture:
    name: str
    released: bool = False
    span: Any = None
    pending_end: Callable[[], None] | None = None


def _read_attribute(span: Any, key: str) -> Any:
    attrs = getattr(span, "attributes", None)
    if attrs is None:
        return None
    return attrs.get(key)


def set_conversation_id_if_absent(span: Any, conversation: str) -> None:
    """Write ``gen_ai.conversation.id`` only when the span does not already carry one."""
    if not conversation or span is None:
        return
    existing = _read_attribute(span, GEN_AI_CONVERSATION_ID)
    if isinstance(existing, str) and existing:
        return
    span.set_attribute(GEN_AI_CONVERSATION_ID, conversation)


def _conversation_id_from(ctx: otel_context.Context | None) -> str | None:
    if ctx is None:
        return None
    value = otel_context.get_value(_CONV_KEY, ctx)
    return value if isinstance(value, str) and value else None


def _record_evaluation(
    span: Any, name: str, score: float, explanation: str | None
) -> None:
    if span is None or not span.is_recording():
        return
    attrs: dict[str, Any] = {
        "gen_ai.evaluation.name": name,
        "gen_ai.evaluation.score.value": score,
    }
    if explanation:
        attrs["gen_ai.evaluation.explanation"] = explanation
    span.add_event("gen_ai.evaluation.result", attrs)
    span.set_attribute("gen_ai.evaluation.name", name)
    span.set_attribute("gen_ai.evaluation.score.value", score)
    if explanation:
        span.set_attribute("gen_ai.evaluation.explanation", explanation)


def _delay_invoke_agent_end(span: Any, capture: _JudgeEvalCapture) -> None:
    original_end = span.end
    ended = False

    def wrapped_end(*args: Any, **kwargs: Any) -> None:
        nonlocal ended
        if ended:
            return
        if capture.released:
            ended = True
            original_end(*args, **kwargs)
            return
        capture.span = span

        def pending() -> None:
            nonlocal ended
            if ended:
                return
            ended = True
            original_end(*args, **kwargs)

        capture.pending_end = pending

    span.end = wrapped_end


@contextmanager
def conversation_id(conversation: str) -> Iterator[None]:
    """Bind a caller-supplied conversation id for the duration of the ``with`` block.

    Every span the SDK creates while this is bound receives ``gen_ai.conversation.id``,
    provided :class:`ConversationIdSpanProcessor` is registered — which ``init_client()``
    does when OTel is installed. An empty or whitespace id is treated as unbound.
    """
    trimmed = conversation.strip()
    if not trimmed:
        yield
        return
    token = otel_context.attach(otel_context.set_value(_CONV_KEY, trimmed))
    try:
        yield
    finally:
        otel_context.detach(token)


RecordEvaluation = Callable[[float, str | None], None]


@asynccontextmanager
async def with_judge_evaluation(name: str) -> AsyncIterator[RecordEvaluation]:
    """Hold the judge ``invoke_agent`` span open until ``record`` runs.

    ``execute_and_track`` returns after the handler has already called ``span.end()``,
    so without this delay the evaluation event would be dropped.
    """
    capture = _JudgeEvalCapture(name=name)

    def record(score: float, explanation: str | None = None) -> None:
        if capture.span is not None:
            _record_evaluation(capture.span, capture.name, score, explanation)

    token = otel_context.attach(otel_context.set_value(_EVAL_KEY, capture))
    try:
        yield record
    finally:
        capture.released = True
        if capture.pending_end is not None:
            capture.pending_end()
        otel_context.detach(token)


class ConversationIdSpanProcessor:
    """Stamps ``gen_ai.conversation.id`` write-if-absent; delays judge ``invoke_agent`` end.

    Duck-typed to the OTel SDK ``SpanProcessor`` interface so this module depends only
    on ``opentelemetry-api``. ``init_client()`` registers it ahead of ``BatchSpanProcessor``.
    """

    def on_start(
        self, span: Span, parent_context: otel_context.Context | None = None
    ) -> None:
        ctx = (
            parent_context if parent_context is not None else otel_context.get_current()
        )
        conv = _conversation_id_from(ctx) or _conversation_id_from(
            otel_context.get_current()
        )
        if conv:
            set_conversation_id_if_absent(span, conv)

        capture = otel_context.get_value(_EVAL_KEY, ctx)
        if capture is None:
            capture = otel_context.get_value(_EVAL_KEY)
        if (
            isinstance(capture, _JudgeEvalCapture)
            and getattr(span, "name", None) == "invoke_agent"
        ):
            _delay_invoke_agent_end(span, capture)

    def on_end(self, span: Any) -> None:
        return None

    def _on_ending(self, span: Any) -> None:
        # Python SDK calls ``_on_ending`` from ``span.end()``; JS uses ``onEnd``.
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int | None = None) -> bool:
        return True
