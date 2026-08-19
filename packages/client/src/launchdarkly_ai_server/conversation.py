"""Caller-supplied ``gen_ai.conversation.id``.

A dedicated OTel context key, not W3C baggage: the id must not leak onto outbound
provider HTTP calls. A multi-tenant process binds a different id per request.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator
from contextlib import aclosing, contextmanager
from typing import TYPE_CHECKING, Any

from opentelemetry import context as otel_context
from opentelemetry.trace import Span

if TYPE_CHECKING:
    # ``opentelemetry-sdk`` is an optional extra (``[otel]``), so it must not be imported at
    # runtime — this module has to work on an api-only install. Subclassing under
    # ``TYPE_CHECKING`` still gets the interface checked, which a ``cast`` at the registration
    # site would not.
    from opentelemetry.sdk.trace import SpanProcessor as _SpanProcessorBase
else:
    _SpanProcessorBase = object

GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"

_CONV_KEY = otel_context.create_key("launchdarkly.gen_ai.conversation.id")


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


async def _stream_with_bound_id(
    generator: AsyncGenerator[Any, None], conversation: str
) -> AsyncGenerator[Any, None]:
    """Re-attach ``conversation`` around each step of ``generator``.

    Only the id is re-applied — everything else about the ambient context, including the parent
    span, is whatever is active at iteration time. Streaming callers therefore see the same span
    parenting they would with no id bound at all.
    """
    async with aclosing(generator):
        while True:
            token = otel_context.attach(otel_context.set_value(_CONV_KEY, conversation))
            try:
                item = await generator.__anext__()
            except StopAsyncIteration:
                return
            finally:
                otel_context.detach(token)
            yield item


def bind_conversation_id(
    generator: AsyncGenerator[Any, None],
) -> AsyncGenerator[Any, None]:
    """Pin the currently-bound conversation id onto ``generator`` at call time.

    An ``async def`` with ``yield`` does not start its body until the first ``__anext__``, which
    for a streaming caller is normally after the :func:`conversation_id` block has exited. Reading
    the id here — eagerly, before any iteration — is what lets a caller bind, hand the generator
    off, and iterate it later.
    """
    conversation = _conversation_id_from(otel_context.get_current())
    if not conversation:
        return generator
    return _stream_with_bound_id(generator, conversation)


class ConversationIdSpanProcessor(_SpanProcessorBase):
    """Stamps ``gen_ai.conversation.id`` write-if-absent on every span.

    Structurally a ``SpanProcessor``; the base is only real to a type checker so that an
    api-only install (no ``[otel]`` extra) still imports this module.
    ``init_client()`` registers it ahead of ``BatchSpanProcessor``.
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

    def on_end(self, span: Any) -> None:
        return None

    def _on_ending(self, span: Any) -> None:
        # Python SDK calls ``_on_ending`` from ``span.end()``; JS uses ``onEnd``.
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int | None = None) -> bool:
        return True
