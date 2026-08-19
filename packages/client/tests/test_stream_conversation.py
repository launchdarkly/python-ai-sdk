"""Conversation id across ``config().stream()``.

An ``async def`` with ``yield`` does not run its body until the first ``__anext__``, which for a
streaming caller is normally after the ``conversation_id`` block has already exited. These tests
pin the id onto spans a handler opens while streaming, and pin that binding no id leaves span
parenting alone.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import ProviderHandler, config
from launchdarkly_ai_server.conversation import (
    GEN_AI_CONVERSATION_ID,
    ConversationIdSpanProcessor,
    conversation_id,
)

CONTEXT = {"kind": "user", "key": "u1"}

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(ConversationIdSpanProcessor())
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
_tracer = _provider.get_tracer("stream-conversation-test")


@pytest.fixture(autouse=True)
def _reset_exporter() -> Iterator[None]:
    _exporter.clear()
    yield
    _exporter.clear()


@pytest.fixture
def mock_ld_client() -> Iterator[MagicMock]:
    c = MagicMock()
    c.track = MagicMock()
    c.flush = AsyncMock()
    c.close = AsyncMock()
    c.variation = AsyncMock(
        return_value={
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "Be helpful.",
            "_ldMeta": {
                "enabled": True,
                "variationKey": "v1",
                "version": 1,
                "mode": "messages",
            },
        }
    )
    lifecycle_module._set_client_for_testing(c)
    yield c
    lifecycle_module._reset_for_testing()


def _span_creating_handler(chunks: list[str] | None = None) -> ProviderHandler:
    """Mirrors a real streaming handler: spans are opened as the generator runs, not at call time."""
    _chunks = chunks or ["Hello", " World"]
    _usage = {"input_tokens": 10, "output_tokens": 5}

    async def fn(cfg, user_input, tool_handlers, variables, history=None) -> dict:  # type: ignore[override]
        return {"output": "".join(_chunks), "usage": _usage}

    async def stream_fn(
        cfg, user_input, tool_handlers, variables, history=None
    ) -> AsyncGenerator:  # type: ignore[override]
        root = _tracer.start_span("invoke_agent")
        for c in _chunks:
            with trace.use_span(root, end_on_exit=False):
                chat = _tracer.start_span("chat gpt-4")
                chat.end()
            yield {"type": "chunk", "text": c}
        root.end()
        yield {"type": "done", "output": "".join(_chunks), "usage": _usage}

    return ProviderHandler(
        fn=fn, provides_for=("TestProvider", "messages"), stream_fn=stream_fn
    )


def finished() -> list[ReadableSpan]:
    return list(_exporter.get_finished_spans())


def _ids_on(prefixes: tuple[str, ...]) -> list[Any]:
    return [
        s.attributes.get(GEN_AI_CONVERSATION_ID) if s.attributes else None
        for s in finished()
        if any(s.name.startswith(p) for p in prefixes)
    ]


class TestStreamConversationId:
    async def test_stamps_id_when_iterated_outside_the_block(
        self, mock_ld_client: MagicMock
    ) -> None:
        m = config(key="flag", handler=_span_creating_handler())

        # The natural shape: bind, build the generator, hand it off, iterate later.
        with conversation_id("thread-123"):
            gen = m.stream("q", CONTEXT)
        async for _ in gen:
            pass

        ids = _ids_on(("invoke_agent", "chat"))
        assert len(ids) > 0
        assert all(i == "thread-123" for i in ids)

    async def test_stamps_id_when_iterated_inside_the_block(
        self, mock_ld_client: MagicMock
    ) -> None:
        m = config(key="flag", handler=_span_creating_handler())

        with conversation_id("thread-inside"):
            async for _ in m.stream("q", CONTEXT):
                pass

        ids = _ids_on(("invoke_agent", "chat"))
        assert len(ids) > 0
        assert all(i == "thread-inside" for i in ids)

    async def test_leaves_parenting_unchanged_when_unbound(
        self, mock_ld_client: MagicMock
    ) -> None:
        m = config(key="flag", handler=_span_creating_handler(["only"]))

        caller = _tracer.start_span("caller")
        with trace.use_span(caller, end_on_exit=False):
            async for _ in m.stream("q", CONTEXT):
                pass
        caller.end()

        root = next(s for s in finished() if s.name == "invoke_agent")
        caller_span = next(s for s in finished() if s.name == "caller")
        assert root.parent is not None
        assert root.parent.span_id == caller_span.context.span_id
        assert (root.attributes or {}).get(GEN_AI_CONVERSATION_ID) is None

    async def test_overlapping_streams_stay_isolated(
        self, mock_ld_client: MagicMock
    ) -> None:
        async def run(tag: str) -> None:
            m = config(key="flag", handler=_span_creating_handler([tag]))
            with conversation_id(tag):
                gen = m.stream("q", CONTEXT)
            async for _ in gen:
                pass

        await asyncio.gather(run("tenant-a"), run("tenant-b"))

        roots = [s for s in finished() if s.name == "invoke_agent"]
        assert len(roots) == 2
        assert sorted(
            (s.attributes or {}).get(GEN_AI_CONVERSATION_ID) for s in roots
        ) == ["tenant-a", "tenant-b"]

        for root in roots:
            expected = (root.attributes or {}).get(GEN_AI_CONVERSATION_ID)
            same_trace = [
                s for s in finished() if s.context.trace_id == root.context.trace_id
            ]
            assert all(
                (s.attributes or {}).get(GEN_AI_CONVERSATION_ID) == expected
                for s in same_trace
            )


class TestConcurrentConversationIsolation:
    async def test_two_overlapping_scopes_stay_isolated(self) -> None:
        async def one(tag: str) -> None:
            with conversation_id(tag):
                span = _tracer.start_span(f"{tag}:invoke_agent")
                await asyncio.sleep(0.01)
                span.end()

        await asyncio.gather(one("tenant-a"), one("tenant-b"))

        by_name = {
            s.name: (s.attributes or {}).get(GEN_AI_CONVERSATION_ID) for s in finished()
        }
        assert by_name["tenant-a:invoke_agent"] == "tenant-a"
        assert by_name["tenant-b:invoke_agent"] == "tenant-b"
