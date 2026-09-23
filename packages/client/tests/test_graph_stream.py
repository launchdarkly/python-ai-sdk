"""
Tests for §3.15a ``graph().stream()``.
Reference: TESTING.md §3.15a, Appendix A.4 / A.13.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import ProviderHandler, graph
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
_tracer = _provider.get_tracer("@launchdarkly/ai-server")


def _node_variation(instructions: str = "Be helpful.") -> dict[str, Any]:
    return {
        "model": {"name": "gpt-4"},
        "provider": {"name": "TestProvider"},
        "instructions": instructions,
        "_ldMeta": {
            "enabled": True,
            "variationKey": "v1",
            "version": 1,
            "mode": "messages",
        },
    }


def _make_client(graph_variation: dict | None = None) -> MagicMock:
    c = MagicMock()
    c.track = MagicMock()
    c.flush = AsyncMock()
    c.close = AsyncMock()

    graph_var = graph_variation or {
        "root": "root-node",
        "edges": {"root-node": [{"key": "leaf-node"}]},
    }

    node_instructions = {
        "root-node": "I am root",
        "leaf-node": "I am leaf",
        "agent-a": "I am A",
        "agent-b": "I am B",
    }

    async def variation_side_effect(key: str, ctx: dict, default: Any) -> Any:
        if key == "graph-key":
            return graph_var
        return _node_variation(node_instructions.get(key, "Be helpful."))

    c.variation = AsyncMock(side_effect=variation_side_effect)
    return c


def _make_streaming_handler(
    chunks: list[str] | None = None,
    usage: dict | None = None,
) -> ProviderHandler:
    _chunks = chunks or ["Hi", "!"]
    _usage = usage or {"input_tokens": 2, "output_tokens": 3}

    async def fn(config, user_input, tool_handlers, variables, history=None) -> dict:  # type: ignore[override]
        return {"output": "".join(_chunks), "usage": _usage}

    async def stream_fn(
        config, user_input, tool_handlers, variables, history=None
    ) -> AsyncGenerator:  # type: ignore[override]
        for c in _chunks:
            yield {"type": "chunk", "text": c}
        yield {"type": "done", "output": "".join(_chunks), "usage": _usage}

    return ProviderHandler(
        fn=fn, provides_for=("TestProvider", "messages"), stream_fn=stream_fn
    )


def _make_blocking_handler(response: str = "blocked") -> ProviderHandler:
    async def fn(config, user_input, tool_handlers, variables, history=None) -> dict:  # type: ignore[override]
        return {
            "output": response,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    return ProviderHandler(fn=fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]


def _make_branch_picking_stream_handler(pick_target: str) -> ProviderHandler:
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in pick_target)
    _usage = {"input_tokens": 1, "output_tokens": 1}
    received: list[tuple[Any, Any]] = []

    async def fn(config, user_input, tool_handlers, variables, history=None) -> dict:  # type: ignore[override]
        return {"output": "ok", "usage": _usage}

    async def stream_fn(
        config, user_input, tool_handlers, variables, history=None
    ) -> AsyncGenerator:  # type: ignore[override]
        received.append((config, tool_handlers))
        handoff = (tool_handlers or {}).get(f"__handoff_{sanitized}")
        if handoff:
            handoff()
        yield {"type": "chunk", "text": "ok"}
        yield {"type": "done", "output": "ok", "usage": _usage}

    h = ProviderHandler(
        fn=fn, provides_for=("TestProvider", "messages"), stream_fn=stream_fn
    )
    h._test_received = received  # type: ignore[attr-defined]
    return h


def _span_creating_stream_handler() -> ProviderHandler:
    _chunks = ["ok"]
    _usage = {"input_tokens": 1, "output_tokens": 1}

    async def fn(config, user_input, tool_handlers, variables, history=None) -> dict:  # type: ignore[override]
        return {"output": "ok", "usage": _usage}

    async def stream_fn(
        config, user_input, tool_handlers, variables, history=None
    ) -> AsyncGenerator:  # type: ignore[override]
        root = _tracer.start_span("handler.invoke_agent")
        for c in _chunks:
            with trace.use_span(root, end_on_exit=False):
                chat = _tracer.start_span("handler.chat")
                chat.end()
            yield {"type": "chunk", "text": c}
        root.end()
        yield {"type": "done", "output": "".join(_chunks), "usage": _usage}

    return ProviderHandler(
        fn=fn, provides_for=("TestProvider", "messages"), stream_fn=stream_fn
    )


async def _collect(gen: Any) -> list[Any]:
    return [e async for e in gen]


def _track_names(client: MagicMock) -> list[str]:
    return [c[0][0] for c in client.track.call_args_list]


def _track_payload(client: MagicMock, name: str) -> Any:
    for c in client.track.call_args_list:
        if c[0][0] == name:
            return c[0][2] if len(c[0]) > 2 else c[1]
    return None


def _finished() -> list[ReadableSpan]:
    return list(_exporter.get_finished_spans())


@pytest.fixture
def mock_ld_client() -> Iterator[MagicMock]:
    client = _make_client()
    lifecycle_module._set_client_for_testing(client)
    yield client
    lifecycle_module._reset_for_testing()


@pytest.fixture(autouse=True)
def _reset_exporter() -> Iterator[None]:
    _exporter.clear()
    # Hand the SDK this file's tracer without touching the global provider:
    # set_tracer_provider is process-wide and first-writer-wins, and another package's
    # suite already claims it at import time, so registering here exports nothing.
    # graph.py imports `trace` inside its functions, so the seam is get_tracer itself.
    with patch.object(trace, "get_tracer", return_value=_tracer):
        yield
    _exporter.clear()


# ---------------------------------------------------------------------------
# Setup / errors / traversal events
# ---------------------------------------------------------------------------


class TestGraphStream:
    def test_returns_async_generator(self, mock_ld_client: MagicMock) -> None:
        gen = graph("graph-key", handlers=[_make_streaming_handler()]).stream(
            "hi", CONTEXT
        )
        assert hasattr(gen, "__aiter__")

    async def test_throws_when_graph_disabled(self, mock_ld_client: MagicMock) -> None:
        mock_ld_client.variation = AsyncMock(return_value={"edges": {}})
        gen = graph("graph-key", handlers=[_make_streaming_handler()]).stream(
            "hi", CONTEXT
        )
        with pytest.raises((ValueError, RuntimeError), match="disabled"):
            await _collect(gen)

    async def test_throws_when_no_handlers(self, mock_ld_client: MagicMock) -> None:
        gen = graph("graph-key").stream("hi", CONTEXT)
        with pytest.raises((ValueError, RuntimeError)):
            await _collect(gen)

    async def test_emits_node_start_in_order(self, mock_ld_client: MagicMock) -> None:
        events = await _collect(
            graph("graph-key", handlers=[_make_streaming_handler(["ok"])]).stream(
                "hi", CONTEXT
            )
        )
        starts = [e for e in events if e["type"] == "node_start"]
        assert [e["nodeKey"] for e in starts] == ["root-node", "leaf-node"]

    async def test_forwards_chunks_tagged_with_node_key(
        self, mock_ld_client: MagicMock
    ) -> None:
        events = await _collect(
            graph("graph-key", handlers=[_make_streaming_handler(["Hi", "!"])]).stream(
                "hi", CONTEXT
            )
        )
        chunks = [e for e in events if e["type"] == "chunk"]
        assert chunks == [
            {"type": "chunk", "text": "Hi", "nodeKey": "root-node"},
            {"type": "chunk", "text": "!", "nodeKey": "root-node"},
            {"type": "chunk", "text": "Hi", "nodeKey": "leaf-node"},
            {"type": "chunk", "text": "!", "nodeKey": "leaf-node"},
        ]

    async def test_emits_node_done_with_response_and_usage(
        self, mock_ld_client: MagicMock
    ) -> None:
        events = await _collect(
            graph(
                "graph-key",
                handlers=[
                    _make_streaming_handler(
                        ["Hi", "!"], {"input_tokens": 2, "output_tokens": 3}
                    )
                ],
            ).stream("hi", CONTEXT)
        )
        dones = [e for e in events if e["type"] == "node_done"]
        assert len(dones) == 2
        assert dones[0] == {
            "type": "node_done",
            "nodeKey": "root-node",
            "response": "Hi!",
            "usage": {"input": 2, "output": 3, "total": 5},
        }
        assert dones[1]["nodeKey"] == "leaf-node"
        assert dones[1]["usage"] == {"input": 2, "output": 3, "total": 5}

    async def test_emits_handoff_between_node_done_and_next_start(
        self, mock_ld_client: MagicMock
    ) -> None:
        events = await _collect(
            graph("graph-key", handlers=[_make_streaming_handler(["ok"])]).stream(
                "hi", CONTEXT
            )
        )
        types = [e["type"] for e in events]
        idx = types.index("handoff")
        assert events[idx] == {
            "type": "handoff",
            "sourceKey": "root-node",
            "targetKey": "leaf-node",
        }
        assert types[idx - 1] == "node_done"
        assert types[idx + 1] == "node_start"

    async def test_final_done_has_leaf_response_and_aggregate_usage(
        self, mock_ld_client: MagicMock
    ) -> None:
        events = await _collect(
            graph(
                "graph-key",
                handlers=[
                    _make_streaming_handler(
                        ["final"], {"input_tokens": 2, "output_tokens": 3}
                    )
                ],
            ).stream("hi", CONTEXT)
        )
        done = events[-1]
        assert done["type"] == "done"
        assert done["response"] == "final"
        assert done["usage"] == {"input": 4, "output": 6, "total": 10}
        assert "path" not in done
        assert "nodes" not in done

    async def test_lifecycle_events_before_final_done(
        self, mock_ld_client: MagicMock
    ) -> None:
        events = await _collect(
            graph("graph-key", handlers=[_make_streaming_handler(["a"])]).stream(
                "hi", CONTEXT
            )
        )
        assert events[-1]["type"] == "done"
        assert all(e["type"] != "done" for e in events[:-1])

    async def test_tracks_invocation_success(self, mock_ld_client: MagicMock) -> None:
        await _collect(
            graph("graph-key", handlers=[_make_streaming_handler(["ok"])]).stream(
                "hi", CONTEXT
            )
        )
        assert "$ld:ai:graph:invocation_success" in _track_names(mock_ld_client)

    async def test_tracks_duration_total(self, mock_ld_client: MagicMock) -> None:
        await _collect(
            graph("graph-key", handlers=[_make_streaming_handler(["ok"])]).stream(
                "hi", CONTEXT
            )
        )
        assert "$ld:ai:graph:duration:total" in _track_names(mock_ld_client)

    async def test_tracks_path(self, mock_ld_client: MagicMock) -> None:
        await _collect(
            graph("graph-key", handlers=[_make_streaming_handler(["ok"])]).stream(
                "hi", CONTEXT
            )
        )
        assert "$ld:ai:graph:path" in _track_names(mock_ld_client)

    async def test_tracks_handoff_success(self, mock_ld_client: MagicMock) -> None:
        await _collect(
            graph("graph-key", handlers=[_make_streaming_handler(["ok"])]).stream(
                "hi", CONTEXT
            )
        )
        payload = _track_payload(mock_ld_client, "$ld:ai:graph:handoff_success")
        assert payload is not None
        assert payload["sourceKey"] == "root-node"
        assert payload["targetKey"] == "leaf-node"

    async def test_tracks_invocation_failure_and_rethrows(
        self, mock_ld_client: MagicMock
    ) -> None:
        async def fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            return {"output": "x", "usage": {"input_tokens": 1, "output_tokens": 1}}

        async def stream_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> AsyncGenerator:  # type: ignore[override]
            raise RuntimeError("stream boom")
            yield  # pragma: no cover

        h = ProviderHandler(
            fn=fn, provides_for=("TestProvider", "messages"), stream_fn=stream_fn
        )
        with pytest.raises(RuntimeError, match="stream boom"):
            await _collect(graph("graph-key", handlers=[h]).stream("hi", CONTEXT))
        assert "$ld:ai:graph:invocation_failure" in _track_names(mock_ld_client)

    async def test_generation_success_includes_graph_key(
        self, mock_ld_client: MagicMock
    ) -> None:
        await _collect(
            graph("graph-key", handlers=[_make_streaming_handler(["ok"])]).stream(
                "hi", CONTEXT
            )
        )
        success = [
            c
            for c in mock_ld_client.track.call_args_list
            if c[0][0] == "$ld:ai:generation:success"
        ]
        assert len(success) >= 2
        for call in success:
            assert call[0][2]["graphKey"] == "graph-key"

    async def test_falls_back_to_blocking_handler(
        self, mock_ld_client: MagicMock
    ) -> None:
        events = await _collect(
            graph("graph-key", handlers=[_make_blocking_handler("blocked")]).stream(
                "hi", CONTEXT
            )
        )
        chunks = [e for e in events if e["type"] == "chunk"]
        assert chunks == [
            {"type": "chunk", "text": "blocked", "nodeKey": "root-node"},
            {"type": "chunk", "text": "blocked", "nodeKey": "leaf-node"},
        ]
        assert events[-1]["type"] == "done"
        assert events[-1]["response"] == "blocked"

    async def test_includes_graph_judge_results_on_done(
        self, mock_ld_client: MagicMock
    ) -> None:
        judge_data = {
            "graph-judge": {
                "usage": {"input": 1, "output": 1, "total": 2},
                "response": "ok",
                "score": 0.8,
            }
        }
        with patch(
            "launchdarkly_ai_server.judges.run_judges",
            new_callable=AsyncMock,
            return_value=judge_data,
        ) as run_judges:
            events = await _collect(
                graph(
                    "graph-key",
                    handlers=[_make_streaming_handler(["final"])],
                    graph_judge="graph-judge",
                ).stream("hi", CONTEXT)
            )
        assert run_judges.await_count >= 1
        assert events[-1]["type"] == "done"
        assert events[-1]["judgeResults"] == judge_data

    async def test_omits_judge_results_when_empty(
        self, mock_ld_client: MagicMock
    ) -> None:
        with patch(
            "launchdarkly_ai_server.judges.run_judges",
            new_callable=AsyncMock,
            return_value={},
        ):
            events = await _collect(
                graph(
                    "graph-key",
                    handlers=[_make_streaming_handler(["final"])],
                    graph_judge="graph-judge",
                ).stream("hi", CONTEXT)
            )
        assert "judgeResults" not in events[-1]


# ---------------------------------------------------------------------------
# Multi-edge routing
# ---------------------------------------------------------------------------


class TestGraphStreamMultiEdge:
    @pytest.fixture
    def mock_ld_client(self) -> Iterator[MagicMock]:
        client = _make_client(
            {
                "root": "root-node",
                "edges": {
                    "root-node": [{"key": "agent-a"}, {"key": "agent-b"}],
                },
            }
        )
        lifecycle_module._set_client_for_testing(client)
        yield client
        lifecycle_module._reset_for_testing()

    async def test_model_pick_emits_handoff_success(
        self, mock_ld_client: MagicMock
    ) -> None:
        h = _make_branch_picking_stream_handler("agent-b")
        await _collect(graph("graph-key", handlers=[h]).stream("hi", CONTEXT))
        handoffs = [
            c
            for c in mock_ld_client.track.call_args_list
            if c[0][0] == "$ld:ai:graph:handoff_success"
        ]
        assert len(handoffs) >= 1
        assert handoffs[0][0][2]["sourceKey"] == "root-node"
        assert handoffs[0][0][2]["targetKey"] == "agent-b"

    async def test_handoff_failure_when_node_throws_after_choice(
        self, mock_ld_client: MagicMock
    ) -> None:
        sanitized = "agent_a"
        _usage = {"input_tokens": 1, "output_tokens": 1}

        async def fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            return {"output": "ok", "usage": _usage}

        async def stream_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> AsyncGenerator:  # type: ignore[override]
            handoff = (tool_handlers or {}).get(f"__handoff_{sanitized}")
            if handoff:
                handoff()
            raise RuntimeError("boom after choice")
            yield  # pragma: no cover

        h = ProviderHandler(
            fn=fn, provides_for=("TestProvider", "messages"), stream_fn=stream_fn
        )
        with pytest.raises(RuntimeError, match="boom after choice"):
            await _collect(graph("graph-key", handlers=[h]).stream("hi", CONTEXT))
        payload = _track_payload(mock_ld_client, "$ld:ai:graph:handoff_failure")
        assert payload is not None
        assert payload["sourceKey"] == "root-node"
        assert payload["targetKey"] == "agent-a"

    async def test_judges_receive_original_config(
        self, mock_ld_client: MagicMock
    ) -> None:
        h = _make_branch_picking_stream_handler("agent-b")
        with patch(
            "launchdarkly_ai_server.judges.run_judges",
            new_callable=AsyncMock,
            return_value={},
        ) as run_judges:
            await _collect(graph("graph-key", handlers=[h]).stream("hi", CONTEXT))

        root_call = next(
            (
                c
                for c in run_judges.await_args_list
                if (c.kwargs.get("config") or {}).get("instructions") == "I am root"
            ),
            None,
        )
        assert root_call is not None
        judged = root_call.kwargs["config"]
        assert judged["instructions"] == "I am root"
        assert not any(k.startswith("__handoff_") for k in (judged.get("tools") or {}))

    async def test_handoff_tools_and_instructions_match_invoke(
        self, mock_ld_client: MagicMock
    ) -> None:
        stream_h = _make_branch_picking_stream_handler("agent-b")
        await _collect(graph("graph-key", handlers=[stream_h]).stream("hi", CONTEXT))
        stream_cfg, stream_tools = stream_h._test_received[0]  # type: ignore[attr-defined]

        invoke_received: list[tuple[Any, Any]] = []

        async def invoke_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            invoke_received.append((config, tool_handlers))
            handoff = (tool_handlers or {}).get("__handoff_agent_b")
            if handoff:
                handoff()
            return {"output": "ok", "usage": {"input_tokens": 1, "output_tokens": 1}}

        invoke_h = ProviderHandler(
            fn=invoke_fn, provides_for=("TestProvider", "messages")
        )  # type: ignore[arg-type]
        await graph("graph-key", handlers=[invoke_h]).invoke("hi", CONTEXT)
        invoke_cfg, invoke_tools = invoke_received[0]

        assert (
            stream_cfg["tools"]["__handoff_agent_a"]["description"]
            == invoke_cfg["tools"]["__handoff_agent_a"]["description"]
        )
        assert (
            stream_cfg["tools"]["__handoff_agent_b"]["description"]
            == invoke_cfg["tools"]["__handoff_agent_b"]["description"]
        )
        assert stream_cfg["instructions"] == invoke_cfg["instructions"]
        assert (
            stream_tools["__handoff_agent_b"]() == invoke_tools["__handoff_agent_b"]()
        )


# ---------------------------------------------------------------------------
# Conversation id + OTel parenting + abandonment (§3.15a / A.4)
# ---------------------------------------------------------------------------


class TestGraphStreamOtel:
    async def test_stamps_conversation_id_when_bound_at_call_time(
        self, mock_ld_client: MagicMock
    ) -> None:
        with conversation_id("thread-graph-stream"):
            gen = graph("graph-key", handlers=[_span_creating_stream_handler()]).stream(
                "hi", CONTEXT
            )
        await _collect(gen)

        graph_spans = [s for s in _finished() if s.name == "ld.ai.graph"]
        assert len(graph_spans) >= 1
        assert graph_spans[0].attributes
        assert (
            graph_spans[0].attributes.get(GEN_AI_CONVERSATION_ID)
            == "thread-graph-stream"
        )

    async def test_handler_spans_nest_under_graph_on_stream(
        self, mock_ld_client: MagicMock
    ) -> None:
        await _collect(
            graph("graph-key", handlers=[_span_creating_stream_handler()]).stream(
                "hi", CONTEXT
            )
        )
        spans = _finished()
        graph_spans = [s for s in spans if s.name == "ld.ai.graph"]
        handler_spans = [s for s in spans if s.name.startswith("handler.")]
        assert len(graph_spans) >= 1
        assert len(handler_spans) >= 1
        gctx = graph_spans[0].get_span_context()
        for hs in handler_spans:
            assert hs.get_span_context().trace_id == gctx.trace_id
            # Ancestor: parent chain reaches the graph span
            parent = hs.parent
            assert parent is not None

    async def test_handler_spans_nest_under_graph_on_invoke(
        self, mock_ld_client: MagicMock
    ) -> None:
        async def fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            root = _tracer.start_span("handler.invoke_agent")
            with trace.use_span(root, end_on_exit=False):
                chat = _tracer.start_span("handler.chat")
                chat.end()
            root.end()
            return {"output": "ok", "usage": {"input_tokens": 1, "output_tokens": 1}}

        h = ProviderHandler(fn=fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]
        await graph("graph-key", handlers=[h]).invoke("hi", CONTEXT)

        spans = _finished()
        graph_spans = [s for s in spans if s.name == "ld.ai.graph"]
        handler_spans = [s for s in spans if s.name.startswith("handler.")]
        assert len(graph_spans) >= 1
        assert len(handler_spans) >= 1
        gctx = graph_spans[0].get_span_context()
        for hs in handler_spans:
            assert hs.get_span_context().trace_id == gctx.trace_id

    async def test_graph_parent_captured_at_stream_call_time(
        self, mock_ld_client: MagicMock
    ) -> None:
        caller = _tracer.start_span("caller")
        with trace.use_span(caller, end_on_exit=False):
            gen = graph("graph-key", handlers=[_make_streaming_handler(["ok"])]).stream(
                "hi", CONTEXT
            )
        caller.end()
        await _collect(gen)

        graph_spans = [s for s in _finished() if s.name == "ld.ai.graph"]
        assert len(graph_spans) >= 1
        assert graph_spans[0].parent is not None
        assert graph_spans[0].parent.span_id == caller.get_span_context().span_id

    async def test_abandoned_on_consumer_break(self, mock_ld_client: MagicMock) -> None:
        gen = graph("graph-key", handlers=[_make_streaming_handler(["a", "b"])]).stream(
            "hi", CONTEXT
        )
        async for event in gen:
            if event["type"] == "chunk":
                break
        await gen.aclose()

        graph_spans = [s for s in _finished() if s.name == "ld.ai.graph"]
        assert len(graph_spans) >= 1
        attrs = graph_spans[0].attributes or {}
        assert attrs.get("launchdarkly.stream.abandoned") is True
        assert "$ld:ai:graph:invocation_success" not in _track_names(mock_ld_client)

    async def test_graph_judge_spans_nest_under_graph(
        self, mock_ld_client: MagicMock
    ) -> None:
        async def run_judges_with_span(*args: Any, **kwargs: Any) -> dict:
            span = _tracer.start_span("graph.judge")
            span.end()
            return {
                "graph-judge": {
                    "usage": {"input": 1, "output": 1, "total": 2},
                    "response": "ok",
                    "score": 0.9,
                }
            }

        with patch(
            "launchdarkly_ai_server.judges.run_judges",
            new_callable=AsyncMock,
            side_effect=run_judges_with_span,
        ):
            await _collect(
                graph(
                    "graph-key",
                    handlers=[_make_streaming_handler(["final"])],
                    graph_judge="graph-judge",
                ).stream("hi", CONTEXT)
            )

        spans = _finished()
        graph_spans = [s for s in spans if s.name == "ld.ai.graph"]
        judge_spans = [s for s in spans if s.name == "graph.judge"]
        assert len(graph_spans) >= 1
        assert len(judge_spans) >= 1
        assert (
            judge_spans[0].get_span_context().trace_id
            == graph_spans[0].get_span_context().trace_id
        )
