"""
Tests for the native graph adapter (to_lang_graph) and LangChain-specific specs.
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_openai import ChatOpenAI as _REAL_CHAT_OPENAI

from launchdarkly_ai_langchain_agents.native_graph import (
    _build_node_tools,
    _extract_usage,
    _to_neutral_history,
    to_lang_graph,
)
from launchdarkly_ai_server import GraphDefinition, GraphEdge, GraphNode
from tests.forwarding_spec import (
    LANGCHAIN_CHAT_OPENAI,
    candidate_keys,
    probe_forwarded_keys,
)
from tests.never_forwarded import NEVER_FORWARDED_BAG

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ai_msg(
    content: str = "answer", input_tokens: int = 10, output_tokens: int = 5
) -> Any:
    msg = MagicMock()
    msg.content = content
    msg.type = "ai"
    msg.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    return msg


def _to_graph_edge(e: dict[str, Any]) -> GraphEdge:
    sk = e.get("source_key", "")
    tk = e.get("target_key", "")
    return GraphEdge(
        key=e.get("key", f"{sk}-{tk}"),
        source_key=sk,
        target_key=tk,
        handoff=e.get("handoff"),
    )


def _to_graph_node(n: dict[str, Any]) -> GraphNode:
    node_edges = [_to_graph_edge(e) for e in (n.get("edges") or [])]
    return GraphNode(
        key=n["key"],
        config=n.get("config", {}),
        meta=n.get("meta", {}),
        edges=node_edges,
        is_terminal=n.get("is_terminal", True),
    )


def _make_graph_def(
    enabled: bool = True,
    nodes: dict[str, Any] | None = None,
    edges: list[dict[str, Any]] | None = None,
    root_key: str = "root",
) -> GraphDefinition:
    raw_nodes = nodes or {
        root_key: {
            "key": root_key,
            "config": {"model": {"name": "gpt-4o"}, "instructions": "help"},
            "meta": {"variationKey": "v1", "version": 1},
            "edges": [],
            "is_terminal": True,
        }
    }
    _edge_objects = [_to_graph_edge(e) for e in (edges or [])]
    _node_objs: dict[str, GraphNode] = {
        k: _to_graph_node(n) for k, n in raw_nodes.items()
    }

    def edges_from(k: str) -> list[GraphEdge]:
        return [e for e in _edge_objects if e.source_key == k]

    async def _noop_run_node(*a: Any, **kw: Any) -> Any:
        raise NotImplementedError

    async def _noop_traverse(fn: Any, ctx: Any = None) -> None:
        return None

    return GraphDefinition(
        key="test-graph",
        enabled=enabled,
        root=_node_objs.get(root_key),
        get_node=lambda k: _node_objs.get(k),
        get_child_nodes=lambda k: [
            _node_objs[e.target_key]
            for e in edges_from(k)
            if e.target_key in _node_objs
        ],
        get_parent_nodes=lambda k: [
            _node_objs[e.source_key]
            for e in _edge_objects
            if e.target_key == k and e.source_key in _node_objs
        ],
        terminal_nodes=lambda: [
            n for n in _node_objs.values() if len(edges_from(n.key)) == 0
        ],
        is_terminal=lambda k: len(edges_from(k)) == 0,
        edges_from=edges_from,
        run_node=_noop_run_node,
        route=_noop_run_node,
        traverse=_noop_traverse,
        reverse_traverse=_noop_traverse,
    )


async def _make_def_promise(def_obj: GraphDefinition) -> GraphDefinition:
    return def_obj


def _make_langgraph_mocks(ai_msg: Any) -> dict[str, Any]:
    """Create minimal mocks for LangGraph modules, tracked via sys.modules patch."""
    node_fns: dict[str, Any] = {}
    edges_added: list[tuple[str, str]] = []

    class MockStateGraph:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        def add_node(self, name: str, fn: Any) -> None:
            node_fns[name] = fn

        def add_edge(self, src: str, tgt: str) -> None:
            edges_added.append((src, tgt))

        def add_conditional_edges(self, *a: Any, **kw: Any) -> None:
            pass

        def compile(self) -> Any:
            compiled = MagicMock()
            compiled.ainvoke = AsyncMock(return_value={"messages": [ai_msg]})
            return compiled

    mock_langgraph_graph = MagicMock()
    mock_langgraph_graph.StateGraph = MockStateGraph
    mock_langgraph_graph.START = "__start__"
    mock_langgraph_graph.END = "__end__"

    mock_prebuilt = MagicMock()
    mock_prebuilt.ToolNode = MagicMock(return_value=MagicMock())
    mock_prebuilt.tools_condition = MagicMock()

    mock_lc_msgs = MagicMock()
    mock_lc_msgs.HumanMessage = MagicMock(
        side_effect=lambda c: MagicMock(content=c, type="human")
    )
    mock_lc_msgs.SystemMessage = MagicMock(
        side_effect=lambda c: MagicMock(content=c, type="system")
    )

    mock_lc_tools = MagicMock()
    # tool(name_or_callable, fn=None, ...) — first arg is name string when called as
    # tool(name, fn, ...), or the callable itself when called as tool(fn, ...).
    mock_lc_tools.tool = MagicMock(
        side_effect=lambda *args, **kw: (
            args[1] if len(args) > 1 and callable(args[1]) else args[0]
        )
    )

    mock_chat_model = MagicMock()
    mock_chat_model.ainvoke = AsyncMock(return_value=ai_msg)
    mock_chat_model.bind_tools = MagicMock(return_value=mock_chat_model)

    mock_lc_openai = MagicMock()
    mock_lc_openai.ChatOpenAI = MagicMock(return_value=mock_chat_model)
    mock_lc_openai.ChatOpenAI.model_fields = _REAL_CHAT_OPENAI.model_fields

    mock_gm = MagicMock()
    mock_gm.add_messages = MagicMock(return_value=MagicMock())

    mock_types = MagicMock()
    mock_types.Command = MagicMock(side_effect=lambda **kw: kw)

    return {
        "langgraph.graph": mock_langgraph_graph,
        "langgraph.prebuilt": mock_prebuilt,
        "langchain_core.messages": mock_lc_msgs,
        "langchain_core.tools": mock_lc_tools,
        "langchain_openai": mock_lc_openai,
        "langgraph.types": mock_types,
        "langgraph.graph.message": mock_gm,
        "_node_fns": node_fns,
        "_edges": edges_added,
        "_chat_model": mock_chat_model,
    }


def _capture_compiled_state(mocks: dict[str, Any], ai_msg: Any) -> list[dict[str, Any]]:
    """Swaps in a StateGraph whose compiled graph records the state it is invoked
    with, so a test can assert on the root's initial messages."""
    states: list[dict[str, Any]] = []

    class _CapturingStateGraph:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        def add_node(self, *a: Any, **kw: Any) -> None:
            pass

        def add_edge(self, *a: Any, **kw: Any) -> None:
            pass

        def add_conditional_edges(self, *a: Any, **kw: Any) -> None:
            pass

        def compile(self) -> Any:
            compiled = MagicMock()

            async def _ainvoke(state: dict[str, Any]) -> Any:
                states.append(state)
                return {"messages": [ai_msg]}

            compiled.ainvoke = _ainvoke
            return compiled

    mocks["langgraph.graph"].StateGraph = _CapturingStateGraph
    return states


@contextmanager
def _patch_imports(mocks: dict[str, Any]) -> Any:
    """Patch sys.modules so importlib.import_module picks up our mocks."""
    _skip = {"_node_fns", "_edges", "_chat_model"}
    module_map = {k: v for k, v in mocks.items() if k not in _skip}

    with patch.dict(sys.modules, module_map, clear=False):
        yield


# ---------------------------------------------------------------------------
# Generic topology
# ---------------------------------------------------------------------------


def _recording_trace() -> tuple[Any, Any, list[Any]]:
    """A real tracer behind the module's ``trace`` lookup, with every span's ``end`` spied.

    Returns the stand-in ``trace`` module, the exporter holding finished spans, and the
    ``end`` spies, so a test can check both what was exported and how often each span ended.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    end_spies: list[Any] = []

    def _start_span(name: str, *args: Any, **kwargs: Any) -> Any:
        span = tracer.start_span(name, *args, **kwargs)
        span.end = MagicMock(wraps=span.end)  # type: ignore[method-assign]
        end_spies.append(span.end)
        return span

    fake_trace = MagicMock()
    fake_trace.get_tracer.return_value.start_span.side_effect = _start_span
    return fake_trace, exporter, end_spies


def _assert_one_failed_graph_span(
    exporter: Any, end_spies: list[Any], message: str, track_calls: list[Any]
) -> None:
    from opentelemetry.trace import StatusCode

    assert [spy.call_count for spy in end_spies] == [1]
    spans = [s for s in exporter.get_finished_spans() if s.name == "launchdarkly.graph"]
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == message
    assert [e.name for e in span.events].count("exception") == 1
    assert (span.attributes or {}).get("launchdarkly.graph.key") == "test-graph"

    failures = [d for e, d in track_calls if e == "$ld:ai:graph:invocation_failure"]
    assert len(failures) == 1
    assert failures[0]["configKey"] == "test-graph"
    assert failures[0]["graphKey"] == "test-graph"


def _assert_two_runs_two_run_ids(
    exporter: Any, track_calls_per_run: list[list[tuple[str, Any]]]
) -> None:
    """Each run's graph span carries the ``runId`` of that run's graph-level events,
    and the two runs carry different ids."""
    spans = [s for s in exporter.get_finished_spans() if s.name == "launchdarkly.graph"]
    assert len(spans) == 2
    span_run_ids = [(s.attributes or {}).get("launchdarkly.run.id") for s in spans]
    for span_run_id, run_calls in zip(span_run_ids, track_calls_per_run, strict=True):
        assert span_run_id
        graph_events = [d for e, d in run_calls if e.startswith("$ld:ai:graph:")]
        assert graph_events
        for data in graph_events:
            assert data["runId"] == span_run_id
    assert span_run_ids[0] != span_run_ids[1]


class TestToLangGraphTopology:
    @pytest.mark.asyncio
    async def test_each_graph_node_translated(self) -> None:
        ai_msg = _make_ai_msg("final")
        mocks = _make_langgraph_mocks(ai_msg)
        graph_def = _make_graph_def()

        with _patch_imports(mocks):
            result = await to_lang_graph(_make_def_promise(graph_def)).invoke("hi")

        assert result["response"] == "final"

    @pytest.mark.asyncio
    async def test_root_node_is_entry_point(self) -> None:
        ai_msg = _make_ai_msg("answer")
        mocks = _make_langgraph_mocks(ai_msg)
        graph_def = _make_graph_def()

        with _patch_imports(mocks):
            await to_lang_graph(_make_def_promise(graph_def)).invoke("input-text")

        # __start__ edge should be added to root node
        start_edges = [e for e in mocks["_edges"] if e[0] == "__start__"]
        assert start_edges

    @pytest.mark.asyncio
    async def test_runner_returns_final_output(self) -> None:
        ai_msg = _make_ai_msg("my-answer")
        mocks = _make_langgraph_mocks(ai_msg)
        graph_def = _make_graph_def()

        with _patch_imports(mocks):
            result = await to_lang_graph(_make_def_promise(graph_def)).invoke(
                "question"
            )

        assert result["response"] == "my-answer"


# ---------------------------------------------------------------------------
# LangChain-specific specs
# ---------------------------------------------------------------------------


class TestToLangGraphLangChainSpecific:
    @pytest.mark.asyncio
    async def test_disabled_graph_throws(self) -> None:
        ai_msg = _make_ai_msg()
        mocks = _make_langgraph_mocks(ai_msg)
        graph_def = _make_graph_def(enabled=False)

        with _patch_imports(mocks):
            with pytest.raises(ValueError, match="disabled"):
                await to_lang_graph(_make_def_promise(graph_def)).invoke("hi")

    @pytest.mark.asyncio
    async def test_default_chat_openai_receives_model_parameters(self) -> None:
        ai_msg = _make_ai_msg("final")
        mocks = _make_langgraph_mocks(ai_msg)
        graph_def = _make_graph_def(
            nodes={
                "root": {
                    "key": "root",
                    "config": {
                        "model": {
                            "name": "gpt-4o",
                            "parameters": {
                                "temperature": 0.2,
                                "max_tokens": 512,
                                "tools": [{"name": "openai-tool"}],
                            },
                        },
                        "instructions": "help",
                    },
                    "meta": {"variationKey": "v1", "version": 1},
                    "edges": [],
                    "is_terminal": True,
                }
            }
        )

        async def _visit(fn: Any, ctx: Any = None) -> None:
            if graph_def.root is not None:
                await fn(graph_def.root)

        graph_def.traverse = _visit

        with _patch_imports(mocks):
            await to_lang_graph(_make_def_promise(graph_def)).invoke("hi")

        mocks["langchain_openai"].ChatOpenAI.assert_called_once()
        assert mocks["langchain_openai"].ChatOpenAI.call_args.kwargs == {
            "temperature": 0.2,
            "max_tokens": 512,
            "model": "gpt-4o",
        }

    @pytest.mark.asyncio
    async def test_null_root_throws(self) -> None:
        ai_msg = _make_ai_msg()
        mocks = _make_langgraph_mocks(ai_msg)
        # nodes has key "other-node" but root_key is "root" → root=None
        graph_def = _make_graph_def(
            nodes={
                "other-node": {
                    "key": "other-node",
                    "config": {"model": {"name": "gpt-4o"}, "instructions": "hi"},
                    "meta": {},
                    "edges": [],
                    "is_terminal": True,
                }
            },
            root_key="root",
        )

        with _patch_imports(mocks):
            with pytest.raises(ValueError):
                await to_lang_graph(_make_def_promise(graph_def)).invoke("hi")

    @pytest.mark.asyncio
    async def test_two_node_graph_calls_add_node_twice(self) -> None:
        ai_msg = _make_ai_msg()
        mocks = _make_langgraph_mocks(ai_msg)

        child = {
            "key": "child",
            "config": {"model": {"name": "gpt-4o"}, "instructions": "child"},
            "meta": {},
            "edges": [],
            "is_terminal": True,
        }
        root = {
            "key": "root",
            "config": {"model": {"name": "gpt-4o"}, "instructions": "root"},
            "meta": {},
            "edges": [{"source_key": "root", "target_key": "child"}],
            "is_terminal": False,
        }
        graph_def = _make_graph_def(
            nodes={"root": root, "child": child},
            edges=[{"source_key": "root", "target_key": "child"}],
        )

        with _patch_imports(mocks):
            await to_lang_graph(_make_def_promise(graph_def)).invoke("hi")

        assert len(mocks["_node_fns"]) >= 2

    @pytest.mark.asyncio
    async def test_root_node_connected_from_start(self) -> None:
        ai_msg = _make_ai_msg()
        mocks = _make_langgraph_mocks(ai_msg)
        graph_def = _make_graph_def()

        with _patch_imports(mocks):
            await to_lang_graph(_make_def_promise(graph_def)).invoke("hi")

        start_edges = [e for e in mocks["_edges"] if e[0] == "__start__"]
        assert start_edges

    @pytest.mark.asyncio
    async def test_call_returns_response_and_usage(self) -> None:
        ai_msg = _make_ai_msg("answer", input_tokens=20, output_tokens=8)
        mocks = _make_langgraph_mocks(ai_msg)
        graph_def = _make_graph_def()

        with _patch_imports(mocks):
            result = await to_lang_graph(_make_def_promise(graph_def)).invoke("hi")

        assert "response" in result
        assert "usage" in result

    @pytest.mark.asyncio
    async def test_extract_usage_when_usage_metadata_absent(self) -> None:
        msg = MagicMock(spec=[])  # spec=[] means no attributes
        usage = _extract_usage(msg)
        assert usage == {"input": 0, "output": 0, "total": 0}

    @pytest.mark.asyncio
    async def test_invocation_success_and_duration_tracked(self) -> None:
        track_calls: list[str] = []
        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: track_calls.append(evt)
        )

        ai_msg = _make_ai_msg("done")
        mocks = _make_langgraph_mocks(ai_msg)
        graph_def = _make_graph_def()
        ctx = {"kind": "user", "key": "test"}

        with _patch_imports(mocks):
            with patch(
                "launchdarkly_ai_langchain_agents.native_graph.get_client",
                return_value=mock_ld_client,
            ):
                await to_lang_graph(
                    _make_def_promise(graph_def),
                    opts={"context": ctx},
                ).invoke("hi")

        assert "$ld:ai:graph:invocation_success" in track_calls
        assert "$ld:ai:graph:path" not in track_calls

    @pytest.mark.asyncio
    async def test_graph_span_carries_config_identity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The graph span must identify the config so Monitoring can link the trace."""
        import launchdarkly_ai_langchain_agents.native_graph as ng_mod

        monkeypatch.setenv("LD_ENVIRONMENT_ID", "env-123")
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        ai_msg = _make_ai_msg("done")
        mocks = _make_langgraph_mocks(ai_msg)
        graph_def = _make_graph_def()
        ctx = {"kind": "user", "key": "u1"}

        with _patch_imports(mocks):
            with patch.object(ng_mod, "get_client", return_value=MagicMock()):
                with patch.object(ng_mod, "trace", mock_trace):
                    with patch.object(ng_mod, "_HAS_OTEL", True):
                        await to_lang_graph(
                            _make_def_promise(graph_def),
                            opts={"context": ctx},
                        ).invoke("hi")

        attrs = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert attrs["launchdarkly.operation.type"] == "gen_ai"
        assert attrs["launchdarkly.config.key"] == "test-graph"
        assert attrs["launchdarkly.graph.key"] == "test-graph"
        assert attrs["launchdarkly.run.id"]
        assert "launchdarkly.variation.key" in attrs
        assert attrs["context.contextKeys.user"] == "u1"

        events = {c[0][0]: c[0][1] for c in mock_span.add_event.call_args_list}
        assert events["feature_flag"] == {
            "feature_flag.key": "test-graph",
            "feature_flag.provider.name": "LaunchDarkly",
            "feature_flag.set.id": "env-123",
            "feature_flag.context.id": "u1",
            "feature_flag.contextKeys": '{"user":"u1"}',
        }

    @pytest.mark.asyncio
    async def test_tracking_events_carry_the_environment_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LD_ENVIRONMENT_ID", "env-123")
        payloads: list[Any] = []
        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: payloads.append(data)
        )

        ai_msg = _make_ai_msg("done")
        mocks = _make_langgraph_mocks(ai_msg)
        graph_def = _make_graph_def()
        ctx = {"kind": "user", "key": "u1"}

        with _patch_imports(mocks):
            with patch(
                "launchdarkly_ai_langchain_agents.native_graph.get_client",
                return_value=mock_ld_client,
            ):
                await to_lang_graph(
                    _make_def_promise(graph_def),
                    opts={"context": ctx},
                ).invoke("hi")
                node_fn = mocks["_node_fns"]["root"]
                await node_fn({"messages": []})

        assert payloads
        for data in payloads:
            assert data["environmentId"] == "env-123"

    @pytest.mark.asyncio
    async def test_graph_events_are_keyed_to_the_graph(self) -> None:
        """Graph-level events carry the graph key, like the graph span and graph()."""
        calls: list[tuple[str, Any]] = []
        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: calls.append((evt, data))
        )

        mocks = _make_langgraph_mocks(_make_ai_msg("done"))
        graph_def = _make_graph_def()
        ctx = {"kind": "user", "key": "u1"}

        with _patch_imports(mocks):
            with patch(
                "launchdarkly_ai_langchain_agents.native_graph.get_client",
                return_value=mock_ld_client,
            ):
                await to_lang_graph(
                    _make_def_promise(graph_def),
                    opts={"context": ctx},
                ).invoke("hi")

        graph_events = [
            (e, d)
            for e, d in calls
            if e
            in (
                "$ld:ai:graph:invocation_success",
                "$ld:ai:graph:duration:total",
                "$ld:ai:graph:total_tokens",
            )
        ]
        assert len(graph_events) == 3
        for _, data in graph_events:
            assert data["configKey"] == "test-graph"
            assert data["graphKey"] == "test-graph"

    @pytest.mark.asyncio
    async def test_each_runner_call_gets_its_own_run_id(self) -> None:
        """Two calls on one runner: each span shares its run's id, and the ids differ."""
        import launchdarkly_ai_langchain_agents.native_graph as ng_mod

        fake_trace, exporter, _ = _recording_trace()
        track_calls: list[tuple[str, Any]] = []
        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: track_calls.append((evt, data))
        )
        mocks = _make_langgraph_mocks(_make_ai_msg("done"))
        graph_def = _make_graph_def()

        with _patch_imports(mocks):
            with patch.object(ng_mod, "trace", fake_trace):
                with patch.object(ng_mod, "_HAS_OTEL", True):
                    with patch.object(
                        ng_mod, "get_client", return_value=mock_ld_client
                    ):
                        # A task, unlike a bare coroutine, can be awaited on each call.
                        runner = to_lang_graph(
                            asyncio.ensure_future(_make_def_promise(graph_def)),
                            opts={"context": {"kind": "user", "key": "u1"}},
                        )
                        await runner.invoke("first")
                        first_run_calls = len(track_calls)
                        await runner.invoke("second")

        _assert_two_runs_two_run_ids(
            exporter,
            [track_calls[:first_run_calls], track_calls[first_run_calls:]],
        )

    @pytest.mark.asyncio
    async def test_graph_setup_error_fails_the_graph_span(self) -> None:
        """A compile error before the run ends one ERROR span and tracks the failure."""
        import launchdarkly_ai_langchain_agents.native_graph as ng_mod

        fake_trace, exporter, end_spies = _recording_trace()
        track_calls: list[tuple[str, Any]] = []
        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: track_calls.append((evt, data))
        )

        mocks = _make_langgraph_mocks(_make_ai_msg("done"))

        class _BadCompileStateGraph:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            def add_node(self, *a: Any, **kw: Any) -> None:
                pass

            def add_edge(self, *a: Any, **kw: Any) -> None:
                pass

            def add_conditional_edges(self, *a: Any, **kw: Any) -> None:
                pass

            def compile(self) -> Any:
                raise ValueError("bad graph")

        mocks["langgraph.graph"].StateGraph = _BadCompileStateGraph
        graph_def = _make_graph_def()

        with _patch_imports(mocks):
            with patch.object(ng_mod, "trace", fake_trace):
                with patch.object(ng_mod, "_HAS_OTEL", True):
                    with patch.object(
                        ng_mod, "get_client", return_value=mock_ld_client
                    ):
                        with pytest.raises(ValueError, match="bad graph"):
                            await to_lang_graph(
                                _make_def_promise(graph_def),
                                opts={"context": {"kind": "user", "key": "u1"}},
                            ).invoke("hi")

        _assert_one_failed_graph_span(exporter, end_spies, "bad graph", track_calls)

    @pytest.mark.asyncio
    async def test_node_function_emits_graph_node(self) -> None:
        track_calls: list[tuple[str, Any, Any]] = []
        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: track_calls.append((evt, data, val))
        )

        ai_msg = _make_ai_msg("done")
        mocks = _make_langgraph_mocks(ai_msg)
        graph_def = _make_graph_def()
        ctx = {"kind": "user", "key": "test"}

        with _patch_imports(mocks):
            with patch(
                "launchdarkly_ai_langchain_agents.native_graph.get_client",
                return_value=mock_ld_client,
            ):
                await to_lang_graph(
                    _make_def_promise(graph_def),
                    opts={"context": ctx},
                ).invoke("hi")
                node_fn = mocks["_node_fns"]["root"]
                await node_fn({"messages": []})

        node_events = [item for item in track_calls if item[0] == "$ld:ai:graph:node"]
        assert len(node_events) == 1
        assert node_events[0][1]["nodeKey"] == "root"
        assert node_events[0][1]["index"] == 0
        assert node_events[0][2] == 1

    @pytest.mark.asyncio
    async def test_invocation_failure_tracked(self) -> None:
        track_calls: list[str] = []
        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: track_calls.append(evt)
        )

        ai_msg = _make_ai_msg()
        mocks = _make_langgraph_mocks(ai_msg)

        # Make the compiled graph raise
        class _FailStateGraph:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            def add_node(self, *a: Any, **kw: Any) -> None:
                pass

            def add_edge(self, *a: Any, **kw: Any) -> None:
                pass

            def add_conditional_edges(self, *a: Any, **kw: Any) -> None:
                pass

            def compile(self) -> Any:
                c = MagicMock()
                c.ainvoke = AsyncMock(side_effect=RuntimeError("graph fail"))
                return c

        mocks["langgraph.graph"].StateGraph = _FailStateGraph
        graph_def = _make_graph_def()
        ctx = {"kind": "user", "key": "test"}

        with _patch_imports(mocks):
            with patch(
                "launchdarkly_ai_langchain_agents.native_graph.get_client",
                return_value=mock_ld_client,
            ):
                with pytest.raises(RuntimeError):
                    await to_lang_graph(
                        _make_def_promise(graph_def),
                        opts={"context": ctx},
                    ).invoke("hi")

        assert "$ld:ai:graph:invocation_failure" in track_calls

    @pytest.mark.asyncio
    async def test_no_context_path_no_tracking(self) -> None:
        mock_ld_client = MagicMock()
        ai_msg = _make_ai_msg()
        mocks = _make_langgraph_mocks(ai_msg)
        graph_def = _make_graph_def()

        with _patch_imports(mocks):
            with patch(
                "launchdarkly_ai_langchain_agents.native_graph.get_client",
                return_value=mock_ld_client,
            ):
                await to_lang_graph(_make_def_promise(graph_def)).invoke("hi")

        mock_ld_client.track.assert_not_called()

    @pytest.mark.asyncio
    async def test_span_end_called_on_success(self) -> None:
        import launchdarkly_ai_langchain_agents.native_graph as ng_mod

        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        ai_msg = _make_ai_msg()
        mocks = _make_langgraph_mocks(ai_msg)
        graph_def = _make_graph_def()

        with _patch_imports(mocks):
            with patch.object(ng_mod, "trace", mock_trace):
                with patch.object(ng_mod, "_HAS_OTEL", True):
                    await to_lang_graph(_make_def_promise(graph_def)).invoke("hi")

        mock_span.end.assert_called()

    @pytest.mark.asyncio
    async def test_span_end_called_on_error(self) -> None:
        import launchdarkly_ai_langchain_agents.native_graph as ng_mod

        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        ai_msg = _make_ai_msg()
        mocks = _make_langgraph_mocks(ai_msg)

        class _FailStateGraph:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            def add_node(self, *a: Any, **kw: Any) -> None:
                pass

            def add_edge(self, *a: Any, **kw: Any) -> None:
                pass

            def add_conditional_edges(self, *a: Any, **kw: Any) -> None:
                pass

            def compile(self) -> Any:
                c = MagicMock()
                c.ainvoke = AsyncMock(side_effect=RuntimeError("fail"))
                return c

        mocks["langgraph.graph"].StateGraph = _FailStateGraph
        graph_def = _make_graph_def()

        with _patch_imports(mocks):
            with patch.object(ng_mod, "trace", mock_trace):
                with patch.object(ng_mod, "_HAS_OTEL", True):
                    with pytest.raises(RuntimeError):
                        await to_lang_graph(_make_def_promise(graph_def)).invoke("hi")

        mock_span.end.assert_called()

    @pytest.mark.asyncio
    async def test_otel_span_has_graph_key_attribute(self) -> None:
        import launchdarkly_ai_langchain_agents.native_graph as ng_mod

        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        ai_msg = _make_ai_msg()
        mocks = _make_langgraph_mocks(ai_msg)
        graph_def = _make_graph_def()

        with _patch_imports(mocks):
            with patch.object(ng_mod, "trace", mock_trace):
                with patch.object(ng_mod, "_HAS_OTEL", True):
                    await to_lang_graph(_make_def_promise(graph_def)).invoke("hi")

        mock_trace.get_tracer.return_value.start_span.assert_called_with(
            "launchdarkly.graph"
        )
        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert "launchdarkly.graph.key" in calls

    @pytest.mark.asyncio
    async def test_terminal_leaf_connected_to_end(self) -> None:
        """Terminal leaf node with no tools must be connected to END via add_edge."""
        ai_msg = _make_ai_msg()
        mocks = _make_langgraph_mocks(ai_msg)

        child = {
            "key": "child",
            "config": {"model": {"name": "gpt-4o"}, "instructions": "child"},
            "meta": {},
            "edges": [],
            "is_terminal": True,
        }
        root = {
            "key": "root",
            "config": {"model": {"name": "gpt-4o"}, "instructions": "root"},
            "meta": {},
            "edges": [{"source_key": "root", "target_key": "child"}],
            "is_terminal": False,
        }
        edges = [{"source_key": "root", "target_key": "child"}]
        graph_def = _make_graph_def(nodes={"root": root, "child": child}, edges=edges)

        with _patch_imports(mocks):
            await to_lang_graph(_make_def_promise(graph_def)).invoke("hi")

        end_edges = [e for e in mocks["_edges"] if e[1] == "__end__"]
        assert any("child" in e[0] for e in end_edges), (
            f"Expected child node connected to END, edges: {mocks['_edges']}"
        )

    @pytest.mark.asyncio
    async def test_non_terminal_node_has_transfer_to_tools(self) -> None:
        """Non-terminal nodes must have transfer_to_<target> handoff tools built."""
        ai_msg = _make_ai_msg()
        tool_names_built: list[str] = []

        # Track tool() calls to capture tool names.
        # tool() is called as tool(name, fn, description=...) so name is args[0].
        mock_lc_tools_spy = MagicMock()

        def _tool_spy(*args: Any, **kw: Any) -> Any:
            name_arg = (
                args[0] if args and isinstance(args[0], str) else kw.get("name", "")
            )
            tool_names_built.append(name_arg)
            return args[1] if len(args) > 1 and callable(args[1]) else args[0]

        mock_lc_tools_spy.tool = MagicMock(side_effect=_tool_spy)

        child = {
            "key": "child",
            "config": {"model": {"name": "gpt-4o"}, "instructions": "child"},
            "meta": {},
            "edges": [],
            "is_terminal": True,
        }
        root = {
            "key": "root",
            "config": {"model": {"name": "gpt-4o"}, "instructions": "root"},
            "meta": {},
            "edges": [{"source_key": "root", "target_key": "child"}],
            "is_terminal": False,
        }
        edges = [{"source_key": "root", "target_key": "child"}]
        graph_def = _make_graph_def(nodes={"root": root, "child": child}, edges=edges)

        mocks = _make_langgraph_mocks(ai_msg)
        mocks["langchain_core.tools"] = mock_lc_tools_spy

        with _patch_imports(mocks):
            await to_lang_graph(_make_def_promise(graph_def)).invoke("hi")

        transfer_tools = [n for n in tool_names_built if n.startswith("transfer_to_")]
        assert transfer_tools, f"Expected transfer_to_* tools, got: {tool_names_built}"
        assert "transfer_to_child" in transfer_tools

    @pytest.mark.asyncio
    async def test_terminal_node_has_no_transfer_to_tools(self) -> None:
        """Terminal nodes must not have handoff (transfer_to_*) tools injected."""
        ai_msg = _make_ai_msg()
        tool_names_built: list[str] = []

        mock_lc_tools_spy = MagicMock()

        def _tool_spy(*args: Any, **kw: Any) -> Any:
            name_arg = (
                args[0] if args and isinstance(args[0], str) else kw.get("name", "")
            )
            tool_names_built.append(name_arg)
            return args[1] if len(args) > 1 and callable(args[1]) else args[0]

        mock_lc_tools_spy.tool = MagicMock(side_effect=_tool_spy)

        # Single terminal root with no edges
        mocks = _make_langgraph_mocks(ai_msg)
        mocks["langchain_core.tools"] = mock_lc_tools_spy
        graph_def = _make_graph_def()

        with _patch_imports(mocks):
            await to_lang_graph(_make_def_promise(graph_def)).invoke("hi")

        transfer_tools = [n for n in tool_names_built if n.startswith("transfer_to_")]
        assert not transfer_tools, (
            f"Terminal node should have no handoff tools, got: {transfer_tools}"
        )

    @pytest.mark.asyncio
    async def test_instructions_produce_system_message(self) -> None:
        """config.instructions must become a SystemMessage as the first message in node_fn."""
        ai_msg = _make_ai_msg()
        mocks = _make_langgraph_mocks(ai_msg)
        graph_def = _make_graph_def()

        with _patch_imports(mocks):
            await to_lang_graph(_make_def_promise(graph_def)).invoke("hi")

        node_fn = mocks["_node_fns"].get("root")
        assert node_fn is not None, "root node function not captured"

        captured_messages: list[Any] = []

        async def _capture_invoke(msgs: list[Any]) -> Any:
            captured_messages.extend(msgs)
            return ai_msg

        mocks["_chat_model"].ainvoke = _capture_invoke

        with _patch_imports(mocks):
            await node_fn({"messages": []})

        system_msgs = [
            m for m in captured_messages if getattr(m, "type", None) == "system"
        ]
        assert system_msgs, f"No SystemMessage found in: {captured_messages}"
        assert (
            "help" in system_msgs[0].content
        )  # "help" is the test graph's instructions

    @pytest.mark.asyncio
    async def test_system_prompt_from_config_messages(self) -> None:
        """When config.messages has a system role entry, it must be used as system prompt."""
        ai_msg = _make_ai_msg()
        mocks = _make_langgraph_mocks(ai_msg)

        graph_def = _make_graph_def(
            nodes={
                "root": {
                    "key": "root",
                    "config": {
                        "model": {"name": "gpt-4o"},
                        "messages": [
                            {"role": "system", "content": "you are an expert"}
                        ],
                    },
                    "meta": {"variationKey": "v1", "version": 1},
                    "edges": [],
                    "is_terminal": True,
                }
            }
        )

        with _patch_imports(mocks):
            await to_lang_graph(_make_def_promise(graph_def)).invoke("hi")

        node_fn = mocks["_node_fns"].get("root")
        assert node_fn is not None

        captured_messages: list[Any] = []

        async def _capture_invoke(msgs: list[Any]) -> Any:
            captured_messages.extend(msgs)
            return ai_msg

        mocks["_chat_model"].ainvoke = _capture_invoke

        with _patch_imports(mocks):
            await node_fn({"messages": []})

        system_msgs = [
            m for m in captured_messages if getattr(m, "type", None) == "system"
        ]
        assert system_msgs, "No SystemMessage found"
        assert "expert" in system_msgs[0].content

    @pytest.mark.asyncio
    async def test_no_history_seeds_root_with_plain_human_message(self) -> None:
        ai_msg = _make_ai_msg()
        mocks = _make_langgraph_mocks(ai_msg)
        states = _capture_compiled_state(mocks, ai_msg)
        graph_def = _make_graph_def()

        with _patch_imports(mocks):
            await to_lang_graph(_make_def_promise(graph_def)).invoke("hi")

        messages = states[0]["messages"]
        assert len(messages) == 1
        assert messages[0].content == "hi"

    @pytest.mark.asyncio
    async def test_history_seeds_root_with_native_image_content(self) -> None:
        """A multimodal history reaches the root as LangChain image content."""
        ai_msg = _make_ai_msg()
        mocks = _make_langgraph_mocks(ai_msg)
        states = _capture_compiled_state(mocks, ai_msg)
        graph_def = _make_graph_def()

        history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "abc123",
                        },
                    }
                ],
            }
        ]

        with _patch_imports(mocks):
            await to_lang_graph(_make_def_promise(graph_def)).invoke(
                "describe", {}, history
            )

        serialized = json.dumps([m.content for m in states[0]["messages"]])
        assert "image_url" in serialized or '"type": "image"' in serialized
        assert "abc123" in serialized
        assert "describe" in serialized

    @pytest.mark.asyncio
    async def test_config_tools_creates_tool_node(self) -> None:
        """When config.tools is non-empty, a ToolNode must be created and wired."""
        ai_msg = _make_ai_msg()
        mocks = _make_langgraph_mocks(ai_msg)

        graph_def = _make_graph_def(
            nodes={
                "root": {
                    "key": "root",
                    "config": {
                        "model": {"name": "gpt-4o"},
                        "instructions": "use the calculator",
                        "tools": {
                            "calculate": {"description": "do math", "parameters": {}}
                        },
                    },
                    "meta": {"variationKey": "v1", "version": 1},
                    "edges": [],
                    "is_terminal": True,
                }
            }
        )

        with _patch_imports(mocks):
            await to_lang_graph(_make_def_promise(graph_def)).invoke("hi")

        assert mocks["langgraph.prebuilt"].ToolNode.called, (
            "ToolNode was not called despite config.tools being non-empty"
        )


class TestBuildNodeToolsSyncHandlers:
    """``_build_node_tools`` must accept sync handlers (native path skips wrap_tool_handlers)."""

    @pytest.mark.asyncio
    async def test_sync_handler_returns_string(self) -> None:
        captured: list[Any] = []

        def _capture_tool(_name: Any, fn: Any = None, **_kw: Any) -> Any:
            captured.append(fn)
            return fn

        mock_lc_tools = MagicMock()
        mock_lc_tools.tool = MagicMock(side_effect=_capture_tool)

        node = _to_graph_node(
            {
                "key": "root",
                "config": {
                    "tools": {"weather": {"description": "d", "parameters": {}}},
                },
                "meta": {},
                "edges": [],
                "is_terminal": True,
            }
        )

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_lc_tools if n == "langchain_core.tools" else __import__(n)
            ),
        ):
            _build_node_tools(node, {"weather": lambda a: "sunny"})

        assert captured, "tool was not registered"
        assert await captured[0](city="Paris") == "sunny"

    @pytest.mark.asyncio
    async def test_async_handler_is_awaited(self) -> None:
        captured: list[Any] = []

        def _capture_tool(_name: Any, fn: Any = None, **_kw: Any) -> Any:
            captured.append(fn)
            return fn

        mock_lc_tools = MagicMock()
        mock_lc_tools.tool = MagicMock(side_effect=_capture_tool)

        async def async_handler(args: dict[str, Any]) -> str:
            return f"async:{args.get('q')}"

        node = _to_graph_node(
            {
                "key": "root",
                "config": {
                    "tools": {"lookup": {"description": "d", "parameters": {}}},
                },
                "meta": {},
                "edges": [],
                "is_terminal": True,
            }
        )

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_lc_tools if n == "langchain_core.tools" else __import__(n)
            ),
        ):
            _build_node_tools(node, {"lookup": async_handler})

        assert await captured[0](q="hi") == "async:hi"


# ---------------------------------------------------------------------------
# WorkflowState annotations resolve — real StateGraph (no mock)
# ---------------------------------------------------------------------------


class TestWorkflowStateAnnotationsResolve:
    """
    Guards against NameError from from __future__ import annotations deferring
    annotation evaluation: LangGraph calls get_type_hints(WorkflowState) during
    StateGraph(WorkflowState), which resolves symbols in the *module* global
    namespace.  Any annotation symbol that is only a local variable inside the
    invoke() closure will raise NameError at that point.

    This test does NOT mock StateGraph or langgraph.graph so the real
    get_type_hints() path is exercised.  Only the LLM call is mocked.
    """

    @pytest.mark.asyncio
    async def test_workflow_state_annotations_resolve(self) -> None:
        """to_lang_graph must build a StateGraph without NameError when the real
        langgraph package is available.  Only the ChatOpenAI.ainvoke is mocked."""
        # We need the real langgraph.graph.StateGraph to exercise get_type_hints.
        # Skip the test if langgraph is not installed.
        pytest.importorskip("langgraph")

        from unittest.mock import AsyncMock, MagicMock, patch

        from langchain_core.messages import AIMessage  # type: ignore[import]

        ai_msg = AIMessage(content="resolved answer")
        ai_msg.usage_metadata = {"input_tokens": 5, "output_tokens": 3}  # type: ignore[assignment]

        mock_chat_model = MagicMock()
        mock_chat_model.ainvoke = AsyncMock(return_value=ai_msg)
        mock_chat_model.bind_tools = MagicMock(return_value=mock_chat_model)

        def model_factory(_node: Any) -> Any:
            return mock_chat_model

        graph_def = _make_graph_def()

        # Patch only the OTel tracer so we don't need a real OTel setup, and
        # patch ChatOpenAI so no real HTTP call is made.
        with patch("launchdarkly_ai_langchain_agents.native_graph._HAS_OTEL", False):
            result = await to_lang_graph(
                _make_def_promise(graph_def),
                {"model_factory": model_factory},
            ).invoke("hello")

        assert isinstance(result, dict)
        assert "response" in result


class TestNativeGraphNeverForwardedParameters:
    @pytest.mark.asyncio
    async def test_default_chat_openai_receives_no_never_forwarded_key(self) -> None:
        ai_msg = _make_ai_msg("final")
        mocks = _make_langgraph_mocks(ai_msg)
        graph_def = _make_graph_def(
            nodes={
                "root": {
                    "key": "root",
                    "config": {
                        "model": {
                            "name": "gpt-4o",
                            "parameters": {**NEVER_FORWARDED_BAG, "temperature": 0.2},
                        },
                        "instructions": "help",
                    },
                    "meta": {"variationKey": "v1", "version": 1},
                    "edges": [],
                    "is_terminal": True,
                }
            }
        )

        async def _visit(fn: Any, ctx: Any = None) -> None:
            if graph_def.root is not None:
                await fn(graph_def.root)

        graph_def.traverse = _visit

        with _patch_imports(mocks):
            await to_lang_graph(_make_def_promise(graph_def)).invoke("hi")

        assert mocks["langchain_openai"].ChatOpenAI.call_args.kwargs == {
            "temperature": 0.2,
            "model": "gpt-4o",
        }


class TestNativeGraphForwardsExactlyTheCrossSdkList:
    """The native graph's default ``ChatOpenAI`` takes the same forwarded list as the handler's,
    so a probe of one node finds exactly the cross-SDK ChatOpenAI list."""

    @pytest.mark.asyncio
    async def test_default_chat_openai(self) -> None:
        async def call(parameters: dict[str, Any]) -> object:
            mocks = _make_langgraph_mocks(_make_ai_msg("final"))
            graph_def = _make_graph_def(
                nodes={
                    "root": {
                        "key": "root",
                        "config": {
                            "model": {"name": "gpt-4o", "parameters": parameters},
                            "instructions": "help",
                        },
                        "meta": {"variationKey": "v1", "version": 1},
                        "edges": [],
                        "is_terminal": True,
                    }
                }
            )

            async def _visit(fn: Any, ctx: Any = None) -> None:
                if graph_def.root is not None:
                    await fn(graph_def.root)

            graph_def.traverse = _visit
            with _patch_imports(mocks):
                await to_lang_graph(_make_def_promise(graph_def)).invoke("hi")
            return mocks["langchain_openai"].ChatOpenAI.call_args.kwargs

        expected = LANGCHAIN_CHAT_OPENAI
        candidates = candidate_keys(_REAL_CHAT_OPENAI.model_fields, expected)
        assert await probe_forwarded_keys(candidates, call) == expected


# ---------------------------------------------------------------------------
# Hand-offs through a real StateGraph (AIC-3592)
# ---------------------------------------------------------------------------


def _handoff_graph_def(
    children: list[str], root_tools: dict[str, Any] | None = None
) -> GraphDefinition:
    def _node(
        key: str, edges: list[str], tools: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "model": {"name": "m"},
            "instructions": f"you are {key}",
        }
        if tools:
            config["tools"] = tools
        return {
            "key": key,
            "config": config,
            "meta": {"variationKey": "v1", "version": 1},
            "edges": [{"source_key": key, "target_key": c} for c in edges],
            "is_terminal": not edges,
        }

    nodes = {"router": _node("router", children, root_tools)}
    nodes.update({c: _node(c, []) for c in children})
    return _make_graph_def(
        nodes=nodes,
        edges=[{"source_key": "router", "target_key": c} for c in children],
        root_key="router",
    )


def _scripted_models(replies: dict[str, list[Any]]) -> tuple[Any, dict[str, Any]]:
    """One mock chat model per node; each ainvoke pops that node's next reply.
    bind_tools is a MagicMock, so a test can assert on its kwargs."""
    models: dict[str, Any] = {}
    for key, msgs in replies.items():
        m = MagicMock()
        m.ainvoke = AsyncMock(side_effect=list(msgs))
        m.bind_tools = MagicMock(return_value=m)
        models[key] = m
    return (lambda node: models[node.key]), models


def _tool_call_msg(*calls: tuple[Any, ...], content: Any = "") -> Any:
    """An AI message with tool calls; each call is (name, id) or (name, id, args)."""
    from langchain_core.messages import AIMessage

    return AIMessage(
        content=content,
        tool_calls=[
            {
                "name": c[0],
                "args": c[2] if len(c) > 2 else {},
                "id": c[1],
                "type": "tool_call",
            }
            for c in calls
        ],
    )


async def _run_handoff(
    graph_def: GraphDefinition,
    factory: Any,
    tool_handlers: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    from langchain_core.messages import AIMessage  # noqa: F401

    tracked: list[str] = []
    client = MagicMock()
    client.track = MagicMock(side_effect=lambda evt, *_a: tracked.append(evt))
    with (
        patch("launchdarkly_ai_langchain_agents.native_graph._HAS_OTEL", False),
        patch(
            "launchdarkly_ai_langchain_agents.native_graph.get_client",
            return_value=client,
        ),
    ):
        result = await to_lang_graph(
            _make_def_promise(graph_def),
            {
                "model_factory": factory,
                "tool_handlers": tool_handlers or {},
                "context": {"kind": "user", "key": "u"},
            },
        ).invoke("who did it?")
    return result, tracked


def _sent(model: Any, call: int = 0) -> list[tuple[str, Any]]:
    """(type, content) of each message a node's model received on one call."""
    return [(m.type, m.content) for m in model.ainvoke.call_args_list[call].args[0]]


def _assert_neutral(model: Any) -> None:
    """No tool role, tool calls or non-text blocks reach a node that was handed to."""
    for m in model.ainvoke.call_args.args[0]:
        assert m.type != "tool"
        assert not getattr(m, "tool_calls", None)
        assert isinstance(m.content, str)


# Same literal as the TypeScript tests' REASONING, so the two rewrite tests share one input.
_REASONING = {"type": "reasoning_content", "reasoning_content": {"text": "hmm"}}


class TestNativeHandoffRealStateGraph:
    @pytest.mark.asyncio
    async def test_multi_child_router_hands_off_once(self) -> None:
        from langchain_core.messages import AIMessage

        factory, models = _scripted_models(
            {
                "router": [
                    _tool_call_msg(
                        ("transfer_to_manor_butler", "call_1"),
                        content=[_REASONING, {"type": "text", "text": "Routing."}],
                    )
                ],
                "manor.butler": [AIMessage(content="the butler did it")],
                "cook": [],
            }
        )
        result, tracked = await _run_handoff(
            _handoff_graph_def(["manor.butler", "cook"]), factory
        )

        assert result["response"] == "the butler did it"
        assert models["router"].ainvoke.call_count == 1  # not re-run after the hand-off
        assert models["cook"].ainvoke.call_count == 0
        # A target with no tools gets the router's turn as plain text: reasoning
        # dropped, the call and its result as lines. The result text uses the raw
        # target key; the tool name uses the sanitized one.
        assert _sent(models["manor.butler"]) == [
            ("system", "you are manor.butler"),
            ("human", "who did it?"),
            ("ai", "Routing.\n[Tool call] transfer_to_manor_butler {}"),
            (
                "human",
                "[Tool result] transfer_to_manor_butler: Handed off to manor.butler.",
            ),
        ]
        _assert_neutral(models["manor.butler"])
        assert tracked.count("$ld:ai:graph:handoff_success") == 1

    @pytest.mark.asyncio
    async def test_bind_tools_gets_no_parallel_tool_calls(self) -> None:
        """Bedrock Converse rejects parallel_tool_calls, so it is never passed."""
        from langchain_core.messages import AIMessage

        factory, models = _scripted_models(
            {
                "router": [_tool_call_msg(("transfer_to_butler", "call_1"))],
                "butler": [AIMessage(content="done")],
                "cook": [],
            }
        )
        await _run_handoff(_handoff_graph_def(["butler", "cook"]), factory)

        assert models["router"].bind_tools.call_args.kwargs == {}

    @pytest.mark.asyncio
    async def test_normal_tool_and_handoff_in_one_message_both_run(self) -> None:
        from langchain_core.messages import AIMessage

        lookups: list[Any] = []
        factory, models = _scripted_models(
            {
                "router": [
                    _tool_call_msg(
                        ("lookup", "call_a", {"room": "library"}),
                        ("transfer_to_butler", "call_b"),
                    )
                ],
                "butler": [AIMessage(content="done")],
                "cook": [],
            }
        )
        result, _ = await _run_handoff(
            _handoff_graph_def(
                ["butler", "cook"],
                root_tools={
                    "lookup": {
                        "description": "look up",
                        "parameters": {
                            "type": "object",
                            "properties": {"room": {"type": "string"}},
                        },
                    }
                },
            ),
            factory,
            tool_handlers={"lookup": lambda args: lookups.append(args) or "clue"},
        )

        assert result["response"] == "done"
        assert lookups == [{"room": "library"}]
        assert models["router"].ainvoke.call_count == 1
        # Both results become one user message, in the call order.
        assert _sent(models["butler"])[2:] == [
            (
                "ai",
                '[Tool call] lookup {"room":"library"}\n[Tool call] transfer_to_butler {}',
            ),
            (
                "human",
                "[Tool result] lookup: clue\n"
                "[Tool result] transfer_to_butler: Handed off to butler.",
            ),
        ]
        _assert_neutral(models["butler"])

    @pytest.mark.asyncio
    async def test_first_handoff_wins(self) -> None:
        from langchain_core.messages import AIMessage

        factory, models = _scripted_models(
            {
                "router": [
                    _tool_call_msg(
                        ("transfer_to_butler", "call_1"), ("transfer_to_cook", "call_2")
                    )
                ],
                "butler": [AIMessage(content="butler answer")],
                "cook": [],
            }
        )
        result, tracked = await _run_handoff(
            _handoff_graph_def(["butler", "cook"]), factory
        )

        assert result["response"] == "butler answer"
        assert models["cook"].ainvoke.call_count == 0
        assert _sent(models["butler"])[-1] == (
            "human",
            "[Tool result] transfer_to_butler: Handed off to butler.\n"
            "[Tool result] transfer_to_cook: Ignored: only the first transfer in a turn routes.",
        )
        assert tracked.count("$ld:ai:graph:handoff_success") == 1

    @pytest.mark.asyncio
    async def test_single_child_hands_off_once(self) -> None:
        from langchain_core.messages import AIMessage

        factory, models = _scripted_models(
            {
                "router": [_tool_call_msg(("transfer_to_butler", "call_1"))],
                "butler": [AIMessage(content="the butler did it")],
            }
        )
        result, _ = await _run_handoff(_handoff_graph_def(["butler"]), factory)

        assert result["response"] == "the butler did it"
        assert models["router"].ainvoke.call_count == 1
        assert _sent(models["butler"])[2:] == [
            ("ai", "[Tool call] transfer_to_butler {}"),
            ("human", "[Tool result] transfer_to_butler: Handed off to butler."),
        ]
        _assert_neutral(models["butler"])

    @pytest.mark.asyncio
    async def test_own_tool_loop_stays_native(self) -> None:
        """Only the history from before a node was entered is rewritten: a node's
        own tool call and its result reach its next model call unchanged."""
        from langchain_core.messages import AIMessage

        factory, models = _scripted_models(
            {
                "router": [
                    _tool_call_msg(("lookup", "call_x"), content=[_REASONING]),
                    AIMessage(content="answer"),
                ],
            }
        )
        result, _ = await _run_handoff(
            _handoff_graph_def(
                [],
                root_tools={
                    "lookup": {
                        "description": "look up",
                        "parameters": {"type": "object", "properties": {}},
                    }
                },
            ),
            factory,
            tool_handlers={"lookup": lambda args: "clue"},
        )

        assert result["response"] == "answer"
        second = models["router"].ainvoke.call_args_list[1].args[0]
        assert [m.type for m in second] == ["system", "human", "ai", "tool"]
        assert second[2].tool_calls[0]["id"] == "call_x"
        assert second[2].content == [_REASONING]
        assert second[3].tool_call_id == "call_x"


class TestToNeutralHistory:
    """The rewrite rules in spec §2.x.3, on one input that hits each of them.
    The TypeScript test uses the same input and expects the same output."""

    def test_rewrites_every_case(self) -> None:
        from langchain_core.messages import (
            AIMessage,
            AIMessageChunk,
            HumanMessage,
            SystemMessage,
            ToolMessage,
            ToolMessageChunk,
        )

        image = [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:x"}},
        ]
        history = [
            SystemMessage("sys"),
            HumanMessage(image),
            AIMessage(
                content=[
                    _REASONING,
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": None},
                    {"type": "text", "text": "b"},
                ],
                tool_calls=[
                    {
                        "name": "first",
                        "args": {"q": "café", "n": {"x": [1, 2]}},
                        "id": "c1",
                    },
                    {"name": "second", "args": {}, "id": "c2"},
                ],
            ),
            ToolMessage("orphan", tool_call_id="zz", name="lost"),
            ToolMessage([{"type": "text", "text": "two"}], tool_call_id="c2"),
            ToolMessage("one", tool_call_id="c1"),
            AIMessage(content=[_REASONING]),
            AIMessageChunk(
                content="chunk", tool_calls=[{"name": "t", "args": {}, "id": "c3"}]
            ),
            ToolMessageChunk(content="three", tool_call_id="c3"),
        ]

        out = _to_neutral_history(history, AIMessage, HumanMessage)

        assert [(m.type, m.content) for m in out] == [
            ("system", "sys"),
            ("human", image),
            (
                "ai",
                "ab\n"
                '[Tool call] first {"q":"café","n":{"x":[1,2]}}\n'
                "[Tool call] second {}",
            ),
            (
                "human",
                "[Tool result] first: one\n"
                "[Tool result] second: two\n"
                "[Tool result] lost: orphan",
            ),
            ("ai", "chunk\n[Tool call] t {}"),
            ("human", "[Tool result] t: three"),
        ]
        assert all(not getattr(m, "tool_calls", None) for m in out)
