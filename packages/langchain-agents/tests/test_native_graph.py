"""
Tests for §2.2 native graph adapter (to_lang_graph) and LangChain-specific specs.
Reference: TESTING.md §2.2, §2.x.3
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from launchdarkly_ai_langchain_agents.native_graph import _extract_usage, to_lang_graph
from launchdarkly_ai_server import GraphDefinition, GraphEdge, GraphNode

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


@contextmanager
def _patch_imports(mocks: dict[str, Any]) -> Any:
    """Patch sys.modules so importlib.import_module picks up our mocks."""
    _skip = {"_node_fns", "_edges", "_chat_model"}
    module_map = {k: v for k, v in mocks.items() if k not in _skip}

    with patch.dict(sys.modules, module_map, clear=False):
        yield


# ---------------------------------------------------------------------------
# §2.2 Generic topology
# ---------------------------------------------------------------------------


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
# §2.x.3 LangChain-specific specs
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

        mock_trace.get_tracer.return_value.start_span.assert_called_with("ld.ai.graph")
        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert "ld.ai.graph.key" in calls

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


# ---------------------------------------------------------------------------
# §2.x.3 WorkflowState annotations resolve — real StateGraph (no mock)
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
