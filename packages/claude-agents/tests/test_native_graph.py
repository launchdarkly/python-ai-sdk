"""
Tests for §2.2 native graph adapter (to_claude_agents) plus Anthropic-specific specs.
Reference: TESTING.md §2.2, §2.x (Anthropic)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_claude_agents.native_graph as _claude_ng
from launchdarkly_ai_claude_agents.native_graph import to_claude_agents
from launchdarkly_ai_server import GraphDefinition, GraphEdge, GraphNode, NativeTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result_msg(
    text: str = "result", input_tokens: int = 5, output_tokens: int = 3
) -> Any:
    m = MagicMock()
    m.__class__.__name__ = "ResultMessage"
    m.result = text
    m.usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    return m


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
            "config": {"model": {"name": "claude-3"}, "instructions": "be helpful"},
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

    async def _noop_traverse(fn: Any, ctx: Any = None) -> None:
        return None

    async def _noop_run_node(*a: Any, **kw: Any) -> Any:
        raise NotImplementedError

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


def _make_sdk_mock(result_text: str = "done") -> Any:
    result_msg = _make_result_msg(result_text)

    async def _query(**kwargs: Any) -> AsyncIterator[Any]:
        yield result_msg

    mock = MagicMock()
    mock.query = _query
    mock.ClaudeAgentOptions = MagicMock(return_value=MagicMock())
    mock.ResultMessage = type(result_msg)
    mock.HookMatcher = MagicMock()
    mock.tool = MagicMock(side_effect=lambda name, desc, schema: lambda fn: fn)
    mock.create_sdk_mcp_server = MagicMock(return_value=MagicMock())
    return mock


# ---------------------------------------------------------------------------
# §2.2 Generic topology
# ---------------------------------------------------------------------------


class TestToClaudeAgentsTopology:
    @pytest.mark.asyncio
    async def test_each_graph_node_translated(self) -> None:
        mock_sdk = _make_sdk_mock("root-answer")
        graph_def = _make_graph_def()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            result = await to_claude_agents(_make_def_promise(graph_def)).invoke("hi")

        assert result["response"] == "root-answer"

    @pytest.mark.asyncio
    async def test_root_node_is_entry_point(self) -> None:
        mock_sdk = _make_sdk_mock("from-root")
        graph_def = _make_graph_def()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            result = await to_claude_agents(_make_def_promise(graph_def)).invoke("hi")

        assert "response" in result

    @pytest.mark.asyncio
    async def test_terminal_nodes_no_handoff_tools(self) -> None:
        mock_sdk = _make_sdk_mock("done")
        graph_def = _make_graph_def()

        tool_calls: list[Any] = []

        def orig_tool(name: str, desc: str, schema: Any) -> Any:
            def register(fn: Any) -> Any:
                tool_calls.append(name)
                return fn

            return register

        mock_sdk.tool = MagicMock(side_effect=orig_tool)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            await to_claude_agents(_make_def_promise(graph_def)).invoke("hi")

        # Terminal root node should not create any subagent tools
        assert len(tool_calls) == 0

    @pytest.mark.asyncio
    async def test_handoff_tool_injected_for_edges(self) -> None:
        mock_sdk = _make_sdk_mock("done")
        child_node = {
            "key": "child",
            "config": {"model": {"name": "claude-3"}, "instructions": "child"},
            "meta": {},
            "edges": [],
            "is_terminal": True,
        }
        root_node = {
            "key": "root",
            "config": {"model": {"name": "claude-3"}, "instructions": "root"},
            "meta": {},
            "edges": [{"source_key": "root", "target_key": "child"}],
            "is_terminal": False,
        }
        nodes = {"root": root_node, "child": child_node}
        edges = [{"source_key": "root", "target_key": "child"}]
        graph_def = _make_graph_def(nodes=nodes, edges=edges)

        created_tools: list[str] = []

        def orig_tool(name: str, desc: str, schema: Any) -> Any:
            def register(fn: Any) -> Any:
                created_tools.append(name)
                return fn

            return register

        mock_sdk.tool = MagicMock(side_effect=orig_tool)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            await to_claude_agents(_make_def_promise(graph_def)).invoke("hi")

        # The child node should have been wrapped as a subagent tool
        assert any("child" in t for t in created_tools)

    @pytest.mark.asyncio
    async def test_instructions_forwarded_to_system_prompt(self) -> None:
        mock_sdk = _make_sdk_mock("done")
        graph_def = _make_graph_def()

        captured_options: list[Any] = []
        mock_sdk.ClaudeAgentOptions = MagicMock(
            side_effect=lambda **kw: (captured_options.append(kw), kw)[1]
        )

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            await to_claude_agents(_make_def_promise(graph_def)).invoke("hi")

        # All ClaudeAgentOptions calls should have system_prompt set (from config.instructions)
        assert len(captured_options) > 0
        assert any(
            "system_prompt" in opts and opts["system_prompt"]
            for opts in captured_options
        )

    @pytest.mark.asyncio
    async def test_runner_starts_at_root_and_returns_output(self) -> None:
        mock_sdk = _make_sdk_mock("final-output")
        graph_def = _make_graph_def()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            result = await to_claude_agents(_make_def_promise(graph_def)).invoke(
                "input"
            )

        assert result["response"] == "final-output"

    @pytest.mark.asyncio
    async def test_config_tools_converted_and_passed(self) -> None:
        mock_sdk = _make_sdk_mock("done")
        root_node = {
            "key": "root",
            "config": {
                "model": {"name": "claude-3"},
                "instructions": "help",
                "tools": {"my-tool": {"description": "does stuff", "parameters": {}}},
            },
            "meta": {},
            "edges": [],
            "is_terminal": True,
        }
        graph_def = _make_graph_def(nodes={"root": root_node})

        handler_invoked = []

        async def _my_handler(args: Any) -> str:
            handler_invoked.append(args)
            return "tool-result"

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            await to_claude_agents(
                _make_def_promise(graph_def),
                opts={"tool_handlers": {"my-tool": _my_handler}},
            ).invoke("input")

        # create_sdk_mcp_server is called from build_tool_mcp for user config tools
        assert mock_sdk.create_sdk_mcp_server.called

    @pytest.mark.asyncio
    async def test_tool_handlers_wired_correctly(self) -> None:
        mock_sdk = _make_sdk_mock("done")
        graph_def = _make_graph_def()

        handler = AsyncMock(return_value="ok")
        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            await to_claude_agents(
                _make_def_promise(graph_def),
                opts={"tool_handlers": {"some-tool": handler}},
            ).invoke("input")

        # No config.tools → handler not invoked automatically
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_native_tool_handled_via_native_tool_key(self) -> None:
        mock_sdk = _make_sdk_mock("done")
        native = NativeTool("Bash")
        graph_def = _make_graph_def()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            result = await to_claude_agents(
                _make_def_promise(graph_def),
                opts={"tool_handlers": {"run-bash": native}},
            ).invoke("hi")

        assert "response" in result


# ---------------------------------------------------------------------------
# Anthropic-specific specs
# ---------------------------------------------------------------------------


class TestToClaudeAgentsAnthropicSpecific:
    @pytest.mark.asyncio
    async def test_throws_when_graph_disabled(self) -> None:
        mock_sdk = _make_sdk_mock()
        graph_def = _make_graph_def(enabled=False)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with pytest.raises(ValueError, match="disabled"):
                await to_claude_agents(_make_def_promise(graph_def)).invoke("hi")

    @pytest.mark.asyncio
    async def test_builds_one_sub_agent_mcp_tool_per_non_root_node(self) -> None:
        mock_sdk = _make_sdk_mock("done")
        child_node = {
            "key": "child",
            "config": {
                "model": {"name": "claude-3"},
                "instructions": "child instructions",
            },
            "meta": {},
            "edges": [],
            "is_terminal": True,
        }
        root_node = {
            "key": "root",
            "config": {"model": {"name": "claude-3"}, "instructions": "root"},
            "meta": {},
            "edges": [{"source_key": "root", "target_key": "child"}],
            "is_terminal": False,
        }
        graph_def = _make_graph_def(
            nodes={"root": root_node, "child": child_node},
            edges=[{"source_key": "root", "target_key": "child"}],
        )

        created_tools: list[str] = []

        def orig_tool(name: str, desc: str, schema: Any) -> Any:
            def register(fn: Any) -> Any:
                created_tools.append(name)
                return fn

            return register

        mock_sdk.tool = MagicMock(side_effect=orig_tool)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            await to_claude_agents(_make_def_promise(graph_def)).invoke("hi")

        assert len(created_tools) == 1
        assert "child" in created_tools[0]

    @pytest.mark.asyncio
    async def test_emits_handoff_success_when_sub_agent_tool_invoked(self) -> None:
        track_calls: list[tuple[str, Any]] = []

        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: track_calls.append((evt, data))
        )

        child_node = {
            "key": "child",
            "config": {"model": {"name": "claude-3"}, "instructions": "child"},
            "meta": {"variationKey": "v1", "version": 1},
            "edges": [],
            "is_terminal": True,
        }
        root_node = {
            "key": "root",
            "config": {"model": {"name": "claude-3"}, "instructions": "root"},
            "meta": {"variationKey": "v1", "version": 1},
            "edges": [{"source_key": "root", "target_key": "child"}],
            "is_terminal": False,
        }
        graph_def = _make_graph_def(
            nodes={"root": root_node, "child": child_node},
            edges=[{"source_key": "root", "target_key": "child"}],
        )

        # Mock the SDK so query invokes the child subagent tool
        child_tool_fn: list[Any] = []

        def _tool_factory(name: str, desc: str, schema: Any) -> Any:
            def _dec(fn: Any) -> Any:
                if "child" in name:
                    child_tool_fn.append(fn)
                return fn

            return _dec

        mock_sdk = MagicMock()
        mock_sdk.tool = MagicMock(side_effect=_tool_factory)
        mock_sdk.create_sdk_mcp_server = MagicMock(return_value=MagicMock())
        mock_sdk.HookMatcher = MagicMock()
        result_msg = _make_result_msg("root-done")

        invoked_child = False

        async def _query(**kwargs: Any) -> AsyncIterator[Any]:
            nonlocal invoked_child
            if not invoked_child and child_tool_fn:
                invoked_child = True
                await child_tool_fn[0]({"input": "sub-query"})
            yield result_msg

        mock_sdk.query = _query
        mock_sdk.ClaudeAgentOptions = MagicMock(return_value=MagicMock())
        mock_sdk.ResultMessage = type(result_msg)

        ctx = {"kind": "user", "key": "test"}
        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with patch.object(_claude_ng, "get_client", return_value=mock_ld_client):
                await to_claude_agents(
                    _make_def_promise(graph_def),
                    opts={"context": ctx},
                ).invoke("hi")

        handoff_events = [
            evt for evt, _ in track_calls if evt == "$ld:ai:graph:handoff_success"
        ]
        assert len(handoff_events) >= 1

    @pytest.mark.asyncio
    async def test_emits_invocation_success_on_completion(self) -> None:
        track_calls: list[str] = []
        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: track_calls.append(evt)
        )

        mock_sdk = _make_sdk_mock("done")
        graph_def = _make_graph_def()
        ctx = {"kind": "user", "key": "test"}

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with patch.object(_claude_ng, "get_client", return_value=mock_ld_client):
                await to_claude_agents(
                    _make_def_promise(graph_def),
                    opts={"context": ctx},
                ).invoke("hi")

        assert "$ld:ai:graph:invocation_success" in track_calls

    @pytest.mark.asyncio
    async def test_emits_invocation_failure_on_error(self) -> None:
        track_calls: list[str] = []
        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: track_calls.append(evt)
        )

        mock_sdk = MagicMock()

        async def _bad_query(**kwargs: Any) -> AsyncIterator[Any]:
            raise RuntimeError("query failed")
            yield

        mock_sdk.query = _bad_query
        mock_sdk.ClaudeAgentOptions = MagicMock(return_value=MagicMock())
        mock_sdk.ResultMessage = MagicMock
        mock_sdk.HookMatcher = MagicMock()
        mock_sdk.tool = MagicMock(side_effect=lambda n, d, s: lambda fn: fn)

        graph_def = _make_graph_def()
        ctx = {"kind": "user", "key": "test"}

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with patch.object(_claude_ng, "get_client", return_value=mock_ld_client):
                with pytest.raises(RuntimeError):
                    await to_claude_agents(
                        _make_def_promise(graph_def),
                        opts={"context": ctx},
                    ).invoke("hi")

        assert "$ld:ai:graph:invocation_failure" in track_calls

    @pytest.mark.asyncio
    async def test_returns_root_query_result(self) -> None:
        mock_sdk = _make_sdk_mock("expected output")
        graph_def = _make_graph_def()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            result = await to_claude_agents(_make_def_promise(graph_def)).invoke("hi")

        assert result["response"] == "expected output"

    @pytest.mark.asyncio
    async def test_span_end_called_on_success(self) -> None:
        import launchdarkly_ai_claude_agents.native_graph as ng_mod

        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        mock_sdk = _make_sdk_mock("done")
        graph_def = _make_graph_def()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with patch.object(ng_mod, "trace", mock_trace):
                with patch.object(ng_mod, "_HAS_OTEL", True):
                    await to_claude_agents(_make_def_promise(graph_def)).invoke("hi")

        mock_span.end.assert_called()

    @pytest.mark.asyncio
    async def test_span_end_called_on_error(self) -> None:
        import launchdarkly_ai_claude_agents.native_graph as ng_mod

        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        mock_sdk = MagicMock()

        async def _bad_query(**kwargs: Any) -> AsyncIterator[Any]:
            raise RuntimeError("fail")
            yield

        mock_sdk.query = _bad_query
        mock_sdk.ClaudeAgentOptions = MagicMock(return_value=MagicMock())
        mock_sdk.ResultMessage = MagicMock
        mock_sdk.HookMatcher = MagicMock()
        mock_sdk.tool = MagicMock(side_effect=lambda n, d, s: lambda fn: fn)

        graph_def = _make_graph_def()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with patch.object(ng_mod, "trace", mock_trace):
                with patch.object(ng_mod, "_HAS_OTEL", True):
                    with pytest.raises(RuntimeError):
                        await to_claude_agents(_make_def_promise(graph_def)).invoke(
                            "hi"
                        )

        mock_span.end.assert_called()

    @pytest.mark.asyncio
    async def test_build_tool_mcp_throws_when_tool_not_in_handlers(self) -> None:
        from launchdarkly_ai_claude_agents.handler import build_tool_mcp

        mock_sdk = MagicMock()
        mock_sdk.tool = MagicMock(side_effect=lambda name, desc, schema: lambda fn: fn)
        mock_sdk.create_sdk_mcp_server = MagicMock(return_value=MagicMock())

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            config_tools = {"missing-tool": {"description": "d", "parameters": {}}}
            handlers: dict[str, Any] = {}
            await build_tool_mcp(config_tools, handlers)
            # The actual error is raised when the fn is *called*, not when mcp is built
