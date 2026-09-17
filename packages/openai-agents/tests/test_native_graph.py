"""
Tests for §2.2 native graph adapter (to_openai_agents) and OpenAI-specific specs.
Reference: TESTING.md §2.2, §2.x.6
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_openai_agents.native_graph as _openai_ng
from launchdarkly_ai_openai_agents.native_graph import to_openai_agents
from launchdarkly_ai_server import GraphDefinition, GraphEdge, GraphNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_run_result(output: str = "done") -> Any:
    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 5
    usage.total_tokens = 15
    raw_resp = MagicMock()
    raw_resp.usage = usage
    result = MagicMock()
    result.final_output = output
    result.raw_responses = [raw_resp]
    return result


def _make_agents_mock(run_result: Any) -> Any:
    mock = MagicMock()

    created_agents: list[Any] = []

    def _make_agent(**kw: Any) -> MagicMock:
        a = MagicMock()
        a.name = kw.get("name", "agent")
        a._kw = kw
        created_agents.append(a)
        return a

    mock.Agent = MagicMock(side_effect=_make_agent)
    mock.Runner.run = AsyncMock(return_value=run_result)
    mock.handoff = MagicMock(side_effect=lambda agent: agent)
    mock.tool = MagicMock(side_effect=lambda **kw: lambda fn: fn)

    # RunHooks base class
    class _FakeRunHooks:
        async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
            pass

        async def on_handoff(
            self, context: Any, from_agent: Any, to_agent: Any
        ) -> None:
            pass

        async def on_agent_start(self, context: Any, agent: Any) -> None:
            pass

    mock.RunHooks = _FakeRunHooks
    mock._created_agents = created_agents
    return mock


# ---------------------------------------------------------------------------
# §2.2 Generic topology
# ---------------------------------------------------------------------------


class TestToOpenAIAgentsTopology:
    @pytest.mark.asyncio
    async def test_each_graph_node_translated(self) -> None:
        run_result = _make_run_result("out")
        agents_mock = _make_agents_mock(run_result)
        graph_def = _make_graph_def()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            result = await to_openai_agents(_make_def_promise(graph_def)).invoke("hi")

        assert result["response"] == "out"

    @pytest.mark.asyncio
    async def test_root_node_is_entry_point(self) -> None:
        run_result = _make_run_result("out")
        agents_mock = _make_agents_mock(run_result)
        graph_def = _make_graph_def()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            await to_openai_agents(_make_def_promise(graph_def)).invoke("input-text")

        # Runner.run should be called with the root agent and input text
        agents_mock.Runner.run.assert_called_once()
        call_args = agents_mock.Runner.run.call_args
        assert "input-text" in call_args[0] or "input-text" == call_args[0][1]

    @pytest.mark.asyncio
    async def test_multimodal_history_is_structured_root_input(self) -> None:
        run_result = _make_run_result("out")
        agents_mock = _make_agents_mock(run_result)
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

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            await to_openai_agents(_make_def_promise(graph_def)).invoke(
                "describe", None, history
            )

        root_input = agents_mock.Runner.run.call_args.args[1]
        assert isinstance(root_input, list)
        serialized = str(root_input)
        assert "input_image" in serialized
        assert "data:image/png;base64,abc123" in serialized

    @pytest.mark.asyncio
    async def test_terminal_nodes_no_handoff_tools(self) -> None:
        run_result = _make_run_result("out")
        agents_mock = _make_agents_mock(run_result)
        graph_def = _make_graph_def()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            await to_openai_agents(_make_def_promise(graph_def)).invoke("hi")

        # Terminal root with no edges → handoff never called
        agents_mock.handoff.assert_not_called()

    @pytest.mark.asyncio
    async def test_handoff_tool_injected_for_edges(self) -> None:
        run_result = _make_run_result("out")
        agents_mock = _make_agents_mock(run_result)

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

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            await to_openai_agents(_make_def_promise(graph_def)).invoke("hi")

        # handoff() should be called for the child agent
        agents_mock.handoff.assert_called()

    @pytest.mark.asyncio
    async def test_runner_starts_at_root_and_returns_output(self) -> None:
        run_result = _make_run_result("final-output")
        agents_mock = _make_agents_mock(run_result)
        graph_def = _make_graph_def()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            result = await to_openai_agents(_make_def_promise(graph_def)).invoke(
                "input"
            )

        assert result["response"] == "final-output"


# ---------------------------------------------------------------------------
# §2.x.6 OpenAI-specific specs
# ---------------------------------------------------------------------------


class TestToOpenAIAgentsOpenAISpecific:
    @pytest.mark.asyncio
    async def test_disabled_graph_throws(self) -> None:
        run_result = _make_run_result()
        agents_mock = _make_agents_mock(run_result)
        graph_def = _make_graph_def(enabled=False)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with pytest.raises(ValueError, match="disabled"):
                await to_openai_agents(_make_def_promise(graph_def)).invoke("hi")

    @pytest.mark.asyncio
    async def test_null_root_throws(self) -> None:
        run_result = _make_run_result()
        agents_mock = _make_agents_mock(run_result)
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

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with pytest.raises(ValueError):
                await to_openai_agents(_make_def_promise(graph_def)).invoke("hi")

    @pytest.mark.asyncio
    async def test_two_node_graph_creates_two_agents(self) -> None:
        run_result = _make_run_result("out")
        agents_mock = _make_agents_mock(run_result)

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

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            await to_openai_agents(_make_def_promise(graph_def)).invoke("hi")

        assert agents_mock.Agent.call_count == 2

    @pytest.mark.asyncio
    async def test_returns_response_and_usage(self) -> None:
        run_result = _make_run_result("answer")
        agents_mock = _make_agents_mock(run_result)
        graph_def = _make_graph_def()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            result = await to_openai_agents(_make_def_promise(graph_def)).invoke("hi")

        assert result["response"] == "answer"
        assert "usage" in result

    @pytest.mark.asyncio
    async def test_success_path_emits_invocation_success_and_tokens(self) -> None:
        track_calls: list[str] = []
        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: track_calls.append(evt)
        )

        run_result = _make_run_result("done")
        agents_mock = _make_agents_mock(run_result)
        graph_def = _make_graph_def()
        ctx = {"kind": "user", "key": "test"}

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(_openai_ng, "get_client", return_value=mock_ld_client):
                await to_openai_agents(
                    _make_def_promise(graph_def),
                    opts={"context": ctx},
                ).invoke("hi")

        assert "$ld:ai:graph:invocation_success" in track_calls

    @pytest.mark.asyncio
    async def test_error_path_emits_invocation_failure_and_rethrows(self) -> None:
        track_calls: list[str] = []
        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: track_calls.append(evt)
        )

        agents_mock = MagicMock()
        agents_mock.Agent = MagicMock(return_value=MagicMock(name="root"))
        agents_mock.Runner.run = AsyncMock(side_effect=RuntimeError("provider error"))
        agents_mock.handoff = MagicMock()
        agents_mock.tool = MagicMock(side_effect=lambda **kw: lambda fn: fn)

        class _FakeRunHooks:
            async def on_agent_end(self, *a: Any) -> None:
                pass

            async def on_handoff(self, *a: Any) -> None:
                pass

            async def on_agent_start(self, *a: Any) -> None:
                pass

        agents_mock.RunHooks = _FakeRunHooks
        graph_def = _make_graph_def()
        ctx = {"kind": "user", "key": "test"}

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(_openai_ng, "get_client", return_value=mock_ld_client):
                with pytest.raises(RuntimeError, match="provider error"):
                    await to_openai_agents(
                        _make_def_promise(graph_def),
                        opts={"context": ctx},
                    ).invoke("hi")

        assert "$ld:ai:graph:invocation_failure" in track_calls

    @pytest.mark.asyncio
    async def test_no_tracking_when_context_omitted(self) -> None:
        mock_ld_client = MagicMock()
        run_result = _make_run_result("done")
        agents_mock = _make_agents_mock(run_result)
        graph_def = _make_graph_def()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(_openai_ng, "get_client", return_value=mock_ld_client):
                await to_openai_agents(_make_def_promise(graph_def)).invoke("hi")

        mock_ld_client.track.assert_not_called()

    @pytest.mark.asyncio
    async def test_span_end_called_on_success(self) -> None:
        import launchdarkly_ai_openai_agents.native_graph as ng_mod

        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        run_result = _make_run_result("done")
        agents_mock = _make_agents_mock(run_result)
        graph_def = _make_graph_def()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(ng_mod, "trace", mock_trace):
                with patch.object(ng_mod, "_HAS_OTEL", True):
                    await to_openai_agents(_make_def_promise(graph_def)).invoke("hi")

        mock_span.end.assert_called()

    @pytest.mark.asyncio
    async def test_span_end_called_on_error(self) -> None:
        import launchdarkly_ai_openai_agents.native_graph as ng_mod

        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        agents_mock = MagicMock()
        agents_mock.Agent = MagicMock(return_value=MagicMock(name="root"))
        agents_mock.Runner.run = AsyncMock(side_effect=RuntimeError("fail"))
        agents_mock.handoff = MagicMock()
        agents_mock.tool = MagicMock(side_effect=lambda **kw: lambda fn: fn)

        class _FakeRunHooks:
            async def on_agent_end(self, *a: Any) -> None:
                pass

            async def on_handoff(self, *a: Any) -> None:
                pass

            async def on_agent_start(self, *a: Any) -> None:
                pass

        agents_mock.RunHooks = _FakeRunHooks
        graph_def = _make_graph_def()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(ng_mod, "trace", mock_trace):
                with patch.object(ng_mod, "_HAS_OTEL", True):
                    with pytest.raises(RuntimeError):
                        await to_openai_agents(_make_def_promise(graph_def)).invoke(
                            "hi"
                        )

        mock_span.end.assert_called()

    @pytest.mark.asyncio
    async def test_total_tokens_tracked_on_success(self) -> None:
        """Success path emits $ld:ai:graph:total_tokens."""
        track_calls: list[str] = []
        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: track_calls.append(evt)
        )

        run_result = _make_run_result("done")
        agents_mock = _make_agents_mock(run_result)
        graph_def = _make_graph_def()
        ctx = {"kind": "user", "key": "test"}

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(_openai_ng, "get_client", return_value=mock_ld_client):
                await to_openai_agents(
                    _make_def_promise(graph_def),
                    opts={"context": ctx},
                ).invoke("hi")

        assert "$ld:ai:graph:total_tokens" in track_calls

    @pytest.mark.asyncio
    async def test_otel_span_has_graph_key_attribute(self) -> None:
        """OTel span must have ld.ai.graph.key attribute set to the graph key."""
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        run_result = _make_run_result("done")
        agents_mock = _make_agents_mock(run_result)
        graph_def = _make_graph_def()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(_openai_ng, "trace", mock_trace):
                with patch.object(_openai_ng, "_HAS_OTEL", True):
                    await to_openai_agents(_make_def_promise(graph_def)).invoke("hi")

        set_attr_calls = {
            c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list
        }
        assert "ld.ai.graph.key" in set_attr_calls
        assert set_attr_calls["ld.ai.graph.key"] == "test-graph"

    @pytest.mark.asyncio
    async def test_agent_end_hook_emits_generation_success(self) -> None:
        """agent_end hook must emit $ld:ai:generation:success for the agent's node."""
        track_calls: list[str] = []
        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: track_calls.append(evt)
        )

        captured_hooks: list[Any] = []
        run_result = _make_run_result("done")
        agents_mock = _make_agents_mock(run_result)

        # Fire on_agent_end inside Runner.run so the get_client patch is still active
        async def _run_and_fire_hook(agent: Any, text: str, hooks: Any = None) -> Any:
            if hooks:
                captured_hooks.append(hooks)
                root_agent_mock = MagicMock()
                root_agent_mock.name = "root"
                await hooks.on_agent_end(None, root_agent_mock, "output")
            return run_result

        agents_mock.Runner.run = _run_and_fire_hook
        graph_def = _make_graph_def()
        ctx = {"kind": "user", "key": "test"}

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(_openai_ng, "get_client", return_value=mock_ld_client):
                await to_openai_agents(
                    _make_def_promise(graph_def),
                    opts={"context": ctx},
                ).invoke("hi")

        assert captured_hooks, "hooks were not passed to Runner.run"
        assert "$ld:ai:generation:success" in track_calls

    @pytest.mark.asyncio
    async def test_path_entries_are_unique(self) -> None:
        """§2.x.6 — each node key must appear at most once in path.
        When on_handoff adds the child key, on_agent_start must not add it again.
        """
        # Two-node graph: root -> child
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

        _captured_path: list[str] = []
        run_result = _make_run_result("done")
        agents_mock = _make_agents_mock(run_result)

        # Fire both on_handoff (for child) and on_agent_start (for child)
        # to simulate what the OpenAI Agents SDK would do on a real handoff.
        async def _run_and_fire_hooks(agent: Any, text: str, hooks: Any = None) -> Any:
            if hooks:
                from_agent_mock = MagicMock()
                from_agent_mock.name = "root"
                to_agent_mock = MagicMock()
                to_agent_mock.name = "child"
                # on_handoff adds child to path
                await hooks.on_handoff(None, from_agent_mock, to_agent_mock)
                # on_agent_start should NOT add child again
                await hooks.on_agent_start(None, to_agent_mock)
            return run_result

        agents_mock.Runner.run = _run_and_fire_hooks

        # Capture OTel path attribute to inspect the path list
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(_openai_ng, "trace", mock_trace):
                with patch.object(_openai_ng, "_HAS_OTEL", True):
                    await to_openai_agents(_make_def_promise(graph_def)).invoke("hi")

        # Extract the path from the span set_attribute call for "ld.ai.graph.path"
        path_val: str | None = None
        for call in mock_span.set_attribute.call_args_list:
            if call[0][0] == "ld.ai.graph.path":
                path_val = call[0][1]
                break

        assert path_val is not None, "ld.ai.graph.path attribute was not set"
        path_parts = [p for p in path_val.split("->") if p]
        child_occurrences = path_parts.count("child")
        assert child_occurrences <= 1, (
            f"'child' appeared {child_occurrences} times in path '{path_val}'. "
            "Each node key must appear at most once (on_agent_start must not re-add keys already in path)."
        )

    @pytest.mark.asyncio
    async def test_agent_handoff_hook_emits_handoff_success(self) -> None:
        """on_handoff hook must emit $ld:ai:graph:handoff_success."""
        track_calls: list[str] = []
        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: track_calls.append(evt)
        )

        captured_hooks: list[Any] = []

        # Two-node graph: root -> child
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

        run_result = _make_run_result("done")
        agents_mock = _make_agents_mock(run_result)

        # Fire on_handoff inside Runner.run so the get_client patch is still active
        async def _run_and_fire_hook(agent: Any, text: str, hooks: Any = None) -> Any:
            if hooks:
                captured_hooks.append(hooks)
                from_agent_mock = MagicMock()
                from_agent_mock.name = "root"
                to_agent_mock = MagicMock()
                to_agent_mock.name = "child"
                await hooks.on_handoff(None, from_agent_mock, to_agent_mock)
            return run_result

        agents_mock.Runner.run = _run_and_fire_hook
        ctx = {"kind": "user", "key": "test"}

        with patch(
            "importlib.import_module",
            side_effect=lambda n: agents_mock if n == "agents" else __import__(n),
        ):
            with patch.object(_openai_ng, "get_client", return_value=mock_ld_client):
                await to_openai_agents(
                    _make_def_promise(graph_def),
                    opts={"context": ctx},
                ).invoke("hi")

        assert captured_hooks, "hooks were not passed to Runner.run"
        assert "$ld:ai:graph:handoff_success" in track_calls
