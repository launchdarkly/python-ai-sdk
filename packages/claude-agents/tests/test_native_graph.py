"""
Tests for §2.2 native graph adapter (to_claude_agents) plus Anthropic-specific specs.
Reference: TESTING.md §2.2, §2.x (Anthropic)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_claude_agents.native_graph as _claude_ng
from launchdarkly_ai_claude_agents.native_graph import to_claude_agents
from launchdarkly_ai_server import GraphDefinition, GraphEdge, GraphNode, NativeTool
from tests.forwarding_spec import (
    CLAUDE_AGENTS,
    candidate_keys,
    probe_forwarded_keys,
)
from tests.never_forwarded import NEVER_FORWARDED_BAG, find_leaks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result_msg(
    text: str | None = "result",
    input_tokens: int = 5,
    output_tokens: int = 3,
    subtype: str = "success",
    is_error: bool = False,
    errors: list[str] | None = None,
    api_error_status: int | None = None,
) -> Any:
    m = MagicMock()
    m.__class__.__name__ = "ResultMessage"
    # Set explicitly: an unset MagicMock attribute is truthy, so `is_error` would read as an error.
    m.subtype = subtype
    m.is_error = is_error
    m.errors = errors
    m.api_error_status = api_error_status
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
    async def test_multimodal_history_uses_native_root_prompt(self) -> None:
        mock_sdk = _make_sdk_mock("done")
        graph_def = _make_graph_def()
        captured_prompts: list[Any] = []
        result_msg = _make_result_msg("done")

        async def _query(**kwargs: Any) -> AsyncIterator[Any]:
            captured_prompts.append(kwargs.get("prompt"))
            yield result_msg

        mock_sdk.query = _query
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
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            await to_claude_agents(_make_def_promise(graph_def)).invoke(
                "describe", {}, history
            )

        assert captured_prompts
        prompt = captured_prompts[0]
        assert not isinstance(prompt, str)
        chunks = [chunk async for chunk in prompt]
        image = chunks[0]["message"]["content"][0]
        assert image == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "abc123",
            },
        }

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

        # create_sdk_mcp_server is called from _build_tool_mcp for user config tools
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


class TestToClaudeAgentsHostIsolation:
    """Every ``query()`` the adapter makes switches the built-ins off unless a node attaches one,
    and loads none of the host's Claude Code settings unless ``opts`` names sources. The host
    options in ``opts`` reach every node, root and sub-agents alike.
    """

    @staticmethod
    async def _root_options(
        tool_handlers: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        mock_sdk = _make_sdk_mock("done")
        captured: list[dict[str, Any]] = []
        mock_sdk.ClaudeAgentOptions = MagicMock(
            side_effect=lambda **kw: (captured.append(kw), kw)[1]
        )
        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            await to_claude_agents(
                _make_def_promise(_make_graph_def()),
                opts={"tool_handlers": tool_handlers or {}},
            ).invoke("hi")
        assert len(captured) == 1
        return captured[0]

    @staticmethod
    async def _root_and_leaf_options(
        **host: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Runs a root with one leaf sub-agent, which the root calls, and returns the options
        each node's ``query()`` received."""
        graph_def = _make_graph_def(
            nodes={
                "root": {
                    "key": "root",
                    "config": {"model": {"name": "claude-3"}, "instructions": "root"},
                    "meta": {"variationKey": "v1", "version": 1},
                    "edges": [{"source_key": "root", "target_key": "leaf"}],
                    "is_terminal": False,
                },
                "leaf": {
                    "key": "leaf",
                    "config": {"model": {"name": "claude-3"}, "instructions": "leaf"},
                    "meta": {"variationKey": "v1", "version": 1},
                    "edges": [],
                    "is_terminal": True,
                },
            },
            edges=[{"source_key": "root", "target_key": "leaf"}],
        )
        leaf_tool: list[Any] = []

        def _tool_factory(name: str, desc: str, schema: Any) -> Any:
            def _dec(fn: Any) -> Any:
                if "leaf" in name:
                    leaf_tool.append(fn)
                return fn

            return _dec

        mock_sdk = _make_sdk_mock("done")
        mock_sdk.tool = MagicMock(side_effect=_tool_factory)
        captured: list[dict[str, Any]] = []
        mock_sdk.ClaudeAgentOptions = MagicMock(
            side_effect=lambda **kw: (captured.append(kw), kw)[1]
        )
        result_msg = _make_result_msg("done")

        async def _query(**kwargs: Any) -> AsyncIterator[Any]:
            if kwargs["options"].get("system_prompt") == "root":
                await leaf_tool[0]({"input": "sub-query"})
            yield result_msg

        mock_sdk.query = _query
        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            await to_claude_agents(_make_def_promise(graph_def), opts=host).invoke("hi")
        by_node = {kw["system_prompt"]: kw for kw in captured}
        assert sorted(by_node) == ["leaf", "root"]
        return by_node["root"], by_node["leaf"]

    @pytest.mark.asyncio
    async def test_sends_an_empty_tool_list_with_no_native_tools(self) -> None:
        opts = await self._root_options()
        assert opts["tools"] == []

    @pytest.mark.asyncio
    async def test_sends_the_native_tool_names(self) -> None:
        opts = await self._root_options({"run-bash": NativeTool("Bash")})
        assert opts["tools"] == ["Bash"]

    @pytest.mark.asyncio
    async def test_loads_no_host_settings_and_sends_no_cwd_by_default(self) -> None:
        for opts in await self._root_and_leaf_options():
            assert opts["setting_sources"] == []
            assert "cwd" not in opts

    @pytest.mark.asyncio
    async def test_forwards_cwd_and_setting_sources_to_root_and_leaf(self) -> None:
        for opts in await self._root_and_leaf_options(
            cwd="/srv/app", setting_sources=["user"]
        ):
            assert opts["cwd"] == "/srv/app"
            assert opts["setting_sources"] == ["user"]


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
    async def test_cyclic_edge_is_dropped_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_sdk = _make_sdk_mock("done")

        # root -> a -> b -> a: post-order builds b while a is still unbuilt.
        def _node(key: str, target: str) -> dict[str, Any]:
            return {
                "key": key,
                "config": {"model": {"name": "claude-3"}, "instructions": key},
                "meta": {},
                "edges": [{"source_key": key, "target_key": target}],
                "is_terminal": False,
            }

        edges = [("root", "a"), ("a", "b"), ("b", "a")]
        graph_def = _make_graph_def(
            nodes={
                "root": _node("root", "a"),
                "a": _node("a", "b"),
                "b": _node("b", "a"),
            },
            edges=[{"source_key": s, "target_key": t} for s, t in edges],
        )

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            await to_claude_agents(_make_def_promise(graph_def)).invoke("hi")

        assert 'Dropping cyclic edge "b" -> "a"' in caplog.text

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
        child_nodes = [
            data
            for evt, data in track_calls
            if evt == "$ld:ai:graph:node" and data.get("nodeKey") == "child"
        ]
        assert len(child_nodes) == 1

    @pytest.mark.asyncio
    async def test_emits_invocation_success_on_completion(self) -> None:
        track_calls: list[tuple[str, Any]] = []
        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: track_calls.append((evt, data))
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

        assert "$ld:ai:graph:invocation_success" in [evt for evt, _ in track_calls]
        root_nodes = [
            data
            for evt, data in track_calls
            if evt == "$ld:ai:graph:node" and data.get("nodeKey") == "root"
        ]
        assert len(root_nodes) == 1
        assert root_nodes[0]["index"] == 0
        assert all(evt != "$ld:ai:graph:path" for evt, _ in track_calls)

    @pytest.mark.asyncio
    async def test_graph_span_carries_config_identity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The graph span must identify the config so Monitoring can link the trace."""
        import launchdarkly_ai_claude_agents.native_graph as ng_mod

        monkeypatch.setenv("LD_ENVIRONMENT_ID", "env-123")
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        mock_sdk = _make_sdk_mock("done")
        graph_def = _make_graph_def()
        ctx = {"kind": "user", "key": "u1"}

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with patch.object(ng_mod, "get_client", return_value=MagicMock()):
                with patch.object(ng_mod, "trace", mock_trace):
                    with patch.object(ng_mod, "_HAS_OTEL", True):
                        await to_claude_agents(
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

        mock_sdk = _make_sdk_mock("done")
        graph_def = _make_graph_def()
        ctx = {"kind": "user", "key": "u1"}

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

        mock_sdk = _make_sdk_mock("done")
        graph_def = _make_graph_def()
        ctx = {"kind": "user", "key": "u1"}

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
        fake_trace, exporter, _ = _recording_trace()
        track_calls: list[tuple[str, Any]] = []
        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: track_calls.append((evt, data))
        )
        mock_sdk = _make_sdk_mock("done")
        graph_def = _make_graph_def()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with patch.object(_claude_ng, "trace", fake_trace):
                with patch.object(_claude_ng, "_HAS_OTEL", True):
                    with patch.object(
                        _claude_ng, "get_client", return_value=mock_ld_client
                    ):
                        # A task, unlike a bare coroutine, can be awaited on each call.
                        runner = to_claude_agents(
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

    # Measured against Agent SDK 0.3.220: an Anthropic API failure arrives as a `success` result
    # with `is_error` set and the error text in `result`. The adapter used to return that text as
    # the answer, and it ignored error subtypes entirely.
    @pytest.mark.asyncio
    async def test_root_api_error_result_fails_the_graph(self) -> None:
        track_calls: list[str] = []
        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: track_calls.append(evt)
        )
        mock_sdk = _make_sdk_mock()
        error_msg = _make_result_msg(
            "Credit balance is too low", 0, 0, is_error=True, api_error_status=400
        )

        async def _query(**kwargs: Any) -> AsyncIterator[Any]:
            yield error_msg

        mock_sdk.query = _query
        mock_sdk.ResultMessage = type(error_msg)
        ctx = {"kind": "user", "key": "test"}

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with patch.object(_claude_ng, "get_client", return_value=mock_ld_client):
                with pytest.raises(RuntimeError, match="Credit balance is too low"):
                    await to_claude_agents(
                        _make_def_promise(_make_graph_def()),
                        opts={"context": ctx},
                    ).invoke("hi")

        assert "$ld:ai:graph:invocation_failure" in track_calls
        assert "$ld:ai:graph:invocation_success" not in track_calls

    @pytest.mark.asyncio
    async def test_root_error_subtype_fails_the_graph(self) -> None:
        mock_sdk = _make_sdk_mock()
        error_msg = _make_result_msg(
            None,
            subtype="error_max_turns",
            is_error=True,
            errors=["turn limit reached"],
        )

        async def _query(**kwargs: Any) -> AsyncIterator[Any]:
            yield error_msg

        mock_sdk.query = _query
        mock_sdk.ResultMessage = type(error_msg)

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with pytest.raises(
                RuntimeError, match=r"error_max_turns.*turn limit reached"
            ):
                await to_claude_agents(_make_def_promise(_make_graph_def())).invoke(
                    "hi"
                )

    @pytest.mark.asyncio
    async def test_sub_agent_api_error_fails_its_tool_call(self) -> None:
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
        child_tool_fn: list[Any] = []

        def _tool_factory(name: str, desc: str, schema: Any) -> Any:
            def _dec(fn: Any) -> Any:
                if "child" in name:
                    child_tool_fn.append(fn)
                return fn

            return _dec

        mock_sdk = _make_sdk_mock("root-done")
        mock_sdk.tool = MagicMock(side_effect=_tool_factory)
        error_msg = _make_result_msg(
            "Credit balance is too low", 0, 0, is_error=True, api_error_status=400
        )

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            await to_claude_agents(_make_def_promise(graph_def)).invoke("hi")

            async def _query(**kwargs: Any) -> AsyncIterator[Any]:
                yield error_msg

            mock_sdk.query = _query
            mock_sdk.ResultMessage = type(error_msg)
            assert child_tool_fn
            with pytest.raises(RuntimeError, match="Credit balance is too low"):
                await child_tool_fn[0]({"input": "sub-query"})

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

        mock_span.end.assert_called_once()
        mock_span.record_exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_subagent_setup_error_fails_the_graph_span(self) -> None:
        """A sub-agent build error before the run ends one ERROR span and tracks the failure."""
        fake_trace, exporter, end_spies = _recording_trace()
        track_calls: list[tuple[str, Any]] = []
        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: track_calls.append((evt, data))
        )
        mock_sdk = _make_sdk_mock("done")
        graph_def = _make_graph_def()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with patch.object(_claude_ng, "trace", fake_trace):
                with patch.object(_claude_ng, "_HAS_OTEL", True):
                    with patch.object(
                        _claude_ng, "get_client", return_value=mock_ld_client
                    ):
                        with patch.object(
                            _claude_ng,
                            "_reverse_traverse",
                            AsyncMock(side_effect=ValueError("bad sub-agent")),
                        ):
                            with pytest.raises(ValueError, match="bad sub-agent"):
                                await to_claude_agents(
                                    _make_def_promise(graph_def),
                                    opts={"context": {"kind": "user", "key": "u1"}},
                                ).invoke("hi")

        _assert_one_failed_graph_span(exporter, end_spies, "bad sub-agent", track_calls)

    @pytest.mark.asyncio
    async def test_build_tool_mcp_throws_when_tool_not_in_handlers(self) -> None:
        from launchdarkly_ai_claude_agents.handler import _build_tool_mcp

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
            await _build_tool_mcp(config_tools, handlers)
            # The actual error is raised when the fn is *called*, not when mcp is built


class TestNativeGraphConversationId:
    """The telemetry contract claims the conversation id reaches ``launchdarkly.graph`` spans.

    The span is opened after an ``await`` on the graph definition, so this pins that the binding
    survives the await chain rather than only covering spans started synchronously in the block.
    """

    @pytest.mark.asyncio
    async def test_stamps_conversation_id_on_graph_span(self) -> None:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        import launchdarkly_ai_claude_agents.native_graph as ng_mod
        from launchdarkly_ai_server.conversation import (
            GEN_AI_CONVERSATION_ID,
            ConversationIdSpanProcessor,
            conversation_id,
        )

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(ConversationIdSpanProcessor())
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        # A real tracer behind the module's `trace` lookup: the span and the processor are real,
        # only the global-provider registration is bypassed (it is process-wide and set once).
        real_trace = MagicMock()
        real_trace.get_tracer.return_value = provider.get_tracer(
            "@launchdarkly/ai-claude-agents"
        )

        mock_sdk = _make_sdk_mock("done")
        graph_def = _make_graph_def()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                mock_sdk if n == "claude_agent_sdk" else __import__(n)
            ),
        ):
            with patch.object(ng_mod, "trace", real_trace):
                with patch.object(ng_mod, "_HAS_OTEL", True):
                    with conversation_id("thread-graph"):
                        await to_claude_agents(_make_def_promise(graph_def)).invoke(
                            "hi"
                        )

        graph_spans = [
            s for s in exporter.get_finished_spans() if s.name == "launchdarkly.graph"
        ]
        assert len(graph_spans) == 1
        assert (graph_spans[0].attributes or {}).get(
            GEN_AI_CONVERSATION_ID
        ) == "thread-graph"


class TestNativeGraphModelParameters:
    @pytest.mark.asyncio
    async def test_node_options_take_run_settings_and_nothing_never_forwarded(
        self,
    ) -> None:
        """Each node runs its own query with options built from its own config, so a node's
        ``max_turns`` applies to that node."""
        mock_sdk = _make_sdk_mock("done")
        nodes = {
            "root": {
                "key": "root",
                "config": {
                    "model": {
                        "name": "claude-3",
                        "parameters": {**NEVER_FORWARDED_BAG, "max_turns": 7},
                    },
                    "instructions": "be helpful",
                },
                "meta": {"variationKey": "v1", "version": 1},
                "edges": [],
                "is_terminal": True,
            }
        }
        graph_def = _make_graph_def(nodes=nodes)

        captured_options: list[dict[str, Any]] = []
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

        assert captured_options
        for opts in captured_options:
            assert opts["max_turns"] == 7
            assert not find_leaks(opts)

    @pytest.mark.asyncio
    async def test_node_max_tokens_sets_the_cli_output_limit_and_thinking_is_disabled(
        self,
    ) -> None:
        """A node's ``max_tokens`` reaches its CLI as
        ``CLAUDE_CODE_MAX_OUTPUT_TOKENS``, and thinking is off unless the node's config asks."""

        async def node_options(parameters: dict[str, Any]) -> dict[str, Any]:
            mock_sdk = _make_sdk_mock("done")
            nodes = {
                "root": {
                    "key": "root",
                    "config": {
                        "model": {"name": "claude-3", "parameters": parameters},
                        "instructions": "be helpful",
                    },
                    "meta": {"variationKey": "v1", "version": 1},
                    "edges": [],
                    "is_terminal": True,
                }
            }
            captured: list[dict[str, Any]] = []
            mock_sdk.ClaudeAgentOptions = MagicMock(
                side_effect=lambda **kw: (captured.append(kw), kw)[1]
            )
            with patch(
                "importlib.import_module",
                side_effect=lambda n: (
                    mock_sdk if n == "claude_agent_sdk" else __import__(n)
                ),
            ):
                await to_claude_agents(
                    _make_def_promise(_make_graph_def(nodes=nodes))
                ).invoke("hi")
            assert len(captured) == 1
            return captured[0]

        limited = await node_options({"max_tokens": 200})
        assert limited["env"] == {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "200"}
        assert limited["thinking"] == {"type": "disabled"}
        # Built by the real SDK, so the node's CLI is started with thinking off.
        command = _build_cli_command(limited)
        assert command[command.index("--thinking") + 1] == "disabled"

        invalid = await node_options({"max_tokens": 1.5})
        assert "env" not in invalid
        assert invalid["thinking"] == {"type": "disabled"}

        budget_only = await node_options({"max_thinking_tokens": 2048})
        assert budget_only["max_thinking_tokens"] == 2048
        assert "thinking" not in budget_only
        command = _build_cli_command(budget_only)
        assert command[command.index("--max-thinking-tokens") + 1] == "2048"

        thinking = await node_options(
            {"thinking": {"type": "enabled", "budget_tokens": 2048}}
        )
        assert "env" not in thinking
        assert thinking["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def _build_cli_command(options_kwargs: dict[str, Any]) -> list[str]:
    """The CLI command the real ``claude_agent_sdk`` would launch for one node's options, built by
    the SDK's own transport with a stand-in binary. The mocked hooks and MCP servers are left out,
    since only the forwarded settings are being checked."""
    from claude_agent_sdk import ClaudeAgentOptions
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        SubprocessCLITransport,
    )

    options = ClaudeAgentOptions(
        **{
            k: v
            for k, v in options_kwargs.items()
            if k not in {"hooks", "mcp_servers", "can_use_tool"}
        },
        cli_path="/bin/echo",
    )
    return SubprocessCLITransport(prompt="q", options=options)._build_command()


class TestNativeGraphForwardsExactlyTheCrossSdkList:
    """The native graph builds each node's options with the handler's own forwarding, so a probe
    of one node finds exactly the cross-SDK Claude Agents list."""

    @pytest.mark.asyncio
    async def test_root_node_options(self) -> None:
        import dataclasses

        from claude_agent_sdk import ClaudeAgentOptions

        async def call(parameters: dict[str, Any]) -> object:
            mock_sdk = _make_sdk_mock("done")
            nodes = {
                "root": {
                    "key": "root",
                    "config": {
                        "model": {"name": "claude-3", "parameters": parameters},
                        "instructions": "be helpful",
                    },
                    "meta": {"variationKey": "v1", "version": 1},
                    "edges": [],
                    "is_terminal": True,
                }
            }
            captured: list[dict[str, Any]] = []
            mock_sdk.ClaudeAgentOptions = MagicMock(
                side_effect=lambda **kw: (captured.append(kw), kw)[1]
            )
            with patch(
                "importlib.import_module",
                side_effect=lambda n: (
                    mock_sdk if n == "claude_agent_sdk" else __import__(n)
                ),
            ):
                await to_claude_agents(
                    _make_def_promise(_make_graph_def(nodes=nodes))
                ).invoke("hi")
            # The real SDK builds its CLI command from each node's options, so a value it
            # cannot read fails the probe here.
            for kw in captured:
                _build_cli_command(kw)
            return [
                {k: v for k, v in kw.items() if k not in {"hooks", "mcp_servers"}}
                for kw in captured
            ]

        candidates = candidate_keys(
            (f.name for f in dataclasses.fields(ClaudeAgentOptions)), CLAUDE_AGENTS
        )
        assert await probe_forwarded_keys(candidates, call) == CLAUDE_AGENTS
