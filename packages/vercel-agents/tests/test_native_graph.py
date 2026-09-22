"""Test-first contract for ``to_vercel_agents``.

The ``ai`` runtime is fully mocked. The suite describes topology, handoff,
history, usage, telemetry, and cleanup without gateway credentials or I/O.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import launchdarkly_ai_vercel_agents.native_graph as native_graph_mod
from launchdarkly_ai_server import GraphDefinition, GraphEdge, GraphNode
from launchdarkly_ai_vercel_agents.native_graph import to_vercel_agents


def _edge(source: str, target: str) -> GraphEdge:
    return GraphEdge(
        key=f"{source}-{target}",
        source_key=source,
        target_key=target,
        handoff={"description": f"Transfer to {target}"},
    )


def _node(
    key: str,
    *,
    edges: list[GraphEdge] | None = None,
    tools: dict[str, Any] | None = None,
) -> GraphNode:
    node_edges = edges or []
    return GraphNode(
        key=key,
        config={
            "model": {"name": f"gateway/{key}", "parameters": {"temperature": 0.1}},
            "provider": {"name": "Gateway"},
            "instructions": f"instructions for {key}",
            "tools": tools or {},
        },
        meta={"variationKey": f"variation-{key}", "version": 1},
        edges=node_edges,
        is_terminal=not node_edges,
    )


def _graph(
    *,
    enabled: bool = True,
    include_node_tool: bool = False,
) -> GraphDefinition:
    root_edge = _edge("root", "leaf")
    root = _node(
        "root",
        edges=[root_edge],
        tools=(
            {"lookup": {"description": "Look up", "parameters": {"type": "object"}}}
            if include_node_tool
            else None
        ),
    )
    leaf = _node("leaf")
    nodes = {"root": root, "leaf": leaf}

    async def _traverse(fn: Any, ctx: Any = None) -> None:
        for node in (root, leaf):
            value = fn(node)
            if hasattr(value, "__await__"):
                await value

    async def _unused(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("native graph must not call GraphDefinition.run_node")

    return GraphDefinition(
        key="vercel-graph",
        enabled=enabled,
        root=root,
        get_node=lambda key: nodes.get(key),
        get_child_nodes=lambda key: [leaf] if key == "root" else [],
        get_parent_nodes=lambda key: [root] if key == "leaf" else [],
        terminal_nodes=lambda: [leaf],
        is_terminal=lambda key: key == "leaf",
        edges_from=lambda key: [root_edge] if key == "root" else [],
        run_node=_unused,
        route=_unused,
        traverse=_traverse,
        reverse_traverse=_traverse,
    )


async def _definition(value: GraphDefinition) -> GraphDefinition:
    return value


class RunStream:
    def __init__(
        self,
        text: str,
        *,
        input_tokens: int,
        output_tokens: int,
        on_enter: Any = None,
    ) -> None:
        # ``AgentStream`` exposes ``output``, not ``text``.
        self.output = text
        self.messages: list[Any] = [
            SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=input_tokens, output_tokens=output_tokens
                )
            )
        ]
        self.exited = 0
        self.on_enter = on_enter

    async def __aenter__(self) -> RunStream:
        if self.on_enter is not None:
            value = self.on_enter()
            if hasattr(value, "__await__"):
                await value
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.exited += 1

    def __aiter__(self) -> RunStream:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


@pytest.fixture
def ai_runtime() -> MagicMock:
    runtime = MagicMock()
    runtime.get_model = MagicMock(side_effect=lambda name: f"model:{name}")
    runtime.system_message = MagicMock(
        side_effect=lambda content: {"role": "system", "content": content}
    )
    runtime.user_message = MagicMock(
        side_effect=lambda *content: {
            "role": "user",
            "content": content[0] if len(content) == 1 else list(content),
        }
    )
    runtime.assistant_message = MagicMock(
        side_effect=lambda *content: {
            "role": "assistant",
            "content": content[0] if len(content) == 1 else list(content),
        }
    )
    runtime.file_part = MagicMock(
        side_effect=lambda value, media_type=None: {
            "type": "file",
            "value": value,
            "media_type": media_type,
        }
    )
    runtime.tool = MagicMock(
        side_effect=lambda fn=None, **kwargs: (
            SimpleNamespace(execute=fn, **kwargs) if fn is not None else kwargs
        )
    )
    runtime.Tool = MagicMock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    runtime.AgentTool = MagicMock(
        side_effect=lambda tool, fn: SimpleNamespace(tool=tool, fn=fn, name=tool.name)
    )
    runtime.InferenceRequestParams = MagicMock(return_value=object())
    runtime._follow_handoff = False
    created: list[Any] = []

    def _agent(**kwargs: Any) -> MagicMock:
        agent = MagicMock()
        agent.kwargs = kwargs
        index = len(created)

        async def _select_handoff() -> None:
            if index != 0 or not runtime._follow_handoff:
                return
            handoff = next(
                tool
                for tool in kwargs.get("tools", [])
                if tool.name == "transfer_to_leaf"
            )
            await handoff.execute()

        agent.run = MagicMock(
            return_value=RunStream(
                "root answer" if index == 0 else "leaf answer",
                input_tokens=10 if index == 0 else 4,
                output_tokens=3 if index == 0 else 2,
                on_enter=_select_handoff,
            )
        )
        created.append(agent)
        return agent

    runtime.Agent = MagicMock(side_effect=_agent)
    runtime._created = created
    with patch.object(native_graph_mod, "ai", runtime):
        yield runtime


class TestTopology:
    @pytest.mark.asyncio
    async def test_constructs_one_native_agent_per_node(
        self, ai_runtime: MagicMock
    ) -> None:
        runner = to_vercel_agents(_definition(_graph()))
        await runner.invoke("hello")
        assert ai_runtime.Agent.call_count == 2
        assert ai_runtime.get_model.call_args_list[0].args == ("gateway/root",)
        assert ai_runtime.get_model.call_args_list[1].args == ("gateway/leaf",)

    @pytest.mark.asyncio
    async def test_each_agent_gets_its_own_instructions_and_settings(
        self, ai_runtime: MagicMock
    ) -> None:
        ai_runtime._follow_handoff = True
        await to_vercel_agents(_definition(_graph())).invoke("hello")
        root_messages = ai_runtime._created[0].run.call_args.kwargs["messages"]
        leaf_messages = ai_runtime._created[1].run.call_args.kwargs["messages"]
        assert "instructions for root" in str(root_messages)
        assert "instructions for leaf" in str(leaf_messages)
        assert all(
            call.kwargs["params"] is ai_runtime.InferenceRequestParams.return_value
            for agent in ai_runtime._created
            for call in agent.run.call_args_list
        )
        assert ai_runtime.TemperatureSamplerParams.call_count == 2
        assert all(
            call.kwargs == {"temperature": 0.1}
            for call in ai_runtime.TemperatureSamplerParams.call_args_list
        )

    @pytest.mark.asyncio
    async def test_non_terminal_node_has_transfer_tool_per_edge(
        self, ai_runtime: MagicMock
    ) -> None:
        await to_vercel_agents(_definition(_graph())).invoke("hello")
        root_tools = ai_runtime.Agent.call_args_list[0].kwargs["tools"]
        assert [
            tool.name for tool in root_tools if tool.name.startswith("transfer_to_")
        ] == ["transfer_to_leaf"]

    @pytest.mark.asyncio
    async def test_terminal_node_has_no_handoff_tools(
        self, ai_runtime: MagicMock
    ) -> None:
        await to_vercel_agents(_definition(_graph())).invoke("hello")
        leaf_tools = ai_runtime.Agent.call_args_list[1].kwargs.get("tools", [])
        assert not [tool for tool in leaf_tools if tool.name.startswith("transfer_to_")]

    @pytest.mark.asyncio
    async def test_node_tools_are_included_with_global_handlers(
        self, ai_runtime: MagicMock
    ) -> None:
        lookup = MagicMock(return_value="found")
        await to_vercel_agents(
            _definition(_graph(include_node_tool=True)),
            {"tool_handlers": {"lookup": lookup}},
        ).invoke("hello")
        root_tools = ai_runtime.Agent.call_args_list[0].kwargs["tools"]
        node_tool = next(tool for tool in root_tools if tool.name == "lookup")
        assert await node_tool.execute(key="x") == "found"
        lookup.assert_called_once_with({"key": "x"})

    @pytest.mark.asyncio
    async def test_disabled_graph_raises(self, ai_runtime: MagicMock) -> None:
        with pytest.raises(ValueError, match="disabled"):
            await to_vercel_agents(_definition(_graph(enabled=False))).invoke("hello")


class TestInvocation:
    @pytest.mark.asyncio
    async def test_stops_at_root_when_no_handoff_tool_is_called(
        self, ai_runtime: MagicMock
    ) -> None:
        result = await to_vercel_agents(_definition(_graph())).invoke("hello")
        assert result["response"] == "root answer"
        ai_runtime._created[0].run.assert_called_once()
        ai_runtime._created[1].run.assert_not_called()

    @pytest.mark.asyncio
    async def test_starts_at_root_and_applies_history_only_there(
        self, ai_runtime: MagicMock
    ) -> None:
        ai_runtime._follow_handoff = True
        history = [
            {"role": "user", "content": "past"},
            {"role": "assistant", "content": "reply"},
        ]
        await to_vercel_agents(_definition(_graph())).invoke("latest", {}, history)
        root_messages = ai_runtime._created[0].run.call_args.kwargs["messages"]
        leaf_messages = ai_runtime._created[1].run.call_args.kwargs.get("messages", [])
        assert "past" in str(root_messages)
        assert "latest" in str(root_messages)
        assert "past" not in str(leaf_messages)

    @pytest.mark.asyncio
    async def test_multimodal_history_is_mapped_for_root(
        self, ai_runtime: MagicMock
    ) -> None:
        history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "YWJj",
                        },
                    }
                ],
            }
        ]
        await to_vercel_agents(_definition(_graph())).invoke("describe", {}, history)
        ai_runtime.file_part.assert_called_once()
        root_messages = ai_runtime._created[0].run.call_args.kwargs["messages"]
        assert "b'abc'" in str(root_messages)

    @pytest.mark.asyncio
    async def test_returns_final_leaf_text_and_accumulated_usage(
        self, ai_runtime: MagicMock
    ) -> None:
        ai_runtime._follow_handoff = True
        result = await to_vercel_agents(_definition(_graph())).invoke("hello")
        assert result["response"] == "leaf answer"
        assert result["usage"] == {"input": 14, "output": 5, "total": 19}

    @pytest.mark.asyncio
    async def test_handoff_path_has_no_duplicate_entries(
        self, ai_runtime: MagicMock
    ) -> None:
        ai_runtime._follow_handoff = True
        span = MagicMock()
        tracer = MagicMock()
        tracer.start_span.return_value = span
        with patch.object(native_graph_mod.trace, "get_tracer", return_value=tracer):
            await to_vercel_agents(_definition(_graph())).invoke("hello")
        path_values = [
            call.args[1]
            for call in span.set_attribute.call_args_list
            if call.args[0] == "ld.ai.graph.path"
        ]
        assert path_values
        path = path_values[-1].split("->")
        assert path == ["root", "leaf"]
        assert len(path) == len(set(path))


class TestTelemetryAndCleanup:
    @pytest.mark.asyncio
    async def test_context_emits_graph_success_handoff_duration_and_tokens(
        self, ai_runtime: MagicMock
    ) -> None:
        ai_runtime._follow_handoff = True
        client = MagicMock()
        context = {"kind": "user", "key": "u"}
        with patch.object(native_graph_mod, "get_client", return_value=client):
            await to_vercel_agents(_definition(_graph()), {"context": context}).invoke(
                "hello"
            )
        event_names = [call.args[0] for call in client.track.call_args_list]
        assert "$ld:ai:graph:invocation_success" in event_names
        assert "$ld:ai:graph:handoff_success" in event_names
        assert "$ld:ai:graph:duration:total" in event_names
        assert "$ld:ai:graph:total_tokens" in event_names

    @pytest.mark.asyncio
    async def test_no_context_emits_no_launchdarkly_tracking(
        self, ai_runtime: MagicMock
    ) -> None:
        client = MagicMock()
        with patch.object(native_graph_mod, "get_client", return_value=client):
            await to_vercel_agents(_definition(_graph())).invoke("hello")
        client.track.assert_not_called()

    @pytest.mark.asyncio
    async def test_graph_span_ends_on_success(self, ai_runtime: MagicMock) -> None:
        span = MagicMock()
        tracer = MagicMock()
        tracer.start_span.return_value = span
        with patch.object(native_graph_mod.trace, "get_tracer", return_value=tracer):
            await to_vercel_agents(_definition(_graph())).invoke("hello")
        tracer.start_span.assert_called_once_with("ld.ai.graph")
        span.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_graph_span_ends_and_failure_is_tracked_on_error(
        self, ai_runtime: MagicMock
    ) -> None:
        class BrokenContext:
            async def __aenter__(self) -> Any:
                raise RuntimeError("provider failed")

            async def __aexit__(self, *args: Any) -> None:
                return None

        client = MagicMock()
        span = MagicMock()
        tracer = MagicMock()
        tracer.start_span.return_value = span
        ai_runtime.Agent.side_effect = None
        broken_agent = MagicMock()
        broken_agent.run.return_value = BrokenContext()
        ai_runtime.Agent.return_value = broken_agent
        with (
            patch.object(native_graph_mod, "get_client", return_value=client),
            patch.object(native_graph_mod.trace, "get_tracer", return_value=tracer),
            pytest.raises(RuntimeError, match="provider failed"),
        ):
            await to_vercel_agents(
                _definition(_graph()), {"context": {"kind": "user", "key": "u"}}
            ).invoke("hello")
        assert "$ld:ai:graph:invocation_failure" in [
            call.args[0] for call in client.track.call_args_list
        ]
        span.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_graph_span_ends_on_cancellation(self, ai_runtime: MagicMock) -> None:
        class CancelledContext:
            async def __aenter__(self) -> Any:
                raise asyncio.CancelledError

            async def __aexit__(self, *args: Any) -> None:
                return None

        span = MagicMock()
        tracer = MagicMock()
        tracer.start_span.return_value = span
        ai_runtime.Agent.side_effect = None
        cancelled_agent = MagicMock()
        cancelled_agent.run.return_value = CancelledContext()
        ai_runtime.Agent.return_value = cancelled_agent
        with (
            patch.object(native_graph_mod.trace, "get_tracer", return_value=tracer),
            pytest.raises(asyncio.CancelledError),
        ):
            await to_vercel_agents(_definition(_graph())).invoke("hello")
        span.end.assert_called_once()
