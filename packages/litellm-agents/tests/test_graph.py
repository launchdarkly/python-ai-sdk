"""Graph contracts for the LiteLLM Agents SDK adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_litellm_agents.graph as graph_mod
import launchdarkly_ai_litellm_agents.native_graph as native_graph_mod
from launchdarkly_ai_litellm_agents import litellm_graph, to_litellm_agents
from launchdarkly_ai_server import GraphDefinition, GraphEdge, GraphNode


class _FakeRunHooks:
    async def on_agent_end(self, *args: Any) -> None:
        return None

    async def on_handoff(self, *args: Any) -> None:
        return None

    async def on_agent_start(self, *args: Any) -> None:
        return None


def _definition(*, enabled: bool = True) -> GraphDefinition:
    edge = GraphEdge(
        key="root-child",
        source_key="root",
        target_key="child",
        handoff={"description": "delegate"},
    )
    child = GraphNode(
        key="child",
        config={
            "provider": {"name": "Gemini"},
            "model": {"name": "gemini/gemini-2.5-pro"},
            "instructions": "child",
        },
        meta={},
        edges=[],
        is_terminal=True,
    )
    root = GraphNode(
        key="root",
        config={
            "provider": {"name": "Anthropic"},
            "model": {"name": "anthropic/claude-sonnet-4"},
            "instructions": "root",
        },
        meta={},
        edges=[edge],
        is_terminal=False,
    )
    nodes = {"root": root, "child": child}

    async def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def traverse(*args: Any, **kwargs: Any) -> None:
        return None

    return GraphDefinition(
        key="litellm-graph",
        enabled=enabled,
        root=root if enabled else None,
        get_node=lambda key: nodes.get(key),
        get_child_nodes=lambda key: [child] if key == "root" else [],
        get_parent_nodes=lambda key: [root] if key == "child" else [],
        terminal_nodes=lambda: [child],
        is_terminal=lambda key: key == "child",
        edges_from=lambda key: [edge] if key == "root" else [],
        run_node=unavailable,
        route=unavailable,
        traverse=traverse,
        reverse_traverse=traverse,
    )


async def _definition_awaitable(*, enabled: bool = True) -> GraphDefinition:
    return _definition(enabled=enabled)


async def _definition_awaitable_from(
    definition: GraphDefinition,
) -> GraphDefinition:
    return definition


class TestGraphConvenience:
    def test_binds_wildcard_litellm_agent_handler(self) -> None:
        graph = MagicMock(return_value=object())
        with patch.object(graph_mod, "graph", graph):
            result = litellm_graph(
                "graph-key",
                context={"kind": "user", "key": "u"},
                model_factory=MagicMock(),
            )
        assert result is graph.return_value
        assert graph.call_args.args == ("graph-key",)
        assert graph.call_args.kwargs["context"] == {"kind": "user", "key": "u"}
        handlers = graph.call_args.kwargs["handlers"]
        assert len(handlers) == 1
        assert handlers[0].provides_for == ("*", "agent")
        assert "model_factory" not in graph.call_args.kwargs


class TestNativeGraphAdapter:
    async def test_creates_every_node_with_its_authoritative_litellm_model(
        self,
    ) -> None:
        models: dict[str, object] = {}

        def model_factory(name: str, parameters: dict[str, Any]) -> object:
            models[name] = object()
            return models[name]

        created: list[dict[str, Any]] = []
        agent = MagicMock(
            side_effect=lambda **kwargs: (
                created.append(kwargs),
                SimpleNamespace(name=kwargs["name"], **kwargs),
            )[1]
        )
        runner = SimpleNamespace(
            run=AsyncMock(
                return_value=SimpleNamespace(
                    final_output="done",
                    context_wrapper=SimpleNamespace(
                        usage=SimpleNamespace(input_tokens=9, output_tokens=3)
                    ),
                )
            )
        )
        agents = SimpleNamespace(
            Agent=agent,
            Runner=runner,
            handoff=MagicMock(side_effect=lambda target: target),
            FunctionTool=MagicMock(),
            RunHooks=_FakeRunHooks,
        )
        real_import = __import__
        with patch(
            "importlib.import_module",
            side_effect=lambda name: agents if name == "agents" else real_import(name),
        ):
            result = await to_litellm_agents(
                _definition_awaitable(), opts={"model_factory": model_factory}
            ).invoke("hello")
        assert set(models) == {
            "anthropic/claude-sonnet-4",
            "gemini/gemini-2.5-pro",
        }
        assert {entry["model"] for entry in created} == set(models.values())
        assert agents.handoff.call_count == 1
        assert runner.run.call_args.args[0].name == "root"
        assert result == {
            "response": "done",
            "usage": {"input": 9, "output": 3, "total": 12},
        }

    async def test_forwards_multimodal_history_to_the_root_run(self) -> None:
        runner = SimpleNamespace(
            run=AsyncMock(
                return_value=SimpleNamespace(
                    final_output="done",
                    context_wrapper=SimpleNamespace(
                        usage=SimpleNamespace(input_tokens=1, output_tokens=1)
                    ),
                )
            )
        )
        agents = SimpleNamespace(
            Agent=MagicMock(
                side_effect=lambda **kwargs: SimpleNamespace(
                    name=kwargs["name"], **kwargs
                )
            ),
            Runner=runner,
            handoff=MagicMock(side_effect=lambda target: target),
            FunctionTool=MagicMock(),
            RunHooks=_FakeRunHooks,
        )
        history = [
            {"role": "system", "content": "ignore this"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "image-data",
                        },
                    }
                ],
            },
        ]
        real_import = __import__
        with patch(
            "importlib.import_module",
            side_effect=lambda name: agents if name == "agents" else real_import(name),
        ):
            await to_litellm_agents(
                _definition_awaitable(),
                opts={"model_factory": lambda *_: object()},
            ).invoke("", {}, history)

        assert runner.run.call_args.args[1] == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,image-data",
                    }
                ],
            }
        ]

    async def test_binds_registered_tools_to_their_graph_nodes(self) -> None:
        definition = _definition()
        assert definition.root is not None
        definition.root.config["tools"] = {
            "search": {
                "description": "Search",
                "parameters": {"type": "object", "properties": {}},
            }
        }
        created: list[dict[str, Any]] = []
        function_tool = MagicMock(
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs)
        )
        agents = SimpleNamespace(
            Agent=MagicMock(
                side_effect=lambda **kwargs: (
                    created.append(kwargs),
                    SimpleNamespace(name=kwargs["name"], **kwargs),
                )[1]
            ),
            Runner=SimpleNamespace(
                run=AsyncMock(
                    return_value=SimpleNamespace(
                        final_output="done",
                        context_wrapper=SimpleNamespace(
                            usage=SimpleNamespace(input_tokens=1, output_tokens=1)
                        ),
                    )
                )
            ),
            handoff=MagicMock(side_effect=lambda target: target),
            FunctionTool=function_tool,
            RunHooks=_FakeRunHooks,
        )
        search = AsyncMock(return_value="found")
        real_import = __import__
        with patch(
            "importlib.import_module",
            side_effect=lambda name: agents if name == "agents" else real_import(name),
        ):
            await to_litellm_agents(
                _definition_awaitable_from(definition),
                opts={
                    "model_factory": lambda *_: object(),
                    "tool_handlers": {"search": search},
                },
            ).invoke("query")

        root = next(entry for entry in created if entry["name"] == "root")
        assert len(root["tools"]) == 1
        assert root["tools"][0].name == "search"
        assert (
            await root["tools"][0].on_invoke_tool(None, '{"query":"docs"}') == "found"
        )
        search.assert_awaited_once_with({"query": "docs"})

    async def test_default_factory_uses_litellm_model_for_each_node(self) -> None:
        created_models: list[Any] = []
        litellm_model = MagicMock(
            side_effect=lambda **kwargs: (
                created_models.append(kwargs),
                object(),
            )[1]
        )
        agents = SimpleNamespace(
            Agent=MagicMock(
                side_effect=lambda **kwargs: SimpleNamespace(
                    name=kwargs["name"], **kwargs
                )
            ),
            Runner=SimpleNamespace(
                run=AsyncMock(
                    return_value=SimpleNamespace(
                        final_output="done",
                        context_wrapper=SimpleNamespace(
                            usage=SimpleNamespace(input_tokens=1, output_tokens=1)
                        ),
                    )
                )
            ),
            handoff=MagicMock(side_effect=lambda target: target),
            FunctionTool=MagicMock(),
            RunHooks=_FakeRunHooks,
        )
        real_import = __import__
        with (
            patch(
                "importlib.import_module",
                side_effect=lambda name: (
                    agents if name == "agents" else real_import(name)
                ),
            ),
            patch.object(native_graph_mod, "LitellmModel", litellm_model),
        ):
            await to_litellm_agents(_definition_awaitable()).invoke("hello")
        assert {call["model"] for call in created_models} == {
            "anthropic/claude-sonnet-4",
            "gemini/gemini-2.5-pro",
        }

    async def test_adapter_has_no_default_openai_fallback(self) -> None:
        fallback = MagicMock(side_effect=AssertionError("OpenAI fallback"))
        agents = SimpleNamespace(
            Agent=MagicMock(
                side_effect=lambda **kwargs: SimpleNamespace(
                    name=kwargs["name"], **kwargs
                )
            ),
            Runner=SimpleNamespace(
                run=AsyncMock(
                    return_value=SimpleNamespace(
                        final_output="done",
                        context_wrapper=SimpleNamespace(
                            usage=SimpleNamespace(input_tokens=0, output_tokens=0)
                        ),
                    )
                )
            ),
            handoff=MagicMock(side_effect=lambda target: target),
            FunctionTool=MagicMock(),
            RunHooks=_FakeRunHooks,
        )
        real_import = __import__
        with (
            patch(
                "importlib.import_module",
                side_effect=lambda name: (
                    agents if name == "agents" else real_import(name)
                ),
            ),
            patch.dict(
                "sys.modules",
                {
                    "openai": SimpleNamespace(AsyncOpenAI=fallback),
                    "agents.models.openai_provider": SimpleNamespace(
                        OpenAIProvider=fallback
                    ),
                },
            ),
        ):
            await to_litellm_agents(
                _definition_awaitable(),
                opts={"model_factory": lambda *_: object()},
            ).invoke("hello")
        fallback.assert_not_called()

    async def test_success_path_emits_invocation_success_and_tokens(self) -> None:
        track_calls: list[str] = []
        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: track_calls.append(evt)
        )
        agents = SimpleNamespace(
            Agent=MagicMock(
                side_effect=lambda **kwargs: SimpleNamespace(
                    name=kwargs["name"], **kwargs
                )
            ),
            Runner=SimpleNamespace(
                run=AsyncMock(
                    return_value=SimpleNamespace(
                        final_output="done",
                        context_wrapper=SimpleNamespace(
                            usage=SimpleNamespace(
                                input_tokens=4, output_tokens=2, total_tokens=6
                            )
                        ),
                    )
                )
            ),
            handoff=MagicMock(side_effect=lambda target: target),
            FunctionTool=MagicMock(),
            RunHooks=_FakeRunHooks,
        )
        real_import = __import__
        with (
            patch(
                "importlib.import_module",
                side_effect=lambda name: (
                    agents if name == "agents" else real_import(name)
                ),
            ),
            patch.object(native_graph_mod, "get_client", return_value=mock_ld_client),
        ):
            await to_litellm_agents(
                _definition_awaitable(),
                opts={
                    "model_factory": lambda *_: object(),
                    "context": {"kind": "user", "key": "test"},
                },
            ).invoke("hello")
        assert "$ld:ai:graph:invocation_success" in track_calls
        assert "$ld:ai:graph:total_tokens" in track_calls

    async def test_error_path_emits_invocation_failure_and_rethrows(self) -> None:
        track_calls: list[str] = []
        mock_ld_client = MagicMock()
        mock_ld_client.track = MagicMock(
            side_effect=lambda evt, ctx, data, val: track_calls.append(evt)
        )
        agents = SimpleNamespace(
            Agent=MagicMock(
                side_effect=lambda **kwargs: SimpleNamespace(
                    name=kwargs["name"], **kwargs
                )
            ),
            Runner=SimpleNamespace(
                run=AsyncMock(side_effect=RuntimeError("provider error"))
            ),
            handoff=MagicMock(side_effect=lambda target: target),
            FunctionTool=MagicMock(),
            RunHooks=_FakeRunHooks,
        )
        real_import = __import__
        with (
            patch(
                "importlib.import_module",
                side_effect=lambda name: (
                    agents if name == "agents" else real_import(name)
                ),
            ),
            patch.object(native_graph_mod, "get_client", return_value=mock_ld_client),
        ):
            with pytest.raises(RuntimeError, match="provider error"):
                await to_litellm_agents(
                    _definition_awaitable(),
                    opts={
                        "model_factory": lambda *_: object(),
                        "context": {"kind": "user", "key": "test"},
                    },
                ).invoke("hello")
        assert "$ld:ai:graph:invocation_failure" in track_calls

    async def test_disabled_graph_fails_before_runner_or_model_creation(self) -> None:
        model_factory = MagicMock()
        with pytest.raises(ValueError, match="disabled"):
            await to_litellm_agents(
                _definition_awaitable(enabled=False),
                opts={"model_factory": model_factory},
            ).invoke("hello")
        model_factory.assert_not_called()
