"""Native ADK workflow adapter. Reference: TESTING.md §2.2 and §2.x.4."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

import launchdarkly_ai_google_adk_agents.handler as handler_mod
import launchdarkly_ai_google_adk_agents.native_graph as native_mod
from launchdarkly_ai_google_adk_agents.native_graph import to_adk_agents


async def _resolved(graph: Any) -> Any:
    return graph


def _adapter(graph: Any, **opts: Any) -> Any:
    """``to_adk_agents`` as callers use it: an awaitable definition and an options dict."""
    return to_adk_agents(_resolved(graph), opts)


class _FakeTool:
    created: ClassVar[list[_FakeTool]] = []

    def __init__(self, func: Any, **kwargs: Any) -> None:
        self.func = func
        self.name = kwargs.get("name", "")
        _FakeTool.created.append(self)


class _FakeRunner:
    created: ClassVar[list[_FakeRunner]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.plugins = kwargs.get("plugins", [])
        self.session_service = SimpleNamespace(
            create_session=AsyncMock(return_value=SimpleNamespace(id="sess")),
            append_event=AsyncMock(),
        )
        _FakeRunner.created.append(self)

    async def run_async(self, **kwargs: Any) -> Any:
        yield SimpleNamespace(
            content=SimpleNamespace(parts=[SimpleNamespace(text="leaf-answer")]),
            partial=False,
            usage_metadata=SimpleNamespace(
                prompt_token_count=2,
                candidates_token_count=3,
                total_token_count=5,
            ),
            is_final_response=lambda: True,
        )


def _node(
    key: str, *, terminal: bool, edges: list[Any] | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        key=key,
        config={
            "model": {"name": "gemini-2.5-flash"},
            "provider": {"name": "Google"},
            "instructions": f"instructions-{key}",
        },
        meta={"variationKey": "v", "version": 1},
        edges=edges or [],
        is_terminal=terminal,
    )


def _graph(enabled: bool = True, root: Any = None) -> SimpleNamespace:
    leaf = _node("leaf", terminal=True)
    root_node = root or _node(
        "root",
        terminal=False,
        edges=[SimpleNamespace(target_key="leaf", key="root-leaf", source_key="root")],
    )
    nodes = {"root": root_node, "leaf": leaf}

    def get_node(key: str) -> Any:
        return nodes[key]

    return SimpleNamespace(
        enabled=enabled,
        key="graph-flag",
        root=root_node if enabled else None,
        get_node=get_node,
    )


@pytest.fixture(autouse=True)
def _mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeTool.created = []
    _FakeRunner.created = []
    monkeypatch.setattr(
        native_mod,
        "LlmAgent",
        MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw)),
    )
    monkeypatch.setattr(native_mod, "FunctionTool", _FakeTool)
    monkeypatch.setattr(native_mod, "InMemoryRunner", _FakeRunner)
    monkeypatch.setattr(
        handler_mod, "Gemini", MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw))
    )


class TestToAdkAgents:
    async def test_disabled_graph_throws(self) -> None:
        with pytest.raises(Exception, match="graph-flag"):
            await _adapter(_graph(enabled=False)).invoke("hi")

    async def test_missing_root_throws(self) -> None:
        graph = _graph()
        graph.root = None
        with pytest.raises(Exception, match="root"):
            await _adapter(graph).invoke("hi")

    async def test_two_node_graph_builds_two_agents_and_a_transfer_tool(self) -> None:
        result = await _adapter(_graph()).invoke("hi")
        assert native_mod.LlmAgent.call_count == 2
        instructions = [
            call.kwargs["instruction"] for call in native_mod.LlmAgent.call_args_list
        ]
        assert instructions == ["instructions-root", "instructions-leaf"] or set(
            instructions
        ) == {
            "instructions-root",
            "instructions-leaf",
        }
        names = [tool.name for tool in _FakeTool.created]
        assert names.count("transfer_to_leaf") == 1
        assert not any(
            name.startswith("transfer_to_") and name != "transfer_to_leaf"
            for name in names
        )
        assert result["response"] == "leaf-answer"
        assert result["usage"]["total"] == 5

    async def test_transfer_tool_selects_the_target(self) -> None:
        await _adapter(_graph()).invoke("hi")
        tool = next(
            item for item in _FakeTool.created if item.name == "transfer_to_leaf"
        )
        selected = tool.func()
        assert "leaf" in str(selected)

    async def test_no_context_does_not_track(self) -> None:
        tracker = MagicMock()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(native_mod, "get_client", lambda: tracker)
            await _adapter(_graph()).invoke("hi")
        tracker.track.assert_not_called()

    async def test_history_is_seeded_on_the_root_session_only(self) -> None:
        await _adapter(_graph()).invoke(
            "hi", history=[{"role": "user", "content": "earlier"}]
        )
        assert _FakeRunner.created[0].session_service.append_event.await_count == 1

    async def test_node_local_tools_are_forwarded(self) -> None:
        graph = _graph()
        graph.root.config["tools"] = {
            "lookup": {
                "name": "lookup",
                "description": "find",
                "parameters": {"type": "object"},
            }
        }

        def lookup(args: dict[str, Any]) -> str:
            return str(args["q"])

        await _adapter(graph, tool_handlers={"lookup": lookup}).invoke("hi")
        names = [tool.name for tool in _FakeTool.created]
        assert "lookup" in names
        assert names.count("transfer_to_leaf") == 1

    async def test_transfer_during_the_run_visits_the_target(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _HandoffRunner(_FakeRunner):
            hops: ClassVar[int] = 0

            async def run_async(self, **kwargs: Any) -> Any:
                if _HandoffRunner.hops == 0:
                    tool = next(
                        item
                        for item in _FakeTool.created
                        if item.name == "transfer_to_leaf"
                    )
                    tool.func()
                    _HandoffRunner.hops += 1
                    yield SimpleNamespace(
                        content=SimpleNamespace(parts=[SimpleNamespace(text="root")]),
                        partial=False,
                        usage_metadata=SimpleNamespace(
                            prompt_token_count=1,
                            candidates_token_count=1,
                            total_token_count=2,
                        ),
                        is_final_response=lambda: True,
                    )
                    return
                yield SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[SimpleNamespace(text="leaf-answer")]
                    ),
                    partial=False,
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=2,
                        candidates_token_count=3,
                        total_token_count=5,
                    ),
                    is_final_response=lambda: True,
                )

        _HandoffRunner.hops = 0
        monkeypatch.setattr(native_mod, "InMemoryRunner", _HandoffRunner)
        result = await _adapter(_graph()).invoke("hi")
        assert _HandoffRunner.hops == 1
        assert len(_FakeRunner.created) == 2
        assert result["response"] == "leaf-answer"
        assert result["usage"]["total"] == 7

    async def test_context_emits_graph_span_and_events(self) -> None:
        tracker = MagicMock()
        span = MagicMock()
        tracer = MagicMock()
        tracer.start_span.return_value = span
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(native_mod, "get_client", lambda: tracker)
            patch.setattr(
                native_mod, "trace", SimpleNamespace(get_tracer=lambda _name: tracer)
            )
            await _adapter(_graph(), context={"kind": "user", "key": "user-1"}).invoke(
                "hi"
            )
        assert tracer.start_span.call_args.args[0] == "launchdarkly.graph"
        span.set_attribute.assert_any_call("launchdarkly.config.key", "graph-flag")
        attrs = [call.args[0] for call in span.set_attribute.call_args_list]
        assert "launchdarkly.run.id" in attrs
        events = [call.args[0] for call in tracker.track.call_args_list]
        assert "$ld:ai:graph:invocation_success" in events
        assert "$ld:ai:graph:duration:total" in events
        assert "$ld:ai:graph:total_tokens" in events
        span.end.assert_called()

    async def test_tracks_graph_node_and_graph_level_totals(self) -> None:
        tracker = MagicMock()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(native_mod, "get_client", lambda: tracker)
            await _adapter(_graph(), context={"kind": "user", "key": "user-1"}).invoke(
                "hi"
            )
        calls = [(call.args[0], call.args[2]) for call in tracker.track.call_args_list]
        assert [name for name, _ in calls] == [
            "$ld:ai:graph:node",
            "$ld:ai:graph:duration:total",
            "$ld:ai:graph:total_tokens",
            "$ld:ai:graph:invocation_success",
        ]
        node = calls[0][1]
        assert (node["nodeKey"], node["index"], node["configKey"]) == (
            "root",
            0,
            "root",
        )
        assert {data["configKey"] for _, data in calls[1:]} == {"graph-flag"}

    async def test_runner_error_tracks_invocation_failure_and_ends_the_span(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Boom(_FakeRunner):
            async def run_async(self, **kwargs: Any) -> Any:
                if False:
                    yield None
                raise RuntimeError("runner broke")

        tracker = MagicMock()
        span = MagicMock()
        tracer = MagicMock()
        tracer.start_span.return_value = span
        monkeypatch.setattr(native_mod, "InMemoryRunner", _Boom)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(native_mod, "get_client", lambda: tracker)
            patch.setattr(
                native_mod, "trace", SimpleNamespace(get_tracer=lambda _name: tracer)
            )
            with pytest.raises(RuntimeError, match="runner broke"):
                await _adapter(
                    _graph(), context={"kind": "user", "key": "user-1"}
                ).invoke("hi")
        events = [call.args[0] for call in tracker.track.call_args_list]
        assert events == ["$ld:ai:graph:node", "$ld:ai:graph:invocation_failure"]
        assert tracker.track.call_args_list[-1].args[2]["configKey"] == "graph-flag"
        span.end.assert_called()


class TestNodeInstructions:
    async def test_system_messages_become_the_instruction(self) -> None:
        graph = _graph()
        graph.root.config.pop("instructions")
        graph.root.config["messages"] = [
            {"role": "system", "content": "Route for {{team}}."}
        ]
        await _adapter(graph).invoke("hi", variables={"team": "ops"})
        instructions = {
            call.kwargs["name"]: call.kwargs["instruction"]
            for call in native_mod.LlmAgent.call_args_list
        }
        assert instructions["root"] == "Route for ops."

    async def test_config_turns_are_seeded_before_history(self) -> None:
        graph = _graph()
        graph.root.config["messages"] = [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "Example for {{team}}"},
        ]
        await _adapter(graph).invoke(
            "hi",
            variables={"team": "ops"},
            history=[{"role": "user", "content": "earlier"}],
        )
        appended = _FakeRunner.created[0].session_service.append_event.await_args_list
        texts = [call.args[1].content.parts[0].text for call in appended]
        assert texts == ["Example for ops", "earlier"]


class TestModelTransport:
    async def test_nodes_resolve_their_model_with_the_factory_options(self) -> None:
        await _adapter(
            _graph(), use_vertexai=True, project="p", location="us-central1"
        ).invoke("hi")
        kwargs = handler_mod.Gemini.call_args.kwargs
        assert kwargs["model"] == "gemini-2.5-flash"
        assert kwargs["client_kwargs"] == {
            "vertexai": True,
            "project": "p",
            "location": "us-central1",
        }

    def test_vertex_without_project_fails_up_front(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
        definition = _resolved(_graph())
        with pytest.raises(ValueError, match="Vertex"):
            to_adk_agents(definition, {"use_vertexai": True})
        definition.close()

    async def test_new_message_is_the_handler_user_message(self) -> None:
        sent: list[Any] = []

        class _Spy(_FakeRunner):
            async def run_async(self, **kwargs: Any) -> Any:
                sent.append(kwargs["new_message"])
                async for event in super().run_async(**kwargs):
                    yield event

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(native_mod, "InMemoryRunner", _Spy)
            await _adapter(_graph()).invoke("hi")
        assert sent[0] == handler_mod._user_message("hi")

    async def test_usage_sums_every_non_partial_event(self) -> None:
        class _TwoTurns(_FakeRunner):
            async def run_async(self, **kwargs: Any) -> Any:
                yield SimpleNamespace(
                    content=None,
                    partial=False,
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=4,
                        candidates_token_count=1,
                        total_token_count=5,
                    ),
                    is_final_response=lambda: False,
                )
                async for event in super().run_async(**kwargs):
                    yield event

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(native_mod, "InMemoryRunner", _TwoTurns)
            result = await _adapter(_graph()).invoke("hi")
        assert result["usage"] == {"input": 6, "output": 4, "total": 10}
