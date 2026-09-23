"""Native ADK workflow adapter. Reference: TESTING.md §2.2 and §2.x.4."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

import launchdarkly_ai_google_adk_agents.native_graph as native_mod
from launchdarkly_ai_google_adk_agents.native_graph import to_adk_agents


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
    monkeypatch.setattr(
        native_mod,
        "Workflow",
        MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw)),
    )
    monkeypatch.setattr(native_mod, "InMemoryRunner", _FakeRunner)
    monkeypatch.setattr(native_mod, "START", "START")


class TestToAdkAgents:
    async def test_disabled_graph_throws(self) -> None:
        with pytest.raises(Exception, match="graph-flag"):
            await to_adk_agents(_graph(enabled=False)).invoke("hi")

    async def test_missing_root_throws(self) -> None:
        graph = _graph()
        graph.root = None
        with pytest.raises(Exception, match="root"):
            await to_adk_agents(graph).invoke("hi")

    async def test_two_node_graph_builds_two_agents_and_a_transfer_tool(self) -> None:
        result = await to_adk_agents(_graph()).invoke("hi")
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
        edges = native_mod.Workflow.call_args.kwargs["edges"]
        assert edges[0][0] == "START"
        assert result["response"] == "leaf-answer"
        assert result["usage"]["total"] == 5

    async def test_transfer_tool_selects_the_target(self) -> None:
        await to_adk_agents(_graph()).invoke("hi")
        tool = next(
            item for item in _FakeTool.created if item.name == "transfer_to_leaf"
        )
        selected = tool.func()
        assert "leaf" in str(selected)

    async def test_no_context_does_not_track(self) -> None:
        tracker = MagicMock()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(native_mod, "get_client", lambda: tracker)
            await to_adk_agents(_graph()).invoke("hi")
        tracker.track.assert_not_called()

    async def test_history_is_seeded_on_the_root_session_only(self) -> None:
        await to_adk_agents(_graph()).invoke(
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

        await to_adk_agents(graph, tool_handlers={"lookup": lookup}).invoke("hi")
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
        result = await to_adk_agents(_graph()).invoke("hi")
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
            await to_adk_agents(_graph()).invoke(
                "hi", context={"kind": "user", "key": "user-1"}
            )
        assert tracer.start_span.call_args.args[0] == "ld.ai.graph"
        events = [call.args[0] for call in tracker.track.call_args_list]
        assert "$ld:ai:graph:invocation_success" in events
        assert "$ld:ai:graph:duration:total" in events
        assert "$ld:ai:graph:total_tokens" in events
        span.end.assert_called()

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
                await to_adk_agents(_graph()).invoke(
                    "hi", context={"kind": "user", "key": "user-1"}
                )
        events = [call.args[0] for call in tracker.track.call_args_list]
        assert events == ["$ld:ai:graph:invocation_failure"]
        span.end.assert_called()
