"""
Tests for launchdarkly-ai-langchain-agents handler.
Covers §1.1–1.9 (generic) and §2.x.1 span name.
Reference: TESTING.md §1, §2.x (LangChain)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_langchain_agents.handler as handler_mod
from launchdarkly_ai_langchain_agents.handler import (
    _build_initial_messages,
    _extract_system_prompt,
    create_langchain_agents_handler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs: Any) -> dict[str, Any]:
    base = {"model": {"name": "gpt-4o"}, "provider": {"name": "LangChain"}}
    base.update(kwargs)
    return base


def _make_ai_msg(
    content: str = "answer", input_tokens: int = 10, output_tokens: int = 5
) -> Any:
    msg = MagicMock()
    msg.content = content
    msg.type = "ai"
    msg.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    return msg


def _make_langchain_mock(response: str = "answer") -> Any:
    """Returns a mock LangChain-like module."""
    ai_msg = _make_ai_msg(response)
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=ai_msg)
    mock_llm.astream = AsyncMock(return_value=_empty_astream())

    lc_msgs = MagicMock()
    lc_msgs.HumanMessage = MagicMock(
        side_effect=lambda c: MagicMock(content=c, type="human")
    )
    lc_msgs.AIMessage = MagicMock(side_effect=lambda c: MagicMock(content=c, type="ai"))
    lc_msgs.SystemMessage = MagicMock(
        side_effect=lambda c: MagicMock(content=c, type="system")
    )

    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(return_value={"messages": [ai_msg]})

    mock_langgraph_prebuilt = MagicMock()
    mock_langgraph_prebuilt.create_react_agent = MagicMock(return_value=mock_agent)

    lc_tools = MagicMock()
    lc_tools.tool = MagicMock(side_effect=lambda name, fn=None, **kw: fn)

    return {
        "langgraph.prebuilt": mock_langgraph_prebuilt,
        "langchain_core.messages": lc_msgs,
        "langchain_core.tools": lc_tools,
        "langchain_openai": MagicMock(ChatOpenAI=MagicMock(return_value=mock_llm)),
        "_agent": mock_agent,
        "_llm": mock_llm,
        "_ai_msg": ai_msg,
    }


async def _empty_astream() -> AsyncIterator[Any]:
    return
    yield


def _patch_lc(mocks: dict[str, Any]) -> Any:
    _skip = {"_agent", "_llm", "_ai_msg"}

    def _side_effect(name: str) -> Any:
        if name in mocks and name not in _skip:
            return mocks[name]
        # Stub out packages that aren't installed in the test environment
        if name in ("langchain", "langchain_openai"):
            return MagicMock()
        return __import__(name)

    return patch("importlib.import_module", side_effect=_side_effect)


# ---------------------------------------------------------------------------
# §2.x.0 Agent creation API (Python: create_react_agent from langgraph.prebuilt)
# ---------------------------------------------------------------------------


class TestAgentCreationAPI:
    @pytest.mark.asyncio
    async def test_calls_create_react_agent_not_createAgent(self) -> None:
        """§2.x.0 — Python handler must call create_react_agent from langgraph.prebuilt."""
        captured: dict[str, Any] = {"called": False, "args": None, "kwargs": None}

        mock_agent = AsyncMock()
        ai_msg = _make_ai_msg("answer")
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [ai_msg]})

        def _fake_create_react_agent(*args: Any, **kwargs: Any) -> Any:
            captured["called"] = True
            captured["args"] = args
            captured["kwargs"] = kwargs
            return mock_agent

        mock_langgraph_prebuilt = MagicMock()
        mock_langgraph_prebuilt.create_react_agent = _fake_create_react_agent

        lc_msgs = MagicMock()
        lc_msgs.HumanMessage = MagicMock(
            side_effect=lambda c: MagicMock(content=c, type="human")
        )
        lc_msgs.AIMessage = MagicMock()
        lc_msgs.SystemMessage = MagicMock()

        def _import_side_effect(name: str) -> Any:
            if name == "langgraph.prebuilt":
                return mock_langgraph_prebuilt
            if name == "langchain":
                # Return a mock that does NOT have createAgent
                # so the handler would fail if it tries to call langchain.createAgent
                m = MagicMock(spec=[])  # empty spec means no attributes allowed
                return m
            if name == "langchain_core.messages":
                return lc_msgs
            if name == "langchain_core.tools":
                tools_mod = MagicMock()
                tools_mod.tool = MagicMock(side_effect=lambda name, fn=None, **kw: fn)
                return tools_mod
            if name == "langchain_openai":
                return MagicMock()
            return __import__(name)

        with patch("importlib.import_module", side_effect=_import_side_effect):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_langchain_agents_handler(llm=MagicMock())
                await h(_make_config(instructions="Be helpful."), "hi")

        assert captured["called"], (
            "create_react_agent from langgraph.prebuilt was not called. "
            "Handler must use create_react_agent, not langchain.createAgent."
        )


# ---------------------------------------------------------------------------
# §1.1 Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_returns_callable(self) -> None:
        h = create_langchain_agents_handler()
        assert callable(h)

    def test_attaches_provides_for(self) -> None:
        h = create_langchain_agents_handler()
        assert hasattr(h, "provides_for")

    def test_provides_for_values_are_correct(self) -> None:
        h = create_langchain_agents_handler()
        assert h.provides_for == ("*", "agent")

    def test_multiple_calls_return_independent_instances(self) -> None:
        h1 = create_langchain_agents_handler()
        h2 = create_langchain_agents_handler()
        assert h1 is not h2


# ---------------------------------------------------------------------------
# §1.2 Prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    def test_path_a_instructions(self) -> None:
        config = _make_config(instructions="Be helpful.")
        system = _extract_system_prompt(config, {})
        assert system == "Be helpful."

    def test_path_a_variable_substitution(self) -> None:
        config = _make_config(instructions="Hello {{name}}!")
        system = _extract_system_prompt(config, {"name": "Alice"})
        assert system == "Hello Alice!"

    def test_path_a_unresolved_placeholder_preserved(self) -> None:
        config = _make_config(instructions="Hello {{name}}!")
        system = _extract_system_prompt(config, {})
        assert "{{name}}" in (system or "")

    def test_path_b_messages_system_extracted(self) -> None:
        config = _make_config(
            messages=[
                {"role": "system", "content": "System msg."},
                {"role": "user", "content": "hello"},
            ]
        )
        system = _extract_system_prompt(config, {})
        assert "System msg." in (system or "")

    def test_path_b_variable_substitution_in_messages(self) -> None:
        config = _make_config(messages=[{"role": "user", "content": "I am {{name}}"}])
        _mocks = _make_langchain_mock()

        lc_msgs = MagicMock()
        lc_msgs.HumanMessage = MagicMock(
            side_effect=lambda c: MagicMock(content=c, type="human")
        )
        lc_msgs.AIMessage = MagicMock()
        lc_msgs.SystemMessage = MagicMock()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                lc_msgs if n == "langchain_core.messages" else __import__(n)
            ),
        ):
            msgs = _build_initial_messages(config, "q", {"name": "Bob"})
        # Verify the variable was substituted
        all_content = " ".join(m.content for m in msgs)
        assert "Bob" in all_content

    def test_path_b_user_input_not_duplicated_when_last_msg_is_user(self) -> None:
        config = _make_config(messages=[{"role": "user", "content": "old"}])
        lc_msgs = MagicMock()
        lc_msgs.HumanMessage = MagicMock(
            side_effect=lambda c: MagicMock(content=c, type="human")
        )
        lc_msgs.AIMessage = MagicMock()

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                lc_msgs if n == "langchain_core.messages" else __import__(n)
            ),
        ):
            msgs = _build_initial_messages(config, "new input", {})
        assert len(msgs) == 1
        assert msgs[0].content == "old"

    def test_path_b_user_input_appended_when_last_msg_is_assistant(self) -> None:
        config = _make_config(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        )
        lc_msgs = MagicMock()
        lc_msgs.HumanMessage = MagicMock(
            side_effect=lambda c: MagicMock(content=c, type="human")
        )
        lc_msgs.AIMessage = MagicMock(
            side_effect=lambda c: MagicMock(content=c, type="ai")
        )

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                lc_msgs if n == "langchain_core.messages" else __import__(n)
            ),
        ):
            msgs = _build_initial_messages(config, "new input", {})
        all_content = " ".join(m.content for m in msgs)
        assert "new input" in all_content

    def test_path_c_empty_user_input_no_throw(self) -> None:
        config = _make_config(instructions="help")
        lc_msgs = MagicMock()
        lc_msgs.HumanMessage = MagicMock(
            side_effect=lambda c: MagicMock(content=c, type="human")
        )
        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                lc_msgs if n == "langchain_core.messages" else __import__(n)
            ),
        ):
            msgs = _build_initial_messages(config, "", {})
        assert any(m.content == "" for m in msgs)

    def test_path_c_instructions_takes_priority_over_messages(self) -> None:
        config = _make_config(
            instructions="Use instructions.",
            messages=[{"role": "system", "content": "Use messages."}],
        )
        system = _extract_system_prompt(config, {})
        assert system == "Use instructions."


# ---------------------------------------------------------------------------
# §1.3 Tool conversion
# ---------------------------------------------------------------------------


class TestToolConversion:
    @pytest.mark.asyncio
    async def test_all_fields_forwarded(self) -> None:
        mocks = _make_langchain_mock()
        captured: list[dict[str, Any]] = []

        def _capture_tool(
            name: Any, fn: Any = None, description: str = "", args_schema: Any = None
        ) -> Any:
            captured.append({"name": name, "description": description})
            return fn

        mocks["langchain_core.tools"].tool = MagicMock(side_effect=_capture_tool)

        with _patch_lc(mocks):
            h = create_langchain_agents_handler(
                llm=MagicMock(ainvoke=AsyncMock(return_value=mocks["_ai_msg"]))
            )
            config = _make_config(
                tools={"my-tool": {"description": "does stuff", "parameters": {}}}
            )
            await h(config, "hi", {"my-tool": AsyncMock(return_value="ok")})

        names = [c["name"] for c in captured]
        assert "my-tool" in names

    @pytest.mark.asyncio
    async def test_multiple_tools_all_included(self) -> None:
        mocks = _make_langchain_mock()
        captured: list[str] = []

        def _capture_tool(name: Any, fn: Any = None, **kw: Any) -> Any:
            captured.append(name)
            return fn

        mocks["langchain_core.tools"].tool = MagicMock(side_effect=_capture_tool)

        with _patch_lc(mocks):
            h = create_langchain_agents_handler(
                llm=MagicMock(ainvoke=AsyncMock(return_value=mocks["_ai_msg"]))
            )
            config = _make_config(
                tools={
                    "tool-a": {"description": "a", "parameters": {}},
                    "tool-b": {"description": "b", "parameters": {}},
                }
            )
            await h(config, "hi", {"tool-a": AsyncMock(), "tool-b": AsyncMock()})

        assert "tool-a" in captured
        assert "tool-b" in captured

    @pytest.mark.asyncio
    async def test_empty_tools_no_tools_sent(self) -> None:
        mocks = _make_langchain_mock()
        captured_tools: list[Any] = []

        def _capture_tool(fn: Any, **kw: Any) -> Any:
            captured_tools.append(kw)
            return fn

        mocks["langchain_core.tools"].tool = MagicMock(side_effect=_capture_tool)

        with _patch_lc(mocks):
            h = create_langchain_agents_handler(
                llm=MagicMock(ainvoke=AsyncMock(return_value=mocks["_ai_msg"]))
            )
            await h(_make_config(), "hi")

        assert len(captured_tools) == 0


# ---------------------------------------------------------------------------
# §1.4 Tool execution loop
# ---------------------------------------------------------------------------


class TestToolExecutionLoop:
    @pytest.mark.asyncio
    async def test_tool_not_found_throws(self) -> None:
        mocks = _make_langchain_mock()
        captured_fns: list[Any] = []

        def _capture_tool(name: Any, fn: Any = None, **kw: Any) -> Any:
            captured_fns.append(fn)
            return fn

        mocks["langchain_core.tools"].tool = MagicMock(side_effect=_capture_tool)

        with _patch_lc(mocks):
            config = _make_config(
                tools={"my-tool": {"description": "d", "parameters": {}}}
            )
            from launchdarkly_ai_langchain_agents.handler import _build_agent_tools

            _build_agent_tools(config["tools"], {})  # no handler registered

        if captured_fns:
            with pytest.raises(ValueError, match="No handler"):
                await captured_fns[0](key="val")

    @pytest.mark.asyncio
    async def test_tool_handler_throws_propagates(self) -> None:
        mocks = _make_langchain_mock()
        captured_fns: list[Any] = []

        def _capture_tool(name: Any, fn: Any = None, **kw: Any) -> Any:
            captured_fns.append(fn)
            return fn

        mocks["langchain_core.tools"].tool = MagicMock(side_effect=_capture_tool)

        async def _bad_handler(args: Any) -> str:
            raise RuntimeError("tool error")

        with _patch_lc(mocks):
            from launchdarkly_ai_langchain_agents.handler import _build_agent_tools

            _build_agent_tools(
                {"my-tool": {"description": "d", "parameters": {}}},
                {"my-tool": _bad_handler},
            )

        if captured_fns:
            with pytest.raises(RuntimeError, match="tool error"):
                await captured_fns[0](key="val")

    @pytest.mark.asyncio
    async def test_no_tools_in_config_handler_never_invoked(self) -> None:
        mocks = _make_langchain_mock()
        handler_fn = AsyncMock(return_value="ok")

        with _patch_lc(mocks):
            h = create_langchain_agents_handler(
                llm=MagicMock(ainvoke=AsyncMock(return_value=mocks["_ai_msg"]))
            )
            await h(_make_config(), "hi", {"my-tool": handler_fn})

        handler_fn.assert_not_called()


# ---------------------------------------------------------------------------
# §1.5 Telemetry
# ---------------------------------------------------------------------------


class TestTelemetry:
    @pytest.mark.asyncio
    async def test_span_name(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        mocks = _make_langchain_mock()
        with _patch_lc(mocks):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_langchain_agents_handler(
                        llm=MagicMock(ainvoke=AsyncMock(return_value=mocks["_ai_msg"]))
                    )
                    await h(_make_config(), "hi")

        mock_trace.get_tracer.return_value.start_span.assert_called_with(
            "langchain.agent"
        )

    @pytest.mark.asyncio
    async def test_gen_ai_system(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        mocks = _make_langchain_mock()
        with _patch_lc(mocks):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_langchain_agents_handler(
                        llm=MagicMock(ainvoke=AsyncMock(return_value=mocks["_ai_msg"]))
                    )
                    await h(_make_config(), "hi")

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("gen_ai.system") == "langchain"

    @pytest.mark.asyncio
    async def test_gen_ai_operation_name(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        mocks = _make_langchain_mock()
        with _patch_lc(mocks):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_langchain_agents_handler(
                        llm=MagicMock(ainvoke=AsyncMock(return_value=mocks["_ai_msg"]))
                    )
                    await h(_make_config(), "hi")

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("gen_ai.operation.name") == "chat"

    @pytest.mark.asyncio
    async def test_span_status_ok(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        mocks = _make_langchain_mock()
        with _patch_lc(mocks):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_langchain_agents_handler(
                        llm=MagicMock(ainvoke=AsyncMock(return_value=mocks["_ai_msg"]))
                    )
                    await h(_make_config(), "hi")

        mock_span.set_status.assert_called()

    @pytest.mark.asyncio
    async def test_span_end_always_called(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        mocks = _make_langchain_mock()
        with _patch_lc(mocks):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_langchain_agents_handler(
                        llm=MagicMock(ainvoke=AsyncMock(return_value=mocks["_ai_msg"]))
                    )
                    await h(_make_config(), "hi")

        mock_span.end.assert_called()

    @pytest.mark.asyncio
    async def test_gen_ai_request_model(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        mocks = _make_langchain_mock()
        with _patch_lc(mocks):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_langchain_agents_handler(
                        llm=MagicMock(ainvoke=AsyncMock(return_value=mocks["_ai_msg"]))
                    )
                    await h(_make_config(model={"name": "gpt-4o"}), "hi")

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("gen_ai.request.model") == "gpt-4o"

    @pytest.mark.asyncio
    async def test_gen_ai_content_prompt_event(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        mocks = _make_langchain_mock()
        with _patch_lc(mocks):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_langchain_agents_handler(
                        llm=MagicMock(ainvoke=AsyncMock(return_value=mocks["_ai_msg"]))
                    )
                    await h(_make_config(), "my question")

        event_calls = [
            c
            for c in mock_span.add_event.call_args_list
            if c[0][0] == "gen_ai.content.prompt"
        ]
        assert event_calls
        # The gen_ai.prompt attribute must include the user input text
        prompt_attr = event_calls[0][0][1].get("gen_ai.prompt", "")
        assert "my question" in prompt_attr, (
            f"gen_ai.prompt must include user input 'my question', got: {prompt_attr!r}"
        )

    @pytest.mark.asyncio
    async def test_token_attributes_set(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        ai_msg = _make_ai_msg("answer", input_tokens=30, output_tokens=12)
        mocks = _make_langchain_mock()

        with _patch_lc(mocks):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_langchain_agents_handler(
                        llm=MagicMock(ainvoke=AsyncMock(return_value=ai_msg))
                    )
                    await h(_make_config(), "hi")

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert (
            "gen_ai.usage.input_tokens" in calls
            or "gen_ai.usage.output_tokens" in calls
        )

    @pytest.mark.asyncio
    async def test_gen_ai_content_completion_event(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        mocks = _make_langchain_mock()
        with _patch_lc(mocks):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_langchain_agents_handler(
                        llm=MagicMock(ainvoke=AsyncMock(return_value=mocks["_ai_msg"]))
                    )
                    await h(_make_config(), "hi")

        event_calls = [
            c
            for c in mock_span.add_event.call_args_list
            if c[0][0] == "gen_ai.content.completion"
        ]
        assert event_calls

    @pytest.mark.asyncio
    async def test_gen_ai_response_model(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        mocks = _make_langchain_mock()
        with _patch_lc(mocks):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_langchain_agents_handler(
                        llm=MagicMock(ainvoke=AsyncMock(return_value=mocks["_ai_msg"]))
                    )
                    await h(_make_config(model={"name": "gpt-4o"}), "hi")

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert "gen_ai.response.model" in calls
        assert calls["gen_ai.response.model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_ld_span_attributes(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        mocks = _make_langchain_mock()
        variables = {
            "__ld": {
                "configKey": "my-config",
                "variationKey": "v1",
                "runId": "run-abc",
            }
        }
        with _patch_lc(mocks):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_langchain_agents_handler(
                        llm=MagicMock(ainvoke=AsyncMock(return_value=mocks["_ai_msg"]))
                    )
                    await h(_make_config(), "hi", variables=variables)

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("launchdarkly.operation.type") == "gen_ai"
        assert calls.get("launchdarkly.config.key") == "my-config"
        assert calls.get("launchdarkly.variation.key") == "v1"
        assert calls.get("launchdarkly.run.id") == "run-abc"
        assert "launchdarkly.graph.key" not in calls

    async def test_ld_graph_key_set_when_present(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        mocks = _make_langchain_mock()
        variables = {
            "__ld": {
                "configKey": "my-config",
                "variationKey": "v1",
                "runId": "run-abc",
                "graphKey": "my-graph",
            }
        }
        with _patch_lc(mocks):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_langchain_agents_handler(
                        llm=MagicMock(ainvoke=AsyncMock(return_value=mocks["_ai_msg"]))
                    )
                    await h(_make_config(), "hi", variables=variables)

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("launchdarkly.graph.key") == "my-graph"


# ---------------------------------------------------------------------------
# §1.6 Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_records_exception_on_span(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        mocks = _make_langchain_mock()
        mocks["_agent"].ainvoke = AsyncMock(side_effect=RuntimeError("lc error"))
        mocks["langgraph.prebuilt"].create_react_agent = MagicMock(
            return_value=mocks["_agent"]
        )

        with _patch_lc(mocks):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_langchain_agents_handler(
                        llm=MagicMock(ainvoke=AsyncMock(return_value=None))
                    )
                    with pytest.raises(RuntimeError):
                        await h(_make_config(), "hi")

        mock_span.record_exception.assert_called()

    @pytest.mark.asyncio
    async def test_sets_span_status_error(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        mocks = _make_langchain_mock()
        mocks["_agent"].ainvoke = AsyncMock(side_effect=RuntimeError("lc error"))

        with _patch_lc(mocks):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_langchain_agents_handler(
                        llm=MagicMock(ainvoke=AsyncMock())
                    )
                    with pytest.raises(RuntimeError):
                        await h(_make_config(), "hi")

        mock_span.set_status.assert_called()

    @pytest.mark.asyncio
    async def test_ends_span_on_error(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        mocks = _make_langchain_mock()
        mocks["_agent"].ainvoke = AsyncMock(side_effect=RuntimeError("lc error"))

        with _patch_lc(mocks):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_langchain_agents_handler(
                        llm=MagicMock(ainvoke=AsyncMock())
                    )
                    with pytest.raises(RuntimeError):
                        await h(_make_config(), "hi")

        mock_span.end.assert_called()

    @pytest.mark.asyncio
    async def test_rethrows_error(self) -> None:
        mocks = _make_langchain_mock()
        mocks["_agent"].ainvoke = AsyncMock(side_effect=RuntimeError("specific error"))

        with _patch_lc(mocks):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_langchain_agents_handler(llm=MagicMock())
                with pytest.raises(RuntimeError, match="specific error"):
                    await h(_make_config(), "hi")


# ---------------------------------------------------------------------------
# §1.7 Convenience export
# ---------------------------------------------------------------------------


class TestConvenienceExport:
    def test_calls_through_to_model_call(self) -> None:
        from launchdarkly_ai_langchain_agents.handler import langchain_agents

        assert callable(langchain_agents)

    def test_passes_config_key_user_input_and_context(self) -> None:
        import inspect

        from launchdarkly_ai_langchain_agents.handler import langchain_agents

        sig = inspect.signature(langchain_agents)
        assert "config_key" in sig.parameters
        assert "user_input" in sig.parameters
        assert "context" in sig.parameters

    def test_config_key_forwarded_as_key(self) -> None:
        import launchdarkly_ai_langchain_agents.handler as handler_mod

        mock_config_instance = MagicMock()
        mock_config_fn = MagicMock(return_value=mock_config_instance)
        mock_config_instance.invoke = MagicMock(return_value="result")

        with patch.object(handler_mod, "config", mock_config_fn):
            from launchdarkly_ai_langchain_agents.handler import langchain_agents

            ctx = {"kind": "user", "key": "u1"}
            langchain_agents("my-flag", "hello", ctx)

        mock_config_fn.assert_called_once()
        call_kwargs = mock_config_fn.call_args.kwargs
        assert call_kwargs.get("key") == "my-flag"
        handler = call_kwargs.get("handler")
        assert handler is not None
        assert handler.provides_for == ("*", "agent")
        mock_config_instance.invoke.assert_called_once_with(
            "hello", ctx, variables=None
        )

    def test_callable_without_extra_kwargs(self) -> None:
        import launchdarkly_ai_langchain_agents.handler as handler_mod

        mock_config_instance = MagicMock()
        mock_config_fn = MagicMock(return_value=mock_config_instance)
        mock_config_instance.invoke = MagicMock(return_value="result")

        with patch.object(handler_mod, "config", mock_config_fn):
            from launchdarkly_ai_langchain_agents.handler import langchain_agents

            ctx = {"kind": "user", "key": "u1"}
            langchain_agents("my-flag", "hello", ctx)

        mock_config_fn.assert_called_once()
        mock_config_instance.invoke.assert_called_once_with(
            "hello", ctx, variables=None
        )


# ---------------------------------------------------------------------------
# §1.8 Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_stream_is_defined(self) -> None:
        h = create_langchain_agents_handler()
        assert hasattr(h, "stream")

    @pytest.mark.asyncio
    async def test_stream_returns_async_generator(self) -> None:
        import inspect

        mocks = _make_langchain_mock()

        async def _mock_astream(*a: Any, **kw: Any) -> AsyncIterator[Any]:
            return
            yield

        mocks["_agent"].astream = _mock_astream

        with _patch_lc(mocks):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_langchain_agents_handler(llm=MagicMock())
                gen = await h.stream(_make_config(), "hi")
                assert inspect.isasyncgen(gen) or hasattr(gen, "__aiter__")

    @pytest.mark.asyncio
    async def test_yields_exactly_one_done_event(self) -> None:
        mocks = _make_langchain_mock()

        async def _mock_astream(*a: Any, **kw: Any) -> AsyncIterator[Any]:
            return
            yield

        mocks["_agent"].astream = _mock_astream

        with _patch_lc(mocks):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_langchain_agents_handler(llm=MagicMock())
                events = [e async for e in await h.stream(_make_config(), "hi")]

        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1

    @pytest.mark.asyncio
    async def test_generator_throws_on_provider_error(self) -> None:
        mocks = _make_langchain_mock()

        async def _bad_astream(*a: Any, **kw: Any) -> AsyncIterator[Any]:
            raise RuntimeError("stream fail")
            yield

        mocks["_agent"].astream = _bad_astream

        with _patch_lc(mocks):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_langchain_agents_handler(llm=MagicMock())
                with pytest.raises(RuntimeError, match="stream fail"):
                    async for _ in await h.stream(_make_config(), "hi"):
                        pass


# ---------------------------------------------------------------------------
# §1.5 Streaming telemetry (Appendix A.5 — do not patch _HAS_OTEL=False)
# ---------------------------------------------------------------------------


class TestStreamingTelemetry:
    @pytest.mark.asyncio
    async def test_span_started_during_stream(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        mocks = _make_langchain_mock()

        async def _empty_astream(*a: Any, **kw: Any) -> AsyncIterator[Any]:
            return
            yield

        mocks["_agent"].astream = _empty_astream

        with _patch_lc(mocks):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_langchain_agents_handler(llm=MagicMock())
                    async for _ in await h.stream(_make_config(), "hi"):
                        pass

        mock_trace.get_tracer.return_value.start_span.assert_called_with(
            "langchain.agent.stream"
        )

    @pytest.mark.asyncio
    async def test_ld_span_attributes_set_during_stream(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        mocks = _make_langchain_mock()

        async def _empty_astream(*a: Any, **kw: Any) -> AsyncIterator[Any]:
            return
            yield

        mocks["_agent"].astream = _empty_astream
        variables = {"__ld": {"configKey": "k", "variationKey": "v", "runId": "r"}}

        with _patch_lc(mocks):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_langchain_agents_handler(llm=MagicMock())
                    async for _ in await h.stream(
                        _make_config(), "hi", None, variables
                    ):
                        pass

        calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert calls.get("launchdarkly.operation.type") == "gen_ai"
        assert calls.get("launchdarkly.config.key") == "k"
        assert calls.get("launchdarkly.variation.key") == "v"
        assert calls.get("launchdarkly.run.id") == "r"

    @pytest.mark.asyncio
    async def test_span_ended_after_stream_completes(self) -> None:
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value.start_span.return_value = mock_span

        mocks = _make_langchain_mock()

        async def _empty_astream(*a: Any, **kw: Any) -> AsyncIterator[Any]:
            return
            yield

        mocks["_agent"].astream = _empty_astream

        with _patch_lc(mocks):
            with patch.object(handler_mod, "trace", mock_trace):
                with patch.object(handler_mod, "_HAS_OTEL", True):
                    h = create_langchain_agents_handler(llm=MagicMock())
                    async for _ in await h.stream(_make_config(), "hi"):
                        pass

        mock_span.end.assert_called()


# ---------------------------------------------------------------------------
# §1.9 Output format
# ---------------------------------------------------------------------------


class TestOutputFormat:
    @pytest.mark.asyncio
    async def test_absent_output_format_no_change(self) -> None:
        mocks = _make_langchain_mock()
        with _patch_lc(mocks):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_langchain_agents_handler(
                    llm=MagicMock(ainvoke=AsyncMock(return_value=mocks["_ai_msg"]))
                )
                result = await h(_make_config(), "hi")
        assert "output" in result

    @pytest.mark.asyncio
    async def test_output_format_appends_schema_instruction(self) -> None:
        mocks = _make_langchain_mock()
        captured_calls: list[tuple[Any, ...]] = []

        def _capture_agent(*args: Any, **kw: Any) -> Any:
            captured_calls.append((args, kw))
            return mocks["_agent"]

        mocks["langgraph.prebuilt"].create_react_agent = MagicMock(
            side_effect=_capture_agent
        )

        with _patch_lc(mocks):
            with patch.object(handler_mod, "_HAS_OTEL", False):
                h = create_langchain_agents_handler(
                    llm=MagicMock(ainvoke=AsyncMock(return_value=mocks["_ai_msg"]))
                )
                config = _make_config(
                    outputFormat={
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    }
                )
                await h(config, "hi")

        if captured_calls:
            # The system prompt is passed as the "prompt" keyword argument
            kw = captured_calls[0][1]
            system = kw.get("prompt", "")
            assert (
                "json" in (system or "").lower() or "schema" in (system or "").lower()
            )


# ---------------------------------------------------------------------------
# §1.2 Path C — None user_input must not raise or produce None content
# ---------------------------------------------------------------------------


class TestNoneUserInput:
    """TESTING.md §1.2 Path C: _build_initial_messages must not pass None to
    HumanMessage when user_input is None."""

    def test_none_user_input_no_none_human_message_content(self) -> None:
        """HumanMessage must be constructed with '' not None when user_input=None."""
        config = _make_config(instructions="help")
        captured_human_args: list[Any] = []

        lc_msgs = MagicMock()
        lc_msgs.HumanMessage = MagicMock(
            side_effect=lambda c: (
                captured_human_args.append(c) or MagicMock(content=c, type="human")
            )
        )
        lc_msgs.AIMessage = MagicMock(
            side_effect=lambda c: MagicMock(content=c, type="ai")
        )

        with patch(
            "importlib.import_module",
            side_effect=lambda n: (
                lc_msgs if n == "langchain_core.messages" else __import__(n)
            ),
        ):
            _build_initial_messages(config, None, {})

        # Every HumanMessage must have non-None content
        assert len(captured_human_args) > 0
        for arg in captured_human_args:
            assert arg is not None, (
                "HumanMessage was constructed with None content; "
                "must use '' when user_input is None (TESTING.md §1.2 Path C)"
            )


# ---------------------------------------------------------------------------
# History parameter
# ---------------------------------------------------------------------------


class TestHistory:
    SAMPLE_HISTORY: ClassVar[list[dict[str, Any]]] = [
        {"role": "user", "content": "What is feature flagging?"},
        {"role": "assistant", "content": "Feature flagging is a technique..."},
    ]

    def test_history_appended_to_system_prompt(self) -> None:
        config = _make_config(instructions="Be concise.")
        system = _extract_system_prompt(config, {}, self.SAMPLE_HISTORY)
        assert system is not None
        assert "Conversation History:" in system
        assert "Be concise." in system

    def test_history_format_is_correct(self) -> None:
        config = _make_config(instructions="Be helpful.")
        system = _extract_system_prompt(config, {}, self.SAMPLE_HISTORY)
        assert system is not None
        assert "user: What is feature flagging?" in system
        assert "assistant: Feature flagging is a technique..." in system

    def test_empty_history_treated_like_no_history(self) -> None:
        config = _make_config(instructions="Be concise.")
        system_with_empty = _extract_system_prompt(config, {}, [])
        system_without = _extract_system_prompt(config, {})
        assert system_with_empty == system_without
        assert "Conversation History:" not in (system_with_empty or "")

    def test_history_without_prior_system_prompt(self) -> None:
        config = _make_config()
        system = _extract_system_prompt(config, {}, self.SAMPLE_HISTORY)
        assert system is not None
        assert "Conversation History:" in system
        assert "user: What is feature flagging?" in system
