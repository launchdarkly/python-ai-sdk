"""
Tests for launchdarkly-ai-langchain-agents handler.
Covers §1.1–1.9 (generic) and §2.x.1 span name.
Reference: TESTING.md §1, §2.x (LangChain)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pydantic
import pytest
from langchain_core.language_models.chat_models import BaseChatModel

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


class _FakeToolModel(BaseChatModel):
    """A real ``BaseChatModel`` (so ``create_react_agent`` accepts it) that returns canned replies
    in sequence and never touches the network. ``bind_tools`` is required by the agent graph and
    the base class raises ``NotImplementedError`` for it.
    """

    replies: list[Any] = pydantic.Field(default_factory=list)
    fail_after: int | None = None
    fail_with: Exception | None = None
    calls: int = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> Any:
        raise NotImplementedError

    async def _agenerate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> Any:
        from langchain_core.outputs import ChatGeneration, ChatResult

        if self.fail_after is not None and self.calls >= self.fail_after:
            raise self.fail_with or RuntimeError("model down")
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-model"


def _ai_message(
    content: str = "",
    input_tokens: int = 10,
    output_tokens: int = 5,
    tool_calls: list[dict[str, Any]] | None = None,
    response_metadata: dict[str, Any] | None = None,
) -> Any:
    from langchain_core.messages import AIMessage

    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        tool_calls=tool_calls or [],
        response_metadata=response_metadata or {},
    )


class RecordedSpan:
    """A span that remembers what a handler did to it, so a test can assert on the whole thing."""

    def __init__(self, name: str, context: Any = None) -> None:
        self.name = name
        self.context = context
        self.attributes: dict[str, Any] = {}
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.statuses: list[Any] = []
        self.exceptions: list[BaseException] = []
        self.ended = 0

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append((name, attributes or {}))

    def set_status(self, code: Any, description: str | None = None) -> None:
        self.statuses.append(code)

    def record_exception(self, exc: BaseException) -> None:
        self.exceptions.append(exc)

    def end(self) -> None:
        self.ended += 1


class SpanRecorder:
    """Stands in for the ``trace`` module inside ``spans.py`` and records every span opened.

    A single MagicMock cannot see a span tree at all: every span would be the same object, so a
    parent and its children would be indistinguishable.
    """

    def __init__(self) -> None:
        self.spans: list[RecordedSpan] = []

    def get_tracer(self, name: str) -> SpanRecorder:
        return self

    def start_span(self, name: str, context: Any = None) -> RecordedSpan:
        span = RecordedSpan(name, context)
        self.spans.append(span)
        return span

    def set_span_in_context(self, span: RecordedSpan) -> Any:
        return ("context-of", span)

    @property
    def root(self) -> RecordedSpan:
        return self.spans[0]

    def named(self, prefix: str) -> list[RecordedSpan]:
        return [s for s in self.spans if s.name.startswith(prefix)]

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.spans]


def _recording() -> Any:
    """Patches the tracer that ``spans.py`` holds, and yields the recorder.

    ``AsyncCallbackHandler`` (the base class of ``SpanCallbackHandler``) stays real: LangChain's own
    callback machinery decides when ``on_chat_model_start`` / ``on_llm_end`` / ``on_tool_start`` /
    ``on_tool_end`` fire, and that dispatch is exactly what these tests need to prove, not something
    to mock away.
    """
    import launchdarkly_ai_langchain_agents.spans as spans_mod

    recorder = SpanRecorder()
    return patch.object(spans_mod, "trace", recorder), recorder


BASE_CONFIG: dict[str, Any] = {
    "model": {"name": "gpt-4o"},
    "provider": {"name": "OpenAI"},
    "instructions": "You are helpful.",
}

TOOL_CONFIG: dict[str, Any] = {
    **BASE_CONFIG,
    "tools": {
        "search": {
            "description": "search the web",
            "parameters": {"type": "object", "properties": {}},
        }
    },
}


class TestSpanTree:
    """TELEMETRY-CONTRACT.md section 1. A real langgraph agent and a real LangChain callback
    dispatch, against a tracer this file controls."""

    @pytest.mark.asyncio
    async def test_opens_a_root_span_named_invoke_agent(self) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(replies=[_ai_message("Hello!")])
        with ctx:
            await create_langchain_agents_handler(llm)(BASE_CONFIG, "q")
        assert rec.root.name == "invoke_agent"
        assert rec.root.attributes["gen_ai.operation.name"] == "invoke_agent"

    @pytest.mark.asyncio
    async def test_emits_one_chat_child_per_model_turn(self) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(replies=[_ai_message("Hello!")])
        with ctx:
            await create_langchain_agents_handler(llm)(BASE_CONFIG, "q")
        chats = rec.named("chat ")
        assert len(chats) == 1
        assert chats[0].name == "chat gpt-4o"
        assert chats[0].attributes["gen_ai.operation.name"] == "chat"
        assert chats[0].context == ("context-of", rec.root)

    @pytest.mark.asyncio
    async def test_names_the_chat_span_after_the_model(self) -> None:
        ctx, rec = _recording()
        cfg = {**BASE_CONFIG, "model": {"name": "claude-sonnet-4-5"}}
        llm = _FakeToolModel(replies=[_ai_message("hi")])
        with ctx:
            await create_langchain_agents_handler(llm)(cfg, "q")
        assert "chat claude-sonnet-4-5" in rec.names

    @pytest.mark.asyncio
    async def test_emits_a_chat_span_per_turn_of_a_tool_loop(self) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(
            replies=[
                _ai_message(
                    tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call_1"}]
                ),
                _ai_message("done"),
            ]
        )
        with ctx:
            await create_langchain_agents_handler(llm)(
                TOOL_CONFIG, "q", {"search": AsyncMock(return_value="r")}
            )
        assert len(rec.named("chat ")) == 2

    @pytest.mark.asyncio
    async def test_emits_an_execute_tool_span_per_tool_call(self) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(
            replies=[
                _ai_message(
                    tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call_1"}]
                ),
                _ai_message("done"),
            ]
        )
        with ctx:
            await create_langchain_agents_handler(llm)(
                TOOL_CONFIG, "q", {"search": AsyncMock(return_value="r")}
            )
        tools = rec.named("execute_tool ")
        assert len(tools) == 1
        assert tools[0].name == "execute_tool search"
        assert tools[0].attributes["gen_ai.operation.name"] == "execute_tool"
        assert tools[0].attributes["gen_ai.tool.name"] == "search"
        assert tools[0].attributes["gen_ai.tool.call.id"] == "call_1"

    @pytest.mark.asyncio
    async def test_tool_spans_are_siblings_of_chat_not_children(self) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(
            replies=[
                _ai_message(
                    tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call_1"}]
                ),
                _ai_message("done"),
            ]
        )
        with ctx:
            await create_langchain_agents_handler(llm)(
                TOOL_CONFIG, "q", {"search": AsyncMock(return_value="r")}
            )
        assert rec.named("execute_tool ")[0].context == ("context-of", rec.root)

    @pytest.mark.asyncio
    async def test_every_span_is_ended(self) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(
            replies=[
                _ai_message(
                    tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call_1"}]
                ),
                _ai_message("done"),
            ]
        )
        with ctx:
            await create_langchain_agents_handler(llm)(
                TOOL_CONFIG, "q", {"search": AsyncMock(return_value="r")}
            )
        assert [s.ended for s in rec.spans] == [1] * len(rec.spans)

    @pytest.mark.asyncio
    async def test_nests_the_chat_span_under_the_root_in_the_streaming_path(
        self,
    ) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(replies=[_ai_message("Hello!")])
        with ctx:
            async for _ in await create_langchain_agents_handler(llm).stream(
                BASE_CONFIG, "q", {}, {}
            ):
                pass
        chats = rec.named("chat ")
        assert chats
        assert chats[0].context == ("context-of", rec.root)


class TestRootSpanAttributes:
    """TELEMETRY-CONTRACT.md sections 2, 2a and 9."""

    @pytest.mark.asyncio
    async def test_gen_ai_provider_name_is_openai_for_a_non_anthropic_config(
        self,
    ) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(replies=[_ai_message("hi")])
        with ctx:
            await create_langchain_agents_handler(llm)(BASE_CONFIG, "q")
        attrs = rec.root.attributes
        # gen_ai.provider.name names who really served the model: ChatOpenAI here. gen_ai.system
        # keeps the framework name so existing dashboards do not break.
        assert attrs["gen_ai.provider.name"] == "openai"
        assert attrs["gen_ai.system"] == "langchain"

    @pytest.mark.asyncio
    async def test_gen_ai_provider_name_is_anthropic_when_config_names_it(self) -> None:
        ctx, rec = _recording()
        cfg = {
            **BASE_CONFIG,
            "provider": {"name": "Anthropic"},
            "model": {"name": "claude-sonnet-4-5"},
        }
        llm = _FakeToolModel(replies=[_ai_message("hi")])
        with ctx:
            await create_langchain_agents_handler(llm)(cfg, "q")
        assert rec.root.attributes["gen_ai.provider.name"] == "anthropic"
        assert rec.root.attributes["gen_ai.system"] == "langchain"

    @pytest.mark.asyncio
    async def test_gen_ai_provider_name_falls_back_to_openai_for_anything_else(
        self,
    ) -> None:
        # A binary choice, not a passthrough: Bedrock, Azure, Cohere, a typo, or nothing at all all
        # report `openai`, because that mirrors which chat model class is really instantiated.
        ctx, rec = _recording()
        cfg = {**BASE_CONFIG, "provider": {"name": "Bedrock"}}
        llm = _FakeToolModel(replies=[_ai_message("hi")])
        with ctx:
            await create_langchain_agents_handler(llm)(cfg, "q")
        assert rec.root.attributes["gen_ai.provider.name"] == "openai"

    @pytest.mark.asyncio
    async def test_response_model_is_the_requested_name(self) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(replies=[_ai_message("hi")])
        with ctx:
            await create_langchain_agents_handler(llm)(BASE_CONFIG, "q")
        assert rec.root.attributes["gen_ai.response.model"] == "gpt-4o"
        assert rec.root.attributes["gen_ai.request.model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_carries_the_launchdarkly_attributes_and_feature_flag_event(
        self,
    ) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(replies=[_ai_message("hi")])
        variables = {
            "__ld": {
                "configKey": "k",
                "variationKey": "v",
                "runId": "r",
                "graphKey": "g",
            }
        }
        with ctx:
            await create_langchain_agents_handler(llm)(BASE_CONFIG, "q", {}, variables)
        attrs = rec.root.attributes
        assert attrs["launchdarkly.operation.type"] == "gen_ai"
        assert attrs["launchdarkly.config.key"] == "k"
        assert attrs["launchdarkly.variation.key"] == "v"
        assert attrs["launchdarkly.run.id"] == "r"
        assert attrs["launchdarkly.graph.key"] == "g"
        assert [n for n, _ in rec.root.events] == ["feature_flag"]

    @pytest.mark.asyncio
    async def test_child_spans_carry_no_launchdarkly_identity(self) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(
            replies=[
                _ai_message(
                    tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call_1"}]
                ),
                _ai_message("done"),
            ]
        )
        variables = {"__ld": {"configKey": "k", "variationKey": "v", "runId": "r"}}
        with ctx:
            await create_langchain_agents_handler(llm)(
                TOOL_CONFIG, "q", {"search": AsyncMock(return_value="r")}, variables
            )
        for child in rec.spans[1:]:
            assert not [k for k in child.attributes if k.startswith("launchdarkly.")]
            assert "feature_flag" not in [n for n, _ in child.events]

    @pytest.mark.asyncio
    async def test_carries_the_run_total_not_one_turn(self) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(
            replies=[
                _ai_message(
                    input_tokens=10,
                    output_tokens=1,
                    tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call_1"}],
                ),
                _ai_message("done", input_tokens=20, output_tokens=2),
            ]
        )
        with ctx:
            await create_langchain_agents_handler(llm)(
                TOOL_CONFIG, "q", {"search": AsyncMock(return_value="r")}
            )
        assert rec.root.attributes["gen_ai.usage.input_tokens"] == 30
        assert rec.root.attributes["gen_ai.usage.output_tokens"] == 3
        assert rec.root.attributes["gen_ai.usage.total_tokens"] == 33


class TestChatSpanAttributes:
    """TELEMETRY-CONTRACT.md sections 3, 5 and 8."""

    @pytest.mark.asyncio
    async def test_writes_all_seven_usage_attributes_including_zeros(self) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(
            replies=[_ai_message("hi", input_tokens=10, output_tokens=5)]
        )
        with ctx:
            await create_langchain_agents_handler(llm)(BASE_CONFIG, "q")
        attrs = rec.named("chat ")[0].attributes
        assert attrs["gen_ai.usage.input_tokens"] == 10
        assert attrs["gen_ai.usage.output_tokens"] == 5
        assert attrs["gen_ai.usage.total_tokens"] == 15
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 0
        assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 0
        assert attrs["gen_ai.usage.prompt_tokens"] == 10
        assert attrs["gen_ai.usage.completion_tokens"] == 5

    @pytest.mark.asyncio
    async def test_input_tokens_pass_through_untouched_cache_included(self) -> None:
        # LangChain already includes cached tokens inside input_tokens, unlike Anthropic: the input
        # figure must pass through as reported, and the cache figures are for parity only. This is
        # the assertion that catches a fold applied in the wrong direction.
        from langchain_core.messages import AIMessage

        reply = AIMessage(
            content="hi",
            usage_metadata={
                "input_tokens": 23554,
                "output_tokens": 10,
                "total_tokens": 23564,
                "input_token_details": {"cache_read": 19971, "cache_creation": 3580},
            },
        )
        ctx, rec = _recording()
        llm = _FakeToolModel(replies=[reply])
        with ctx:
            await create_langchain_agents_handler(llm)(BASE_CONFIG, "q")
        attrs = rec.named("chat ")[0].attributes
        assert attrs["gen_ai.usage.input_tokens"] == 23554
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 19971
        assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 3580
        assert attrs["gen_ai.usage.total_tokens"] == 23564 + 10 - 10  # input + output

    @pytest.mark.asyncio
    async def test_reports_the_mapped_finish_reason(self) -> None:
        # LangChain does not normalise the field; the handler reads response_metadata.stop_reason
        # and maps Anthropic's `end_turn` onto semconv's `stop`.
        ctx, rec = _recording()
        llm = _FakeToolModel(
            replies=[_ai_message("hi", response_metadata={"stop_reason": "end_turn"})]
        )
        with ctx:
            await create_langchain_agents_handler(llm)(BASE_CONFIG, "q")
        assert rec.named("chat ")[0].attributes["gen_ai.response.finish_reasons"] == [
            "stop"
        ]

    @pytest.mark.asyncio
    async def test_maps_openai_finish_reason_too(self) -> None:
        # This handler can serve either vendor; both spellings must map onto one vocabulary.
        ctx, rec = _recording()
        llm = _FakeToolModel(
            replies=[_ai_message("hi", response_metadata={"finish_reason": "stop"})]
        )
        with ctx:
            await create_langchain_agents_handler(llm)(BASE_CONFIG, "q")
        assert rec.named("chat ")[0].attributes["gen_ai.response.finish_reasons"] == [
            "stop"
        ]

    @pytest.mark.asyncio
    async def test_omits_the_finish_reason_when_the_provider_gives_none(self) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(replies=[_ai_message("hi")])
        with ctx:
            await create_langchain_agents_handler(llm)(BASE_CONFIG, "q")
        assert "gen_ai.response.finish_reasons" not in rec.named("chat ")[0].attributes

    @pytest.mark.asyncio
    async def test_sets_status_ok_on_a_successful_turn(self) -> None:
        from opentelemetry.trace import StatusCode

        ctx, rec = _recording()
        llm = _FakeToolModel(replies=[_ai_message("hi")])
        with ctx:
            await create_langchain_agents_handler(llm)(BASE_CONFIG, "q")
        assert StatusCode.OK in rec.named("chat ")[0].statuses


class TestContentCapture:
    """TELEMETRY-CONTRACT.md section 7."""

    @pytest.mark.asyncio
    async def test_emits_no_content_at_all_by_default(self) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(replies=[_ai_message("hi")])
        with ctx:
            await create_langchain_agents_handler(llm)(BASE_CONFIG, "q")
        for span in rec.spans:
            content_keys = [
                k
                for k in span.attributes
                if k.startswith(("gen_ai.prompt", "gen_ai.completion"))
                or k
                in (
                    "gen_ai.input.messages",
                    "gen_ai.output.messages",
                    "gen_ai.system_instructions",
                    "gen_ai.tool.definitions",
                )
            ]
            assert content_keys == []
            assert [n for n, _ in span.events if n.startswith("gen_ai.content")] == []

    @pytest.mark.asyncio
    async def test_puts_prompt_and_completion_on_the_root_when_enabled(self) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(replies=[_ai_message("Hello World")])
        with ctx:
            await create_langchain_agents_handler(llm, capture_content=True)(
                BASE_CONFIG, "q"
            )
        root = rec.root
        assert root.attributes["gen_ai.system_instructions"]
        assert "gen_ai.input.messages" in root.attributes
        assert root.attributes["gen_ai.completion.0.content"] == "Hello World"
        assert "gen_ai.output.messages" in root.attributes

    @pytest.mark.asyncio
    async def test_records_the_tool_catalog_on_the_chat_span_when_enabled(self) -> None:
        import json

        ctx, rec = _recording()
        llm = _FakeToolModel(replies=[_ai_message("hi")])
        with ctx:
            await create_langchain_agents_handler(llm, capture_content=True)(
                TOOL_CONFIG, "q", {"search": AsyncMock(return_value="r")}
            )
        definitions = json.loads(
            rec.named("chat ")[0].attributes["gen_ai.tool.definitions"]
        )
        assert definitions[0]["name"] == "search"
        assert definitions[0]["type"] == "function"

    @pytest.mark.asyncio
    async def test_records_tool_arguments_and_results_when_enabled(self) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(
            replies=[
                _ai_message(
                    tool_calls=[
                        {"name": "search", "args": {"q": "weather"}, "id": "call_1"}
                    ]
                ),
                _ai_message("done"),
            ]
        )
        with ctx:
            await create_langchain_agents_handler(llm, capture_content=True)(
                TOOL_CONFIG, "q", {"search": AsyncMock(return_value="72F")}
            )
        tool = rec.named("execute_tool ")[0]
        assert "weather" in tool.attributes["gen_ai.tool.call.arguments"]
        assert tool.attributes["gen_ai.tool.call.result"] == "72F"


class TestErrorHandling:
    """TELEMETRY-CONTRACT.md section 6."""

    @pytest.mark.asyncio
    async def test_fails_the_chat_span_when_the_provider_call_raises(self) -> None:
        from opentelemetry.trace import StatusCode

        ctx, rec = _recording()
        llm = _FakeToolModel(
            replies=[], fail_after=0, fail_with=RuntimeError("model down")
        )
        with ctx, pytest.raises(RuntimeError, match="model down"):
            await create_langchain_agents_handler(llm)(BASE_CONFIG, "q")
        chat = rec.named("chat ")[0]
        assert len(chat.exceptions) == 1
        assert StatusCode.ERROR in chat.statuses
        assert chat.ended == 1

    @pytest.mark.asyncio
    async def test_fails_the_root_span_too(self) -> None:
        from opentelemetry.trace import StatusCode

        ctx, rec = _recording()
        llm = _FakeToolModel(
            replies=[], fail_after=0, fail_with=RuntimeError("model down")
        )
        with ctx, pytest.raises(RuntimeError):
            await create_langchain_agents_handler(llm)(BASE_CONFIG, "q")
        assert len(rec.root.exceptions) == 1
        assert StatusCode.ERROR in rec.root.statuses
        assert rec.root.ended == 1

    @pytest.mark.asyncio
    async def test_fails_the_execute_tool_span_when_a_tool_raises(self) -> None:
        from opentelemetry.trace import StatusCode

        async def _boom(_args: Any) -> str:
            raise RuntimeError("tool exploded")

        ctx, rec = _recording()
        llm = _FakeToolModel(
            replies=[
                _ai_message(
                    tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call_1"}]
                )
            ]
        )
        with ctx, pytest.raises(Exception, match="tool exploded"):
            await create_langchain_agents_handler(llm)(
                TOOL_CONFIG, "q", {"search": _boom}
            )
        tool = rec.named("execute_tool ")[0]
        assert len(tool.exceptions) == 1
        assert StatusCode.ERROR in tool.statuses
        assert tool.ended == 1

    @pytest.mark.asyncio
    async def test_reports_the_spend_of_completed_turns_on_a_failed_run(self) -> None:
        # The first turn was billed. The root is the only span a config-scoped cost query finds it
        # on. There is no `result` to sum on this path, so the total comes from the callbacks.
        ctx, rec = _recording()
        llm = _FakeToolModel(
            replies=[
                _ai_message(
                    input_tokens=40,
                    output_tokens=7,
                    tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call_1"}],
                )
            ],
            fail_after=1,
            fail_with=RuntimeError("second turn died"),
        )
        with ctx, pytest.raises(RuntimeError):
            await create_langchain_agents_handler(llm)(
                TOOL_CONFIG, "q", {"search": AsyncMock(return_value="r")}
            )
        assert rec.root.attributes["gen_ai.usage.input_tokens"] == 40
        assert rec.root.attributes["gen_ai.usage.output_tokens"] == 7

    @pytest.mark.asyncio
    async def test_writes_no_usage_when_no_turn_ever_reported_any(self) -> None:
        # All-zero attributes would assert the run cost nothing; an absent attribute says "unknown".
        ctx, rec = _recording()
        llm = _FakeToolModel(
            replies=[], fail_after=0, fail_with=RuntimeError("died first")
        )
        with ctx, pytest.raises(RuntimeError):
            await create_langchain_agents_handler(llm)(BASE_CONFIG, "q")
        assert "gen_ai.usage.input_tokens" not in rec.root.attributes

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
    """TELEMETRY-CONTRACT.md sections 1 and 6. The streaming path emits the same tree."""

    @pytest.mark.asyncio
    async def test_opens_the_same_root_span_name_as_the_blocking_path(self) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(replies=[_ai_message("hi")])
        with ctx:
            async for _ in await create_langchain_agents_handler(llm).stream(
                BASE_CONFIG, "q", {}, {}
            ):
                pass
        assert rec.root.name == "invoke_agent"
        assert "chat gpt-4o" in rec.names

    @pytest.mark.asyncio
    async def test_carries_the_launchdarkly_attributes_on_the_root(self) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(replies=[_ai_message("hi")])
        variables = {"__ld": {"configKey": "k", "variationKey": "v", "runId": "r"}}
        with ctx:
            async for _ in await create_langchain_agents_handler(llm).stream(
                BASE_CONFIG, "q", {}, variables
            ):
                pass
        attrs = rec.root.attributes
        assert attrs["launchdarkly.operation.type"] == "gen_ai"
        assert attrs["launchdarkly.config.key"] == "k"
        assert attrs["launchdarkly.variation.key"] == "v"
        assert attrs["launchdarkly.run.id"] == "r"

    @pytest.mark.asyncio
    async def test_ends_every_span_once_when_the_stream_completes(self) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(replies=[_ai_message("hi")])
        with ctx:
            async for _ in await create_langchain_agents_handler(llm).stream(
                BASE_CONFIG, "q", {}, {}
            ):
                pass
        assert [s.ended for s in rec.spans] == [1] * len(rec.spans)
        assert "launchdarkly.stream.abandoned" not in rec.root.attributes

    @pytest.mark.asyncio
    async def test_writes_the_run_total_to_the_root(self) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(
            replies=[_ai_message("hi", input_tokens=11, output_tokens=4)]
        )
        with ctx:
            async for _ in await create_langchain_agents_handler(llm).stream(
                BASE_CONFIG, "q", {}, {}
            ):
                pass
        assert rec.root.attributes["gen_ai.usage.input_tokens"] == 11
        assert rec.root.attributes["gen_ai.usage.total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_an_abandoned_stream_still_ends_and_exports_every_span(self) -> None:
        # A consumer that breaks out of `async for` makes this generator run `finally` without ever
        # entering `except`: GeneratorExit is a BaseException. Without the cleanup there the root is
        # never ended, so it is never exported.
        ctx, rec = _recording()
        llm = _FakeToolModel(replies=[_ai_message("hi")])
        with ctx:
            gen = await create_langchain_agents_handler(llm).stream(
                BASE_CONFIG, "q", {}, {}
            )
            async for _ in gen:
                break
            await gen.aclose()
        assert rec.root.ended == 1
        assert [s.ended for s in rec.spans] == [1] * len(rec.spans)

    @pytest.mark.asyncio
    async def test_an_abandoned_stream_is_marked_but_not_failed(self) -> None:
        from opentelemetry.trace import StatusCode

        ctx, rec = _recording()
        llm = _FakeToolModel(replies=[_ai_message("hi")])
        with ctx:
            gen = await create_langchain_agents_handler(llm).stream(
                BASE_CONFIG, "q", {}, {}
            )
            async for _ in gen:
                break
            await gen.aclose()
        assert rec.root.attributes["launchdarkly.stream.abandoned"] is True
        assert StatusCode.ERROR not in rec.root.statuses
        assert rec.root.exceptions == []

    @pytest.mark.asyncio
    async def test_fails_the_spans_when_the_stream_raises(self) -> None:
        from opentelemetry.trace import StatusCode

        ctx, rec = _recording()
        llm = _FakeToolModel(
            replies=[], fail_after=0, fail_with=RuntimeError("stream died")
        )
        with ctx, pytest.raises(RuntimeError, match="stream died"):
            async for _ in await create_langchain_agents_handler(llm).stream(
                BASE_CONFIG, "q", {}, {}
            ):
                pass
        assert StatusCode.ERROR in rec.root.statuses
        assert rec.root.ended == 1

    @pytest.mark.asyncio
    async def test_emits_no_content_by_default_on_the_streaming_path(self) -> None:
        ctx, rec = _recording()
        llm = _FakeToolModel(replies=[_ai_message("hi")])
        with ctx:
            async for _ in await create_langchain_agents_handler(llm).stream(
                BASE_CONFIG, "q", {}, {}
            ):
                pass
        for span in rec.spans:
            assert [k for k in span.attributes if k.startswith("gen_ai.prompt")] == []
            assert [n for n, _ in span.events if n.startswith("gen_ai.content")] == []


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


class TestAbandonOpenSpans:
    """An early consumer stop must not look like a provider failure.

    The abandonment path used to reuse `close_open_spans`, which records a synthetic exception and
    sets ERROR on every span still open. TELEMETRY-CONTRACT.md section 6 says an abandoned span stays
    UNSET and carries `launchdarkly.stream.abandoned`, and `openai-agents` already did that.

    Tested directly on the callback handler rather than through the streaming path. Reaching the
    state that matters, a chat or tool span still open at the break, needs a fake model that yields
    mid-turn, and with the fixtures here LangGraph has already run every callback by the time the
    first chunk reaches the consumer. A test driven through `stream` therefore passes whether or not
    the fix is present, which is worse than no test.
    """

    def _handler_with_open_spans(self) -> tuple[Any, Any, Any]:
        import launchdarkly_ai_langchain_agents.spans as spans_mod
        from launchdarkly_ai_langchain_agents.spans import build_span_callbacks

        recorder = SpanRecorder()
        with patch.object(spans_mod, "trace", recorder):
            bundle = build_span_callbacks(BASE_CONFIG, None, capture_content=False)
            handler = bundle._handler
            chat = recorder.start_span("chat gpt-4o")
            tool = recorder.start_span("execute_tool search")
        handler.model_spans["run-1"] = chat
        handler.tool_spans["run-2"] = tool
        return bundle, chat, tool

    def test_marks_open_spans_abandoned_and_leaves_them_unset(self) -> None:
        from opentelemetry.trace import StatusCode

        bundle, chat, tool = self._handler_with_open_spans()
        bundle.abandon_open_spans(set())
        for span in (chat, tool):
            assert span.ended == 1
            assert span.attributes["launchdarkly.stream.abandoned"] is True
            assert StatusCode.ERROR not in span.statuses
            assert span.exceptions == []

    def test_close_open_spans_still_fails_them_for_a_real_error(self) -> None:
        # The failure path keeps its behaviour; only abandonment changed.
        from opentelemetry.trace import StatusCode

        bundle, chat, tool = self._handler_with_open_spans()
        bundle.close_open_spans(RuntimeError("provider died"))
        for span in (chat, tool):
            assert span.ended == 1
            assert StatusCode.ERROR in span.statuses
            assert len(span.exceptions) == 1
            assert "launchdarkly.stream.abandoned" not in span.attributes

    def test_abandoning_twice_ends_each_span_once(self) -> None:
        bundle, chat, tool = self._handler_with_open_spans()
        ended: set[int] = set()
        bundle.abandon_open_spans(ended)
        bundle.abandon_open_spans(ended)
        assert chat.ended == 1
        assert tool.ended == 1


class TestConvenienceWrapperForwardsCaptureContent:
    """`capture_content` must reach the handler, not fall through into `config()`.

    `config()` takes no such argument, so leaving it in kwargs raised TypeError: a caller asking for
    content on spans got an exception instead. Five of the six wrappers had this.
    """

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        import launchdarkly_ai_langchain_agents.handler as handler_mod

        seen: dict[str, Any] = {}

        def _factory(*args: Any, capture_content: bool = False, **kw: Any) -> Any:
            seen["capture_content"] = capture_content
            return MagicMock()

        fake_config = MagicMock()
        fake_config.return_value.invoke = MagicMock(return_value="ok")
        with (
            patch.object(handler_mod, "create_langchain_agents_handler", _factory),
            patch.object(handler_mod, "config", fake_config),
        ):
            handler_mod.langchain_agents("k", "q", {}, **kwargs)
        seen["config_kwargs"] = fake_config.call_args.kwargs
        return seen

    def test_capture_content_reaches_the_factory(self) -> None:
        seen = self._run(capture_content=True)
        assert seen["capture_content"] is True
        # And it must not have been forwarded to config(), which does not accept it.
        assert "capture_content" not in seen["config_kwargs"]

    def test_defaults_to_off(self) -> None:
        assert self._run()["capture_content"] is False


class TestCallbackSpansNeverLeak:
    """A raise inside a callback must not leave a span both untracked and unended.

    The end callbacks pop the span before doing work that can raise, so after the pop nothing else
    can reach it and ending it is that callback's job alone. `on_tool_start` had the mirror problem:
    it created the span before inserting it, so a raise in between left a span no cleanup path knew
    about.
    """

    class _Unserialisable:
        __slots__ = ()

    def _recording_handler(self) -> Any:
        """The patch must stay active while the callbacks run, not only while they are built."""
        import launchdarkly_ai_langchain_agents.spans as spans_mod

        recorder = SpanRecorder()
        return patch.object(spans_mod, "trace", recorder), recorder

    @pytest.mark.asyncio
    async def test_a_raise_in_on_tool_end_still_ends_the_span(self) -> None:
        from opentelemetry.trace import StatusCode

        from launchdarkly_ai_langchain_agents.spans import build_span_callbacks

        ctx, recorder = self._recording_handler()
        with ctx:
            handler = build_span_callbacks(
                BASE_CONFIG, None, capture_content=True
            )._handler
            await handler.on_tool_start({"name": "search"}, "{}", run_id="r1")
            assert "r1" in handler.tool_spans
            with pytest.raises(TypeError):
                await handler.on_tool_end(self._Unserialisable(), run_id="r1")

        span = recorder.named("execute_tool ")[0]
        assert span.ended == 1, "the tool span leaked"
        assert StatusCode.ERROR in span.statuses

    @pytest.mark.asyncio
    async def test_on_tool_start_tracks_the_span_before_it_can_raise(self) -> None:
        from launchdarkly_ai_langchain_agents.spans import build_span_callbacks

        ctx, recorder = self._recording_handler()
        with ctx:
            bundle = build_span_callbacks(BASE_CONFIG, None, capture_content=True)
            handler = bundle._handler
            with pytest.raises(TypeError):
                await handler.on_tool_start(
                    {"name": "search"}, "{}", run_id="r1", inputs=self._Unserialisable()
                )
            # Tracked despite the raise, so the caller's cleanup can still close it.
            assert "r1" in handler.tool_spans
            bundle.abandon_open_spans(set())

        assert recorder.named("execute_tool ")[0].ended == 1
