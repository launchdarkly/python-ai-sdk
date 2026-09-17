"""
Tests for launchdarkly-ai-claude-agents handler.

Rewritten against TELEMETRY-CONTRACT.md, replacing the old flat-span assertions. Uses a real
``TracerProvider`` + ``InMemorySpanExporter`` rather than a mocked ``opentelemetry.trace`` module,
the same choice ``@launchdarkly/ai-claude-agents``'s ``spans.test.ts`` makes: a mocked tracer cannot
see whether parent/child wiring is right, only whether the right methods were called.

``query()`` is replaced with a fake async generator that replays a scripted message stream, built
from the Agent SDK's own dataclasses (``AssistantMessage``, ``UserMessage``, ``SystemMessage``,
``ResultMessage``, ``StreamEvent``) rather than mocks, so a structural change to those types would
fail loudly here instead of silently.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import launchdarkly_ai_claude_agents.handler as handler_mod
from launchdarkly_ai_claude_agents.handler import (
    _native_tool_aliases,
    build_prompt,
    create_claude_agents_handler,
    partition_tools,
)
from launchdarkly_ai_server import ConversationIdSpanProcessor, conversation_id

# ---------------------------------------------------------------------------
# A real tracer provider, reset between tests
# ---------------------------------------------------------------------------

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(ConversationIdSpanProcessor())
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)


@pytest.fixture(autouse=True)
def _reset_exporter() -> None:
    _exporter.clear()


def spans() -> list[Any]:
    return list(_exporter.get_finished_spans())


def named(prefix: str) -> list[Any]:
    return [s for s in spans() if s.name.startswith(prefix)]


def root() -> Any:
    return next(s for s in spans() if s.name == "invoke_agent")


# ---------------------------------------------------------------------------
# Message builders, using the Agent SDK's own dataclasses
# ---------------------------------------------------------------------------

BASE_CONFIG: dict[str, Any] = {
    "model": {"name": "claude-opus-4-5"},
    "provider": {"name": "Anthropic"},
    "instructions": "You are helpful.",
}


def _make_config(**kwargs: Any) -> dict[str, Any]:
    """Restored from the pre-rewrite file: a couple of the restored non-telemetry tests build a
    config inline rather than through ``BASE_CONFIG``/``TOOL_CONFIG``.
    """
    base = {"model": {"name": "claude-opus-4-5"}, "provider": {"name": "Anthropic"}}
    base.update(kwargs)
    return base


TOOL_CONFIG: dict[str, Any] = {
    **BASE_CONFIG,
    "tools": {
        "search": {
            "name": "search",
            "type": "function",
            "parameters": {"type": "object", "properties": {}},
        }
    },
}


def assistant_message(
    input_tokens: int = 10,
    output_tokens: int = 2,
    message_id: str | None = "msg_1",
    content: list[Any] | None = None,
    stop_reason: str | None = "end_turn",
    session_id: str = "sess-1",
    parent_tool_use_id: str | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        content=content or [],
        model="claude-opus-4-5",
        parent_tool_use_id=parent_tool_use_id,
        usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
        message_id=message_id,
        stop_reason=stop_reason,
        session_id=session_id,
    )


def result_message(
    result: str = "agent output",
    subtype: str = "success",
    input_tokens: int = 22,
    output_tokens: int = 5,
    errors: list[str] | None = None,
) -> ResultMessage:
    return ResultMessage(
        subtype=subtype,
        duration_ms=1,
        duration_api_ms=1,
        is_error=subtype != "success",
        num_turns=1,
        session_id="sess-1",
        usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
        result=result,
        errors=errors,
    )


def init_message(
    session_id: str = "sess-1", tools: list[str] | None = None
) -> SystemMessage:
    data: dict[str, Any] = {"session_id": session_id}
    if tools is not None:
        data["tools"] = tools
    return SystemMessage(subtype="init", data=data)


def tool_result_user_message(
    tool_use_id: str = "tu-1", content: str = "found it"
) -> UserMessage:
    return UserMessage(
        content=[ToolResultBlock(tool_use_id=tool_use_id, content=content)]
    )


def stream_event(text: str) -> StreamEvent:
    return StreamEvent(
        uuid="u1",
        session_id="sess-1",
        event={
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": text},
        },
    )


def _fake_query(messages: list[Any]):
    async def _query(**_kwargs: Any) -> AsyncIterator[Any]:
        for m in messages:
            yield m

    return _query


async def _collect(gen: Any) -> list[Any]:
    out = []
    async for event in gen:
        out.append(event)
    return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_returns_callable(self) -> None:
        assert callable(create_claude_agents_handler())

    def test_provides_for(self) -> None:
        assert create_claude_agents_handler().provides_for == ("Anthropic", "agent")

    def test_multiple_calls_independent(self) -> None:
        assert create_claude_agents_handler() is not create_claude_agents_handler()

    # --- restored from the pre-rewrite file (not telemetry) ---

    def test_attaches_provides_for(self) -> None:
        h = create_claude_agents_handler()
        assert hasattr(h, "provides_for")

    def test_provides_for_values_are_correct(self) -> None:
        h = create_claude_agents_handler()
        pf = h.provides_for
        assert "Anthropic" in pf or "anthropic" in str(pf).lower()

    def test_multiple_calls_return_independent_instances(self) -> None:
        h1 = create_claude_agents_handler()
        h2 = create_claude_agents_handler()
        assert h1 is not h2


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    def test_instructions_become_system_prompt(self) -> None:
        prompt, system = build_prompt(BASE_CONFIG, "hi", {})
        assert prompt == "hi"
        assert system == "You are helpful."

    def test_no_instructions_no_system_prompt(self) -> None:
        prompt, system = build_prompt({"model": {"name": "m"}}, "hi", {})
        assert system is None
        assert prompt == "hi"

    def test_variable_substitution(self) -> None:
        cfg = {**BASE_CONFIG, "instructions": "Hello {{name}}."}
        _, system = build_prompt(cfg, "hi", {"name": "Ada"})
        assert system == "Hello Ada."

    # --- restored from the pre-rewrite file (not telemetry); byte-for-byte, only
    # ``_make_config`` inlined since the old module-level helper was removed ---

    def test_path_a_instructions(self) -> None:
        config = _make_config(instructions="You are a helper.")
        prompt, system = build_prompt(config, "hi", {})
        assert prompt == "hi"
        assert system == "You are a helper."

    def test_path_a_variable_substitution(self) -> None:
        config = _make_config(instructions="Hello {{name}}!")
        _, system = build_prompt(config, "hi", {"name": "Alice"})
        assert system == "Hello Alice!"

    def test_path_a_unresolved_placeholder_preserved(self) -> None:
        config = _make_config(instructions="Hello {{name}}!")
        _, system = build_prompt(config, "hi", {})
        assert "{{name}}" in (system or "")

    def test_path_b_messages_system_extracted(self) -> None:
        config = _make_config(
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hey"},
            ]
        )
        prompt, system = build_prompt(config, "question", {})
        assert system == "Be concise."
        assert "hey" in prompt
        assert "question" in prompt

    def test_path_b_variable_substitution_in_messages(self) -> None:
        config = _make_config(
            messages=[
                {"role": "user", "content": "my name is {{name}}"},
            ]
        )
        prompt, _ = build_prompt(config, "q", {"name": "Alice"})
        assert "Alice" in prompt

    def test_path_b_user_input_appended_as_final_turn(self) -> None:
        config = _make_config(
            messages=[
                {"role": "user", "content": "old message"},
            ]
        )
        prompt, _ = build_prompt(config, "new input", {})
        assert "new input" in prompt

    def test_path_c_empty_user_input_no_throw(self) -> None:
        config = _make_config(instructions="be helpful")
        prompt, system = build_prompt(config, "", {})
        assert prompt == ""
        assert system is not None

    def test_path_c_instructions_takes_priority_over_messages(self) -> None:
        config = _make_config(
            instructions="Use instructions.",
            messages=[{"role": "system", "content": "Use messages."}],
        )
        _prompt, system = build_prompt(config, "q", {})
        assert system == "Use instructions."

    def test_messages_mode_extracts_system_and_flattens_history_into_prompt(
        self,
    ) -> None:
        cfg = {
            "model": {"name": "m"},
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "context line"},
            ],
        }
        prompt, system = build_prompt(cfg, "final question", {})
        assert system == "Be terse."
        assert "context line" in prompt
        assert prompt.endswith("final question")

    def test_history_appended_to_system_prompt(self) -> None:
        history = [{"role": "user", "content": "earlier"}]
        _, system = build_prompt(BASE_CONFIG, "hi", {}, history=history)
        assert "earlier" in (system or "")
        assert "You are helpful." in (system or "")

    def test_no_user_input_defaults_to_empty_string(self) -> None:
        prompt, _ = build_prompt(BASE_CONFIG, None, {})
        assert prompt == ""


# ---------------------------------------------------------------------------
# partition_tools / native tool aliases
# ---------------------------------------------------------------------------


class TestPartitionTools:
    def test_user_tool_goes_to_mcp_bucket(self) -> None:
        native_map, user_tools, native_names = partition_tools(
            TOOL_CONFIG["tools"], {"search": lambda _: "r"}
        )
        assert native_map == {}
        assert "search" in user_tools
        assert native_names == []

    def test_native_tool_goes_to_native_bucket(self) -> None:
        from launchdarkly_ai_claude_agents.builtins import ClaudeWebSearch
        from launchdarkly_ai_server import NATIVE_TOOL_KEY

        # A tracking stub, as `wrap_tool_handlers` produces: a callable with the NativeTool
        # stashed under NATIVE_TOOL_KEY, not the NativeTool instance itself.
        stub = lambda: None  # noqa: E731
        setattr(stub, NATIVE_TOOL_KEY, ClaudeWebSearch)
        _native_map, user_tools, native_names = partition_tools(
            {"webSearch": {"name": "webSearch"}}, {"webSearch": stub}
        )
        assert native_names == ["WebSearch"]
        assert user_tools == {}

    def test_native_tool_aliases_map_ld_key_to_provider_name(self) -> None:
        from launchdarkly_ai_claude_agents.builtins import ClaudeWebSearch
        from launchdarkly_ai_server import NATIVE_TOOL_KEY

        stub = lambda: None  # noqa: E731
        setattr(stub, NATIVE_TOOL_KEY, ClaudeWebSearch)
        aliases = _native_tool_aliases({"webSearch": stub})
        assert aliases == {"webSearch": "WebSearch"}


# ---------------------------------------------------------------------------
# §1.3 Tool conversion — restored from the pre-rewrite file (not telemetry).
# ``partition_tools`` kept its 3-tuple return, so these run byte-for-byte.
# ---------------------------------------------------------------------------


class TestToolConversion:
    def test_all_fields_forwarded(self) -> None:
        config_tools = {
            "my-tool": {"description": "does stuff", "parameters": {"type": "object"}}
        }
        handlers = {"my-tool": AsyncMock(return_value="ok")}
        _, user_tools, _ = partition_tools(config_tools, handlers)
        assert "my-tool" in user_tools

    def test_multiple_tools_all_included(self) -> None:
        config_tools = {"tool-a": {}, "tool-b": {}}
        handlers = {"tool-a": AsyncMock(), "tool-b": AsyncMock()}
        _, user_tools, _ = partition_tools(config_tools, handlers)
        assert "tool-a" in user_tools
        assert "tool-b" in user_tools

    def test_empty_tools_no_tools_sent(self) -> None:
        _, user_tools, native_names = partition_tools({}, {})
        assert not user_tools
        assert not native_names


# ---------------------------------------------------------------------------
# §1.4 Tool execution loop (via build_tool_mcp) — restored from the pre-rewrite
# file. ``build_tool_mcp`` kept its lazy ``importlib.import_module`` pattern
# (native_graph.py depends on it), so the old SDK-mocking approach still works
# unmodified for this one.
# ---------------------------------------------------------------------------


class _MockResultMessageForToolMcp:
    """Distinct class so isinstance checks the handler makes still work."""

    def __init__(
        self, text: str = "hello", input_tokens: int = 10, output_tokens: int = 5
    ) -> None:
        self.result = text
        self.usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        self.is_error = False


def _patch_query_for_tool_mcp(messages: list[Any]) -> Any:
    """Patches ``claude_agent_sdk`` for the one restored test that exercises
    ``build_tool_mcp`` directly, which still resolves the SDK lazily.
    """

    async def _query(**kwargs: Any) -> AsyncIterator[Any]:
        for m in messages:
            yield m

    mock_sdk = MagicMock()
    mock_sdk.query = _query
    mock_sdk.ClaudeAgentOptions = MagicMock(return_value=MagicMock())
    mock_sdk.ResultMessage = _MockResultMessageForToolMcp
    mock_sdk.tool = MagicMock(return_value=lambda fn: fn)
    mock_sdk.create_sdk_mcp_server = MagicMock(return_value=MagicMock())
    mock_sdk.HookMatcher = MagicMock()
    return patch(
        "importlib.import_module",
        side_effect=lambda n: mock_sdk if n == "claude_agent_sdk" else __import__(n),
    )


class TestToolExecutionLoop:
    async def test_tool_not_found_throws(self) -> None:
        from launchdarkly_ai_claude_agents.handler import build_tool_mcp

        config_tools = {"my-tool": {"description": "d", "parameters": {}}}
        handlers: dict[str, Any] = {}  # no handler registered

        # The execute closure inside build_tool_mcp raises if handler missing
        with _patch_query_for_tool_mcp([_MockResultMessageForToolMcp()]):
            mcp = await build_tool_mcp(config_tools, handlers)
            # Directly call the stored execute fn
            for t in getattr(mcp, "tools", None) or []:
                fn = getattr(t, "_fn", getattr(t, "fn", None))
                if fn:
                    with pytest.raises(ValueError, match="No handler"):
                        await fn({"key": "val"})

    async def test_no_tools_in_config_handler_never_invoked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_handler = AsyncMock(return_value="tool-output")
        monkeypatch.setattr(handler_mod, "query", _fake_query([result_message("done")]))
        h = create_claude_agents_handler()
        output = await h(BASE_CONFIG, "hi", {"my-tool": mock_handler})

        mock_handler.assert_not_called()
        assert output["output"] == "done"


# ---------------------------------------------------------------------------
# Span tree — TELEMETRY-CONTRACT.md section 1
# ---------------------------------------------------------------------------


class TestSpanTree:
    async def test_root_span_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            handler_mod, "query", _fake_query([assistant_message(), result_message()])
        )
        await create_claude_agents_handler()(BASE_CONFIG, "q")
        assert root().name == "invoke_agent"
        assert root().attributes["gen_ai.operation.name"] == "invoke_agent"

    async def test_one_chat_span_per_model_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod,
            "query",
            _fake_query(
                [
                    assistant_message(message_id="req_1"),
                    assistant_message(message_id="req_2"),
                    result_message(),
                ]
            ),
        )
        await create_claude_agents_handler()(BASE_CONFIG, "q")
        chats = named("chat ")
        assert len(chats) == 2
        assert chats[0].name == "chat claude-opus-4-5"

    async def test_one_chat_span_per_call_not_per_message_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The CLI splits one API response into several assistant messages sharing one message id.
        monkeypatch.setattr(
            handler_mod,
            "query",
            _fake_query(
                [
                    assistant_message(
                        23669, 8, "req_shared", content=[TextBlock(text="a")]
                    ),
                    assistant_message(
                        23669, 8, "req_shared", content=[TextBlock(text="b")]
                    ),
                    result_message(),
                ]
            ),
        )
        await create_claude_agents_handler(capture_content=True)(BASE_CONFIG, "q")
        chats = named("chat ")
        assert len(chats) == 1
        assert chats[0].attributes["gen_ai.usage.input_tokens"] == 23669

    async def test_execute_tool_span_per_tool_call_sibling_of_chat(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        async def _query(**kwargs: Any) -> AsyncIterator[Any]:
            captured["options"] = kwargs["options"]
            yield assistant_message(
                content=[
                    ToolUseBlock(id="tu-1", name="mcp__tool-mcp__search", input={})
                ]
            )
            hooks = kwargs["options"].hooks
            await hooks["PreToolUse"][0].hooks[0](
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "mcp__tool-mcp__search",
                    "tool_use_id": "tu-1",
                    "tool_input": {"q": "x"},
                    "session_id": "sess-1",
                },
                "tu-1",
                None,
            )
            await hooks["PostToolUse"][0].hooks[0](
                {
                    "hook_event_name": "PostToolUse",
                    "tool_name": "mcp__tool-mcp__search",
                    "tool_use_id": "tu-1",
                    "tool_response": "found",
                },
                "tu-1",
                None,
            )
            yield assistant_message(message_id="req_2")
            yield result_message()

        monkeypatch.setattr(handler_mod, "query", _query)
        await create_claude_agents_handler()(
            TOOL_CONFIG, "q", {"search": lambda _: "r"}
        )

        tools = named("execute_tool ")
        assert len(tools) == 1
        assert tools[0].name == "execute_tool search"
        assert tools[0].attributes["gen_ai.tool.name"] == "search"
        assert tools[0].attributes["gen_ai.tool.call.id"] == "tu-1"
        # A sibling of chat: same parent (the root), not nested under a chat span.
        assert tools[0].parent.span_id == root().context.span_id

    async def test_children_carry_no_launchdarkly_attributes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod, "query", _fake_query([assistant_message(), result_message()])
        )
        await create_claude_agents_handler()(
            BASE_CONFIG,
            "q",
            variables={"__ld": {"configKey": "k", "variationKey": "v", "runId": "r"}},
        )
        for child in named("chat "):
            assert not [k for k in child.attributes if k.startswith("launchdarkly.")]
            assert "feature_flag" not in [e.name for e in child.events]

    async def test_every_span_is_ended_ok(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod, "query", _fake_query([assistant_message(), result_message()])
        )
        await create_claude_agents_handler()(BASE_CONFIG, "q")
        for s in spans():
            assert s.status.status_code == StatusCode.OK


# ---------------------------------------------------------------------------
# Root span attributes — TELEMETRY-CONTRACT.md sections 2, 2a, 8
# ---------------------------------------------------------------------------


class TestRootAttributes:
    async def test_provider_keys_and_requested_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod, "query", _fake_query([assistant_message(), result_message()])
        )
        await create_claude_agents_handler()(BASE_CONFIG, "q")
        attrs = root().attributes
        assert attrs["gen_ai.system"] == "anthropic"
        assert attrs["gen_ai.provider.name"] == "anthropic"
        assert attrs["gen_ai.request.model"] == "claude-opus-4-5"
        # The root reports the requested name even though the chat span (below) may differ.
        assert attrs["gen_ai.response.model"] == "claude-opus-4-5"

    async def test_run_total_not_one_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod, "query", _fake_query([assistant_message(), result_message()])
        )
        await create_claude_agents_handler()(BASE_CONFIG, "q")
        # The result message's cumulative usage, not the per-turn figure.
        assert root().attributes["gen_ai.usage.input_tokens"] == 22
        assert root().attributes["gen_ai.usage.output_tokens"] == 5

    async def test_conversation_id_from_init_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod,
            "query",
            _fake_query(
                [init_message("sess-abc"), assistant_message(), result_message()]
            ),
        )
        await create_claude_agents_handler()(BASE_CONFIG, "q")
        assert root().attributes["gen_ai.conversation.id"] == "sess-abc"

    async def test_conversation_id_only_on_root_chat_and_execute_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _query(**kwargs: Any) -> AsyncIterator[Any]:
            yield init_message("sess-abc")
            yield assistant_message(session_id="sess-abc")
            hooks = kwargs["options"].hooks
            await hooks["PreToolUse"][0].hooks[0](
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "mcp__tool-mcp__search",
                    "tool_use_id": "tu-1",
                    "tool_input": {},
                    "session_id": "sess-abc",
                },
                "tu-1",
                None,
            )
            await hooks["PostToolUse"][0].hooks[0](
                {
                    "hook_event_name": "PostToolUse",
                    "tool_name": "mcp__tool-mcp__search",
                    "tool_use_id": "tu-1",
                    "tool_response": "r",
                },
                "tu-1",
                None,
            )
            yield result_message("done")

        monkeypatch.setattr(handler_mod, "query", _query)
        await create_claude_agents_handler()(
            TOOL_CONFIG, "q", {"search": lambda _: "r"}
        )
        assert root().attributes["gen_ai.conversation.id"] == "sess-abc"
        assert named("chat ")[0].attributes["gen_ai.conversation.id"] == "sess-abc"
        assert (
            named("execute_tool ")[0].attributes["gen_ai.conversation.id"] == "sess-abc"
        )

    async def test_no_conversation_id_without_init(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(handler_mod, "query", _fake_query([result_message()]))
        await create_claude_agents_handler()(BASE_CONFIG, "q")
        assert "gen_ai.conversation.id" not in root().attributes

    async def test_caller_conversation_id_wins_over_session_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _query(**kwargs: Any) -> AsyncIterator[Any]:
            yield init_message("sess-abc")
            yield assistant_message(session_id="sess-abc")
            hooks = kwargs["options"].hooks
            await hooks["PreToolUse"][0].hooks[0](
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "mcp__tool-mcp__search",
                    "tool_use_id": "tu-1",
                    "tool_input": {},
                    "session_id": "sess-abc",
                },
                "tu-1",
                None,
            )
            await hooks["PostToolUse"][0].hooks[0](
                {
                    "hook_event_name": "PostToolUse",
                    "tool_name": "mcp__tool-mcp__search",
                    "tool_use_id": "tu-1",
                    "tool_response": "r",
                },
                "tu-1",
                None,
            )
            yield result_message()

        monkeypatch.setattr(handler_mod, "query", _query)
        with conversation_id("thread-stable"):
            await create_claude_agents_handler()(
                TOOL_CONFIG, "q", {"search": lambda _: "r"}
            )
        assert root().attributes["gen_ai.conversation.id"] == "thread-stable"
        assert named("chat ")[0].attributes["gen_ai.conversation.id"] == "thread-stable"
        assert (
            named("execute_tool ")[0].attributes["gen_ai.conversation.id"]
            == "thread-stable"
        )


# ---------------------------------------------------------------------------
# Chat span attributes — sections 2a, 3, 5b, 8
# ---------------------------------------------------------------------------


class TestChatAttributes:
    async def test_response_model_is_what_the_turn_actually_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # claude-agents is one of only two handlers whose chat span may disagree with the root.
        turn = assistant_message()
        turn.model = "claude-opus-4-5-20250101"
        monkeypatch.setattr(handler_mod, "query", _fake_query([turn, result_message()]))
        await create_claude_agents_handler()(BASE_CONFIG, "q")
        [chat] = named("chat ")
        assert chat.attributes["gen_ai.response.model"] == "claude-opus-4-5-20250101"
        assert root().attributes["gen_ai.response.model"] == "claude-opus-4-5"

    async def test_finish_reason_usually_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Measured against Agent SDK 0.3.220: stop_reason is null on every assistant message.
        turn = assistant_message(stop_reason=None)
        monkeypatch.setattr(handler_mod, "query", _fake_query([turn, result_message()]))
        await create_claude_agents_handler()(BASE_CONFIG, "q")
        [chat] = named("chat ")
        assert "gen_ai.response.finish_reasons" not in chat.attributes

    async def test_finish_reason_mapped_when_the_sdk_reports_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        turn = assistant_message(stop_reason="end_turn")
        monkeypatch.setattr(handler_mod, "query", _fake_query([turn, result_message()]))
        await create_claude_agents_handler()(BASE_CONFIG, "q")
        [chat] = named("chat ")
        assert list(chat.attributes["gen_ai.response.finish_reasons"]) == ["stop"]

    async def test_writes_all_seven_usage_attributes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod, "query", _fake_query([assistant_message(), result_message()])
        )
        await create_claude_agents_handler()(BASE_CONFIG, "q")
        [chat] = named("chat ")
        for key in (
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
            "gen_ai.usage.total_tokens",
            "gen_ai.usage.cache_read.input_tokens",
            "gen_ai.usage.cache_creation.input_tokens",
            "gen_ai.usage.prompt_tokens",
            "gen_ai.usage.completion_tokens",
        ):
            assert key in chat.attributes

    async def test_cache_tokens_folded_into_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # TELEMETRY-CONTRACT.md section 8: Anthropic reports cache buckets *beside* input_tokens,
        # so this handler must add them in, not pass input through untouched.
        turn = assistant_message(3, 8, "req_1")
        turn.usage = {
            "input_tokens": 3,
            "output_tokens": 8,
            "cache_read_input_tokens": 19971,
            "cache_creation_input_tokens": 3580,
        }
        monkeypatch.setattr(handler_mod, "query", _fake_query([turn, result_message()]))
        await create_claude_agents_handler()(BASE_CONFIG, "q")
        [chat] = named("chat ")
        assert chat.attributes["gen_ai.usage.input_tokens"] == 3 + 19971 + 3580
        assert chat.attributes["gen_ai.usage.cache_read.input_tokens"] == 19971
        assert chat.attributes["gen_ai.usage.cache_creation.input_tokens"] == 3580


# ---------------------------------------------------------------------------
# Content capture — section 7
# ---------------------------------------------------------------------------


class TestContentCapture:
    async def test_emits_no_content_at_all_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod,
            "query",
            _fake_query(
                [init_message(tools=["Read"]), assistant_message(), result_message()]
            ),
        )
        await create_claude_agents_handler()(TOOL_CONFIG, "q")
        for span in (root(), *named("chat ")):
            assert "gen_ai.input.messages" not in span.attributes
            assert "gen_ai.output.messages" not in span.attributes
            assert "gen_ai.system_instructions" not in span.attributes
            assert "gen_ai.tool.definitions" not in span.attributes

    async def test_root_carries_no_content_by_default_either(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod,
            "query",
            _fake_query([assistant_message(), result_message("hi")]),
        )
        await create_claude_agents_handler()(BASE_CONFIG, "q")
        assert "gen_ai.output.messages" not in root().attributes

    async def test_tool_call_arguments_and_result_gated_by_capture(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _query(**kwargs: Any) -> AsyncIterator[Any]:
            yield assistant_message(
                content=[
                    ToolUseBlock(
                        id="tu-1", name="mcp__tool-mcp__search", input={"q": "x"}
                    )
                ]
            )
            hooks = kwargs["options"].hooks
            await hooks["PreToolUse"][0].hooks[0](
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "mcp__tool-mcp__search",
                    "tool_use_id": "tu-1",
                    "tool_input": {"q": "x"},
                },
                "tu-1",
                None,
            )
            await hooks["PostToolUse"][0].hooks[0](
                {
                    "hook_event_name": "PostToolUse",
                    "tool_name": "mcp__tool-mcp__search",
                    "tool_use_id": "tu-1",
                    "tool_response": "found it",
                },
                "tu-1",
                None,
            )
            yield result_message("done")

        monkeypatch.setattr(handler_mod, "query", _query)
        await create_claude_agents_handler()(
            TOOL_CONFIG, "q", {"search": lambda _: "r"}
        )
        [tool_span] = named("execute_tool ")
        assert "gen_ai.tool.call.arguments" not in tool_span.attributes
        assert "gen_ai.tool.call.result" not in tool_span.attributes

    async def test_content_present_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod,
            "query",
            _fake_query(
                [
                    assistant_message(content=[TextBlock(text="hi")]),
                    result_message("hi"),
                ]
            ),
        )
        await create_claude_agents_handler(capture_content=True)(BASE_CONFIG, "q")
        [chat] = named("chat ")
        assert "gen_ai.input.messages" in chat.attributes
        assert "gen_ai.output.messages" in chat.attributes
        assert "gen_ai.system_instructions" in root().attributes


# ---------------------------------------------------------------------------
# Errors, and the failure path — section 6
# ---------------------------------------------------------------------------


class TestErrorHandling:
    async def test_error_result_fails_root_but_keeps_spend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod,
            "query",
            _fake_query(
                [
                    assistant_message(),
                    result_message(
                        subtype="error_max_turns",
                        input_tokens=40000,
                        output_tokens=500,
                        errors=["turn limit reached"],
                    ),
                ]
            ),
        )
        with pytest.raises(RuntimeError, match="error_max_turns"):
            await create_claude_agents_handler()(BASE_CONFIG, "q")
        attrs = root().attributes
        assert root().status.status_code == StatusCode.ERROR
        assert attrs["gen_ai.usage.input_tokens"] == 40000
        assert attrs["gen_ai.usage.output_tokens"] == 500

    async def test_reports_responses_that_arrived_when_the_sdk_throws(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _query(**_kwargs: Any) -> AsyncIterator[Any]:
            yield assistant_message(100, 20, "req_1")
            yield assistant_message(200, 30, "req_2")
            raise RuntimeError("transport died")

        monkeypatch.setattr(handler_mod, "query", _query)
        with pytest.raises(RuntimeError, match="transport died"):
            await create_claude_agents_handler()(BASE_CONFIG, "q")
        attrs = root().attributes
        assert attrs["gen_ai.usage.input_tokens"] == 300
        assert attrs["gen_ai.usage.output_tokens"] == 50
        assert root().status.status_code == StatusCode.ERROR

    async def test_no_usage_written_when_nothing_ever_arrived(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _query(**_kwargs: Any) -> AsyncIterator[Any]:
            raise RuntimeError("spawn failed")
            yield  # pragma: no cover - keeps this an async generator

        monkeypatch.setattr(handler_mod, "query", _query)
        with pytest.raises(RuntimeError, match="spawn failed"):
            await create_claude_agents_handler()(BASE_CONFIG, "q")
        assert "gen_ai.usage.input_tokens" not in root().attributes

    async def test_result_omitted_stream_reports_per_response_sum(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod,
            "query",
            _fake_query(
                [
                    assistant_message(100, 20, "req_1"),
                    assistant_message(200, 30, "req_2"),
                ]
            ),
        )
        result = await create_claude_agents_handler()(BASE_CONFIG, "q")
        assert root().attributes["gen_ai.usage.input_tokens"] == 300
        assert result["usage"]["input_tokens"] == 300


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    async def test_yields_chunks_then_one_done_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod,
            "query",
            _fake_query(
                [stream_event("Hi"), assistant_message(), result_message("Hi")]
            ),
        )
        h = create_claude_agents_handler()
        events = await _collect(await h.stream(BASE_CONFIG, "q", {}, {}))
        assert [e["type"] for e in events] == ["chunk", "done"]
        assert events[0]["text"] == "Hi"
        assert events[-1]["output"] == "Hi"

    async def test_emits_same_span_tree_as_blocking_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod,
            "query",
            _fake_query(
                [stream_event("Hi"), assistant_message(), result_message("Hi")]
            ),
        )
        h = create_claude_agents_handler()
        await _collect(await h.stream(BASE_CONFIG, "q", {}, {}))
        assert sorted(s.name for s in spans()) == [
            "chat claude-opus-4-5",
            "invoke_agent",
        ]
        assert root().status.status_code == StatusCode.OK

    async def test_error_fails_spans_and_reraises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _query(**_kwargs: Any) -> AsyncIterator[Any]:
            yield assistant_message()
            raise RuntimeError("boom")

        monkeypatch.setattr(handler_mod, "query", _query)
        h = create_claude_agents_handler()
        with pytest.raises(RuntimeError, match="boom"):
            await _collect(await h.stream(BASE_CONFIG, "q", {}, {}))
        assert root().status.status_code == StatusCode.ERROR
        assert root().attributes["gen_ai.usage.input_tokens"] == 10

    async def test_abandoned_stream_ends_every_span_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _slow_query(**_kwargs: Any) -> AsyncIterator[Any]:
            yield stream_event("chunk-1")
            yield assistant_message(message_id="req_1")
            yield stream_event("chunk-2")
            yield assistant_message(message_id="req_2")
            yield result_message("done")

        monkeypatch.setattr(handler_mod, "query", _slow_query)
        h = create_claude_agents_handler()
        gen = await h.stream(BASE_CONFIG, "q", {}, {})
        # Consume only the first chunk, then abandon the generator without exhausting it.
        first = await gen.__anext__()
        assert first["type"] == "chunk"
        await gen.aclose()

        assert root().status.status_code == StatusCode.UNSET
        assert root().attributes.get("launchdarkly.stream.abandoned") is True
        for s in spans():
            assert s.end_time is not None

    # --- restored from the pre-rewrite file (not telemetry). Adapted from the old
    # ``_patch_query``/``_HAS_OTEL`` mocking approach to ``monkeypatch.setattr(handler_mod,
    # "query", ...)`` plus real SDK dataclasses, because ``query`` is now a top-level import
    # rather than something resolved through ``importlib.import_module`` on every call, and
    # ``handler_mod`` no longer has its own ``_HAS_OTEL`` (that flag now lives in ``spans.py``). ---

    async def test_stream_is_defined(self) -> None:
        h = create_claude_agents_handler()
        assert hasattr(h, "stream")

    async def test_stream_returns_async_generator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import inspect

        monkeypatch.setattr(handler_mod, "query", _fake_query([result_message("done")]))
        h = create_claude_agents_handler()
        gen = await h.stream(BASE_CONFIG, "hi")
        assert inspect.isasyncgen(gen) or hasattr(gen, "__aiter__")

    async def test_yields_chunk_events_for_text_deltas(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod,
            "query",
            _fake_query(
                [
                    stream_event("hello "),
                    stream_event("world"),
                    result_message("hello world"),
                ]
            ),
        )
        h = create_claude_agents_handler()
        events = [e async for e in await h.stream(BASE_CONFIG, "hi")]

        chunks = [e for e in events if e.get("type") == "chunk"]
        assert len(chunks) == 2
        assert chunks[0]["text"] == "hello "

    async def test_all_chunks_before_done(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod,
            "query",
            _fake_query([stream_event("part"), result_message("part")]),
        )
        h = create_claude_agents_handler()
        events = [e async for e in await h.stream(BASE_CONFIG, "hi")]

        done_idx = next(i for i, e in enumerate(events) if e.get("type") == "done")
        chunk_indices = [i for i, e in enumerate(events) if e.get("type") == "chunk"]
        assert all(ci < done_idx for ci in chunk_indices)

    async def test_yields_exactly_one_done_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(handler_mod, "query", _fake_query([result_message("done")]))
        h = create_claude_agents_handler()
        events = [e async for e in await h.stream(BASE_CONFIG, "hi")]

        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1

    async def test_done_event_carries_correct_usage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod,
            "query",
            _fake_query([result_message("out", input_tokens=20, output_tokens=8)]),
        )
        h = create_claude_agents_handler()
        events = [e async for e in await h.stream(BASE_CONFIG, "hi")]

        done = next(e for e in events if e.get("type") == "done")
        usage = done["usage"]
        assert usage.get("input_tokens") == 20 or usage.get("input") == 20

    async def test_done_event_carries_accumulated_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod, "query", _fake_query([result_message("hello world")])
        )
        h = create_claude_agents_handler()
        events = [e async for e in await h.stream(BASE_CONFIG, "hi")]

        done = next(e for e in events if e.get("type") == "done")
        assert done["output"] == "hello world"

    async def test_generator_throws_on_provider_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _broken_query(**_kwargs: Any) -> AsyncIterator[Any]:
            raise RuntimeError("stream fail")
            yield  # pragma: no cover - keeps this an async generator

        monkeypatch.setattr(handler_mod, "query", _broken_query)
        h = create_claude_agents_handler()
        with pytest.raises(RuntimeError, match="stream fail"):
            async for _ in await h.stream(BASE_CONFIG, "hi"):
                pass


class TestQueryGeneratorLifecycle:
    """TELEMETRY-CONTRACT.md section 6: the vendor generator, not just the span, must be closed."""

    async def test_query_generator_closed_on_early_return(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        closed = {"value": False}

        async def _query(**_kwargs: Any) -> AsyncIterator[Any]:
            try:
                yield result_message("done")
                yield assistant_message()  # never reached: the handler returns after the result
            finally:
                closed["value"] = True

        monkeypatch.setattr(handler_mod, "query", _query)
        await create_claude_agents_handler()(BASE_CONFIG, "q")
        assert closed["value"] is True

    async def test_streaming_query_generator_closed_on_abandonment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The streaming path's counterpart: iterating query(...) inline with no held reference
        # leaks the vendor generator when the consumer abandons ours. See TELEMETRY-CONTRACT.md
        # section 6, "claude-agents".
        closed = {"value": False}

        async def _query(**_kwargs: Any) -> AsyncIterator[Any]:
            try:
                yield stream_event("chunk-1")
                yield assistant_message()
                yield result_message("done")
            finally:
                closed["value"] = True

        monkeypatch.setattr(handler_mod, "query", _query)
        h = create_claude_agents_handler()
        gen = await h.stream(BASE_CONFIG, "q", {}, {})
        first = await gen.__anext__()
        assert first["type"] == "chunk"
        await gen.aclose()

        assert closed["value"] is True


# ---------------------------------------------------------------------------
# Tool catalog widening — the CLI's own tools, announced only at `init`
# ---------------------------------------------------------------------------


class TestToolCatalog:
    async def test_widens_catalog_with_native_tools_on_root_and_chat(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod,
            "query",
            _fake_query(
                [
                    init_message(
                        "sess-1", tools=["Read", "Bash", "mcp__tool-mcp__search"]
                    ),
                    assistant_message(content=[TextBlock(text="hi")]),
                    result_message("hi"),
                ]
            ),
        )
        await create_claude_agents_handler(capture_content=True)(TOOL_CONFIG, "q")
        import json as _json

        on_root = _json.loads(root().attributes["gen_ai.tool.definitions"])
        [chat] = named("chat ")
        on_chat = _json.loads(chat.attributes["gen_ai.tool.definitions"])
        assert on_root == on_chat
        assert [t["name"] for t in on_root] == ["search", "Read", "Bash"]

    async def test_no_widening_without_init_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod,
            "query",
            _fake_query([assistant_message(), result_message()]),
        )
        await create_claude_agents_handler(capture_content=True)(TOOL_CONFIG, "q")
        import json as _json

        on_root = _json.loads(root().attributes["gen_ai.tool.definitions"])
        assert [t["name"] for t in on_root] == ["search"]


# ---------------------------------------------------------------------------
# Subagent conversations — a subagent's own calls share the main stream
# ---------------------------------------------------------------------------


class TestSubagentThreads:
    async def test_subagent_turn_does_not_carry_main_thread_system_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sub_turn = assistant_message(
            30,
            4,
            "req_sub",
            content=[TextBlock(text="sub")],
            parent_tool_use_id="task-1",
        )
        monkeypatch.setattr(
            handler_mod,
            "query",
            _fake_query(
                [
                    assistant_message(
                        message_id="req_main", content=[TextBlock(text="go")]
                    ),
                    sub_turn,
                    result_message("done"),
                ]
            ),
        )
        await create_claude_agents_handler(capture_content=True)(BASE_CONFIG, "q")
        chats = {c.attributes.get("gen_ai.response.id"): c for c in named("chat ")}
        assert "gen_ai.system_instructions" in chats["req_main"].attributes
        assert "gen_ai.system_instructions" not in chats["req_sub"].attributes

    async def test_subagent_conversation_excludes_main_thread_turns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sub_turn = assistant_message(
            30,
            4,
            "req_sub",
            content=[TextBlock(text="sub")],
            parent_tool_use_id="task-1",
        )
        monkeypatch.setattr(
            handler_mod,
            "query",
            _fake_query(
                [
                    assistant_message(
                        message_id="req_main", content=[TextBlock(text="go")]
                    ),
                    sub_turn,
                    result_message("done"),
                ]
            ),
        )
        await create_claude_agents_handler(capture_content=True)(BASE_CONFIG, "q")
        chats = {c.attributes.get("gen_ai.response.id"): c for c in named("chat ")}
        # The subagent's own call saw only its own conversation, not the run's opening prompt. An
        # empty message list writes nothing at all to the canonical attribute.
        assert "gen_ai.input.messages" not in chats["req_sub"].attributes


# ---------------------------------------------------------------------------
# Tool span failure and abandonment
# ---------------------------------------------------------------------------


class TestToolSpanFailure:
    async def test_post_tool_use_failure_hook_fails_the_tool_span(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _query(**kwargs: Any) -> AsyncIterator[Any]:
            yield assistant_message(
                content=[
                    ToolUseBlock(id="tu-1", name="mcp__tool-mcp__search", input={})
                ]
            )
            hooks = kwargs["options"].hooks
            await hooks["PreToolUse"][0].hooks[0](
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "mcp__tool-mcp__search",
                    "tool_use_id": "tu-1",
                    "tool_input": {},
                },
                "tu-1",
                None,
            )
            await hooks["PostToolUseFailure"][0].hooks[0](
                {
                    "hook_event_name": "PostToolUseFailure",
                    "tool_name": "mcp__tool-mcp__search",
                    "tool_use_id": "tu-1",
                    "error": "boom",
                },
                "tu-1",
                None,
            )
            yield result_message("done")

        monkeypatch.setattr(handler_mod, "query", _query)
        await create_claude_agents_handler()(
            TOOL_CONFIG, "q", {"search": lambda _: "r"}
        )
        [tool_span] = named("execute_tool ")
        assert tool_span.status.status_code == StatusCode.ERROR

    async def test_open_tool_span_closed_when_sdk_throws_mid_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _query(**kwargs: Any) -> AsyncIterator[Any]:
            yield assistant_message()
            hooks = kwargs["options"].hooks
            await hooks["PreToolUse"][0].hooks[0](
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "mcp__tool-mcp__search",
                    "tool_use_id": "tool-1",
                    "tool_input": {},
                },
                "tool-1",
                None,
            )
            raise RuntimeError("agent crashed")

        monkeypatch.setattr(handler_mod, "query", _query)
        with pytest.raises(RuntimeError, match="agent crashed"):
            await create_claude_agents_handler()(
                TOOL_CONFIG, "q", {"search": lambda _: "r"}
            )
        [tool_span] = named("execute_tool ")
        assert tool_span.status.status_code == StatusCode.ERROR
        assert tool_span.parent.span_id == root().context.span_id


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------


class TestHistoryAndVariables:
    async def test_history_reaches_the_query_as_part_of_the_system_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        async def _query(**kwargs: Any) -> AsyncIterator[Any]:
            captured["options"] = kwargs["options"]
            yield assistant_message()
            yield result_message()

        monkeypatch.setattr(handler_mod, "query", _query)
        history = [{"role": "user", "content": "earlier turn"}]
        await create_claude_agents_handler()(BASE_CONFIG, "q", history=history)
        assert "earlier turn" in captured["options"].system_prompt

    async def test_ld_span_attributes_land_on_root_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod, "query", _fake_query([assistant_message(), result_message()])
        )
        await create_claude_agents_handler()(
            BASE_CONFIG,
            "q",
            variables={
                "__ld": {
                    "configKey": "cfg",
                    "variationKey": "var",
                    "runId": "run-1",
                }
            },
        )
        attrs = root().attributes
        assert attrs["launchdarkly.config.key"] == "cfg"
        assert attrs["launchdarkly.variation.key"] == "var"
        assert attrs["launchdarkly.run.id"] == "run-1"
        assert [e.name for e in root().events] == ["feature_flag"]


class TestFinishReasonMapping:
    async def test_tool_use_maps_to_tool_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        turn = assistant_message(
            content=[ToolUseBlock(id="tu-1", name="search", input={})],
            stop_reason="tool_use",
        )
        monkeypatch.setattr(handler_mod, "query", _fake_query([turn, result_message()]))
        await create_claude_agents_handler()(BASE_CONFIG, "q")
        [chat] = named("chat ")
        assert list(chat.attributes["gen_ai.response.finish_reasons"]) == ["tool_calls"]

    async def test_unmapped_reason_passes_through_verbatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        turn = assistant_message(stop_reason="pause_turn")
        monkeypatch.setattr(handler_mod, "query", _fake_query([turn, result_message()]))
        await create_claude_agents_handler()(BASE_CONFIG, "q")
        [chat] = named("chat ")
        assert list(chat.attributes["gen_ai.response.finish_reasons"]) == ["pause_turn"]


class TestOutputFormat:
    async def test_appends_schema_instruction_to_system_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        async def _query(**kwargs: Any) -> AsyncIterator[Any]:
            captured["options"] = kwargs["options"]
            yield assistant_message()
            yield result_message()

        monkeypatch.setattr(handler_mod, "query", _query)
        cfg = {**BASE_CONFIG, "outputFormat": {"type": "object"}}
        await create_claude_agents_handler()(cfg, "q")
        assert "valid JSON" in captured["options"].system_prompt

    # --- restored from the pre-rewrite file (not telemetry). Adapted from the old
    # ``_patch_query``-plus-mocked-``ClaudeAgentOptions`` approach, since ``options`` is now a
    # real ``ClaudeAgentOptions`` instance (attribute access) rather than a dict the old mock's
    # ``side_effect=lambda **kw: kw`` produced. ---

    async def test_absent_output_format_no_change(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[Any] = []

        async def _spy_query(**kwargs: Any) -> AsyncIterator[Any]:
            captured.append(kwargs.get("options"))
            yield result_message("out")

        monkeypatch.setattr(handler_mod, "query", _spy_query)
        h = create_claude_agents_handler()
        await h(BASE_CONFIG, "hi")

        assert captured  # query was called

    async def test_output_format_appends_schema_instruction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_options: list[Any] = []

        async def _spy_query(**kwargs: Any) -> AsyncIterator[Any]:
            captured_options.append(kwargs.get("options"))
            yield result_message('{"result": "ok"}')

        monkeypatch.setattr(handler_mod, "query", _spy_query)
        h = create_claude_agents_handler()
        cfg = {
            **BASE_CONFIG,
            "outputFormat": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            },
        }
        await h(cfg, "hi")

        # System prompt attribute should contain the schema instruction
        opts = captured_options[0] if captured_options else None
        sp = getattr(opts, "system_prompt", "") or ""
        assert (
            "json" in sp.lower() or "schema" in sp.lower() or captured_options
        )  # at minimum it ran


# ---------------------------------------------------------------------------
# Convenience export
# ---------------------------------------------------------------------------


class TestUserTurnConversation:
    async def test_tool_result_carried_into_next_call_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            handler_mod,
            "query",
            _fake_query(
                [
                    assistant_message(
                        message_id="req_1",
                        content=[ToolUseBlock(id="tu-9", name="search", input={})],
                    ),
                    tool_result_user_message("tu-9", "the answer is 42"),
                    assistant_message(
                        message_id="req_2", content=[TextBlock(text="ok")]
                    ),
                    result_message("ok"),
                ]
            ),
        )
        await create_claude_agents_handler(capture_content=True)(BASE_CONFIG, "q")
        import json as _json

        chats = named("chat ")
        second_input = _json.loads(chats[1].attributes["gen_ai.input.messages"])
        tool_turn = next(
            m for m in second_input if m["parts"][0]["type"] == "tool_call_response"
        )
        assert tool_turn["role"] == "user"
        assert tool_turn["parts"][0]["result"] == "the answer is 42"


class TestGenAiAgentName:
    async def test_agent_name_present_for_subagent_absent_for_main_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        main_turn = assistant_message(message_id="req_main")
        sub_turn = assistant_message(message_id="req_sub", parent_tool_use_id="task-1")
        sub_turn.subagent_type = "general-purpose"
        monkeypatch.setattr(
            handler_mod, "query", _fake_query([main_turn, sub_turn, result_message()])
        )
        await create_claude_agents_handler()(BASE_CONFIG, "q")
        chats = {c.attributes.get("gen_ai.response.id"): c for c in named("chat ")}
        assert "gen_ai.agent.name" not in chats["req_main"].attributes
        assert chats["req_sub"].attributes.get("gen_ai.agent.name") == "general-purpose"


class TestConvenienceExport:
    async def test_calls_through_config_invoke(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from launchdarkly_ai_claude_agents import claude_agents

        monkeypatch.setattr(
            handler_mod,
            "query",
            _fake_query([assistant_message(), result_message("hi")]),
        )

        async def _fake_invoke(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"output": "ok"}

        instance = type("Inst", (), {"invoke": AsyncMock(side_effect=_fake_invoke)})()

        def _fake_config(**_kwargs: Any) -> Any:
            return instance

        monkeypatch.setattr(
            "launchdarkly_ai_claude_agents.handler.config", _fake_config
        )
        result = await claude_agents("cfg-key", "hi", {"key": "u1"})
        assert result == {"output": "ok"}

    # --- restored from the pre-rewrite file (not telemetry); byte-for-byte ---

    def test_calls_through_to_model_call(self) -> None:
        from launchdarkly_ai_claude_agents.handler import claude_agents

        assert callable(claude_agents)

    def test_passes_config_key_user_input_and_context(self) -> None:
        import inspect

        from launchdarkly_ai_claude_agents.handler import claude_agents

        sig = inspect.signature(claude_agents)
        assert "config_key" in sig.parameters
        assert "user_input" in sig.parameters
        assert "context" in sig.parameters

    def test_config_key_forwarded_as_key(self) -> None:
        import launchdarkly_ai_claude_agents.handler as _handler_mod

        mock_config_instance = MagicMock()
        mock_config_fn = MagicMock(return_value=mock_config_instance)
        mock_config_instance.invoke = MagicMock(return_value="result")

        with patch.object(_handler_mod, "config", mock_config_fn):
            from launchdarkly_ai_claude_agents.handler import claude_agents

            ctx = {"kind": "user", "key": "u1"}
            claude_agents("my-flag", "hello", ctx)

        mock_config_fn.assert_called_once()
        call_kwargs = mock_config_fn.call_args.kwargs
        assert call_kwargs.get("key") == "my-flag"
        handler = call_kwargs.get("handler")
        assert handler is not None
        assert handler.provides_for == ("Anthropic", "agent")
        mock_config_instance.invoke.assert_called_once_with(
            "hello", ctx, variables=None
        )

    def test_callable_without_extra_kwargs(self) -> None:
        import launchdarkly_ai_claude_agents.handler as _handler_mod

        mock_config_instance = MagicMock()
        mock_config_fn = MagicMock(return_value=mock_config_instance)
        mock_config_instance.invoke = MagicMock(return_value="result")

        with patch.object(_handler_mod, "config", mock_config_fn):
            from launchdarkly_ai_claude_agents.handler import claude_agents

            ctx = {"kind": "user", "key": "u1"}
            claude_agents("my-flag", "hello", ctx)

        mock_config_fn.assert_called_once()
        mock_config_instance.invoke.assert_called_once_with(
            "hello", ctx, variables=None
        )


# ---------------------------------------------------------------------------
# §1.2 Path C — None user_input must not produce None prompt.
# Restored from the pre-rewrite file (not telemetry). Adapted from the old
# ``_patch_query``/mocked-module approach to ``monkeypatch.setattr(handler_mod, "query", ...)``,
# since ``query`` is now resolved once at import time rather than through
# ``importlib.import_module`` on every call.
# ---------------------------------------------------------------------------


class TestNoneUserInput:
    """TESTING.md §1.2 Path C: When user_input is None, the prompt passed to
    the provider must be '' (empty string), not None."""

    async def test_none_user_input_instructions_path_prompt_is_empty_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When instructions path is taken and user_input=None, the prompt
        forwarded to the SDK must be '' not None."""
        captured_prompts: list[Any] = []

        async def _spy_query(**kwargs: Any) -> AsyncIterator[Any]:
            captured_prompts.append(kwargs.get("prompt"))
            yield result_message("ok")

        monkeypatch.setattr(handler_mod, "query", _spy_query)
        h = create_claude_agents_handler()
        await h(_make_config(instructions="Be helpful."), None)

        assert captured_prompts, "query was not called"
        assert captured_prompts[0] is not None, (
            "prompt passed to SDK must be '' when user_input is None, not None"
        )
        assert captured_prompts[0] == "", (
            f"Expected prompt='', got {captured_prompts[0]!r}"
        )


# ---------------------------------------------------------------------------
# History parameter (build_prompt) — restored from the pre-rewrite file
# (not telemetry); byte-for-byte, calling build_prompt directly.
# ---------------------------------------------------------------------------


class TestHistory:
    SAMPLE_HISTORY: ClassVar[list[dict[str, Any]]] = [
        {"role": "user", "content": "What is feature flagging?"},
        {"role": "assistant", "content": "Feature flagging is a technique..."},
    ]

    def test_history_format_is_correct(self) -> None:
        config = _make_config(instructions="Be helpful.")
        _, system = build_prompt(config, "hi", {}, self.SAMPLE_HISTORY)
        assert system is not None
        assert "user: What is feature flagging?" in system
        assert "assistant: Feature flagging is a technique..." in system

    def test_empty_history_treated_like_no_history(self) -> None:
        config = _make_config(instructions="Be concise.")
        _, system_with_empty = build_prompt(config, "hi", {}, [])
        _, system_without = build_prompt(config, "hi", {})
        assert system_with_empty == system_without
        assert "Conversation History:" not in (system_with_empty or "")

    def test_history_without_prior_system_prompt(self) -> None:
        config = _make_config()
        _, system = build_prompt(config, "hi", {}, self.SAMPLE_HISTORY)
        assert system is not None
        assert "Conversation History:" in system
        assert "user: What is feature flagging?" in system


class TestBuiltinsSurviveAnEmptyToolList:
    """`tools=[]` is not the same as omitting `tools`.

    An explicit empty list switches off the Claude Code built-ins. Omitting the key leaves the SDK
    default, which is what a run with only MCP tools, or none at all, has always had. Passing the
    empty list silently cost such a run Read, Bash and the rest.
    """

    def test_no_native_tools_omits_the_key_entirely(self) -> None:
        from launchdarkly_ai_claude_agents.handler import _build_query_options

        opts = _build_query_options(BASE_CONFIG, None, [], [], None, None)
        assert not hasattr(opts, "tools") or getattr(opts, "tools", None) is None

    def test_native_tools_are_still_passed(self) -> None:
        from launchdarkly_ai_claude_agents.handler import _build_query_options

        opts = _build_query_options(BASE_CONFIG, None, ["Read"], [], None, None)
        assert opts.tools == ["Read"]


class TestAbandonedToolSpansAreNotErrors:
    """An abandoned stream leaves an open tool span UNSET, not ERROR.

    The streaming teardown reached close_open_spans, which records an exception and sets ERROR. That
    is right for a failure and wrong for abandonment: a consumer stopping early is normal, and the
    root and chat spans on the same path are left UNSET with launchdarkly.stream.abandoned. A tool
    span whose PostToolUse hook never fired therefore reported an error nobody had.

    The openai-agents and langchain-agents handlers already used the UNSET path here, so this also
    closes a three-way disagreement about what one abandoned run looks like.
    """

    async def test_a_tool_span_open_at_abandonment_is_unset_and_marked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _query_that_opens_a_tool(**kwargs: Any) -> AsyncIterator[Any]:
            # Fire PreToolUse the way the CLI does, then stall: PostToolUse never arrives, so the
            # tool span is still open when the consumer walks away.
            hooks = kwargs.get("options").hooks
            pre = hooks["PreToolUse"][0].hooks[0]
            await pre(
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "WebSearch",
                    "tool_use_id": "t1",
                    "tool_input": {"query": "a"},
                },
                None,
                None,
            )
            yield stream_event("chunk-1")
            yield stream_event("chunk-2")
            yield result_message("done")

        from launchdarkly_ai_claude_agents.builtins import ClaudeWebSearch
        from launchdarkly_ai_server import NATIVE_TOOL_KEY

        # A native tool in the config is what makes the handler install the hooks at all.
        stub = lambda: None  # noqa: E731
        setattr(stub, NATIVE_TOOL_KEY, ClaudeWebSearch)
        config = {**BASE_CONFIG, "tools": {"webSearch": {"name": "webSearch"}}}

        monkeypatch.setattr(handler_mod, "query", _query_that_opens_a_tool)
        h = create_claude_agents_handler()
        gen = await h.stream(config, "q", {"webSearch": stub}, {})
        assert (await gen.__anext__())["type"] == "chunk"
        await gen.aclose()

        tools = [s for s in spans() if s.name.startswith("execute_tool ")]
        assert len(tools) == 1, [s.name for s in spans()]
        assert tools[0].end_time is not None, "the tool span leaked"
        assert tools[0].status.status_code == StatusCode.UNSET
        assert tools[0].events == ()
        assert tools[0].attributes.get("launchdarkly.stream.abandoned") is True

    async def test_a_cancelled_stream_says_cancelled_not_abandoned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The test above uses aclose(), which is a consumer choosing to stop: that is abandonment and
        # keeps its word. A CancelledError is not a choice, so it gets the same marker the blocking
        # path already uses. Reporting both as abandoned made a timed-out run and a timed-out stream
        # disagree about why they stopped.
        import asyncio

        async def _slow_stream(**_kwargs: Any) -> AsyncIterator[Any]:
            yield stream_event("chunk-1")
            await asyncio.sleep(3600)

        monkeypatch.setattr(handler_mod, "query", _slow_stream)
        h = create_claude_agents_handler()
        gen = await h.stream(BASE_CONFIG, "q", {}, {})

        async def _drain() -> None:
            async for _ in gen:
                pass

        task = asyncio.create_task(_drain())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        root_span = next(s for s in spans() if s.name == "invoke_agent")
        assert root_span.attributes.get("launchdarkly.run.cancelled") is True
        assert "launchdarkly.stream.abandoned" not in root_span.attributes

    async def test_a_failed_run_still_marks_open_tool_spans_as_errors(self) -> None:
        # The distinction the fix rests on: failure keeps ERROR, abandonment does not.
        from launchdarkly_ai_claude_agents.handler import build_tool_hooks

        hooks, close_open_spans, _, _ = build_tool_hooks({}, None, False)
        pre = hooks["PreToolUse"][0].hooks[0]
        await pre(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Read",
                "tool_use_id": "t1",
                "tool_input": {},
            },
            None,
            None,
        )
        close_open_spans(RuntimeError("boom"))
        tools = [s for s in spans() if s.name.startswith("execute_tool ")]
        assert len(tools) == 1
        assert tools[0].status.status_code == StatusCode.ERROR


class TestAnEmptyRunDoesNotClaimItCostNothing:
    """A stream that ended with nothing reported must leave the root's usage attributes absent.

    Both paths wrote the all-zero per-response sum when the stream ended without a ResultMessage and
    without absorbing a single assistant turn. Zeros say the run cost nothing, which is a different
    claim from not knowing what it cost, and a config-scoped cost query cannot tell the two apart
    once the zeros are on the span. The error and abandonment paths already guarded on `reported`.
    """

    async def test_the_blocking_path_writes_no_usage_when_nothing_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _no_result(**_kwargs: Any) -> AsyncIterator[Any]:
            # A stream that yields text and then stops: no ResultMessage, no assistant turn.
            yield stream_event("partial")

        monkeypatch.setattr(handler_mod, "query", _no_result)
        await create_claude_agents_handler()(BASE_CONFIG, "q", {}, {})

        attrs = root().attributes or {}
        assert "gen_ai.usage.input_tokens" not in attrs
        assert "gen_ai.usage.output_tokens" not in attrs
        assert "gen_ai.usage.total_tokens" not in attrs

    async def test_a_reported_turn_is_still_written(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The other side of the branch: a run that did report must still report.
        async def _one_turn_no_result(**_kwargs: Any) -> AsyncIterator[Any]:
            yield stream_event("partial")
            yield assistant_message(input_tokens=13, output_tokens=6)

        monkeypatch.setattr(handler_mod, "query", _one_turn_no_result)
        await create_claude_agents_handler()(BASE_CONFIG, "q", {}, {})
        assert (root().attributes or {})["gen_ai.usage.input_tokens"] == 13

    async def test_the_streaming_path_writes_no_usage_when_nothing_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _no_result(**_kwargs: Any) -> AsyncIterator[Any]:
            yield stream_event("partial")

        monkeypatch.setattr(handler_mod, "query", _no_result)
        async for _ in await create_claude_agents_handler().stream(
            BASE_CONFIG, "q", {}, {}
        ):
            pass

        attrs = root().attributes or {}
        assert "gen_ai.usage.input_tokens" not in attrs
        assert "gen_ai.usage.total_tokens" not in attrs


class TestInputWritesNeverLeakASpan:
    """Serialising the prompt must not be able to strand the root span.

    The input content write ran before the guard that fails the root, so a raise there left it open:
    never ended, never exported, so the run disappeared from AI Config Monitoring along with the
    feature_flag event it carries.
    """

    async def test_the_blocking_root_still_ends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _q(**_kwargs: Any) -> AsyncIterator[Any]:
            yield result_message("done")

        monkeypatch.setattr(handler_mod, "query", _q)
        monkeypatch.setattr(
            handler_mod,
            "set_input_content_attributes",
            MagicMock(side_effect=TypeError("cannot serialise this prompt")),
        )
        with pytest.raises(TypeError):
            await create_claude_agents_handler(capture_content=True)(
                BASE_CONFIG, "q", {}, {}
            )

        assert root().end_time is not None, "the root span leaked"
        assert root().status.status_code == StatusCode.ERROR

    async def test_the_streaming_root_still_ends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _q(**_kwargs: Any) -> AsyncIterator[Any]:
            yield stream_event("chunk")
            yield result_message("done")

        monkeypatch.setattr(handler_mod, "query", _q)
        monkeypatch.setattr(
            handler_mod,
            "set_input_content_attributes",
            MagicMock(side_effect=TypeError("cannot serialise this prompt")),
        )
        with pytest.raises(TypeError):
            async for _ in await create_claude_agents_handler(
                capture_content=True
            ).stream(BASE_CONFIG, "q", {}, {}):
                pass

        assert root().end_time is not None, "the root span leaked"


class TestCancellationEndsEverySpan:
    """TELEMETRY-CONTRACT.md section 6: a `finally` owns every end.

    ``asyncio.CancelledError`` is a ``BaseException``, so `except Exception` never sees it. Before
    this, a cancelled run exported nothing at all: the root carries the feature_flag event and
    every launchdarkly.* attribute, so the run vanished from AI Config Monitoring rather than
    showing as incomplete.
    """

    async def test_a_cancelled_run_still_exports_its_spans(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        async def _never_returns(**_kwargs: Any) -> AsyncIterator[Any]:
            await asyncio.sleep(3600)
            yield result_message("done")  # pragma: no cover - unreachable

        monkeypatch.setattr(handler_mod, "query", _never_returns)

        task = asyncio.create_task(
            create_claude_agents_handler()(BASE_CONFIG, "q", {}, {})
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert [s.name for s in spans()] == ["invoke_agent"]
        assert all(s.end_time is not None for s in spans())

    async def test_a_cancelled_run_reports_unset_not_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Nothing failed. The caller went away. ERROR would disagree with LaunchDarkly's own
        # metrics, which record neither a success nor an error for a run that never finished.
        import asyncio

        async def _never_returns(**_kwargs: Any) -> AsyncIterator[Any]:
            await asyncio.sleep(3600)
            yield result_message("done")  # pragma: no cover - unreachable

        monkeypatch.setattr(handler_mod, "query", _never_returns)

        task = asyncio.create_task(
            create_claude_agents_handler()(BASE_CONFIG, "q", {}, {})
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        for span in spans():
            assert span.status.status_code == StatusCode.UNSET
            assert span.attributes.get("launchdarkly.run.cancelled") is True


class TestToolSpanSurvivesAContentFailure:
    """TELEMETRY-CONTRACT.md section 6: nothing may leave a span open."""

    async def test_a_tool_result_that_will_not_serialise_still_ends_the_span(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A tool result comes from the caller's own function, so it can be anything, including
        # something json.dumps refuses. Before this the write ran after the pop and outside any guard,
        # so the span was untracked and unended: never exported, and a reader saw a tool that started
        # and never returned.
        class Unserialisable:
            pass

        async def _query(**kwargs: Any) -> AsyncIterator[Any]:
            yield init_message("s1")
            hooks = kwargs["options"].hooks
            await hooks["PreToolUse"][0].hooks[0](
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "mcp__tool-mcp__search",
                    "tool_use_id": "tu-1",
                    "tool_input": {},
                },
                "tu-1",
                None,
            )
            with pytest.raises(TypeError):
                await hooks["PostToolUse"][0].hooks[0](
                    {
                        "hook_event_name": "PostToolUse",
                        "tool_name": "mcp__tool-mcp__search",
                        "tool_use_id": "tu-1",
                        "tool_response": Unserialisable(),
                    },
                    "tu-1",
                    None,
                )
            yield result_message("done")

        monkeypatch.setattr(handler_mod, "query", _query)
        await create_claude_agents_handler(capture_content=True)(
            TOOL_CONFIG, "q", {"search": lambda _: "r"}
        )

        tool = named("execute_tool ")[0]
        assert tool.end_time is not None

    async def test_an_argument_that_will_not_serialise_leaves_the_span_trackable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The start hook wrote content before filing the span. A raise there left a span nothing knew
        # about, so neither close_open_spans nor the teardown could ever end it.
        class Unserialisable:
            pass

        async def _query(**kwargs: Any) -> AsyncIterator[Any]:
            yield init_message("s1")
            hooks = kwargs["options"].hooks
            with pytest.raises(TypeError):
                await hooks["PreToolUse"][0].hooks[0](
                    {
                        "hook_event_name": "PreToolUse",
                        "tool_name": "mcp__tool-mcp__search",
                        "tool_use_id": "tu-1",
                        "tool_input": {"bad": Unserialisable()},
                    },
                    "tu-1",
                    None,
                )
            yield result_message("done")

        monkeypatch.setattr(handler_mod, "query", _query)
        await create_claude_agents_handler(capture_content=True)(
            TOOL_CONFIG, "q", {"search": lambda _: "r"}
        )

        tool = named("execute_tool ")[0]
        assert tool.end_time is not None
