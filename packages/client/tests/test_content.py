"""Tests for the span content layer.

Covers TELEMETRY-CONTRACT.md sections 5 (finish reasons) and 7 (content capture).

The recurring assertion in this file is the gate: with ``capture=False`` nothing at all reaches the
span. Conversation content is PII, so that is the behaviour worth pinning hardest.
"""

from __future__ import annotations

import json
from typing import Any

from launchdarkly_ai_server.content import (
    SpanMessage,
    SpanMessagePart,
    ToolDefinitionInput,
    lang_chain_finish_reasons,
    lang_chain_span_messages,
    set_input_content_attributes,
    set_output_content_attributes,
    set_tool_call_content_attributes,
    set_tool_definition_attributes,
    text_message,
    to_semconv_finish_reason,
)


class FakeSpan:
    """Records what a handler wrote, so a test can assert on the whole span at once."""

    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}
        self.events: list[tuple[str, dict[str, Any]]] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append((name, attributes or {}))


# ─── to_semconv_finish_reason ────────────────────────────────────────────────


class TestToSemconvFinishReason:
    def test_maps_every_anthropic_spelling(self) -> None:
        assert to_semconv_finish_reason("end_turn") == "stop"
        assert to_semconv_finish_reason("stop_sequence") == "stop"
        assert to_semconv_finish_reason("max_tokens") == "length"
        assert to_semconv_finish_reason("tool_use") == "tool_calls"
        assert to_semconv_finish_reason("refusal") == "content_filter"

    def test_maps_every_openai_spelling(self) -> None:
        assert to_semconv_finish_reason("stop") == "stop"
        assert to_semconv_finish_reason("length") == "length"
        assert to_semconv_finish_reason("tool_calls") == "tool_calls"
        assert to_semconv_finish_reason("content_filter") == "content_filter"
        assert to_semconv_finish_reason("function_call") == "tool_calls"

    def test_compares_lower_cased(self) -> None:
        assert to_semconv_finish_reason("END_TURN") == "stop"
        assert to_semconv_finish_reason("Max_Tokens") == "length"

    def test_passes_an_unmapped_reason_through_verbatim(self) -> None:
        # The passthrough is the signal to add a row to the table, so it must not become 'stop'.
        assert to_semconv_finish_reason("brand_new_reason") == "brand_new_reason"

    def test_pause_turn_is_deliberately_unmapped(self) -> None:
        # No semconv value means "did not finish", so this must not be flattened into 'stop'.
        assert to_semconv_finish_reason("pause_turn") == "pause_turn"

    def test_absent_and_empty_produce_nothing(self) -> None:
        assert to_semconv_finish_reason(None) is None
        assert to_semconv_finish_reason("") is None


# ─── lang_chain_finish_reasons ───────────────────────────────────────────────


class _Message:
    def __init__(self, response_metadata: dict[str, Any] | None = None) -> None:
        self.response_metadata = response_metadata or {}


class TestLangChainFinishReasons:
    def test_reads_generation_info_first(self) -> None:
        result = {"generations": [[{"generation_info": {"finish_reason": "length"}}]]}
        assert lang_chain_finish_reasons(result) == ["length"]

    def test_falls_back_to_response_metadata_finish_reason(self) -> None:
        msg = _Message({"finish_reason": "tool_calls"})
        assert lang_chain_finish_reasons(msg) == ["tool_calls"]

    def test_falls_back_to_response_metadata_stop_reason(self) -> None:
        # Anthropic through LangChain: the reason lands under a different key and still maps.
        msg = _Message({"stop_reason": "end_turn"})
        assert lang_chain_finish_reasons(msg) == ["stop"]

    def test_accepts_a_bare_ai_message(self) -> None:
        assert lang_chain_finish_reasons(_Message({"finish_reason": "stop"})) == [
            "stop"
        ]

    def test_flattens_nested_generations(self) -> None:
        result = {
            "generations": [
                [{"generation_info": {"finish_reason": "stop"}}],
                [{"generation_info": {"finish_reason": "length"}}],
            ]
        }
        assert lang_chain_finish_reasons(result) == ["stop", "length"]

    def test_returns_none_not_empty_list_when_nothing_is_present(self) -> None:
        # None leaves the attribute off; [] would assert the turn finished for no reason.
        assert lang_chain_finish_reasons(_Message()) is None
        assert lang_chain_finish_reasons({"generations": []}) is None


# ─── lang_chain_span_messages ────────────────────────────────────────────────


class _LCMessage:
    def __init__(
        self,
        msg_type: str,
        content: Any = "",
        tool_calls: list[Any] | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        self._type = msg_type
        self.content = content
        if tool_calls is not None:
            self.tool_calls = tool_calls
        if tool_call_id is not None:
            self.tool_call_id = tool_call_id

    def _get_type(self) -> str:
        return self._type


class TestToolCallArgumentsAgreeAcrossCarriers:
    def test_absent_arguments_are_omitted_not_written_as_null(self) -> None:
        # to_canonical omits absent arguments. to_text used to write them as null, so the OpenLLMetry
        # carrier said the model passed a null argument bag where the canonical one said it passed
        # none. The module docstring promises the three carriers cannot disagree.
        part = SpanMessagePart(type="tool_call", name="f")
        assert part.to_canonical() == {"type": "tool_call", "name": "f"}
        assert part.to_text() == json.dumps({"name": "f"})

    def test_arguments_that_are_present_still_appear(self) -> None:
        part = SpanMessagePart(type="tool_call", name="f", arguments={"q": 1})
        assert part.to_text() == json.dumps({"name": "f", "arguments": {"q": 1}})

    def test_an_empty_argument_bag_is_not_treated_as_absent(self) -> None:
        # {} is a real thing a model can send: a tool called with no arguments. Only None is absent.
        part = SpanMessagePart(type="tool_call", name="f", arguments={})
        assert part.to_text() == json.dumps({"name": "f", "arguments": {}})


class TestContentWritersToleratePeopleWithoutOpenTelemetry:
    """Handlers hold None for every span when the `otel` extra is absent."""

    def test_no_writer_raises_on_a_none_span(self) -> None:
        # capture_content=True without the extra installed used to raise AttributeError from inside
        # the telemetry path, after the provider had already billed the turn. Telemetry may report
        # nothing; it may not break the call it is reporting on.
        messages = [
            SpanMessage(
                role="assistant", parts=[SpanMessagePart(type="text", content="hi")]
            )
        ]
        set_input_content_attributes(None, True, messages=messages)
        set_output_content_attributes(None, True, messages)
        set_tool_call_content_attributes(None, True, arguments={"a": 1}, result="r")


class TestLangChainRoleAccessor:
    def test_reads_the_type_field_when_the_method_is_gone(self) -> None:
        # langchain-core replaced `_get_type()` with a plain `type` field and dropped the method.
        # Reading only the method sent every message to role `user`, silently, because getattr
        # returns None for a missing attribute rather than raising.
        from types import SimpleNamespace

        messages = [
            SimpleNamespace(type="system", content="Be helpful."),
            SimpleNamespace(type="human", content="hi"),
            SimpleNamespace(type="ai", content="hello"),
        ]
        system, converted = lang_chain_span_messages(messages)

        assert system == "Be helpful."
        assert [m.role for m in converted] == ["user", "assistant"]

    def test_the_method_still_wins_where_it_exists(self) -> None:
        # Older releases expose both, and the method was the canonical accessor.
        class _Old:
            type = "human"
            content = "hi"

            def _get_type(self) -> str:
                return "ai"

        _, converted = lang_chain_span_messages([_Old()])
        assert [m.role for m in converted] == ["assistant"]


class TestLangChainSpanMessages:
    def test_lifts_the_system_prompt_out(self) -> None:
        system, messages = lang_chain_span_messages(
            [_LCMessage("system", "Be helpful."), _LCMessage("human", "hi")]
        )
        assert system == "Be helpful."
        assert [m.role for m in messages] == ["user"]

    def test_keeps_a_bare_string_a_list_holds_beside_typed_blocks(self) -> None:
        # LangChain types content as `str | list[str | dict]`, so a bare string in the list is what
        # the library documents. Keeping only `type: "text"` blocks dropped it, and the span then
        # showed less of the conversation than the model was given.
        _, messages = lang_chain_span_messages(
            [
                _LCMessage(
                    "human",
                    ["a bare string", {"type": "text", "text": "a typed block"}],
                )
            ]
        )
        assert [p.to_canonical() for p in messages[0].parts] == [
            {"type": "text", "content": "a bare stringa typed block"}
        ]

    def test_still_ignores_a_block_that_is_not_text(self) -> None:
        _, messages = lang_chain_span_messages(
            [
                _LCMessage(
                    "human",
                    [
                        {
                            "type": "image_url",
                            "image_url": "https://example.test/x.png",
                        },
                        {"type": "text", "text": "caption"},
                    ],
                )
            ]
        )
        assert [p.to_canonical() for p in messages[0].parts] == [
            {"type": "text", "content": "caption"}
        ]

    def test_still_reads_a_plain_string_content_unchanged(self) -> None:
        _, messages = lang_chain_span_messages([_LCMessage("human", "just a string")])
        assert [p.to_canonical() for p in messages[0].parts] == [
            {"type": "text", "content": "just a string"}
        ]

    def test_treats_developer_as_system(self) -> None:
        system, messages = lang_chain_span_messages([_LCMessage("developer", "rules")])
        assert system == "rules"
        assert messages == []

    def test_renames_langchain_roles_to_semconv_roles(self) -> None:
        _, messages = lang_chain_span_messages(
            [_LCMessage("human", "q"), _LCMessage("ai", "a")]
        )
        assert [m.role for m in messages] == ["user", "assistant"]

    def test_converts_a_tool_message_to_a_response_part(self) -> None:
        _, messages = lang_chain_span_messages(
            [_LCMessage("tool", "42", tool_call_id="call_1")]
        )
        assert messages[0].role == "tool"
        part = messages[0].parts[0]
        assert part.type == "tool_call_response"
        assert part.id == "call_1"
        assert part.result == "42"

    def test_carries_tool_calls_off_an_assistant_message(self) -> None:
        _, messages = lang_chain_span_messages(
            [
                _LCMessage(
                    "ai",
                    "calling",
                    tool_calls=[
                        {"id": "c1", "name": "get_weather", "args": {"city": "NYC"}}
                    ],
                )
            ]
        )
        parts = messages[0].parts
        assert parts[0].type == "text"
        assert parts[1].type == "tool_call"
        assert parts[1].name == "get_weather"
        assert parts[1].arguments == {"city": "NYC"}

    def test_keeps_only_text_blocks_from_block_list_content(self) -> None:
        _, messages = lang_chain_span_messages(
            [
                _LCMessage(
                    "human",
                    [
                        {"type": "text", "text": "look at "},
                        {"type": "image_url", "image_url": "http://x"},
                        {"type": "text", "text": "this"},
                    ],
                )
            ]
        )
        assert messages[0].parts[0].content == "look at this"

    def test_joins_several_system_messages(self) -> None:
        system, _ = lang_chain_span_messages(
            [_LCMessage("system", "one"), _LCMessage("system", "two")]
        )
        assert system == "one\ntwo"

    def test_returns_none_for_system_when_there_is_none(self) -> None:
        system, _ = lang_chain_span_messages([_LCMessage("human", "hi")])
        assert system is None


# ─── The capture gate ────────────────────────────────────────────────────────


class TestCaptureGate:
    def test_input_writes_nothing_when_capture_is_off(self) -> None:
        span = FakeSpan()
        set_input_content_attributes(
            span,
            False,
            system_instructions="secret",
            messages=[text_message("user", "PII")],
            tool_definitions=[ToolDefinitionInput(name="t")],
        )
        assert span.attributes == {}
        assert span.events == []

    def test_output_writes_nothing_when_capture_is_off(self) -> None:
        span = FakeSpan()
        set_output_content_attributes(span, False, [text_message("assistant", "PII")])
        assert span.attributes == {}
        assert span.events == []

    def test_tool_definitions_write_nothing_when_capture_is_off(self) -> None:
        span = FakeSpan()
        set_tool_definition_attributes(span, False, [ToolDefinitionInput(name="t")])
        assert span.attributes == {}

    def test_tool_call_io_writes_nothing_when_capture_is_off(self) -> None:
        span = FakeSpan()
        set_tool_call_content_attributes(span, False, arguments={"a": 1}, result="r")
        assert span.attributes == {}


# ─── Input content ───────────────────────────────────────────────────────────


class TestInputContentAttributes:
    def test_writes_all_three_carriers(self) -> None:
        span = FakeSpan()
        set_input_content_attributes(
            span,
            True,
            system_instructions="Be brief.",
            messages=[text_message("user", "hi")],
        )

        assert json.loads(span.attributes["gen_ai.system_instructions"]) == [
            {"type": "text", "content": "Be brief."}
        ]
        assert json.loads(span.attributes["gen_ai.input.messages"]) == [
            {"role": "user", "parts": [{"type": "text", "content": "hi"}]}
        ]
        # The system prompt takes index 0 of the OpenLLMetry carrier, which has no slot of its own.
        assert span.attributes["gen_ai.prompt.0.role"] == "system"
        assert span.attributes["gen_ai.prompt.0.content"] == "Be brief."
        assert span.attributes["gen_ai.prompt.1.role"] == "user"
        assert span.attributes["gen_ai.prompt.1.content"] == "hi"
        assert span.events[0][0] == "gen_ai.content.prompt"
        assert span.events[0][1]["gen_ai.prompt"] == "system: Be brief.\nuser: hi"

    def test_omits_the_canonical_keys_when_there_is_nothing_to_say(self) -> None:
        span = FakeSpan()
        set_input_content_attributes(span, True, messages=[])
        assert "gen_ai.system_instructions" not in span.attributes
        assert "gen_ai.input.messages" not in span.attributes
        assert "gen_ai.prompt.0.role" not in span.attributes

    def test_still_adds_the_legacy_prompt_event_when_empty(self) -> None:
        # Asymmetric with the output side on purpose. This matches the TypeScript SDK; see the
        # docstring on set_input_content_attributes.
        span = FakeSpan()
        set_input_content_attributes(span, True, messages=[])
        assert span.events == [("gen_ai.content.prompt", {"gen_ai.prompt": ""})]

    def test_writes_the_tool_catalog_when_given_one(self) -> None:
        span = FakeSpan()
        set_input_content_attributes(
            span,
            True,
            messages=[text_message("user", "hi")],
            tool_definitions=[
                ToolDefinitionInput(
                    name="get_weather", description="d", parameters={"type": "object"}
                )
            ],
        )
        assert json.loads(span.attributes["gen_ai.tool.definitions"]) == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "d",
                "parameters": {"type": "object"},
            }
        ]

    def test_a_message_with_no_system_prompt_starts_at_index_zero(self) -> None:
        span = FakeSpan()
        set_input_content_attributes(span, True, messages=[text_message("user", "hi")])
        assert span.attributes["gen_ai.prompt.0.role"] == "user"


# ─── Output content ──────────────────────────────────────────────────────────


class TestOutputContentAttributes:
    def test_writes_all_three_carriers(self) -> None:
        span = FakeSpan()
        set_output_content_attributes(span, True, [text_message("assistant", "hello")])
        assert json.loads(span.attributes["gen_ai.output.messages"]) == [
            {"role": "assistant", "parts": [{"type": "text", "content": "hello"}]}
        ]
        assert span.attributes["gen_ai.completion.0.role"] == "assistant"
        assert span.attributes["gen_ai.completion.0.content"] == "hello"
        assert span.events == [
            ("gen_ai.content.completion", {"gen_ai.completion": "hello"})
        ]

    def test_writes_nothing_at_all_for_an_empty_message_list(self) -> None:
        span = FakeSpan()
        set_output_content_attributes(span, True, [])
        assert span.attributes == {}
        assert span.events == []

    def test_carries_the_finish_reason_into_the_canonical_shape(self) -> None:
        span = FakeSpan()
        message = SpanMessage(
            role="assistant",
            parts=[SpanMessagePart(type="text", content="hi")],
            finish_reason="stop",
        )
        set_output_content_attributes(span, True, [message])
        assert (
            json.loads(span.attributes["gen_ai.output.messages"])[0]["finish_reason"]
            == "stop"
        )

    def test_omits_the_finish_reason_key_when_absent(self) -> None:
        span = FakeSpan()
        set_output_content_attributes(span, True, [text_message("assistant", "hi")])
        assert (
            "finish_reason"
            not in json.loads(span.attributes["gen_ai.output.messages"])[0]
        )


# ─── Part flattening ─────────────────────────────────────────────────────────


class TestPartFlattening:
    def test_tool_call_parts_become_json_in_the_flat_carrier(self) -> None:
        span = FakeSpan()
        message = SpanMessage(
            role="assistant",
            parts=[SpanMessagePart(type="tool_call", name="f", arguments={"a": 1})],
        )
        set_output_content_attributes(span, True, [message])
        assert json.loads(span.attributes["gen_ai.completion.0.content"]) == {
            "name": "f",
            "arguments": {"a": 1},
        }

    def test_reasoning_parts_contribute_their_text(self) -> None:
        span = FakeSpan()
        message = SpanMessage(
            role="assistant",
            parts=[
                SpanMessagePart(type="reasoning", content="thinking"),
                SpanMessagePart(type="text", content="answer"),
            ],
        )
        set_output_content_attributes(span, True, [message])
        assert span.attributes["gen_ai.completion.0.content"] == "thinking\nanswer"

    def test_empty_parts_are_dropped_from_the_join(self) -> None:
        span = FakeSpan()
        message = SpanMessage(
            role="assistant",
            parts=[
                SpanMessagePart(type="text", content=""),
                SpanMessagePart(type="text", content="only this"),
            ],
        )
        set_output_content_attributes(span, True, [message])
        assert span.attributes["gen_ai.completion.0.content"] == "only this"

    def test_a_tool_result_the_provider_never_sent_leaves_the_transcript_empty(
        self,
    ) -> None:
        # Every carrier is written from the same messages so they cannot disagree, and the canonical
        # one omits an absent result. Serialising it would render the literal text `null`, so the
        # trace view showed a tool that returned nothing as one that returned a null value.
        span = FakeSpan()
        message = SpanMessage(
            role="tool",
            parts=[SpanMessagePart(type="tool_call_response", id="c1")],
        )
        set_output_content_attributes(span, True, [message])

        assert span.attributes["gen_ai.completion.0.content"] == ""
        # The carrier the LaunchDarkly reader parses, and the canonical one, agree there is no result.
        assert json.loads(span.attributes["gen_ai.output.messages"]) == [
            {"role": "tool", "parts": [{"type": "tool_call_response", "id": "c1"}]}
        ]

    def test_a_falsy_tool_result_that_is_not_absent_survives(self) -> None:
        # 0, False and "" are results. Only None means the provider sent nothing, which is also how
        # to_canonical reads it.
        span = FakeSpan()
        message = SpanMessage(
            role="tool",
            parts=[SpanMessagePart(type="tool_call_response", id="c1", result=0)],
        )
        set_output_content_attributes(span, True, [message])
        assert span.attributes["gen_ai.completion.0.content"] == "0"


# ─── Tool call arguments and results ─────────────────────────────────────────


class TestToolCallContentAttributes:
    def test_a_string_passes_through_unchanged(self) -> None:
        span = FakeSpan()
        set_tool_call_content_attributes(span, True, result="72F")
        assert span.attributes["gen_ai.tool.call.result"] == "72F"

    def test_anything_else_becomes_json(self) -> None:
        span = FakeSpan()
        set_tool_call_content_attributes(span, True, arguments={"city": "NYC"})
        assert span.attributes["gen_ai.tool.call.arguments"] == '{"city": "NYC"}'

    def test_writes_only_the_side_it_was_given(self) -> None:
        span = FakeSpan()
        set_tool_call_content_attributes(span, True, arguments={"a": 1})
        assert "gen_ai.tool.call.arguments" in span.attributes
        assert "gen_ai.tool.call.result" not in span.attributes

    def test_an_empty_tool_catalog_writes_nothing(self) -> None:
        span = FakeSpan()
        set_tool_definition_attributes(span, True, [])
        assert span.attributes == {}
