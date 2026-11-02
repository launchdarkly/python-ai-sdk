from __future__ import annotations

from typing import Any

import pytest

from launchdarkly_ai_server.judge_scoring import (
    FORMATTING_INSTRUCTIONS,
    build_message_history,
    judge_text,
    parse_judge_response,
    typesafe_judge_entries,
)


class TestParseJudgeResponse:
    def test_parses_plain_json(self) -> None:
        assert parse_judge_response('{"score": 0.9, "reasoning": "solid"}') == (
            0.9,
            "solid",
        )

    def test_parses_fenced_json(self) -> None:
        raw = '```json\n{"score": 1, "reasoning": "ok"}\n```'
        assert parse_judge_response(raw) == (1, "ok")

    def test_accepts_already_decoded_mapping(self) -> None:
        assert parse_judge_response({"score": 0.5, "reasoning": "meh"}) == (
            0.5,
            "meh",
        )

    def test_falls_back_to_reason_key(self) -> None:
        assert parse_judge_response({"score": 0.5, "reason": "alt key"}) == (
            0.5,
            "alt key",
        )

    def test_null_reasoning_becomes_empty_string_not_none_literal(self) -> None:
        assert parse_judge_response({"score": 0.5, "reasoning": None}) == (0.5, "")

    def test_non_string_reasoning_is_json_not_a_repr(self) -> None:
        assert parse_judge_response({"score": 1, "reasoning": {"ok": True}}) == (
            1,
            '{"ok":true}',
        )
        assert parse_judge_response({"score": 1, "reasoning": 0}) == (1, "0")

    @pytest.mark.parametrize("empty", [None, "", [], {}])
    def test_empty_reasoning_falls_back_to_reason(self, empty: Any) -> None:
        assert parse_judge_response(
            {"score": 1, "reasoning": empty, "reason": "fallback"}
        ) == (1, "fallback")

    def test_score_returned_untouched_for_caller_policy(self) -> None:
        score, _ = parse_judge_response({"score": "high", "reasoning": "?"})
        assert score == "high"

    @pytest.mark.parametrize(
        "raw",
        [
            "the answer looks correct",
            "{}",
            {},
            None,
            42,
            ["not", "a", "mapping"],
            '["not", "a", "mapping"]',
        ],
    )
    def test_rejects_non_object_responses(self, raw: Any) -> None:
        with pytest.raises(ValueError, match="Invalid JSON from judge"):
            parse_judge_response(raw)


def test_typesafe_entries_leave_classic_judges_alone() -> None:
    assert typesafe_judge_entries('{"score": 0.9, "reasoning": "solid"}') is None


def test_typesafe_entries_keep_label_order() -> None:
    raw = (
        '{"kind":"typesafe","results":['
        '{"key":"migration_intent","eventKey":"$ld:ai:judge:jev:migration_intent","score":0.25},'
        '{"key":"migration","eventKey":"$ld:ai:judge:jev:migration","score":0.9}]}'
    )
    assert typesafe_judge_entries(raw) == [
        {
            "key": "migration_intent",
            "score": 0.25,
            "eventKey": "$ld:ai:judge:jev:migration_intent",
            "reason": "",
        },
        {
            "key": "migration",
            "score": 0.9,
            "eventKey": "$ld:ai:judge:jev:migration",
            "reason": "",
        },
    ]


def test_message_history_renders_a_structured_output_as_json() -> None:
    history = build_message_history(
        user_input="q", output={"answer": "oui", "ok": True, "note": None}
    )
    assert history == "\n\n".join(
        ["q", '{"answer":"oui","ok":true,"note":null}', FORMATTING_INSTRUCTIONS]
    )


def test_message_history_keeps_non_ascii() -> None:
    assert '{"city":"Zürich"}' in build_message_history(output={"city": "Zürich"})


@pytest.mark.parametrize("empty", [None, "", {}, []])
def test_message_history_skips_an_empty_output(empty: Any) -> None:
    # Tested before rendering, so an empty container adds no block.
    assert build_message_history(user_input="q", output=empty) == "\n\n".join(
        ["q", FORMATTING_INSTRUCTIONS]
    )


@pytest.mark.parametrize(("output", "text"), [(0, "0"), (False, "false")])
def test_message_history_shows_a_falsy_answer(output: Any, text: str) -> None:
    # Each is an answer the judge has to grade, not an empty part.
    assert build_message_history(user_input="q", output=output) == "\n\n".join(
        ["q", text, FORMATTING_INSTRUCTIONS]
    )


# Each expected string is what JavaScript's JSON.stringify writes for the same value.
@pytest.mark.parametrize(
    ("value", "text"),
    [
        ("as is", "as is"),
        (None, ""),
        (0, "0"),
        (False, "false"),
        (1.0, "1"),
        (-2.0, "-2"),
        (0.5, "0.5"),
        (1e21, "1e+21"),
        ([1.0, "a", None], '[1,"a",null]'),
        ({"b": 1.0, "a": [True, {}]}, '{"b":1,"a":[true,{}]}'),
        ({"city": "Zürich"}, '{"city":"Zürich"}'),
        ({"n": float("nan")}, '{"n":null}'),
    ],
)
def test_judge_text_is_json_stringify_text(value: Any, text: str) -> None:
    assert judge_text(value) == text
