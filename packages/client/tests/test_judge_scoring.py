from __future__ import annotations

from typing import Any

import pytest

from launchdarkly_ai_server.judge_scoring import parse_judge_response


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
