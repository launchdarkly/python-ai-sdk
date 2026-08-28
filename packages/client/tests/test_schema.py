"""Tests for ``parse_ai_config`` — AI Config variation validation."""

from typing import Any

import pytest

from launchdarkly_ai_server import parse_ai_config


class TestParseAiConfig:
    def _valid_base(self) -> dict:
        return {
            "model": {"name": "claude-3"},
            "provider": {"name": "Anthropic"},
            "instructions": "You are helpful.",
        }

    def test_valid_with_instructions(self) -> None:
        result = parse_ai_config(self._valid_base())
        assert result.success is True

    def test_valid_with_messages(self) -> None:
        raw = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "OpenAI"},
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = parse_ai_config(raw)
        assert result.success is True

    def test_fails_with_neither(self) -> None:
        raw = {"model": {"name": "gpt-4"}, "provider": {"name": "OpenAI"}}
        result = parse_ai_config(raw)
        assert result.success is False

    def test_fails_with_empty_messages(self) -> None:
        raw = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "OpenAI"},
            "messages": [],
        }
        result = parse_ai_config(raw)
        assert result.success is False

    def test_messages_with_wrong_role_fails(self) -> None:
        raw = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "OpenAI"},
            "messages": [{"role": "admin", "content": "bad"}],
        }
        result = parse_ai_config(raw)
        assert result.success is False

    def test_missing_model_name_fails(self) -> None:
        raw = {"model": {}, "provider": {"name": "OpenAI"}, "instructions": "hi"}
        result = parse_ai_config(raw)
        assert result.success is False

    def test_missing_provider_fails(self) -> None:
        raw = {"model": {"name": "gpt-4"}, "instructions": "hi"}
        result = parse_ai_config(raw)
        assert result.success is False

    def test_optional_fields_accepted(self) -> None:
        raw = self._valid_base()
        raw["tools"] = {
            "search": {"name": "search", "type": "function", "parameters": {}}
        }
        raw["judgeConfiguration"] = {"judges": [{"key": "j1", "samplingRate": 1.0}]}
        raw["evaluationMetricKey"] = "my-metric"
        result = parse_ai_config(raw)
        assert result.success is True

    def test_tool_with_wrong_type_fails(self) -> None:
        raw = self._valid_base()
        raw["tools"] = {"bad": {"name": "bad", "type": "class", "parameters": {}}}
        result = parse_ai_config(raw)
        assert result.success is False

    def test_output_format_accepted(self) -> None:
        raw = self._valid_base()
        raw["outputFormat"] = {"type": "object", "properties": {}}
        result = parse_ai_config(raw)
        assert result.success is True


class TestParseAiConfigSkills:
    """
    Fail-closed validation of the optional ``skills`` array.
    """

    def _base(self, **extra: Any) -> dict[str, Any]:
        raw: dict[str, Any] = {
            "model": {"name": "claude-3"},
            "provider": {"name": "Anthropic"},
            "instructions": "You are helpful.",
        }
        raw.update(extra)
        return raw

    def test_absent_skills_is_valid(self) -> None:
        assert parse_ai_config(self._base()).success is True

    def test_empty_skills_is_valid(self) -> None:
        assert parse_ai_config(self._base(skills=[])).success is True

    def test_valid_entries_accepted(self) -> None:
        raw = self._base(skills=[{"key": "pdf-extraction", "version": 2}])
        result = parse_ai_config(raw)
        assert result.success is True
        assert result.data["skills"] == [{"key": "pdf-extraction", "version": 2}]

    def test_multiple_valid_entries_accepted(self) -> None:
        raw = self._base(
            skills=[{"key": "a", "version": 1}, {"key": "b-2", "version": 10}]
        )
        assert parse_ai_config(raw).success is True

    def test_key_at_length_bound_accepted(self) -> None:
        raw = self._base(skills=[{"key": "a" * 256, "version": 1}])
        assert parse_ai_config(raw).success is True

    @pytest.mark.parametrize("bad_skills", ["pdf", {"key": "a"}, 3, True])
    def test_non_array_skills_fails(self, bad_skills: Any) -> None:
        assert parse_ai_config(self._base(skills=bad_skills)).success is False

    @pytest.mark.parametrize("entry", ["pdf-extraction", 1, None, ["a", 1]])
    def test_non_object_entry_fails(self, entry: Any) -> None:
        assert parse_ai_config(self._base(skills=[entry])).success is False

    @pytest.mark.parametrize("bad_key", [None, 1, True, {"a": 1}, ["a"]])
    def test_missing_or_non_string_key_fails(self, bad_key: Any) -> None:
        raw = self._base(skills=[{"key": bad_key, "version": 1}])
        assert parse_ai_config(raw).success is False

    def test_absent_key_fails(self) -> None:
        assert parse_ai_config(self._base(skills=[{"version": 1}])).success is False

    @pytest.mark.parametrize(
        "bad_key",
        [
            "",
            "Evil",
            "-leading-dash",
            ".hidden",
            "_underscore",
            "has space",
            "a/b",
            "a\\b",
            "../escape",
            "trailing-space ",
            "under_score",
            "a" * 257,
        ],
    )
    def test_pattern_and_length_violations_fail(self, bad_key: str) -> None:
        raw = self._base(skills=[{"key": bad_key, "version": 1}])
        assert parse_ai_config(raw).success is False

    @pytest.mark.parametrize("bad_version", [0, -1, 2.5, "2", None, True, [1]])
    def test_invalid_version_fails(self, bad_version: Any) -> None:
        raw = self._base(skills=[{"key": "a", "version": bad_version}])
        assert parse_ai_config(raw).success is False

    def test_absent_version_fails(self) -> None:
        assert parse_ai_config(self._base(skills=[{"key": "a"}])).success is False

    def test_one_bad_entry_fails_the_whole_config(self) -> None:
        raw = self._base(
            skills=[{"key": "good", "version": 1}, {"key": "../bad", "version": 1}]
        )
        assert parse_ai_config(raw).success is False

    def test_error_message_mentions_skills(self) -> None:
        raw = self._base(skills=[{"key": "../bad", "version": 1}])
        result = parse_ai_config(raw)
        assert result.success is False
        assert "skills" in result.error["message"]
