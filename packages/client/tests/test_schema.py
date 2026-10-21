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
    ``skills`` is passed through unvalidated.

    Agent Skills is experimental, so a malformed ``skills`` field must not fail
    a core config call (TESTING.md §0.3). ``skill_refs`` validates the field
    where the references are used; see ``TestSkillRefs`` in test_skills.py.
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

    def test_valid_entries_pass_through(self) -> None:
        raw = self._base(skills=[{"key": "pdf-extraction", "version": 2}])
        result = parse_ai_config(raw)
        assert result.success is True
        assert result.data["skills"] == [{"key": "pdf-extraction", "version": 2}]

    @pytest.mark.parametrize(
        "malformed",
        [
            None,
            "pdf",
            {"key": "a"},
            3,
            ["pdf-extraction"],
            [{"version": 1}],
            [{"key": "Evil", "version": 1}],
            [{"key": "../escape", "version": 1}],
            [{"key": "a", "version": 0}],
            [{"key": "a"}],
            [{"key": "good", "version": 1}, {"key": "My_Skill", "version": 1}],
        ],
    )
    def test_a_malformed_skills_field_does_not_fail_the_parse(
        self, malformed: Any
    ) -> None:
        result = parse_ai_config(self._base(skills=malformed))
        assert result.success is True
        assert result.data["skills"] == malformed
