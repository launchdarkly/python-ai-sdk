"""
Tests for §3.5 parse_ai_config (AiConfig validation).
Reference: TESTING.md §3.5
"""

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
