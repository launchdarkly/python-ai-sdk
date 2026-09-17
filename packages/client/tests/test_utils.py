"""
Tests for parse_template, parse_json_with_possible_fences,
parse_usage, normalize_mode, create_handler.
Reference: TESTING.md s3.1-3.4, s3.15
"""

from typing import Any

import pytest

from launchdarkly_ai_server import (
    create_handler,
    make_track_data,
    model_stamps_from_meta,
    normalize_mode,
    omit_model_stamps,
    parse_json_with_possible_fences,
    parse_template,
    parse_usage,
)

# ---------------------------------------------------------------------------
# ?3.1 parse_template
# ---------------------------------------------------------------------------


class TestParseTemplate:
    def test_simple_substitution(self) -> None:
        assert parse_template("Hello {{name}}", {"name": "world"}) == "Hello world"

    def test_multiple_placeholders(self) -> None:
        result = parse_template(
            "{{greeting}} {{name}}", {"greeting": "Hi", "name": "Alice"}
        )
        assert result == "Hi Alice"

    def test_unknown_placeholder_preserved(self) -> None:
        result = parse_template("Hello {{missing}}", {})
        assert result == "Hello {{missing}}"

    def test_dot_notation_access(self) -> None:
        result = parse_template("Hi {{user.name}}", {"user": {"name": "Alice"}})
        assert result == "Hi Alice"

    def test_partial_dot_notation_miss(self) -> None:
        result = parse_template("{{user.missing}}", {"user": {}})
        assert result == "{{user.missing}}"

    def test_deeply_nested_value(self) -> None:
        result = parse_template("{{a.b.c}}", {"a": {"b": {"c": 42}}})
        assert result == "42"

    def test_non_string_value_coerced(self) -> None:
        assert parse_template("{{n}}", {"n": 123}) == "123"
        assert parse_template("{{b}}", {"b": True}) == "True"

    def test_empty_variables_map(self) -> None:
        result = parse_template("Hello {{name}}", {})
        assert result == "Hello {{name}}"

    def test_no_placeholders_in_template(self) -> None:
        result = parse_template("Hello world", {"name": "ignored"})
        assert result == "Hello world"


# ---------------------------------------------------------------------------
# ?3.2 parse_json_with_possible_fences
# ---------------------------------------------------------------------------


class TestParseJsonWithPossibleFences:
    def test_plain_json(self) -> None:
        assert parse_json_with_possible_fences('{"a":1}') == {"a": 1}

    def test_fenced_with_json_tag(self) -> None:
        result = parse_json_with_possible_fences('```json\n{"a":1}\n```')
        assert result == {"a": 1}

    def test_fenced_with_bare_backticks(self) -> None:
        result = parse_json_with_possible_fences('```\n{"a":1}\n```')
        assert result == {"a": 1}

    def test_whitespace_around_fences(self) -> None:
        result = parse_json_with_possible_fences('```json\n\n{"a":1}\n\n```')
        assert result == {"a": 1}

    def test_invalid_json_returns_none(self) -> None:
        assert parse_json_with_possible_fences("not json") is None

    def test_invalid_json_no_error_log(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        parse_json_with_possible_fences("not json")
        captured = capsys.readouterr()
        assert "Error" not in captured.err
        assert "error" not in captured.err

    def test_nested_json_objects_and_arrays(self) -> None:
        data = {"a": [1, {"b": True}]}
        import json

        result = parse_json_with_possible_fences(json.dumps(data))
        assert result == data

    def test_windows_crlf_json_fence(self) -> None:
        result = parse_json_with_possible_fences('```json\r\n{"a":1}\r\n```')
        assert result == {"a": 1}

    def test_windows_crlf_bare_fence(self) -> None:
        result = parse_json_with_possible_fences('```\r\n{"a":1}\r\n```')
        assert result == {"a": 1}

    def test_leading_whitespace_before_fence(self) -> None:
        result = parse_json_with_possible_fences('  ```json\n{"a":1}\n```')
        assert result == {"a": 1}


# ---------------------------------------------------------------------------
# ?3.3 parse_usage
# ---------------------------------------------------------------------------


class TestParseUsage:
    def test_input_tokens_output_tokens(self) -> None:
        result = parse_usage({"input_tokens": 10, "output_tokens": 5})
        assert result == {"input": 10, "output": 5, "total": 15}

    def test_camel_case_keys(self) -> None:
        result = parse_usage({"inputTokens": 10, "outputTokens": 5})
        assert result == {"input": 10, "output": 5, "total": 15}

    def test_short_keys(self) -> None:
        result = parse_usage({"input": 10, "output": 5})
        assert result == {"input": 10, "output": 5, "total": 15}

    def test_total_is_computed(self) -> None:
        result = parse_usage({"input_tokens": 3, "output_tokens": 7})
        assert result["total"] == 10

    def test_unknown_keys_returns_zeros(self) -> None:
        result = parse_usage({"foo": 1, "bar": 2})
        assert result == {"input": 0, "output": 0, "total": 0}

    def test_first_matching_pair_wins(self) -> None:
        result = parse_usage(
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "inputTokens": 99,
                "outputTokens": 99,
            }
        )
        assert result["input"] == 10
        assert result["output"] == 5


# ---------------------------------------------------------------------------
# ?3.4 normalize_mode
# ---------------------------------------------------------------------------


class TestNormalizeMode:
    def test_agent_stays_agent(self) -> None:
        assert normalize_mode("agent") == "agent"

    def test_completion_becomes_messages(self) -> None:
        assert normalize_mode("completion") == "messages"

    def test_judge_becomes_messages(self) -> None:
        assert normalize_mode("judge") == "messages"

    def test_none_becomes_messages(self) -> None:
        assert normalize_mode(None) == "messages"

    def test_unrecognized_string_becomes_messages(self) -> None:
        assert normalize_mode("unknown") == "messages"


# ---------------------------------------------------------------------------
# ?3.15 create_handler
# ---------------------------------------------------------------------------


class TestCreateHandler:
    def test_attaches_provides_for(self) -> None:
        async def fn(
            config: object,
            user_input: object,
            tool_handlers: object,
            variables: object,
            history: object = None,
        ) -> dict:  # type: ignore[override]
            return {"output": "ok"}

        h = create_handler(("MyProvider", "messages"), fn)
        assert h.provides_for == ("MyProvider", "messages")

    def test_returns_callable(self) -> None:
        async def fn(
            config: object,
            user_input: object,
            tool_handlers: object,
            variables: object,
            history: object = None,
        ) -> dict:  # type: ignore[override]
            return {"output": "ok"}

        h = create_handler(("MyProvider", "messages"), fn)
        assert callable(h)

    async def test_callable_behaves_identically(self) -> None:
        async def fn(
            config: object,
            user_input: object,
            tool_handlers: object,
            variables: object,
            history: object = None,
        ) -> dict:  # type: ignore[override]
            return {"output": "test-output"}

        h = create_handler(("MyProvider", "messages"), fn)
        result = await h({}, "hi", {}, {})  # type: ignore[arg-type]
        assert result["output"] == "test-output"

    def test_works_with_agent_mode(self) -> None:
        async def fn(
            config: object,
            user_input: object,
            tool_handlers: object,
            variables: object,
            history: object = None,
        ) -> dict:  # type: ignore[override]
            return {}

        h = create_handler(("MyProvider", "agent"), fn)
        assert h.provides_for == ("MyProvider", "agent")

    def test_works_with_messages_mode(self) -> None:
        async def fn(
            config: object,
            user_input: object,
            tool_handlers: object,
            variables: object,
            history: object = None,
        ) -> dict:  # type: ignore[override]
            return {}

        h = create_handler(("MyProvider", "messages"), fn)
        assert h.provides_for == ("MyProvider", "messages")

    def test_returned_handler_wraps_original_callable(self) -> None:
        """In Python, create_handler returns a ProviderHandler wrapper (a callable class)
        that holds the original function. The returned object is callable and callable
        invocation reaches the original function. This is the Python adaptation of the
        TS 'same callable reference' spec."""

        async def fn(
            config: object,
            user_input: object,
            tool_handlers: object,
            variables: object,
            history: object = None,
        ) -> dict:  # type: ignore[override]
            return {"output": "original"}

        h = create_handler(("MyProvider", "messages"), fn)
        assert callable(h)
        # The original function is accessible via _fn attribute
        assert h._fn is fn


class TestMakeTrackData:
    """§3.10 model stamps — shared node-trackData builder for native graph adapters."""

    def _node(self, meta: dict) -> Any:
        from launchdarkly_ai_server.types import GraphNode

        return GraphNode(
            key="node-a",
            config={"model": {"name": "gpt-4"}, "provider": {"name": "OpenAI"}},
            meta=meta,
        )

    def test_copies_model_key_and_version_from_meta(self) -> None:
        td = make_track_data(
            self._node(
                {
                    "variationKey": "v1",
                    "version": 1,
                    "modelKey": "my-model",
                    "modelVersion": 3,
                }
            ),
            "graph-key",
            "run-1",
        )
        assert td["modelKey"] == "my-model"
        assert td["modelVersion"] == 3
        assert td["graphKey"] == "graph-key"

    def test_omits_model_key_and_version_when_absent(self) -> None:
        td = make_track_data(
            self._node({"variationKey": "v1", "version": 1}), "graph-key", "run-1"
        )
        assert "modelKey" not in td
        assert "modelVersion" not in td


class TestModelStampsFromMeta:
    """§3.10 model stamps — malformed ``modelVersion`` is omitted, never raises."""

    def test_copies_int_version_and_non_empty_key(self) -> None:
        assert model_stamps_from_meta({"modelKey": "m", "modelVersion": 3}) == {
            "modelKey": "m",
            "modelVersion": 3,
        }

    @pytest.mark.parametrize("value", ["3", 3.0])
    def test_coerces_integral_string_and_float(self, value: Any) -> None:
        assert model_stamps_from_meta({"modelVersion": value}) == {"modelVersion": 3}

    @pytest.mark.parametrize(
        "value", ["abc", "1.5", 1.5, {}, [], None, True, False, float("nan")]
    )
    def test_omits_malformed_version_without_raising(self, value: Any) -> None:
        stamps = model_stamps_from_meta({"modelVersion": value, "modelKey": "m"})
        assert "modelVersion" not in stamps
        assert stamps["modelKey"] == "m"

    def test_non_dict_meta_returns_empty(self) -> None:
        assert model_stamps_from_meta(None) == {}
        assert model_stamps_from_meta("nope") == {}


class TestOmitModelStamps:
    def test_removes_only_the_two_stamp_keys(self) -> None:
        td = {"runId": "r", "graphKey": "g", "modelKey": "m", "modelVersion": 1}
        assert omit_model_stamps(td) == {"runId": "r", "graphKey": "g"}

    def test_does_not_mutate_input(self) -> None:
        td = {"runId": "r", "modelKey": "m"}
        omit_model_stamps(td)
        assert td == {"runId": "r", "modelKey": "m"}
