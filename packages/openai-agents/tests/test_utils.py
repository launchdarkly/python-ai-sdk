"""
Tests for §2.x.5 build_output_type utility.
Reference: TESTING.md §2.x.5
"""

from launchdarkly_ai_openai_agents.utils import build_output_type


class TestBuildOutputType:
    def test_returns_none_when_output_format_none(self) -> None:
        assert build_output_type(None) is None

    def test_returns_none_when_output_format_empty_object(self) -> None:
        assert build_output_type({}) is None

    def test_wraps_schema_in_correct_envelope(self) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = build_output_type(schema)
        assert result is not None
        assert result["type"] == "json_schema"
        assert "schema" in result
        assert result["schema"]["properties"]["x"]["type"] == "string"

    def test_ensures_type_object_in_schema(self) -> None:
        schema = {"properties": {"name": {"type": "string"}}}
        result = build_output_type(schema)
        assert result is not None
        assert result["schema"]["type"] == "object"

    def test_preserves_existing_type_object(self) -> None:
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        result = build_output_type(schema)
        assert result is not None
        assert result["schema"]["type"] == "object"

    def test_name_is_output_and_strict_is_false(self) -> None:
        schema = {"type": "object", "properties": {"v": {"type": "number"}}}
        result = build_output_type(schema)
        assert result is not None
        assert result["name"] == "output"
        assert result["strict"] is False

    def test_required_populated_from_all_property_keys(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        }
        result = build_output_type(schema)
        assert result is not None
        required = result["schema"]["required"]
        assert "a" in required
        assert "b" in required

    def test_required_omitted_when_properties_empty(self) -> None:
        """§2.x.5 — required must not appear in schema when properties is empty.

        When ``outputFormat`` has a ``properties`` key that maps to ``{}``,
        the resulting schema must not include ``"required": []``. An empty
        required list is meaningless and may confuse provider validators.
        The condition must be ``if required`` (falsy check) not ``if required is not None``.
        """
        schema = {"type": "object", "properties": {}}
        result = build_output_type(schema)
        assert result is not None
        assert "required" not in result["schema"], (
            "Schema must not include 'required: []' when properties is empty. "
            "Use 'if required' not 'if required is not None'."
        )
