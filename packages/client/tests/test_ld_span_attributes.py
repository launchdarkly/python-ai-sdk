"""Context identity on the root feature_flag span. TESTING.md §3.18."""

from __future__ import annotations

from typing import Any

import pytest

from launchdarkly_ai_server.utils import set_ld_span_attributes

LD_FIXTURE = {
    "configKey": "test-config",
    "variationKey": "variation-a",
    "runId": "run-123",
    "version": 1,
    "modelName": "test-model",
    "providerName": "TestProvider",
}


class FakeSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}
        self.events: list[tuple[str, dict[str, Any]]] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append((name, attributes or {}))


def _vars(ld_context: Any) -> dict[str, Any]:
    return {"__ld": LD_FIXTURE, "ldContext": ld_context}


def _feature_flag(span: FakeSpan) -> dict[str, Any]:
    for name, attrs in span.events:
        if name == "feature_flag":
            return attrs
    return {}


def _assert_no_context_identity(span: FakeSpan) -> None:
    assert [k for k in span.attributes if k.startswith("context.contextKeys.")] == []
    event = _feature_flag(span)
    assert "feature_flag.context.id" not in event
    assert "feature_flag.contextKeys" not in event
    assert "feature_flag.context.key.user" not in event


class TestContextIdentity:
    def test_legacy_user_with_no_kind_is_the_bare_key(self) -> None:
        span = FakeSpan()
        set_ld_span_attributes(span, _vars({"key": "u-1"}))
        event = _feature_flag(span)
        assert event["feature_flag.context.id"] == "u-1"
        assert event["feature_flag.contextKeys"] == '{"user":"u-1"}'
        assert span.attributes["context.contextKeys.user"] == "u-1"

    def test_kind_user_is_the_bare_key(self) -> None:
        span = FakeSpan()
        set_ld_span_attributes(span, _vars({"kind": "user", "key": "u-1"}))
        event = _feature_flag(span)
        assert event["feature_flag.context.id"] == "u-1"
        assert event["feature_flag.contextKeys"] == '{"user":"u-1"}'
        assert span.attributes["context.contextKeys.user"] == "u-1"

    def test_non_user_single_kind_is_prefixed(self) -> None:
        span = FakeSpan()
        set_ld_span_attributes(span, _vars({"kind": "org", "key": "o-1"}))
        event = _feature_flag(span)
        assert event["feature_flag.context.id"] == "org:o-1"
        assert event["feature_flag.contextKeys"] == '{"org":"o-1"}'
        assert span.attributes["context.contextKeys.org"] == "o-1"

    def test_multi_kind_is_sorted_by_kind_not_declaration_order(self) -> None:
        # user before org is the reverse of sorted order (org < user).
        span = FakeSpan()
        set_ld_span_attributes(
            span,
            _vars({"kind": "multi", "user": {"key": "u-1"}, "org": {"key": "o-1"}}),
        )
        event = _feature_flag(span)
        assert event["feature_flag.context.id"] == "org:o-1:user:u-1"
        assert event["feature_flag.contextKeys"] == '{"org":"o-1","user":"u-1"}'
        assert span.attributes["context.contextKeys.user"] == "u-1"
        assert span.attributes["context.contextKeys.org"] == "o-1"

    def test_percent_is_escaped_before_colon_and_the_map_stays_raw(self) -> None:
        span = FakeSpan()
        set_ld_span_attributes(span, _vars({"kind": "org", "key": "a%b:c"}))
        event = _feature_flag(span)
        assert event["feature_flag.context.id"] == "org:a%25b%3Ac"
        assert event["feature_flag.contextKeys"] == '{"org":"a%b:c"}'
        assert span.attributes["context.contextKeys.org"] == "a%b:c"

    def test_non_ascii_key_is_not_unicode_escaped(self) -> None:
        span = FakeSpan()
        set_ld_span_attributes(span, _vars({"kind": "user", "key": "José"}))
        event = _feature_flag(span)
        assert event["feature_flag.context.id"] == "José"
        assert event["feature_flag.contextKeys"] == '{"user":"José"}'
        assert span.attributes["context.contextKeys.user"] == "José"

    def test_integer_like_kinds_are_lexicographic_not_json_index_order(self) -> None:
        span = FakeSpan()
        set_ld_span_attributes(
            span,
            _vars({"kind": "multi", "2": {"key": "b"}, "10": {"key": "a"}}),
        )
        event = _feature_flag(span)
        assert event["feature_flag.context.id"] == "10:a:2:b"
        assert event["feature_flag.contextKeys"] == '{"10":"a","2":"b"}'
        assert span.attributes["context.contextKeys.10"] == "a"
        assert span.attributes["context.contextKeys.2"] == "b"

    def test_multi_kind_with_one_usable_pair_keeps_the_prefixed_form(self) -> None:
        span = FakeSpan()
        set_ld_span_attributes(
            span,
            _vars({"kind": "multi", "user": {"key": "u-1"}, "org": {}}),
        )
        event = _feature_flag(span)
        assert event["feature_flag.context.id"] == "user:u-1"
        assert event["feature_flag.contextKeys"] == '{"user":"u-1"}'
        assert span.attributes["context.contextKeys.user"] == "u-1"
        assert "context.contextKeys.org" not in span.attributes

    def test_emits_keys_only_never_context_attribute_values(self) -> None:
        span = FakeSpan()
        set_ld_span_attributes(
            span,
            _vars({"key": "u-1", "email": "ada@example.com", "name": "Ada"}),
        )
        values = [*span.attributes.values(), *_feature_flag(span).values()]
        assert "u-1" in values
        assert "ada@example.com" not in values
        assert "Ada" not in values


@pytest.mark.parametrize(
    "variables",
    [
        {"__ld": LD_FIXTURE},
        _vars(None),
        _vars("user-123"),
        _vars(123),
        _vars({}),
        _vars({"kind": "user", "key": 123}),
        _vars({"kind": "user", "key": ""}),
        _vars({"kind": "multi"}),
        _vars({"kind": "multi", "user": {"name": "Ada"}}),
    ],
    ids=[
        "missing",
        "none",
        "string",
        "number",
        "empty-object",
        "non-string-key",
        "empty-key",
        "empty-multi",
        "multi-with-no-usable-key",
    ],
)
def test_malformed_ld_context_emits_none_of_the_three_and_does_not_throw(
    variables: dict[str, Any],
) -> None:
    span = FakeSpan()
    set_ld_span_attributes(span, variables)
    _assert_no_context_identity(span)
