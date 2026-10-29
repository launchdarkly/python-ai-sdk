"""Context identity on the root feature_flag span."""

from __future__ import annotations

import json
from typing import Any

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import SDKOptions, configure
from launchdarkly_ai_server.utils import set_ld_span_attributes

LD_FIXTURE = {
    "configKey": "test-config",
    "variationKey": "variation-a",
    "runId": "run-123",
    "version": 1,
    "modelName": "test-model",
    "providerName": "TestProvider",
}


RAW_VARIATION: dict[str, Any] = {
    "model": {"name": "gpt-x", "parameters": {"temperature": 0.5}},
    "provider": {"name": "openai"},
    "messages": [{"role": "system", "content": "Olá"}],
    "_ldMeta": {"enabled": True, "variationKey": "variation-a"},
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

    @pytest.mark.parametrize(
        "context",
        [
            {"key": "u%1:west"},
            {"kind": "user", "key": "u%1:west"},
        ],
        ids=["legacy-user", "explicit-user"],
    )
    def test_user_percent_and_colon_stay_raw_in_all_identity_values(
        self, context: dict[str, Any]
    ) -> None:
        span = FakeSpan()
        set_ld_span_attributes(span, _vars(context))
        event = _feature_flag(span)
        assert event["feature_flag.context.id"] == "u%1:west"
        assert event["feature_flag.contextKeys"] == '{"user":"u%1:west"}'
        assert span.attributes["context.contextKeys.user"] == "u%1:west"

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
        _vars({"kind": "", "key": "u-1"}),
        _vars({"kind": None, "key": "u-1"}),
        _vars({"kind": 123, "key": "u-1"}),
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
        "empty-explicit-kind",
        "none-explicit-kind",
        "number-explicit-kind",
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


class TestResultVariant:
    def test_variant_is_the_variation_key(self) -> None:
        span = FakeSpan()
        set_ld_span_attributes(span, _vars(None))
        assert _feature_flag(span)["feature_flag.result.variant"] == "variation-a"

    @pytest.mark.parametrize("variation_key", ["", None])
    def test_empty_or_missing_variation_key_omits_the_variant(
        self, variation_key: Any
    ) -> None:
        ld = {**LD_FIXTURE, "variationKey": variation_key}
        if variation_key is None:
            del ld["variationKey"]
        span = FakeSpan()
        set_ld_span_attributes(span, {"__ld": ld})
        assert "feature_flag.result.variant" not in _feature_flag(span)

    def test_capture_on_writes_the_raw_variation_like_the_tracing_hook(
        self,
    ) -> None:
        span = FakeSpan()
        set_ld_span_attributes(
            span, {**_vars(None), "__ldValue": RAW_VARIATION}, capture_content=True
        )
        # ldotel's TracingHook writes json.dumps(detail.value) with defaults.
        assert _feature_flag(span)["feature_flag.result.value"] == json.dumps(
            RAW_VARIATION
        )

    def test_capture_off_writes_the_variation_key_not_the_prompt(self) -> None:
        span = FakeSpan()
        set_ld_span_attributes(
            span, {**_vars(None), "__ldValue": RAW_VARIATION}, capture_content=False
        )
        assert _feature_flag(span)["feature_flag.result.value"] == '"variation-a"'

    @pytest.mark.parametrize(("default", "raw"), [(True, True), (False, False)])
    def test_an_unset_capture_follows_the_process_default(
        self, default: bool, raw: bool
    ) -> None:
        """A custom handler that passes no capture_content gets what configure() says."""
        lifecycle_module._reset_for_testing()
        configure(options=SDKOptions(capture_content=default))
        try:
            span = FakeSpan()
            set_ld_span_attributes(span, {**_vars(None), "__ldValue": RAW_VARIATION})
        finally:
            lifecycle_module._reset_for_testing()
        expected = json.dumps(RAW_VARIATION) if raw else '"variation-a"'
        assert _feature_flag(span)["feature_flag.result.value"] == expected

    def test_capture_on_without_a_raw_variation_falls_back_to_the_key(self) -> None:
        span = FakeSpan()
        set_ld_span_attributes(span, _vars(None), capture_content=True)
        assert _feature_flag(span)["feature_flag.result.value"] == '"variation-a"'

    @pytest.mark.parametrize("capture", [True, False])
    def test_no_raw_variation_and_no_key_omits_the_value(self, capture: bool) -> None:
        span = FakeSpan()
        set_ld_span_attributes(
            span,
            {"__ld": {**LD_FIXTURE, "variationKey": ""}},
            capture_content=capture,
        )
        assert "feature_flag.result.value" not in _feature_flag(span)
