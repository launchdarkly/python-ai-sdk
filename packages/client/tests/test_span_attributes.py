"""Emission tests for `set_ld_span_attributes`.

Nothing covered this function before. The cross-handler parity suite in
`tests/test_cross_handler_parity.py` asserts every handler reaches it; these
tests assert what it writes once reached.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from launchdarkly_ai_server.utils import set_ld_span_attributes

LD = {
    "runId": "run-123",
    "configKey": "test-config",
    "variationKey": "variation-a",
    "version": 1,
    "modelName": "test-model",
    "providerName": "TestProvider",
}

MULTI = {
    "kind": "multi",
    "user": {"kind": "user", "key": "u1"},
    "org": {"kind": "org", "key": "o1"},
}


class RecordingSpan:
    """Records what was written, so a test can assert on keys and on absence."""

    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}
        self.events: list[tuple[str, dict[str, Any]]] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any]) -> None:
        self.events.append((name, attributes))

    @property
    def feature_flag_event(self) -> dict[str, Any]:
        return next(attrs for name, attrs in self.events if name == "feature_flag")


def test_the_canonical_context_key_lands_on_the_feature_flag_event() -> None:
    span = RecordingSpan()
    set_ld_span_attributes(
        span, {"__ld": LD, "ldContext": {"kind": "user", "key": "bob"}}
    )
    assert span.feature_flag_event["feature_flag.context.id"] == "bob"


def test_a_multi_kind_context_uses_its_composite_canonical_key() -> None:
    span = RecordingSpan()
    set_ld_span_attributes(span, {"__ld": LD, "ldContext": MULTI})
    assert span.feature_flag_event["feature_flag.context.id"] == "org:o1:user:u1"


def test_the_per_kind_keys_land_on_the_event_as_json() -> None:
    span = RecordingSpan()
    set_ld_span_attributes(span, {"__ld": LD, "ldContext": MULTI})
    assert json.loads(span.feature_flag_event["feature_flag.contextKeys"]) == {
        "org": "o1",
        "user": "u1",
    }


def test_the_json_is_byte_identical_to_what_json_stringify_produces() -> None:
    # `json.dumps` defaults to `", "` / `": "` separators; JSON.stringify uses
    # none. This value lands verbatim in ClickHouse's ContextKeys column, and
    # the js-ai-sdk and browser SDK both write the compact form, so Python must
    # too or the same context yields two different strings by language.
    span = RecordingSpan()
    set_ld_span_attributes(span, {"__ld": LD, "ldContext": MULTI})
    assert (
        span.feature_flag_event["feature_flag.contextKeys"]
        == '{"org":"o1","user":"u1"}'
    )


def test_each_kind_gets_its_own_span_attribute() -> None:
    # The composite canonical key cannot answer "filter to this one user" for a
    # multi-kind context. These can.
    span = RecordingSpan()
    set_ld_span_attributes(span, {"__ld": LD, "ldContext": MULTI})
    assert span.attributes["context.contextKeys.user"] == "u1"
    assert span.attributes["context.contextKeys.org"] == "o1"


def test_no_context_attributes_without_an_ld_context() -> None:
    span = RecordingSpan()
    set_ld_span_attributes(span, {"__ld": LD})
    assert "feature_flag.context.id" not in span.feature_flag_event
    assert "feature_flag.contextKeys" not in span.feature_flag_event
    assert [k for k in span.attributes if k.startswith("context.")] == []


@pytest.mark.parametrize("bad", ["bob", 42, None, {}, {"kind": "multi", "user": {}}])
def test_a_malformed_ld_context_emits_nothing_and_does_not_raise(bad: Any) -> None:
    span = RecordingSpan()
    set_ld_span_attributes(span, {"__ld": LD, "ldContext": bad})
    assert "feature_flag.context.id" not in span.feature_flag_event
    assert [k for k in span.attributes if k.startswith("context.")] == []


def test_only_keys_are_emitted_never_attribute_values() -> None:
    # AC 5: attribute values are where the PII lives. Nothing but keys leaves
    # the SDK, and no option exists to change that.
    span = RecordingSpan()
    set_ld_span_attributes(
        span,
        {
            "__ld": LD,
            "ldContext": {
                "kind": "user",
                "key": "bob",
                "email": "bob@example.com",
                "name": "Bob",
            },
        },
    )
    emitted = json.dumps([span.attributes, span.events])
    assert "bob@example.com" not in emitted
    assert "Bob" not in emitted
