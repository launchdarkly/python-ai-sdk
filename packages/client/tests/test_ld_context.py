"""The Python port must agree with the TypeScript port, key for key.

The fixtures here are the same ones in js-ai-sdk's
`packages/client/src/__tests__/context.test.ts`, which are in turn the
observability browser SDK's. A canonical key that differs between emitters
breaks context-instance linking, and nothing else would catch it.
"""

from __future__ import annotations

from typing import Any

import pytest

from launchdarkly_ai_server.ld_context import (
    context_identity,
    get_canonical_key,
    get_context_keys,
)


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        ({"key": "bob"}, {"user": "bob"}),
        ({"kind": "user", "key": "bob"}, {"user": "bob"}),
        ({"kind": "org", "key": "org123"}, {"org": "org123"}),
        ({"kind": "device", "key": "device456"}, {"device": "device456"}),
        (
            {
                "kind": "multi",
                "user": {"kind": "user", "key": "user-key", "name": "Test User"},
                "org": {"kind": "org", "key": "org-key"},
            },
            {"org": "org-key", "user": "user-key"},
        ),
        (
            {
                "kind": "multi",
                "device": {"kind": "device", "key": "device-key"},
                "user": {"kind": "user", "key": "user-key"},
            },
            {"device": "device-key", "user": "user-key"},
        ),
    ],
)
def test_get_context_keys(context: dict[str, Any], expected: dict[str, str]) -> None:
    assert get_context_keys(context) == expected


def test_get_context_keys_does_not_escape_the_key() -> None:
    # Only the canonical key is escaped. The map holds the key the customer
    # actually sent, because that is what a filter compares against.
    assert get_context_keys({"kind": "org", "key": "a:b%c"}) == {"org": "a:b%c"}


def test_get_context_keys_skips_a_multi_kind_entry_with_no_usable_key() -> None:
    context = {"kind": "multi", "user": {"kind": "user", "key": "bob"}, "org": {}}
    assert get_context_keys(context) == {"user": "bob"}


def test_get_context_keys_is_empty_without_a_key() -> None:
    assert get_context_keys({}) == {}


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        ({"key": "bob"}, "bob"),
        ({"kind": "user", "key": "bob"}, "bob"),
        ({"kind": "org", "key": "org123"}, "org:org123"),
        (
            {
                "kind": "multi",
                "user": {"kind": "user", "key": "user-key"},
                "org": {"kind": "org", "key": "org-key"},
            },
            "org:org-key:user:user-key",
        ),
        (
            {
                "kind": "multi",
                "device": {"kind": "device", "key": "device-key"},
                "user": {"kind": "user", "key": "user-key"},
            },
            "device:device-key:user:user-key",
        ),
    ],
)
def test_get_canonical_key(context: dict[str, Any], expected: str) -> None:
    assert get_canonical_key(context) == expected


def test_get_canonical_key_escapes_percent_before_colon() -> None:
    # `%` first, then `:`, so an escape sequence is never double-escaped.
    assert get_canonical_key({"kind": "org", "key": "a:b%c"}) == "org:a%3Ab%25c"


def test_get_canonical_key_is_empty_without_a_key() -> None:
    assert get_canonical_key({}) == ""


def test_context_identity_returns_the_canonical_key_and_the_per_kind_keys() -> None:
    context = {
        "kind": "multi",
        "user": {"kind": "user", "key": "u1"},
        "org": {"kind": "org", "key": "o1"},
    }
    assert context_identity(context) == ("org:o1:user:u1", {"org": "o1", "user": "u1"})


@pytest.mark.parametrize(
    "context",
    [
        None,
        "user-key",
        42,
        {},
        {"kind": "user", "key": 42},
        {"kind": "multi", "user": {}},
    ],
)
def test_context_identity_is_none_for_anything_unusable(context: Any) -> None:
    assert context_identity(context) is None
