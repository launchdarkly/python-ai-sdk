"""RFC 8785 canonical JSON. Each expected string is what JavaScript's JSON.stringify writes."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from launchdarkly_ai_server.canonical_json import canonical_json


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (1.0, "1"),
        (-2.0, "-2"),
        (-0.0, "0"),
        (0.1, "0.1"),
        (1 / 3, "0.3333333333333333"),
        (1e16, "10000000000000000"),
        (1e20, "100000000000000000000"),
        (1e21, "1e+21"),
        (1.5e300, "1.5e+300"),
        (1e-6, "0.000001"),
        (1e-7, "1e-7"),
        (-2.5e-7, "-2.5e-7"),
        (5e-324, "5e-324"),
        (2**53 - 1, "9007199254740991"),
        (2**64, "18446744073709552000"),
        (10**400, "null"),
        (float("nan"), "null"),
        (float("-inf"), "null"),
    ],
)
def test_writes_numbers_as_ecmascript_does(value: Any, text: str) -> None:
    assert canonical_json(value) == text


def test_writes_strings_as_json_stringify_does() -> None:
    assert canonical_json("café 東京 😀") == '"café 東京 😀"'
    assert (
        canonical_json('\x00\x1f"\\\n\t\u2028\x7f')
        == '"\\u0000\\u001f\\"\\\\\\n\\t\u2028\x7f"'
    )
    # A lone surrogate is escaped; a pair held as two code points is one character.
    assert canonical_json("a\udc00b") == '"a\\udc00b"'
    assert canonical_json("\ud83d\ude00") == '"😀"'


def test_sorts_keys_by_utf16_code_unit_at_every_depth() -> None:
    # U+1F600 is the surrogate pair D83D DE00, so it sorts before U+FF61.
    value = {"z": 1, "\uff61": 2, "\U0001f600": 3, "a": [{"b": 1, "a": 2}]}
    assert canonical_json(value) == '{"a":[{"a":2,"b":1}],"z":1,"😀":3,"｡":2}'


def test_normalizes_values_as_json_dumps_does() -> None:
    assert canonical_json({1: (True, None)}) == '{"1":[true,null]}'
    assert canonical_json({"when": object}) == '{"when":"<class \'object\'>"}'


# Shared with the JavaScript SDK's contentHash tests: the same generated content
# must hash the same in every language.
@pytest.mark.parametrize(
    ("generated", "digest"),
    [
        (
            {
                "status": "COMPLETE",
                "output": "café 東京 😀",
                "error": None,
                "usage": {"inputTokens": 1, "outputTokens": 2},
            },
            "c413938130f9e5d632edbd07d529139b942e1c43c62fe3fb9ae09552dc7e7f71",
        ),
        (
            {
                "status": "ERROR",
                "output": None,
                "error": {"code": 5001, "message": "handler raised: boom"},
                "errorMessage": "handler raised: boom",
            },
            "04e26552bf66f7afa68ba6307c9f9c162abd2bbd963cb7e42d9dcea337152848",
        ),
        (
            {
                "status": "COMPLETE",
                "output": {"b": 1, "a": [True, None, "x"]},
                "error": None,
            },
            "504de7cbb5f7d8b3ef260a99c69460c6b7f4fb3ea449c16cedde0ac0e120008c",
        ),
        (
            {
                "status": "COMPLETE",
                "output": {
                    "score": 1.0,
                    "p": 0.1,
                    "tiny": 1e-5,
                    "big": 1e16,
                    "list": [3, -0.0, 2.5e-7],
                    "neg": -2,
                },
                "error": None,
                "usage": {"inputTokens": 7, "outputTokens": 9},
            },
            "ae920410735c7fe62767503ea2121965d264b0d26abe396736d03be6aac77ae7",
        ),
    ],
)
def test_content_hash_matches_the_javascript_sdk(
    generated: dict[str, Any], digest: str
) -> None:
    assert hashlib.sha256(canonical_json(generated).encode()).hexdigest() == digest


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (1.0, "1"),
        ("high", '"high"'),
        ({"b": 2.0, "a": [None]}, '{"b":2,"a":[null]}'),
        (float("nan"), "NaN"),
        (float("-inf"), "-Infinity"),
    ],
)
def test_describe_value_is_json_stringify_text_except_non_finite(
    value: Any, text: str
) -> None:
    from launchdarkly_ai_server.evaluations.runner import _describe_value

    assert _describe_value(value) == text
