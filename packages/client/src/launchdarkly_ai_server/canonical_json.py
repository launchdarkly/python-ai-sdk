"""RFC 8785 JSON Canonicalization Scheme (JCS), for hashes every SDK must agree on.

JCS spells JSON the way ECMAScript's ``JSON.stringify`` does: no whitespace,
object keys sorted by UTF-16 code unit, non-ASCII written as is, and numbers in
ECMAScript's shortest round-trip form. That is the native form in JavaScript
and close to it in Go, so the only language that needs this module is Python,
whose ``json.dumps`` writes ``1.0`` where JCS writes ``1``.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from math import isfinite
from typing import Any

# Above this an int has no exact IEEE double, so it is written as the double
# nearest to it, as a JavaScript JSON.parse reads it.
_MAX_SAFE_INTEGER = 2**53 - 1
_SURROGATE = re.compile("[\ud800-\udfff]")


def canonical_json(value: Any, *, sort_keys: bool = True) -> str:
    """The JCS text of a JSON-like value.

    With ``sort_keys=False`` object keys keep their insertion order, which is
    ``JSON.stringify``'s output exactly: the form judge text uses, so a judge
    reads the same text in every SDK. It is not JCS, so never hash it.

    The value is first normalized through ``json.dumps(..., default=str)``, so
    a tuple is a list, a non-string key is the string ``json.dumps`` writes for
    it, and a value JSON cannot hold is its ``str()``. A non-finite number is
    written ``null``, as ``JSON.stringify`` writes it.
    """
    return _encode(json.loads(json.dumps(value, default=str)), sort_keys)


def _encode(value: Any, sort_keys: bool = True) -> str:
    if value is None or value is True or value is False:
        return json.dumps(value)
    if isinstance(value, int):
        if abs(value) <= _MAX_SAFE_INTEGER:
            return str(value)
        try:
            return _number(float(value))
        except OverflowError:
            return "null"
    if isinstance(value, float):
        return _number(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, list):
        return "[" + ",".join(_encode(item, sort_keys) for item in value) + "]"
    keys = sorted(value, key=_utf16) if sort_keys else list(value)
    return (
        "{"
        + ",".join(f"{_string(key)}:{_encode(value[key], sort_keys)}" for key in keys)
        + "}"
    )


def _number(value: float) -> str:
    """ECMAScript's Number::toString, from the shortest round-trip digits ``repr`` gives."""
    if not isfinite(value):
        return "null"
    if value == 0:
        return "0"
    _, digit_tuple, exponent = Decimal(repr(abs(value))).as_tuple()
    # The value is 0.<digits> x 10^n, the form Number::toString starts from.
    n = len(digit_tuple) + int(exponent)
    digits = "".join(map(str, digit_tuple)).rstrip("0")
    k = len(digits)
    if k <= n <= 21:
        text = digits + "0" * (n - k)
    elif 0 < n <= 21:
        text = f"{digits[:n]}.{digits[n:]}"
    elif -6 < n <= 0:
        text = f"0.{'0' * -n}{digits}"
    else:
        fraction = f".{digits[1:]}" if k > 1 else ""
        text = f"{digits[0]}{fraction}e{'+' if n - 1 >= 0 else '-'}{abs(n - 1)}"
    return ("-" if value < 0 else "") + text


def _string(text: str) -> str:
    # A surrogate pair held as two code points is joined into one character, as
    # it is in a JavaScript string; a lone surrogate is escaped, as
    # JSON.stringify escapes it, so the text always has a UTF-8 encoding.
    joined = text.encode("utf-16", "surrogatepass").decode("utf-16", "surrogatepass")
    return _SURROGATE.sub(
        lambda match: f"\\u{ord(match.group()):04x}",
        json.dumps(joined, ensure_ascii=False),
    )


def _utf16(key: str) -> bytes:
    return key.encode("utf-16-be", "surrogatepass")
