"""
Drift test for the OpenAI Responses API parameter classification in ``handler.py``.

``_RESPONSES_FORWARDED_KEYS`` is a literal, hand-maintained list, and the same list serves ``invoke`` and ``stream``.
This test reads ``AsyncResponses.create``/``.stream``'s own signatures and asserts every parameter either
accepts is classified in exactly one of forwarded, handler-owned, or excluded, so an SDK parameter
nobody has classified yet fails loudly by name. It also asserts every forwarded or handler-owned key
is accepted by both calls, so ``invoke`` and ``stream`` cannot drift apart again, and that every
excluded key is a real parameter of at least one of them.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from openai.resources.responses import AsyncResponses

from launchdarkly_ai_openai_messages.handler import (
    _RESPONSES_EXCLUDED_KEYS,
    _RESPONSES_FORWARDED_KEYS,
)

#: Handler-owned: set by each call site itself, never taken from the config.
_OWNED_KEYS = frozenset({"model", "input", "previous_response_id", "tools", "text"})


def _signature_keys(fn: Callable[..., object]) -> frozenset[str]:
    keys: set[str] = set()
    for name, param in inspect.signature(fn).parameters.items():
        if name == "self" or param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        assert param.kind is not inspect.Parameter.VAR_KEYWORD, (
            f"{fn!r} now accepts **{name}; this test can no longer enumerate its accept-set"
        )
        keys.add(name)
    return frozenset(keys)


_CREATE_KEYS = _signature_keys(AsyncResponses.create)
_STREAM_KEYS = _signature_keys(AsyncResponses.stream)


class TestResponsesCreateAcceptsExactlyTheseKeys:
    def test_every_accepted_key_is_classified_exactly_once(self) -> None:
        classified = _RESPONSES_FORWARDED_KEYS | _OWNED_KEYS | _RESPONSES_EXCLUDED_KEYS

        unclassified = _CREATE_KEYS - classified
        assert not unclassified, (
            f"AsyncResponses.create now accepts {sorted(unclassified)}, not classified as "
            "forwarded, handler-owned, or excluded in openai-messages handler.py"
        )

        overlap = (
            (_RESPONSES_FORWARDED_KEYS & _OWNED_KEYS)
            | (_RESPONSES_FORWARDED_KEYS & _RESPONSES_EXCLUDED_KEYS)
            | (_OWNED_KEYS & _RESPONSES_EXCLUDED_KEYS)
        )
        assert not overlap, f"keys classified more than once: {sorted(overlap)}"

    def test_every_forwarded_or_owned_key_is_a_real_parameter(self) -> None:
        stale = (_RESPONSES_FORWARDED_KEYS | _OWNED_KEYS) - _CREATE_KEYS
        assert not stale, (
            f"{sorted(stale)} forwarded or handler-owned in openai-messages handler.py but "
            "AsyncResponses.create does not accept them"
        )


class TestResponsesStreamAcceptsExactlyTheseKeys:
    def test_every_accepted_key_is_classified_exactly_once(self) -> None:
        classified = _RESPONSES_FORWARDED_KEYS | _OWNED_KEYS | _RESPONSES_EXCLUDED_KEYS

        unclassified = _STREAM_KEYS - classified
        assert not unclassified, (
            f"AsyncResponses.stream now accepts {sorted(unclassified)}, not classified as "
            "forwarded, handler-owned, or excluded in openai-messages handler.py"
        )

    def test_every_forwarded_or_owned_key_is_a_real_parameter(self) -> None:
        stale = (_RESPONSES_FORWARDED_KEYS | _OWNED_KEYS) - _STREAM_KEYS
        assert not stale, (
            f"{sorted(stale)} forwarded or handler-owned in openai-messages handler.py but "
            "AsyncResponses.stream does not accept them"
        )


def test_every_excluded_key_is_a_real_parameter() -> None:
    stale = _RESPONSES_EXCLUDED_KEYS - (_CREATE_KEYS | _STREAM_KEYS)
    assert not stale, (
        f"{sorted(stale)} excluded in openai-messages handler.py but neither "
        "AsyncResponses.create nor .stream accepts them"
    )
