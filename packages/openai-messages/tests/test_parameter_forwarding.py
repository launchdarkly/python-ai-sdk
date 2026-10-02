"""
Drift test for the OpenAI Responses API parameter classification in ``handler.py``.

``_RESPONSES_CREATE_FORWARDED_KEYS`` / ``_RESPONSES_STREAM_FORWARDED_KEYS`` are literal,
hand-maintained lists. This test reads ``AsyncResponses.create``/``.stream``'s own signature and
asserts every parameter they accept is classified in exactly one of forwarded, handler-owned, or
excluded, so an SDK parameter nobody has classified yet fails loudly by name, and so does a list
entry that is not a real SDK parameter.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from openai.resources.responses import AsyncResponses

from launchdarkly_ai_openai_messages.handler import (
    _RESPONSES_CREATE_FORWARDED_KEYS,
    _RESPONSES_EXCLUDED_KEYS,
    _RESPONSES_STREAM_FORWARDED_KEYS,
)

#: Handler-owned: always popped from the filtered params before the call, at both call sites.
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
    return keys


class TestResponsesCreateAcceptsExactlyTheseKeys:
    def test_every_accepted_key_is_classified_exactly_once(self) -> None:
        accepted = _signature_keys(AsyncResponses.create)
        classified = (
            _RESPONSES_CREATE_FORWARDED_KEYS | _OWNED_KEYS | _RESPONSES_EXCLUDED_KEYS
        )

        unclassified = accepted - classified
        assert not unclassified, (
            f"AsyncResponses.create now accepts {sorted(unclassified)}, not classified as "
            "forwarded, handler-owned, or excluded in openai-messages handler.py"
        )

        overlap = (
            (_RESPONSES_CREATE_FORWARDED_KEYS & _OWNED_KEYS)
            | (_RESPONSES_CREATE_FORWARDED_KEYS & _RESPONSES_EXCLUDED_KEYS)
            | (_OWNED_KEYS & _RESPONSES_EXCLUDED_KEYS)
        )
        assert not overlap, f"keys classified more than once: {sorted(overlap)}"

    def test_every_classified_key_is_a_real_parameter(self) -> None:
        accepted = _signature_keys(AsyncResponses.create)
        stale = (_RESPONSES_CREATE_FORWARDED_KEYS | _OWNED_KEYS) - accepted
        assert not stale, (
            f"{sorted(stale)} classified in openai-messages handler.py but "
            "AsyncResponses.create does not accept them"
        )


class TestResponsesStreamAcceptsExactlyTheseKeys:
    def test_every_accepted_key_is_classified_exactly_once(self) -> None:
        accepted = _signature_keys(AsyncResponses.stream)
        classified = (
            _RESPONSES_STREAM_FORWARDED_KEYS | _OWNED_KEYS | _RESPONSES_EXCLUDED_KEYS
        )

        unclassified = accepted - classified
        assert not unclassified, (
            f"AsyncResponses.stream now accepts {sorted(unclassified)}, not classified as "
            "forwarded, handler-owned, or excluded in openai-messages handler.py"
        )

    def test_every_classified_key_is_a_real_parameter(self) -> None:
        accepted = _signature_keys(AsyncResponses.stream)
        stale = (_RESPONSES_STREAM_FORWARDED_KEYS | _OWNED_KEYS) - accepted
        assert not stale, (
            f"{sorted(stale)} classified in openai-messages handler.py but "
            "AsyncResponses.stream does not accept them"
        )
