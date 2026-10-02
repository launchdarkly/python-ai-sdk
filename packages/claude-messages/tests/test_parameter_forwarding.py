"""
Drift test for the Anthropic Messages API parameter classification in ``handler.py``.

``_MESSAGES_CREATE_FORWARDED_KEYS`` / ``_MESSAGES_STREAM_FORWARDED_KEYS`` are literal,
hand-maintained lists (see the module docstring there for why). This test is what keeps them
honest: it reads ``AsyncMessages.create``/``.stream``'s own signature and asserts every parameter
they accept is classified in exactly one of forwarded, handler-owned, or excluded, so an SDK
parameter nobody has classified yet fails loudly by name, and so does a list entry that is not a
real SDK parameter.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import anthropic

from launchdarkly_ai_claude_messages.handler import (
    _MESSAGES_CREATE_FORWARDED_KEYS,
    _MESSAGES_EXCLUDED_KEYS,
    _MESSAGES_STREAM_FORWARDED_KEYS,
)

#: Handler-owned: always popped from the filtered params before the call, at both call sites.
_OWNED_KEYS = frozenset({"model", "messages", "system", "tools"})


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


class TestMessagesCreateAcceptsExactlyTheseKeys:
    def test_every_accepted_key_is_classified_exactly_once(self) -> None:
        accepted = _signature_keys(anthropic.resources.messages.AsyncMessages.create)
        classified = (
            _MESSAGES_CREATE_FORWARDED_KEYS | _OWNED_KEYS | _MESSAGES_EXCLUDED_KEYS
        )

        unclassified = accepted - classified
        assert not unclassified, (
            f"AsyncMessages.create now accepts {sorted(unclassified)}, not classified as "
            "forwarded, handler-owned, or excluded in claude-messages handler.py"
        )

        overlap = (
            (_MESSAGES_CREATE_FORWARDED_KEYS & _OWNED_KEYS)
            | (_MESSAGES_CREATE_FORWARDED_KEYS & _MESSAGES_EXCLUDED_KEYS)
            | (_OWNED_KEYS & _MESSAGES_EXCLUDED_KEYS)
        )
        assert not overlap, f"keys classified more than once: {sorted(overlap)}"

    def test_every_classified_key_is_a_real_parameter(self) -> None:
        accepted = _signature_keys(anthropic.resources.messages.AsyncMessages.create)
        stale = (_MESSAGES_CREATE_FORWARDED_KEYS | _OWNED_KEYS) - accepted
        assert not stale, (
            f"{sorted(stale)} classified in claude-messages handler.py but "
            "AsyncMessages.create does not accept them"
        )


class TestMessagesStreamAcceptsExactlyTheseKeys:
    def test_every_accepted_key_is_classified_exactly_once(self) -> None:
        accepted = _signature_keys(anthropic.resources.messages.AsyncMessages.stream)
        classified = (
            _MESSAGES_STREAM_FORWARDED_KEYS | _OWNED_KEYS | _MESSAGES_EXCLUDED_KEYS
        )

        unclassified = accepted - classified
        assert not unclassified, (
            f"AsyncMessages.stream now accepts {sorted(unclassified)}, not classified as "
            "forwarded, handler-owned, or excluded in claude-messages handler.py"
        )

    def test_every_classified_key_is_a_real_parameter(self) -> None:
        accepted = _signature_keys(anthropic.resources.messages.AsyncMessages.stream)
        stale = (_MESSAGES_STREAM_FORWARDED_KEYS | _OWNED_KEYS) - accepted
        assert not stale, (
            f"{sorted(stale)} classified in claude-messages handler.py but "
            "AsyncMessages.stream does not accept them"
        )
