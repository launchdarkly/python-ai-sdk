"""
Drift test for the ``litellm.acompletion`` parameter classification in ``handler.py``.

``_COMPLETION_FORWARDED_KEYS`` is a literal, hand-maintained list. This test reads
``acompletion``'s own signature and asserts every named parameter it accepts is classified in
exactly one of forwarded, handler-owned, or excluded, so a parameter nobody has classified yet fails
loudly by name, and so does a list entry that is not a real parameter. ``acompletion`` also takes
``**kwargs``; nothing reaches it that way, because the forwarded list is the only way in.
"""

from __future__ import annotations

import inspect

import litellm

from launchdarkly_ai_litellm_messages.handler import (
    _COMPLETION_EXCLUDED_KEYS,
    _COMPLETION_FORWARDED_KEYS,
)

#: Handler-owned: set by the call site itself, never taken from the config.
_OWNED_KEYS = frozenset(
    {
        "model",
        "messages",
        "stream",
        "stream_options",
        "tools",
        "response_format",
        "api_key",
    }
)


def _accepted_keys() -> frozenset[str]:
    return frozenset(
        name
        for name, param in inspect.signature(litellm.acompletion).parameters.items()
        if param.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    )


class TestAcompletionAcceptsExactlyTheseKeys:
    def test_every_accepted_key_is_classified_exactly_once(self) -> None:
        classified = (
            _COMPLETION_FORWARDED_KEYS | _OWNED_KEYS | _COMPLETION_EXCLUDED_KEYS
        )

        unclassified = _accepted_keys() - classified
        assert not unclassified, (
            f"litellm.acompletion now accepts {sorted(unclassified)}, not classified as "
            "forwarded, handler-owned, or excluded in litellm-messages handler.py"
        )

        overlap = (
            (_COMPLETION_FORWARDED_KEYS & _OWNED_KEYS)
            | (_COMPLETION_FORWARDED_KEYS & _COMPLETION_EXCLUDED_KEYS)
            | (_OWNED_KEYS & _COMPLETION_EXCLUDED_KEYS)
        )
        assert not overlap, f"keys classified more than once: {sorted(overlap)}"

    def test_every_classified_key_is_a_real_parameter(self) -> None:
        stale = (
            _COMPLETION_FORWARDED_KEYS | _OWNED_KEYS | _COMPLETION_EXCLUDED_KEYS
        ) - _accepted_keys()
        assert not stale, (
            f"{sorted(stale)} classified in litellm-messages handler.py but "
            "litellm.acompletion does not accept them"
        )
