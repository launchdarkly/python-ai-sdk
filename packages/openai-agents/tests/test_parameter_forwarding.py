"""
Drift test for the ``agents.ModelSettings`` parameter classification in ``handler.py``.

``_MODEL_SETTINGS_FORWARDED_KEYS`` is a literal, hand-maintained list. This test reads
``ModelSettings``'s own dataclass fields and asserts every field it declares is classified in
exactly one of forwarded or excluded, so an SDK field nobody has classified yet fails loudly by
name, and so does a list entry that is not a real SDK field. ``ModelSettings`` has no handler-owned
fields: ``model`` and ``max_turns`` live outside it (on ``Agent``/``Runner.run``).
"""

from __future__ import annotations

import dataclasses

from agents import ModelSettings

from launchdarkly_ai_openai_agents.handler import (
    _MODEL_SETTINGS_EXCLUDED_KEYS as _EXCLUDED_KEYS,
)
from launchdarkly_ai_openai_agents.handler import _MODEL_SETTINGS_FORWARDED_KEYS


class TestModelSettingsAcceptsExactlyTheseFields:
    def test_every_field_is_classified_exactly_once(self) -> None:
        accepted = frozenset(f.name for f in dataclasses.fields(ModelSettings))
        classified = _MODEL_SETTINGS_FORWARDED_KEYS | _EXCLUDED_KEYS

        unclassified = accepted - classified
        assert not unclassified, (
            f"ModelSettings now declares {sorted(unclassified)}, not classified as "
            "forwarded or excluded in openai-agents handler.py"
        )

        overlap = _MODEL_SETTINGS_FORWARDED_KEYS & _EXCLUDED_KEYS
        assert not overlap, f"fields classified more than once: {sorted(overlap)}"

    def test_every_classified_field_is_real(self) -> None:
        accepted = frozenset(f.name for f in dataclasses.fields(ModelSettings))
        stale = (_MODEL_SETTINGS_FORWARDED_KEYS | _EXCLUDED_KEYS) - accepted
        assert not stale, (
            f"{sorted(stale)} classified in openai-agents handler.py but "
            "ModelSettings does not declare them"
        )
