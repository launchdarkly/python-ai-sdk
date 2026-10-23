"""
Drift test for the ``agents.ModelSettings`` parameter classification in ``handler.py``.

``_MODEL_SETTINGS_FORWARDED_KEYS`` is a literal, hand-maintained list. This test reads
``ModelSettings``'s own dataclass fields and asserts every field it declares is classified in
exactly one of forwarded or excluded, so an SDK field nobody has classified yet fails loudly by
name, and so does a list entry that is not a real SDK field. It also checks every forwarded field
is one ``LitellmModel`` actually reads, so a key it would silently drop is not on the list.
"""

from __future__ import annotations

import dataclasses
import inspect
import re

from agents import ModelSettings
from agents.extensions.models import litellm_model

from launchdarkly_ai_litellm_agents.handler import (
    _MODEL_SETTINGS_EXCLUDED_KEYS,
    _MODEL_SETTINGS_FORWARDED_KEYS,
)

_FIELDS = frozenset(f.name for f in dataclasses.fields(ModelSettings))


class TestModelSettingsAcceptsExactlyTheseFields:
    def test_every_field_is_classified_exactly_once(self) -> None:
        classified = _MODEL_SETTINGS_FORWARDED_KEYS | _MODEL_SETTINGS_EXCLUDED_KEYS

        unclassified = _FIELDS - classified
        assert not unclassified, (
            f"ModelSettings now declares {sorted(unclassified)}, not classified as "
            "forwarded or excluded in litellm-agents handler.py"
        )

        overlap = _MODEL_SETTINGS_FORWARDED_KEYS & _MODEL_SETTINGS_EXCLUDED_KEYS
        assert not overlap, f"fields classified more than once: {sorted(overlap)}"

    def test_every_classified_field_is_real(self) -> None:
        stale = (
            _MODEL_SETTINGS_FORWARDED_KEYS | _MODEL_SETTINGS_EXCLUDED_KEYS
        ) - _FIELDS
        assert not stale, (
            f"{sorted(stale)} classified in litellm-agents handler.py but "
            "ModelSettings does not declare them"
        )

    def test_every_forwarded_field_is_read_by_litellm_model(self) -> None:
        source = inspect.getsource(litellm_model)
        read = frozenset(re.findall(r"model_settings\.(\w+)", source))
        unread = _MODEL_SETTINGS_FORWARDED_KEYS - read
        assert not unread, (
            f"{sorted(unread)} forwarded in litellm-agents handler.py but LitellmModel "
            "never reads them"
        )
