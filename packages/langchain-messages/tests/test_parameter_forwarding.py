"""
Drift test for the LangChain chat model parameter classification in ``handler.py``.

``_CHAT_OPENAI_FORWARDED_KEYS`` / ``_CHAT_ANTHROPIC_FORWARDED_KEYS`` /
``_CHAT_BEDROCK_CONVERSE_FORWARDED_KEYS`` are literal, hand-maintained lists. This test reads each
chat model's own ``model_fields`` (field names plus pydantic aliases) and asserts every key it
accepts is classified in exactly one of forwarded, handler-owned, or excluded, so a field nobody has
classified yet fails loudly by name, and so does a list entry that is not a real field.

``langchain-aws`` is not a dependency of this package (Bedrock support is opt-in), so the
``ChatBedrockConverse`` test skips itself when it is not installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import langchain_anthropic
import langchain_openai
import pytest

from launchdarkly_ai_langchain_messages.handler import (
    _CHAT_ANTHROPIC_EXCLUDED_KEYS,
    _CHAT_ANTHROPIC_FORWARDED_KEYS,
    _CHAT_ANTHROPIC_OWNED_KEYS,
    _CHAT_BEDROCK_CONVERSE_EXCLUDED_KEYS,
    _CHAT_BEDROCK_CONVERSE_FORWARDED_KEYS,
    _CHAT_BEDROCK_CONVERSE_OWNED_KEYS,
    _CHAT_OPENAI_EXCLUDED_KEYS,
    _CHAT_OPENAI_FORWARDED_KEYS,
    _CHAT_OPENAI_OWNED_KEYS,
    _model_constructor_kwargs,
)


def _accepted_keys(cls: Any) -> frozenset[str]:
    """Every key *cls* (a pydantic model) accepts by construction: each field's own name plus any
    string alias it declares."""
    fields: Mapping[str, Any] = cls.model_fields
    accepted: set[str] = set()
    for name, field in fields.items():
        accepted.add(name)
        alias = getattr(field, "alias", None)
        if isinstance(alias, str):
            accepted.add(alias)
        validation_alias = getattr(field, "validation_alias", None)
        if isinstance(validation_alias, str):
            accepted.add(validation_alias)
        else:
            choices = getattr(validation_alias, "choices", None)
            if choices:
                accepted.update(c for c in choices if isinstance(c, str))
    return frozenset(accepted)


class TestChatOpenAIAcceptsExactlyTheseKeys:
    def test_every_accepted_key_is_classified_exactly_once(self) -> None:
        accepted = _accepted_keys(langchain_openai.ChatOpenAI)
        classified = (
            _CHAT_OPENAI_FORWARDED_KEYS
            | _CHAT_OPENAI_OWNED_KEYS
            | _CHAT_OPENAI_EXCLUDED_KEYS
        )

        unclassified = accepted - classified
        assert not unclassified, (
            f"ChatOpenAI now accepts {sorted(unclassified)}, not classified as forwarded, "
            "handler-owned, or excluded in langchain-messages handler.py"
        )

        overlap = (
            (_CHAT_OPENAI_FORWARDED_KEYS & _CHAT_OPENAI_OWNED_KEYS)
            | (_CHAT_OPENAI_FORWARDED_KEYS & _CHAT_OPENAI_EXCLUDED_KEYS)
            | (_CHAT_OPENAI_OWNED_KEYS & _CHAT_OPENAI_EXCLUDED_KEYS)
        )
        assert not overlap, f"keys classified more than once: {sorted(overlap)}"

    def test_every_classified_key_is_real(self) -> None:
        accepted = _accepted_keys(langchain_openai.ChatOpenAI)
        stale = (
            _CHAT_OPENAI_FORWARDED_KEYS
            | _CHAT_OPENAI_OWNED_KEYS
            | _CHAT_OPENAI_EXCLUDED_KEYS
        ) - accepted
        assert not stale, (
            f"{sorted(stale)} classified in langchain-messages handler.py but "
            "ChatOpenAI does not accept them"
        )


class TestChatAnthropicAcceptsExactlyTheseKeys:
    def test_every_accepted_key_is_classified_exactly_once(self) -> None:
        accepted = _accepted_keys(langchain_anthropic.ChatAnthropic)
        classified = (
            _CHAT_ANTHROPIC_FORWARDED_KEYS
            | _CHAT_ANTHROPIC_OWNED_KEYS
            | _CHAT_ANTHROPIC_EXCLUDED_KEYS
        )

        unclassified = accepted - classified
        assert not unclassified, (
            f"ChatAnthropic now accepts {sorted(unclassified)}, not classified as forwarded, "
            "handler-owned, or excluded in langchain-messages handler.py"
        )

    def test_every_classified_key_is_real(self) -> None:
        accepted = _accepted_keys(langchain_anthropic.ChatAnthropic)
        stale = (
            _CHAT_ANTHROPIC_FORWARDED_KEYS
            | _CHAT_ANTHROPIC_OWNED_KEYS
            | _CHAT_ANTHROPIC_EXCLUDED_KEYS
        ) - accepted
        assert not stale, (
            f"{sorted(stale)} classified in langchain-messages handler.py but "
            "ChatAnthropic does not accept them"
        )


class TestChatBedrockConverseAcceptsExactlyTheseKeys:
    def test_every_accepted_key_is_classified_exactly_once(self) -> None:
        lc_aws = pytest.importorskip("langchain_aws")
        accepted = _accepted_keys(lc_aws.ChatBedrockConverse)
        classified = (
            _CHAT_BEDROCK_CONVERSE_FORWARDED_KEYS
            | _CHAT_BEDROCK_CONVERSE_OWNED_KEYS
            | _CHAT_BEDROCK_CONVERSE_EXCLUDED_KEYS
        )

        unclassified = accepted - classified
        assert not unclassified, (
            f"ChatBedrockConverse now accepts {sorted(unclassified)}, not classified as "
            "forwarded, handler-owned, or excluded in langchain-messages handler.py"
        )

    def test_every_classified_key_is_real(self) -> None:
        lc_aws = pytest.importorskip("langchain_aws")
        accepted = _accepted_keys(lc_aws.ChatBedrockConverse)
        stale = (
            _CHAT_BEDROCK_CONVERSE_FORWARDED_KEYS
            | _CHAT_BEDROCK_CONVERSE_OWNED_KEYS
            | _CHAT_BEDROCK_CONVERSE_EXCLUDED_KEYS
        ) - accepted
        assert not stale, (
            f"{sorted(stale)} classified in langchain-messages handler.py but "
            "ChatBedrockConverse does not accept them"
        )


class TestModelFieldAliasesAreNeverForwarded:
    """``model_name`` / ``model_id`` are the same constructor field as ``model``. A config value
    for them must not be forwarded, or it would collide with the model the handler resolves."""

    def _config(self, provider: str, parameters: dict[str, Any]) -> Any:
        return {
            "model": {"name": "configured-model", "parameters": parameters},
            "provider": {"name": provider},
        }

    def test_openai_model_name_is_dropped(self) -> None:
        kwargs = _model_constructor_kwargs(
            self._config("openai", {"model_name": "other", "temperature": 0.2}),
            "fallback",
            _CHAT_OPENAI_FORWARDED_KEYS,
        )
        assert kwargs == {"model": "configured-model", "temperature": 0.2}

    def test_anthropic_model_name_is_dropped(self) -> None:
        kwargs = _model_constructor_kwargs(
            self._config("anthropic", {"model_name": "other", "temperature": 0.2}),
            "fallback",
            _CHAT_ANTHROPIC_FORWARDED_KEYS,
        )
        assert kwargs == {"model": "configured-model", "temperature": 0.2}

    def test_bedrock_model_id_is_dropped(self) -> None:
        kwargs = _model_constructor_kwargs(
            self._config("bedrock", {"model_id": "other", "temperature": 0.2}),
            "fallback",
            _CHAT_BEDROCK_CONVERSE_FORWARDED_KEYS,
        )
        assert kwargs == {"model": "configured-model", "temperature": 0.2}
