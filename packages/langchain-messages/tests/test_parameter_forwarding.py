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

import importlib
from collections.abc import Mapping
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

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
    _make_default_chat_model,
    _model_constructor_kwargs,
)
from tests.forwarding_spec import (
    LANGCHAIN_CHAT_ANTHROPIC,
    LANGCHAIN_CHAT_BEDROCK_CONVERSE,
    LANGCHAIN_CHAT_OPENAI,
    LANGCHAIN_CHAT_OPENAI_UNSUPPORTED,
    candidate_keys,
    probe_forwarded_keys,
    sample,
)
from tests.never_forwarded import NEVER_FORWARDED_BAG


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


class TestNeverForwardedKeys:
    """No credential, endpoint, request-injection, or remote-tool key in ``model.parameters``
    reaches any chat model constructor."""

    def _config(self, provider: str, parameters: dict[str, Any]) -> Any:
        return {
            "model": {"name": "configured-model", "parameters": parameters},
            "provider": {"name": provider},
        }

    @pytest.mark.parametrize(
        ("provider", "forwarded_keys"),
        [
            ("openai", _CHAT_OPENAI_FORWARDED_KEYS),
            ("anthropic", _CHAT_ANTHROPIC_FORWARDED_KEYS),
            ("bedrock", _CHAT_BEDROCK_CONVERSE_FORWARDED_KEYS),
        ],
    )
    def test_constructor_kwargs_hold_none_of_them(
        self, provider: str, forwarded_keys: frozenset[str]
    ) -> None:
        kwargs = _model_constructor_kwargs(
            self._config(provider, {**NEVER_FORWARDED_BAG, "temperature": 0.2}),
            "fallback",
            forwarded_keys,
        )
        assert kwargs == {"model": "configured-model", "temperature": 0.2}


class TestModelKwargsCannotSmuggleRequestKeys:
    """``ChatOpenAI`` merges ``model_kwargs`` straight into the request payload, so forwarding it
    would carry ``extra_headers``/``extra_query`` past every exclusion."""

    _SMUGGLED: ClassVar[dict[str, Any]] = {
        "model_kwargs": {
            "extra_headers": {"X-Smuggled": "1"},
            "extra_query": {"smuggled": "1"},
        }
    }

    def _payload(self, **kwargs: Any) -> dict[str, Any]:
        from langchain_core.messages import HumanMessage

        model = langchain_openai.ChatOpenAI(api_key="sk-test-not-a-real-key", **kwargs)
        payload: dict[str, Any] = model._get_request_payload([HumanMessage("hi")])
        return payload

    def test_model_kwargs_would_reach_the_payload_if_forwarded(self) -> None:
        """The control: passed to ``ChatOpenAI`` directly, both keys reach the payload."""
        payload = self._payload(model="gpt-4o", **self._SMUGGLED)
        assert payload["extra_headers"] == {"X-Smuggled": "1"}
        assert payload["extra_query"] == {"smuggled": "1"}

    def test_forwarded_model_parameters_keep_them_out_of_the_payload(self) -> None:
        kwargs = _model_constructor_kwargs(
            {
                "model": {"name": "gpt-4o", "parameters": dict(self._SMUGGLED)},
                "provider": {"name": "openai"},
            },
            "fallback",
            _CHAT_OPENAI_FORWARDED_KEYS,
        )
        payload = self._payload(**kwargs)
        assert "extra_headers" not in payload
        assert "extra_query" not in payload


def _probe_config(provider: str, parameters: dict[str, Any]) -> Any:
    return {
        "model": {"name": "configured-model", "parameters": parameters},
        "provider": {"name": provider},
    }


def _build(provider: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Runs ``_make_default_chat_model`` for *provider* against a recording constructor and
    returns the kwargs it passed."""
    ctor = MagicMock()
    module = MagicMock(ChatOpenAI=ctor, ChatAnthropic=ctor, ChatBedrockConverse=ctor)
    importlib = MagicMock()
    importlib.import_module.return_value = module
    _make_default_chat_model(_probe_config(provider, parameters), importlib)
    kwargs: dict[str, Any] = ctor.call_args.kwargs
    return kwargs


def _accepted_or_empty(module_name: str, cls_name: str) -> frozenset[str]:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return frozenset()
    return _accepted_keys(getattr(module, cls_name))


class TestForwardsExactlyTheCrossSdkList:
    """Probes each chat model constructor one key at a time: the keys that change what it is
    built with are exactly the cross-SDK list for that class. ``invoke`` and ``stream`` both
    build the model through ``_make_default_chat_model``, so this covers both paths."""

    @pytest.mark.parametrize(
        ("provider", "module_name", "cls_name", "expected"),
        [
            (
                "openai",
                "langchain_openai",
                "ChatOpenAI",
                LANGCHAIN_CHAT_OPENAI - LANGCHAIN_CHAT_OPENAI_UNSUPPORTED,
            ),
            (
                "anthropic",
                "langchain_anthropic",
                "ChatAnthropic",
                LANGCHAIN_CHAT_ANTHROPIC,
            ),
            (
                "bedrock",
                "langchain_aws",
                "ChatBedrockConverse",
                LANGCHAIN_CHAT_BEDROCK_CONVERSE,
            ),
        ],
    )
    async def test_forwarded_keys(
        self, provider: str, module_name: str, cls_name: str, expected: frozenset[str]
    ) -> None:
        baseline = _build(provider, {})

        async def call(parameters: dict[str, Any]) -> object:
            return _build(provider, parameters)

        candidates = candidate_keys(_accepted_or_empty(module_name, cls_name), expected)
        assert await probe_forwarded_keys(candidates, call) == expected
        assert baseline == {"model": "configured-model"}

    def test_chat_openai_has_no_prompt_cache_key_field(self) -> None:
        """Why ``prompt_cache_key`` is left off: ``ChatOpenAI`` cannot take it. When it can,
        this fails and the key goes on the list."""
        assert not LANGCHAIN_CHAT_OPENAI_UNSUPPORTED & _accepted_keys(
            langchain_openai.ChatOpenAI
        )

    @pytest.mark.parametrize(
        ("cls", "keys"),
        [
            (
                langchain_openai.ChatOpenAI,
                LANGCHAIN_CHAT_OPENAI - LANGCHAIN_CHAT_OPENAI_UNSUPPORTED,
            ),
            (langchain_anthropic.ChatAnthropic, LANGCHAIN_CHAT_ANTHROPIC),
        ],
    )
    def test_the_real_class_accepts_every_forwarded_key(
        self, cls: Any, keys: frozenset[str]
    ) -> None:
        for key in sorted(keys):
            cls(model="m", api_key="sk-test-not-a-real-key", **{key: sample(key)})


class TestMalformedObjectValuesAreDropped:
    @pytest.mark.parametrize(
        ("provider", "parameters"),
        [
            ("openai", {"logit_bias": "none"}),
            ("anthropic", {"thinking": "enabled", "output_config": "json"}),
            ("bedrock", {"performance_config": "optimized"}),
        ],
    )
    def test_dropped(self, provider: str, parameters: dict[str, Any]) -> None:
        kwargs = _build(provider, {**parameters, "temperature": 0.2})
        assert kwargs == {"model": "configured-model", "temperature": 0.2}


class _Built(Exception):
    """Stops a run once the chat model has been built."""


class TestInvokeAndStreamBuildTheSameModel:
    """Both paths hand the same config to ``_make_default_chat_model``, so the probe above
    covers both."""

    async def test_same_config_reaches_the_builder(self) -> None:
        import launchdarkly_ai_langchain_messages.handler as handler_mod
        from launchdarkly_ai_langchain_messages import create_langchain_messages_handler

        config = {
            "model": {
                "name": "gpt-4o",
                "parameters": {**NEVER_FORWARDED_BAG, "temperature": 0.2, "top_p": 0.5},
            },
            "provider": {"name": "openai"},
            "instructions": "help",
        }
        seen: list[Any] = []

        def _builder(cfg: Any, *_rest: Any) -> Any:
            seen.append(cfg)
            raise _Built

        with patch.object(handler_mod, "_make_default_chat_model", _builder):
            h = create_langchain_messages_handler()
            with pytest.raises(_Built):
                await h(config, "q")
            with pytest.raises(_Built):
                async for _event in await h.stream(config, "q"):
                    pass

        assert len(seen) == 2
        assert seen[0] == seen[1]
