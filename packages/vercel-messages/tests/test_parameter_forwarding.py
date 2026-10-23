"""Forwarding of ``model.parameters`` into ``ai.InferenceRequestParams``.

A drift test classifies every field of ``InferenceRequestParams`` and its parts, value tests cover
the renames and the typed rebuilds, and a probe runs the real call site one key at a time on invoke
and stream. These use the real ``ai`` library with only ``ai.stream`` and ``ai.get_model`` replaced.
"""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import MagicMock

import ai
import pytest

import launchdarkly_ai_vercel_messages.handler as handler_mod
from launchdarkly_ai_vercel_messages import create_vercel_messages_handler
from launchdarkly_ai_vercel_messages.handler import (
    _INFERENCE_PARAMS_EXCLUDED_FIELDS,
    _INFERENCE_PARAMS_FORWARDED_KEYS,
    _request_params,
)
from tests.forwarding_spec import VERCEL, candidate_keys, probe_forwarded_keys
from tests.never_forwarded import NEVER_FORWARDED_BAG, find_leaks

from .test_handler import CONFIG, FakeStream

#: ``InferenceRequestParams`` fields that group typed parts, and the part classes in each.
_CONTAINERS: dict[str, tuple[type[Any], ...]] = {
    "sampling": (
        ai.TemperatureSamplerParams,
        ai.TopKSamplerParams,
        ai.TopPSamplerParams,
        ai.MinPSamplerParams,
        ai.RepetitionPenaltyParams,
        ai.SeedSamplerParams,
    ),
    "reasoning": (ai.ReasoningParams,),
    "tool_calling": (ai.ToolCallingParams,),
    "provider_service": (ai.ProviderServiceParams,),
    "output": (ai.OutputParams,),
    "cache": (ai.CacheParams,),
}

#: Forwarded config keys whose library field has another name.
_FIELD_FOR_KEY = {
    "max_output_tokens": "max_tokens",
    "prompt_cache_key": "key",
    "reasoning": "effort",
}


def _library_fields() -> frozenset[str]:
    fields = {
        f.name
        for f in dataclasses.fields(ai.InferenceRequestParams)
        if f.name not in _CONTAINERS
    }
    for classes in _CONTAINERS.values():
        for cls in classes:
            fields.update(f.name for f in dataclasses.fields(cls))
    return frozenset(fields)


class TestInferenceRequestParamsAreClassified:
    def test_every_field_is_classified_exactly_once(self) -> None:
        forwarded = frozenset(
            _FIELD_FOR_KEY.get(key, key) for key in _INFERENCE_PARAMS_FORWARDED_KEYS
        )
        unclassified = _library_fields() - forwarded - _INFERENCE_PARAMS_EXCLUDED_FIELDS
        assert not unclassified, (
            f"InferenceRequestParams now has {sorted(unclassified)}, not classified as "
            "forwarded or excluded in vercel-messages handler.py"
        )
        overlap = forwarded & _INFERENCE_PARAMS_EXCLUDED_FIELDS
        assert not overlap, f"fields classified more than once: {sorted(overlap)}"

    def test_every_classified_field_is_real(self) -> None:
        forwarded = frozenset(
            _FIELD_FOR_KEY.get(key, key) for key in _INFERENCE_PARAMS_FORWARDED_KEYS
        )
        stale = (forwarded | _INFERENCE_PARAMS_EXCLUDED_FIELDS) - _library_fields()
        assert not stale, (
            f"{sorted(stale)} classified in vercel-messages handler.py but "
            "InferenceRequestParams has no such field"
        )


def _params(parameters: dict[str, Any]) -> Any:
    return _request_params({"model": {"name": "m", "parameters": parameters}})


class TestRenames:
    def test_max_output_tokens_wins_over_the_other_spellings(self) -> None:
        params = _params(
            {"max_output_tokens": 3, "max_completion_tokens": 2, "max_tokens": 1}
        )
        assert params.output == ai.OutputParams(max_tokens=3)

    def test_max_completion_tokens_wins_over_max_tokens(self) -> None:
        params = _params({"max_completion_tokens": 2, "max_tokens": 1})
        assert params.output == ai.OutputParams(max_tokens=2)

    def test_max_tokens_alone_applies(self) -> None:
        assert _params({"max_tokens": 1}).output == ai.OutputParams(max_tokens=1)

    def test_reasoning_wins_over_reasoning_effort(self) -> None:
        params = _params({"reasoning": "high", "reasoning_effort": "low"})
        assert params.reasoning == ai.ReasoningParams(effort="high")

    def test_reasoning_effort_alone_applies(self) -> None:
        params = _params({"reasoning_effort": "low"})
        assert params.reasoning == ai.ReasoningParams(effort="low")

    def test_camel_case_spellings_are_not_read(self) -> None:
        assert _params({"topP": 0.4, "maxTokens": 9}) == ai.InferenceRequestParams()


class TestTypedValues:
    def test_penalties_share_one_sampler(self) -> None:
        params = _params({"frequency_penalty": 0.3, "presence_penalty": 0.4})
        assert params.sampling == {
            ai.RepetitionPenaltyParams: ai.RepetitionPenaltyParams(
                frequency_penalty=0.3, presence_penalty=0.4
            )
        }

    def test_reasoning_object_is_read_for_its_effort(self) -> None:
        params = _params({"reasoning": {"effort": "low", "summary": "auto"}})
        assert params.reasoning == ai.ReasoningParams(effort="low")
        assert params.output is None

    def test_reasoning_none_disables_reasoning(self) -> None:
        assert _params({"reasoning": "none"}).reasoning == ai.ReasoningParams(
            effort=None
        )

    @pytest.mark.parametrize(
        "reasoning", ["provider-default", {"summary": "auto"}, {"effort": 1}, 3]
    )
    def test_unusable_reasoning_is_dropped(self, reasoning: Any) -> None:
        assert _params({"reasoning": reasoning}) == ai.InferenceRequestParams()

    @pytest.mark.parametrize("mode", ["auto", "none", "required"])
    def test_tool_choice_mode_becomes_the_library_type(self, mode: str) -> None:
        assert _params({"tool_choice": mode}).tool_calling == ai.ToolCallingParams(
            tool_choice=ai.ToolChoiceMode(mode)
        )

    @pytest.mark.parametrize(
        "tool_choice",
        [{"type": "auto"}, {"type": "tool", "toolName": "lookup"}, "lookup", 1],
    )
    def test_other_tool_choice_values_are_dropped(self, tool_choice: Any) -> None:
        assert _params({"tool_choice": tool_choice}) == ai.InferenceRequestParams()

    def test_tool_calling_settings_without_a_tool_choice_use_auto(self) -> None:
        params = _params({"parallel_tool_calls": False, "max_tool_calls": 3})
        assert params.tool_calling == ai.ToolCallingParams(
            tool_choice=ai.ToolChoiceMode.AUTO,
            parallel_tool_calls=False,
            max_tool_calls=3,
        )

    def test_service_tier_and_prompt_cache_key(self) -> None:
        params = _params({"service_tier": "flex", "prompt_cache_key": "k"})
        assert params.provider_service == ai.ProviderServiceParams(service_tier="flex")
        assert params.cache == ai.CacheParams(key="k")


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every ``params`` handed to ``ai.stream``, with the real ``ai`` building them."""
    calls: list[Any] = []

    def stream(**kwargs: Any) -> FakeStream:
        calls.append(kwargs["params"])
        return FakeStream()

    monkeypatch.setattr(handler_mod.ai, "stream", stream)
    monkeypatch.setattr(handler_mod.ai, "get_model", MagicMock(return_value=object()))
    return calls


def _config(parameters: dict[str, Any]) -> dict[str, Any]:
    return {**CONFIG, "model": {**CONFIG["model"], "parameters": parameters}}


def _candidates() -> frozenset[str]:
    return candidate_keys(_library_fields(), VERCEL)


#: The Python library takes ``tool_choice`` only as a mode string.
_SAMPLE_OVERRIDES = {"tool_choice": "required"}


class TestNeverForwardedParameters:
    async def test_invoke_forwards_none_of_them(self, captured: list[Any]) -> None:
        await create_vercel_messages_handler()(
            _config({**NEVER_FORWARDED_BAG, "temperature": 0.1}), "q"
        )
        assert captured[-1] == ai.InferenceRequestParams(
            sampling={
                ai.TemperatureSamplerParams: ai.TemperatureSamplerParams(
                    temperature=0.1
                )
            }
        )
        assert not find_leaks(captured[-1])

    async def test_stream_forwards_none_of_them(self, captured: list[Any]) -> None:
        stream = await create_vercel_messages_handler().stream(
            _config({**NEVER_FORWARDED_BAG, "temperature": 0.1}), "q"
        )
        async for _event in stream:
            pass
        assert not find_leaks(captured[-1])
        assert captured[-1].sampling


class TestForwardsExactlyTheVercelList:
    """Probes ``invoke`` and ``stream`` one key at a time: the keys that change the request params
    handed to ``ai.stream`` are exactly the Vercel list, on both paths."""

    async def test_invoke(self, captured: list[Any]) -> None:
        async def call(parameters: dict[str, Any]) -> object:
            await create_vercel_messages_handler()(_config(parameters), "q")
            return captured[-1]

        forwarded = await probe_forwarded_keys(
            _candidates(), call, overrides=_SAMPLE_OVERRIDES
        )
        assert forwarded == VERCEL

    async def test_stream(self, captured: list[Any]) -> None:
        async def call(parameters: dict[str, Any]) -> object:
            stream = await create_vercel_messages_handler().stream(
                _config(parameters), "q"
            )
            async for _event in stream:
                pass
            return captured[-1]

        forwarded = await probe_forwarded_keys(
            _candidates(), call, overrides=_SAMPLE_OVERRIDES
        )
        assert forwarded == VERCEL
