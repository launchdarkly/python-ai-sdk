from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from launchdarkly_ai_server.evaluations.api import EvaluationsError
from launchdarkly_ai_server.evaluations.judges import (
    Accuracy,
    AnswerRelevancy,
    Bias,
    Judge,
    JudgeReference,
    Likeness,
    Misinformation,
    Toxicity,
    resolve_launchdarkly_judges,
)
from launchdarkly_ai_server.types import ProviderHandler
from launchdarkly_ai_server.utils import create_handler


def judge_variation(
    *,
    key: str = "served-variation",
    version: int = 12,
    provider: str = "OpenAI",
    mode: str = "messages",
    inverted: bool = False,
) -> dict[str, Any]:
    return {
        "config": {
            "provider": {"name": provider},
            "model": {"name": "judge-model"},
            "instructions": "Evaluate {{response_to_evaluate}}",
            "isInverted": inverted,
            "evaluationMetricKey": "must-not-be-emitted-offline",
        },
        "meta": {
            "enabled": True,
            "variationKey": key,
            "version": version,
            "mode": mode,
        },
    }


def handler(
    fn: Any,
    *,
    provider: str = "OpenAI",
    mode: Literal["agent", "messages"] = "messages",
) -> ProviderHandler:
    return create_handler((provider, mode), fn)


def test_judge_reference_defaults_and_criteria_wire_shape() -> None:
    assert Accuracy().to_criterion() == {
        "criterionType": "$ld:ai:judge:accuracy",
        "options": {"threshold": 0.5, "passRateThreshold": 1.0},
    }
    assert AnswerRelevancy().key == "$ld:ai:judge:relevance"
    assert Toxicity().threshold == 0.5
    assert Bias().threshold == 0.3
    assert Likeness().ground_truth_context == "{{expected_output}}"
    assert Misinformation().to_criterion()["options"] == {
        "threshold": 0.3,
        "passRateThreshold": 1.0,
        "groundTruthContext": "{{expected_output}}",
    }


def test_judge_references_forbid_typos_and_invalid_thresholds() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Accuracy(threshhold=0.7)  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="less_than_equal"):
        Judge(key="security", threshold=1.1)
    with pytest.raises(ValidationError, match="judge key must not be blank"):
        Judge(key="  ")
    with pytest.raises(ValidationError, match="literal_error"):
        Accuracy(key="different")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_missing_sdk_key_fails_before_initialization_or_resolution() -> None:
    initialize = AsyncMock()
    resolver = AsyncMock(return_value=judge_variation())

    with pytest.raises(EvaluationsError, match="LD_SDK_KEY"):
        await resolve_launchdarkly_judges(
            [Accuracy()],
            [],
            sdk_key=" ",
            resolver=resolver,
            initialize_client=initialize,
        )

    initialize.assert_not_awaited()
    resolver.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_typed_judge_references_are_accepted() -> None:
    raw_references = cast(Sequence[JudgeReference], ["security-judge"])

    with pytest.raises(EvaluationsError, match="typed JudgeReference"):
        await resolve_launchdarkly_judges(
            raw_references,
            [],
            sdk_key="sdk-key",
            initialize_client=AsyncMock(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reference", "expected_key"),
    [
        (Judge(key="security-judge"), "security-judge"),
        (Accuracy(), "$ld:ai:judge:accuracy"),
    ],
)
async def test_unknown_judge_fails_clearly_during_preflight(
    reference: JudgeReference, expected_key: str
) -> None:
    async def missing(key: str, context: dict[str, Any]) -> dict[str, Any]:
        del key, context
        raise RuntimeError("variation returned None")

    with pytest.raises(
        EvaluationsError, match=rf"{re.escape(expected_key)}.*LaunchDarkly UI"
    ):
        await resolve_launchdarkly_judges(
            [reference],
            [],
            sdk_key="sdk-key",
            resolver=missing,
            initialize_client=AsyncMock(),
        )


@pytest.mark.asyncio
async def test_resolved_evaluation_preserves_context_score_and_judge_identity() -> None:
    received: dict[str, Any] = {}

    async def judge_handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Any] | None,
        variables: dict[str, Any] | None,
        history: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        received.update(
            config=config,
            user_input=user_input,
            tool_handlers=tool_handlers,
            variables=variables,
            history=history,
        )
        return {
            "output": '```json\n{"score": 0.82, "reasoning": "matches"}\n```',
            "usage": {"input_tokens": 31, "output_tokens": 7},
        }

    resolver = AsyncMock(
        return_value=judge_variation(key="variation-abc", version=27, inverted=True)
    )
    methods = await resolve_launchdarkly_judges(
        [
            Judge(
                key="security-judge",
                threshold=0.7,
                ground_truth_context="Known: {{expected_output}} / {{account}}",
            )
        ],
        [handler(judge_handler)],
        sdk_key="sdk-key",
        resolver=resolver,
        initialize_client=AsyncMock(),
    )

    result = await methods[0].evaluate(
        "generated answer",
        row_index=41,
        rendered_input="Where is order A19?",
        expected_output="Order A19 shipped",
        variables={"account": "enterprise", "input": "unrendered"},
        metadata={"suite": "orders", "case_id": "stable-41"},
    )

    resolver.assert_awaited_once_with(
        "security-judge",
        {"kind": "user", "key": "offline-evaluation-judge-resolution"},
    )
    assert result.status == "complete"
    assert result.row_index == 41
    assert result.score == 0.82
    assert result.reasoning == "matches"
    assert result.usage.model_dump() == {"input": 31, "output": 7, "total": 38}
    assert result.judge.model_dump() == {
        "key": "security-judge",
        "variation_key": "variation-abc",
        "version": 27,
        "provider": "OpenAI",
        "model": "judge-model",
        "mode": "messages",
        "is_inverted": True,
    }

    assert received["user_input"] == "generated answer"
    assert received["tool_handlers"] is None
    assert received["history"] is None
    judge_variables = cast(Mapping[str, Any], received["variables"])
    assert judge_variables["row_index"] == 41
    assert judge_variables["input"] == "Where is order A19?"
    assert judge_variables["expected_output"] == "Order A19 shipped"
    assert judge_variables["account"] == "enterprise"
    assert judge_variables["metadata"] == {
        "suite": "orders",
        "case_id": "stable-41",
    }
    assert judge_variables["response_to_evaluate"] == "generated answer"
    assert judge_variables["ground_truth_context"] == (
        "Known: Order A19 shipped / enterprise"
    )
    assert "Where is order A19?" in judge_variables["message_history"]
    assert "generated answer" in judge_variables["message_history"]


@pytest.mark.asyncio
async def test_offline_evaluation_never_emits_online_metric_events() -> None:
    tracked: list[tuple[Any, ...]] = []

    class FakeClient:
        def track(self, *args: Any) -> None:
            tracked.append(args)

    client = FakeClient()

    async def initialize(options: dict[str, Any]) -> FakeClient:
        assert options == {"sdkKey": "sdk-key"}
        return client

    async def judge_handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Any] | None,
        variables: dict[str, Any] | None,
        history: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        del config, user_input, tool_handlers, variables, history
        return {"output": '{"score": 1, "reasoning": "safe"}', "usage": {}}

    methods = await resolve_launchdarkly_judges(
        [Judge(key="security-judge")],
        [handler(judge_handler)],
        sdk_key="sdk-key",
        resolver=AsyncMock(return_value=judge_variation()),
        initialize_client=initialize,
    )
    result = await methods[0].evaluate(
        "answer",
        row_index=0,
        rendered_input="question",
        expected_output=None,
        variables={},
        metadata=None,
    )

    assert result.status == "complete"
    assert tracked == []


@pytest.mark.asyncio
async def test_unparseable_judge_response_becomes_diagnosable_error_result() -> None:
    async def judge_handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Any] | None,
        variables: dict[str, Any] | None,
        history: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        del config, user_input, tool_handlers, variables, history
        return {
            "output": "not json",
            "usage": {"inputTokens": 4, "outputTokens": 2},
        }

    methods = await resolve_launchdarkly_judges(
        [Accuracy()],
        [handler(judge_handler)],
        sdk_key="sdk-key",
        resolver=AsyncMock(return_value=judge_variation()),
        initialize_client=AsyncMock(),
    )
    result = await methods[0].evaluate(
        "answer",
        row_index=3,
        rendered_input="question",
        expected_output="expected",
        variables={},
        metadata=None,
    )

    assert result.status == "error"
    assert result.score is None
    assert result.reasoning is None
    assert result.usage.total == 6
    assert result.error is not None
    assert result.error.code == "judge_parse_error"
    assert "invalid JSON" in result.error.message


@pytest.mark.asyncio
async def test_messages_judge_uses_agent_fallback_with_collapsed_prompt() -> None:
    received_configs: list[dict[str, Any]] = []

    async def agent_handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Any] | None,
        variables: dict[str, Any] | None,
        history: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        del user_input, tool_handlers, variables, history
        received_configs.append(config)
        return {"output": '{"score": 0.5, "reasoning": "ok"}', "usage": {}}

    variation = judge_variation()
    variation["config"].pop("instructions")
    variation["config"]["messages"] = [
        {"role": "system", "content": "Apply the rubric."},
        {"role": "user", "content": "Score the response."},
    ]
    methods = await resolve_launchdarkly_judges(
        [Accuracy()],
        [handler(agent_handler, mode="agent")],
        sdk_key="sdk-key",
        resolver=AsyncMock(return_value=variation),
        initialize_client=AsyncMock(),
    )

    await methods[0].evaluate(
        "answer",
        row_index=1,
        rendered_input="question",
        expected_output=None,
        variables={},
        metadata=None,
    )

    assert received_configs == [
        {
            **variation["config"],
            "instructions": "Apply the rubric.\n\nScore the response.",
            "messages": [],
        }
    ]


@pytest.mark.asyncio
async def test_missing_compatible_handler_fails_during_resolution() -> None:
    async def unused_handler(
        config: dict[str, Any],
        user_input: str | None,
        tool_handlers: dict[str, Any] | None,
        variables: dict[str, Any] | None,
        history: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        del config, user_input, tool_handlers, variables, history
        return {}

    with pytest.raises(EvaluationsError, match=r"No handler.*Anthropic"):
        await resolve_launchdarkly_judges(
            [Accuracy()],
            [handler(unused_handler, provider="OpenAI")],
            sdk_key="sdk-key",
            resolver=AsyncMock(
                return_value=judge_variation(provider="Anthropic", mode="messages")
            ),
            initialize_client=AsyncMock(),
        )
