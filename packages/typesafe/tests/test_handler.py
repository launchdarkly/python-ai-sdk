import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from launchdarkly_ai_server.judge_scoring import FORMATTING_INSTRUCTIONS
from launchdarkly_ai_typesafe import create_typesafe_handler

CLASSIFIERS = [
    {
        "key": "tone",
        "eventKey": "$ld:ai:judge:jev:tone",
        "instructions": "Is the tone appropriate?",
        "type": "noul",
        "criteria": {"true": "Polite", "false": "Harmful"},
    },
    {
        "key": "accuracy",
        "eventKey": "$ld:ai:judge:jev:accuracy",
        "instructions": "How accurate is the response?",
        "type": "score",
        "criteria": ["Great", "Ok", "Bad"],
    },
]


def _client(response: object) -> MagicMock:
    client = MagicMock()
    client.system_one = AsyncMock(return_value=response)
    entered = MagicMock()
    entered.__aenter__ = AsyncMock(return_value=client)
    entered.__aexit__ = AsyncMock(return_value=None)
    factory = MagicMock(return_value=entered)
    return factory


@pytest.mark.asyncio
async def test_provides_for_typesafe_messages() -> None:
    handler = create_typesafe_handler()
    assert handler.provides_for == ("TypeSafe", "messages")
    assert handler.capture_content is False


@pytest.mark.asyncio
async def test_sends_labels_and_returns_one_score_per_label() -> None:
    response = SimpleNamespace(
        answers={
            "tone": SimpleNamespace(noul=0.25),
            "accuracy": SimpleNamespace(score=1),
        },
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )
    factory = _client(response)
    handler = create_typesafe_handler()
    with patch("launchdarkly_ai_typesafe.handler.AsyncTypeSafeClient", factory):
        result = await handler(
            {
                "model": {"name": "jev-latest"},
                "provider": {"name": "TypeSafe"},
                "messages": [],
                "classifiers": CLASSIFIERS,
            },
            "assistant output",
            None,
            {
                "input": "user question",
                "response_to_evaluate": "assistant output",
                "message_history": (
                    "user question\n\nassistant output\n\n" + FORMATTING_INSTRUCTIONS
                ),
            },
        )

    state = factory.return_value.__aenter__.return_value.system_one.await_args.kwargs
    request_state = state["state"]
    assert state["model"] == "jev-latest"
    tone = state["questions"]["tone"]
    assert tone.instructions == "Is the tone appropriate?"
    assert tone.criteria["true"] == "Polite"
    assert state["questions"]["accuracy"].criteria == ["Great", "Ok", "Bad"]
    assert "valid JSON format" not in str(request_state)
    assert request_state["input"] == "user question"
    assert request_state["output"] == "assistant output"
    assert json.loads(result["output"]) == {
        "kind": "typesafe",
        "results": [
            {
                "key": "tone",
                "eventKey": "$ld:ai:judge:jev:tone",
                "score": 0.25,
                "reason": "Harmful",
            },
            {
                "key": "accuracy",
                "eventKey": "$ld:ai:judge:jev:accuracy",
                "score": 0.5,
                "reason": "Ok",
            },
        ],
    }
    assert result["usage"] == {"input_tokens": 100, "output_tokens": 20}


@pytest.mark.asyncio
async def test_propagates_client_failure() -> None:
    factory = _client(SimpleNamespace())
    factory.return_value.__aenter__.return_value.system_one.side_effect = RuntimeError(
        "typesafe down"
    )
    handler = create_typesafe_handler()
    with (
        patch("launchdarkly_ai_typesafe.handler.AsyncTypeSafeClient", factory),
        pytest.raises(RuntimeError, match="typesafe down"),
    ):
        await handler(
            {
                "model": {"name": "jev"},
                "messages": [],
                "classifiers": CLASSIFIERS,
                "provider": {"name": "TypeSafe"},
                "instructions": "ignored",
            },
            "output",
        )
