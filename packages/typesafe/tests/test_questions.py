import pytest

from launchdarkly_ai_server.judge_scoring import FORMATTING_INSTRUCTIONS
from launchdarkly_ai_typesafe.questions import (
    TypesafeQuestionError,
    build_typesafe_state,
    extract_typesafe_questions,
    reason_from_answer,
    score_from_answer,
)

CLASSIFIERS = [
    {
        "key": "accuracy",
        "eventKey": "$ld:ai:judge:jev:accuracy",
        "instructions": "How accurate is the response?",
        "type": "score",
        "criteria": ["Great", "Ok", "Bad"],
    },
    {
        "key": "tone",
        "eventKey": "$ld:ai:judge:jev:tone",
        "instructions": "Is the tone appropriate?",
        "type": "noul",
        "criteria": {"true": "Polite", "false": "Harmful"},
    },
    {
        "key": "route",
        "eventKey": "$ld:ai:judge:jev:route",
        "instructions": "What is the tone?",
        "type": "choice",
        "criteria": {"calm": "Calm", "angry": "Angry"},
    },
]


def _config(labels: object = CLASSIFIERS) -> dict:
    return {
        "model": {"name": "jev-latest"},
        "provider": {"name": "TypeSafe"},
        "messages": [],
        "classifiers": labels,
    }


def test_extracts_all_three_question_types() -> None:
    assert extract_typesafe_questions(_config()) == [
        {
            "key": "accuracy",
            "eventKey": "$ld:ai:judge:jev:accuracy",
            "type": "score",
            "instructions": "How accurate is the response?",
            "criteria": ["Great", "Ok", "Bad"],
        },
        {
            "key": "tone",
            "eventKey": "$ld:ai:judge:jev:tone",
            "type": "noul",
            "instructions": "Is the tone appropriate?",
            "criteria": {"true": "Polite", "false": "Harmful"},
        },
        {
            "key": "route",
            "eventKey": "$ld:ai:judge:jev:route",
            "type": "choice",
            "instructions": "What is the tone?",
            "criteria": {"calm": "Calm", "angry": "Angry"},
        },
    ]


def test_rejects_a_config_without_labels() -> None:
    with pytest.raises(TypesafeQuestionError, match="classifiers"):
        extract_typesafe_questions(_config([]))
    with pytest.raises(TypesafeQuestionError, match="classifiers"):
        extract_typesafe_questions(
            {
                "model": {"name": "jev"},
                "provider": {"name": "TypeSafe"},
                "instructions": "unused",
            }
        )


def test_state_drops_formatting_instructions() -> None:
    state = build_typesafe_state(
        user_input="the answer",
        variables={
            "input": "the question",
            "response_to_evaluate": "the answer",
            "message_history": f"the question\n\n{FORMATTING_INSTRUCTIONS}",
            "trajectory": "called search",
        },
        history=[{"role": "user", "content": "earlier"}],
    )
    assert state == {
        "input": "the question",
        "output": "the answer",
        "message_history": "the question",
        "trajectory": "called search",
        "history": [{"role": "user", "content": "earlier"}],
    }
    assert "valid JSON format" not in str(state)


def test_score_mapping() -> None:
    noul = {"key": "migration", "type": "noul", "instructions": "migration?"}
    choice = {"key": "tone", "type": "choice", "instructions": "tone?"}
    scored = {
        "key": "urgency",
        "type": "score",
        "instructions": "urgency?",
        "criteria": ["a", "b", "c"],
    }
    assert score_from_answer(noul, {"noul": 0.82}) == 0.82
    assert (
        score_from_answer(
            choice,
            {
                "choice": "angry",
                "probabilities": {"calm": 0.2, "angry": 0.8},
                "confidence": 0.1,
            },
        )
        == 0.8
    )
    legend = {0: "a", 1: "b", 2: "c"}
    assert score_from_answer(scored, {"score": 0.4, "legend": legend}) == pytest.approx(
        0.2
    )
    assert score_from_answer(scored, {"score": 1, "legend": legend}) == 0.5
    assert score_from_answer(scored, {"score": 1.7, "legend": legend}) == pytest.approx(
        0.85
    )
    binary = {**scored, "criteria": ["a", "b"]}
    assert score_from_answer(binary, {"score": 0.4, "legend": {0: "a", 1: "b"}}) == 0.4


def test_reason_is_the_selected_label_value() -> None:
    noul = {
        "key": "tone",
        "type": "noul",
        "instructions": "tone?",
        "criteria": {"true": "Polite", "false": "Harmful"},
    }
    choice = {
        "key": "route",
        "type": "choice",
        "instructions": "route?",
        "criteria": {"calm": "A neutral message", "angry": None},
    }
    scored = {
        "key": "urgency",
        "type": "score",
        "instructions": "urgency?",
        "criteria": ["Great", "Ok", "Bad"],
    }
    legend = {0: "Great", 1: "Ok", 2: "Bad"}
    assert reason_from_answer(noul, {"noul": 0.5}) == "Polite"
    assert reason_from_answer(noul, {"noul": 0.49}) == "Harmful"
    assert reason_from_answer({"key": "tone", "type": "noul"}, {"noul": 0.9}) == "true"
    assert reason_from_answer(choice, {"choice": "calm"}) == "A neutral message"
    assert reason_from_answer(choice, {"choice": "angry"}) == "angry"
    assert reason_from_answer(scored, {"score": 0.4, "legend": legend}) == "Great"
    assert reason_from_answer(scored, {"score": 1.5, "legend": legend}) == "Ok"
    assert reason_from_answer(scored, {"score": 1.7, "legend": legend}) == "Bad"
    assert reason_from_answer(scored, {"score": 1}) == "Ok"
