"""Question extraction and Jev answer scoring for a TypeSafe judge.

``extract_typesafe_questions`` is the only reader of the judge's ``classifiers``
list.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from launchdarkly_ai_server.judge_scoring import FORMATTING_INSTRUCTIONS

QuestionType = Literal["noul", "choice", "score"]
_QUESTION_TYPES = {"noul", "choice", "score"}


class TypesafeQuestionError(ValueError):
    """The judge config does not describe a usable set of Jev questions."""


def extract_typesafe_questions(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the Jev questions declared on ``config['classifiers']``."""
    labels = config.get("classifiers")
    if not isinstance(labels, list) or not labels:
        raise TypesafeQuestionError("classifiers must be a non-empty list")

    seen: set[str] = set()
    seen_events: set[str] = set()
    questions: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        if not isinstance(label, Mapping):
            raise TypesafeQuestionError(f"classifiers[{index}] must be an object")
        key = label.get("key")
        if not isinstance(key, str) or not key.strip():
            raise TypesafeQuestionError(f"classifiers[{index}] is missing a key")
        if key in seen:
            raise TypesafeQuestionError(f"classifiers has a duplicate key {key!r}")
        seen.add(key)
        event_key = label.get("eventKey")
        if not isinstance(event_key, str) or not event_key.strip():
            raise TypesafeQuestionError(
                f"classifiers label {key!r} is missing an eventKey"
            )
        if event_key in seen_events:
            raise TypesafeQuestionError(
                f"classifiers has a duplicate eventKey {event_key!r}"
            )
        seen_events.add(event_key)
        question_type = label.get("type")
        if question_type not in _QUESTION_TYPES:
            raise TypesafeQuestionError(
                f"classifiers label {key!r} has an unsupported type"
            )
        instructions = label.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise TypesafeQuestionError(
                f"classifiers label {key!r} is missing instructions"
            )
        question: dict[str, Any] = {
            "key": key,
            "eventKey": event_key,
            "type": question_type,
            "instructions": instructions,
        }
        criteria = _criteria_for(key, question_type, label.get("criteria"))
        if criteria is not None:
            question["criteria"] = criteria
        questions.append(question)
    return questions


def _criteria_for(key: str, question_type: str, raw: Any) -> Any:
    if question_type == "noul":
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise TypesafeQuestionError(
                f"classifiers label {key!r} has invalid noul criteria"
            )
        criteria: dict[str, str | None] = {}
        for name in ("true", "false"):
            if name not in raw:
                continue
            description = raw[name]
            if description is not None and not isinstance(description, str):
                raise TypesafeQuestionError(
                    f"classifiers label {key!r} has a non-string noul description"
                )
            criteria[name] = description
        return criteria or None
    if question_type == "choice":
        if not isinstance(raw, Mapping) or not raw:
            raise TypesafeQuestionError(
                f"classifiers label {key!r} requires choice criteria"
            )
        choice_criteria: dict[str, str | None] = {}
        for name, description in raw.items():
            if not isinstance(name, str) or not name.strip():
                raise TypesafeQuestionError(
                    f"classifiers label {key!r} has an empty choice"
                )
            if description is not None and not isinstance(description, str):
                raise TypesafeQuestionError(
                    f"classifiers label {key!r} has a non-string choice description"
                )
            choice_criteria[name] = description
        return choice_criteria
    if not isinstance(raw, list) or len(raw) < 2:
        raise TypesafeQuestionError(
            f"classifiers label {key!r} requires at least two score levels"
        )
    levels: list[str | None] = []
    for level in raw:
        if level is not None and not isinstance(level, str):
            raise TypesafeQuestionError(
                f"classifiers label {key!r} has a non-string score level"
            )
        levels.append(level)
    return levels


def without_formatting_instructions(value: Any) -> str | None:
    """Drop the shared judge JSON-format block. Jev does not use it."""
    if not isinstance(value, str):
        return None
    index = value.find(FORMATTING_INSTRUCTIONS)
    text = value if index == -1 else value[:index]
    text = text.strip()
    return text or None


def build_typesafe_state(
    *,
    user_input: str | None = None,
    variables: Mapping[str, Any] | None = None,
    history: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """The state sent to Jev: the conversation, without score-format instructions."""
    variables = variables or {}
    state: dict[str, Any] = {}

    def assign(key: str, value: Any) -> None:
        if value is None or value == "":
            return
        state[key] = value

    assign("input", variables.get("input"))
    output = variables.get("response_to_evaluate", user_input)
    assign("output", output)
    assign(
        "message_history",
        without_formatting_instructions(variables.get("message_history")),
    )
    assign("trajectory", variables.get("trajectory"))
    assign("expected_output", variables.get("expected_output"))
    assign("ground_truth_context", variables.get("ground_truth_context"))
    if user_input and user_input != output:
        assign("user_input", user_input)
    if history:
        state["history"] = list(history)
    return state


def _read(answer: Any, field: str) -> Any:
    if isinstance(answer, Mapping):
        return answer.get(field)
    return getattr(answer, field, None)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def score_from_answer(question: Mapping[str, Any], answer: Any) -> float:
    """Map one Jev answer onto the 0–1 score stored in ``judge_results``."""
    key = question["key"]
    question_type = question["type"]
    if question_type == "noul":
        noul = _finite(_read(answer, "noul"))
        if noul is None:
            raise TypesafeQuestionError(
                f"TypeSafe answer {key!r} is missing a noul probability"
            )
        return _clamp(noul)
    if question_type == "choice":
        selected = _read(answer, "choice")
        probabilities = _read(answer, "probabilities")
        if isinstance(selected, str) and isinstance(probabilities, Mapping):
            selected_probability = _finite(probabilities.get(selected))
            if selected_probability is not None:
                return _clamp(selected_probability)
        confidence = _finite(_read(answer, "confidence"))
        if confidence is not None:
            return _clamp(confidence)
        if isinstance(selected, str) and selected:
            return 1.0
        raise TypesafeQuestionError(f"TypeSafe answer {key!r} is missing a choice")

    score = _finite(_read(answer, "score"))
    if score is None:
        raise TypesafeQuestionError(f"TypeSafe answer {key!r} is missing a score")
    legend = _read(answer, "legend")
    criteria = question.get("criteria")
    levels = (
        len(legend)
        if isinstance(legend, Mapping)
        else len(criteria)
        if isinstance(criteria, list)
        else 0
    )
    # Jev's score is an expected rubric index. A value in 0–1 is the low end of
    # a longer rubric, not a probability. Two levels divide by 1.
    if levels > 1:
        return _clamp(score / (levels - 1))
    return _clamp(score)


def _criterion_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _legend_level(key: Any) -> int | None:
    if isinstance(key, bool):
        return None
    if isinstance(key, int):
        return key
    if isinstance(key, str) and key.lstrip("-").isdigit():
        return int(key)
    return None


def _nearest_key(score: float, keys: Sequence[int]) -> int:
    """Integer closest to ``score``. Ties resolve to the lower key."""
    return min(keys, key=lambda key: (abs(key - score), key))


def reason_from_answer(question: Mapping[str, Any], answer: Any) -> str:
    """The selected label's value, stored as the judge result reason.

    A noul at or above 0.5 selects ``true``. A score uses the legend entry
    nearest the raw score.
    """
    question_type = question.get("type")
    criteria = question.get("criteria")
    if question_type == "noul":
        noul = _finite(_read(answer, "noul"))
        side = "true" if noul is not None and noul >= 0.5 else "false"
        if isinstance(criteria, Mapping):
            described = _criterion_text(criteria.get(side))
            if described is not None:
                return described
        return side
    if question_type == "choice":
        selected = _read(answer, "choice")
        name = selected if isinstance(selected, str) else ""
        if name and isinstance(criteria, Mapping):
            described = _criterion_text(criteria.get(name))
            if described is not None:
                return described
        return name

    score = _finite(_read(answer, "score"))
    if score is None:
        return ""
    legend = _read(answer, "legend")
    if isinstance(legend, Mapping) and legend:
        keys: list[int] = []
        for legend_key in legend:
            level = _legend_level(legend_key)
            if level is not None:
                keys.append(level)
        if keys:
            level = _nearest_key(score, keys)
            described = _criterion_text(legend.get(level))
            if described is None:
                described = _criterion_text(legend.get(str(level)))
            if described is not None:
                return described
    if isinstance(criteria, list) and criteria:
        level = _nearest_key(score, list(range(len(criteria))))
        described = _criterion_text(criteria[level])
        if described is not None:
            return described
    return ""


def typesafe_output(results: list[dict[str, Any]]) -> str:
    import json

    return json.dumps({"kind": "typesafe", "results": results})
