"""Shared contract for LaunchDarkly AI Judge invocations.

Three judge execution paths exist — the online inline path
(``judges.run_judges``, sampled per invocation), the online deferred path
(``judges.run_judge``, from a ``JudgeTask`` on a background thread), and the
offline evaluations path (``evaluations.runner``). All three prompt a judge
model for the same ``{"score": <0-1>, "reasoning": <string>}`` JSON shape, and
all three must show the judge the same conversation. This module owns both
halves of that contract so the paths cannot drift.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from .utils import parse_json_with_possible_fences

FORMATTING_INSTRUCTIONS = "\n".join(
    [
        "Your response MUST be in valid JSON format with the following structure:",
        '{ "score": <number, 0-1>, "reasoning": <string> }',
        "The output must be valid, parseable JSON. Do not include additional tags, comments, "
        "formatting, or newlines.",
        "It should be returned in a format that is immediately parseable by a JSON parsing "
        "function. Do not include ```json tags.",
    ]
)


def build_message_history(
    *,
    user_input: Any = None,
    trajectory: Any = None,
    output: Any = None,
) -> str:
    """The conversation a judge is shown, as its ``message_history`` variable.

    Ordered as it happened: what was asked, what the agent did, what it
    answered, then how to format the verdict. Empty parts are skipped, so a
    run with no tools yields the history it did before trajectories existed.

    The formatting block is appended here, not by callers: judges built from
    the AI Library's default templates read the JSON shape from
    ``{{message_history}}``, and one that stopped being told it would return
    prose and fail every result as invalid output.
    """
    return "\n\n".join(
        str(part)
        for part in (user_input, trajectory, output, FORMATTING_INSTRUCTIONS)
        if part
    )


def numeric_score(score: Any) -> float | None:
    """Return ``score`` as a float only when it already is a finite number.

    Never raises. A judge that returns ``"0.9 (high)"`` or ``None`` must not take down the
    evaluation metric track that follows, and must not put a string where semconv defines a double.
    """
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    value = float(score)
    return value if isfinite(value) else None


def typesafe_judge_entries(raw: Any) -> list[dict[str, Any]] | None:
    """Parse a TypeSafe handler payload into question key, score, event key, and reason.

    Returns ``None`` when ``raw`` is not a TypeSafe multi-result payload, so a
    classic ``{"score", "reasoning"}`` judge stays on :func:`parse_judge_response`.
    Raises ``ValueError`` when the payload claims to be TypeSafe but is malformed.
    """
    parsed: Any
    if isinstance(raw, str):
        parsed = parse_json_with_possible_fences(raw)
    elif isinstance(raw, Mapping):
        parsed = raw
    else:
        parsed = None
    if not isinstance(parsed, Mapping) or parsed.get("kind") != "typesafe":
        return None
    results = parsed.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("TypeSafe judge output is missing results")
    entries: list[dict[str, Any]] = []
    for item in results:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("key"), str)
            or not item["key"]
        ):
            raise ValueError("TypeSafe judge result is missing a key")
        score = numeric_score(item.get("score"))
        if score is None:
            raise ValueError(
                f"TypeSafe judge result {item['key']!r} is missing a finite score"
            )
        event_key = item.get("eventKey")
        if not isinstance(event_key, str) or not event_key:
            raise ValueError(
                f"TypeSafe judge result {item['key']!r} is missing an eventKey"
            )
        reason = item.get("reason")
        entries.append(
            {
                "key": item["key"],
                "score": score,
                "eventKey": event_key,
                "reason": reason if isinstance(reason, str) else "",
            }
        )
    return entries


def parse_judge_response(raw: Any) -> tuple[Any, str]:
    """Parse a judge model response into ``(score, reasoning)``.

    Accepts a JSON string (possibly wrapped in markdown fences) or an
    already-decoded mapping. The score is returned untouched — callers apply
    their own policy to non-numeric values via :func:`numeric_score`.

    Raises ``ValueError`` when the response is not a non-empty JSON object.
    """
    parsed: Any
    if isinstance(raw, Mapping):
        parsed = raw
    elif isinstance(raw, str):
        parsed = parse_json_with_possible_fences(raw)
    else:
        parsed = None
    if not isinstance(parsed, Mapping) or not parsed:
        raise ValueError("Invalid JSON from judge")
    reasoning = parsed.get("reasoning") or parsed.get("reason") or ""
    return parsed.get("score"), str(reasoning)
