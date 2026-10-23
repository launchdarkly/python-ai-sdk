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

import logging
from collections.abc import Mapping
from math import isfinite
from typing import Any

from .content import tool_result_string
from .utils import parse_json_with_possible_fences

logger = logging.getLogger(__name__)

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
    answered, then how to format the verdict. An empty part (see
    :func:`_is_empty_part`) is skipped, so a run with no tools yields the
    history it did before trajectories existed.

    The formatting block is appended here, not by callers: judges built from
    the AI Library's default templates read the JSON shape from
    ``{{message_history}}``, and one that stopped being told it would return
    prose and fail every result as invalid output.

    Every other part is rendered by the rule tool results follow, so a
    non-string output is shown as JSON and a rubric reads the same text in
    every SDK. An output of ``0`` or ``False`` is an answer, and is shown.
    """
    return "\n\n".join(
        tool_result_string(part)
        for part in (user_input, trajectory, output, FORMATTING_INSTRUCTIONS)
        if not _is_empty_part(part)
    )


def _is_empty_part(value: Any) -> bool:
    """``None``, ``""``, an empty list or an empty mapping, tested before rendering.

    An empty container is skipped, as it was before structured outputs were
    rendered as JSON. Python truthiness is not the test: it would also drop
    ``0`` and ``False``, which are answers.
    """
    if value is None or value == "":
        return True
    return isinstance(value, (list, tuple, Mapping)) and not value


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

    The reasoning is the first of ``reasoning`` and ``reason`` that is not an
    empty part (:func:`_is_empty_part`), rendered by the tool-result rule: a
    string as it is, anything else as JSON, never as a Python repr.

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
    reasoning = next(
        (
            tool_result_string(value)
            for value in (parsed.get("reasoning"), parsed.get("reason"))
            if not _is_empty_part(value)
        ),
        "",
    )
    return parsed.get("score"), reasoning


def without_output_format(config: Any, judge_key: str) -> Any:
    """Return a judge config with its ``outputFormat`` removed.

    A judge is an ordinary AI Config, so it can carry an ``outputFormat`` JSON
    Schema. The verdict shape belongs to this module, not to the judge's
    author, and every provider handler that honors ``outputFormat`` would
    constrain the judge model to the author's schema instead, so the verdict
    could never parse. Every judge path strips the field before a config
    reaches a handler or a :class:`JudgeTask`, and says so once per judge.

    Returns ``config`` itself when the key is absent. Never mutates it: it
    came from ``extract_variation`` and may be cached.
    """
    if not isinstance(config, Mapping) or "outputFormat" not in config:
        return config
    logger.warning(
        "Judge '%s': ignoring outputFormat - a judge must return {score, reasoning}.",
        judge_key,
    )
    return {k: v for k, v in config.items() if k != "outputFormat"}
