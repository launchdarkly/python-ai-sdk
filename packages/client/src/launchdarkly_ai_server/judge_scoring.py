"""Shared contract for LaunchDarkly AI Judge invocations.

Three judge execution paths exist — the online inline path
(``judges.run_judges``, sampled per invocation), the online deferred path
(``judges.run_judge``, from a ``JudgeTask`` on a background thread), and the
offline evaluations path (``evaluations.runner``). All three prompt a judge
model for the same ``{"score": <0-1>, "reasoning": <string>}`` JSON shape, and
all three must show the judge the same conversation. This module owns both
halves of that contract so the paths cannot drift.

They did drift: each path joined its own, and the deferred one carried neither
input nor trajectory -- so a judge grading the same response saw a different
conversation depending on which path reached it.
:func:`build_message_history` is now the only place it is built.
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
