from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from .types import DatasetRow

type ScorerFn = Callable[[DatasetRow, Any], float | bool | Awaitable[float | bool]]

type SuccessDirection = Literal["higher_is_better", "lower_is_better"]

#: The threshold a judge is held to when its reference sets none. A criterion
#: with no threshold gives LaunchDarkly nothing to compare a score against, so
#: it would be recorded and never ruled on; defaulting is what keeps a judge
#: usable as a gate without every caller restating the obvious.
DEFAULT_JUDGE_THRESHOLD = 0.5


@dataclass(frozen=True)
class Judge:
    """Reference to a LaunchDarkly AI Judge config to run for each eval row.

    The SDK does not create or provide built-in judges. Pass the key of a judge
    that exists in LaunchDarkly. Resolution uses LaunchDarkly flag delivery for
    the currently served variation.

    A judge's success direction is not set here. It lives on the judge's AI
    Config as ``isInverted`` and is injected onto the criterion by LaunchDarkly
    when the evaluation is created, so the one input a verdict is ruled on stays
    server-attested even though the score beside it is client-reported.
    """

    key: str
    threshold: float | None = DEFAULT_JUDGE_THRESHOLD
    pass_rate_threshold: float | None = None
    ground_truth_context: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("judge key must not be blank")
        _validate_thresholds(
            threshold=self.threshold,
            pass_rate_threshold=self.pass_rate_threshold,
        )

    @property
    def criterion_type(self) -> str:
        return self.key

    def to_criteria_wire(self) -> dict[str, Any]:
        options = _criteria_options(
            threshold=self.threshold,
            pass_rate_threshold=self.pass_rate_threshold,
            ground_truth_context=self.ground_truth_context,
        )
        return {
            "criterionType": self.criterion_type,
            "kind": "judge",
            "judgeKey": self.key,
            "options": options,
        }


@dataclass(frozen=True)
class Scorer:
    """Local deterministic scorer run for each generated evaluation row.

    ``fn`` may be sync or async and receives ``(row, output)``, where ``row``
    is the :class:`~launchdarkly_ai_server.evaluations.types.DatasetRow` the
    output was generated from and ``output`` is the generated output. It must
    return a boolean or a numeric score from 0 to 1. Boolean results are
    converted to 1.0 or 0.0 before being emitted as evaluation events.

    ``threshold`` defaults to 1.0: a row passes only on a perfect score, which
    matches the common case of boolean scorers. Pass a lower threshold for
    graded numeric scorers.

    ``success_direction`` says which way the score points, and defaults to
    higher-is-better. Unlike a judge, a scorer has no LaunchDarkly-side config
    to read a direction from, so the declaration here is the only source --
    LaunchDarkly derives each row's verdict by comparing score to threshold in
    this direction. Set ``"lower_is_better"`` for a scorer that counts
    something unwanted, e.g. a regex hit count or an edit distance.
    """

    name: str
    fn: ScorerFn
    threshold: float | None = 1.0
    pass_rate_threshold: float | None = None
    success_direction: SuccessDirection = "higher_is_better"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("scorer name must not be blank")
        if not callable(self.fn):
            raise ValueError("scorer fn must be callable")
        _validate_thresholds(
            threshold=self.threshold,
            pass_rate_threshold=self.pass_rate_threshold,
        )

    @property
    def criterion_type(self) -> str:
        return self.name

    def to_criteria_wire(self) -> dict[str, Any]:
        return {
            "criterionType": self.criterion_type,
            "kind": "scorer",
            "successDirection": self.success_direction,
            "options": _criteria_options(
                threshold=self.threshold,
                pass_rate_threshold=self.pass_rate_threshold,
            ),
        }


type Criterion = Judge | Scorer


def _validate_thresholds(
    *,
    threshold: float | None,
    pass_rate_threshold: float | None,
) -> None:
    for name, value in (
        ("threshold", threshold),
        ("pass_rate_threshold", pass_rate_threshold),
    ):
        if value is not None and (value < 0 or value > 1):
            raise ValueError(f"{name} must be between 0 and 1")


def _criteria_options(
    *,
    threshold: float | None,
    pass_rate_threshold: float | None,
    ground_truth_context: str | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if threshold is not None:
        options["threshold"] = threshold
    if pass_rate_threshold is not None:
        options["passRateThreshold"] = pass_rate_threshold
    if ground_truth_context is not None:
        options["groundTruthContext"] = ground_truth_context
    return options
