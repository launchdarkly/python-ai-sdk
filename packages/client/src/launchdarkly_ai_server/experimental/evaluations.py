"""
Evaluations from Code (experimental).

Create a harness with ``init_evaluations``, then call ``run()`` on it to generate an
output for every dataset row and score it against LaunchDarkly judges or local
scorers. These names may change in a minor release, so review the changelog when
you upgrade.
"""

from ..evaluations.api import EvaluationsError
from ..evaluations.criteria import Criterion, Judge, Scorer
from ..evaluations.module import EvaluationsModule, init_evaluations
from ..evaluations.tools import EvalTool
from ..evaluations.types import (
    AIConfig,
    DatasetRow,
    EvalRunResult,
    GenerationConfig,
    GenerationMode,
    GenerationOverrides,
    RunSummary,
)

__all__ = [  # noqa: RUF022
    # entry point
    "init_evaluations",
    "EvaluationsModule",
    # run() arguments
    "AIConfig",
    "GenerationConfig",
    "GenerationMode",
    "GenerationOverrides",
    "DatasetRow",
    "EvalTool",
    # criteria
    "Criterion",
    "Judge",
    "Scorer",
    # results and errors
    "EvalRunResult",
    "RunSummary",
    "EvaluationsError",
]
