from __future__ import annotations

from typing import Any, cast

import pytest

from launchdarkly_ai_server.evaluations.scorers import (
    Scorer,
    ScorerRow,
    ScoreValue,
)


def scorer_row() -> ScorerRow:
    return ScorerRow(
        row_index=7,
        input="Rendered order A19",
        expected_output="Refund A19",
        variables={"order_id": "A19", "input": "Rendered order A19"},
        metadata={"suite": "refunds", "priority": 1},
    )


@pytest.mark.asyncio
async def test_sync_scorer_receives_generation_output_and_row_context() -> None:
    received: dict[str, Any] = {}

    def score(row: ScorerRow, output: str | None) -> float:
        received.update(
            {
                "row_index": row.row_index,
                "input": row.input,
                "expected_output": row.expected_output,
                "variables": dict(row.variables),
                "metadata": dict(row.metadata or {}),
                "output": output,
            }
        )
        return 0.75

    result = await Scorer(name="refund-exists", fn=score).execute(
        scorer_row(), "Refund created"
    )

    assert received == {
        "row_index": 7,
        "input": "Rendered order A19",
        "expected_output": "Refund A19",
        "variables": {"order_id": "A19", "input": "Rendered order A19"},
        "metadata": {"suite": "refunds", "priority": 1},
        "output": "Refund created",
    }
    assert result.scorer_name == "refund-exists"
    assert result.row_index == 7
    assert result.score == 0.75
    assert result.status == "COMPLETE"
    assert result.error is None
    assert result.started_at.tzinfo is not None
    assert result.evaluated_at >= result.started_at
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_async_scorer_is_awaited() -> None:
    called = False

    async def score(row: ScorerRow, output: str | None) -> float:
        nonlocal called
        called = True
        assert row.metadata == {"suite": "refunds", "priority": 1}
        assert output == "done"
        return 0.4

    result = await Scorer(name="async-check", fn=score).execute(scorer_row(), "done")

    assert called is True
    assert result.status == "COMPLETE"
    assert result.score == 0.4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_score", "normalized"),
    [(True, 1.0), (False, 0.0), (0, 0.0), (1, 1.0), (0.625, 0.625)],
)
async def test_bool_and_numeric_scores_are_normalized(
    raw_score: ScoreValue, normalized: float
) -> None:
    def score(row: ScorerRow, output: str | None) -> ScoreValue:
        del row, output
        return raw_score

    result = await Scorer(name="normalized", fn=score).execute(scorer_row(), "ok")

    assert result.status == "COMPLETE"
    assert result.score == normalized


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_score", [None, "1", -0.01, 1.01, float("nan"), float("inf")]
)
async def test_invalid_scorer_results_are_clear_error_results(
    raw_score: object,
) -> None:
    def score(row: ScorerRow, output: str | None) -> ScoreValue:
        del row, output
        return cast(ScoreValue, raw_score)

    result = await Scorer(name="bad-result", fn=score).execute(scorer_row(), "ok")

    assert result.scorer_name == "bad-result"
    assert result.row_index == 7
    assert result.status == "ERROR"
    assert result.score is None
    assert result.error is not None
    assert result.error.code == "invalid_score"
    assert "between 0 and 1" in result.error.message
    assert result.started_at.tzinfo is not None
    assert result.evaluated_at >= result.started_at
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_scorer_exception_is_preserved_as_error_result() -> None:
    def score(row: ScorerRow, output: str | None) -> float:
        del row, output
        raise RuntimeError("database unavailable")

    result = await Scorer(name="db-check", fn=score).execute(scorer_row(), "ok")

    assert result.status == "ERROR"
    assert result.score is None
    assert result.error is not None
    assert result.error.code == "scorer_error"
    assert result.error.exception_type == "RuntimeError"
    assert result.error.message == "database unavailable"


def test_scorer_and_row_dtos_validate_strictly() -> None:
    with pytest.raises(ValueError, match="name"):
        Scorer(name=" ", fn=lambda row, output: True)
    with pytest.raises(ValueError, match="between 0 and 1"):
        Scorer(name="check", fn=lambda row, output: True, threshold=1.1)
    with pytest.raises(ValueError, match="row_index"):
        ScorerRow(
            row_index=-1,
            input=None,
            expected_output=None,
            variables={},
            metadata=None,
        )
    with pytest.raises(TypeError, match="metadata keys"):
        ScorerRow(
            row_index=0,
            input=None,
            expected_output=None,
            variables={},
            metadata=cast(dict[str, Any], {1: "invalid"}),
        )
