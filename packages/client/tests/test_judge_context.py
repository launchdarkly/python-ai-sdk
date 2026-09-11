"""
Tests for grounded judge context and per-judge diagnostics.

Everything here uses hand-written doubles: no real LaunchDarkly client, no network.
"""

import asyncio
import json
import random
from collections.abc import AsyncGenerator, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import (
    JsonValue,
    JudgeTask,
    ProviderHandler,
    build_judge_tasks,
    config,
    run_judge,
)
from launchdarkly_ai_server.judges import EVIDENCE_BEGIN, EVIDENCE_END

CONTEXT = {"kind": "user", "key": "u1"}

MAIN_META = {
    "enabled": True,
    "variationKey": "v1",
    "version": 1,
    "mode": "messages",
}
JUDGE_META = {
    "enabled": True,
    "variationKey": "j1",
    "version": 1,
    "mode": "judge",
}


def _main_variation(
    judges: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    variation: dict[str, Any] = {
        "model": {"name": "gpt-4"},
        "provider": {"name": "TestProvider"},
        "instructions": "Be helpful.",
        "_ldMeta": MAIN_META,
    }
    if judges is not None:
        variation["judgeConfiguration"] = {"judges": judges}
    return variation


def _judge_variation(evaluation_metric_key: str | None = None) -> dict[str, Any]:
    variation: dict[str, Any] = {
        "model": {"name": "gpt-4"},
        "provider": {"name": "TestProvider"},
        "instructions": "You are a judge.",
        "_ldMeta": JUDGE_META,
    }
    if evaluation_metric_key:
        variation["evaluationMetricKey"] = evaluation_metric_key
    return variation


def _client(variations: dict[str, Any]) -> MagicMock:
    """LD client double that answers `variation(key, ...)` from a key -> value map."""
    client = MagicMock()
    client.track = MagicMock()
    client.flush = AsyncMock()
    client.close = AsyncMock()

    async def variation(key: str, *_args: object, **_kwargs: object) -> Any:
        value = variations[key]
        if isinstance(value, Exception):
            raise value
        return value

    client.variation = AsyncMock(side_effect=variation)
    return client


def _install(variations: dict[str, Any]) -> MagicMock:
    client = _client(variations)
    lifecycle_module._set_client_for_testing(client)
    return client


def _handler(
    *,
    primary_output: str = "primary answer",
    judge_output: str = '{"score": 0.9, "reasoning": "good"}',
    seen_variables: list[dict[str, Any]] | None = None,
    order: list[str] | None = None,
    judge_error: Exception | None = None,
    judge_delay_s: float = 0.0,
    stream_chunks: list[str] | None = None,
) -> ProviderHandler:
    """One handler serving both the primary call and any judge call.

    A judge call is recognised by the `message_history` variable the SDK injects.
    """

    async def fn(
        cfg: Any,
        user_input: Any,
        tool_handlers: Any,
        variables: Any,
        history: Any = None,
    ) -> dict[str, Any]:
        is_judge = bool(variables and "message_history" in variables)
        if not is_judge:
            if order is not None:
                order.append("primary")
            return {
                "output": primary_output,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        if seen_variables is not None:
            seen_variables.append(dict(variables))
        if order is not None:
            order.append("judge")
        if judge_delay_s:
            await asyncio.sleep(judge_delay_s)
        if judge_error is not None:
            raise judge_error
        return {
            "output": judge_output,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    async def stream_fn(
        cfg: Any,
        user_input: Any,
        tool_handlers: Any,
        variables: Any,
        history: Any = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        chunks = stream_chunks or ["Hel", "lo"]
        for chunk in chunks:
            yield {"type": "chunk", "text": chunk}
        yield {
            "type": "done",
            "output": "".join(chunks),
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    return ProviderHandler(
        fn=fn,
        provides_for=("TestProvider", "messages"),
        stream_fn=stream_fn if stream_chunks is not None else None,
    )


def _always_sample() -> Any:
    return patch.object(random, "random", return_value=0.0)


def _evidence_json(message_history: str) -> Any:
    """Parse the JSON sitting between the two delimiter lines."""
    lines = message_history.split("\n")
    begin = lines.index(EVIDENCE_BEGIN)
    end = lines.index(EVIDENCE_END)
    return json.loads("\n".join(lines[begin + 1 : end]))


# ---------------------------------------------------------------------------
# Resolution of the callback
# ---------------------------------------------------------------------------


class TestJudgeContextResolution:
    async def test_resolved_once_after_primary_and_before_parsing(self) -> None:
        order: list[str] = []
        calls = [0]
        variations = {
            "flag": {
                **_main_variation([{"key": "judge-key", "samplingRate": 1.0}]),
                "outputFormat": {"type": "object"},
            },
            "judge-key": _judge_variation(),
        }
        _install(variations)

        def judge_context() -> Any:
            calls[0] += 1
            order.append("context")
            return {"tool": "ok"}

        handler = _handler(primary_output='{"answer": 1}', order=order)

        import launchdarkly_ai_server.client as client_module

        real_parse = client_module._resolve_output_format_response

        def recording_parse(raw: Any, output_format: Any) -> Any:
            order.append("parse")
            return real_parse(raw, output_format)

        try:
            with (
                _always_sample(),
                patch.object(
                    client_module,
                    "_resolve_output_format_response",
                    recording_parse,
                ),
            ):
                result = await config(
                    key="flag", handler=handler, judge_context=judge_context
                ).invoke("q", CONTEXT)
        finally:
            lifecycle_module._reset_for_testing()

        assert calls[0] == 1
        assert order == ["primary", "context", "parse", "judge"]
        # Parsing still happened: outputFormat turned the raw JSON into a dict.
        assert result.response == {"answer": 1}

    async def test_resolved_even_when_no_judge_is_sampled(self) -> None:
        calls = [0]
        _install(
            {
                "flag": _main_variation([{"key": "judge-key", "samplingRate": 0.0}]),
                "judge-key": _judge_variation(),
            }
        )

        def judge_context() -> Any:
            calls[0] += 1
            return {"tool": "ok"}

        try:
            result = await config(
                key="flag", handler=_handler(), judge_context=judge_context
            ).invoke("q", CONTEXT)
        finally:
            lifecycle_module._reset_for_testing()

        assert calls[0] == 1
        assert result.judge_context == {"tool": "ok"}
        assert result.judge_results is None
        assert result.judge_diagnostics is None

    async def test_async_callback_is_awaited_and_value_unchanged(self) -> None:
        payload: JsonValue = {
            "steps": [{"name": "search", "status": "not_found"}],
            "count": 2,
        }
        _install({"flag": _main_variation()})

        async def judge_context() -> Any:
            await asyncio.sleep(0)
            return payload

        try:
            result = await config(
                key="flag", handler=_handler(), judge_context=judge_context
            ).invoke("q", CONTEXT)
        finally:
            lifecycle_module._reset_for_testing()

        assert result.judge_context == payload
        assert result.judge_context is payload


# ---------------------------------------------------------------------------
# Injection into message_history
# ---------------------------------------------------------------------------


class TestEvidenceBlock:
    async def test_message_history_block_round_trips_to_judge_context(self) -> None:
        payload: JsonValue = {"tool_calls": [{"name": "lookup", "result": "not_found"}]}
        seen: list[dict[str, Any]] = []
        _install(
            {
                "flag": _main_variation([{"key": "judge-key", "samplingRate": 1.0}]),
                "judge-key": _judge_variation(),
            }
        )
        try:
            with _always_sample():
                result = await config(
                    key="flag",
                    handler=_handler(seen_variables=seen),
                    judge_context=lambda: payload,
                ).invoke("q", CONTEXT)
        finally:
            lifecycle_module._reset_for_testing()

        assert len(seen) == 1
        history = seen[0]["message_history"]
        assert _evidence_json(history) == result.judge_context
        # Placement: after the user input and the response, before the instructions.
        assert history.index("q") < history.index(EVIDENCE_BEGIN)
        assert history.index("primary answer") < history.index(EVIDENCE_BEGIN)
        assert history.index(EVIDENCE_END) < history.index(
            "Your response MUST be in valid JSON format"
        )

    async def test_no_context_leaves_message_history_byte_identical(self) -> None:
        seen: list[dict[str, Any]] = []
        _install(
            {
                "flag": _main_variation([{"key": "judge-key", "samplingRate": 1.0}]),
                "judge-key": _judge_variation(),
            }
        )
        try:
            with _always_sample():
                await config(key="flag", handler=_handler(seen_variables=seen)).invoke(
                    "q", CONTEXT
                )
        finally:
            lifecycle_module._reset_for_testing()

        from launchdarkly_ai_server.judges import _FORMATTING_INSTRUCTIONS

        assert seen[0]["message_history"] == "\n\n".join(
            ["q", "primary answer", _FORMATTING_INSTRUCTIONS]
        )

    async def test_context_never_reaches_the_primary_model(self) -> None:
        primary_variables: list[dict[str, Any]] = []
        _install({"flag": _main_variation()})

        async def fn(
            cfg: Any,
            user_input: Any,
            tool_handlers: Any,
            variables: Any,
            history: Any = None,
        ) -> dict[str, Any]:
            primary_variables.append(dict(variables or {}))
            return {"output": "ok", "usage": {"input_tokens": 1, "output_tokens": 1}}

        handler = ProviderHandler(fn=fn, provides_for=("TestProvider", "messages"))
        try:
            await config(
                key="flag", handler=handler, judge_context=lambda: {"secret": 1}
            ).invoke("q", CONTEXT)
        finally:
            lifecycle_module._reset_for_testing()

        assert all(
            "judge_context" not in variables and "message_history" not in variables
            for variables in primary_variables
        )


# ---------------------------------------------------------------------------
# Context diagnostics
# ---------------------------------------------------------------------------


def _cycle() -> Any:
    value: dict[str, Any] = {}
    value["self"] = value
    return value


class TestContextDiagnostics:
    @pytest.mark.parametrize(
        ("callback", "code"),
        [
            (
                lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                "context_callback_failed",
            ),
            (lambda: object(), "context_invalid_json"),
            (lambda: {"set": {1, 2}}, "context_invalid_json"),
            (_cycle, "context_invalid_json"),
            (lambda: {"blob": "x" * (64 * 1024 + 1)}, "context_too_large"),
        ],
    )
    async def test_bad_context_skips_every_judge_but_keeps_primary(
        self, callback: Callable[[], Any], code: str
    ) -> None:
        seen: list[dict[str, Any]] = []
        _install(
            {
                "flag": _main_variation([{"key": "judge-key", "samplingRate": 1.0}]),
                "judge-key": _judge_variation(),
            }
        )
        try:
            with _always_sample():
                result = await config(
                    key="flag",
                    handler=_handler(seen_variables=seen),
                    judge_context=callback,
                ).invoke("q", CONTEXT)
        finally:
            lifecycle_module._reset_for_testing()

        assert result.response == "primary answer"
        assert result.judge_context is None
        assert result.judge_results is None
        assert seen == []
        assert result.judge_diagnostics is not None
        assert len(result.judge_diagnostics) == 1
        diagnostic = result.judge_diagnostics[0]
        assert (diagnostic.status, diagnostic.stage, diagnostic.code) == (
            "skipped",
            "context",
            code,
        )

    async def test_context_at_the_size_limit_is_accepted_unchanged(self) -> None:
        # 64 KiB exactly, encoded.
        filler = "x" * (64 * 1024 - len(json.dumps({"blob": ""})))
        payload: JsonValue = {"blob": filler}
        assert len(json.dumps(payload).encode("utf-8")) == 64 * 1024
        _install({"flag": _main_variation()})
        try:
            result = await config(
                key="flag", handler=_handler(), judge_context=lambda: payload
            ).invoke("q", CONTEXT)
        finally:
            lifecycle_module._reset_for_testing()

        assert result.judge_context == payload
        assert result.judge_diagnostics is None


# ---------------------------------------------------------------------------
# Per-judge isolation
# ---------------------------------------------------------------------------


class TestJudgeIsolation:
    async def test_duplicate_judge_key_runs_once_and_reports(self) -> None:
        order: list[str] = []
        _install(
            {
                "flag": _main_variation(
                    [
                        {"key": "judge-key", "samplingRate": 1.0},
                        {"key": "judge-key", "samplingRate": 1.0},
                    ]
                ),
                "judge-key": _judge_variation(),
            }
        )
        try:
            with _always_sample():
                result = await config(key="flag", handler=_handler(order=order)).invoke(
                    "q", CONTEXT
                )
        finally:
            lifecycle_module._reset_for_testing()

        assert order.count("judge") == 1
        assert result.judge_results is not None
        assert set(result.judge_results) == {"judge-key"}
        assert result.judge_diagnostics is not None
        diagnostic = result.judge_diagnostics[0]
        assert (diagnostic.judge_key, diagnostic.status, diagnostic.code) == (
            "judge-key",
            "skipped",
            "judge_duplicate_key",
        )

    async def test_config_lookup_failure_is_isolated(self) -> None:
        _install(
            {
                "flag": _main_variation(
                    [
                        {"key": "bad-judge", "samplingRate": 1.0},
                        {"key": "good-judge", "samplingRate": 1.0},
                    ]
                ),
                "bad-judge": RuntimeError("flag exploded"),
                "good-judge": _judge_variation(),
            }
        )
        try:
            with _always_sample():
                result = await config(key="flag", handler=_handler()).invoke(
                    "q", CONTEXT
                )
        finally:
            lifecycle_module._reset_for_testing()

        assert result.response == "primary answer"
        assert result.judge_results is not None
        assert set(result.judge_results) == {"good-judge"}
        assert result.judge_diagnostics is not None
        diagnostic = result.judge_diagnostics[0]
        assert (diagnostic.judge_key, diagnostic.stage, diagnostic.code) == (
            "bad-judge",
            "config",
            "judge_config_failed",
        )
        assert "flag exploded" not in str(diagnostic)

    async def test_provider_failure_is_isolated(self) -> None:
        _install(
            {
                "flag": _main_variation([{"key": "judge-key", "samplingRate": 1.0}]),
                "judge-key": _judge_variation(),
            }
        )
        try:
            with _always_sample():
                result = await config(
                    key="flag",
                    handler=_handler(judge_error=RuntimeError("provider down")),
                ).invoke("q", CONTEXT)
        finally:
            lifecycle_module._reset_for_testing()

        assert result.response == "primary answer"
        assert result.judge_results is None
        assert result.judge_diagnostics is not None
        diagnostic = result.judge_diagnostics[0]
        assert (diagnostic.status, diagnostic.stage, diagnostic.code) == (
            "failed",
            "provider",
            "judge_provider_failed",
        )
        assert "provider down" not in str(diagnostic)

    async def test_invalid_verdict_is_isolated(self) -> None:
        _install(
            {
                "flag": _main_variation([{"key": "judge-key", "samplingRate": 1.0}]),
                "judge-key": _judge_variation(),
            }
        )
        try:
            with _always_sample():
                result = await config(
                    key="flag", handler=_handler(judge_output="not json at all")
                ).invoke("q", CONTEXT)
        finally:
            lifecycle_module._reset_for_testing()

        assert result.response == "primary answer"
        assert result.judge_results is None
        assert result.judge_diagnostics is not None
        diagnostic = result.judge_diagnostics[0]
        assert (diagnostic.status, diagnostic.stage, diagnostic.code) == (
            "failed",
            "parse",
            "judge_response_invalid",
        )

    async def test_track_failure_keeps_the_judge_result(self) -> None:
        client = _install(
            {
                "flag": _main_variation([{"key": "judge-key", "samplingRate": 1.0}]),
                "judge-key": _judge_variation(evaluation_metric_key="judge-metric"),
            }
        )

        def track(metric: str, *_args: object, **_kwargs: object) -> None:
            if metric == "judge-metric":
                raise RuntimeError("track exploded")

        client.track = MagicMock(side_effect=track)
        try:
            with _always_sample():
                result = await config(key="flag", handler=_handler()).invoke(
                    "q", CONTEXT
                )
        finally:
            lifecycle_module._reset_for_testing()

        assert result.judge_results is not None
        assert result.judge_results["judge-key"].score == 0.9
        assert result.judge_diagnostics is not None
        diagnostic = result.judge_diagnostics[0]
        assert (diagnostic.status, diagnostic.stage, diagnostic.code) == (
            "failed",
            "track",
            "judge_tracking_failed",
        )

    async def test_reasoning_is_capped_at_4_kib(self) -> None:
        long_reasoning = "é" * 5000
        _install(
            {
                "flag": _main_variation([{"key": "judge-key", "samplingRate": 1.0}]),
                "judge-key": _judge_variation(),
            }
        )
        try:
            with _always_sample():
                result = await config(
                    key="flag",
                    handler=_handler(
                        judge_output=json.dumps(
                            {"score": 0.5, "reasoning": long_reasoning}
                        )
                    ),
                ).invoke("q", CONTEXT)
        finally:
            lifecycle_module._reset_for_testing()

        assert result.judge_results is not None
        capped = result.judge_results["judge-key"].response
        assert len(capped.encode("utf-8")) <= 4 * 1024
        assert long_reasoning.startswith(capped)


class TestJudgeTimeout:
    async def test_slow_judge_times_out_without_touching_results(self) -> None:
        client = _install(
            {
                "flag": _main_variation([{"key": "slow-judge", "samplingRate": 1.0}]),
                "slow-judge": _judge_variation(evaluation_metric_key="judge-metric"),
            }
        )
        try:
            with _always_sample():
                result = await config(
                    key="flag",
                    handler=_handler(judge_delay_s=0.2),
                    judge_timeout_ms=10,
                ).invoke("q", CONTEXT)
            # Let the straggler finish: it must change nothing.
            await asyncio.sleep(0.3)
        finally:
            lifecycle_module._reset_for_testing()

        assert result.response == "primary answer"
        assert result.judge_results is None
        assert result.judge_diagnostics is not None
        diagnostic = result.judge_diagnostics[0]
        assert (
            diagnostic.judge_key,
            diagnostic.status,
            diagnostic.stage,
            diagnostic.code,
        ) == (
            "slow-judge",
            "failed",
            "timeout",
            "judge_timed_out",
        )
        assert all(
            call.args[0] != "judge-metric" for call in client.track.call_args_list
        )

    async def test_a_second_judge_still_runs_after_a_timeout(self) -> None:
        _install(
            {
                "flag": _main_variation(
                    [
                        {"key": "slow-judge", "samplingRate": 1.0},
                        {"key": "fast-judge", "samplingRate": 1.0},
                    ]
                ),
                "slow-judge": _judge_variation(),
                "fast-judge": _judge_variation(),
            }
        )

        async def fn(
            cfg: Any,
            user_input: Any,
            tool_handlers: Any,
            variables: Any,
            history: Any = None,
        ) -> dict[str, Any]:
            if not (variables and "message_history" in variables):
                return {
                    "output": "primary answer",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            if cfg.get("_slow"):
                await asyncio.sleep(0.2)
            return {
                "output": '{"score": 0.4, "reasoning": "ok"}',
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        slow = _judge_variation()
        slow["_slow"] = True
        client = lifecycle_module.get_client()

        async def variation(key: str, *_a: object, **_k: object) -> Any:
            if key == "flag":
                return _main_variation(
                    [
                        {"key": "slow-judge", "samplingRate": 1.0},
                        {"key": "fast-judge", "samplingRate": 1.0},
                    ]
                )
            return slow if key == "slow-judge" else _judge_variation()

        client.variation = AsyncMock(side_effect=variation)
        handler = ProviderHandler(fn=fn, provides_for=("TestProvider", "messages"))
        try:
            with _always_sample():
                result = await config(
                    key="flag", handler=handler, judge_timeout_ms=20
                ).invoke("q", CONTEXT)
            await asyncio.sleep(0.3)
        finally:
            lifecycle_module._reset_for_testing()

        assert result.judge_results is not None
        assert set(result.judge_results) == {"fast-judge"}
        assert result.judge_diagnostics is not None
        assert [d.judge_key for d in result.judge_diagnostics] == ["slow-judge"]


# ---------------------------------------------------------------------------
# skip_judges path
# ---------------------------------------------------------------------------


class TestSkipJudgesPath:
    async def test_tasks_carry_context_and_run_judge_injects_the_same_block(
        self,
    ) -> None:
        payload: JsonValue = {"tool_calls": [{"name": "lookup", "result": "not_found"}]}
        seen: list[dict[str, Any]] = []
        _install(
            {
                "flag": _main_variation([{"key": "judge-key", "samplingRate": 1.0}]),
                "judge-key": _judge_variation(),
            }
        )
        handler = _handler(seen_variables=seen)
        try:
            with _always_sample():
                result = await config(
                    key="flag",
                    handler=handler,
                    skip_judges=True,
                    judge_context=lambda: payload,
                ).invoke("q", CONTEXT)

            assert result.judge_context == payload
            assert result.judge_tasks is not None
            task = result.judge_tasks[0]
            assert task.judge_context == payload
            assert seen == []

            run_result = await run_judge(task, [handler])
        finally:
            lifecycle_module._reset_for_testing()

        assert run_result is not None
        assert len(seen) == 1
        assert _evidence_json(seen[0]["message_history"]) == payload

    async def test_build_step_diagnostics_come_back_with_the_tasks(self) -> None:
        _install(
            {
                "flag": _main_variation([{"key": "judge-key", "samplingRate": 1.0}]),
                "judge-key": _judge_variation(),
            }
        )

        def bad_context() -> Any:
            raise RuntimeError("boom")

        try:
            with _always_sample():
                result = await config(
                    key="flag",
                    handler=_handler(),
                    skip_judges=True,
                    judge_context=bad_context,
                ).invoke("q", CONTEXT)
        finally:
            lifecycle_module._reset_for_testing()

        assert result.response == "primary answer"
        assert result.judge_tasks == []
        assert result.judge_context is None
        assert result.judge_diagnostics is not None
        assert result.judge_diagnostics[0].code == "context_callback_failed"

    async def test_build_judge_tasks_resolves_the_callback_itself(self) -> None:
        calls = [0]
        _install(
            {
                "flag": _main_variation([{"key": "judge-key", "samplingRate": 1.0}]),
                "judge-key": _judge_variation(),
            }
        )

        def judge_context() -> Any:
            calls[0] += 1
            return [1, 2, 3]

        handler = _handler()
        try:
            with _always_sample():
                build = await build_judge_tasks(
                    config=_main_variation([{"key": "judge-key", "samplingRate": 1.0}]),
                    user_context=CONTEXT,
                    handler=handler,
                    handlers=[handler],
                    llm_response="primary answer",
                    base_track_data={"runId": "r1"},
                    judge_context=judge_context,
                )
        finally:
            lifecycle_module._reset_for_testing()

        assert calls[0] == 1
        assert build.judge_diagnostics == []
        assert [task.judge_context for task in build.judge_tasks] == [[1, 2, 3]]

    def test_judge_task_stays_json_serialisable(self) -> None:
        task = JudgeTask(
            config_key="judge-key",
            judge_config={"provider": {"name": "TestProvider"}},
            judge_meta={"mode": "judge"},
            actual_output="answer",
            user_context=CONTEXT,
            judge_provider="TestProvider",
            judge_mode="messages",
            collapse_messages=False,
            parent_track_data={"runId": "r1"},
            judge_context={"tool": "ok"},
        )
        from dataclasses import asdict

        assert json.loads(json.dumps(asdict(task)))["judge_context"] == {"tool": "ok"}


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    async def test_stream_yields_chunks_then_one_done_event(self) -> None:
        payload: JsonValue = {"tool_calls": [{"name": "lookup", "result": "ok"}]}
        seen: list[dict[str, Any]] = []
        _install(
            {
                "flag": _main_variation([{"key": "judge-key", "samplingRate": 1.0}]),
                "judge-key": _judge_variation(),
            }
        )
        handler = _handler(seen_variables=seen, stream_chunks=["Hel", "lo"])
        try:
            with _always_sample():
                events = [
                    event
                    async for event in config(
                        key="flag", handler=handler, judge_context=lambda: payload
                    ).stream("q", CONTEXT)
                ]
        finally:
            lifecycle_module._reset_for_testing()

        assert [e["text"] for e in events if e["type"] == "chunk"] == ["Hel", "lo"]
        done = [e for e in events if e["type"] == "done"]
        assert len(done) == 1
        assert done[0]["response"] == "Hello"
        assert done[0]["judge_context"] == payload
        assert set(done[0]["judge_results"]) == {"judge-key"}
        assert done[0]["judge_diagnostics"] is None
        assert _evidence_json(seen[0]["message_history"]) == payload

    async def test_stream_done_carries_context_diagnostics(self) -> None:
        _install(
            {
                "flag": _main_variation([{"key": "judge-key", "samplingRate": 1.0}]),
                "judge-key": _judge_variation(),
            }
        )
        handler = _handler(stream_chunks=["a"])

        def bad_context() -> Any:
            return {1, 2}

        try:
            with _always_sample():
                events = [
                    event
                    async for event in config(
                        key="flag", handler=handler, judge_context=bad_context
                    ).stream("q", CONTEXT)
                ]
        finally:
            lifecycle_module._reset_for_testing()

        done = events[-1]
        assert done["type"] == "done"
        assert done["judge_context"] is None
        assert done["judge_results"] is None
        assert done["judge_diagnostics"][0].code == "context_invalid_json"
