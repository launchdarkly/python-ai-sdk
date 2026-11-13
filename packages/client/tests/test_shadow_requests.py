"""Tests for experimental shadow requests (``shadow_config``)."""

import asyncio
import dataclasses
import logging
import pickle
import threading
import time
from collections.abc import Iterator
from contextlib import aclosing
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from opentelemetry import context as otel_context

import launchdarkly_ai_server
import launchdarkly_ai_server.client as client_module
import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import (
    ProviderHandler,
    agui,
    config,
    conversation_id,
    shutdown,
)
from launchdarkly_ai_server.conversation import _conversation_id_from
from launchdarkly_ai_server.experimental import shadow_requests as sr
from launchdarkly_ai_server.experimental.shadow_requests import (
    ShadowResult,
    ShadowTask,
    run_shadow,
    shadow_config,
)
from launchdarkly_ai_server.sdk_usage import reset_sdk_usage
from launchdarkly_ai_server.shutdown_hooks import _hooks
from launchdarkly_ai_server.utils import set_ld_span_attributes

CONTEXT = {"kind": "user", "key": "u1"}
FUTURE = int(time.time() * 1000) + 3_600_000
PAST = int(time.time() * 1000) - 1
FALLTHROUGH = {"kind": "FALLTHROUGH"}
SHADOW_FIELDS = {
    "isShadow",
    "primaryVariationKey",
    "shadowRequestId",
    "ownerKind",
    "ownerId",
}
BOOKKEEPING = {"$ld:ai:sdk:usage", "$ld:ai:sdk:info"}


def _entry(**overrides: Any) -> dict[str, Any]:
    entry = {
        "id": "shadow-1",
        "variationKey": "b",
        "samplingRate": 1.0,
        "endsAt": FUTURE,
        "ownerKind": "member",
        "ownerId": "member-1",
    }
    entry.update(overrides)
    return entry


def _variation_b(
    enabled: bool = True, judges: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    variation: dict[str, Any] = {
        "_ldMeta": {
            "enabled": enabled,
            "variationKey": "b",
            "version": 7,
            "mode": "messages",
            "modelKey": "claude",
            "modelVersion": 2,
        },
        "model": {"name": "claude-x"},
        "provider": {"name": "ShadowProvider"},
        "instructions": "Shadow.",
        "tools": {"search": {"name": "search", "type": "function", "parameters": {}}},
    }
    if judges is not None:
        variation["judgeConfiguration"] = {"judges": judges}
    return variation


def _variation_a(
    shadows: dict[str, Any] | None = None, variations: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "_ldMeta": {
            "enabled": True,
            "variationKey": "a",
            "version": 1,
            "mode": "messages",
        },
        "model": {"name": "gpt-4"},
        "provider": {"name": "TestProvider"},
        "instructions": "Be helpful.",
        "shadowConfiguration": {
            "shadows": shadows if shadows is not None else {"fallthrough": [_entry()]},
            "variations": variations
            if variations is not None
            else {"b": _variation_b()},
        },
    }


def _make_client(
    value: dict[str, Any], reason: dict[str, Any] | None = FALLTHROUGH
) -> MagicMock:
    c = MagicMock()
    c.track = MagicMock()
    c.flush = AsyncMock()
    c.close = AsyncMock()
    c.variation = AsyncMock(return_value=value)
    c.variation_detail = AsyncMock(
        return_value=SimpleNamespace(value=value, reason=reason)
    )
    return c


class _Recorder:
    def __init__(
        self,
        provider: str,
        output: str = "ok",
        *,
        sleep_s: float = 0,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.streamed = 0
        self._output = output
        self._sleep_s = sleep_s
        self._error = error
        self.handler = ProviderHandler(
            fn=self._fn,  # type: ignore[arg-type]
            provides_for=(provider, "messages"),
            stream_fn=self._stream,  # type: ignore[arg-type]
        )

    async def _fn(
        self,
        cfg: Any,
        user_input: Any,
        tool_handlers: Any,
        variables: Any,
        history: Any = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "config": cfg,
                "input": user_input,
                "tools": tool_handlers,
                "variables": variables,
                "history": history,
                "conversation_id": _conversation_id_from(otel_context.get_current()),
            }
        )
        if self._sleep_s:
            await asyncio.sleep(self._sleep_s)
        if self._error is not None:
            raise self._error
        return {
            "output": self._output,
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }

    async def _stream(self, cfg: Any, *args: Any) -> Any:
        self.streamed += 1
        self.calls.append(
            {
                "config": cfg,
                "streamed": True,
                "conversation_id": _conversation_id_from(otel_context.get_current()),
            }
        )
        yield {"type": "chunk", "text": self._output}
        yield {"type": "done", "usage": {"input_tokens": 3, "output_tokens": 2}}


def _events(client: MagicMock) -> list[tuple[str, dict[str, Any]]]:
    return [
        (c[0][0], c[0][2])
        for c in client.track.call_args_list
        if c[0][0] not in BOOKKEEPING
    ]


def _shadow_events(client: MagicMock) -> list[tuple[str, dict[str, Any]]]:
    return [e for e in _events(client) if e[1].get("isShadow")]


def _real_events(client: MagicMock) -> list[tuple[str, dict[str, Any]]]:
    return [e for e in _events(client) if not e[1].get("isShadow")]


def _without_run_id(
    events: list[tuple[str, dict[str, Any]]],
) -> list[tuple[str, dict[str, Any]]]:
    return [(n, {k: v for k, v in d.items() if k != "runId"}) for n, d in events]


def _usage_helpers(client: MagicMock) -> list[str]:
    return [
        c[0][2].get("helper")
        for c in client.track.call_args_list
        if c[0][0] == "$ld:ai:sdk:usage"
    ]


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    lifecycle_module._reset_for_testing()
    reset_sdk_usage()
    yield
    lifecycle_module._reset_for_testing()


def _use(client: MagicMock) -> MagicMock:
    lifecycle_module._set_client_for_testing(client)
    return client


async def _invoke(
    client: MagicMock,
    *,
    user_input: str = "hi",
    history: list[dict[str, Any]] | None = None,
    b: _Recorder | None = None,
    **shadow_args: Any,
) -> tuple[Any, _Recorder, _Recorder]:
    _use(client)
    a = _Recorder("TestProvider", "real answer")
    b = b or _Recorder("ShadowProvider", "shadow answer")
    response = await shadow_config(
        key="flag", handler=[a.handler, b.handler], **shadow_args
    ).invoke(user_input, CONTEXT, {"topic": "x"}, history)
    await sr._wait_for_shadows_for_testing()
    return response, a, b


async def _deferred_task(client: MagicMock) -> ShadowTask:
    tasks: list[ShadowTask] = []
    await _invoke(client, on_task=tasks.append)
    assert len(tasks) == 1
    return tasks[0]


class TestPlainConfig:
    async def test_config_never_calls_variation_detail_or_runs_shadows(self) -> None:
        client = _use(_make_client(_variation_a()))
        a = _Recorder("TestProvider", "real answer")
        b = _Recorder("ShadowProvider")
        response = await config(key="flag", handler=[a.handler, b.handler]).invoke(
            "hi", CONTEXT
        )
        await sr._wait_for_shadows_for_testing()
        assert response.response == "real answer"
        client.variation.assert_called_once()
        client.variation_detail.assert_not_called()
        assert b.calls == []
        assert _shadow_events(client) == []

    async def test_config_stream_never_runs_shadows(self) -> None:
        client = _use(_make_client(_variation_a()))
        a = _Recorder("TestProvider", "real answer")
        b = _Recorder("ShadowProvider")
        async for _ in config(key="flag", handler=[a.handler, b.handler]).stream(
            "hi", CONTEXT
        ):
            pass
        await sr._wait_for_shadows_for_testing()
        client.variation_detail.assert_not_called()
        assert b.calls == []

    def test_root_exports_no_shadow_names(self) -> None:
        for name in ("shadow_config", "run_shadow", "ShadowTask", "ShadowResult"):
            assert name not in launchdarkly_ai_server.__all__
            assert not hasattr(launchdarkly_ai_server, name)


class TestCoreSeams:
    def test_forwards_every_config_instance_option(self) -> None:
        import inspect

        from launchdarkly_ai_server.client import ConfigInstance

        options = set(inspect.signature(ConfigInstance.__init__).parameters) - {"self"}
        assert set(sr._Evaluated._FORWARDED) == options

    async def test_fetch_variation_reads_the_reason_only_when_asked(self) -> None:
        detail = SimpleNamespace(value={"v": 1}, reason={"kind": "FALLTHROUGH"})
        client = MagicMock()
        client.variation.return_value = {"v": 0}
        client.variation_detail.return_value = detail
        _use(client)
        assert await lifecycle_module._fetch_variation("flag", CONTEXT) == (
            {"v": 0},
            None,
        )
        assert await lifecycle_module._fetch_variation(
            "flag", CONTEXT, with_reason=True
        ) == ({"v": 1}, {"kind": "FALLTHROUGH"})
        client.variation_detail.assert_called_once()


class TestPrimary:
    async def test_primary_matches_plain_config(self) -> None:
        value = _variation_a(shadows={})
        baseline = _use(_make_client(value))
        a = _Recorder("TestProvider", "real answer")
        expected = await config(key="flag", handler=a.handler).invoke(
            "hi", CONTEXT, {"topic": "x"}
        )
        lifecycle_module._reset_for_testing()
        client = _make_client(value)
        response, a2, _ = await _invoke(client)
        assert response.response == expected.response
        assert response.usage == expected.usage
        assert _without_run_id(_events(client)) == _without_run_id(_events(baseline))
        served = {
            k: v for k, v in a.calls[0]["config"].items() if k != "shadowConfiguration"
        }
        assert a2.calls[0]["config"] == served

    async def test_shadow_configuration_never_reaches_a_handler(self) -> None:
        _, a, b = await _invoke(_make_client(_variation_a()))
        assert "shadowConfiguration" not in a.calls[0]["config"]
        assert "shadowConfiguration" not in b.calls[0]["config"]

    async def test_captured_variation_value_has_no_shadow_configuration(self) -> None:
        _use(_make_client(_variation_a()))
        a = _Recorder("TestProvider", "real answer")
        capturing = ProviderHandler(
            fn=a._fn,  # type: ignore[arg-type]
            provides_for=("TestProvider", "messages"),
            capture_content=True,
        )
        await shadow_config(
            key="flag", handler=[capturing, _Recorder("ShadowProvider").handler]
        ).invoke("hi", CONTEXT)
        await sr._wait_for_shadows_for_testing()
        captured = a.calls[0]["variables"]["__ldValue"]
        assert captured["_ldMeta"]["variationKey"] == "a"
        assert "shadowConfiguration" not in captured

    async def test_primary_failure_starts_no_shadows(self) -> None:
        client = _use(_make_client(_variation_a()))
        a = _Recorder("TestProvider", error=RuntimeError("primary failed"))
        b = _Recorder("ShadowProvider")
        with pytest.raises(RuntimeError, match="primary failed"):
            await shadow_config(key="flag", handler=[a.handler, b.handler]).invoke(
                "hi", CONTEXT
            )
        await sr._wait_for_shadows_for_testing()
        assert b.calls == []
        assert _shadow_events(client) == []

    async def test_disabled_primary_raises_like_config(self) -> None:
        value = _variation_a()
        value["_ldMeta"]["enabled"] = False
        _use(_make_client(value))
        a = _Recorder("TestProvider")
        with pytest.raises(RuntimeError, match="disabled"):
            await shadow_config(key="flag", handler=a.handler).invoke("hi", CONTEXT)

    @pytest.mark.parametrize(
        "shadow_configuration",
        [
            "nope",
            {"shadows": "nope", "variations": {"b": _variation_b()}},
            {"shadows": {"fallthrough": "nope"}, "variations": {"b": _variation_b()}},
            {"shadows": {"fallthrough": ["nope"]}, "variations": {"b": _variation_b()}},
            {"shadows": {"fallthrough": [{"id": "x"}]}, "variations": {"b": {}}},
            {
                "shadows": {"fallthrough": [_entry(id=None)]},
                "variations": {"b": _variation_b()},
            },
            {
                "shadows": {"fallthrough": [_entry(variationKey=None)]},
                "variations": {"b": _variation_b()},
            },
            {
                "shadows": {"fallthrough": [_entry(samplingRate="high")]},
                "variations": {"b": _variation_b()},
            },
            {
                "shadows": {"fallthrough": [_entry(endsAt=None)]},
                "variations": {"b": _variation_b()},
            },
        ],
    )
    async def test_malformed_configuration_is_ignored(
        self, shadow_configuration: Any
    ) -> None:
        value = _variation_a()
        value["shadowConfiguration"] = shadow_configuration
        response, a, b = await _invoke(_make_client(value))
        assert response.response == "real answer"
        assert len(a.calls) == 1
        assert b.calls == []


class TestSelection:
    async def test_evaluates_once_with_variation_detail(self) -> None:
        client = _make_client(_variation_a())
        await _invoke(client)
        client.variation_detail.assert_called_once()
        client.variation.assert_not_called()

    async def test_client_without_variation_detail_runs_no_shadows(self) -> None:
        client = _make_client(_variation_a())
        del client.variation_detail
        response, a, b = await _invoke(client)
        assert response.response == "real answer"
        assert len(a.calls) == 1
        assert b.calls == []

    async def test_rule_match_uses_only_that_rule(self) -> None:
        shadows = {
            "r1": [_entry(id="rule-shadow")],
            "fallthrough": [_entry(id="ft-shadow")],
        }
        client = _make_client(
            _variation_a(shadows), {"kind": "RULE_MATCH", "ruleId": "r1"}
        )
        await _invoke(client)
        assert {e[1]["shadowRequestId"] for e in _shadow_events(client)} == {
            "rule-shadow"
        }

    async def test_fallthrough_uses_the_fallthrough_slot(self) -> None:
        shadows = {"r1": [_entry(id="rule-shadow")], "fallthrough": [_entry(id="ft")]}
        client = _make_client(_variation_a(shadows))
        await _invoke(client)
        assert {e[1]["shadowRequestId"] for e in _shadow_events(client)} == {"ft"}

    @pytest.mark.parametrize(
        "reason",
        [
            {"kind": "TARGET_MATCH"},
            {"kind": "OFF"},
            {"kind": "PREREQUISITE_FAILED"},
            {"kind": "ERROR"},
            {"kind": "RULE_MATCH"},
            None,
        ],
    )
    async def test_other_reasons_run_nothing(
        self, reason: dict[str, Any] | None
    ) -> None:
        _, _, b = await _invoke(_make_client(_variation_a(), reason))
        assert b.calls == []

    async def test_entry_past_its_end_is_skipped(self) -> None:
        shadows = {"fallthrough": [_entry(id="ended", endsAt=PAST), _entry(id="live")]}
        client = _make_client(_variation_a(shadows))
        await _invoke(client)
        assert {e[1]["shadowRequestId"] for e in _shadow_events(client)} == {"live"}

    async def test_missing_variation_is_skipped(self) -> None:
        shadows = {"fallthrough": [_entry(variationKey="gone")]}
        _, _, b = await _invoke(_make_client(_variation_a(shadows)))
        assert b.calls == []

    async def test_disabled_variation_is_skipped(self) -> None:
        value = _variation_a(variations={"b": _variation_b(enabled=False)})
        _, _, b = await _invoke(_make_client(value))
        assert b.calls == []

    async def test_no_matching_handler_is_skipped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _use(_make_client(_variation_a()))
        a = _Recorder("TestProvider", "real answer")
        tasks: list[ShadowTask] = []
        with caplog.at_level(logging.DEBUG, logger=sr.logger.name):
            response = await shadow_config(
                key="flag", handler=a.handler, on_task=tasks.append
            ).invoke("hi", CONTEXT)
        await sr._wait_for_shadows_for_testing()
        assert response.response == "real answer"
        assert tasks == []
        assert _shadow_events(client) == []
        assert any("shadow-1" in m for m in caplog.messages)

    def test_sampling_is_sticky(self) -> None:
        answers = {sr._in_sample("s1", CONTEXT, 0.5) for _ in range(20)}
        assert len(answers) == 1

    def test_sampling_honors_the_rate(self) -> None:
        hits = sum(
            sr._in_sample("s1", {"kind": "user", "key": f"k{i}"}, 0.2)
            for i in range(10_000)
        )
        assert 1800 <= hits <= 2200

    def test_sampling_edges(self) -> None:
        contexts = [{"kind": "user", "key": f"k{i}"} for i in range(200)]
        assert all(sr._in_sample("s1", c, 1.0) for c in contexts)
        assert not any(sr._in_sample("s1", c, 0.0) for c in contexts)

    def test_sampling_bucket_value(self) -> None:
        assert sr._in_sample("s1", CONTEXT, 0.732)
        assert not sr._in_sample("s1", CONTEXT, 0.731)

    def test_context_without_key_is_not_sampled(self) -> None:
        assert not sr._in_sample("s1", {"kind": "user"}, 1.0)


class TestStream:
    async def test_stream_runs_the_shadow_as_a_plain_invoke_after_done(self) -> None:
        client = _use(_make_client(_variation_a()))
        a = _Recorder("TestProvider", "real answer")
        b = _Recorder("ShadowProvider", "shadow answer")
        events = [
            e
            async for e in shadow_config(
                key="flag", handler=[a.handler, b.handler]
            ).stream("hi", CONTEXT)
        ]
        await sr._wait_for_shadows_for_testing()
        assert events[-1]["type"] == "done"
        assert events[-1]["response"] == "real answer"
        assert a.streamed == 1
        assert len(b.calls) == 1
        assert b.streamed == 0
        client.variation_detail.assert_called_once()
        client.variation.assert_not_called()

    async def test_stream_matches_plain_config_stream(self) -> None:
        value = _variation_a(shadows={})
        _use(_make_client(value))
        a = _Recorder("TestProvider", "real answer")
        expected = [
            e async for e in config(key="flag", handler=a.handler).stream("hi", CONTEXT)
        ]
        lifecycle_module._reset_for_testing()
        _use(_make_client(value))
        actual = [
            e
            async for e in shadow_config(key="flag", handler=a.handler).stream(
                "hi", CONTEXT
            )
        ]
        assert actual == expected

    async def test_stream_deferred_hands_over_the_task(self) -> None:
        client = _use(_make_client(_variation_a()))
        a = _Recorder("TestProvider", "real answer")
        b = _Recorder("ShadowProvider")
        tasks: list[ShadowTask] = []
        async for _ in shadow_config(
            key="flag", handler=[a.handler, b.handler], on_task=tasks.append
        ).stream("hi", CONTEXT):
            pass
        assert [t.shadow_request_id for t in tasks] == ["shadow-1"]
        assert b.calls == []
        primary_run_ids = {d["runId"] for _, d in _real_events(client)}
        assert tasks[0].track_data["runId"] in primary_run_ids

    async def test_stream_starts_shadows_after_judges(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use(_make_client(_variation_a()))
        tasks: list[ShadowTask] = []
        tasks_when_judged: list[int] = []

        async def judge(*args: Any) -> list[Any]:
            tasks_when_judged.append(len(tasks))
            return []

        monkeypatch.setattr(client_module, "judge_completed_stream", judge)
        async for _ in shadow_config(
            key="flag",
            handler=[
                _Recorder("TestProvider", "real answer").handler,
                _Recorder("ShadowProvider").handler,
            ],
            on_task=tasks.append,
        ).stream("hi", CONTEXT):
            pass
        assert tasks_when_judged == [0]
        assert len(tasks) == 1

    async def test_agui_stream_starts_shadows_after_judges(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use(_make_client(_variation_a()))
        tasks: list[ShadowTask] = []
        tasks_when_judged: list[int] = []

        async def judge(*args: Any) -> list[Any]:
            tasks_when_judged.append(len(tasks))
            return []

        monkeypatch.setattr(client_module, "judge_completed_stream", judge)
        instance = shadow_config(
            key="flag",
            handler=[
                _Recorder("TestProvider", "real answer").handler,
                _Recorder("ShadowProvider").handler,
            ],
            on_task=tasks.append,
        )
        async for _ in agui.stream(instance, "hi", CONTEXT):
            pass
        assert tasks_when_judged == [0]
        assert len(tasks) == 1

    async def test_stream_binds_the_conversation_id_at_call_time(self) -> None:
        _use(_make_client(_variation_a()))
        a = _Recorder("TestProvider", "real answer")
        b = _Recorder("ShadowProvider")
        with conversation_id("conv-1"):
            stream = shadow_config(key="flag", handler=[a.handler, b.handler]).stream(
                "hi", CONTEXT
            )
        async for _ in stream:
            pass
        await sr._wait_for_shadows_for_testing()
        assert a.calls[0]["conversation_id"] == "conv-1"
        assert b.calls[0]["conversation_id"] == "conv-1"

    async def test_agui_stream_runs_shadows(self) -> None:
        _use(_make_client(_variation_a()))
        a = _Recorder("TestProvider", "real answer")
        b = _Recorder("ShadowProvider")
        instance = shadow_config(key="flag", handler=[a.handler, b.handler])
        async for _ in agui.stream(instance, "hi", CONTEXT):
            pass
        await sr._wait_for_shadows_for_testing()
        assert len(b.calls) == 1


class TestRunning:
    async def test_same_input_on_the_stripped_config(self) -> None:
        history = [{"role": "user", "content": "earlier"}]
        _, _, b = await _invoke(
            _make_client(_variation_a()), user_input="question", history=history
        )
        call = b.calls[0]
        assert call["input"] == "question"
        assert call["history"] == history
        assert call["variables"]["topic"] == "x"
        assert call["config"]["instructions"] == "Shadow."
        assert "tools" not in call["config"]
        assert "_ldMeta" not in call["config"]
        assert not call["tools"]

    async def test_shadow_answer_never_joins_history(self) -> None:
        history = [{"role": "user", "content": "earlier"}]
        await _invoke(_make_client(_variation_a()), history=history)
        assert history == [{"role": "user", "content": "earlier"}]

    async def test_background_shadow_does_not_delay_the_caller(self) -> None:
        _use(_make_client(_variation_a()))
        a = _Recorder("TestProvider", "real answer")
        b = _Recorder("ShadowProvider", sleep_s=5)
        response = await asyncio.wait_for(
            shadow_config(key="flag", handler=[a.handler, b.handler]).invoke(
                "hi", CONTEXT
            ),
            timeout=1,
        )
        assert response.response == "real answer"
        assert sr._running
        await shutdown()

    async def test_failing_shadow_never_reaches_the_caller(self) -> None:
        client = _make_client(_variation_a())
        response, _, _ = await _invoke(
            client, b=_Recorder("ShadowProvider", error=RuntimeError("boom"))
        )
        assert response.response == "real answer"
        assert "$ld:ai:generation:success" in [e[0] for e in _real_events(client)]
        assert "$ld:ai:generation:error" in [e[0] for e in _shadow_events(client)]

    async def test_timed_out_shadow_never_reaches_the_caller(self) -> None:
        client = _make_client(_variation_a())
        response, _, _ = await _invoke(
            client, b=_Recorder("ShadowProvider", sleep_s=1), shadow_timeout_s=0.05
        )
        assert response.response == "real answer"
        assert "$ld:ai:generation:success" in [e[0] for e in _real_events(client)]

    @pytest.mark.parametrize("timeout_s", [0, -1, float("inf"), float("nan")])
    def test_timeout_must_be_finite_and_positive(self, timeout_s: float) -> None:
        with pytest.raises(ValueError):
            shadow_config(key="flag", shadow_timeout_s=timeout_s)

    async def test_timeout_is_tracked_once(self) -> None:
        client = _make_client(_variation_a())
        task = await _deferred_task(client)
        task = dataclasses.replace(task, timeout_s=0.05)
        result = await run_shadow(
            task, [_Recorder("ShadowProvider", sleep_s=1).handler]
        )
        names = [e[0] for e in _shadow_events(client)]
        assert result.error == "timeout"
        assert result.track_data["configKey"] == "flag"
        assert result.track_data["variationKey"] == "b"
        assert names.count("$ld:ai:generation:error") == 1
        assert names.count("$ld:ai:duration:total") == 1
        assert "$ld:ai:generation:success" not in names

    async def test_handler_timeout_error_is_an_ordinary_failure(self) -> None:
        client = _make_client(_variation_a())
        task = await _deferred_task(client)
        b = _Recorder("ShadowProvider", error=TimeoutError("provider timeout"))
        result = await run_shadow(task, [b.handler])
        names = [e[0] for e in _shadow_events(client)]
        assert result.error == "provider timeout"
        assert names.count("$ld:ai:generation:error") == 1

    async def test_shutdown_cancels_running_shadows(self) -> None:
        _use(_make_client(_variation_a()))
        a = _Recorder("TestProvider", "real answer")
        b = _Recorder("ShadowProvider", sleep_s=5)
        await shadow_config(key="flag", handler=[a.handler, b.handler]).invoke(
            "hi", CONTEXT
        )
        running = list(sr._running)
        assert running
        await shutdown()
        assert not sr._running
        await asyncio.gather(*running, return_exceptions=True)
        assert all(t.cancelled() for t in running)

    def test_shutdown_hook_is_registered(self) -> None:
        assert "Shadow requests" in _hooks

    async def test_reports_only_shadow_config_usage(self) -> None:
        client = _make_client(_variation_a())
        await _invoke(client)
        assert _usage_helpers(client) == ["experimental.shadowConfig"]


class TestDeferred:
    async def test_on_task_receives_each_sampled_shadow_and_nothing_runs(
        self,
    ) -> None:
        shadows = {"fallthrough": [_entry(id="one"), _entry(id="two")]}
        client = _make_client(_variation_a(shadows))
        tasks: list[ShadowTask] = []
        _, a, b = await _invoke(client, on_task=tasks.append)
        assert len(a.calls) == 1
        assert b.calls == []
        assert [t.shadow_request_id for t in tasks] == ["one", "two"]
        assert _shadow_events(client) == []
        assert not sr._running

    async def test_on_task_errors_are_logged_and_swallowed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def explode(task: ShadowTask) -> None:
            raise RuntimeError("queue full")

        with caplog.at_level(logging.WARNING, logger=sr.logger.name):
            response, _, _ = await _invoke(
                _make_client(_variation_a()), on_task=explode
            )
        assert response.response == "real answer"
        assert any("on_task" in m for m in caplog.messages)

    async def test_task_copies_the_context(self) -> None:
        client = _make_client(_variation_a())
        _use(client)
        context: dict[str, Any] = {
            "kind": "multi",
            "user": {"key": "u1", "name": "Ann"},
        }
        tasks: list[ShadowTask] = []
        await shadow_config(
            key="flag",
            handler=[
                _Recorder("TestProvider", "a").handler,
                _Recorder("ShadowProvider", "b").handler,
            ],
            on_task=tasks.append,
        ).invoke("hi", context)
        context["user"]["key"] = "someone-else"
        assert tasks[0].user_context == {
            "kind": "multi",
            "user": {"key": "u1", "name": "Ann"},
        }

    async def test_served_response_survives_a_failure_building_shadows(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _make_client(_variation_a())
        _use(client)
        context: dict[str, Any] = {
            "kind": "user",
            "key": "u1",
            "lock": threading.Lock(),
        }
        tasks: list[ShadowTask] = []
        with caplog.at_level(logging.WARNING, logger=sr.logger.name):
            response = await shadow_config(
                key="flag",
                handler=[
                    _Recorder("TestProvider", "real answer").handler,
                    _Recorder("ShadowProvider", "b").handler,
                ],
                on_task=tasks.append,
            ).invoke("hi", context)
        assert response.response == "real answer"
        assert tasks == []
        assert caplog.messages

    async def test_task_is_self_contained_and_survives_pickle(self) -> None:
        history = [{"role": "user", "content": "earlier"}]
        tasks: list[ShadowTask] = []
        response, _, _ = await _invoke(
            _make_client(_variation_a()),
            user_input="question",
            history=history,
            on_task=tasks.append,
            shadow_timeout_s=12.5,
        )
        task = tasks[0]
        restored = pickle.loads(pickle.dumps(task))
        assert restored == task
        assert restored.config_key == "flag"
        assert restored.shadow_request_id == "shadow-1"
        assert restored.shadow_meta["variationKey"] == "b"
        assert "tools" not in restored.shadow_config
        assert restored.user_input == "question"
        assert restored.history == history
        assert restored.variables == {"topic": "x"}
        assert restored.user_context == CONTEXT
        assert restored.timeout_s == 12.5
        assert restored.track_data["runId"] == response.track_data["runId"]
        assert restored.span_attributes["launchdarkly.is_shadow"] is True
        with pytest.raises(dataclasses.FrozenInstanceError):
            task.timeout_s = 1  # type: ignore[misc]

    async def test_run_shadow_matches_background_mode(self) -> None:
        deferred_client = _make_client(_variation_a())
        task = await _deferred_task(deferred_client)
        result = await run_shadow(task, [_Recorder("ShadowProvider").handler])
        assert isinstance(result, ShadowResult)
        assert result.error is None
        assert result.usage["total"] == 5
        assert result.track_data["isShadow"] is True
        deferred = _without_run_id(_shadow_events(deferred_client))

        lifecycle_module._reset_for_testing()
        background_client = _make_client(_variation_a())
        await _invoke(background_client)
        background = _without_run_id(_shadow_events(background_client))
        assert deferred == background

    async def test_run_shadow_never_raises(self) -> None:
        task = await _deferred_task(_make_client(_variation_a()))
        failed = await run_shadow(
            task, [_Recorder("ShadowProvider", error=RuntimeError("boom")).handler]
        )
        unrouted = await run_shadow(task, [])
        assert failed.error == "boom"
        assert unrouted.error
        for result in (failed, unrouted):
            assert result.track_data["configKey"] == "flag"
            assert result.track_data["variationKey"] == "b"
            assert result.track_data["providerName"] == "ShadowProvider"
            assert SHADOW_FIELDS <= set(result.track_data)
            assert result.track_data["runId"] == task.track_data["runId"]

    async def test_task_carries_the_conversation_id(self) -> None:
        client = _make_client(_variation_a())
        with conversation_id("conv-1"):
            task = await _deferred_task(client)
        assert task.conversation_id == "conv-1"
        assert pickle.loads(pickle.dumps(task)).conversation_id == "conv-1"
        b = _Recorder("ShadowProvider")
        await run_shadow(task, [b.handler])
        assert b.calls[0]["conversation_id"] == "conv-1"

    async def test_task_without_a_bound_conversation_id(self) -> None:
        task = await _deferred_task(_make_client(_variation_a()))
        assert task.conversation_id is None
        b = _Recorder("ShadowProvider")
        await run_shadow(task, [b.handler])
        assert b.calls[0]["conversation_id"] is None

    async def test_task_without_an_id_ignores_the_workers_conversation(self) -> None:
        task = await _deferred_task(_make_client(_variation_a()))
        b = _Recorder("ShadowProvider")
        with conversation_id("worker-conversation"):
            await run_shadow(task, [b.handler])
        assert b.calls[0]["conversation_id"] is None

    async def test_task_id_replaces_the_workers_conversation(self) -> None:
        client = _make_client(_variation_a())
        with conversation_id("conv-1"):
            task = await _deferred_task(client)
        b = _Recorder("ShadowProvider")
        with conversation_id("worker-conversation"):
            await run_shadow(task, [b.handler])
        assert b.calls[0]["conversation_id"] == "conv-1"

    async def test_run_shadow_reports_usage(self) -> None:
        client = _make_client(_variation_a())
        task = await _deferred_task(client)
        await run_shadow(task, [_Recorder("ShadowProvider").handler])
        assert "experimental.runShadow" in _usage_helpers(client)


class TestEvents:
    async def test_shadow_events_describe_the_shadow_variation(self) -> None:
        client = _make_client(_variation_a())
        response, _, _ = await _invoke(client)
        events = _shadow_events(client)
        assert {e[0] for e in events} >= {
            "$ld:ai:duration:total",
            "$ld:ai:generation:success",
            "$ld:ai:tokens:total",
        }
        data = events[0][1]
        assert data["configKey"] == "flag"
        assert data["variationKey"] == "b"
        assert data["version"] == 7
        assert data["modelName"] == "claude-x"
        assert data["providerName"] == "ShadowProvider"
        assert data["modelKey"] == "claude"
        assert data["modelVersion"] == 2
        assert data["runId"] == response.track_data["runId"]
        assert data["isShadow"] is True
        assert data["primaryVariationKey"] == "a"
        assert data["shadowRequestId"] == "shadow-1"
        assert data["ownerKind"] == "member"
        assert data["ownerId"] == "member-1"

    async def test_nothing_is_added_to_primary_events(self) -> None:
        client = _make_client(_variation_a())
        response, _, _ = await _invoke(client)
        assert not (SHADOW_FIELDS & set(response.track_data))
        for _, data in _real_events(client):
            assert not (SHADOW_FIELDS & set(data))

    async def test_shadow_span_attributes_reach_the_root_span(
        self, mock_span: MagicMock
    ) -> None:
        _, _, b = await _invoke(_make_client(_variation_a()))
        set_ld_span_attributes(mock_span, b.calls[0]["variables"])
        mock_span.set_attribute.assert_any_call("launchdarkly.is_shadow", True)
        mock_span.set_attribute.assert_any_call(
            "launchdarkly.shadow.request.id", "shadow-1"
        )
        mock_span.set_attribute.assert_any_call(
            "launchdarkly.shadow.primary.variation.key", "a"
        )

    async def test_shadow_root_span_has_no_feature_flag_event(self) -> None:
        _, a, b = await _invoke(_make_client(_variation_a()))
        primary_span, shadow_span = MagicMock(), MagicMock()
        set_ld_span_attributes(primary_span, a.calls[0]["variables"])
        set_ld_span_attributes(shadow_span, b.calls[0]["variables"])
        assert "feature_flag" in [
            c[0][0] for c in primary_span.add_event.call_args_list
        ]
        shadow_span.add_event.assert_not_called()

    async def test_template_variables_cannot_set_span_attributes(
        self, mock_span: MagicMock
    ) -> None:
        _use(_make_client(_variation_a(shadows={})))
        a = _Recorder("TestProvider", "real answer")
        await config(key="flag", handler=a.handler).invoke(
            "hi", CONTEXT, {"__ld_span": {"launchdarkly.is_shadow": True}}
        )
        set_ld_span_attributes(mock_span, a.calls[0]["variables"])
        names = [c[0][0] for c in mock_span.set_attribute.call_args_list]
        assert "launchdarkly.is_shadow" not in names

    async def test_template_variables_cannot_skip_the_feature_flag_event(
        self, mock_span: MagicMock
    ) -> None:
        _use(_make_client(_variation_a(shadows={})))
        a = _Recorder("TestProvider", "real answer")
        await config(key="flag", handler=a.handler).invoke(
            "hi", CONTEXT, {"__ld_no_feature_flag_event": True}
        )
        set_ld_span_attributes(mock_span, a.calls[0]["variables"])
        assert "feature_flag" in [c[0][0] for c in mock_span.add_event.call_args_list]


class TestFailureEdges:
    async def test_handler_that_swallows_cancellation_is_still_a_timeout(self) -> None:
        client = _make_client(_variation_a())
        task = await _deferred_task(client)
        task = dataclasses.replace(task, timeout_s=0.05)

        async def swallows(*args: Any) -> dict[str, Any]:
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
            return {"output": "late", "usage": {}}

        handler = ProviderHandler(
            fn=swallows, provides_for=("ShadowProvider", "messages")
        )  # type: ignore[arg-type]
        result = await run_shadow(task, [handler])
        names = [e[0] for e in _shadow_events(client)]
        assert result.error == "timeout"
        assert "$ld:ai:generation:success" not in names
        assert names.count("$ld:ai:generation:error") == 1

    async def test_error_without_a_message_uses_the_type_name(self) -> None:
        task = await _deferred_task(_make_client(_variation_a()))
        result = await run_shadow(
            task, [_Recorder("ShadowProvider", error=TimeoutError()).handler]
        )
        assert result.error == "TimeoutError"

    def test_rate_above_one_is_treated_as_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        top = SimpleNamespace(hexdigest=lambda: "f" * 40)
        monkeypatch.setattr(sr.hashlib, "sha1", lambda *a, **k: top)
        assert sr._in_sample("shadow-1", CONTEXT, 1.0)
        assert sr._in_sample("shadow-1", CONTEXT, 1.5)
        assert not sr._in_sample("shadow-1", CONTEXT, 0.999)

    async def test_failed_served_call_hands_over_no_task(self) -> None:
        _use(_make_client(_variation_a()))
        tasks: list[ShadowTask] = []
        failing = _Recorder("TestProvider", error=RuntimeError("primary failed"))
        with pytest.raises(RuntimeError):
            await shadow_config(
                key="flag",
                handler=[failing.handler, _Recorder("ShadowProvider").handler],
                on_task=tasks.append,
            ).invoke("hi", CONTEXT)
        assert tasks == []

    async def test_abandoned_stream_starts_no_shadow(self) -> None:
        client = _use(_make_client(_variation_a()))
        tasks: list[ShadowTask] = []
        b = _Recorder("ShadowProvider")
        for on_task in (None, tasks.append):
            stream = shadow_config(
                key="flag",
                handler=[_Recorder("TestProvider", "real answer").handler, b.handler],
                on_task=on_task,
            ).stream("hi", CONTEXT)
            async with aclosing(stream):
                async for _ in stream:
                    break
        await sr._wait_for_shadows_for_testing()
        assert tasks == []
        assert b.calls == []
        assert _shadow_events(client) == []

    async def test_failing_shadow_leaves_served_events_unchanged(self) -> None:
        with_shadow = _make_client(_variation_a())
        await _invoke(
            with_shadow, b=_Recorder("ShadowProvider", error=RuntimeError("boom"))
        )
        lifecycle_module._reset_for_testing()
        without_shadow = _make_client(_variation_a(shadows={}))
        await _invoke(without_shadow)
        assert _without_run_id(_real_events(with_shadow)) == _without_run_id(
            _events(without_shadow)
        )

    async def test_shadow_with_malformed_tools_still_runs(self) -> None:
        variation = _variation_b()
        variation["tools"] = "not-a-tools-map"
        _, _, b = await _invoke(_make_client(_variation_a(variations={"b": variation})))
        assert len(b.calls) == 1
        assert "tools" not in b.calls[0]["config"]


JUDGE_VERDICT = '{"score": 0.8, "reasoning": "good"}'


def _judge_variation() -> dict[str, Any]:
    return {
        "_ldMeta": {
            "enabled": True,
            "variationKey": "jv",
            "version": 1,
            "mode": "judge",
        },
        "model": {"name": "judge-model"},
        "provider": {"name": "JudgeProvider"},
        "instructions": "Judge.",
        "evaluationMetricKey": "judge-score",
    }


def _judged_client(
    b_rate: float = 1, a_judged: bool = False
) -> tuple[MagicMock, dict[str, Any]]:
    value = _variation_a(
        variations={"b": _variation_b(judges=[{"key": "j", "samplingRate": b_rate}])}
    )
    if a_judged:
        value["judgeConfiguration"] = {"judges": [{"key": "j", "samplingRate": 1}]}
    client = _make_client(value)

    async def variation(key: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return _judge_variation() if key == "j" else value

    client.variation = AsyncMock(side_effect=variation)
    return client, value


def _judge_events(client: MagicMock) -> list[dict[str, Any]]:
    return [d for n, d in _events(client) if n == "judge-score"]


async def _invoke_judged(
    client: MagicMock,
    *,
    b: _Recorder | None = None,
    judge: _Recorder | None = None,
    **shadow_args: Any,
) -> tuple[Any, _Recorder, _Recorder, _Recorder]:
    _use(client)
    a = _Recorder("TestProvider", "real answer")
    b = b or _Recorder("ShadowProvider", "shadow answer")
    judge = judge or _Recorder("JudgeProvider", JUDGE_VERDICT)
    response = await shadow_config(
        key="flag", handler=[a.handler, b.handler, judge.handler], **shadow_args
    ).invoke("question", CONTEXT)
    await sr._wait_for_shadows_for_testing()
    return response, a, b, judge


class TestJudges:
    async def test_judge_scores_the_shadow_response(self) -> None:
        client, _ = _judged_client()
        response, _, b, judge = await _invoke_judged(client)
        assert response.response == "real answer"
        assert len(b.calls) == 1
        assert len(judge.calls) == 1
        assert judge.calls[0]["variables"]["response_to_evaluate"] == "shadow answer"
        assert judge.calls[0]["variables"]["input"] == "question"

    async def test_judge_runs_after_the_caller_returns(self) -> None:
        client, _ = _judged_client()
        _use(client)
        a = _Recorder("TestProvider", "real answer")
        b = _Recorder("ShadowProvider", "shadow answer")
        judge = _Recorder("JudgeProvider", JUDGE_VERDICT)
        await shadow_config(
            key="flag", handler=[a.handler, b.handler, judge.handler]
        ).invoke("question", CONTEXT)
        assert judge.calls == []
        await sr._wait_for_shadows_for_testing()
        assert len(judge.calls) == 1

    async def test_shadow_judge_event_is_tracked_under_the_shadow(self) -> None:
        client, _ = _judged_client(a_judged=True)
        await _invoke_judged(client)
        shadow = [d for d in _judge_events(client) if d.get("isShadow")]
        real = [d for d in _judge_events(client) if not d.get("isShadow")]
        assert len(shadow) == 1
        assert len(real) == 1
        assert shadow[0]["variationKey"] == "b"
        assert shadow[0]["shadowRequestId"] == "shadow-1"
        assert shadow[0]["primaryVariationKey"] == "a"
        assert shadow[0]["judgeConfigKey"] == "j"
        assert shadow[0]["runId"] == real[0]["runId"]
        assert not SHADOW_FIELDS & real[0].keys()

    @pytest.mark.parametrize(("rate", "runs"), [(0, 0), (1, 1)])
    async def test_judge_sampling_rate_applies(self, rate: float, runs: int) -> None:
        client, _ = _judged_client(b_rate=rate)
        _, _, b, judge = await _invoke_judged(client)
        assert len(b.calls) == 1
        assert len(judge.calls) == runs
        assert len(_judge_events(client)) == runs

    @pytest.mark.parametrize(
        "b",
        [
            _Recorder("ShadowProvider", error=RuntimeError("boom")),
            _Recorder("ShadowProvider", sleep_s=1),
        ],
        ids=["error", "timeout"],
    )
    async def test_no_judge_when_the_shadow_fails(self, b: _Recorder) -> None:
        client, _ = _judged_client()
        _, _, _, judge = await _invoke_judged(client, b=b, shadow_timeout_s=0.05)
        assert judge.calls == []
        assert _judge_events(client) == []

    async def test_judge_failure_is_not_a_shadow_failure(self) -> None:
        client, _ = _judged_client()
        task = await _deferred_task_judged(client)
        judge = _Recorder("JudgeProvider", "not json")
        result = await run_shadow(
            task, [_Recorder("ShadowProvider", "shadow answer").handler, judge.handler]
        )
        assert len(judge.calls) == 1
        assert result.error is None
        assert result.judge_results is None
        assert "$ld:ai:generation:success" in [e[0] for e in _shadow_events(client)]

    async def test_timeout_does_not_cover_judges(self) -> None:
        client, _ = _judged_client()
        task = dataclasses.replace(await _deferred_task_judged(client), timeout_s=0.2)
        judge = _Recorder("JudgeProvider", JUDGE_VERDICT, sleep_s=0.3)
        result = await run_shadow(
            task, [_Recorder("ShadowProvider", "shadow answer").handler, judge.handler]
        )
        assert result.error is None
        assert result.judge_results is not None
        assert result.judge_results["j"].score == 0.8

    async def test_result_carries_judge_results_only_when_a_judge_ran(self) -> None:
        client, _ = _judged_client()
        task = await _deferred_task_judged(client)
        handlers = [
            _Recorder("ShadowProvider", "shadow answer").handler,
            _Recorder("JudgeProvider", JUDGE_VERDICT).handler,
        ]
        judged = await run_shadow(task, handlers)
        assert judged.judge_results is not None
        assert judged.judge_results["j"].score == 0.8
        assert "shadow answer" not in repr(judged)

        unjudged = await run_shadow(
            await _deferred_task(_make_client(_variation_a())), handlers
        )
        assert unjudged.judge_results is None

    async def test_skip_judges_applies_to_the_served_variation_only(self) -> None:
        client, _ = _judged_client(a_judged=True)
        response, _, _, judge = await _invoke_judged(client, skip_judges=True)
        assert response.judge_tasks is not None
        assert len(judge.calls) == 1
        assert [d.get("isShadow") for d in _judge_events(client)] == [True]

    async def test_deferred_and_background_emit_the_same_judge_events(self) -> None:
        background, _ = _judged_client()
        await _invoke_judged(background)

        deferred, _ = _judged_client()
        task = await _deferred_task_judged(deferred)
        deferred.track.reset_mock()
        await run_shadow(
            task,
            [
                _Recorder("ShadowProvider", "shadow answer").handler,
                _Recorder("JudgeProvider", JUDGE_VERDICT).handler,
            ],
        )
        assert _without_run_id(_shadow_events(background)) == _without_run_id(
            _shadow_events(deferred)
        )

    async def test_judged_task_survives_pickle(self) -> None:
        client, _ = _judged_client()
        task = await _deferred_task_judged(client)
        assert pickle.loads(pickle.dumps(task)) == task
        assert task.shadow_config["judgeConfiguration"]["judges"][0]["key"] == "j"


async def _deferred_task_judged(client: MagicMock) -> ShadowTask:
    tasks: list[ShadowTask] = []
    await _invoke_judged(client, on_task=tasks.append)
    assert len(tasks) == 1
    return tasks[0]
