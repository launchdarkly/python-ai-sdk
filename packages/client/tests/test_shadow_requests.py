"""Tests for experimental shadow requests (``shadow_config``)."""

import asyncio
import dataclasses
import logging
import pickle
import time
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import launchdarkly_ai_server
import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import ProviderHandler, agui, config, shutdown
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


def _variation_b(enabled: bool = True) -> dict[str, Any]:
    return {
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
        "judgeConfiguration": {"judges": [{"key": "j", "samplingRate": 1}]},
    }


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
        self.calls.append({"config": cfg, "streamed": True})
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
        assert "judgeConfiguration" not in call["config"]
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
        assert unrouted.track_data["isShadow"] is True

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
