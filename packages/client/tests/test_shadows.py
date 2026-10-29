"""Tests for shadow requests."""

import asyncio
import pickle
import time
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import (
    ProviderHandler,
    ShadowTask,
    config,
    run_shadow,
    shutdown,
)
from launchdarkly_ai_server import shadows as sh
from launchdarkly_ai_server.sdk_usage import reset_sdk_usage
from launchdarkly_ai_server.utils import set_ld_span_attributes

CONTEXT = {"kind": "user", "key": "u1"}
FUTURE = int(time.time() * 1000) + 3_600_000
PAST = int(time.time() * 1000) - 1
_SHADOW_FIELDS = {
    "isShadow",
    "primaryVariationKey",
    "shadowRequestId",
    "ownerKind",
    "ownerId",
}
_BOOKKEEPING = {"$ld:ai:sdk:usage", "$ld:ai:sdk:info"}


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


def _variation_b() -> dict[str, Any]:
    return {
        "_ldMeta": {
            "enabled": True,
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


def _detail(value: dict[str, Any], reason: dict[str, Any] | None) -> SimpleNamespace:
    return SimpleNamespace(value=value, reason=reason)


FALLTHROUGH = {"kind": "FALLTHROUGH"}


def _make_client(
    value: dict[str, Any], reason: dict[str, Any] | None = FALLTHROUGH
) -> MagicMock:
    c = MagicMock()
    c.track = MagicMock()
    c.flush = AsyncMock()
    c.close = AsyncMock()
    c.variation = AsyncMock(return_value=value)
    c.variation_detail = AsyncMock(return_value=_detail(value, reason))
    return c


class _Recorder:
    def __init__(
        self,
        provider: str,
        output: str = "ok",
        *,
        raises: bool = False,
        sleep_s: float = 0,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._output = output
        self._raises = raises
        self._sleep_s = sleep_s
        self._error = error
        self.handler = ProviderHandler(fn=self._fn, provides_for=(provider, "messages"))  # type: ignore[arg-type]

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
        if self._raises:
            raise RuntimeError("shadow boom")
        return {
            "output": self._output,
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }


def _events(client: MagicMock) -> list[tuple[str, dict[str, Any], Any]]:
    return [
        (c[0][0], c[0][2], c[0][3])
        for c in client.track.call_args_list
        if c[0][0] not in _BOOKKEEPING
    ]


def _shadow_events(client: MagicMock) -> list[tuple[str, dict[str, Any], Any]]:
    return [e for e in _events(client) if e[1].get("isShadow")]


def _real_events(client: MagicMock) -> list[tuple[str, dict[str, Any], Any]]:
    return [e for e in _events(client) if not e[1].get("isShadow")]


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
    **config_args: Any,
) -> tuple[Any, _Recorder, _Recorder]:
    _use(client)
    a = _Recorder("TestProvider", "real answer")
    b = _Recorder("ShadowProvider", "shadow answer")
    response = await config(
        key="flag", handler=[a.handler, b.handler], **config_args
    ).invoke(user_input, CONTEXT, {"topic": "x"}, history)
    await sh._wait_for_shadows_for_testing()
    return response, a, b


async def _deferred_task(client: MagicMock) -> ShadowTask:
    response, _, _ = await _invoke(client, skip_shadows=True)
    assert response.shadow_tasks is not None
    return response.shadow_tasks[0]


class TestRunning:
    async def test_runs_the_fallthrough_shadow(self) -> None:
        client = _make_client(_variation_a())
        _, a, b = await _invoke(client)
        assert len(a.calls) == 1
        assert len(b.calls) == 1
        client.variation_detail.assert_called_once()
        client.variation.assert_not_called()

    async def test_client_without_variation_detail_runs_no_shadows(self) -> None:
        client = _make_client(_variation_a())
        del client.variation_detail
        response, a, b = await _invoke(client)
        assert response.response == "real answer"
        assert len(a.calls) == 1
        assert b.calls == []

    @pytest.mark.parametrize("timeout_s", [0, -1, float("inf"), float("nan")])
    def test_timeout_must_be_positive_and_finite(self, timeout_s: float) -> None:
        with pytest.raises(ValueError):
            config(key="flag", shadow_timeout_s=timeout_s)

    async def test_shutdown_cancels_running_shadows(self) -> None:
        _use(_make_client(_variation_a()))
        a = _Recorder("TestProvider", "real answer")
        b = _Recorder("ShadowProvider", sleep_s=5)
        await config(key="flag", handler=[a.handler, b.handler]).invoke("hi", CONTEXT)
        assert sh._running
        await shutdown()
        assert not sh._running

    async def test_same_input_and_stripped_config(self) -> None:
        history = [{"role": "user", "content": "earlier"}]
        _, a, b = await _invoke(
            _make_client(_variation_a()), user_input="question", history=history
        )
        call = b.calls[0]
        assert call["input"] == "question"
        assert call["history"] == history
        assert call["variables"]["topic"] == "x"
        assert "tools" not in call["config"]
        assert "judgeConfiguration" not in call["config"]
        assert "shadowConfiguration" not in a.calls[0]["config"]
        assert "shadowConfiguration" not in call["config"]
        assert not call["tools"]

    async def test_shadow_answer_never_joins_history(self) -> None:
        history = [{"role": "user", "content": "earlier"}]
        await _invoke(_make_client(_variation_a()), history=history)
        assert history == [{"role": "user", "content": "earlier"}]

    async def test_failing_shadow_never_reaches_the_caller(self) -> None:
        client = _use(_make_client(_variation_a()))
        a = _Recorder("TestProvider", "real answer")
        b = _Recorder("ShadowProvider", raises=True)
        response = await config(key="flag", handler=[a.handler, b.handler]).invoke(
            "hi", CONTEXT
        )
        await sh._wait_for_shadows_for_testing()
        assert response.response == "real answer"
        assert "$ld:ai:generation:success" in [e[0] for e in _real_events(client)]
        assert "$ld:ai:generation:error" in [e[0] for e in _shadow_events(client)]

    async def test_timeout_is_tracked_once(self) -> None:
        client = _use(_make_client(_variation_a()))
        a = _Recorder("TestProvider", "real answer")
        b = _Recorder("ShadowProvider", sleep_s=1)
        response = await config(
            key="flag", handler=[a.handler, b.handler], shadow_timeout_s=0.05
        ).invoke("hi", CONTEXT)
        await sh._wait_for_shadows_for_testing()
        assert response.response == "real answer"
        names = [e[0] for e in _shadow_events(client)]
        assert names.count("$ld:ai:generation:error") == 1
        assert names.count("$ld:ai:duration:total") == 1

    async def test_handler_timeout_error_is_tracked_once(self) -> None:
        client = _make_client(_variation_a())
        task = await _deferred_task(client)
        b = _Recorder("ShadowProvider", error=TimeoutError("provider timeout"))
        result = await run_shadow(task, [b.handler])
        names = [e[0] for e in _shadow_events(client)]
        assert result.error == "provider timeout"
        assert names.count("$ld:ai:generation:error") == 1

    async def test_real_run_is_unchanged(self) -> None:
        baseline_client = _make_client(_variation_a(shadows={}))
        baseline, _, _ = await _invoke(baseline_client)
        lifecycle_module._reset_for_testing()
        client = _make_client(_variation_a())
        response, _, _ = await _invoke(client)
        assert response.response == baseline.response
        assert response.usage == baseline.usage
        assert [e[0] for e in _real_events(client)] == [
            e[0] for e in _real_events(baseline_client)
        ]

    async def test_stream_runs_the_shadow_as_invoke(self) -> None:
        _use(_make_client(_variation_a()))
        a = _Recorder("TestProvider", "real answer")
        b = _Recorder("ShadowProvider", "shadow answer")
        events = [
            e
            async for e in config(key="flag", handler=[a.handler, b.handler]).stream(
                "hi", CONTEXT
            )
        ]
        await sh._wait_for_shadows_for_testing()
        assert events[-1]["type"] == "done"
        assert "shadow_tasks" not in events[-1]
        assert len(b.calls) == 1


class TestPayload:
    @pytest.mark.parametrize(
        "shadow_configuration",
        [
            "nope",
            {"shadows": {"fallthrough": "nope"}, "variations": {"b": _variation_b()}},
            {
                "shadows": {"fallthrough": [{"id": "x"}]},
                "variations": {"b": _variation_b()},
            },
            {
                "shadows": {"fallthrough": [_entry(samplingRate="high")]},
                "variations": {"b": _variation_b()},
            },
        ],
    )
    async def test_malformed_configuration_is_ignored(
        self, shadow_configuration: Any
    ) -> None:
        value = _variation_a()
        value["shadowConfiguration"] = shadow_configuration
        response, _, b = await _invoke(_make_client(value))
        assert response.response == "real answer"
        assert b.calls == []


class TestSelection:
    async def test_rule_match_uses_only_that_rule(self) -> None:
        shadows = {
            "r1": [_entry(id="rule-shadow")],
            "fallthrough": [_entry(id="ft-shadow")],
        }
        client = _make_client(
            _variation_a(shadows),
            {"kind": "RULE_MATCH", "ruleId": "r1", "ruleIndex": 0},
        )
        await _invoke(client)
        ids = {e[1]["shadowRequestId"] for e in _shadow_events(client)}
        assert ids == {"rule-shadow"}

    @pytest.mark.parametrize(
        "reason",
        [
            {"kind": "TARGET_MATCH"},
            {"kind": "OFF"},
            {"kind": "PREREQUISITE_FAILED"},
            {"kind": "ERROR"},
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
        ids = {e[1]["shadowRequestId"] for e in _shadow_events(client)}
        assert ids == {"live"}

    async def test_missing_variation_is_skipped(self) -> None:
        _, _, b = await _invoke(
            _make_client(_variation_a({"fallthrough": [_entry(variationKey="gone")]}))
        )
        assert b.calls == []

    async def test_no_matching_handler_is_skipped(self) -> None:
        client = _use(_make_client(_variation_a()))
        a = _Recorder("TestProvider", "real answer")
        response = await config(key="flag", handler=a.handler).invoke("hi", CONTEXT)
        await sh._wait_for_shadows_for_testing()
        assert response.response == "real answer"
        assert _shadow_events(client) == []

    def test_sampling_is_sticky(self) -> None:
        assert sh._in_sample("s1", CONTEXT, 0.5) == sh._in_sample("s1", CONTEXT, 0.5)

    def test_sampling_honors_the_rate(self) -> None:
        hits = sum(
            sh._in_sample("s1", {"kind": "user", "key": f"k{i}"}, 0.2)
            for i in range(10_000)
        )
        assert 1800 <= hits <= 2200

    def test_sampling_edges(self) -> None:
        contexts = [{"kind": "user", "key": f"k{i}"} for i in range(200)]
        assert all(sh._in_sample("s1", c, 1.0) for c in contexts)
        assert not any(sh._in_sample("s1", c, 0.0) for c in contexts)

    def test_sampling_matches_typescript(self) -> None:
        assert sh._in_sample("s1", CONTEXT, 0.732)
        assert not sh._in_sample("s1", CONTEXT, 0.731)

    def test_context_without_key_is_not_sampled(self) -> None:
        assert not sh._in_sample("s1", {"kind": "user"}, 1.0)


class TestDeferred:
    async def test_skip_shadows_returns_tasks_and_runs_nothing(self) -> None:
        client = _make_client(_variation_a())
        response, a, b = await _invoke(client, skip_shadows=True)
        assert len(a.calls) == 1
        assert b.calls == []
        assert response.shadow_tasks is not None
        assert len(response.shadow_tasks) == 1
        assert _shadow_events(client) == []

    async def test_tasks_are_none_when_shadows_run_in_the_background(self) -> None:
        response, _, _ = await _invoke(_make_client(_variation_a()))
        assert response.shadow_tasks is None

    async def test_stream_done_event_carries_tasks(self) -> None:
        _use(_make_client(_variation_a()))
        a = _Recorder("TestProvider", "real answer")
        events = [
            e
            async for e in config(
                key="flag", handler=a.handler, skip_shadows=True
            ).stream("hi", CONTEXT)
        ]
        assert events[-1]["type"] == "done"
        assert events[-1]["shadow_tasks"] == []

    async def test_task_survives_pickle(self) -> None:
        task = await _deferred_task(_make_client(_variation_a()))
        restored = pickle.loads(pickle.dumps(task))
        assert restored == task
        assert restored.shadow_request_id == "shadow-1"

    async def test_run_shadow_matches_background_mode(self) -> None:
        deferred_client = _make_client(_variation_a())
        task = await _deferred_task(deferred_client)
        result = await run_shadow(task, [_Recorder("ShadowProvider").handler])
        assert result.error is None
        assert result.track_data["isShadow"] is True
        deferred = [
            (e[0], {k: v for k, v in e[1].items() if k != "runId"})
            for e in _shadow_events(deferred_client)
        ]

        lifecycle_module._reset_for_testing()
        background_client = _make_client(_variation_a())
        await _invoke(background_client)
        background = [
            (e[0], {k: v for k, v in e[1].items() if k != "runId"})
            for e in _shadow_events(background_client)
        ]
        assert deferred == background

    async def test_run_shadow_never_raises(self) -> None:
        task = await _deferred_task(_make_client(_variation_a()))
        result = await run_shadow(
            task, [_Recorder("ShadowProvider", raises=True).handler]
        )
        assert result.error

    async def test_run_shadow_reports_usage(self) -> None:
        client = _make_client(_variation_a())
        task = await _deferred_task(client)
        await run_shadow(task, [_Recorder("ShadowProvider").handler])
        helpers = [
            c[0][2].get("helper")
            for c in client.track.call_args_list
            if c[0][0] == "$ld:ai:sdk:usage"
        ]
        assert "client.runShadow" in helpers


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

    async def test_nothing_is_added_to_real_events(self) -> None:
        client = _make_client(_variation_a())
        response, _, _ = await _invoke(client)
        assert not (_SHADOW_FIELDS & set(response.track_data))
        for _, data, _ in _real_events(client):
            assert not (_SHADOW_FIELDS & set(data))

    def test_span_attributes_are_set_on_the_root_span(
        self, mock_span: MagicMock
    ) -> None:
        set_ld_span_attributes(
            mock_span,
            {
                "__ld": {"configKey": "flag", "variationKey": "b", "runId": "r"},
                "__ld_span": {"launchdarkly.is_shadow": True},
            },
        )
        mock_span.set_attribute.assert_any_call("launchdarkly.is_shadow", True)

    async def test_shadow_span_attributes_reach_the_handler(self) -> None:
        _, _, b = await _invoke(_make_client(_variation_a()))
        assert b.calls[0]["variables"]["__ld_span"] == {
            "launchdarkly.is_shadow": True,
            "launchdarkly.shadow.request.id": "shadow-1",
            "launchdarkly.shadow.primary.variation.key": "a",
        }

    async def test_template_variables_cannot_set_span_attributes(self) -> None:
        _use(_make_client(_variation_a()))
        a = _Recorder("TestProvider", "real answer")
        await config(key="flag", handler=a.handler).invoke(
            "hi", CONTEXT, {"__ld_span": {"x": 1}}
        )
        assert "__ld_span" not in a.calls[0]["variables"]
