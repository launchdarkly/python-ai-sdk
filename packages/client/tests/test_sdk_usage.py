"""Tests for TESTING.md §3.27 helper usage events."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import (
    SDK_INFO_CONTEXT,
    ProviderHandler,
    __version__,
    build_judge_tasks,
    config,
    create_handler,
    graph,
    init_client,
    init_evaluations,
    inspect_config,
    resolve_graph,
    run_judge,
    shutdown,
)

CONTEXT = {"kind": "user", "key": "user-1"}
HELPER = "client.buildJudgeTasks"


def _client(variation: Any = None) -> MagicMock:
    client = MagicMock()
    client.track = MagicMock()
    client.variation = AsyncMock(return_value=variation)
    client.flush = AsyncMock()
    client.close = AsyncMock()
    return client


async def _install(variation: Any = None) -> MagicMock:
    await shutdown()
    client = _client(variation)
    with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
        await init_client(client=client)
    client.track.reset_mock()
    return client


def _usage(client: MagicMock, helper: str | None = None) -> list[Any]:
    return [
        call
        for call in client.track.call_args_list
        if call.args[0] == "$ld:ai:sdk:usage"
        and (helper is None or call.args[2].get("helper") == helper)
    ]


def _expect_usage(client: MagicMock, helper: str) -> None:
    calls = _usage(client, helper)
    assert len(calls) == 1
    assert calls[0].args[1] == SDK_INFO_CONTEXT
    assert calls[0].args[2] == {
        "aiSdkName": "launchdarkly-ai-server",
        "aiSdkVersion": __version__,
        "aiSdkLanguage": "python",
        "helper": helper,
    }
    assert calls[0].args[3] == 1


async def _settle(value: Any) -> None:
    if hasattr(value, "__await__"):
        try:
            await value
        except Exception:
            return


def _judge_kwargs() -> dict[str, Any]:
    return {
        "config": {
            "model": {"name": "m"},
            "provider": {"name": "OpenAI"},
            "instructions": "i",
        },
        "user_context": CONTEXT,
        "handler": MagicMock(),
        "llm_response": "",
        "base_track_data": {
            "runId": "r",
            "configKey": "k",
            "variationKey": "v",
            "version": 1,
            "modelName": "m",
            "providerName": "OpenAI",
        },
    }


def _call(helper: str) -> Any:
    if helper == "client.config.invoke":
        return config(key="k").invoke("q", CONTEXT)
    if helper == "client.config.stream":
        return config(key="k").stream("q", CONTEXT)
    if helper == "client.graph.invoke":
        return graph("k").invoke("q", CONTEXT)
    if helper == "client.graph.stream":
        return graph("k").stream("q", CONTEXT)
    if helper == "client.resolveGraph":
        return resolve_graph("k", context=CONTEXT)
    if helper == "client.inspectConfig":
        return inspect_config("k", CONTEXT)
    if helper == "client.runJudge":
        return run_judge(MagicMock(), [])
    if helper == "client.buildJudgeTasks":
        return build_judge_tasks(**_judge_kwargs())
    if helper == "client.createHandler":

        async def _fn(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {"output": ""}

        create_handler(("OpenAI", "messages"), _fn)
        return None
    try:
        init_evaluations()
    except Exception:
        return None
    return None


@pytest.fixture
async def client() -> Any:
    installed = await _install()
    yield installed
    await shutdown()


@pytest.mark.parametrize(
    "helper",
    [
        "client.config.invoke",
        "client.config.stream",
        "client.graph.invoke",
        "client.graph.stream",
        "client.resolveGraph",
        "client.inspectConfig",
        "client.runJudge",
        "client.buildJudgeTasks",
        "client.createHandler",
        "client.init_evaluations",
    ],
)
async def test_helper_sends_one_usage_event(client: MagicMock, helper: str) -> None:
    await _settle(_call(helper))
    _expect_usage(client, helper)
    if helper == "client.graph.invoke":
        assert _usage(client, "client.graph.stream") == []
    if helper == "client.graph.stream":
        assert _usage(client, "client.graph.invoke") == []


async def test_repeat_call_sends_nothing_until_shutdown(client: MagicMock) -> None:
    await build_judge_tasks(**_judge_kwargs())
    _expect_usage(client, HELPER)
    await build_judge_tasks(**_judge_kwargs())
    assert len(_usage(client, HELPER)) == 1

    await shutdown()
    fresh = await _install()
    await build_judge_tasks(**_judge_kwargs())
    _expect_usage(fresh, HELPER)


async def test_helper_before_client_is_sent_on_init() -> None:
    await shutdown()
    lifecycle_module._reset_for_testing()
    await build_judge_tasks(**_judge_kwargs())
    client = _client()
    assert _usage(client, HELPER) == []
    with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
        await init_client(client=client)
    _expect_usage(client, HELPER)
    await shutdown()


async def test_track_failure_does_not_fail_the_helper_or_retry(
    client: MagicMock,
) -> None:
    client.track.side_effect = RuntimeError("client closed")
    result = await build_judge_tasks(**_judge_kwargs())
    assert result == []
    assert len(_usage(client, HELPER)) == 1

    client.track.reset_mock(side_effect=True)
    await build_judge_tasks(**_judge_kwargs())
    assert _usage(client, HELPER) == []


async def test_generation_events_gain_no_usage_fields() -> None:
    await shutdown()
    try:
        installed = await _install(
            {
                "model": {"name": "gpt-4o"},
                "provider": {"name": "OpenAI"},
                "instructions": "Be helpful.",
                "_ldMeta": {
                    "enabled": True,
                    "variationKey": "v1",
                    "version": 1,
                    "mode": "messages",
                },
            }
        )

        async def _handler(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {"output": "ok", "usage": {"input_tokens": 2, "output_tokens": 3}}

        await config(
            key="flag",
            handler=ProviderHandler(fn=_handler, provides_for=("OpenAI", "messages")),
        ).invoke("q", CONTEXT)

        _expect_usage(installed, "client.config.invoke")
        others = [
            call
            for call in installed.track.call_args_list
            if call.args[0] not in {"$ld:ai:sdk:usage", "$ld:ai:sdk:info"}
        ]
        names = [call.args[0] for call in others]
        assert "$ld:ai:duration:total" in names
        assert "$ld:ai:generation:success" in names
        assert "$ld:ai:tokens:total" in names
        for call in others:
            payload = call.args[2]
            assert "helper" not in payload
            assert "aiSdkName" not in payload
            assert "aiSdkVersion" not in payload
            assert "aiSdkLanguage" not in payload
    finally:
        await shutdown()
