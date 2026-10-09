"""Tests for TESTING.md §3.27 helper usage events."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
import launchdarkly_ai_server.sdk_usage as sdk_usage_module
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


def _helpers(client: MagicMock) -> list[str]:
    return [call.args[2].get("helper") for call in _usage(client)]


_USAGE_FIELDS = (
    "helper",
    "aiSdkName",
    "aiSdkVersion",
    "aiSdkLanguage",
    "helperPackageName",
    "helperPackageVersion",
)

_ENABLED_CONFIG: dict[str, Any] = {
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


def _expect_usage(client: MagicMock, helper: str) -> None:
    calls = _usage(client, helper)
    assert len(calls) == 1
    assert calls[0].args[1] == SDK_INFO_CONTEXT
    assert calls[0].args[2] == {
        "aiSdkName": "launchdarkly-ai-server",
        "aiSdkVersion": __version__,
        "aiSdkLanguage": "python",
        "helper": helper,
        "helperPackageName": "launchdarkly-ai-server",
        "helperPackageVersion": __version__,
    }
    # A client.* helper belongs to the core package.
    payload = calls[0].args[2]
    assert payload["helperPackageName"] == payload["aiSdkName"]
    assert payload["helperPackageVersion"] == payload["aiSdkVersion"]
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
    # One call emits only its own helper: config.invoke emits no client.createHandler,
    # client.buildJudgeTasks, and so on.
    assert _helpers(client) == [helper]
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


async def test_held_helper_is_sent_on_sdk_key_init() -> None:
    await shutdown()
    lifecycle_module._reset_for_testing()
    await build_judge_tasks(**_judge_kwargs())
    stub = _client()
    ld_module = MagicMock()
    ld_module.Config = MagicMock(return_value=MagicMock())
    ld_module.LDClient = MagicMock(return_value=stub)
    with (
        patch.object(lifecycle_module, "_setup_telemetry", return_value=None),
        patch("importlib.import_module", return_value=ld_module),
    ):
        await init_client({"sdkKey": "sdk-key"})
    ld_module.LDClient.assert_called_once()
    _expect_usage(stub, HELPER)
    await shutdown()


async def test_test_setter_does_not_flush_but_already_initialized_init_does() -> None:
    await shutdown()
    lifecycle_module._reset_for_testing()
    await build_judge_tasks(**_judge_kwargs())
    client = _client()
    lifecycle_module._set_client_for_testing(client)
    assert _usage(client) == []

    # init_client returns the existing client on this path and flushes what was held.
    assert await init_client() is client
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


async def test_payload_failure_does_not_fail_the_helper_or_retry(
    client: MagicMock,
) -> None:
    with patch.object(
        sdk_usage_module, "_version", side_effect=RuntimeError("no version")
    ) as version:
        assert await build_judge_tasks(**_judge_kwargs()) == []
        assert await build_judge_tasks(**_judge_kwargs()) == []
    # Building the payload threw before track, once. The helper counts as reported.
    assert version.call_count == 1
    assert _usage(client) == []
    await build_judge_tasks(**_judge_kwargs())
    assert _usage(client) == []


async def test_held_payload_failure_on_flush_is_swallowed() -> None:
    await shutdown()
    lifecycle_module._reset_for_testing()
    await build_judge_tasks(**_judge_kwargs())
    client = _client()
    with (
        patch.object(lifecycle_module, "_setup_telemetry", return_value=None),
        patch.object(
            sdk_usage_module, "_version", side_effect=RuntimeError("no version")
        ),
    ):
        assert await init_client(client=client) is client
    assert _usage(client) == []
    await build_judge_tasks(**_judge_kwargs())
    assert _usage(client) == []
    await shutdown()


async def test_config_invoke_does_not_report_internal_judge_tasks() -> None:
    """``skip_judges`` builds judge tasks through the internal path."""
    judged = {
        **_ENABLED_CONFIG,
        "judgeConfiguration": {"judges": [{"key": "judge", "samplingRate": 1}]},
    }
    installed = await _install(judged)
    try:

        async def _handler(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {"output": "ok"}

        result = await config(
            key="flag",
            handler=ProviderHandler(fn=_handler, provides_for=("OpenAI", "messages")),
            skip_judges=True,
        ).invoke("q", CONTEXT)
        assert result.judge_tasks
        assert _helpers(installed) == ["client.config.invoke"]
    finally:
        await shutdown()


@pytest.mark.parametrize("method", ["invoke", "stream"])
async def test_helpers_called_from_user_handler_and_tool_still_report(
    method: str,
) -> None:
    """Code the application passes in is outside the SDK, under invoke and stream alike."""
    installed = await _install(_ENABLED_CONFIG)
    try:

        async def _lookup() -> str:
            await build_judge_tasks(**_judge_kwargs())
            return "found"

        async def _handler(
            _config: Any,
            _input: Any,
            tool_handlers: dict[str, Any],
            *_a: Any,
            **_k: Any,
        ) -> dict[str, Any]:
            create_handler(("OpenAI", "messages"), _handler)
            return {"output": await tool_handlers["lookup"]()}

        instance = config(
            key="flag",
            handler=ProviderHandler(fn=_handler, provides_for=("OpenAI", "messages")),
            tool_handlers={"lookup": _lookup},
        )
        if method == "invoke":
            await instance.invoke("q", CONTEXT)
        else:
            async for _ in instance.stream("q", CONTEXT):
                pass

        assert _helpers(installed) == [
            f"client.config.{method}",
            "client.createHandler",
            "client.buildJudgeTasks",
        ]
        tool_calls = [
            call
            for call in installed.track.call_args_list
            if call.args[0] == "$ld:ai:tool_call"
        ]
        assert len(tool_calls) == 1
        assert tool_calls[0].args[2]["toolKey"] == "lookup"
        for field in _USAGE_FIELDS:
            assert field not in tool_calls[0].args[2]
    finally:
        await shutdown()


def _graph_variation(key: str, _ctx: Any, _default: Any) -> Any:
    if key == "graph-key":
        return {"root": "root-node", "edges": {"root-node": [{"key": "leaf-node"}]}}
    return {**_ENABLED_CONFIG, "provider": {"name": "TestProvider"}}


@pytest.mark.parametrize("method", ["invoke", "stream"])
async def test_graph_events_gain_no_usage_fields(method: str) -> None:
    installed = await _install()
    installed.variation = AsyncMock(side_effect=_graph_variation)
    try:

        async def _handler(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {"output": "ok", "usage": {"input_tokens": 1, "output_tokens": 1}}

        instance = graph(
            "graph-key",
            handlers=[
                ProviderHandler(fn=_handler, provides_for=("TestProvider", "messages"))
            ],
        )
        if method == "invoke":
            await instance.invoke("q", CONTEXT)
        else:
            async for _ in instance.stream("q", CONTEXT):
                pass

        assert _helpers(installed) == [f"client.graph.{method}"]
        graph_calls = [
            call
            for call in installed.track.call_args_list
            if str(call.args[0]).startswith("$ld:ai:graph:")
        ]
        assert "$ld:ai:graph:invocation_success" in [c.args[0] for c in graph_calls]
        for call in graph_calls:
            for field in _USAGE_FIELDS:
                assert field not in call.args[2]
    finally:
        await shutdown()


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
            for field in _USAGE_FIELDS:
                assert field not in call.args[2]
    finally:
        await shutdown()


def _package_payload(helper: str) -> dict[str, Any]:
    return {
        "aiSdkName": "launchdarkly-ai-server",
        "aiSdkVersion": __version__,
        "aiSdkLanguage": "python",
        "helper": helper,
        "helperPackageName": "launchdarkly-ai-example",
        "helperPackageVersion": "9.8.7",
    }


async def test_package_helper_names_its_package_and_the_core(
    client: MagicMock,
) -> None:
    sdk_usage_module.report_usage("example.helper", "launchdarkly-ai-example", "9.8.7")
    calls = _usage(client, "example.helper")
    assert [call.args[2] for call in calls] == [_package_payload("example.helper")]

    # The package is not part of the dedupe key.
    sdk_usage_module.report_usage("example.helper", "launchdarkly-ai-other", "1.0.0")
    assert len(_usage(client, "example.helper")) == 1


async def test_held_package_helper_flushes_with_its_package() -> None:
    await shutdown()
    lifecycle_module._reset_for_testing()
    sdk_usage_module.report_usage("example.helper", "launchdarkly-ai-example", "9.8.7")
    await build_judge_tasks(**_judge_kwargs())
    client = _client()
    with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
        await init_client(client=client)
    assert [call.args[2] for call in _usage(client, "example.helper")] == [
        _package_payload("example.helper")
    ]
    _expect_usage(client, HELPER)
    await shutdown()


async def test_package_name_without_version_reports_the_core(
    client: MagicMock,
) -> None:
    sdk_usage_module.report_usage("example.helper", "launchdarkly-ai-example")
    calls = _usage(client, "example.helper")
    assert len(calls) == 1
    assert calls[0].args[2]["helperPackageName"] == "launchdarkly-ai-server"
    assert calls[0].args[2]["helperPackageVersion"] == __version__
