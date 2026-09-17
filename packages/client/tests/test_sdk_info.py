"""Tests for TESTING.md §3.9 AI SDK package information events."""

from unittest.mock import MagicMock

import pytest

from launchdarkly_ai_server.sdk_info import (
    SDK_INFO_CONTEXT,
    SDK_INFO_EVENT,
    flush_ai_sdk_info,
    register_ai_sdk_package,
    reset_ai_sdk_info,
)


@pytest.fixture(autouse=True)
def reset_sdk_info_state() -> None:
    reset_ai_sdk_info(clear_known=True)
    yield
    reset_ai_sdk_info(clear_known=True)


def _client() -> MagicMock:
    client = MagicMock()
    client.track = MagicMock()
    return client


def test_emits_one_event_per_registered_package() -> None:
    client = _client()
    register_ai_sdk_package("launchdarkly-ai-server", "0.1.3")
    register_ai_sdk_package("launchdarkly-ai-openai-agents", "0.1.4")

    flush_ai_sdk_info(client)

    assert client.track.call_count == 2
    client.track.assert_any_call(
        "$ld:ai:sdk:info",
        SDK_INFO_CONTEXT,
        {
            "aiSdkName": "launchdarkly-ai-server",
            "aiSdkVersion": "0.1.3",
            "aiSdkLanguage": "python",
        },
        1,
    )
    client.track.assert_any_call(
        "$ld:ai:sdk:info",
        SDK_INFO_CONTEXT,
        {
            "aiSdkName": "launchdarkly-ai-openai-agents",
            "aiSdkVersion": "0.1.4",
            "aiSdkLanguage": "python",
        },
        1,
    )


def test_uses_anonymous_ld_ai_context() -> None:
    assert SDK_INFO_EVENT == "$ld:ai:sdk:info"
    assert SDK_INFO_CONTEXT == {
        "kind": "ld_ai",
        "key": "ld-internal-tracking",
        "anonymous": True,
    }


def test_duplicate_registration_and_flush_emit_once() -> None:
    client = _client()
    register_ai_sdk_package("launchdarkly-ai-server", "0.1.3")
    register_ai_sdk_package("launchdarkly-ai-server", "0.1.3")

    flush_ai_sdk_info(client)
    flush_ai_sdk_info(client)

    client.track.assert_called_once()


def test_two_versions_of_one_package_both_emit() -> None:
    client = _client()
    register_ai_sdk_package("launchdarkly-ai-server", "0.1.2")
    register_ai_sdk_package("launchdarkly-ai-server", "0.1.3")

    flush_ai_sdk_info(client)

    assert [call.args[2]["aiSdkVersion"] for call in client.track.call_args_list] == [
        "0.1.2",
        "0.1.3",
    ]


def test_late_registration_emits_on_next_flush() -> None:
    client = _client()
    register_ai_sdk_package("launchdarkly-ai-server", "0.1.3")
    flush_ai_sdk_info(client)
    client.track.reset_mock()

    register_ai_sdk_package("launchdarkly-ai-claude-agents", "0.1.4")
    flush_ai_sdk_info(client)

    client.track.assert_called_once()
    assert client.track.call_args.args[2]["aiSdkName"] == (
        "launchdarkly-ai-claude-agents"
    )


def test_reset_reemits_known_packages() -> None:
    client = _client()
    register_ai_sdk_package("launchdarkly-ai-server", "0.1.3")
    flush_ai_sdk_info(client)
    client.track.reset_mock()

    reset_ai_sdk_info()
    flush_ai_sdk_info(client)

    client.track.assert_called_once()


def test_track_failure_is_non_fatal_and_not_retried() -> None:
    client = _client()
    client.track.side_effect = RuntimeError("client closed")
    register_ai_sdk_package("launchdarkly-ai-server", "0.1.3")

    flush_ai_sdk_info(client)
    client.track.reset_mock()
    client.track.side_effect = None
    flush_ai_sdk_info(client)

    client.track.assert_not_called()
