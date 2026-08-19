"""
Tests for §3.14 run_judges.
Reference: TESTING.md §3.14
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import ProviderHandler, run_judges

CONTEXT = {"kind": "user", "key": "u1"}


def _make_client() -> MagicMock:
    c = MagicMock()
    c.track = MagicMock()
    c.flush = AsyncMock()
    c.close = AsyncMock()
    c.variation = AsyncMock(return_value=None)
    return c


def _make_handler(response: str = "judge-ok") -> ProviderHandler:
    async def fn(config, user_input, tool_handlers, variables, history=None) -> dict:  # type: ignore[override]
        return {
            "output": '{"score": 0.9, "reasoning": "good"}',
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    return ProviderHandler(fn=fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]


@pytest.fixture
def mock_ld_client() -> MagicMock:
    client = _make_client()
    lifecycle_module._set_client_for_testing(client)
    yield client
    lifecycle_module._reset_for_testing()


class TestRunJudges:
    async def test_returns_empty_dict_when_no_judges(
        self, mock_ld_client: MagicMock
    ) -> None:
        config = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "hi",
        }
        result = await run_judges(
            config=config,
            user_context=CONTEXT,
            handler=_make_handler(),
            user_input="q",
            llm_response="r",
            base_track_data={},
        )
        assert result == {}

    async def test_skips_judges_with_sampling_rate_zero(
        self, mock_ld_client: MagicMock
    ) -> None:
        config = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "hi",
            "judgeConfiguration": {"judges": [{"key": "judge-1", "samplingRate": 0}]},
        }
        result = await run_judges(
            config=config,
            user_context=CONTEXT,
            handler=_make_handler(),
            user_input="q",
            llm_response="r",
            base_track_data={},
        )
        assert result == {}

    async def test_tool_handlers_not_forwarded_to_judge_calls(
        self, mock_ld_client: MagicMock
    ) -> None:
        received_tool_handlers: list[Any] = []

        async def recording_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            received_tool_handlers.append(tool_handlers)
            return {
                "output": '{"score": 0.5, "reasoning": "test"}',
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        h = ProviderHandler(fn=recording_fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]

        judge_variation = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "judge",
            "_ldMeta": {
                "enabled": True,
                "variationKey": "j1",
                "version": 1,
                "mode": "messages",
            },
        }
        mock_ld_client.variation = AsyncMock(return_value=judge_variation)

        config = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "hi",
            "judgeConfiguration": {"judges": [{"key": "judge-1", "samplingRate": 1.0}]},
        }

        import random

        with patch.object(random, "random", return_value=0.0):
            await run_judges(
                config=config,
                user_context=CONTEXT,
                handler=h,
                user_input="q",
                llm_response="response",
                base_track_data={"runId": "x"},
                tool_handlers={"my_tool": lambda: None},
            )

        # judge should receive None as tool_handlers, not the parent's tools
        assert received_tool_handlers[-1] is None or received_tool_handlers[-1] == {}

    async def test_wildcard_agent_handler_used_when_no_messages_handler_and_messages_collapsed(
        self, mock_ld_client: MagicMock
    ) -> None:
        """When only a wildcard agent handler is registered, it should be selected for
        a messages-mode judge config and the messages should be collapsed to instructions."""
        received_configs: list[Any] = []

        async def recording_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            received_configs.append(config)
            return {
                "output": '{"score": 0.8, "reasoning": "ok"}',
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        wildcard_agent_handler = ProviderHandler(
            fn=recording_fn,
            provides_for=("*", "agent"),  # type: ignore[arg-type]
        )

        judge_variation = {
            "model": {"name": "claude-3-5-sonnet"},
            "provider": {"name": "Anthropic"},
            "messages": [
                {"role": "system", "content": "You are a judge."},
                {"role": "user", "content": "Evaluate this."},
            ],
            "_ldMeta": {
                "enabled": True,
                "variationKey": "j1",
                "version": 1,
                "mode": "judge",
            },
        }
        mock_ld_client.variation = AsyncMock(return_value=judge_variation)

        config = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "OpenAI"},
            "instructions": "hi",
            "judgeConfiguration": {"judges": [{"key": "judge-1", "samplingRate": 1.0}]},
        }

        import random

        with patch.object(random, "random", return_value=0.0):
            await run_judges(
                config=config,
                user_context=CONTEXT,
                handler=wildcard_agent_handler,
                handlers=[wildcard_agent_handler],
                user_input="q",
                llm_response="response",
                base_track_data={"runId": "x"},
            )

        assert len(received_configs) == 1
        effective = received_configs[0]
        # Messages should be collapsed into instructions
        assert effective.get("instructions") is not None
        assert effective.get("messages") == []

    async def test_exact_agent_handler_fallback_collapses_messages(
        self, mock_ld_client: MagicMock
    ) -> None:
        """When an agent handler for the same provider is registered but no messages
        handler exists, it should be used with messages collapsed to instructions."""
        received_configs: list[Any] = []

        async def recording_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            received_configs.append(config)
            return {
                "output": '{"score": 0.7, "reasoning": "ok"}',
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        claude_agent_handler = ProviderHandler(
            fn=recording_fn,
            provides_for=("Anthropic", "agent"),  # type: ignore[arg-type]
        )

        judge_variation = {
            "model": {"name": "claude-3-5-sonnet"},
            "provider": {"name": "Anthropic"},
            "messages": [{"role": "user", "content": "Judge this response."}],
            "_ldMeta": {
                "enabled": True,
                "variationKey": "j1",
                "version": 1,
                "mode": "judge",
            },
        }
        mock_ld_client.variation = AsyncMock(return_value=judge_variation)

        config = {
            "model": {"name": "claude-3-5-sonnet"},
            "provider": {"name": "Anthropic"},
            "instructions": "hi",
            "judgeConfiguration": {"judges": [{"key": "judge-1", "samplingRate": 1.0}]},
        }

        import random

        with patch.object(random, "random", return_value=0.0):
            await run_judges(
                config=config,
                user_context=CONTEXT,
                handler=claude_agent_handler,
                handlers=[claude_agent_handler],
                user_input="q",
                llm_response="response",
                base_track_data={"runId": "x"},
            )

        assert len(received_configs) == 1
        effective = received_configs[0]
        assert effective.get("instructions") == "Judge this response."
        assert effective.get("messages") == []

    async def test_exact_messages_handler_preferred_over_agent_fallback(
        self, mock_ld_client: MagicMock
    ) -> None:
        """When both messages and agent handlers exist, the messages handler wins."""
        called_handlers: list[str] = []

        async def messages_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            called_handlers.append("messages")
            return {
                "output": '{"score": 0.9, "reasoning": "precise"}',
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        async def agent_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            called_handlers.append("agent")
            return {
                "output": '{"score": 0.5, "reasoning": "fallback"}',
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        messages_handler = ProviderHandler(
            fn=messages_fn, provides_for=("TestProvider", "messages")
        )  # type: ignore[arg-type]
        agent_handler = ProviderHandler(
            fn=agent_fn, provides_for=("TestProvider", "agent")
        )  # type: ignore[arg-type]

        judge_variation = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "judge",
            "_ldMeta": {
                "enabled": True,
                "variationKey": "j1",
                "version": 1,
                "mode": "messages",
            },
        }
        mock_ld_client.variation = AsyncMock(return_value=judge_variation)

        config = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "hi",
            "judgeConfiguration": {"judges": [{"key": "judge-1", "samplingRate": 1.0}]},
        }

        import random

        with patch.object(random, "random", return_value=0.0):
            await run_judges(
                config=config,
                user_context=CONTEXT,
                handler=messages_handler,
                handlers=[agent_handler, messages_handler],
                user_input="q",
                llm_response="response",
                base_track_data={"runId": "x"},
            )

        assert called_handlers == ["messages"]

    async def test_returns_empty_dict_when_judges_array_is_empty(
        self, mock_ld_client: MagicMock
    ) -> None:
        config = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "hi",
            "judgeConfiguration": {"judges": []},
        }
        result = await run_judges(
            config=config,
            user_context=CONTEXT,
            handler=_make_handler(),
            user_input="q",
            llm_response="r",
            base_track_data={},
        )
        assert result == {}


class TestScoreGuard:
    """`float(score)` used to sit ahead of the evaluation-metric track, so a junk score killed it."""

    def test_rejects_non_numeric_scores_without_raising(self) -> None:
        from launchdarkly_ai_server.judges import _numeric_score

        for junk in ("0.9 (high)", "85%", None, {"v": 1}, [], True, False):
            assert _numeric_score(junk) is None

    def test_accepts_finite_numbers(self) -> None:
        from math import inf, nan

        from launchdarkly_ai_server.judges import _numeric_score

        assert _numeric_score(0.9) == 0.9
        assert _numeric_score(1) == 1.0
        assert _numeric_score(0) == 0.0
        assert _numeric_score(inf) is None
        assert _numeric_score(nan) is None
