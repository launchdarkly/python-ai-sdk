"""
Tests for §3.14 run_judges.
Reference: TESTING.md §3.14
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_server.judges as judges_module
import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import JudgeTask, ProviderHandler, run_judge, run_judges
from launchdarkly_ai_server.utils import JUDGE_REASONING_MAX_LENGTH

CONTEXT = {"kind": "user", "key": "u1"}


class FakeSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}
        self.ended = 0

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def end(self) -> None:
        self.ended += 1


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


class TestJudgeReasoning:
    """Reasoning leaves the process on the metric event and on the judge span.

    Reference: TELEMETRY-CONTRACT.md §4a
    """

    @staticmethod
    def _judge_variation() -> dict[str, Any]:
        return {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "judge",
            "evaluationMetricKey": "quality",
            "_ldMeta": {
                "enabled": True,
                "variationKey": "j1",
                "version": 1,
                "mode": "messages",
            },
        }

    @staticmethod
    def _parent_config() -> dict[str, Any]:
        return {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "hi",
            "judgeConfiguration": {"judges": [{"key": "judge-1", "samplingRate": 1.0}]},
        }

    async def _run(
        self, client: MagicMock, reasoning: str = "clear and correct"
    ) -> None:
        async def fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            return {
                "output": json.dumps({"score": 0.9, "reasoning": reasoning}),
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        handler = ProviderHandler(fn=fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]
        client.variation = AsyncMock(return_value=self._judge_variation())

        import random

        with patch.object(random, "random", return_value=0.0):
            await run_judges(
                config=self._parent_config(),
                user_context=CONTEXT,
                handler=handler,
                user_input="q",
                llm_response="response",
                base_track_data={"runId": "run-1", "configKey": "parent"},
            )

    async def test_metric_event_carries_reasoning(
        self, mock_ld_client: MagicMock
    ) -> None:
        await self._run(mock_ld_client)

        metric_key, _context, track_data, score = mock_ld_client.track.call_args[0]
        assert metric_key == "quality"
        assert score == 0.9
        assert track_data["judgeReasoning"] == "clear and correct"
        assert track_data["judgeConfigKey"] == "judge-1"
        assert track_data["runId"] == "run-1"

    async def test_reasoning_can_be_suppressed_without_losing_the_score(
        self, mock_ld_client: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LD_CAPTURE_JUDGE_REASONING", "false")
        await self._run(mock_ld_client)

        _metric_key, _context, track_data, score = mock_ld_client.track.call_args[0]
        assert "judgeReasoning" not in track_data
        assert score == 0.9

    async def test_empty_reasoning_is_omitted(self, mock_ld_client: MagicMock) -> None:
        await self._run(mock_ld_client, reasoning="")

        _metric_key, _context, track_data, _score = mock_ld_client.track.call_args[0]
        assert "judgeReasoning" not in track_data

    async def test_long_reasoning_is_truncated(self, mock_ld_client: MagicMock) -> None:
        await self._run(
            mock_ld_client, reasoning="a" * (JUDGE_REASONING_MAX_LENGTH + 50)
        )

        _metric_key, _context, track_data, _score = mock_ld_client.track.call_args[0]
        assert len(track_data["judgeReasoning"]) == JUDGE_REASONING_MAX_LENGTH + 1
        assert track_data["judgeReasoning"].endswith("…")

    async def test_judge_span_carries_reasoning_and_score(
        self, mock_ld_client: MagicMock
    ) -> None:
        spans: list[FakeSpan] = []

        def start_span(_span_name: str) -> FakeSpan:
            spans.append(FakeSpan())
            return spans[-1]

        tracer = MagicMock()
        tracer.start_span = start_span

        with patch.object(judges_module.trace, "get_tracer", return_value=tracer):
            await self._run(mock_ld_client)

        assert len(spans) == 1
        span = spans[0]
        assert span.ended == 1
        assert span.attributes == {
            "launchdarkly.operation.type": "judge",
            "launchdarkly.judge.key": "judge-1",
            "launchdarkly.judge.score": 0.9,
            "launchdarkly.judge.reasoning": "clear and correct",
            "launchdarkly.judge.metric.key": "quality",
            "launchdarkly.run.id": "run-1",
        }


class TestRunJudgeTrackData:
    """The deferred path must carry reasoning too, so a background worker's track call
    is not a downgrade from the inline one.
    """

    @staticmethod
    def _task() -> JudgeTask:
        return JudgeTask(
            config_key="judge-1",
            judge_config={
                "model": {"name": "gpt-4"},
                "provider": {"name": "TestProvider"},
                "instructions": "judge",
            },
            judge_meta={"enabled": True, "variationKey": "j1", "version": 1},
            actual_output="response",
            user_context=CONTEXT,
            judge_provider="TestProvider",
            judge_mode="messages",
            collapse_messages=False,
            parent_track_data={"runId": "run-1", "configKey": "parent"},
            evaluation_metric_key="quality",
        )

    async def test_track_data_carries_reasoning(
        self, mock_ld_client: MagicMock
    ) -> None:
        async def fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            return {
                "output": '{"score": 0.4, "reasoning": "missed the question"}',
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        handler = ProviderHandler(fn=fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]

        result = await run_judge(self._task(), [handler])

        assert result is not None
        assert result.response == "missed the question"
        assert result.track_data["judgeReasoning"] == "missed the question"
        assert result.track_data["judgeConfigKey"] == "judge-1"
