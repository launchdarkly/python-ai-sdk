"""
Tests for §3.14 run_judges.
Reference: TESTING.md §3.14
"""

from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import JudgeResult, ProviderHandler, run_judges

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

    async def test_returns_judge_result_objects_with_score_and_reasoning(
        self, mock_ld_client: MagicMock
    ) -> None:
        """Inline results must be ``JudgeResult`` instances.

        The conversation example (and ``ProviderResponse.judge_results``) read
        ``.score`` / ``.response`` as attributes. A plain dict makes those always
        ``None`` even when a judge ran.
        """
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
            result = await run_judges(
                config=config,
                user_context=CONTEXT,
                handler=_make_handler(),
                user_input="q",
                llm_response="r",
                base_track_data={"runId": "x"},
            )

        assert "judge-1" in result
        judge = result["judge-1"]
        assert isinstance(judge, JudgeResult)
        # Attribute access — the pattern the conversation example uses.
        assert getattr(judge, "score", None) == 0.9
        assert getattr(judge, "response", None) == "good"

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
        from launchdarkly_ai_server.judge_scoring import numeric_score

        for junk in ("0.9 (high)", "85%", None, {"v": 1}, [], True, False):
            assert numeric_score(junk) is None

    def test_accepts_finite_numbers(self) -> None:
        from math import inf, nan

        from launchdarkly_ai_server.judge_scoring import numeric_score

        assert numeric_score(0.9) == 0.9
        assert numeric_score(1) == 1.0
        assert numeric_score(0) == 0.0
        assert numeric_score(inf) is None
        assert numeric_score(nan) is None


class TestRunJudgeScoreReporting:
    """A judge that returns no score has not scored the output a zero."""

    @pytest.mark.asyncio
    async def test_null_score_skips_the_metric_instead_of_recording_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from contextlib import asynccontextmanager

        import launchdarkly_ai_server.judges as judges_module
        import launchdarkly_ai_server.tracking as tracking_module
        from launchdarkly_ai_server import JudgeTask, run_judge

        recorded: list[tuple[float, str | None]] = []

        @asynccontextmanager
        async def fake_with_judge_evaluation(name: str) -> Any:
            def record(score: float, explanation: str | None = None) -> None:
                recorded.append((score, explanation))

            yield record

        monkeypatch.setattr(
            judges_module, "with_judge_evaluation", fake_with_judge_evaluation
        )

        async def fake_execute_and_track(**kwargs: Any) -> dict[str, Any]:
            return {
                "response": '{"score": null, "reasoning": "cannot tell"}',
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "track_data": {"runId": "run-1"},
            }

        monkeypatch.setattr(
            tracking_module, "execute_and_track", fake_execute_and_track
        )

        async def judge_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            raise AssertionError("execute_and_track is stubbed")

        handler = ProviderHandler(
            fn=judge_fn, provides_for=("TestProvider", "messages")
        )  # type: ignore[arg-type]

        task = JudgeTask(
            config_key="judge-key",
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
            parent_track_data={"runId": "run-1"},
        )

        result = await run_judge(task, [handler])

        assert result is not None
        # A gen_ai.evaluation of 0 is indistinguishable from a judge that
        # scored the output a hard fail, so no metric is recorded at all.
        assert recorded == []
        assert result.score is None
        assert result.response == "cannot tell"


class TestRunJudgeTrackData:
    """§3.13 run_judge result track_data must not inherit the parent's model stamps."""

    PARENT: ClassVar[dict[str, Any]] = {
        "runId": "parent-run",
        "configKey": "main-flag",
        "variationKey": "v1",
        "version": 1,
        "modelName": "gpt-4o",
        "providerName": "OpenAI",
        "modelKey": "parent-model",
        "modelVersion": 7,
        "graphKey": "g1",
    }

    async def _run(
        self, monkeypatch: pytest.MonkeyPatch, judge_track_data: dict[str, Any]
    ) -> Any:
        from contextlib import asynccontextmanager

        import launchdarkly_ai_server.judges as judges_module
        import launchdarkly_ai_server.tracking as tracking_module
        from launchdarkly_ai_server import JudgeTask, run_judge

        @asynccontextmanager
        async def fake_with_judge_evaluation(name: str) -> Any:
            yield lambda score, explanation=None: None

        monkeypatch.setattr(
            judges_module, "with_judge_evaluation", fake_with_judge_evaluation
        )

        async def fake_execute_and_track(**kwargs: Any) -> dict[str, Any]:
            return {
                "response": '{"score": 0.9, "reasoning": "good"}',
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "track_data": judge_track_data,
            }

        monkeypatch.setattr(
            tracking_module, "execute_and_track", fake_execute_and_track
        )

        async def judge_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            raise AssertionError("execute_and_track is stubbed")

        handler = ProviderHandler(
            fn=judge_fn, provides_for=("TestProvider", "messages")
        )  # type: ignore[arg-type]
        task = JudgeTask(
            config_key="judge-key",
            judge_config={
                "model": {"name": "claude"},
                "provider": {"name": "TestProvider"},
                "instructions": "judge",
            },
            judge_meta={"enabled": True, "variationKey": "j1", "version": 1},
            actual_output="response",
            user_context=CONTEXT,
            judge_provider="TestProvider",
            judge_mode="messages",
            collapse_messages=False,
            parent_track_data=dict(self.PARENT),
        )
        return await run_judge(task, [handler])

    async def test_does_not_inherit_parent_model_stamps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = await self._run(
            monkeypatch,
            {
                "runId": "judge-run",
                "configKey": "judge-key",
                "variationKey": "j1",
                "version": 1,
                "modelName": "claude",
                "providerName": "TestProvider",
            },
        )
        assert result is not None
        assert "modelKey" not in result.track_data
        assert "modelVersion" not in result.track_data
        assert result.track_data["judgeConfigKey"] == "judge-key"
        assert result.track_data["graphKey"] == "g1"

    async def test_keeps_judge_own_model_stamps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = await self._run(
            monkeypatch,
            {
                "runId": "judge-run",
                "configKey": "judge-key",
                "variationKey": "j1",
                "version": 1,
                "modelName": "claude",
                "providerName": "TestProvider",
                "modelKey": "judge-model",
                "modelVersion": 2,
            },
        )
        assert result is not None
        assert result.track_data["modelKey"] == "judge-model"
        assert result.track_data["modelVersion"] == 2
