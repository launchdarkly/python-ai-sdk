"""
Tests for §3.14 run_judges.
Reference: TESTING.md §3.14
"""

import json
import logging
from dataclasses import asdict
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import JudgeResult, ProviderHandler, run_judges
from launchdarkly_ai_server.judges import build_judge_tasks, run_judge

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

    async def test_exact_provider_beats_an_earlier_wildcard(
        self, mock_ld_client: MagicMock
    ) -> None:
        seen: list[str] = []

        def handler_for(name: str, provider: str) -> ProviderHandler:
            async def fn(
                config, user_input, tool_handlers, variables, history=None
            ) -> dict:  # type: ignore[override]
                seen.append(name)
                return {
                    "output": '{"score": 0.5, "reasoning": "ok"}',
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }

            return ProviderHandler(fn=fn, provides_for=(provider, "messages"))  # type: ignore[arg-type]

        wildcard = handler_for("wildcard", "*")
        typesafe = handler_for("typesafe", "TypeSafe")
        mock_ld_client.variation = AsyncMock(
            return_value={
                "model": {"name": "jev"},
                "provider": {"name": "TypeSafe"},
                "instructions": "judge",
                "_ldMeta": {
                    "enabled": True,
                    "variationKey": "j1",
                    "version": 1,
                    "mode": "judge",
                },
            }
        )
        config = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "OpenAI"},
            "instructions": "hi",
            "judgeConfiguration": {"judges": [{"key": "jev", "samplingRate": 1.0}]},
        }
        import random

        with patch.object(random, "random", return_value=0.0):
            await run_judges(
                config=config,
                user_context=CONTEXT,
                handler=wildcard,
                handlers=[wildcard, typesafe],
                user_input="question",
                llm_response="answer",
                base_track_data={},
            )
        assert seen == ["typesafe"]

    async def test_scoped_wildcard_skips_a_judge_outside_its_list(
        self, mock_ld_client: MagicMock
    ) -> None:
        called = False

        async def recording_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            nonlocal called
            called = True
            return {"output": '{"score": 0.8, "reasoning": "ok"}', "usage": {}}

        scoped = ProviderHandler(
            fn=recording_fn,
            provides_for=("*", "agent"),  # type: ignore[arg-type]
            providers=["Bedrock"],
        )
        parent = ProviderHandler(
            fn=recording_fn,
            provides_for=("OpenAI", "messages"),  # type: ignore[arg-type]
        )
        mock_ld_client.variation = AsyncMock(
            return_value={
                "model": {"name": "claude-3-5-sonnet"},
                "provider": {"name": "Anthropic"},
                "messages": [{"role": "user", "content": "Evaluate this."}],
                "_ldMeta": {
                    "enabled": True,
                    "variationKey": "j1",
                    "version": 1,
                    "mode": "judge",
                },
            }
        )
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
                handler=parent,
                handlers=[scoped],
                user_input="q",
                llm_response="response",
                base_track_data={"runId": "x"},
            )
        assert called is False

    @pytest.mark.parametrize(
        ("judge_provider", "runs"), [("Bedrock", True), ("Anthropic", False)]
    )
    async def test_scoped_parent_runs_only_a_listed_judge_provider(
        self, mock_ld_client: MagicMock, judge_provider: str, runs: bool
    ) -> None:
        called = False

        async def recording_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            nonlocal called
            called = True
            return {"output": '{"score": 0.8, "reasoning": "ok"}', "usage": {}}

        parent = ProviderHandler(
            fn=recording_fn,
            provides_for=("*", "messages"),  # type: ignore[arg-type]
            providers=["Bedrock"],
        )
        mock_ld_client.variation = AsyncMock(
            return_value={
                "model": {"name": "m"},
                "provider": {"name": judge_provider},
                "messages": [{"role": "user", "content": "Evaluate this."}],
                "_ldMeta": {
                    "enabled": True,
                    "variationKey": "j1",
                    "version": 1,
                    "mode": "judge",
                },
            }
        )
        config = {
            "model": {"name": "m"},
            "provider": {"name": "Bedrock"},
            "instructions": "hi",
            "judgeConfiguration": {"judges": [{"key": "judge-1", "samplingRate": 1.0}]},
        }
        import random

        with patch.object(random, "random", return_value=0.0):
            await run_judges(
                config=config,
                user_context=CONTEXT,
                handler=parent,
                handlers=[],
                user_input="q",
                llm_response="response",
                base_track_data={"runId": "x"},
            )
        assert called is runs

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

    async def test_completion_handler_matches_messages_judge_over_agent(
        self, mock_ld_client: MagicMock
    ) -> None:
        """completion ≡ messages: registry replace must not lose judge routing.

        After registering a completion handler over a messages one, online
        judges still need an exact mode match. Comparing raw provides_for
        would miss completion and fall through to agent (with collapse).
        """
        called_handlers: list[str] = []

        async def completion_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            called_handlers.append("completion")
            return {
                "output": '{"score": 0.9, "reasoning": "ok"}',
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        async def agent_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            called_handlers.append("agent")
            return {
                "output": '{"score": 0.1, "reasoning": "fallback"}',
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        completion_handler = ProviderHandler(
            fn=completion_fn, provides_for=("TestProvider", "completion")
        )  # type: ignore[arg-type]
        agent_handler = ProviderHandler(
            fn=agent_fn, provides_for=("TestProvider", "agent")
        )  # type: ignore[arg-type]

        judge_variation = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "messages": [{"role": "system", "content": "Judge."}],
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
                handler=agent_handler,
                handlers=[completion_handler, agent_handler],
                user_input="q",
                llm_response="response",
                base_track_data={"runId": "x"},
            )

        assert called_handlers == ["completion"]

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


class TestRunJudgeModeAliases:
    """run_judge must treat completion and messages as one mode."""

    @pytest.mark.asyncio
    async def test_completion_handler_matches_messages_task(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
                "response": '{"score": 0.8, "reasoning": "aliased"}',
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "track_data": {"runId": "run-1"},
            }

        monkeypatch.setattr(
            tracking_module, "execute_and_track", fake_execute_and_track
        )

        async def completion_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            raise AssertionError("execute_and_track is stubbed")

        handler = ProviderHandler(
            fn=completion_fn, provides_for=("TestProvider", "completion")
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
        assert result.score == 0.8
        assert result.response == "aliased"

    @pytest.mark.asyncio
    async def test_returns_none_when_only_unrelated_mode(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from launchdarkly_ai_server import JudgeTask, run_judge

        async def agent_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            raise AssertionError("should not run — messages task with no match")

        # Agent-only list still falls back for messages tasks; use a different
        # provider so there is truly no compatible handler.
        handler = ProviderHandler(fn=agent_fn, provides_for=("OtherProvider", "agent"))  # type: ignore[arg-type]

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

        with caplog.at_level(logging.WARNING, logger="launchdarkly_ai_server.judges"):
            assert await run_judge(task, [handler]) is None
        # The same warning run_judges logs when it skips a judge.
        assert (
            "Judge 'judge-key' skipped: no handler covers provider 'TestProvider'"
            in caplog.messages
        )


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
    """§3.13 run_judge result track_data carries the evaluated (parent) config identity."""

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

    async def test_matches_inline_track_data(
        self, monkeypatch: pytest.MonkeyPatch, mock_ld_client: MagicMock
    ) -> None:
        # The judge's own track_data differs on every identity key; none of it
        # may leak into the result, which must equal what run_judges tracks.
        import random

        judge_track_data = {
            "runId": "judge-run",
            "configKey": "judge-key",
            "variationKey": "j1",
            "version": 3,
            "modelName": "claude",
            "providerName": "TestProvider",
            "modelKey": "judge-model",
            "modelVersion": 2,
        }
        result = await self._run(monkeypatch, judge_track_data)
        assert result is not None
        assert result.track_data == {**self.PARENT, "judgeConfigKey": "judge-key"}

        mock_ld_client.variation = AsyncMock(
            return_value={
                "model": {"name": "claude"},
                "provider": {"name": "TestProvider"},
                "instructions": "judge",
                "evaluationMetricKey": "judge-metric",
                "_ldMeta": {"enabled": True, "variationKey": "j1", "version": 3},
            }
        )
        with patch.object(random, "random", return_value=0.0):
            await run_judges(
                config={
                    "model": {"name": "gpt-4o"},
                    "provider": {"name": "TestProvider"},
                    "instructions": "hi",
                    "judgeConfiguration": {
                        "judges": [{"key": "judge-key", "samplingRate": 1.0}]
                    },
                },
                user_context=CONTEXT,
                handler=_make_handler(),
                user_input="q",
                llm_response="response",
                base_track_data=dict(self.PARENT),
            )
        inline_track_data = mock_ld_client.track.call_args.args[2]
        assert result.track_data == inline_track_data


class TestTypesafeJudgeResults:
    async def test_expands_labels_and_tracks_each_event_key(
        self, mock_ld_client: MagicMock
    ) -> None:
        import json
        import random

        payload = json.dumps(
            {
                "kind": "typesafe",
                "results": [
                    {
                        "key": "migration_intent",
                        "eventKey": "$ld:ai:judge:jev:migration_intent",
                        "score": 0.25,
                        "reason": "Unclear",
                    },
                    {
                        "key": "migration",
                        "eventKey": "$ld:ai:judge:jev:migration",
                        "score": 0.9,
                        "reason": "Yes",
                    },
                ],
            }
        )

        seen_variables: list[dict[str, Any]] = []

        async def fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            seen_variables.append(variables)
            return {
                "output": payload,
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }

        handler = ProviderHandler(fn=fn, provides_for=("TypeSafe", "messages"))  # type: ignore[arg-type]
        mock_ld_client.variation = AsyncMock(
            return_value={
                "model": {"name": "jev"},
                "provider": {"name": "TypeSafe"},
                "instructions": "judge",
                "evaluationMetricKey": "judge-metric",
                "_ldMeta": {
                    "enabled": True,
                    "variationKey": "j1",
                    "version": 1,
                    "mode": "judge",
                },
            }
        )
        with patch.object(random, "random", return_value=0.0):
            result = await run_judges(
                config={
                    "model": {"name": "gpt-4"},
                    "provider": {"name": "OpenAI"},
                    "instructions": "hi",
                    "judgeConfiguration": {
                        "judges": [{"key": "jev-judge", "samplingRate": 1.0}]
                    },
                },
                user_context=CONTEXT,
                handler=handler,
                user_input="question",
                llm_response="answer",
                base_track_data={"runId": "x"},
            )

        assert seen_variables[0]["input"] == "question"
        assert seen_variables[0]["response_to_evaluate"] == "answer"
        assert set(result) == {
            "jev-judge.migration_intent",
            "jev-judge.migration",
        }
        assert result["jev-judge.migration_intent"].score == 0.25
        assert result["jev-judge.migration"].score == 0.9
        assert result["jev-judge.migration_intent"].response == "Unclear"
        assert result["jev-judge.migration"].response == "Yes"
        assert result["jev-judge.migration_intent"].usage.input == 100
        assert result["jev-judge.migration"].usage.input == 100
        assert result["jev-judge.migration_intent"].usage.output == 20
        metric_calls = [
            call
            for call in mock_ld_client.track.call_args_list
            if str(call.args[0]).startswith("$ld:ai:judge:jev:")
        ]
        token_calls = [
            call
            for call in mock_ld_client.track.call_args_list
            if call.args[0] == "$ld:ai:tokens:total"
        ]
        assert [call.args[0] for call in metric_calls] == [
            "$ld:ai:judge:jev:migration_intent",
            "$ld:ai:judge:jev:migration",
        ]
        assert metric_calls[0].args[2]["judgeConfigKey"] == "jev-judge.migration_intent"
        assert metric_calls[0].args[3] == 0.25
        assert metric_calls[1].args[3] == 0.9
        assert len(token_calls) == 1

    async def test_run_judge_keeps_the_first_label_event_key(
        self, mock_ld_client: MagicMock
    ) -> None:
        import json

        from launchdarkly_ai_server import JudgeTask, run_judge

        payload = json.dumps(
            {
                "kind": "typesafe",
                "results": [
                    {
                        "key": "migration_intent",
                        "eventKey": "$ld:ai:judge:jev:migration_intent",
                        "score": 0.25,
                        "reason": "Unclear",
                    },
                    {
                        "key": "migration",
                        "eventKey": "$ld:ai:judge:jev:migration",
                        "score": 0.9,
                        "reason": "Yes",
                    },
                ],
            }
        )

        seen_variables: list[dict[str, Any]] = []

        async def fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            seen_variables.append(variables)
            return {
                "output": payload,
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }

        handler = ProviderHandler(fn=fn, provides_for=("TypeSafe", "messages"))  # type: ignore[arg-type]
        result = await run_judge(
            JudgeTask(
                config_key="jev-judge",
                judge_config={
                    "model": {"name": "jev"},
                    "provider": {"name": "TypeSafe"},
                    "instructions": "judge",
                },
                judge_meta={
                    "enabled": True,
                    "variationKey": "j1",
                    "version": 1,
                    "mode": "judge",
                },
                actual_output="answer",
                user_input="question",
                user_context=CONTEXT,
                judge_provider="TypeSafe",
                judge_mode="messages",
                collapse_messages=False,
                parent_track_data={"runId": "x"},
            ),
            [handler],
        )
        assert result is not None
        assert seen_variables[0]["input"] == "question"
        assert result.event_key == "$ld:ai:judge:jev:migration_intent"
        assert result.score == 0.25
        assert result.response == "Unclear"
        assert result.track_data == {
            "runId": "x",
            "judgeConfigKey": "jev-judge.migration_intent",
        }


def _judge_variation_with_output_format() -> dict[str, Any]:
    return {
        "model": {"name": "gpt-4"},
        "provider": {"name": "TestProvider"},
        "instructions": "judge",
        "outputFormat": {
            "type": "json_schema",
            "properties": {"message": "string"},
            "required": ["message"],
        },
        "_ldMeta": {
            "enabled": True,
            "variationKey": "j1",
            "version": 1,
            "mode": "messages",
        },
    }


class TestJudgeOutputFormatStripped:
    """A judge's own ``outputFormat`` must never reach a handler or a JudgeTask.

    The judge verdict contract (``{score, reasoning}``) is owned by this SDK. A schema
    on the judge config itself would hard-constrain the provider away from that shape
    and the judge would never produce a score.
    """

    async def test_handler_never_sees_output_format(
        self, mock_ld_client: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        received_configs: list[Any] = []

        async def recording_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            received_configs.append(config)
            return {
                "output": '{"score": 0.8, "reasoning": "ok"}',
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        handler = ProviderHandler(
            fn=recording_fn, provides_for=("TestProvider", "messages")
        )  # type: ignore[arg-type]

        judge_variation = _judge_variation_with_output_format()
        mock_ld_client.variation = AsyncMock(return_value=judge_variation)

        config = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "hi",
            "judgeConfiguration": {"judges": [{"key": "judge-1", "samplingRate": 1.0}]},
        }

        import random

        with caplog.at_level("WARNING", logger="launchdarkly_ai_server.judge_scoring"):
            with patch.object(random, "random", return_value=0.0):
                result = await run_judges(
                    config=config,
                    user_context=CONTEXT,
                    handler=handler,
                    user_input="q",
                    llm_response="response",
                    base_track_data={"runId": "x"},
                )

        assert len(received_configs) == 1
        effective = received_configs[0]
        assert "outputFormat" not in effective
        assert effective["model"] == {"name": "gpt-4"}
        assert effective["provider"] == {"name": "TestProvider"}
        assert effective["instructions"] == "judge"

        # A valid verdict still parses.
        assert result["judge-1"].score == 0.8
        assert result["judge-1"].response == "ok"

        # This is what was broken: before the fix, a strict provider schema on the judge
        # config made a valid {score, reasoning} verdict impossible. It must be present now.
        assert "judge-1" in result

        # The reason is stated once, naming the judge key.
        warnings = [
            r
            for r in caplog.records
            if r.levelname == "WARNING" and "judge-1" in r.message
        ]
        assert len(warnings) == 1
        assert "outputFormat" in warnings[0].message

    async def test_input_config_not_mutated(self, mock_ld_client: MagicMock) -> None:
        judge_variation = _judge_variation_with_output_format()
        original_output_format = judge_variation["outputFormat"]
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
                handler=_make_handler(),
                user_input="q",
                llm_response="response",
                base_track_data={"runId": "x"},
            )

        # The caller's original variation dict must be untouched.
        assert judge_variation["outputFormat"] is original_output_format

    async def test_no_output_format_means_no_change_and_no_log(
        self, mock_ld_client: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
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

        with caplog.at_level("WARNING", logger="launchdarkly_ai_server.judge_scoring"):
            with patch.object(random, "random", return_value=0.0):
                result = await run_judges(
                    config=config,
                    user_context=CONTEXT,
                    handler=_make_handler(),
                    user_input="q",
                    llm_response="response",
                    base_track_data={"runId": "x"},
                )

        assert result["judge-1"].score == 0.9
        assert not any("outputFormat" in r.message for r in caplog.records)

    async def test_collapsed_messages_still_apply(
        self, mock_ld_client: MagicMock
    ) -> None:
        received_configs: list[Any] = []

        async def recording_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            received_configs.append(config)
            return {
                "output": '{"score": 0.7, "reasoning": "ok"}',
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        agent_handler = ProviderHandler(
            fn=recording_fn, provides_for=("Anthropic", "agent")
        )  # type: ignore[arg-type]

        judge_variation = {
            "model": {"name": "claude-3-5-sonnet"},
            "provider": {"name": "Anthropic"},
            "messages": [{"role": "user", "content": "Judge this response."}],
            "outputFormat": {"type": "json_schema", "properties": {"x": "string"}},
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
                handler=agent_handler,
                handlers=[agent_handler],
                user_input="q",
                llm_response="response",
                base_track_data={"runId": "x"},
            )

        assert len(received_configs) == 1
        effective = received_configs[0]
        # Collapsed to instructions...
        assert effective.get("instructions") == "Judge this response."
        assert effective.get("messages") == []
        # ...and stripped.
        assert "outputFormat" not in effective

    async def test_judge_task_carries_no_output_format(
        self, mock_ld_client: MagicMock
    ) -> None:
        judge_variation = _judge_variation_with_output_format()
        mock_ld_client.variation = AsyncMock(return_value=judge_variation)

        config = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "hi",
            "judgeConfiguration": {"judges": [{"key": "judge-1", "samplingRate": 1.0}]},
        }

        import random

        with patch.object(random, "random", return_value=0.0):
            tasks = await build_judge_tasks(
                config=config,
                user_context=CONTEXT,
                handler=_make_handler(),
                llm_response="response",
                base_track_data={"runId": "x"},
            )

        assert len(tasks) == 1
        task = tasks[0]
        assert "outputFormat" not in task.judge_config

        # Must still survive a JSON round-trip (serialisable for a background thread).
        roundtripped = json.loads(json.dumps(asdict(task)))
        assert "outputFormat" not in roundtripped["judge_config"]

    async def test_run_judge_strips_defensively(
        self, mock_ld_client: MagicMock
    ) -> None:
        from launchdarkly_ai_server.types import JudgeTask

        received_configs: list[Any] = []

        async def recording_fn(
            config, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            received_configs.append(config)
            return {
                "output": '{"score": 0.6, "reasoning": "ok"}',
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        handler = ProviderHandler(
            fn=recording_fn, provides_for=("TestProvider", "messages")
        )  # type: ignore[arg-type]

        # Simulate an older serialized task that still carries outputFormat.
        task = JudgeTask(
            config_key="judge-1",
            judge_config={
                "model": {"name": "gpt-4"},
                "provider": {"name": "TestProvider"},
                "instructions": "judge",
                "outputFormat": {"type": "json_schema", "properties": {"x": "string"}},
            },
            judge_meta={
                "enabled": True,
                "variationKey": "j1",
                "version": 1,
                "mode": "messages",
            },
            actual_output="response",
            user_context=CONTEXT,
            judge_provider="TestProvider",
            judge_mode="messages",
            collapse_messages=False,
            parent_track_data={"runId": "x"},
        )

        result = await run_judge(task, handlers=[handler])

        assert len(received_configs) == 1
        assert "outputFormat" not in received_configs[0]
        assert result is not None
        assert result.score == 0.6
