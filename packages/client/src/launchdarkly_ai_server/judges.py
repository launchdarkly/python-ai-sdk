from __future__ import annotations

import logging
import random
from collections.abc import Callable
from typing import Any

from .types import (
    AiConfigRep,
    JudgeRunResult,
    JudgeTask,
    LDContext,
    NativeTool,
    ProviderHandler,
    TrackData,
    UsageDict,
)
from .utils import (
    collapse_messages_to_instructions as _collapse_messages_to_instructions,
)
from .utils import (
    normalize_mode,
    parse_json_with_possible_fences,
    to_ld_context,
)


def _provider_matches(handler: ProviderHandler, provider: str | None) -> bool:
    """Returns True when the handler covers the given provider or is a wildcard."""
    return bool(
        handler.provides_for
        and (handler.provides_for[0] == provider or handler.provides_for[0] == "*")
    )


logger = logging.getLogger(__name__)

_FORMATTING_INSTRUCTIONS = "\n".join(
    [
        "Your response MUST be in valid JSON format with the following structure:",
        '{ "score": <number, 0-1>, "reasoning": <string> }',
        "The output must be valid, parseable JSON. Do not include additional tags, comments, "
        "formatting, or newlines.",
        "It should be returned in a format that is immediately parseable by a JSON parsing "
        "function. Do not include ```json tags.",
    ]
)


async def run_judges(
    *,
    config: AiConfigRep,
    user_context: LDContext,
    handler: ProviderHandler,
    handlers: list[ProviderHandler] | None = None,
    user_input: str | None,
    llm_response: str,
    base_track_data: TrackData,
    tool_handlers: dict[str, Callable[..., Any] | NativeTool] | None = None,
    graph_key: str | None = None,
) -> dict[str, Any]:
    """
    Runs any judges configured on ``config['judgeConfiguration']`` against the
    produced output. Each judge is itself a tracked AI call.
    """
    from .lifecycle import extract_variation
    from .tracking import execute_and_track

    judge_results: dict[str, Any] = {}

    judge_config = (
        config.get("judgeConfiguration") or {} if isinstance(config, dict) else {}
    )
    judges = judge_config.get("judges", [])

    has_active_judge = any(j.get("samplingRate", 0) > 0 for j in judges)
    if not judges or not has_active_judge:
        return judge_results

    for judge in judges:
        sampling_rate = judge.get("samplingRate", 0)
        if random.random() >= sampling_rate:
            continue

        judge_key = judge["key"]

        try:
            variation = await extract_variation(judge_key, user_context)
            judge_ai_config: AiConfigRep = variation["config"]
            judge_meta = variation["meta"]

            judge_provider = (
                judge_ai_config.get("provider", {}).get("name")
                if isinstance(judge_ai_config, dict)
                else None
            )
            judge_mode = normalize_mode(
                judge_meta.get("mode") if isinstance(judge_meta, dict) else None
            )

            # Select judge handler. Priority:
            #   1. Exact provider + mode (or wildcard provider + same mode)
            #   2. Agent-mode handler for same provider / wildcard (messages-mode fallback)
            #   3. Parent handler when it covers the same provider or is a wildcard
            # When falling back to an agent-mode handler for a messages-mode judge
            # config, collapse messages into a single instructions block.
            judge_handler: ProviderHandler = handler
            collapse_messages = False
            if handlers:
                exact = next(
                    (
                        h
                        for h in handlers
                        if _provider_matches(h, judge_provider)
                        and h.provides_for
                        and h.provides_for[1] == judge_mode
                    ),
                    None,
                )
                agent_fallback = (
                    next(
                        (
                            h
                            for h in handlers
                            if _provider_matches(h, judge_provider)
                            and h.provides_for
                            and h.provides_for[1] == "agent"
                        ),
                        None,
                    )
                    if not exact and judge_mode == "messages"
                    else None
                )
                if exact:
                    judge_handler = exact
                elif agent_fallback:
                    judge_handler = agent_fallback
                    collapse_messages = True
                elif _provider_matches(handler, judge_provider):
                    judge_handler = handler
                    collapse_messages = (
                        judge_mode == "messages"
                        and handler.provides_for is not None
                        and handler.provides_for[1] == "agent"
                    )

            effective_judge_config = (
                _collapse_messages_to_instructions(judge_ai_config)
                if collapse_messages
                else judge_ai_config
            )

            message_history = "\n\n".join(
                filter(None, [user_input, llm_response, _FORMATTING_INSTRUCTIONS])
            )

            result = await execute_and_track(
                config_key=judge_key,
                config=effective_judge_config,
                meta=judge_meta,
                user_context=user_context,
                handler=judge_handler,
                user_input=llm_response,
                tool_handlers=None,
                graph_key=graph_key,
                variables={
                    "message_history": message_history,
                    "response_to_evaluate": llm_response,
                },
            )

            raw = result["response"]
            judge_response = raw if isinstance(raw, str) else str(raw)

            parsed = parse_json_with_possible_fences(judge_response)
            if not parsed:
                raise ValueError("Invalid JSON from judge")

            score = parsed.get("score")
            reasoning = parsed.get("reasoning", "")
            judge_results[judge_key] = {
                "usage": result["usage"],
                "response": reasoning,
                "score": score,
            }

            evaluation_metric_key = (
                judge_ai_config.get("evaluationMetricKey")
                if isinstance(judge_ai_config, dict)
                else None
            )
            if evaluation_metric_key and score is not None:
                from .lifecycle import get_client

                client = get_client()
                client.track(
                    evaluation_metric_key,
                    to_ld_context(client, user_context),
                    {**base_track_data, "judgeConfigKey": judge_key},
                    score,
                )

        except Exception as exc:
            logger.error("Judge '%s' failed: %s", judge_key, exc)

    return judge_results


async def build_judge_tasks(
    *,
    config: AiConfigRep,
    user_context: LDContext,
    handler: ProviderHandler,
    handlers: list[ProviderHandler] | None = None,
    llm_response: str,
    base_track_data: TrackData,
) -> list[JudgeTask]:
    """
    Resolves all judges configured on ``config['judgeConfiguration']`` into
    serialisable :class:`JudgeTask` objects without executing any AI calls.

    Mirrors the iteration and handler-selection logic of :func:`run_judges` but
    returns tasks instead of running them. Pass each task to a background thread
    that calls ``run_judge(task, handlers)``.

    Sampling is applied here (same as :func:`run_judges`): judges whose
    ``samplingRate`` causes them to be skipped are excluded from the list.
    Returns an empty list when no active judges are configured.
    """
    from .lifecycle import extract_variation

    judge_config_block = (
        config.get("judgeConfiguration") or {} if isinstance(config, dict) else {}
    )
    judges = judge_config_block.get("judges", [])
    has_active_judge = any(j.get("samplingRate", 0) > 0 for j in judges)
    if not judges or not has_active_judge:
        return []

    tasks: list[JudgeTask] = []

    for judge in judges:
        sampling_rate = judge.get("samplingRate", 0)
        if random.random() >= sampling_rate:
            continue

        judge_key = judge["key"]

        try:
            variation = await extract_variation(judge_key, user_context)
            judge_ai_config: AiConfigRep = variation["config"]
            judge_meta = variation["meta"]

            judge_provider = (
                judge_ai_config.get("provider", {}).get("name")
                if isinstance(judge_ai_config, dict)
                else None
            )
            judge_mode = normalize_mode(
                judge_meta.get("mode") if isinstance(judge_meta, dict) else None
            )

            collapse_messages = False
            if handlers:
                exact = next(
                    (
                        h
                        for h in handlers
                        if _provider_matches(h, judge_provider)
                        and h.provides_for
                        and h.provides_for[1] == judge_mode
                    ),
                    None,
                )
                agent_fallback = (
                    next(
                        (
                            h
                            for h in handlers
                            if _provider_matches(h, judge_provider)
                            and h.provides_for
                            and h.provides_for[1] == "agent"
                        ),
                        None,
                    )
                    if not exact and judge_mode == "messages"
                    else None
                )
                if exact:
                    collapse_messages = False
                elif agent_fallback:
                    collapse_messages = True
                elif _provider_matches(handler, judge_provider):
                    collapse_messages = (
                        judge_mode == "messages"
                        and handler.provides_for is not None
                        and handler.provides_for[1] == "agent"
                    )
                else:
                    # No compatible handler — skip, same as run_judges.
                    continue

            evaluation_metric_key = (
                judge_ai_config.get("evaluationMetricKey")
                if isinstance(judge_ai_config, dict)
                else None
            )

            tasks.append(
                JudgeTask(
                    config_key=judge_key,
                    judge_config=judge_ai_config,
                    judge_meta=judge_meta,
                    actual_output=llm_response,
                    user_context=user_context,
                    judge_provider=judge_provider,
                    judge_mode=judge_mode,
                    collapse_messages=collapse_messages,
                    parent_track_data=base_track_data,
                    evaluation_metric_key=evaluation_metric_key,
                )
            )
        except Exception as exc:
            logger.error("Failed to build judge task for '%s': %s", judge_key, exc)

    return tasks


async def run_judge(
    task: JudgeTask,
    handlers: list[ProviderHandler],
) -> JudgeRunResult | None:
    """
    Executes a judge evaluation from a pre-resolved :class:`JudgeTask`.

    Designed to run in a background thread (e.g. via ``threading.Thread`` +
    ``asyncio.run()``): it requires no LaunchDarkly client and no global
    registry — only the explicit ``handlers`` list the caller provides.

    The returned :class:`JudgeRunResult` includes ``track_data`` with
    ``judgeConfigKey`` already merged in, ready to hand to
    ``get_client().track()`` on the main thread.

    Returns ``None`` when no compatible handler is found or the response cannot
    be parsed as ``{"score": ..., "reasoning": ...}``.
    """
    from .tracking import execute_and_track

    def _matches(h: ProviderHandler) -> bool:
        return _provider_matches(h, task.judge_provider)

    exact = next(
        (
            h
            for h in handlers
            if _matches(h) and h.provides_for and h.provides_for[1] == task.judge_mode
        ),
        None,
    )
    agent_fallback = (
        next(
            (
                h
                for h in handlers
                if _matches(h) and h.provides_for and h.provides_for[1] == "agent"
            ),
            None,
        )
        if task.judge_mode == "messages" and not exact
        else None
    )

    judge_handler = exact or agent_fallback
    if judge_handler is None:
        return None

    effective_config = (
        _collapse_messages_to_instructions(task.judge_config)
        if task.collapse_messages
        else task.judge_config
    )

    message_history = "\n\n".join(
        filter(None, [task.actual_output, _FORMATTING_INSTRUCTIONS])
    )

    result = await execute_and_track(
        config_key=task.config_key,
        config=effective_config,
        meta=task.judge_meta,
        user_context=task.user_context,
        handler=judge_handler,
        user_input=task.actual_output,
        tool_handlers=None,
        variables={
            **(task.variables or {}),
            "message_history": message_history,
            "response_to_evaluate": task.actual_output,
        },
    )

    raw = result["response"]
    judge_response = raw if isinstance(raw, str) else str(raw)
    parsed = parse_json_with_possible_fences(judge_response)
    if not parsed:
        return None

    score = parsed.get("score", 0.0)
    reasoning = parsed.get("reasoning", "")
    raw_usage = result["usage"]

    usage = UsageDict(
        input=raw_usage.get("input", 0),
        output=raw_usage.get("output", 0),
        total=raw_usage.get("total", 0),
    )

    merged_track_data: TrackData = {
        **task.parent_track_data,
        **result["track_data"],
        "judgeConfigKey": task.config_key,
    }

    return JudgeRunResult(
        score=score, response=reasoning, usage=usage, track_data=merged_track_data
    )
