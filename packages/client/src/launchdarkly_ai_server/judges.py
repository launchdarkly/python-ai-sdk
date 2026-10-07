from __future__ import annotations

import logging
import random
from collections.abc import Callable
from typing import Any

from .conversation import with_judge_evaluation
from .judge_scoring import (
    build_message_history,
    numeric_score,
    parse_judge_response,
    typesafe_judge_entries,
)
from .types import (
    AiConfigRep,
    JudgeResult,
    JudgeRunResult,
    JudgeTask,
    LDContext,
    NativeTool,
    ProviderHandler,
    TrackData,
)
from .utils import (
    collapse_messages_to_instructions as _collapse_messages_to_instructions,
)
from .utils import (
    normalize_mode,
    omit_model_stamps,
    to_ld_context,
    to_usage_dict,
)


def _provider_matches(handler: ProviderHandler, provider: str | None) -> bool:
    """Returns True when the handler covers the given provider or is a wildcard."""
    return bool(
        handler.provides_for
        and (handler.provides_for[0] == provider or handler.provides_for[0] == "*")
    )


def _handler_for(
    handlers: list[ProviderHandler], provider: str | None, mode: str
) -> ProviderHandler | None:
    """Exact ``(provider, mode)`` match, then a same-mode wildcard.

    A wildcard such as LangChain's ``('*', 'messages')`` must not hide a
    handler registered for the judge's own provider.
    """
    exact = next(
        (handler for handler in handlers if handler.provides_for == (provider, mode)),
        None,
    )
    if exact is not None:
        return exact
    return next(
        (
            handler
            for handler in handlers
            if handler.provides_for
            and handler.provides_for[0] == "*"
            and handler.provides_for[1] == mode
        ),
        None,
    )


logger = logging.getLogger(__name__)


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
    trajectory: str = "",
) -> dict[str, JudgeResult]:
    """
    Runs any judges configured on ``config['judgeConfiguration']`` against the
    produced output. Each judge is itself a tracked AI call.

    ``trajectory`` is the rendered tool-call trajectory of the invocation being
    judged, from ``execute_and_track``. It defaults to empty so a caller that
    has none -- a graph-level judge over several nodes, for instance -- is
    unchanged, and so is a judge for a config with no tools.
    """
    from .lifecycle import extract_variation
    from .tracking import execute_and_track

    judge_results: dict[str, JudgeResult] = {}

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
            #   1. Exact provider + mode
            #   2. Wildcard provider + same mode
            #   3. Exact provider agent handler, then a wildcard agent handler
            #   4. Parent handler when it covers the same provider or is a wildcard
            # When falling back to an agent-mode handler for a messages-mode judge
            # config, collapse messages into a single instructions block.
            judge_handler: ProviderHandler = handler
            collapse_messages = False
            if handlers:
                exact = _handler_for(handlers, judge_provider, judge_mode)
                agent_fallback = (
                    _handler_for(handlers, judge_provider, "agent")
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
                else:
                    logger.warning(
                        "Judge '%s' skipped: no handler provides for provider %r",
                        judge_key,
                        judge_provider,
                    )
                    continue

            effective_judge_config = (
                _collapse_messages_to_instructions(judge_ai_config)
                if collapse_messages
                else judge_ai_config
            )

            message_history = build_message_history(
                user_input=user_input,
                trajectory=trajectory,
                output=llm_response,
            )

            async with with_judge_evaluation(judge_key) as record_evaluation:
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
                        **({"input": user_input} if user_input else {}),
                    },
                )

                usage = to_usage_dict(result["usage"])
                entries = typesafe_judge_entries(result["response"])
                if entries is not None:
                    # One Jev call. Every label is returned with that call's
                    # full usage, and each label is tracked under its eventKey.
                    from .lifecycle import get_client

                    client = get_client()
                    for entry in entries:
                        result_key = f"{judge_key}.{entry['key']}"
                        score = entry["score"]
                        judge_results[result_key] = JudgeResult(
                            usage=usage,
                            response=entry["reason"],
                            score=score,
                            event_key=entry["eventKey"],
                        )
                        record_evaluation(score, None, result_key)
                        client.track(
                            entry["eventKey"],
                            to_ld_context(client, user_context),
                            {**base_track_data, "judgeConfigKey": result_key},
                            score,
                        )
                    continue

                score, reasoning = parse_judge_response(result["response"])
                judge_results[judge_key] = JudgeResult(
                    usage=usage,
                    response=reasoning,
                    score=score,
                )
                metric_score = numeric_score(score)
                if metric_score is not None:
                    record_evaluation(
                        metric_score,
                        reasoning if judge_handler.capture_content else None,
                    )

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
    user_input: str | None = None,
    trajectory: str = "",
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
                exact = _handler_for(handlers, judge_provider, judge_mode)
                agent_fallback = (
                    _handler_for(handlers, judge_provider, "agent")
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
                    user_input=user_input,
                    trajectory=trajectory,
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

    exact = _handler_for(handlers, task.judge_provider, task.judge_mode)
    agent_fallback = (
        _handler_for(handlers, task.judge_provider, "agent")
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

    # user_input and trajectory come off the task rather than being omitted:
    # this path used to build a history with neither, so a judge grading the
    # same response saw a different conversation than the inline path did.
    message_history = build_message_history(
        user_input=task.user_input,
        trajectory=task.trajectory,
        output=task.actual_output,
    )

    async with with_judge_evaluation(task.config_key) as record_evaluation:
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
                **({"input": task.user_input} if task.user_input else {}),
            },
        )

        usage = to_usage_dict(result["usage"])
        try:
            entries = typesafe_judge_entries(result["response"])
        except ValueError:
            return None
        if entries is not None:
            first = entries[0]
            result_key = f"{task.config_key}.{first['key']}"
            expanded = {
                f"{task.config_key}.{entry['key']}": JudgeResult(
                    usage=usage,
                    response=entry["reason"],
                    score=entry["score"],
                    event_key=entry["eventKey"],
                )
                for entry in entries
            }
            metrics = [
                {
                    "eventKey": entry["eventKey"],
                    "score": entry["score"],
                    "judgeConfigKey": f"{task.config_key}.{entry['key']}",
                }
                for entry in entries
            ]
            for entry in entries:
                record_evaluation(
                    entry["score"], None, f"{task.config_key}.{entry['key']}"
                )
            typesafe_track_data: TrackData = {
                **omit_model_stamps(task.parent_track_data),
                **result["track_data"],
                "judgeConfigKey": result_key,
            }
            return JudgeRunResult(
                score=first["score"],
                response=first["reason"],
                usage=usage,
                track_data=typesafe_track_data,
                results=expanded,
                metrics=metrics,
                event_key=first["eventKey"],
            )

        try:
            score, reasoning = parse_judge_response(result["response"])
        except ValueError:
            return None

        # The score is reported as the judge gave it. A missing or null score
        # is not a zero: coercing it would record a gen_ai.evaluation of 0 --
        # indistinguishable from a judge that scored the output a hard fail --
        # where every other non-numeric judge output skips the metric instead.
        metric_score = numeric_score(score)
        if metric_score is not None:
            record_evaluation(
                metric_score,
                reasoning if judge_handler.capture_content else None,
            )
        # A judge without a pinned model config must not inherit the parent's
        # modelKey / modelVersion; every other parent-only key (graphKey, ...)
        # is still carried over.
        merged_track_data: TrackData = {
            **omit_model_stamps(task.parent_track_data),
            **result["track_data"],
            "judgeConfigKey": task.config_key,
        }

        return JudgeRunResult(
            score=score, response=reasoning, usage=usage, track_data=merged_track_data
        )
