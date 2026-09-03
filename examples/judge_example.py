"""
Demonstrates background judge evaluation using invoke() with skip_judges=True.

The example:
  1. Makes a single config() call with skip_judges=True — no judge key is ever
     specified by the caller. Judge keys are auto-discovered from the main
     config's judgeConfiguration at invocation time.
  2. invoke() returns judge_tasks: list[JudgeTask] alongside the LLM response.
     Each task is a fully-resolved snapshot ready to pass to a background thread.
  3. One OS thread is spawned per task. The thread handles the AI call AND the
     LaunchDarkly tracking event autonomously. Python threads share the parent
     process's LD client, so no re-initialization is needed in the thread.
  4. In production, omit the thread.join() so the main thread continues
     immediately. This example joins for verification purposes only.

Usage:
  python main.py judge <flag-key> "<user input>"
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from typing import Any

import examples.register  # noqa: F401 – side-effect: populate global_registry
from examples.utils import new_multi_context, write_output
from launchdarkly_ai_server import (
    JudgeRunResult,
    JudgeTask,
    config,
    get_client,
    global_registry,
    resolve_handlers,
    run_judge,
)
from launchdarkly_ai_server.utils import to_ld_context


def _judge_in_thread(
    task: JudgeTask,
    handlers: list[Any],
    result_box: dict[str, Any],
) -> None:
    """Run run_judge() and track the result — fully autonomous within this thread.

    Python threads share the parent process's memory, so get_client() returns
    the already-initialized LD client without any re-initialization.
    """
    try:
        result = asyncio.run(run_judge(task, handlers))
        result_box["value"] = result

        if result and task.evaluation_metric_key:
            client = get_client()
            ld_ctx = to_ld_context(client, task.user_context)
            client.track(
                task.evaluation_metric_key,
                ld_ctx,
                result.track_data,
                result.score,
            )
    except Exception as exc:
        result_box["error"] = str(exc)


async def run(key: str, user_input: str) -> None:
    ctx = new_multi_context()
    print(
        f"[context] {json.dumps(ctx, ensure_ascii=False, separators=(',', ':'))}",
        file=sys.stderr,
    )

    # Single config() call — the caller never touches a judge key.
    # skip_judges=True suppresses automatic inline evaluation so we control when
    # judging happens and on which thread.
    instance = config(key=key, registry=global_registry, skip_judges=True)

    # invoke() calls the LLM, then auto-discovers judges from judgeConfiguration
    # and returns them as pre-packaged JudgeTask objects. No AI call yet for judges.
    resp = await instance.invoke(user_input, ctx)
    llm_response = (
        resp.response if isinstance(resp.response, str) else str(resp.response)
    )
    print(f"[invoke] response: {llm_response[:120]}\n")

    # Resolve handlers once on the main thread and pass them to each worker.
    handlers = resolve_handlers(global_registry, None) or []

    threads: list[tuple[threading.Thread, dict[str, Any], JudgeTask]] = []
    for task in resp.judge_tasks or []:
        print(f"[judge] spawning thread (judge: {task.config_key})")
        result_box: dict[str, Any] = {}
        thread = threading.Thread(
            target=_judge_in_thread,
            args=(task, handlers, result_box),
            daemon=True,
        )
        thread.start()
        threads.append((thread, result_box, task))

    print("[judge] threads spawned — main thread continues.\n")

    # In production: omit join() and let threads run freely.
    for thread, _, _ in threads:
        thread.join()

    # ── Verification (example only) ───────────────────────────────────────────
    print("── verification ──────────────────────────────────────")
    for _, result_box, _ in threads:
        if "error" in result_box:
            print(f"[judge] thread error: {result_box['error']}")
            continue

        judge_result: JudgeRunResult | None = result_box.get("value")
        if judge_result:
            score_ok = (
                isinstance(judge_result.score, (int, float))
                and 0 <= judge_result.score <= 1
            )
            reasoning_ok = (
                isinstance(judge_result.response, str)
                and len(judge_result.response) > 0
            )
            usage_ok = isinstance(judge_result.usage.input, int) and isinstance(
                judge_result.usage.output, int
            )
            run_id = judge_result.track_data.get("runId", "")

            print(
                f"score ∈ [0,1]:      {'✓' if score_ok else '✗'} ({judge_result.score})"
            )
            print(
                f"reasoning present:  {'✓' if reasoning_ok else '✗'} ({judge_result.response[:60]})"
            )
            print(
                f"usage tokens:       {'✓' if usage_ok else '✗'} "
                f"(in={judge_result.usage.input} out={judge_result.usage.output})"
            )
            print(
                f"trackData.runId:    {'✓' if run_id else '✗'} ({str(run_id)[:8]}...)"
            )
        else:
            print(
                "[judge] result was None — check that a handler matches the judge config"
            )

    write_output(
        {
            "response": resp.response,
            "judge_results": [r.get("value") for _, r, _ in threads],
        }
    )
