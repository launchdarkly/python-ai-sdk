"""AG-UI reader over the same handler drive as ``config().stream()``.

The stable reader keeps yielding ``chunk`` then ``done``. This one forwards
text and tool events and brackets them with ``RUN_STARTED`` and ``RUN_FINISHED``.
A raised turn ends the generator with neither ``RUN_FINISHED`` nor ``RUN_ERROR``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from typing import Any

from ..client import ConfigInstance
from ..conversation import bind_conversation_id
from ..judges import run_judges
from ..lifecycle import extract_variation
from ..registry import resolve_handlers, resolve_tools
from ..sdk_usage import report_usage
from ..tracking import _SETTLED_EVENT, _drive_handler_stream
from ..types import LDContext
from ..utils import select_handler

_TEXT = frozenset({"TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END"})
_TOOLS = frozenset(
    {"TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END", "TOOL_CALL_RESULT"}
)


def stream(
    config_instance: ConfigInstance,
    user_input: str | None,
    context: LDContext,
    *,
    variables: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Yield AG-UI events for one ``config()`` call.

    Takes the same arguments as ``ConfigInstance.stream``. ``config().stream()``
    is unchanged. This function is the opt-in reader; it is not a default, and
    it does not read or set a protocol field.
    """
    report_usage("client.experimental.richStream")
    return bind_conversation_id(
        _events(config_instance, user_input, context, variables, history)
    )


async def _events(
    config_instance: ConfigInstance,
    user_input: str | None,
    context: LDContext,
    variables: dict[str, Any] | None,
    history: list[dict[str, Any]] | None,
) -> AsyncGenerator[dict[str, Any], None]:
    resolved_handler_list = resolve_handlers(
        config_instance._registry, config_instance._normalize_handlers()
    )
    resolved_tools = resolve_tools(
        config_instance._registry, config_instance._tool_handlers
    )
    variation = await extract_variation(config_instance._key, context)
    config = variation["config"]
    meta = variation["meta"]
    if not resolved_handler_list:
        raise ValueError("No handlers provided to config()")
    handler = select_handler(config, meta, resolved_handler_list)

    thread_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    yield {
        "type": "RUN_STARTED",
        "threadId": thread_id,
        "runId": run_id,
        "protocolVersion": "1.0",
    }

    settled: dict[str, Any] | None = None
    synthetic_id: str | None = None
    saw_canonical_text = False
    async for event in _drive_handler_stream(
        config_key=config_instance._key,
        config=config,
        meta=meta,
        user_context=context,
        handler=handler,
        user_input=user_input,
        tool_handlers=resolved_tools,
        variables=variables,
        history=history,
    ):
        if event.get("type") == _SETTLED_EVENT:
            settled = event
            continue
        kind = event.get("type")
        if kind in _TEXT:
            if synthetic_id is not None:
                yield {"type": "TEXT_MESSAGE_END", "messageId": synthetic_id}
                synthetic_id = None
            saw_canonical_text = True
            yield event
        elif kind == "chunk" and not saw_canonical_text:
            text = event.get("text")
            if not isinstance(text, str) or not text:
                continue
            if synthetic_id is None:
                synthetic_id = str(uuid.uuid4())
                yield {"type": "TEXT_MESSAGE_START", "messageId": synthetic_id}
            yield {
                "type": "TEXT_MESSAGE_CONTENT",
                "messageId": synthetic_id,
                "delta": text,
            }
        elif kind in _TOOLS:
            yield event

    if synthetic_id is not None:
        yield {"type": "TEXT_MESSAGE_END", "messageId": synthetic_id}

    if settled is None:
        return

    judge_results: dict[str, Any] = {}
    if not config_instance._skip_judges:
        judge_results = await run_judges(
            config=config,
            user_context=context,
            handler=handler,
            handlers=resolved_handler_list,
            user_input=user_input,
            trajectory=settled.get("trajectory", ""),
            llm_response=settled.get("response", ""),
            base_track_data=settled.get("track_data", {}),
            tool_handlers=resolved_tools,
        )

    finished: dict[str, Any] = {
        "type": "RUN_FINISHED",
        "threadId": thread_id,
        "runId": run_id,
    }
    response = settled.get("response")
    if isinstance(response, str) and response:
        finished["result"] = response
    if settled.get("had_usage"):
        usage = settled.get("usage")
        if isinstance(usage, dict):
            finished["usage"] = _token_usage(usage)
    if judge_results:
        finished["metadata"] = {"launchdarkly": {"judgeResults": judge_results}}
    yield finished


def _token_usage(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    entry: dict[str, Any] = {
        "inputTokens": parsed.get("input", 0),
        "outputTokens": parsed.get("output", 0),
        "totalTokens": parsed.get("total", 0),
    }
    details = parsed.get("input_details")
    if isinstance(details, dict):
        entry["cachedInputTokens"] = details.get("cache_read", 0)
        entry["cacheWriteInputTokens"] = details.get("cache_creation", 0)
    return [entry]
