"""Map OpenAI Responses stream events onto the AG-UI text and tool subset.

Both OpenAI handler packages call :func:`map_responses_event`. They only
offer function tools, so hosted and MCP item types are ignored until those
tools are wired.
"""

from __future__ import annotations

from typing import Any

from .agui import TurnEmitter


def _call_id(event: Any, item_to_call: dict[str, str]) -> str:
    direct = getattr(event, "call_id", None)
    if isinstance(direct, str) and direct:
        return direct
    item_id = getattr(event, "item_id", None)
    if isinstance(item_id, str) and item_id in item_to_call:
        return item_to_call[item_id]
    return item_id if isinstance(item_id, str) else ""


def map_responses_event(
    event: Any, emitter: TurnEmitter, item_to_call: dict[str, str]
) -> list[dict[str, Any]]:
    """Map one Responses event. Unknown events, including hosted tools, are ignored."""
    kind = getattr(event, "type", None)
    if kind == "response.created":
        return emitter.end_text()
    if kind == "response.output_text.delta":
        delta = getattr(event, "delta", "")
        if not isinstance(delta, str) or not delta:
            return []
        return emitter.text(delta)
    if kind == "response.output_item.added":
        item = getattr(event, "item", None)
        if getattr(item, "type", None) != "function_call":
            return []
        call_id = getattr(item, "call_id", None) or getattr(item, "id", None) or ""
        name = getattr(item, "name", None) or ""
        item_id = getattr(item, "id", None)
        if isinstance(item_id, str) and isinstance(call_id, str) and call_id:
            item_to_call[item_id] = call_id
        return emitter.tool_start(str(call_id), str(name))
    if kind == "response.function_call_arguments.delta":
        delta = getattr(event, "delta", "")
        if not isinstance(delta, str):
            return []
        return emitter.tool_args(_call_id(event, item_to_call), delta)
    if kind == "response.function_call_arguments.done":
        return emitter.tool_end(
            _call_id(event, item_to_call),
            arguments=getattr(event, "arguments", None),
        )
    return []
