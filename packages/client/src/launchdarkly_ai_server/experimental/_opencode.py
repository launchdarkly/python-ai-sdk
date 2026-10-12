"""Reduce OpenCode session events into handler stream events.

The session is a caller-supplied async iterable of OpenCode bus events. This
module does not start a process. Tool execution stays inside OpenCode: the
events record a call that already ran.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator, AsyncIterable, Callable, Mapping
from typing import Any

from ..types import AiConfigRep, ProviderHandler
from ..utils import _create_handler, number_or_zero

_TOOL_FAILURE = "An error occurred."


class OpenCodeSessionError(Exception):
    """The OpenCode session failed.

    The provider error body is not included. A failed session is not a tool
    result and does not finish the run.
    """


class _Reducer:
    """One pass over a session. Repeated snapshots do not emit a second event."""

    def __init__(self, capture_content: bool) -> None:
        self._capture = capture_content
        self._open_message: str | None = None
        self._deltas: list[str] = []
        self._delta_parts: set[str] = set()
        self._snapshot_parts: set[str] = set()
        self._tools: dict[str, str] = {}
        self._usage_ids: set[str] = set()
        self._input = 0
        self._output = 0
        self._cache_read = 0
        self._cache_write = 0
        self._saw_usage = False
        self._saw_cache = False

    @property
    def deltas(self) -> list[str]:
        return self._deltas

    def usage_payload(self) -> dict[str, int] | None:
        """Raw usage for ``parse_usage``, or ``None`` when no assistant message completed."""
        if not self._saw_usage:
            return None
        payload: dict[str, int] = {
            "input_tokens": self._input,
            "output_tokens": self._output,
        }
        if self._saw_cache:
            payload["cache_read_input_tokens"] = self._cache_read
            payload["cache_creation_input_tokens"] = self._cache_write
        return payload

    async def reduce(
        self, events: AsyncIterable[Mapping[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for raw in events:
            if isinstance(raw, Mapping) and raw.get("type") == "session.error":
                raise OpenCodeSessionError("The OpenCode session failed.")
            for event in self._event(raw):
                yield event
        for event in self._end_open():
            yield event

    def _event(self, raw: Mapping[str, Any] | object) -> list[dict[str, Any]]:
        if not isinstance(raw, Mapping):
            return []
        kind = raw.get("type")
        properties = raw.get("properties")
        if not isinstance(properties, Mapping):
            return []
        if kind == "message.part.delta":
            return self._delta(properties)
        if kind == "message.part.updated":
            part = properties.get("part")
            if not isinstance(part, Mapping):
                return []
            if part.get("type") == "tool":
                return self._tool(part)
            if part.get("type") == "text":
                return self._text_snapshot(part)
            return []
        if kind == "message.updated":
            info = properties.get("info")
            if isinstance(info, Mapping):
                return self._message(info)
        return []

    def _delta(self, properties: Mapping[str, Any]) -> list[dict[str, Any]]:
        if properties.get("field") != "text":
            return []
        delta = properties.get("delta")
        message_id = properties.get("messageID")
        if not isinstance(delta, str) or not delta or not isinstance(message_id, str):
            return []
        part_id = properties.get("partID")
        if isinstance(part_id, str):
            if part_id in self._snapshot_parts:
                return []
            self._delta_parts.add(part_id)
        return self._text(message_id, delta)

    def _text_snapshot(self, part: Mapping[str, Any]) -> list[dict[str, Any]]:
        message_id = part.get("messageID")
        text = part.get("text")
        part_id = part.get("id")
        if (
            not isinstance(message_id, str)
            or not isinstance(text, str)
            or not text
            or not isinstance(part_id, str)
        ):
            return []
        if part_id in self._delta_parts or part_id in self._snapshot_parts:
            return []
        self._snapshot_parts.add(part_id)
        return self._text(message_id, text)

    def _text(self, message_id: str, delta: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self._open_message != message_id:
            events.extend(self._end_open())
            self._open_message = message_id
            events.append({"type": "TEXT_MESSAGE_START", "messageId": message_id})
        events.append(
            {
                "type": "TEXT_MESSAGE_CONTENT",
                "messageId": message_id,
                "delta": delta,
            }
        )
        self._deltas.append(delta)
        return events

    def _end_open(self) -> list[dict[str, Any]]:
        if self._open_message is None:
            return []
        message_id = self._open_message
        self._open_message = None
        return [{"type": "TEXT_MESSAGE_END", "messageId": message_id}]

    def _message(self, info: Mapping[str, Any]) -> list[dict[str, Any]]:
        if info.get("role") != "assistant":
            return []
        completed = info.get("time")
        if not isinstance(completed, Mapping) or "completed" not in completed:
            return []
        message_id = info.get("id")
        events: list[dict[str, Any]] = []
        if isinstance(message_id, str) and message_id == self._open_message:
            events.extend(self._end_open())
        if isinstance(message_id, str) and message_id not in self._usage_ids:
            tokens = info.get("tokens")
            if isinstance(tokens, Mapping):
                self._add_usage(tokens)
                self._usage_ids.add(message_id)
        return events

    def _add_usage(self, tokens: Mapping[str, Any]) -> None:
        self._saw_usage = True
        self._input += number_or_zero(tokens.get("input"))
        self._output += number_or_zero(tokens.get("output"))
        cache = tokens.get("cache")
        if isinstance(cache, Mapping):
            self._saw_cache = True
            self._cache_read += number_or_zero(cache.get("read"))
            self._cache_write += number_or_zero(cache.get("write"))

    def _tool(self, part: Mapping[str, Any]) -> list[dict[str, Any]]:
        state = part.get("state")
        if not isinstance(state, Mapping):
            return []
        call_id = part.get("callID") or part.get("id")
        name = part.get("tool")
        if (
            not isinstance(call_id, str)
            or not call_id
            or not isinstance(name, str)
            or not name
        ):
            return []
        status = state.get("status")
        if status not in ("running", "completed", "error"):
            return []
        events: list[dict[str, Any]] = []
        if self._tools.get(call_id) is None and (
            "input" in state or status != "running"
        ):
            events.extend(self._open_tool(call_id, name, state.get("input")))
            self._tools[call_id] = "open"
        if status == "completed" and self._tools.get(call_id) == "open":
            events.append(self._result(call_id, state.get("output", ""), failed=False))
            self._tools[call_id] = "done"
        elif status == "error" and self._tools.get(call_id) == "open":
            events.append(self._result(call_id, None, failed=True))
            self._tools[call_id] = "done"
        return events

    def _open_tool(
        self, call_id: str, name: str, raw_input: object
    ) -> list[dict[str, Any]]:
        start: dict[str, Any] = {
            "type": "TOOL_CALL_START",
            "toolCallId": call_id,
            "toolCallName": name,
        }
        if self._open_message is not None:
            start["parentMessageId"] = self._open_message
        events = [start]
        if self._capture:
            text = _argument_text(raw_input)
            if text:
                events.append(
                    {"type": "TOOL_CALL_ARGS", "toolCallId": call_id, "delta": text}
                )
        events.append({"type": "TOOL_CALL_END", "toolCallId": call_id})
        return events

    def _result(self, call_id: str, output: object, *, failed: bool) -> dict[str, Any]:
        content = ""
        metadata: dict[str, Any] | None = None
        if failed:
            metadata = {"launchdarkly": {"toolFailed": True}}
            if self._capture:
                content = _TOOL_FAILURE
        else:
            rendered = _result_text(output)
            if rendered is None:
                metadata = {"launchdarkly": {"toolFailed": True}}
                if self._capture:
                    content = _TOOL_FAILURE
            elif self._capture:
                content = rendered
        event: dict[str, Any] = {
            "type": "TOOL_CALL_RESULT",
            "messageId": str(uuid.uuid4()),
            "toolCallId": call_id,
            "content": content,
        }
        if metadata is not None:
            event["metadata"] = metadata
        return event


def _argument_text(value: object) -> str:
    """Argument text for one ``TOOL_CALL_ARGS`` delta. Empty when there is nothing to send."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    rendered = _json_text(value)
    return rendered or ""


def _result_text(value: object) -> str | None:
    """Success content, or ``None`` when the value cannot be placed on the event."""
    if isinstance(value, str):
        return value
    return _json_text(value)


def _json_text(value: object) -> str | None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None
    try:
        return json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError):
        return None


def _open_events(
    events: Callable[[], AsyncIterable[Mapping[str, Any]]] | None,
) -> AsyncIterable[Mapping[str, Any]]:
    if events is None:
        raise OpenCodeSessionError("The OpenCode handler has no session event source.")
    produced = events()
    if not hasattr(produced, "__aiter__"):
        raise TypeError("OpenCode event source must return an async iterable.")
    return produced


def create_opencode_handler(
    *,
    capture_content: bool | None = None,
    events: Callable[[], AsyncIterable[Mapping[str, Any]]] | None = None,
) -> ProviderHandler:
    """Return a handler that runs an OpenCode session supplied by *events*.

    ``provides_for`` is ``('*', 'agent')``, so the handler matches an agent-mode
    variation for any provider. *events* is called on every invocation and must
    return a fresh async iterable of OpenCode session events. This function does
    not start OpenCode.

    *capture_content* follows the other handlers: ``None`` reads the process
    default from ``configure`` when the call runs, and ``True`` or ``False``
    overrides it. Tool argument text and tool results are omitted from the
    stream unless capture is on. ``config().stream()`` still yields assistant
    text as ``chunk`` and one ``done``. Tool events stay on the experimental
    reader.
    """
    from ..lifecycle import effective_capture_content
    from ..sdk_usage import report_usage

    report_usage("client.experimental.opencode.createHandler")
    return _create_opencode_handler(
        capture_content=capture_content,
        events=events,
        resolve_capture=effective_capture_content,
    )


def _create_opencode_handler(
    *,
    capture_content: bool | None,
    events: Callable[[], AsyncIterable[Mapping[str, Any]]] | None,
    resolve_capture: Callable[[bool | None], bool],
) -> ProviderHandler:
    content_override = capture_content

    async def _stream_impl(
        config: AiConfigRep,
        user_input: str | None,
        tool_handlers: dict[str, Any] | None,
        variables: dict[str, Any] | None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        capture = resolve_capture(content_override)
        reducer = _Reducer(capture)
        async for event in reducer.reduce(_open_events(events)):
            yield event
        done: dict[str, Any] = {"type": "done", "output": "".join(reducer.deltas)}
        usage = reducer.usage_payload()
        if usage is not None:
            done["usage"] = usage
        yield done

    async def _call_impl(
        config: AiConfigRep,
        user_input: str | None,
        tool_handlers: dict[str, Any] | None,
        variables: dict[str, Any] | None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        output = ""
        usage: dict[str, Any] = {}
        async for event in _stream_impl(
            config, user_input, tool_handlers, variables, history
        ):
            if event.get("type") == "done":
                raw_output = event.get("output")
                output = raw_output if isinstance(raw_output, str) else ""
                raw_usage = event.get("usage")
                usage = raw_usage if isinstance(raw_usage, dict) else {}
        return {"output": output, "usage": usage}

    return _create_handler(
        ("*", "agent"),
        _call_impl,
        _stream_impl,
        capture_content=content_override,
    )
