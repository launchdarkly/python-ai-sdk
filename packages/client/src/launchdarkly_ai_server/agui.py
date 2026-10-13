"""Opt-in AG-UI reader for one AI Config turn.

Handlers may yield ``chunk`` or the text and tool subset below, then one
``done``. :func:`stream` is the opt-in entry: it brackets that subset with
``RUN_STARTED`` and ``RUN_FINISHED``. ``config().stream()`` still yields
``chunk`` then ``done``. The SDK runs the tool loop. This stream only reports it.

Argument deltas and tool results are content. :class:`TurnEmitter` omits them
when the handler was built with ``capture_content`` off, and still emits the
tool lifecycle.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from .conversation import bind_conversation_id
from .sdk_usage import report_usage
from .types import JudgeResult, LDContext

PROTOCOL_VERSION = "1.0"

TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
TOOL_CALL_START = "TOOL_CALL_START"
TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
TOOL_CALL_END = "TOOL_CALL_END"
TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
RUN_STARTED = "RUN_STARTED"
RUN_FINISHED = "RUN_FINISHED"

TEXT_TYPES = frozenset({TEXT_MESSAGE_START, TEXT_MESSAGE_CONTENT, TEXT_MESSAGE_END})
TOOL_TYPES = frozenset(
    {TOOL_CALL_START, TOOL_CALL_ARGS, TOOL_CALL_END, TOOL_CALL_RESULT}
)
HANDLER_RICH_TYPES = TEXT_TYPES | TOOL_TYPES

_FAILURE_CONTENT = "An error occurred."


def argument_text(value: Any) -> str | None:
    """The argument text for one ``TOOL_CALL_ARGS`` delta.

    A string is that string. Any other JSON value is compact JSON. ``None`` and
    a value that is not JSON under :func:`json.dumps` with ``allow_nan=False``
    produce no delta.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        return json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError):
        return None


def safe_tool_content(value: Any) -> tuple[str, bool]:
    """``(content, failed)`` for a tool result.

    A string is not encoded again. Any other JSON value is compact JSON.
    A value that is not a string and not serializable under that rule is a
    failed result: the caller masks it. The model-facing value is unchanged.
    """
    if isinstance(value, str):
        return value, False
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _FAILURE_CONTENT, True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return _FAILURE_CONTENT, True
    try:
        return (
            json.dumps(
                value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ),
            False,
        )
    except (TypeError, ValueError):
        return _FAILURE_CONTENT, True


class TurnEmitter:
    """Yields the text and tool subset for one handler run.

    ``capture_content`` gates argument deltas and result content. The tool name
    and call id are always present on ``TOOL_CALL_START``. A closed call is not
    reopened.
    """

    def __init__(self, *, capture_content: bool) -> None:
        self.capture_content = capture_content
        self._open: str | None = None
        self._seq = 0
        self._started: set[str] = set()
        self._args: set[str] = set()
        self._arg_text: dict[str, str] = {}
        self._ended: set[str] = set()
        self._resulted: set[str] = set()

    def _mint(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}-{self._seq}"

    def text(self, delta: str) -> list[dict[str, Any]]:
        """Append assistant text. The first slice opens the message."""
        if not delta:
            return []
        events: list[dict[str, Any]] = []
        if self._open is None:
            self._open = self._mint("msg")
            events.append({"type": TEXT_MESSAGE_START, "messageId": self._open})
        events.append(
            {
                "type": TEXT_MESSAGE_CONTENT,
                "messageId": self._open,
                "delta": delta,
            }
        )
        return events

    def end_text(self) -> list[dict[str, Any]]:
        if self._open is None:
            return []
        event = {"type": TEXT_MESSAGE_END, "messageId": self._open}
        self._open = None
        return [event]

    def tool_start(
        self, tool_call_id: str, tool_call_name: str
    ) -> list[dict[str, Any]]:
        if not tool_call_id or not tool_call_name or tool_call_id in self._started:
            return []
        self._started.add(tool_call_id)
        event: dict[str, Any] = {
            "type": TOOL_CALL_START,
            "toolCallId": tool_call_id,
            "toolCallName": tool_call_name,
        }
        if self._open is not None:
            event["parentMessageId"] = self._open
        return [event]

    def has_argument_deltas(self, tool_call_id: str) -> bool:
        """True once a streamed argument slice was recorded for this call."""
        return tool_call_id in self._args

    def tool_args(self, tool_call_id: str, delta: str) -> list[dict[str, Any]]:
        if (
            not self.capture_content
            or not delta
            or tool_call_id not in self._started
            or tool_call_id in self._ended
        ):
            return []
        self._args.add(tool_call_id)
        self._arg_text[tool_call_id] = self._arg_text.get(tool_call_id, "") + delta
        return [{"type": TOOL_CALL_ARGS, "toolCallId": tool_call_id, "delta": delta}]

    def tool_end(
        self, tool_call_id: str, *, arguments: Any = None
    ) -> list[dict[str, Any]]:
        """Close the argument text.

        When nothing was streamed, one delta carries the complete arguments.
        A streamed prefix is extended with the missing suffix of those
        arguments, so a partial delta cannot publish truncated JSON.
        """
        if tool_call_id not in self._started or tool_call_id in self._ended:
            return []
        events: list[dict[str, Any]] = []
        if self.capture_content and arguments is not None:
            text = argument_text(arguments)
            have = self._arg_text.get(tool_call_id, "")
            extra = text[len(have) :] if text and text.startswith(have) else ""
            if extra:
                self._args.add(tool_call_id)
                self._arg_text[tool_call_id] = have + extra
                events.append(
                    {"type": TOOL_CALL_ARGS, "toolCallId": tool_call_id, "delta": extra}
                )
        self._ended.add(tool_call_id)
        events.append({"type": TOOL_CALL_END, "toolCallId": tool_call_id})
        return events

    def tool_result(
        self, tool_call_id: str, value: Any = None, *, failed: bool = False
    ) -> list[dict[str, Any]]:
        """Report a finished call, including failure. Does not run the tool."""
        if tool_call_id not in self._started or tool_call_id in self._resulted:
            return []
        events = self.tool_end(tool_call_id) if tool_call_id not in self._ended else []
        self._resulted.add(tool_call_id)
        event: dict[str, Any] = {
            "type": TOOL_CALL_RESULT,
            "messageId": self._mint("toolmsg"),
            "toolCallId": tool_call_id,
        }
        if failed:
            event["metadata"] = {"launchdarkly": {"toolFailed": True}}
            if self.capture_content:
                event["content"] = _FAILURE_CONTENT
        elif self.capture_content:
            content, bad = safe_tool_content(value)
            if bad:
                event["content"] = _FAILURE_CONTENT
                event["metadata"] = {"launchdarkly": {"toolFailed": True}}
            else:
                event["content"] = content
        events.append(event)
        return events

    def fail_open(self) -> list[dict[str, Any]]:
        """Close every call that started and did not finish, as a masked failure."""
        events: list[dict[str, Any]] = []
        for call_id in list(self._started):
            if call_id not in self._resulted:
                events.extend(self.tool_result(call_id, failed=True))
        return events


class StableProjector:
    """Projects a handler stream onto ``chunk`` then one internal ``done``.

    ``TEXT_MESSAGE_CONTENT`` becomes a chunk. Tool events and any other type are
    dropped and do not close the turn. Only the handler's ``done`` does.
    """

    def __init__(self) -> None:
        self.saw_text = False
        self.done: dict[str, Any] | None = None

    def feed(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        typ = event.get("type")
        if typ in TEXT_TYPES:
            self.saw_text = True
        if typ == "done":
            self.done = event
            return []
        if typ == "chunk":
            if self.saw_text:
                return []
            text = event.get("text", "")
            if not isinstance(text, str):
                return []
            return [{"type": "chunk", "text": text}]
        if typ == TEXT_MESSAGE_CONTENT:
            delta = event.get("delta") or ""
            if not delta:
                return []
            return [{"type": "chunk", "text": delta}]
        return []


def _token_usage(
    parsed: dict[str, Any], *, provider: str, model: str
) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "inputTokens": int(parsed.get("input") or 0),
        "outputTokens": int(parsed.get("output") or 0),
        "totalTokens": int(parsed.get("total") or 0),
    }
    details = parsed.get("input_details")
    if isinstance(details, dict):
        usage["cachedInputTokens"] = int(details.get("cache_read") or 0)
        usage["cacheWriteInputTokens"] = int(details.get("cache_creation") or 0)
    if provider:
        usage["provider"] = provider
    if model:
        usage["model"] = model
    return usage


def _judge_payload(results: dict[str, JudgeResult]) -> dict[str, Any] | None:
    if not results:
        return None
    payload: dict[str, Any] = {}
    for key, result in results.items():
        usage: dict[str, Any] = {
            "input": result.usage.input,
            "output": result.usage.output,
            "total": result.usage.total,
        }
        details = result.usage.input_details
        if details is not None:
            usage["input_details"] = {
                "uncached": details.uncached,
                "cache_read": details.cache_read,
                "cache_creation": details.cache_creation,
            }
        payload[key] = {
            "usage": usage,
            "response": result.response,
            "score": result.score,
        }
    return payload


class RunReader:
    """Maps one handler stream onto one AG-UI run.

    Legacy ``chunk`` events become one text message when the handler yielded no
    ``TEXT_MESSAGE_*`` event. ``RUN_STARTED`` and ``RUN_FINISHED`` are minted
    here. The handler's ``done`` is consumed, not forwarded.
    """

    def __init__(self) -> None:
        self.thread_id = str(uuid.uuid4())
        self.run_id = str(uuid.uuid4())
        self._mode: str | None = None
        self._open: str | None = None
        self._deltas: list[str] = []
        self._started: set[str] = set()
        self._ended: set[str] = set()
        self._resulted: set[str] = set()
        self._seq = 0

    def start(self) -> dict[str, Any]:
        return {
            "type": RUN_STARTED,
            "threadId": self.thread_id,
            "runId": self.run_id,
            "protocolVersion": PROTOCOL_VERSION,
        }

    def _mint(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}-{self._seq}"

    def feed(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        typ = event.get("type")
        if typ == "chunk":
            return self._chunk(event.get("text") or "")
        if typ == TEXT_MESSAGE_START:
            return self._text_start(event.get("messageId"))
        if typ == TEXT_MESSAGE_CONTENT:
            return self._text_content(event.get("messageId"), event.get("delta"))
        if typ == TEXT_MESSAGE_END:
            return self._text_end(event.get("messageId"))
        if typ == TOOL_CALL_START:
            return self._tool_start(event)
        if typ == TOOL_CALL_ARGS:
            return self._tool_args(event)
        if typ == TOOL_CALL_END:
            return self._tool_end(event)
        if typ == TOOL_CALL_RESULT:
            return self._tool_result(event)
        return []

    def finish(
        self,
        done: dict[str, Any],
        judge_results: dict[str, JudgeResult] | None,
        *,
        provider: str,
        model: str,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self._open is not None:
            events.append({"type": TEXT_MESSAGE_END, "messageId": self._open})
            self._open = None
        response = done.get("response")
        if not isinstance(response, str):
            response = "".join(self._deltas)
        finished: dict[str, Any] = {
            "type": RUN_FINISHED,
            "threadId": self.thread_id,
            "runId": self.run_id,
        }
        if response:
            finished["result"] = response
        parsed = done.get("usage")
        if done.get("_had_usage") and isinstance(parsed, dict):
            finished["usage"] = [_token_usage(parsed, provider=provider, model=model)]
        judges = _judge_payload(judge_results or {})
        if judges is not None:
            finished["metadata"] = {"launchdarkly": {"judgeResults": judges}}
        events.append(finished)
        return events

    def _chunk(self, text: str) -> list[dict[str, Any]]:
        if self._mode == "message" or not text:
            return []
        self._mode = "chunk"
        events: list[dict[str, Any]] = []
        if self._open is None:
            self._open = self._mint("msg")
            events.append({"type": TEXT_MESSAGE_START, "messageId": self._open})
        events.append(
            {"type": TEXT_MESSAGE_CONTENT, "messageId": self._open, "delta": text}
        )
        self._deltas.append(text)
        return events

    def _close_open(self, message_id: str) -> list[dict[str, Any]]:
        if self._open is None or self._open == message_id:
            return []
        event = {"type": TEXT_MESSAGE_END, "messageId": self._open}
        self._open = None
        return [event]

    def _text_start(self, message_id: Any) -> list[dict[str, Any]]:
        if not isinstance(message_id, str) or not message_id:
            return []
        self._mode = "message"
        if self._open == message_id:
            return []
        events = self._close_open(message_id)
        self._open = message_id
        events.append({"type": TEXT_MESSAGE_START, "messageId": message_id})
        return events

    def _text_content(self, message_id: Any, delta: Any) -> list[dict[str, Any]]:
        if not isinstance(delta, str):
            return []
        self._mode = "message"
        mid = message_id if isinstance(message_id, str) and message_id else None
        events: list[dict[str, Any]] = []
        if self._open is None:
            if mid is None:
                mid = self._mint("msg")
            self._open = mid
            events.append({"type": TEXT_MESSAGE_START, "messageId": mid})
        elif mid is not None and mid != self._open:
            events.extend(self._close_open(mid))
            self._open = mid
            events.append({"type": TEXT_MESSAGE_START, "messageId": mid})
        events.append(
            {
                "type": TEXT_MESSAGE_CONTENT,
                "messageId": self._open,
                "delta": delta,
            }
        )
        self._deltas.append(delta)
        return events

    def _text_end(self, message_id: Any) -> list[dict[str, Any]]:
        if self._open is None:
            return []
        if isinstance(message_id, str) and message_id and message_id != self._open:
            return []
        event = {"type": TEXT_MESSAGE_END, "messageId": self._open}
        self._open = None
        return [event]

    def _tool_start(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        call_id = event.get("toolCallId")
        name = event.get("toolCallName")
        if (
            not isinstance(call_id, str)
            or not call_id
            or not isinstance(name, str)
            or not name
            or call_id in self._started
        ):
            return []
        self._started.add(call_id)
        started: dict[str, Any] = {
            "type": TOOL_CALL_START,
            "toolCallId": call_id,
            "toolCallName": name,
        }
        parent = event.get("parentMessageId")
        if isinstance(parent, str) and parent:
            started["parentMessageId"] = parent
        return [started]

    def _tool_args(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        call_id = event.get("toolCallId")
        delta = event.get("delta")
        if (
            not isinstance(call_id, str)
            or call_id not in self._started
            or call_id in self._ended
            or not isinstance(delta, str)
        ):
            return []
        return [{"type": TOOL_CALL_ARGS, "toolCallId": call_id, "delta": delta}]

    def _tool_end(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        call_id = event.get("toolCallId")
        if (
            not isinstance(call_id, str)
            or call_id not in self._started
            or call_id in self._ended
        ):
            return []
        self._ended.add(call_id)
        return [{"type": TOOL_CALL_END, "toolCallId": call_id}]

    def _tool_result(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        call_id = event.get("toolCallId")
        if (
            not isinstance(call_id, str)
            or call_id not in self._started
            or call_id in self._resulted
        ):
            return []
        self._resulted.add(call_id)
        message_id = event.get("messageId")
        if not isinstance(message_id, str) or not message_id:
            message_id = self._mint("toolmsg")
        result: dict[str, Any] = {
            "type": TOOL_CALL_RESULT,
            "messageId": message_id,
            "toolCallId": call_id,
        }
        content = event.get("content")
        if isinstance(content, str):
            result["content"] = content
        metadata = event.get("metadata")
        if isinstance(metadata, dict):
            launchdarkly = metadata.get("launchdarkly")
            if (
                isinstance(launchdarkly, dict)
                and launchdarkly.get("toolFailed") is True
            ):
                result["metadata"] = {"launchdarkly": {"toolFailed": True}}
        return [result]


def stream(
    instance: Any,
    user_input: str | None,
    context: LDContext,
    *,
    variables: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Stream one config turn as AG-UI events.

    Same arguments as ``ConfigInstance.stream``. ``RUN_STARTED`` is first and
    ``RUN_FINISHED`` is last. A failure, including cancellation, raises and does
    not yield ``RUN_FINISHED``.
    """
    report_usage("client.agui.stream")
    return bind_conversation_id(
        _stream_events(instance, user_input, context, variables, history)
    )


async def _stream_events(
    instance: Any,
    user_input: str | None,
    context: LDContext,
    variables: dict[str, Any] | None,
    history: list[dict[str, Any]] | None,
) -> AsyncGenerator[dict[str, Any], None]:
    from .judges import run_judges

    sink: dict[str, Any] = {}
    reader = RunReader()
    started = False
    done: dict[str, Any] | None = None
    async for event in instance._execute_stream(
        user_input, context, variables, history, sink
    ):
        if not started:
            yield reader.start()
            started = True
        if event.get("type") == "done":
            done = event
            continue
        for item in reader.feed(event):
            yield item
    if not started:
        yield reader.start()
    if done is None:
        done = {"response": "", "usage": {}, "_had_usage": False}
    judge_results: dict[str, JudgeResult] = {}
    if not instance._skip_judges and sink:
        judge_results = await run_judges(
            config=sink["config"],
            user_context=context,
            handler=sink["handler"],
            handlers=sink["handlers"],
            user_input=user_input,
            trajectory=done.get("trajectory", ""),
            llm_response=done.get("response", "")
            if isinstance(done.get("response"), str)
            else "",
            base_track_data=done.get("track_data", {}),
            tool_handlers=sink["tools"],
        )
    for item in reader.finish(
        done,
        judge_results,
        provider=str(sink.get("provider") or ""),
        model=str(sink.get("model") or ""),
    ):
        yield item
