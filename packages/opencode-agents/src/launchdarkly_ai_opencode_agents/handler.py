"""Reduce OpenCode session events into handler stream events.

The session is a caller-supplied async iterable of OpenCode bus events. This
module does not start a process and does not construct an OpenCode SDK client.
The handler yields the AG-UI text and tool subset, then one ``done``.
``config().stream()`` still yields ``chunk`` then ``done``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable, Callable, Mapping
from typing import Any

from launchdarkly_ai_server import (
    AiConfigRep,
    ProviderHandler,
    _config,
    _create_handler,
)
from launchdarkly_ai_server.agui import TurnEmitter
from launchdarkly_ai_server.lifecycle import effective_capture_content
from launchdarkly_ai_server.sdk_usage import report_usage
from launchdarkly_ai_server.utils import number_or_zero

from ._version import PACKAGE_NAME, __version__


class OpenCodeSessionError(Exception):
    """The OpenCode session failed.

    The provider error body is not included. A failed session does not finish
    the run.
    """


class _Reducer:
    """One pass over a session.

    Text is emitted only for messages ``message.updated`` has marked assistant.
    A part snapshot is that part's full text so far: a later delta still appends,
    and a snapshot that repeats text already emitted adds nothing. The first
    ``running`` tool state with complete ``state.input`` opens the call. OpenCode
    runs the tool. ``state.raw`` is not diffed into argument text.
    """

    def __init__(self, *, capture_content: bool) -> None:
        self._emitter = TurnEmitter(capture_content=capture_content)
        self._text_message_id: str | None = None
        self._started_tools: set[str] = set()
        self._resulted_tools: set[str] = set()
        self._held_tools: dict[str, list[Mapping[str, Any]]] = {}
        self._deltas: list[str] = []
        self._roles: dict[str, str] = {}
        self._held: dict[str, list[tuple[str, str, str]]] = {}
        self._part_text: dict[str, str] = {}
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

    def _accept_text(
        self, message_id: str, part_id: str, kind: str, text: str
    ) -> list[dict[str, Any]]:
        role = self._roles.get(message_id)
        if role is None:
            self._held.setdefault(message_id, []).append((kind, part_id, text))
            return []
        if role != "assistant":
            return []
        events = self._begin_message(message_id)
        events.extend(self._apply_text(part_id, kind, text))
        return events

    def _flush_held(self, message_id: str) -> list[dict[str, Any]]:
        held = self._held.pop(message_id, [])
        events: list[dict[str, Any]] = []
        for kind, part_id, text in held:
            events.extend(self._apply_text(part_id, kind, text))
        return events

    def _apply_text(self, part_id: str, kind: str, text: str) -> list[dict[str, Any]]:
        if kind == "delta":
            if part_id:
                self._part_text[part_id] = self._part_text.get(part_id, "") + text
            return self._chunk(text)
        already = self._part_text.get(part_id, "")
        if text.startswith(already):
            suffix = text[len(already) :]
            if part_id:
                self._part_text[part_id] = text
        elif not already:
            suffix = text
            if part_id:
                self._part_text[part_id] = text
        else:
            return []
        if not suffix:
            return []
        return self._chunk(suffix)

    def _begin_message(self, message_id: str) -> list[dict[str, Any]]:
        if self._text_message_id == message_id:
            return []
        events = self._emitter.end_text()
        self._text_message_id = message_id
        return events

    def close(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        # End the open assistant message before an unlabeled tool. Otherwise
        # tool_start copies that message into parentMessageId, and fail_open
        # then marks the call failed while it is still nested there.
        events.extend(self._emitter.end_text())
        self._text_message_id = None
        for message_id, parts in list(self._held_tools.items()):
            if self._roles.get(message_id) is None:
                for part in parts:
                    events.extend(self._emit_tool(part))
        self._held_tools.clear()
        events.extend(self._emitter.fail_open())
        return events

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
            if part.get("type") != "text":
                return []
            return self._text_snapshot(part)
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
        return self._accept_text(
            message_id, part_id if isinstance(part_id, str) else "", "delta", delta
        )

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
        return self._accept_text(message_id, part_id, "snapshot", text)

    def _chunk(self, text: str) -> list[dict[str, Any]]:
        self._deltas.append(text)
        return self._emitter.text(text)

    def _tool(self, part: Mapping[str, Any]) -> list[dict[str, Any]]:
        message_id = part.get("messageID")
        if isinstance(message_id, str):
            role = self._roles.get(message_id)
            if role is None:
                self._held_tools.setdefault(message_id, []).append(part)
                return []
            if role != "assistant":
                return []
        return self._emit_tool(part)

    def _emit_tool(self, part: Mapping[str, Any]) -> list[dict[str, Any]]:
        state = part.get("state")
        if not isinstance(state, Mapping):
            return []
        status = state.get("status")
        raw_id = part.get("callID") or part.get("id") or ""
        call_id = raw_id if isinstance(raw_id, str) else ""
        name = part.get("tool") if isinstance(part.get("tool"), str) else ""
        if not call_id or status == "pending":
            return []
        if status == "running":
            if "input" not in state or call_id in self._started_tools or not name:
                return []
            self._started_tools.add(call_id)
            events = self._emitter.tool_start(call_id, name)
            events.extend(self._emitter.tool_end(call_id, arguments=state.get("input")))
            return events
        if status not in ("completed", "error") or call_id in self._resulted_tools:
            return []
        opened: list[dict[str, Any]] = []
        if call_id not in self._started_tools and name:
            self._started_tools.add(call_id)
            opened.extend(self._emitter.tool_start(call_id, name))
            opened.extend(self._emitter.tool_end(call_id, arguments=state.get("input")))
        self._resulted_tools.add(call_id)
        if status == "error":
            opened.extend(self._emitter.tool_result(call_id, failed=True))
        else:
            opened.extend(self._emitter.tool_result(call_id, state.get("output")))
        return opened

    def _message(self, info: Mapping[str, Any]) -> list[dict[str, Any]]:
        message_id = info.get("id")
        role = info.get("role")
        events: list[dict[str, Any]] = []
        if isinstance(message_id, str) and isinstance(role, str):
            self._roles[message_id] = role
            if role == "assistant":
                events.extend(self._begin_message(message_id))
                events.extend(self._flush_held(message_id))
                for part in self._held_tools.pop(message_id, []):
                    events.extend(self._emit_tool(part))
            else:
                self._held.pop(message_id, None)
                self._held_tools.pop(message_id, None)
        if role != "assistant" or not isinstance(message_id, str):
            return events
        completed = info.get("time")
        if not isinstance(completed, Mapping) or "completed" not in completed:
            return events
        completed_time = info.get("time")
        if (
            isinstance(completed_time, Mapping)
            and "completed" in completed_time
            and message_id == self._text_message_id
        ):
            events.extend(self._emitter.end_text())
            self._text_message_id = None
        if message_id not in self._usage_ids:
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
    not start OpenCode and does not construct an OpenCode SDK client.

    The stream yields the AG-UI text and tool subset and one ``done``.
    ``config().stream()`` still yields ``chunk`` then ``done``. *capture_content*
    follows the other handlers: ``None`` reads the process default from
    ``configure`` when content is read, and ``True`` or ``False`` overrides it.
    """
    report_usage("opencode-agents.createOpenCodeHandler", PACKAGE_NAME, __version__)
    return _create_opencode_handler(capture_content=capture_content, events=events)


def opencode_agents(
    config_key: str,
    user_input: str,
    context: dict[str, Any],
    **kwargs: Any,
) -> Any:
    """Convenience wrapper: creates a handler and calls ``config(...).invoke()``.

    *events* is the session event source, the same callable
    :func:`create_opencode_handler` takes. This does not construct an OpenCode
    SDK client.
    """
    report_usage("opencode-agents.opencodeAgents", PACKAGE_NAME, __version__)
    variables = kwargs.pop("variables", None)
    history = kwargs.pop("history", None)
    capture_content = kwargs.pop("capture_content", None)
    events = kwargs.pop("events", None)
    return _config(
        key=config_key,
        handler=_create_opencode_handler(
            capture_content=capture_content, events=events
        ),
        **kwargs,
    ).invoke(user_input, context, variables, history)


def _create_opencode_handler(
    *,
    capture_content: bool | None,
    events: Callable[[], AsyncIterable[Mapping[str, Any]]] | None,
) -> ProviderHandler:
    async def _stream_impl(
        config: AiConfigRep,
        user_input: str | None,
        tool_handlers: dict[str, Any] | None,
        variables: dict[str, Any] | None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        reducer = _Reducer(capture_content=effective_capture_content(capture_content))
        async for event in reducer.reduce(_open_events(events)):
            yield event
        for event in reducer.close():
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
        capture_content=capture_content,
    )
