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
    SpanMessage,
    SpanUsage,
    _config,
    _create_handler,
    add_cached_tokens_to_input,
    end_span_once,
    set_input_content_attributes,
    set_output_content_attributes,
    text_message,
)
from launchdarkly_ai_server.agui import TurnEmitter
from launchdarkly_ai_server.lifecycle import effective_capture_content
from launchdarkly_ai_server.sdk_usage import report_usage
from launchdarkly_ai_server.utils import number_or_zero

from ._version import PACKAGE_NAME, __version__
from .spans import fail_span, finish_root_span, mark_ok, start_root_span

Events = Callable[
    [
        AiConfigRep,
        str | None,
        dict[str, Any] | None,
        list[dict[str, Any]] | None,
        str | None,
    ],
    AsyncIterable[Mapping[str, Any]],
]


class OpenCodeSessionError(Exception):
    """The OpenCode session failed.

    The provider error body is not included. The error ``name`` is, when the
    event has one. A failed session does not finish the run.
    """


class _Reducer:
    """One pass over a session.

    Text is emitted only for messages ``message.updated`` has marked assistant.
    A part snapshot is that part's full text so far: a later delta still appends,
    and a snapshot that repeats text already emitted adds nothing. A text delta
    without ``partID`` is dropped. The first ``running`` tool state with complete
    ``state.input`` opens the call. OpenCode runs the tool. ``state.raw`` is not
    diffed into argument text.

    Events whose ``sessionID`` is not this session are ignored. The first
    ``sessionID`` seen is the session when the caller did not pass one. A
    ``session.idle`` for this session ends the iterable.
    """

    def __init__(self, *, capture_content: bool, session_id: str | None) -> None:
        self._emitter = TurnEmitter(capture_content=capture_content)
        self._session_id = session_id
        self._text_message_id: str | None = None
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
            if not isinstance(raw, Mapping):
                continue
            kind = raw.get("type")
            properties = raw.get("properties")
            session_id = _session_id(properties)
            if session_id is not None:
                if self._session_id is None:
                    self._session_id = session_id
                elif session_id != self._session_id:
                    continue
            if kind == "session.idle":
                return
            if kind == "session.error":
                raise OpenCodeSessionError(_session_error_message(properties))
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
        # tool_start copies that message into parentMessageId.
        events.extend(self._emitter.end_text())
        self._text_message_id = None
        for message_id, parts in list(self._held_tools.items()):
            if self._roles.get(message_id) is None:
                for part in parts:
                    events.extend(self._emit_tool(part))
        self._held_tools.clear()
        events.extend(self._emitter.fail_open())
        return events

    def _event(self, raw: Mapping[str, Any]) -> list[dict[str, Any]]:
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
        part_id = properties.get("partID")
        if (
            not isinstance(delta, str)
            or not delta
            or not isinstance(message_id, str)
            or not isinstance(part_id, str)
            or not part_id
        ):
            return []
        return self._accept_text(message_id, part_id, "delta", delta)

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
            if "input" not in state or not name:
                return []
            events = self._emitter.tool_start(call_id, name)
            events.extend(self._emitter.tool_end(call_id, arguments=state.get("input")))
            return events
        if status not in ("completed", "error"):
            return []
        opened: list[dict[str, Any]] = []
        if name:
            opened.extend(self._emitter.tool_start(call_id, name))
            opened.extend(self._emitter.tool_end(call_id, arguments=state.get("input")))
        if status == "error":
            opened.extend(self._emitter.tool_result(call_id, failed=True))
        else:
            opened.extend(self._emitter.tool_result(call_id, state.get("output")))
        return opened

    def _message(self, info: Mapping[str, Any]) -> list[dict[str, Any]]:
        message_id = info.get("id")
        role = info.get("role")
        events: list[dict[str, Any]] = []
        completed_time = info.get("time")
        finished = isinstance(completed_time, Mapping) and "completed" in completed_time
        if isinstance(message_id, str) and isinstance(role, str):
            self._roles[message_id] = role
            if role == "assistant":
                late_completion = (
                    finished
                    and self._text_message_id is not None
                    and message_id != self._text_message_id
                )
                # Held parts for an earlier completed message cannot be
                # inserted without merging them into the current message or
                # splitting that message. Drop those stale parts.
                if late_completion:
                    self._held.pop(message_id, None)
                    self._held_tools.pop(message_id, None)
                else:
                    events.extend(self._begin_message(message_id))
                    events.extend(self._flush_held(message_id))
                    for part in self._held_tools.pop(message_id, []):
                        events.extend(self._emit_tool(part))
            else:
                self._held.pop(message_id, None)
                self._held_tools.pop(message_id, None)
        if role != "assistant" or not isinstance(message_id, str) or not finished:
            return events
        if message_id == self._text_message_id:
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


def _session_id(properties: object) -> str | None:
    if not isinstance(properties, Mapping):
        return None
    session_id = properties.get("sessionID")
    if isinstance(session_id, str) and session_id:
        return session_id
    return None


def _session_error_message(properties: object) -> str:
    name = None
    if isinstance(properties, Mapping):
        error = properties.get("error")
        if isinstance(error, Mapping):
            raw_name = error.get("name")
            if isinstance(raw_name, str) and raw_name:
                name = raw_name
    if name is None:
        return "The OpenCode session failed."
    return f"The OpenCode session failed ({name})."


def _input_messages(
    user_input: str | None,
    history: list[dict[str, Any]] | None,
) -> list[SpanMessage]:
    messages: list[SpanMessage] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if isinstance(role, str) and role and isinstance(content, str):
            messages.append(text_message(role, content))
    if isinstance(user_input, str):
        messages.append(text_message("user", user_input))
    return messages


def _open_events(
    events: Events,
    config: AiConfigRep,
    user_input: str | None,
    variables: dict[str, Any] | None,
    history: list[dict[str, Any]] | None,
    session_id: str | None,
) -> AsyncIterable[Mapping[str, Any]]:
    produced = events(config, user_input, variables, history, session_id)
    if not hasattr(produced, "__aiter__"):
        raise TypeError("OpenCode event source must return an async iterable.")
    return produced


def create_opencode_handler(
    *,
    events: Events,
    session_id: str | None = None,
    capture_content: bool | None = None,
) -> ProviderHandler:
    """Return a handler that runs an OpenCode session supplied by *events*.

    ``provides_for`` is ``('*', 'agent')``, so the handler matches an agent-mode
    variation for any provider. *events* is called on every invocation as
    ``events(config, user_input, variables, history, session_id)`` and must
    return a fresh async iterable of that session's bus events. This function
    does not start OpenCode and does not construct an OpenCode SDK client.

    The stream yields the AG-UI text and tool subset and one ``done``.
    ``config().stream()`` still yields ``chunk`` then ``done``. Pass
    ``skip_judges=True`` to ``config()``: a judge would otherwise be routed
    here and *events* would run again on the judge prompt. *capture_content*
    follows the other handlers.
    """
    report_usage("opencode-agents.createOpenCodeHandler", PACKAGE_NAME, __version__)
    return _create_opencode_handler(
        events=events,
        session_id=session_id,
        capture_content=capture_content,
    )


def opencode_agents(
    config_key: str,
    user_input: str,
    context: dict[str, Any],
    **kwargs: Any,
) -> Any:
    """Convenience wrapper: creates a handler and calls ``config(...).invoke()``.

    *events* is required and is the same callable :func:`create_opencode_handler`
    takes. *variables*, *history*, *session_id*, and *capture_content* configure
    that call. Pass ``skip_judges=True`` so a judged config does not run
    *events* again.
    """
    report_usage("opencode-agents.opencodeAgents", PACKAGE_NAME, __version__)
    variables = kwargs.pop("variables", None)
    history = kwargs.pop("history", None)
    capture_content = kwargs.pop("capture_content", None)
    session_id = kwargs.pop("session_id", None)
    events = kwargs.pop("events")
    return _config(
        key=config_key,
        handler=_create_opencode_handler(
            events=events,
            session_id=session_id,
            capture_content=capture_content,
        ),
        **kwargs,
    ).invoke(user_input, context, variables, history)


def _create_opencode_handler(
    *,
    events: Events,
    session_id: str | None,
    capture_content: bool | None,
) -> ProviderHandler:
    content_override = capture_content
    bound_session = session_id

    async def _stream_impl(
        config: AiConfigRep,
        user_input: str | None,
        tool_handlers: dict[str, Any] | None,
        variables: dict[str, Any] | None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        capture = effective_capture_content(content_override)
        span = start_root_span(config, variables or {})
        ended: set[int] = set()
        reducer = _Reducer(capture_content=capture, session_id=bound_session)
        try:
            instructions = (
                config.get("instructions") if isinstance(config, dict) else None
            )
            set_input_content_attributes(
                span,
                capture,
                system_instructions=instructions
                if isinstance(instructions, str)
                else None,
                messages=_input_messages(user_input, history),
            )
            async for event in reducer.reduce(
                _open_events(
                    events, config, user_input, variables, history, bound_session
                )
            ):
                yield event
            for event in reducer.close():
                yield event
            output = "".join(reducer.deltas)
            if output:
                set_output_content_attributes(
                    span, capture, [text_message("assistant", output)]
                )
            usage = reducer.usage_payload()
            finish_root_span(
                span,
                config,
                add_cached_tokens_to_input(usage) if usage is not None else SpanUsage(),
            )
            mark_ok(span)
            done: dict[str, Any] = {"type": "done", "output": output}
            if usage is not None:
                done["usage"] = usage
            yield done
        except Exception as exc:
            fail_span(span, exc, ended)
            raise
        finally:
            end_span_once(span, ended)

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
