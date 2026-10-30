"""Reduce OpenCode session events into handler stream events.

``async with handler`` starts ``opencode serve`` and waits until it is healthy.
The process stops when that context exits, after ``config().stream()`` returns,
including when the caller times out or closes the stream. The handler yields
the AG-UI text and tool subset, then one ``done``. ``config().stream()`` still
yields ``chunk`` then ``done``.
"""

from __future__ import annotations

import asyncio
import os
import secrets
from collections.abc import AsyncGenerator, AsyncIterable, Callable, Mapping
from contextlib import aclosing
from typing import Any

from launchdarkly_ai_server import (
    NATIVE_TOOL_KEY,
    AiConfigRep,
    NativeTool,
    ProviderHandler,
    SpanMessage,
    SpanUsage,
    _config,
    add_cached_tokens_to_input,
    compose_history,
    content_to_text,
    end_span_once,
    parse_template,
    set_input_content_attributes,
    set_output_content_attributes,
    set_tool_call_content_attributes,
    text_message,
)
from launchdarkly_ai_server.agui import TurnEmitter
from launchdarkly_ai_server.lifecycle import effective_capture_content
from launchdarkly_ai_server.parameter_forwarding import select_forwarded_parameters
from launchdarkly_ai_server.sdk_usage import report_usage
from launchdarkly_ai_server.utils import model_parameters, number_or_zero

from ._version import PACKAGE_NAME, __version__
from .process import Command, OpenCodeServe, serve_from
from .server import (
    MESSAGE_TIMEOUT_SECONDS,
    ModelResolver,
    default_model,
    disconnect_mcp_bridge,
    register_mcp_bridge,
    session_events,
)
from .spans import (
    abandon_open_spans,
    close_open_spans,
    fail_span,
    finish_model_span,
    finish_root_span,
    mark_ok,
    parent_context_of,
    start_model_span,
    start_root_span,
    start_tool_span,
    succeed_span,
)
from .tool_bridge import ToolBridge, bridge_tool_name

#: The ``model.parameters`` keys forwarded into the ``POST /session/{id}/message`` body. None are.
#: The message route takes no sampling, length, stop, seed, tool-choice or reasoning field: OpenCode
#: reads those from its own server, agent and provider config, which the handler does not control
#: per call. The fields the route does take are not forwarded, and why:
#: * ``parts``, ``model``, ``system``, ``agent``, ``tools``: handler-owned. ``agent`` is the
#:   ``create_opencode_handler(agent=...)`` argument and also picks the server's tools and
#:   permissions.
#: * ``format``: switches where the answer lands (``AssistantMessage.structured``), which the
#:   reducer does not read.
#: * ``noReply``: runtime wiring. No assistant turn is produced.
#: * ``messageID``: server-side session state.
#: * ``variant``: names a preset from the server's variant map, which depends on the provider,
#:   the model and the operator's config, and an unknown name is silently ignored.
_PROMPT_FORWARDED_KEYS: frozenset[str] = frozenset()


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

    Events whose ``sessionID`` is not *session_id* are ignored. A missing
    ``sessionID`` is kept. The reducer does not adopt the first session it
    sees. A ``session.idle`` for this session ends the iterable.

    Also opens and closes this run's ``chat`` and ``execute_tool`` spans, at
    the same two places text and tool events are already derived from the
    bus: one ``chat`` span per assistant message id, one ``execute_tool``
    span per tool call id. *open_tool_spans* and *open_model_spans* are
    owned by the caller, so a teardown outside this reducer (abandonment,
    the run raising) can still find and close whatever is left open.
    *tool_display_names* maps the name OpenCode reports for a tool part
    (a native tool's own id, or a bridged tool's composed MCP name) back to
    the AI Config key a span should show; an unmapped name is shown as-is.
    *tool_native_stubs* holds the zero-arg ``$ld:ai:tool_call`` tracking stub
    for a ``NativeTool`` entry, called once per call id when its span opens —
    a bridged tool's own tracked wrapper already does this when the bridge
    calls it, so it has no entry here.
    """

    def __init__(
        self,
        *,
        capture_content: bool,
        session_id: str,
        config: AiConfigRep,
        parent: Any,
        open_tool_spans: dict[str, Any],
        open_model_spans: dict[str, Any],
        tool_display_names: Mapping[str, str],
        tool_native_stubs: Mapping[str, Callable[[], None]],
    ) -> None:
        self._emitter = TurnEmitter(capture_content=capture_content)
        self._capture = capture_content
        self._session_id = session_id
        self._config = config
        self._parent = parent
        self._tool_spans = open_tool_spans
        self._model_spans = open_model_spans
        self._tool_display_names = tool_display_names
        self._tool_native_stubs = tool_native_stubs
        self._started: set[str] = set()
        self._text_message_id: str | None = None
        self._text_open = False
        self._held_tools: dict[str, list[Mapping[str, Any]]] = {}
        self._deltas: list[str] = []
        self._roles: dict[str, str] = {}
        self._held: dict[str, list[tuple[str, str, str]]] = {}
        self._parked: list[
            tuple[str, list[tuple[str, str, str]], list[Mapping[str, Any]]]
        ] = []
        self._part_text: dict[str, str] = {}
        self._usage_ids: set[str] = set()
        # Closed chat spans, including completions that carried no tokens.
        # _usage_ids only records a message once tokens exist, so it cannot
        # be the guard against opening a second chat for the same id.
        self._closed_chat_ids: set[str] = set()
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
            session_id = _session_id(raw)
            if session_id is not None and session_id != self._session_id:
                continue
            if kind == "session.idle":
                if session_id == self._session_id:
                    return
                continue
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
        if self._text_open and message_id != self._text_message_id:
            self._held.setdefault(message_id, []).append((kind, part_id, text))
            return []
        events = self._begin_message(message_id)
        events.extend(self._flush_held(message_id))
        events.extend(self._apply_text(part_id, kind, text))
        if self._text_open and self._text_message_id == message_id:
            events.extend(self._release_tools(message_id))
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
        events = self._end_open_text()
        self._text_message_id = message_id
        return events

    def _end_open_text(self) -> list[dict[str, Any]]:
        events = self._emitter.end_text()
        self._text_message_id = None
        self._text_open = False
        events.extend(self._flush_parked())
        return events

    def _flush_parked(self) -> list[dict[str, Any]]:
        """Emit messages completed while other text was open, after that text ends."""
        parked = self._parked
        self._parked = []
        events: list[dict[str, Any]] = []
        for message_id, held, tools in parked:
            events.extend(self._emit_parked(message_id, held, tools))
        return events

    def _emit_parked(
        self,
        message_id: str,
        held: list[tuple[str, str, str]],
        tools: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if held:
            events.extend(self._begin_message(message_id))
            for kind, part_id, text in held:
                events.extend(self._apply_text(part_id, kind, text))
        if self._text_open and self._text_message_id == message_id:
            for part in tools:
                events.extend(self._emit_tool(part))
            events.extend(self._emitter.end_text())
            self._text_message_id = None
            self._text_open = False
            return events
        for part in tools:
            events.extend(self._emit_tool(part))
        if self._text_message_id == message_id:
            self._text_message_id = None
        return events

    def _release_tools(self, message_id: str) -> list[dict[str, Any]]:
        parts = self._held_tools.pop(message_id, [])
        events: list[dict[str, Any]] = []
        for part in parts:
            events.extend(self._emit_tool(part))
        return events

    def close(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        # End the open assistant message before leftover tools. tool_start
        # copies that message into parentMessageId. Then apply every held
        # assistant call, including one that completed after the text closed,
        # before fail_open marks a started call as a failure.
        events.extend(self._end_open_text())
        for message_id, held in list(self._held.items()):
            if self._roles.get(message_id) != "assistant" or not held:
                continue
            tools = self._held_tools.pop(message_id, [])
            events.extend(self._emit_parked(message_id, held, tools))
            self._held.pop(message_id, None)
        for message_id, parts in list(self._held_tools.items()):
            role = self._roles.get(message_id)
            if role is None or role == "assistant":
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
        events = self._emitter.text(text)
        if events:
            self._text_open = True
        return events

    def _tool(self, part: Mapping[str, Any]) -> list[dict[str, Any]]:
        message_id = part.get("messageID")
        if not isinstance(message_id, str):
            self._held_tools.setdefault("", []).append(part)
            return []
        role = self._roles.get(message_id)
        if role is None:
            self._held_tools.setdefault(message_id, []).append(part)
            return []
        if role != "assistant":
            return []
        raw_id = part.get("callID") or part.get("id") or ""
        call_id = raw_id if isinstance(raw_id, str) else ""
        if call_id and call_id in self._started:
            return self._emit_tool(part)
        if self._text_open and self._text_message_id == message_id:
            return self._emit_tool(part)
        # Another message is streaming, or this message's text has not opened
        # yet. Emitting now would parent the call to the wrong message.
        if self._text_open or self._text_message_id == message_id:
            self._held_tools.setdefault(message_id, []).append(part)
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
            self._open_tool_span(call_id, name, state.get("input"))
            events = self._emitter.tool_start(call_id, name)
            if any(event.get("type") == "TOOL_CALL_START" for event in events):
                self._started.add(call_id)
            events.extend(self._emitter.tool_end(call_id, arguments=state.get("input")))
            return events
        if status not in ("completed", "error"):
            return []
        if name:
            # A call that skipped straight to completed/error without a
            # recorded `running` state still needs its span opened once,
            # with whatever input this status carries.
            self._open_tool_span(call_id, name, state.get("input"))
        opened: list[dict[str, Any]] = []
        if name:
            started = self._emitter.tool_start(call_id, name)
            if any(event.get("type") == "TOOL_CALL_START" for event in started):
                self._started.add(call_id)
            opened.extend(started)
            opened.extend(self._emitter.tool_end(call_id, arguments=state.get("input")))
        if status == "error":
            self._close_tool_span(call_id, failed=True, result=None)
            opened.extend(self._emitter.tool_result(call_id, failed=True))
        else:
            output = state.get("output")
            self._close_tool_span(call_id, failed=False, result=output)
            opened.extend(self._emitter.tool_result(call_id, output))
        return opened

    def _open_tool_span(self, call_id: str, provider_name: str, arguments: Any) -> None:
        """Opens one ``execute_tool`` span for *call_id*, once.

        Also fires a ``NativeTool``'s tracking stub, the first and only time
        its span opens — mirroring the Claude handler's ``PreToolUse`` hook,
        which calls the same kind of stub when it opens its own tool span.
        A bridged (callable) tool has no stub here: its tracked wrapper
        already emits ``$ld:ai:tool_call`` when the bridge invokes it.
        """
        if call_id in self._tool_spans:
            return
        stub = self._tool_native_stubs.get(provider_name)
        if stub is not None:
            stub()
        display_name = self._tool_display_names.get(provider_name, provider_name)
        span = start_tool_span(display_name, call_id, self._parent)
        self._tool_spans[call_id] = span
        set_tool_call_content_attributes(span, self._capture, arguments=arguments)

    def _close_tool_span(self, call_id: str, *, failed: bool, result: Any) -> None:
        span = self._tool_spans.pop(call_id, None)
        if span is None:
            return
        if failed:
            fail_span(span, OpenCodeSessionError(f"tool call {call_id} failed"))
            return
        set_tool_call_content_attributes(span, self._capture, result=result)
        succeed_span(span)

    def _message(self, info: Mapping[str, Any]) -> list[dict[str, Any]]:
        message_id = info.get("id")
        role = info.get("role")
        events: list[dict[str, Any]] = []
        completed_time = info.get("time")
        finished = isinstance(completed_time, Mapping) and "completed" in completed_time
        if isinstance(message_id, str) and isinstance(role, str):
            self._roles[message_id] = role
            if role == "assistant":
                # A completed message.updated closes this chat span and
                # removes it from _model_spans. A later completion for the
                # same id, including one with no tokens, must not open
                # another. _usage_ids is not that set: it is filled only
                # when the completion carries a tokens mapping.
                if (
                    message_id not in self._model_spans
                    and message_id not in self._closed_chat_ids
                ):
                    self._model_spans[message_id] = start_model_span(
                        self._config, self._parent
                    )
                # A reserved id is not an open AG-UI message. Only text that
                # TurnEmitter has already yielded can take a parent or be split.
                late_completion = (
                    finished and self._text_open and message_id != self._text_message_id
                )
                if late_completion:
                    # Keep the open message intact. The held text and tools
                    # still happened; emit them once that text ends.
                    held = self._held.pop(message_id, [])
                    tools = self._held_tools.pop(message_id, [])
                    if held or tools:
                        self._parked.append((message_id, held, tools))
                else:
                    events.extend(self._begin_message(message_id))
                    events.extend(self._flush_held(message_id))
                    if self._text_open and self._text_message_id == message_id:
                        events.extend(self._release_tools(message_id))
                    elif finished:
                        events.extend(self._release_tools(message_id))
            else:
                self._held.pop(message_id, None)
                self._held_tools.pop(message_id, None)
        if role != "assistant" or not isinstance(message_id, str) or not finished:
            return events
        if self._text_message_id == message_id and self._text_open:
            events.extend(self._end_open_text())
        elif self._text_message_id == message_id:
            self._text_message_id = None
        if message_id not in self._usage_ids:
            tokens = info.get("tokens")
            if isinstance(tokens, Mapping):
                self._add_usage(tokens)
                self._usage_ids.add(message_id)
        self._close_model_span(message_id, info.get("tokens"))
        self._closed_chat_ids.add(message_id)
        return events

    def _close_model_span(
        self, message_id: str, tokens: Mapping[str, Any] | None
    ) -> None:
        span = self._model_spans.pop(message_id, None)
        if span is None:
            return
        finish_model_span(span, self._config, _message_usage(tokens))

    def _add_usage(self, tokens: Mapping[str, Any]) -> None:
        self._saw_usage = True
        self._input += number_or_zero(tokens.get("input"))
        self._output += number_or_zero(tokens.get("output"))
        cache = tokens.get("cache")
        if isinstance(cache, Mapping):
            self._saw_cache = True
            self._cache_read += number_or_zero(cache.get("read"))
            self._cache_write += number_or_zero(cache.get("write"))


def _message_usage(tokens: Any) -> SpanUsage:
    """One assistant message's own token counts, as a ``chat`` span wants them.

    Separate from ``_Reducer._add_usage``, which accumulates the run's total
    for the ``done`` event: a ``chat`` span reports the one message it
    belongs to, not the run so far.
    """
    if not isinstance(tokens, Mapping):
        return SpanUsage()
    cache = tokens.get("cache")
    cache = cache if isinstance(cache, Mapping) else {}
    return SpanUsage(
        input=number_or_zero(tokens.get("input")),
        output=number_or_zero(tokens.get("output")),
        cache_read=number_or_zero(cache.get("read")),
        cache_creation=number_or_zero(cache.get("write")),
    )


def _string_id(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _session_id(raw: object) -> str | None:
    """Session id from a bus event.

    Deltas carry it on ``properties``. ``message.updated`` carries it on
    ``properties.info``, and ``message.part.updated`` on ``properties.part``.
    """
    if not isinstance(raw, Mapping):
        return None
    properties = raw.get("properties")
    if not isinstance(properties, Mapping):
        return _string_id(raw.get("sessionID"))
    found = _string_id(properties.get("sessionID"))
    if found is not None:
        return found
    info = properties.get("info")
    if isinstance(info, Mapping):
        found = _string_id(info.get("sessionID"))
        if found is not None:
            return found
    part = properties.get("part")
    if isinstance(part, Mapping):
        return _string_id(part.get("sessionID"))
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
    """History and *user_input* as span messages.

    Block-array history is flattened the same way as the other handlers.
    System turns stay off this list: they belong on the system instructions.
    An empty *user_input* adds nothing, so a history turn that already holds
    the user message is not followed by a blank one.
    """
    items = [item for item in (history or []) if isinstance(item, dict)]
    messages: list[SpanMessage] = []
    for item in compose_history(history=items, user_input=user_input):
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not role:
            continue
        if isinstance(content, str):
            messages.append(text_message(role, content))
            continue
        if isinstance(content, list):
            text = content_to_text(
                [block for block in content if isinstance(block, dict)]
            )
            if text:
                messages.append(text_message(role, text))
    return messages


def create_opencode_handler(
    *,
    cwd: str | os.PathLike[str],
    config: str | os.PathLike[str],
    env: Mapping[str, str],
    capture_content: bool | None = None,
    model: ModelResolver | None = None,
    agent: str | None = None,
    command: Command = None,
) -> ProviderHandler:
    """Return a handler that starts ``opencode serve`` for each call.

    ``provides_for`` is ``('*', 'agent')``. *cwd* is the working directory,
    *config* is the OpenCode config file, and *env* is the child environment.
    The handler sets ``OPENCODE_CONFIG`` from *config*,
    ``OPENCODE_SERVER_USERNAME`` to ``opencode``, and
    ``OPENCODE_SERVER_PASSWORD`` to a generated secret, replacing either value
    in *env*. *command* is the executable, ``opencode`` when omitted. A test
    passes a fake executable here. There is no URL, client, or transport
    argument.

    Entering the handler as an async context starts the process and returns
    only once it is healthy. Exiting waits until the process has stopped, so a
    caller can remove *cwd* after the context returns. A wall-clock budget
    belongs around ``config().stream()``, not around enter or exit. Without
    the context, one call starts and stops its own process. A judge routed
    here runs as a full OpenCode agent: the server's tools, in *cwd*, apply to
    the judge prompt too. This package does not offer a tool-less judge agent.
    *model* overrides provider and model selection.
    """
    bound = serve_from(cwd=cwd, config=config, env=env, command=command)
    report_usage("opencode-agents.createOpenCodeHandler", PACKAGE_NAME, __version__)
    return OpenCodeHandler(
        serve=bound,
        capture_content=capture_content,
        model=model or default_model,
        agent=agent,
    )


def opencode_agents(
    config_key: str,
    user_input: str,
    context: dict[str, Any],
    **kwargs: Any,
) -> Any:
    """Convenience wrapper: creates a handler and calls ``config(...).invoke()``."""
    report_usage("opencode-agents.opencodeAgents", PACKAGE_NAME, __version__)
    variables = kwargs.pop("variables", None)
    history = kwargs.pop("history", None)
    capture_content = kwargs.pop("capture_content", None)
    cwd = kwargs.pop("cwd", None)
    config_path = kwargs.pop("config", None)
    env = kwargs.pop("env", None)
    command = kwargs.pop("command", None)
    if cwd is None or config_path is None or env is None:
        raise TypeError(
            "The OpenCode handler requires a working directory, config file, and env."
        )
    model = kwargs.pop("model", None)
    agent = kwargs.pop("agent", None)
    return _config(
        key=config_key,
        handler=OpenCodeHandler(
            serve=serve_from(cwd=cwd, config=config_path, env=env, command=command),
            capture_content=capture_content,
            model=model or default_model,
            agent=agent,
        ),
        **kwargs,
    ).invoke(user_input, context, variables, history)


class _CallableEntry:
    __slots__ = ("description", "fn", "parameters")

    def __init__(
        self, description: str, parameters: Mapping[str, Any], fn: Callable[[Any], Any]
    ) -> None:
        self.description = description
        self.parameters = parameters
        self.fn = fn


class _ToolPlan:
    """What one call offers OpenCode, derived from ``config["tools"]`` and ``tool_handlers``.

    *offered_names* and *tool_display_names* start populated with every
    ``NativeTool`` entry. The handler appends a bridged (callable) tool's
    composed MCP name to both once the bridge, if any, has started and that
    name is known — a bridged tool's model-visible name depends on the
    per-call bridge's generated server name, which does not exist yet when
    this plan is built.
    """

    def __init__(self) -> None:
        self.callables: dict[str, _CallableEntry] = {}
        self.native_stubs: dict[str, Callable[[], None]] = {}
        self.offered_names: list[str] = []
        self.tool_display_names: dict[str, str] = {}


def _plan_tools(
    config: AiConfigRep, tool_handlers: Mapping[str, Any] | None
) -> _ToolPlan:
    """Partitions the offered tools into native pass-throughs and bridged callables.

    "Offered" is the request-scoped authorization boundary: a config tool key
    with no entry in *tool_handlers* is not served, and a *tool_handlers*
    entry whose key the config does not declare is not served either. See
    AGENTS.md's tool-handling contract and the matching rule already stated
    for the JS SDK's handlers.
    """
    plan = _ToolPlan()
    config_tools = config.get("tools") if isinstance(config, dict) else None
    config_tools = config_tools if isinstance(config_tools, dict) else {}
    handlers = tool_handlers or {}
    for key, tool_cfg in config_tools.items():
        stub = handlers.get(key)
        if stub is None:
            continue
        native = getattr(stub, NATIVE_TOOL_KEY, None)
        if native is not None:
            provider_name = native.tool_name
            plan.offered_names.append(provider_name)
            plan.tool_display_names[provider_name] = key
            plan.native_stubs[provider_name] = stub
            continue
        cfg = tool_cfg if isinstance(tool_cfg, dict) else {}
        description = cfg.get("description")
        parameters = cfg.get("parameters")
        plan.callables[key] = _CallableEntry(
            description=description if isinstance(description, str) else "",
            parameters=parameters if isinstance(parameters, dict) else {},
            fn=stub,
        )
    return plan


#: ``Permission.disabled`` remaps these tool ids onto ``edit`` before
#: ``findLast``. An allow rule whose permission is ``write`` or
#: ``apply_patch`` never matches. An allow for ``edit`` enables all three.
_EDIT_FAMILY = frozenset({"edit", "write", "apply_patch"})


def _permission_for(name: str) -> str:
    """The permission OpenCode actually matches for one offered tool id."""
    if name in _EDIT_FAMILY:
        return "edit"
    return name


def _build_tool_ruleset(offered_names: list[str]) -> list[dict[str, str]]:
    """OpenCode's session-level ``permission`` ruleset for one call's offered tools.

    The first rule denies every tool. ``Permission.disabled`` resolves the
    *last* matching rule (``findLast`` over the agent's static permission
    followed by this ruleset), so each later per-name allow rule takes
    precedence for the tool it names and nothing else gains visibility by
    accident — and this ruleset, being appended last, always wins over the
    opencode.json agent config's own static permission. An empty
    *offered_names* still sends the deny-all rule: a variation with no tools
    gets a model with no tools, not the host's default catalog.

    ``write`` and ``apply_patch`` are not sent as their own permissions.
    OpenCode remaps both onto ``edit`` before matching, so any of the three
    emits one ``edit`` allow, and that one grant covers all three tools.
    """
    ruleset = [{"permission": "*", "action": "deny", "pattern": "*"}]
    sent: set[str] = set()
    for name in offered_names:
        permission = _permission_for(name)
        if permission in sent:
            continue
        sent.add(permission)
        ruleset.append({"permission": permission, "action": "allow", "pattern": "*"})
    return ruleset


class OpenCodeHandler(ProviderHandler):
    """Handler whose async context owns one ``opencode serve`` process."""

    def __init__(
        self,
        *,
        serve: OpenCodeServe,
        capture_content: bool | None,
        model: ModelResolver,
        agent: str | None,
    ) -> None:
        self._serve = serve
        self._model = model
        self._agent = agent
        self._active: OpenCodeServe | None = None
        self._base_url: str | None = None
        self._password: str | None = None
        self._depth = 0
        self._lock = asyncio.Lock()
        super().__init__(
            fn=self._call_impl,
            provides_for=("*", "agent"),
            stream_fn=self._stream_impl,
            capture_content=capture_content,
        )

    async def __aenter__(self) -> OpenCodeHandler:
        """Start ``opencode serve`` and return once it answers on loopback.

        Concurrent entries share that process. The lock covers startup so a
        second entry cannot replace ``_active`` and leave the first process
        running.
        """
        async with self._lock:
            if self._active is None:
                active = self._serve.spawn()
                self._base_url = await active.start()
                self._password = active.password
                self._active = active
            self._depth += 1
            return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        """Stop the process and wait until it has exited."""
        async with self._lock:
            self._depth -= 1
            if self._depth > 0:
                return
            active = self._active
            self._active = None
            self._base_url = None
            self._password = None
            if active is not None:
                await active.stop()

    async def _stream_impl(
        self,
        config: AiConfigRep,
        user_input: str | None,
        tool_handlers: Mapping[str, Callable[..., Any] | NativeTool] | None,
        variables: dict[str, Any] | None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        capture = effective_capture_content(self._capture_content_override)
        span = start_root_span(config, variables or {}, capture_content=capture)
        parent = parent_context_of(span)
        ended: set[int] = set()
        open_tool_spans: dict[str, Any] = {}
        open_model_spans: dict[str, Any] = {}
        # GeneratorExit is a consumer who stopped reading (abandoned).
        # CancelledError is a timeout or task.cancel(); except Exception
        # does not see it, so the finally would otherwise stamp
        # launchdarkly.stream.abandoned. cancelled wins in end_span_once.
        cancelled = False
        provider_id, model_id = self._model(config if isinstance(config, dict) else {})
        system = _system_text(config, variables)
        prompt = _user_prompt(config, user_input, history, variables)
        plan = _plan_tools(config, tool_handlers)
        # A context-owned process outlives this stream, so a caller timeout
        # that cancels the stream cannot cancel its shutdown.
        owned: OpenCodeServe | None = None
        if (
            self._active is not None
            and self._base_url is not None
            and self._password is not None
        ):
            base_url = self._base_url
            password = self._password
        else:
            owned = self._serve.spawn()
            base_url = ""
            password = ""
        bridge: ToolBridge | None = None
        bridge_name: str | None = None
        try:
            set_input_content_attributes(
                span,
                capture,
                system_instructions=system,
                messages=_input_messages(user_input, history),
            )
            if owned is not None:
                base_url = await owned.start()
                password = owned.password
            if plan.callables:
                bridge = ToolBridge()
                for key, entry in plan.callables.items():
                    bridge.register(
                        key,
                        description=entry.description,
                        parameters=entry.parameters,
                        fn=entry.fn,
                    )
                await bridge.start()
                run_id = secrets.token_hex(8)
                bridge_name = f"ld_{run_id}"
                await register_mcp_bridge(
                    base_url=base_url,
                    password=password,
                    name=bridge_name,
                    url=f"{bridge.base_url}/mcp/{run_id}",
                    token=bridge.token,
                    timeout_ms=int(MESSAGE_TIMEOUT_SECONDS * 1000),
                )
                for key in plan.callables:
                    composed = bridge_tool_name(bridge_name, key)
                    plan.offered_names.append(composed)
                    plan.tool_display_names[composed] = key
            permission = _build_tool_ruleset(plan.offered_names)
            created = _Created()
            async with aclosing(
                _reduce_session(
                    created,
                    base_url=base_url,
                    password=password,
                    prompt=prompt,
                    system=system,
                    provider_id=provider_id,
                    model_id=model_id,
                    agent=self._agent,
                    capture=capture,
                    parameters=select_forwarded_parameters(
                        model_parameters(config), _PROMPT_FORWARDED_KEYS
                    ),
                    permission=permission,
                    config=config,
                    parent=parent,
                    open_tool_spans=open_tool_spans,
                    open_model_spans=open_model_spans,
                    tool_display_names=plan.tool_display_names,
                    tool_native_stubs=plan.native_stubs,
                )
            ) as reduced:
                async for event in reduced:
                    yield event
            output = "".join(created.deltas)
            if output:
                set_output_content_attributes(
                    span, capture, [text_message("assistant", output)]
                )
            usage = created.usage
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
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:
            close_open_spans(open_tool_spans, open_model_spans, exc)
            fail_span(span, exc, ended)
            raise
        finally:
            if bridge_name is not None:
                await disconnect_mcp_bridge(
                    base_url=base_url, password=password, name=bridge_name
                )
            if bridge is not None:
                await bridge.stop()
            if owned is not None:
                await owned.stop()
            abandon_open_spans(
                open_tool_spans, open_model_spans, ended, cancelled=cancelled
            )
            end_span_once(span, ended, cancelled=cancelled)

    async def _call_impl(
        self,
        config: AiConfigRep,
        user_input: str | None,
        tool_handlers: Mapping[str, Callable[..., Any] | NativeTool] | None,
        variables: dict[str, Any] | None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        output = ""
        usage: dict[str, Any] = {}
        async with aclosing(
            self._stream_impl(config, user_input, tool_handlers, variables, history)
        ) as stream:
            async for event in stream:
                if event.get("type") != "done":
                    continue
                raw_output = event.get("output")
                output = raw_output if isinstance(raw_output, str) else ""
                raw_usage = event.get("usage")
                usage = raw_usage if isinstance(raw_usage, dict) else {}
        return {"output": output, "usage": usage}


class _Created:
    deltas: list[str]
    usage: dict[str, int] | None

    def __init__(self) -> None:
        self.deltas = []
        self.usage = None


async def _reduce_session(
    created: _Created,
    *,
    base_url: str,
    password: str,
    prompt: str,
    system: str | None,
    provider_id: str,
    model_id: str,
    agent: str | None,
    capture: bool,
    parameters: Mapping[str, Any],
    permission: list[dict[str, str]] | None,
    config: AiConfigRep,
    parent: Any,
    open_tool_spans: dict[str, Any],
    open_model_spans: dict[str, Any],
    tool_display_names: Mapping[str, str],
    tool_native_stubs: Mapping[str, Callable[[], None]],
) -> AsyncGenerator[dict[str, Any], None]:
    """Reduce the session ``POST /session`` created. The reducer is unchanged."""
    session_ids: list[str] = []
    reducer: _Reducer | None = None
    async with aclosing(
        session_events(
            base_url=base_url,
            password=password,
            parameters=parameters,
            prompt=prompt,
            system=system,
            provider_id=provider_id,
            model_id=model_id,
            agent=agent,
            session_ids=session_ids,
            permission=permission,
        )
    ) as raw_events:
        async for raw in raw_events:
            if reducer is None:
                reducer = _Reducer(
                    capture_content=capture,
                    session_id=session_ids[0],
                    config=config,
                    parent=parent,
                    open_tool_spans=open_tool_spans,
                    open_model_spans=open_model_spans,
                    tool_display_names=tool_display_names,
                    tool_native_stubs=tool_native_stubs,
                )
            async with aclosing(reducer.reduce(_once(raw))) as reduced:
                async for event in reduced:
                    yield event
    if reducer is None:
        return
    for event in reducer.close():
        yield event
    created.deltas = list(reducer.deltas)
    created.usage = reducer.usage_payload()


async def _once(raw: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
    yield raw


def _system_text(config: AiConfigRep, variables: dict[str, Any] | None) -> str | None:
    """Agent instructions, with ``{{variables}}`` rendered.

    History and the user turn are not rendered. A judge prompt is rendered
    the same way, because a judge uses this handler.
    """
    if not isinstance(config, dict):
        return None
    instructions = config.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        return parse_template(instructions, variables or {})
    return None


def _user_prompt(
    config: AiConfigRep,
    user_input: str | None,
    history: list[dict[str, Any]] | None,
    variables: dict[str, Any] | None = None,
) -> str:
    values = variables or {}
    messages: list[dict[str, Any]] = []
    if isinstance(config, dict):
        raw_messages = config.get("messages")
        if isinstance(raw_messages, list):
            for message in raw_messages:
                if not isinstance(message, dict) or message.get("role") == "system":
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    message = {**message, "content": parse_template(content, values)}
                messages.append(message)
    turns = compose_history(
        history=[item for item in (history or []) if isinstance(item, dict)],
        user_input=user_input,
        config_messages=messages,
    )
    lines: list[str] = []
    for turn in turns:
        role = turn.get("role")
        text = content_to_text(turn.get("content") or "")
        if not text:
            continue
        label = "Assistant" if role == "assistant" else "User"
        lines.append(f"{label}: {text}")
    if (
        len(lines) == 1
        and lines[0].startswith("User: ")
        and not messages
        and not history
    ):
        return lines[0].removeprefix("User: ")
    return "\n\n".join(lines)
