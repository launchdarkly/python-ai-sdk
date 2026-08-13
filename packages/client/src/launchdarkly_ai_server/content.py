"""Conversation content on spans, per LaunchDarkly's "Richer LLM spans" proposal.

Two rules drive everything in this module.

**Attributes, not events.** Canonical content lives on span attributes. OTEP 4430 deprecated the
span-event recording API, and LaunchDarkly's ingest does not normalise content events into the
canonical shape, so a ``gen_ai.content.prompt`` event, which is what these handlers emitted
before, is read by nothing on the LaunchDarkly side.

**Off by default.** Everything here is conversation content, which is PII. A handler opts in with
``capture_content=True``; every function below takes that decision as its ``capture`` argument and
returns without writing when it is false. Passing the flag rather than reading ambient state keeps
the gate visible at each call site.

Handlers also test the same flag themselves before building the argument, so the check appears
twice on purpose: the guard here is the safety net that makes a forgotten call site harmless, and
the guard there avoids walking a conversation and serialising JSON that would then be discarded,
once per model turn, inside a loop.

Two carriers are written for the same content, deliberately:

* ``gen_ai.input.messages`` / ``gen_ai.output.messages`` / ``gen_ai.system_instructions`` /
  ``gen_ai.tool.definitions`` are the canonical JSON shape from the OTel GenAI semantic
  conventions, and the shape the proposal makes normative.
* ``gen_ai.prompt.{i}.role|content`` / ``gen_ai.completion.{i}.role|content`` are the OpenLLMetry
  shape, one numbered attribute per field. LaunchDarkly's LLM trace view and conversation view
  read *only* this one today, so canonical attributes alone would render as an empty transcript.

A third carrier is written alongside them: the ``gen_ai.content.prompt`` /
``gen_ai.content.completion`` span events. They are redundant today, but they are what every
published version of these handlers has emitted, and removing them would silently break any
consumer that learned to read them. They are written from the same messages and behind the same
gate as everything else here, so the three carriers cannot disagree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

# ─── Message shapes ──────────────────────────────────────────────────────────


@dataclass
class SpanMessagePart:
    """One typed piece of a message, mirroring the ``parts`` union in the GenAI JSON schemas.

    ``type`` selects which of the remaining fields carry meaning:

    * ``text`` and ``reasoning`` use ``content``.
    * ``tool_call`` uses ``name``, and optionally ``id`` and ``arguments``.
    * ``tool_call_response`` uses ``result``, and optionally ``id``.

    One dataclass rather than a union of four, because mypy's strict mode makes a tagged union of
    dataclasses awkward to narrow at the call sites in the handler packages, and the JSON shape is
    the contract rather than the Python type.
    """

    type: Literal["text", "reasoning", "tool_call", "tool_call_response"]
    content: str | None = None
    id: str | None = None
    name: str | None = None
    arguments: Any = None
    result: Any = None

    def to_canonical(self) -> dict[str, Any]:
        """The canonical JSON form: only the members this part's type actually uses."""
        out: dict[str, Any] = {"type": self.type}
        if self.type in ("text", "reasoning"):
            out["content"] = self.content or ""
            return out
        if self.type == "tool_call":
            if self.id is not None:
                out["id"] = self.id
            out["name"] = self.name or ""
            if self.arguments is not None:
                out["arguments"] = self.arguments
            return out
        if self.id is not None:
            out["id"] = self.id
        if self.result is not None:
            out["result"] = self.result
        return out

    def to_text(self) -> str:
        """Flattens this part to the string the OpenLLMetry carrier holds.

        Text and reasoning contribute their text. Tool traffic becomes JSON, because OpenLLMetry
        has no place to put structure.

        An absent tool result contributes nothing, so :meth:`SpanMessage.to_text` drops it and every
        carrier agrees with :meth:`to_canonical`, which omits the key. ``json.dumps(None)`` would
        render it as the literal text ``null``, which reads in the trace view as a tool that returned
        a null value rather than one that returned nothing.

        ``None`` is the only spelling of absent here, because :meth:`to_canonical` already treats it
        that way. The TypeScript SDK can tell an absent result from one explicitly returned as null
        and reports the second as ``null``; Python's dataclass cannot hold that distinction, so there
        is nothing to disagree about.
        """
        if self.type in ("text", "reasoning"):
            return self.content or ""
        if self.type == "tool_call":
            return json.dumps({"name": self.name or "", "arguments": self.arguments})
        if self.result is None:
            return ""
        if isinstance(self.result, str):
            return self.result
        return json.dumps(self.result)


@dataclass
class SpanMessage:
    """One turn of the conversation, in the canonical ``{role, parts}`` shape."""

    role: str
    parts: list[SpanMessagePart] = field(default_factory=list)
    #: Output messages only. Why the model stopped producing this message.
    finish_reason: str | None = None

    def to_canonical(self) -> dict[str, Any]:
        """``snake_case`` keys, and no absent members."""
        out: dict[str, Any] = {
            "role": self.role,
            "parts": [p.to_canonical() for p in self.parts],
        }
        if self.finish_reason is not None:
            out["finish_reason"] = self.finish_reason
        return out

    def to_text(self) -> str:
        """The parts joined by newlines, with empty parts dropped."""
        return "\n".join(t for t in (p.to_text() for p in self.parts) if t)


def text_message(role: str, content: str) -> SpanMessage:
    """Convenience for the common case: a whole message that is one block of text."""
    return SpanMessage(role=role, parts=[SpanMessagePart(type="text", content=content)])


@dataclass
class ToolDefinitionInput:
    """One entry of the tool catalog, as it was actually offered to the model.

    Deliberately not the AI Config's own tool type: a handler filters the configured tools down to
    the ones it has a registered implementation for, and it is that filtered set the model saw.
    Recording the unfiltered config would misreport what the model could have called.
    """

    name: str
    description: str | None = None
    parameters: Any = None


# ─── Finish reasons ──────────────────────────────────────────────────────────

#: Every provider spelling this SDK actually serves, mapped onto semconv's vocabulary.
#:
#: The vocabulary is ``stop``, ``length``, ``content_filter``, ``tool_calls`` and ``error``.
#: Anthropic and OpenAI are the only two vendors behind these six handlers, so they are the only
#: two groups here. Keys are compared lower-cased, which costs one ``lower()`` and means a provider
#: that shouts its enum is mapped rather than passed through as a stray value.
#:
#: ``pause_turn`` is deliberately absent. Anthropic returns it when a long-running server-side tool
#: suspends a turn that has not actually finished, and no semconv value means "did not finish", so
#: it falls through to the passthrough below rather than being flattened into ``stop``.
#:
#: The OpenAI rows are load-bearing only for the two LangChain handlers, which can serve an OpenAI
#: model and do read a ``finish_reason`` string. The two OpenAI handlers use the Responses API,
#: which has no such field, and derive the reason themselves. They must not use this table.
_SEMCONV_FINISH_REASONS: dict[str, str] = {
    # Anthropic `stop_reason`
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
    # OpenAI Chat Completions `finish_reason`, mostly already the vocabulary
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
    "function_call": "tool_calls",
}


def to_semconv_finish_reason(raw: str | None) -> str | None:
    """Maps one provider's finish reason onto semconv's ``gen_ai.response.finish_reasons`` vocabulary.

    This SDK used to pass the provider's string through untranslated, on the argument that
    translating ``end_turn`` into ``stop`` loses information. Measuring it settled the argument the
    other way: a single run emits ``chat`` spans from more than one handler, so a consumer grouping
    by finish reason saw ``stop`` and ``end_turn`` as two different outcomes for the same event.

    Nothing is lost. The provider's own wording is still on the span, because the raw response is
    what ``gen_ai.output.messages`` was built from, and an unrecognised reason is passed through
    verbatim rather than dropped or coerced, so a new vendor spelling shows up as itself instead of
    silently becoming ``stop``. That passthrough is the signal to add a row to the table above.
    """
    if not raw or not isinstance(raw, str):
        return None
    return _SEMCONV_FINISH_REASONS.get(raw.lower(), raw)


def lang_chain_finish_reasons(source: Any) -> list[str] | None:
    """Reads the finish reasons out of a LangChain ``LLMResult`` or a single ``AIMessage``.

    Both shapes are accepted because both are what the handlers hold: the agents handler finishes a
    turn from an ``LLMResult``, while the messages handler finishes one from the ``AIMessage`` that
    ``invoke()`` returned.

    LangChain does not normalise the field, so it is read from every place providers put it:
    ``generation_info["finish_reason"]`` (OpenAI) and ``response_metadata["finish_reason"]`` or
    ``["stop_reason"]`` (Anthropic), then mapped onto the semconv vocabulary. LangChain is the one
    place where the same handler can serve either vendor, so it is where an untranslated
    passthrough is least defensible.

    Returns ``None`` rather than an empty list when nothing is present, so a caller leaves the
    attribute off instead of asserting that the turn finished for no reason.
    """
    generations = _get(source, "generations")
    candidates: list[Any]
    if isinstance(generations, list):
        candidates = [item for group in generations for item in _as_list(group)]
    else:
        candidates = [source]

    reasons: list[str] = []
    for candidate in candidates:
        reason = to_semconv_finish_reason(_finish_reason_of(candidate))
        if reason:
            reasons.append(reason)
    return reasons or None


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _get(obj: Any, key: str) -> Any:
    """Reads *key* off a mapping or an object, whichever *obj* is.

    LangChain hands back objects in some paths and dicts in others, and both shapes reach these
    functions.
    """
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _finish_reason_of(candidate: Any) -> str | None:
    """A ``ChatGeneration`` holds the message under ``message``; an ``AIMessage`` is the message."""
    info = _get(candidate, "generation_info") or {}
    message = _get(candidate, "message")
    if message is None:
        message = candidate
    metadata = _get(message, "response_metadata") or {}
    reason = (
        (info.get("finish_reason") if isinstance(info, dict) else None)
        or (metadata.get("finish_reason") if isinstance(metadata, dict) else None)
        or (metadata.get("stop_reason") if isinstance(metadata, dict) else None)
    )
    return reason if isinstance(reason, str) and reason else None


def lang_chain_span_messages(
    messages: list[Any],
) -> tuple[str | None, list[SpanMessage]]:
    """Converts LangChain ``BaseMessage`` values into canonical span messages.

    Returns ``(system_instructions, messages)``, with the system prompt lifted out.

    Shared here rather than copied into the two LangChain packages: both need the exact same
    conversion, and a copy in each package is how the span code in this SDK drifted apart the last
    time. The narrowing is structural, on ``_get_type()``, ``content`` and ``tool_calls``, so the
    client takes no dependency on LangChain.

    LangChain names its roles ``human`` and ``ai``; the semconv vocabulary is ``user`` and
    ``assistant``.
    """
    system: list[str] = []
    converted: list[SpanMessage] = []

    for raw in messages:
        get_type = getattr(raw, "_get_type", None)
        msg_type = str(get_type()) if callable(get_type) else ""
        text = _lang_chain_content_text(_get(raw, "content"))

        if msg_type in ("system", "developer"):
            if text:
                system.append(text)
            continue

        if msg_type == "tool":
            tool_call_id = _get(raw, "tool_call_id")
            converted.append(
                SpanMessage(
                    role="tool",
                    parts=[
                        SpanMessagePart(
                            type="tool_call_response",
                            id=tool_call_id if isinstance(tool_call_id, str) else None,
                            result=text,
                        )
                    ],
                )
            )
            continue

        parts: list[SpanMessagePart] = []
        if text:
            parts.append(SpanMessagePart(type="text", content=text))

        tool_calls = _get(raw, "tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                call_id = _get(call, "id")
                parts.append(
                    SpanMessagePart(
                        type="tool_call",
                        id=call_id if isinstance(call_id, str) else None,
                        name=str(_get(call, "name") or ""),
                        arguments=_get(call, "args"),
                    )
                )

        if msg_type == "human":
            role = "user"
        elif msg_type == "ai":
            role = "assistant"
        else:
            role = msg_type or "user"
        converted.append(SpanMessage(role=role, parts=parts))

    return ("\n".join(system) if system else None, converted)


def _lang_chain_content_text(content: Any) -> str:
    """LangChain message content is a string, or a list holding typed blocks and bare strings.

    LangChain types it as ``str | list[str | dict]``, so a bare string inside the list is what the
    library documents rather than a malformed input. Keeping only the blocks whose ``type`` is
    ``text`` dropped those strings, and the span then showed less of the conversation than the model
    was given.

    Reachable from a caller: history content is passed straight into ``HumanMessage`` and
    ``AIMessage`` with no conversion, so whatever shape the caller supplies is the shape this reads.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    texts: list[str] = []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
        elif _get(block, "type") == "text":
            texts.append(str(_get(block, "text") or ""))
    return "".join(texts)


# ─── Writers ─────────────────────────────────────────────────────────────────


def _set_openllmetry_messages(
    span: Any, prefix: str, messages: list[SpanMessage]
) -> None:
    """Writes the OpenLLMetry carrier: one numbered attribute per field.

    ``prefix`` is ``gen_ai.prompt`` or ``gen_ai.completion``, the only difference between the input
    and output halves, which is why this is one parameterised function.

    Named for the convention rather than the shape. OpenLLMetry predates the GenAI semantic
    conventions and is what LaunchDarkly's LLM trace view parses today, so a reader who needs to
    know why ``gen_ai.prompt.0.role`` looks the way it does has somewhere to look it up.

    An empty message list writes nothing rather than a zero-length marker: the reader treats a
    missing key and an empty list identically.
    """
    for index, message in enumerate(messages):
        span.set_attribute(f"{prefix}.{index}.role", message.role)
        span.set_attribute(f"{prefix}.{index}.content", message.to_text())


def set_input_content_attributes(
    span: Any,
    capture: bool,
    *,
    system_instructions: str | None = None,
    messages: list[SpanMessage] | None = None,
    tool_definitions: list[ToolDefinitionInput] | None = None,
) -> None:
    """Records what the model was given: system instructions, the messages, and the tool catalog.

    ``system_instructions`` is written both to its own attribute and, when present, as message 0 of
    the OpenLLMetry carrier. That shape has no separate slot for it, and dropping it there would
    hide the system prompt from the only view that renders today.

    The ``gen_ai.content.prompt`` event is written whenever *capture* is true, even with nothing to
    say, which is asymmetric with the output side below. That asymmetry matches the TypeScript SDK
    and is deliberate rather than tidy.
    """
    if not capture:
        return

    msgs = messages or []

    if system_instructions:
        span.set_attribute(
            "gen_ai.system_instructions",
            json.dumps([{"type": "text", "content": system_instructions}]),
        )

    if msgs:
        span.set_attribute(
            "gen_ai.input.messages",
            json.dumps([m.to_canonical() for m in msgs]),
        )

    # The system prompt is a message here, unlike in the canonical carrier where it has its own
    # attribute. OpenLLMetry has no separate slot for it, so it goes in at index 0.
    openllmetry_messages = (
        [text_message("system", system_instructions), *msgs]
        if system_instructions
        else msgs
    )
    _set_openllmetry_messages(span, "gen_ai.prompt", openllmetry_messages)
    span.add_event(
        "gen_ai.content.prompt",
        {
            "gen_ai.prompt": "\n".join(
                f"{m.role}: {m.to_text()}" for m in openllmetry_messages
            )
        },
    )

    if tool_definitions:
        set_tool_definition_attributes(span, capture, tool_definitions)


def set_output_content_attributes(
    span: Any, capture: bool, messages: list[SpanMessage]
) -> None:
    """Records what the model produced.

    Unlike the input side, this writes nothing at all when there are no messages, event included.
    """
    if not capture or not messages:
        return
    span.set_attribute(
        "gen_ai.output.messages",
        json.dumps([m.to_canonical() for m in messages]),
    )
    _set_openllmetry_messages(span, "gen_ai.completion", messages)
    span.add_event(
        "gen_ai.content.completion",
        {"gen_ai.completion": "\n".join(m.to_text() for m in messages)},
    )


def set_tool_definition_attributes(
    span: Any, capture: bool, tools: list[ToolDefinitionInput]
) -> None:
    """Records the tools the model was allowed to call.

    The catalog is content because tool descriptions and parameter schemas routinely embed
    customer-specific detail, so it sits behind the same gate as the messages.
    """
    if not capture or not tools:
        return
    definitions: list[dict[str, Any]] = []
    for tool in tools:
        entry: dict[str, Any] = {"type": "function", "name": tool.name}
        if tool.description:
            entry["description"] = tool.description
        if tool.parameters is not None:
            entry["parameters"] = tool.parameters
        definitions.append(entry)
    span.set_attribute("gen_ai.tool.definitions", json.dumps(definitions))


def set_tool_call_content_attributes(
    span: Any,
    capture: bool,
    *,
    arguments: Any = None,
    result: Any = None,
) -> None:
    """Records one tool call's arguments and result on its ``execute_tool`` span.

    Both are stringified rather than written as native attribute values: an argument bag is an
    object, and OTel attributes hold only primitives and sequences of primitives.
    """
    if not capture:
        return
    if arguments is not None:
        span.set_attribute(
            "gen_ai.tool.call.arguments", _stringify_tool_value(arguments)
        )
    if result is not None:
        span.set_attribute("gen_ai.tool.call.result", _stringify_tool_value(result))


def _stringify_tool_value(value: Any) -> str:
    """A string passes through unchanged; anything else becomes JSON."""
    return value if isinstance(value, str) else json.dumps(value)
