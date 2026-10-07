"""TypeSafe Jev handler for LaunchDarkly AI judges."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score

from launchdarkly_ai_server import (
    AiConfigRep,
    ProviderHandler,
    create_handler,
    set_input_content_attributes,
    set_ld_span_attributes,
    set_model_identity_attributes,
    set_output_content_attributes,
    set_usage_span_attributes,
    text_message,
)
from launchdarkly_ai_server.utils import SpanUsage

from .questions import (
    build_typesafe_state,
    extract_typesafe_questions,
    reason_from_answer,
    score_from_answer,
    typesafe_output,
)

_TRACER = trace.get_tracer("launchdarkly-ai-typesafe")


def _sdk_question(question: Mapping[str, Any]) -> Any:
    instructions = question["instructions"]
    if question["type"] == "noul":
        criteria = question.get("criteria")
        if criteria:
            return Noul(instructions=instructions, criteria=criteria)
        return Noul(instructions=instructions)
    if question["type"] == "choice":
        return Choice(instructions=instructions, criteria=question["criteria"])
    return Score(instructions=instructions, criteria=question["criteria"])


def _usage_of(usage: Any) -> dict[str, int]:
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    if isinstance(usage, Mapping):
        input_tokens = usage.get("input_tokens", usage.get("inputTokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("outputTokens", 0))
    else:
        input_tokens = getattr(usage, "input_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "inputTokens", 0)
        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "outputTokens", 0)
    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
    }


def _answer_for(response: Any, key: str) -> Any:
    answers = getattr(response, "answers", None)
    if isinstance(answers, Mapping) and key in answers:
        return answers[key]
    if isinstance(response, Mapping):
        mapped = response.get("answers")
        if isinstance(mapped, Mapping):
            return mapped.get(key)
    return None


def create_typesafe_handler(*, capture_content: bool = False) -> ProviderHandler:
    """Build a handler with ``provides_for = ('TypeSafe', 'messages')``.

    Judge mode normalizes to messages, so a judge variation whose provider is
    ``TypeSafe`` selects this handler. ``TYPESAFE_API_KEY`` is read by the
    TypeSafe client from the environment.
    """

    async def _call(
        config: AiConfigRep,
        user_input: str | None = None,
        tool_handlers: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        del tool_handlers
        questions = extract_typesafe_questions(config)
        state = build_typesafe_state(
            user_input=user_input, variables=variables, history=history
        )
        model = config.get("model")
        model_name = model.get("name") if isinstance(model, Mapping) else None
        span_model = model_name or "jev"
        root = _TRACER.start_span("invoke_agent")
        chat = _TRACER.start_span(
            f"chat {span_model}", context=trace.set_span_in_context(root)
        )
        set_ld_span_attributes(root, dict(variables) if variables else None)
        set_model_identity_attributes(root, "typesafe", span_model)
        set_model_identity_attributes(chat, "typesafe", span_model)
        if capture_content:
            set_input_content_attributes(
                chat,
                True,
                messages=[text_message("user", json.dumps(state))],
            )
        try:
            async with AsyncTypeSafeClient() as client:
                response = await client.system_one(
                    state=state,
                    questions={
                        question["key"]: _sdk_question(question)
                        for question in questions
                    },
                    model=model_name or None,
                )
            results = []
            for question in questions:
                answer = _answer_for(response, question["key"])
                results.append(
                    {
                        "key": question["key"],
                        "eventKey": question["eventKey"],
                        "score": score_from_answer(question, answer),
                        "reason": reason_from_answer(question, answer),
                    }
                )
            output = typesafe_output(results)
            usage = _usage_of(getattr(response, "usage", None))
            span_usage = SpanUsage(
                input=usage["input_tokens"], output=usage["output_tokens"]
            )
            set_usage_span_attributes(chat, span_usage)
            set_usage_span_attributes(root, span_usage)
            if capture_content:
                set_output_content_attributes(
                    chat, True, [text_message("assistant", output)]
                )
            chat.set_status(Status(StatusCode.OK))
            root.set_status(Status(StatusCode.OK))
            return {"output": output, "usage": usage}
        except Exception as exc:
            chat.record_exception(exc)
            root.record_exception(exc)
            chat.set_status(Status(StatusCode.ERROR, str(exc)))
            root.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            chat.end()
            root.end()

    return create_handler(
        ("TypeSafe", "messages"), _call, capture_content=capture_content
    )
