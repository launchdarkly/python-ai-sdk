"""Span helpers for the ADK telemetry plugin. Reference: TESTING.md §2.x.3."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import launchdarkly_ai_google_adk_agents.spans as spans_mod
from launchdarkly_ai_google_adk_agents.spans import (
    abandon_open_spans,
    start_model_span,
    start_root_span,
    start_tool_span,
)

CONFIG = {
    "model": {"name": "gemini-2.5-flash"},
    "provider": {"name": "Google"},
    "instructions": "Be helpful.",
}
VARIABLES = {
    "__ld": {
        "configKey": "cfg",
        "variationKey": "var",
        "runId": "run-1",
    },
    "ldContext": {"kind": "user", "key": "user-1"},
}


class _Span:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, Any] = {}
        self.ended = False
        self.failed = False

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.attributes.setdefault("events", []).append(name)

    def end(self) -> None:
        self.ended = True

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        return None

    def record_exception(self, exc: BaseException) -> None:
        self.failed = True


def test_root_span_identity(monkeypatch: Any) -> None:
    spans: list[_Span] = []

    def start_span(name: str, **kwargs: Any) -> _Span:
        span = _Span(name)
        spans.append(span)
        return span

    tracer = SimpleNamespace(start_span=start_span)
    monkeypatch.setattr(spans_mod.trace, "get_tracer", lambda name: tracer)
    root = start_root_span(CONFIG, VARIABLES)
    assert root is spans[0]
    assert root.name == "invoke_agent"
    assert root.attributes["gen_ai.system"] == "google_adk"
    assert root.attributes["gen_ai.provider.name"] == "gcp.gemini"
    assert root.attributes["gen_ai.request.model"] == "gemini-2.5-flash"
    assert root.attributes["launchdarkly.config.key"] == "cfg"
    assert "feature_flag" in root.attributes["events"]


def test_child_spans_do_not_repeat_launchdarkly_identity(monkeypatch: Any) -> None:
    def start_span(name: str, **kwargs: Any) -> _Span:
        return _Span(name)

    monkeypatch.setattr(
        spans_mod.trace,
        "get_tracer",
        lambda name: SimpleNamespace(start_span=start_span),
    )
    model = start_model_span(CONFIG, parent=MagicMock())
    tool = start_tool_span("lookup", "call-1", parent=MagicMock())
    assert model.name == "chat gemini-2.5-flash"
    assert tool.name == "execute_tool lookup"
    assert "launchdarkly.config.key" not in model.attributes
    assert "launchdarkly.config.key" not in tool.attributes


def test_abandon_does_not_fail_the_tool_span(monkeypatch: Any) -> None:
    span = _Span("execute_tool lookup")
    fail = MagicMock()
    monkeypatch.setattr(spans_mod, "fail_span", fail)
    abandon_open_spans([span], ended=set())
    assert span.ended is True
    fail.assert_not_called()
