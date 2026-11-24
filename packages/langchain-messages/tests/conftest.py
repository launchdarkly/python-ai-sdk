from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_span() -> MagicMock:
    span = MagicMock()
    span.add_event = MagicMock()
    span.set_attribute = MagicMock()
    span.set_status = MagicMock()
    span.end = MagicMock()
    span.record_exception = MagicMock()
    return span


@pytest.fixture
def mock_tracer(mock_span: MagicMock) -> MagicMock:
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__ = MagicMock(
        return_value=mock_span
    )
    tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
    tracer.start_span.return_value = mock_span
    return tracer


@pytest.fixture(autouse=True)
def gemini_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Gemini config needs an API key in the environment before the handler builds its model.
    Every test starts with a fake ``GOOGLE_API_KEY`` and no ``GEMINI_API_KEY``, and a test of the
    key rules sets its own."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-not-a-real-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
