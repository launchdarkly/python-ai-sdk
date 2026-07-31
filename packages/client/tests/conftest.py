from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_ld_client() -> MagicMock:
    """Stub LaunchDarkly client with variation/track/flush/close."""
    client = MagicMock()
    client.variation = AsyncMock(return_value=None)
    client.track = MagicMock()
    client.flush = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_span() -> MagicMock:
    """OTel span stub with add_event/set_status/end/record_exception spies."""
    span = MagicMock()
    span.add_event = MagicMock()
    span.set_attribute = MagicMock()
    span.set_status = MagicMock()
    span.end = MagicMock()
    span.record_exception = MagicMock()
    return span


@pytest.fixture
def mock_tracer(mock_span: MagicMock) -> MagicMock:
    """OTel tracer that returns mock_span from start_as_current_span and start_span."""
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__ = MagicMock(
        return_value=mock_span
    )
    tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
    tracer.start_span.return_value = mock_span
    return tracer
