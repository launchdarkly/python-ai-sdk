"""Tests for the shared model.parameters filter: select_forwarded_parameters."""

from __future__ import annotations

from launchdarkly_ai_server import select_forwarded_parameters


class TestSelectForwardedParameters:
    def test_keeps_only_forwarded_keys(self) -> None:
        params = {"temperature": 0.5, "unknown": 1}
        result = select_forwarded_parameters(params, frozenset({"temperature"}))
        assert result == {"temperature": 0.5}

    def test_drops_a_key_the_handler_never_classified_as_forwarded(self) -> None:
        params = {"temperature": 0.5, "api_key": "secret"}
        result = select_forwarded_parameters(params, frozenset({"temperature"}))
        assert result == {"temperature": 0.5}

    def test_empty_params_returns_empty(self) -> None:
        assert select_forwarded_parameters({}, frozenset({"temperature"})) == {}

    def test_empty_forwarded_keys_returns_empty(self) -> None:
        params = {"temperature": 0.5}
        assert select_forwarded_parameters(params, frozenset()) == {}

    def test_does_not_mutate_input(self) -> None:
        params = {"temperature": 0.5, "unknown": 1}
        select_forwarded_parameters(params, frozenset({"temperature"}))
        assert params == {"temperature": 0.5, "unknown": 1}
