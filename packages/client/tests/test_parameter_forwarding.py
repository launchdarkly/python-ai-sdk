"""
Tests for the shared model.parameters filter: is_transport_parameter,
strip_transport_parameters, accepted_parameter_keys_from_signature,
accepted_parameter_keys_from_dataclass, accepted_parameter_keys_from_pydantic_model,
filter_forwardable_parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from launchdarkly_ai_server import (
    accepted_parameter_keys_from_dataclass,
    accepted_parameter_keys_from_pydantic_model,
    accepted_parameter_keys_from_signature,
    filter_forwardable_parameters,
    is_transport_parameter,
    strip_transport_parameters,
)


class TestIsTransportParameter:
    @pytest.mark.parametrize(
        "key",
        [
            "timeout",
            "extra_headers",
            "extra_query",
            "extra_body",
            "extra_args",
            "extra_x",
        ],
    )
    def test_matches_known_transport_keys(self, key: str) -> None:
        assert is_transport_parameter(key)

    @pytest.mark.parametrize("key", ["temperature", "top_p", "model", "extraordinary"])
    def test_does_not_match_generation_keys(self, key: str) -> None:
        assert not is_transport_parameter(key)


class TestStripTransportParameters:
    def test_removes_every_transport_key(self) -> None:
        params = {
            "temperature": 0.5,
            "timeout": 30,
            "extra_body": {"a": 1},
            "extra_headers": {"h": "v"},
        }
        assert strip_transport_parameters(params) == {"temperature": 0.5}

    def test_does_not_mutate_input(self) -> None:
        params = {"timeout": 30, "temperature": 0.5}
        strip_transport_parameters(params)
        assert params == {"timeout": 30, "temperature": 0.5}

    def test_empty_input_returns_empty(self) -> None:
        assert strip_transport_parameters({}) == {}


class TestAcceptedParameterKeysFromSignature:
    def test_derives_names_from_plain_function(self) -> None:
        def fn(a: int, b: str = "x") -> None: ...

        assert accepted_parameter_keys_from_signature(fn) == frozenset({"a", "b"})

    def test_excludes_self(self) -> None:
        class C:
            def method(self, a: int) -> None: ...

        assert accepted_parameter_keys_from_signature(C.method) == frozenset({"a"})

    def test_excludes_var_positional(self) -> None:
        def fn(a: int, *args: Any) -> None: ...

        assert accepted_parameter_keys_from_signature(fn) == frozenset({"a"})

    def test_raises_on_var_keyword_catch_all(self) -> None:
        def fn(a: int, **kwargs: Any) -> None: ...

        with pytest.raises(ValueError, match="kwargs"):
            accepted_parameter_keys_from_signature(fn)


class TestAcceptedParameterKeysFromDataclass:
    def test_derives_field_names(self) -> None:
        @dataclass
        class Options:
            a: int = 0
            b: str = ""

        assert accepted_parameter_keys_from_dataclass(Options) == frozenset({"a", "b"})


class TestAcceptedParameterKeysFromPydanticModel:
    def test_includes_field_names_and_string_aliases(self) -> None:
        cls = SimpleNamespace(
            model_fields={
                "model_name": SimpleNamespace(alias="model", validation_alias="model"),
                "temperature": SimpleNamespace(alias=None, validation_alias=None),
            }
        )
        keys = accepted_parameter_keys_from_pydantic_model(cls)
        assert keys == frozenset({"model_name", "model", "temperature"})

    def test_reads_alias_choices(self) -> None:
        cls = SimpleNamespace(
            model_fields={
                "field_a": SimpleNamespace(
                    alias=None,
                    validation_alias=SimpleNamespace(
                        choices=["alias_one", "alias_two"]
                    ),
                ),
            }
        )
        keys = accepted_parameter_keys_from_pydantic_model(cls)
        assert keys == frozenset({"field_a", "alias_one", "alias_two"})

    def test_raises_when_model_fields_missing(self) -> None:
        with pytest.raises(ValueError, match="model_fields"):
            accepted_parameter_keys_from_pydantic_model(SimpleNamespace())

    def test_raises_when_model_fields_not_a_mapping(self) -> None:
        with pytest.raises(ValueError, match="model_fields"):
            accepted_parameter_keys_from_pydantic_model(
                SimpleNamespace(model_fields="not-a-mapping")
            )


class TestFilterForwardableParameters:
    def test_keeps_only_accepted_keys(self) -> None:
        params = {"temperature": 0.5, "unknown": 1}
        result = filter_forwardable_parameters(params, frozenset({"temperature"}))
        assert result == {"temperature": 0.5}

    def test_strips_transport_keys_even_if_accepted(self) -> None:
        params = {"temperature": 0.5, "timeout": 30, "extra_body": {"a": 1}}
        accepted = frozenset({"temperature", "timeout", "extra_body"})
        result = filter_forwardable_parameters(params, accepted)
        assert result == {"temperature": 0.5}

    def test_strips_excluded_keys_even_if_accepted(self) -> None:
        params = {"temperature": 0.5, "stream": True}
        accepted = frozenset({"temperature", "stream"})
        result = filter_forwardable_parameters(params, accepted, frozenset({"stream"}))
        assert result == {"temperature": 0.5}

    def test_empty_params_returns_empty(self) -> None:
        assert filter_forwardable_parameters({}, frozenset({"temperature"})) == {}

    def test_does_not_mutate_input(self) -> None:
        params = {"temperature": 0.5, "unknown": 1}
        filter_forwardable_parameters(params, frozenset({"temperature"}))
        assert params == {"temperature": 0.5, "unknown": 1}
