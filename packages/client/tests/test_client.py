"""
Tests for §3.10 config(), §3.12 LD context interpolation,
§3.15 config().stream().
Reference: TESTING.md §3.10, §3.12, §3.15
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import (
    JudgeResult,
    ProviderHandler,
    config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONTEXT = {"kind": "user", "key": "u1"}


def _make_client(response: Any = "Hello", usage: dict | None = None) -> MagicMock:
    c = MagicMock()
    c.track = MagicMock()
    c.flush = AsyncMock()
    c.close = AsyncMock()
    raw_variation = {
        "model": {"name": "gpt-4"},
        "provider": {"name": "TestProvider"},
        "instructions": "Be helpful.",
        "_ldMeta": {
            "enabled": True,
            "variationKey": "v1",
            "version": 1,
            "mode": "messages",
        },
    }
    c.variation = AsyncMock(return_value=raw_variation)
    return c


def _make_handler(
    response: str = "Hello",
    usage: dict | None = None,
    stream_chunks: list[str] | None = None,
) -> ProviderHandler:
    """Creates a mock ProviderHandler."""
    _usage = usage or {"input_tokens": 10, "output_tokens": 5}

    async def fn(cfg, user_input, tool_handlers, variables, history=None) -> dict:  # type: ignore[override]
        return {"output": response, "usage": _usage}

    async def stream_fn(
        cfg, user_input, tool_handlers, variables, history=None
    ) -> AsyncGenerator:  # type: ignore[override]
        chunks = stream_chunks or ["Hello", " World"]
        for c in chunks:
            yield {"type": "chunk", "text": c}
        yield {"type": "done", "output": "".join(chunks), "usage": _usage}

    return ProviderHandler(
        fn=fn,
        provides_for=("TestProvider", "messages"),
        stream_fn=stream_fn if stream_chunks is not None else None,
    )


@pytest.fixture
def mock_ld_client() -> MagicMock:
    client = _make_client()
    lifecycle_module._set_client_for_testing(client)
    yield client
    lifecycle_module._reset_for_testing()


# ---------------------------------------------------------------------------
# §3.10 config() — single handler
# ---------------------------------------------------------------------------


class TestConfigSingleHandler:
    async def test_calls_extract_variation_with_correct_key_and_context(
        self, mock_ld_client: MagicMock
    ) -> None:
        h = _make_handler()
        m = config(key="my-flag", handler=h)
        await m.invoke("hi", CONTEXT)
        mock_ld_client.variation.assert_called_once_with("my-flag", CONTEXT, None)

    async def test_calls_the_handler(self, mock_ld_client: MagicMock) -> None:
        calls: list[Any] = []

        async def recording_fn(
            cfg, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            calls.append((cfg, user_input))
            return {"output": "ok", "usage": {"input_tokens": 1, "output_tokens": 1}}

        h = ProviderHandler(fn=recording_fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]
        m = config(key="flag", handler=h)
        await m.invoke("hello", CONTEXT)
        assert len(calls) == 1
        assert calls[0][1] == "hello"

    async def test_returns_provider_response(self, mock_ld_client: MagicMock) -> None:
        m = config(key="flag", handler=_make_handler("answer"))
        result = await m.invoke("q", CONTEXT)
        assert result.response == "answer"

    async def test_generation_success_on_success(
        self, mock_ld_client: MagicMock
    ) -> None:
        m = config(key="flag", handler=_make_handler())
        await m.invoke("q", CONTEXT)
        events = [c[0][0] for c in mock_ld_client.track.call_args_list]
        assert "$ld:ai:generation:success" in events

    async def test_generation_error_on_failure(self, mock_ld_client: MagicMock) -> None:
        async def bad_fn(
            cfg, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            raise RuntimeError("fail")

        h = ProviderHandler(fn=bad_fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]
        m = config(key="flag", handler=h)
        with pytest.raises(RuntimeError):
            await m.invoke("q", CONTEXT)
        events = [c[0][0] for c in mock_ld_client.track.call_args_list]
        assert "$ld:ai:generation:error" in events

    async def test_duration_tracked_on_success(self, mock_ld_client: MagicMock) -> None:
        m = config(key="flag", handler=_make_handler())
        await m.invoke("q", CONTEXT)
        events = [c[0][0] for c in mock_ld_client.track.call_args_list]
        assert "$ld:ai:duration:total" in events

    async def test_duration_tracked_even_on_failure(
        self, mock_ld_client: MagicMock
    ) -> None:
        async def bad_fn(
            cfg, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            raise RuntimeError("fail")

        h = ProviderHandler(fn=bad_fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]
        m = config(key="flag", handler=h)
        with pytest.raises(RuntimeError):
            await m.invoke("q", CONTEXT)
        events = [c[0][0] for c in mock_ld_client.track.call_args_list]
        assert "$ld:ai:duration:total" in events

    async def test_token_tracking_when_usage_nonzero(
        self, mock_ld_client: MagicMock
    ) -> None:
        m = config(
            key="flag",
            handler=_make_handler(usage={"input_tokens": 10, "output_tokens": 5}),
        )
        await m.invoke("q", CONTEXT)
        events = [c[0][0] for c in mock_ld_client.track.call_args_list]
        assert "$ld:ai:tokens:total" in events
        assert "$ld:ai:tokens:input" in events
        assert "$ld:ai:tokens:output" in events

    async def test_token_events_skipped_when_zero(
        self, mock_ld_client: MagicMock
    ) -> None:
        m = config(
            key="flag",
            handler=_make_handler(usage={"input_tokens": 0, "output_tokens": 0}),
        )
        await m.invoke("q", CONTEXT)
        events = [c[0][0] for c in mock_ld_client.track.call_args_list]
        assert "$ld:ai:tokens:total" not in events

    async def test_judge_integration_absent_config(
        self, mock_ld_client: MagicMock
    ) -> None:
        m = config(key="flag", handler=_make_handler())
        result = await m.invoke("q", CONTEXT)
        assert result.judge_results is None

    async def test_disabled_variation_propagates_error(
        self, mock_ld_client: MagicMock
    ) -> None:
        mock_ld_client.variation = AsyncMock(return_value=None)
        m = config(key="flag", handler=_make_handler())
        with pytest.raises(RuntimeError):
            await m.invoke("q", CONTEXT)

    async def test_throws_when_single_handler_provider_does_not_match(
        self, mock_ld_client: MagicMock
    ) -> None:
        # variation provider is "TestProvider", but handler claims "OtherProvider"
        h = ProviderHandler(
            fn=_make_handler()._fn,  # type: ignore[arg-type]
            provides_for=("OtherProvider", "messages"),
        )
        m = config(key="flag", handler=h)
        with pytest.raises((ValueError, RuntimeError)):
            await m.invoke("q", CONTEXT)

    async def test_throws_when_single_handler_mode_does_not_match(
        self, mock_ld_client: MagicMock
    ) -> None:
        raw = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "hi",
            "_ldMeta": {
                "enabled": True,
                "variationKey": "v1",
                "version": 1,
                "mode": "agent",
            },
        }
        mock_ld_client.variation = AsyncMock(return_value=raw)
        h = ProviderHandler(
            fn=_make_handler()._fn,  # type: ignore[arg-type]
            provides_for=("TestProvider", "messages"),
        )
        m = config(key="flag", handler=h)
        with pytest.raises((ValueError, RuntimeError), match="agent"):
            await m.invoke("q", CONTEXT)

    async def test_string_output_json_parsed_when_output_format_set(
        self, mock_ld_client: MagicMock
    ) -> None:
        raw = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "Reply in JSON",
            "_ldMeta": {
                "enabled": True,
                "variationKey": "v1",
                "version": 1,
                "mode": "messages",
            },
            "outputFormat": {"type": "object"},
        }
        mock_ld_client.variation = AsyncMock(return_value=raw)
        handler = _make_handler(response='{"key": "value"}')
        m = config(key="flag", handler=handler)
        result = await m.invoke("q", CONTEXT)
        assert result.response == {"key": "value"}

    async def test_fenced_json_stripped_when_output_format_set(
        self, mock_ld_client: MagicMock
    ) -> None:
        raw = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "hi",
            "_ldMeta": {
                "enabled": True,
                "variationKey": "v1",
                "version": 1,
                "mode": "messages",
            },
            "outputFormat": {"type": "object"},
        }
        mock_ld_client.variation = AsyncMock(return_value=raw)
        handler = _make_handler(response='```json\n{"a": 1}\n```')
        m = config(key="flag", handler=handler)
        result = await m.invoke("q", CONTEXT)
        assert result.response == {"a": 1}

    async def test_object_output_returned_as_is_when_output_format_set(
        self, mock_ld_client: MagicMock
    ) -> None:
        raw = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "hi",
            "_ldMeta": {
                "enabled": True,
                "variationKey": "v1",
                "version": 1,
                "mode": "messages",
            },
            "outputFormat": {"type": "object"},
        }
        mock_ld_client.variation = AsyncMock(return_value=raw)

        async def obj_fn(
            cfg, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            return {
                "output": {"already": "parsed"},
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        h = ProviderHandler(fn=obj_fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]
        m = config(key="flag", handler=h)
        result = await m.invoke("q", CONTEXT)
        assert result.response == {"already": "parsed"}

    async def test_no_parsing_when_output_format_absent(
        self, mock_ld_client: MagicMock
    ) -> None:
        handler = _make_handler(response='{"a":1}')
        m = config(key="flag", handler=handler)
        result = await m.invoke("q", CONTEXT)
        assert result.response == '{"a":1}'

    async def test_parse_failure_returns_raw_string_when_output_format_set(
        self, mock_ld_client: MagicMock
    ) -> None:
        """When outputFormat is set but the handler returns an unparseable string,
        invoke() returns the raw string rather than raising — agents and streaming
        handlers cannot guarantee structured output (best-effort, consistent with
        TypeScript SDK behavior). See TESTING.md §3.10."""
        raw = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "hi",
            "_ldMeta": {
                "enabled": True,
                "variationKey": "v1",
                "version": 1,
                "mode": "messages",
            },
            "outputFormat": {"type": "object"},
        }
        mock_ld_client.variation = AsyncMock(return_value=raw)
        handler = _make_handler(response="not json")
        m = config(key="flag", handler=handler)
        result = await m.invoke("q", CONTEXT)
        assert result.response == "not json"


# ---------------------------------------------------------------------------
# §3.10 config() — multi-handler routing
# ---------------------------------------------------------------------------


def _handler_for(provider: str, mode: str, response: str = "ok") -> ProviderHandler:
    async def fn(cfg, user_input, tool_handlers, variables, history=None) -> dict:  # type: ignore[override]
        return {"output": response, "usage": {"input_tokens": 1, "output_tokens": 1}}

    return ProviderHandler(fn=fn, provides_for=(provider, mode))  # type: ignore[arg-type]


class TestConfigMultiHandler:
    async def test_selects_right_handler_by_provider_and_mode(
        self, mock_ld_client: MagicMock
    ) -> None:
        h_messages = _handler_for("TestProvider", "messages", "messages-result")
        h_agent = _handler_for("TestProvider", "agent", "agent-result")
        rm = config(key="flag", handler=[h_messages, h_agent])
        result = await rm.invoke("hi", CONTEXT)
        assert result.response == "messages-result"

    async def test_mode_normalization(self, mock_ld_client: MagicMock) -> None:
        raw = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "hi",
            "_ldMeta": {
                "enabled": True,
                "variationKey": "v1",
                "version": 1,
                "mode": "completion",
            },
        }
        mock_ld_client.variation = AsyncMock(return_value=raw)
        h = _handler_for("TestProvider", "messages", "normalized")
        rm = config(key="flag", handler=[h])
        result = await rm.invoke("hi", CONTEXT)
        assert result.response == "normalized"

    async def test_throws_when_no_provider(self, mock_ld_client: MagicMock) -> None:
        raw = {
            "model": {"name": "gpt-4"},
            "instructions": "hi",
            "_ldMeta": {"enabled": True, "variationKey": "v1", "version": 1},
        }
        mock_ld_client.variation = AsyncMock(return_value=raw)
        rm = config(key="flag", handler=[_handler_for("X", "messages")])
        with pytest.raises((ValueError, RuntimeError)):
            await rm.invoke("hi", CONTEXT)

    async def test_throws_when_no_matching_handler(
        self, mock_ld_client: MagicMock
    ) -> None:
        rm = config(key="flag", handler=[_handler_for("OtherProvider", "messages")])
        with pytest.raises((ValueError, RuntimeError)):
            await rm.invoke("hi", CONTEXT)

    async def test_wildcard_handler_selected_when_no_exact_match(
        self, mock_ld_client: MagicMock
    ) -> None:
        raw = {
            "model": {"name": "claude-3"},
            "provider": {"name": "Anthropic"},
            "instructions": "hi",
            "_ldMeta": {
                "enabled": True,
                "variationKey": "v1",
                "version": 1,
                "mode": "messages",
            },
        }
        mock_ld_client.variation = AsyncMock(return_value=raw)
        h = _handler_for("*", "messages", "from-wildcard")
        rm = config(key="flag", handler=[h])
        result = await rm.invoke("hi", CONTEXT)
        assert result.response == "from-wildcard"

    async def test_explicit_provider_wins_over_wildcard(
        self, mock_ld_client: MagicMock
    ) -> None:
        h_explicit = _handler_for("TestProvider", "messages", "explicit")
        h_wildcard = _handler_for("*", "messages", "wildcard")
        rm = config(key="flag", handler=[h_wildcard, h_explicit])
        result = await rm.invoke("hi", CONTEXT)
        assert result.response == "explicit"

    async def test_wildcard_handler_does_not_match_different_mode(
        self, mock_ld_client: MagicMock
    ) -> None:
        raw = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "hi",
            "_ldMeta": {
                "enabled": True,
                "variationKey": "v1",
                "version": 1,
                "mode": "agent",
            },
        }
        mock_ld_client.variation = AsyncMock(return_value=raw)
        h = _handler_for("*", "messages")
        rm = config(key="flag", handler=[h])
        with pytest.raises((ValueError, RuntimeError)):
            await rm.invoke("hi", CONTEXT)

    async def test_resolves_handlers_from_registry(
        self, mock_ld_client: MagicMock
    ) -> None:
        from launchdarkly_ai_server import Registry

        h = _handler_for("TestProvider", "messages", "from-registry")
        reg = Registry(handlers=[h])
        rm = config(key="flag", registry=reg)
        result = await rm.invoke("hi", CONTEXT)
        assert result.response == "from-registry"

    async def test_local_handler_takes_precedence_over_registry(
        self, mock_ld_client: MagicMock
    ) -> None:
        from launchdarkly_ai_server import Registry

        h_reg = _handler_for("TestProvider", "messages", "from-registry")
        h_local = _handler_for("TestProvider", "messages", "local")
        reg = Registry(handlers=[h_reg])
        rm = config(key="flag", handler=[h_local], registry=reg)
        result = await rm.invoke("hi", CONTEXT)
        assert result.response == "local"

    async def test_invoke_exposes_track_data_for_prepare_judge(
        self, mock_ld_client: MagicMock
    ) -> None:
        rm = config(key="flag", handler=[_handler_for("TestProvider", "messages")])
        result = await rm.invoke("hi", CONTEXT)
        assert result.track_data is not None
        assert result.track_data.get("configKey") == "flag"

    async def test_throws_mode_specific_error_when_provider_matches_but_mode_does_not(
        self, mock_ld_client: MagicMock
    ) -> None:
        """Provider matches but mode doesn't — error should reference mode."""
        raw = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "hi",
            "_ldMeta": {
                "enabled": True,
                "variationKey": "v1",
                "version": 1,
                "mode": "agent",
            },
        }
        mock_ld_client.variation = AsyncMock(return_value=raw)
        h = _handler_for(
            "TestProvider", "messages"
        )  # messages handler, but config requests agent
        rm = config(key="flag", handler=[h])
        with pytest.raises((ValueError, RuntimeError), match="agent"):
            await rm.invoke("hi", CONTEXT)


# ---------------------------------------------------------------------------
# §3.12 LD context interpolation
# ---------------------------------------------------------------------------


class TestLDContextInterpolation:
    async def test_context_attributes_exposed_as_ld_context(
        self, mock_ld_client: MagicMock
    ) -> None:
        received_variables: list[Any] = []

        async def capturing_fn(
            cfg, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            received_variables.append(variables)
            return {"output": "ok", "usage": {}}

        h = ProviderHandler(fn=capturing_fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]
        ctx = {"kind": "user", "key": "bob", "name": "Bob"}
        m = config(key="flag", handler=h)
        await m.invoke("hi", ctx)
        assert received_variables[0]["ldContext"] == ctx

    async def test_user_variables_preserved_alongside_ld_context(
        self, mock_ld_client: MagicMock
    ) -> None:
        received_variables: list[Any] = []

        async def capturing_fn(
            cfg, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            received_variables.append(variables)
            return {"output": "ok", "usage": {}}

        h = ProviderHandler(fn=capturing_fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]
        m = config(key="flag", handler=h)
        await m.invoke("hi", CONTEXT, variables={"custom": "value"})
        assert received_variables[0]["custom"] == "value"
        assert "ldContext" in received_variables[0]

    async def test_user_supplied_ld_context_always_discarded(
        self, mock_ld_client: MagicMock
    ) -> None:
        received_variables: list[Any] = []

        async def capturing_fn(
            cfg, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            received_variables.append(variables)
            return {"output": "ok", "usage": {}}

        h = ProviderHandler(fn=capturing_fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]
        m = config(key="flag", handler=h)
        await m.invoke("hi", CONTEXT, variables={"ldContext": {"key": "injected"}})
        assert received_variables[0]["ldContext"]["key"] == CONTEXT["key"]

    async def test_kind_field_included(self, mock_ld_client: MagicMock) -> None:
        received_variables: list[Any] = []

        async def capturing_fn(
            cfg, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            received_variables.append(variables)
            return {"output": "ok", "usage": {}}

        h = ProviderHandler(fn=capturing_fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]
        ctx = {"kind": "user", "key": "u"}
        m = config(key="flag", handler=h)
        await m.invoke("hi", ctx)
        assert received_variables[0]["ldContext"]["kind"] == "user"

    async def test_ld_context_injected_via_multi_handler_config(
        self, mock_ld_client: MagicMock
    ) -> None:
        received_variables: list[Any] = []

        async def capturing_fn(
            cfg, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            received_variables.append(variables)
            return {"output": "ok", "usage": {}}

        h = ProviderHandler(fn=capturing_fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]
        ctx = {"kind": "user", "key": "routed-user", "email": "x@y.com"}
        rm = config(key="flag", handler=[h])
        await rm.invoke("hi", ctx)
        assert received_variables[0]["ldContext"]["key"] == "routed-user"


# ---------------------------------------------------------------------------
# §3.15 config().stream()
# ---------------------------------------------------------------------------


class TestConfigStream:
    async def test_returns_async_generator(self, mock_ld_client: MagicMock) -> None:
        h = _make_handler(stream_chunks=["hi"])
        m = config(key="flag", handler=h)
        events = [e async for e in m.stream("q", CONTEXT)]
        assert len(events) > 0

    async def test_forwards_chunk_events(self, mock_ld_client: MagicMock) -> None:
        h = _make_handler(stream_chunks=["Hello", " World"])
        m = config(key="flag", handler=h)
        events = [e async for e in m.stream("q", CONTEXT)]
        chunks = [e for e in events if e.get("type") == "chunk"]
        assert len(chunks) == 2

    async def test_yields_done_event_as_last(self, mock_ld_client: MagicMock) -> None:
        h = _make_handler(stream_chunks=["a", "b"])
        m = config(key="flag", handler=h)
        events = [e async for e in m.stream("q", CONTEXT)]
        assert events[-1]["type"] == "done"

    async def test_done_response_is_concatenated_text(
        self, mock_ld_client: MagicMock
    ) -> None:
        h = _make_handler(stream_chunks=["foo", "bar"])
        m = config(key="flag", handler=h)
        events = [e async for e in m.stream("q", CONTEXT)]
        done = events[-1]
        assert done["response"] == "foobar"

    async def test_emits_generation_success_after_stream(
        self, mock_ld_client: MagicMock
    ) -> None:
        h = _make_handler(stream_chunks=["ok"])
        m = config(key="flag", handler=h)
        async for _ in m.stream("q", CONTEXT):
            pass
        events_tracked = [c[0][0] for c in mock_ld_client.track.call_args_list]
        assert "$ld:ai:generation:success" in events_tracked

    async def test_emits_token_events_when_usage_nonzero(
        self, mock_ld_client: MagicMock
    ) -> None:
        h = _make_handler(
            stream_chunks=["ok"], usage={"input_tokens": 5, "output_tokens": 3}
        )
        m = config(key="flag", handler=h)
        async for _ in m.stream("q", CONTEXT):
            pass
        events_tracked = [c[0][0] for c in mock_ld_client.track.call_args_list]
        assert "$ld:ai:tokens:total" in events_tracked

    async def test_fallback_to_blocking_handler_when_no_stream(
        self, mock_ld_client: MagicMock
    ) -> None:
        # Handler with no stream method — should fall back to blocking call
        async def fn(cfg, user_input, tool_handlers, variables, history=None) -> dict:  # type: ignore[override]
            return {
                "output": "blocking",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        h = ProviderHandler(fn=fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]
        m = config(key="flag", handler=h)
        events = [e async for e in m.stream("q", CONTEXT)]
        done = events[-1]
        assert done["type"] == "done"
        assert done["response"] == "blocking"

    async def test_emits_duration_total_after_stream_completes(
        self, mock_ld_client: MagicMock
    ) -> None:
        h = _make_handler(stream_chunks=["ok"])
        m = config(key="flag", handler=h)
        async for _ in m.stream("q", CONTEXT):
            pass
        events_tracked = [c[0][0] for c in mock_ld_client.track.call_args_list]
        assert "$ld:ai:duration:total" in events_tracked

    async def test_no_token_events_when_usage_is_zero(
        self, mock_ld_client: MagicMock
    ) -> None:
        h = _make_handler(
            stream_chunks=["ok"], usage={"input_tokens": 0, "output_tokens": 0}
        )
        m = config(key="flag", handler=h)
        async for _ in m.stream("q", CONTEXT):
            pass
        events_tracked = [c[0][0] for c in mock_ld_client.track.call_args_list]
        assert "$ld:ai:tokens:total" not in events_tracked

    async def test_generation_error_and_rethrow_on_stream_error(
        self, mock_ld_client: MagicMock
    ) -> None:
        def _bad_stream_fn(
            cfg, user_input, tool_handlers, variables, history=None
        ) -> Any:
            async def _gen() -> Any:
                raise RuntimeError("stream-fail")
                yield  # make it a generator

            return _gen()

        async def _noop_fn(
            cfg, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            return {"output": "", "usage": {}}

        h = ProviderHandler(
            fn=_noop_fn,
            stream_fn=_bad_stream_fn,
            provides_for=("TestProvider", "messages"),
        )  # type: ignore[arg-type]
        m = config(key="flag", handler=h)
        with pytest.raises(RuntimeError, match="stream-fail"):
            async for _ in m.stream("q", CONTEXT):
                pass
        events_tracked = [c[0][0] for c in mock_ld_client.track.call_args_list]
        assert "$ld:ai:generation:error" in events_tracked

    async def test_ld_context_injected_into_stream_variables(
        self, mock_ld_client: MagicMock
    ) -> None:
        received_variables: list[Any] = []

        def _capturing_stream_fn(
            cfg, user_input, tool_handlers, variables, history=None
        ) -> Any:
            received_variables.append(variables)

            async def _gen() -> Any:
                yield {"type": "chunk", "text": "hi"}
                yield {
                    "type": "done",
                    "output": "hi",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }

            return _gen()

        async def _noop_fn(
            cfg, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            return {"output": "", "usage": {}}

        h = ProviderHandler(
            fn=_noop_fn,
            stream_fn=_capturing_stream_fn,
            provides_for=("TestProvider", "messages"),
        )  # type: ignore[arg-type]
        ctx = {"kind": "user", "key": "stream-user"}
        m = config(key="flag", handler=h)
        async for _ in m.stream("q", ctx):
            pass
        assert len(received_variables) > 0
        assert received_variables[0]["ldContext"]["key"] == "stream-user"


class TestConfigStreamMultiHandler:
    async def test_selects_correct_handler_then_streams(
        self, mock_ld_client: MagicMock
    ) -> None:
        h = _make_handler(stream_chunks=["routed"])
        h.provides_for = ("TestProvider", "messages")
        rm = config(key="flag", handler=[h])
        events = [e async for e in rm.stream("q", CONTEXT)]
        done = events[-1]
        assert done["response"] == "routed"

    async def test_yields_chunks_and_done(self, mock_ld_client: MagicMock) -> None:
        h = _make_handler(stream_chunks=["a", "b"])
        h.provides_for = ("TestProvider", "messages")
        rm = config(key="flag", handler=[h])
        events = [e async for e in rm.stream("q", CONTEXT)]
        assert any(e["type"] == "chunk" for e in events)
        assert events[-1]["type"] == "done"

    async def test_throws_when_no_matching_handler(
        self, mock_ld_client: MagicMock
    ) -> None:
        rm = config(key="flag", handler=[_handler_for("OtherProvider", "messages")])
        with pytest.raises((ValueError, RuntimeError)):
            async for _ in rm.stream("q", CONTEXT):
                pass


# ---------------------------------------------------------------------------
# skip_judges option
# ---------------------------------------------------------------------------


class TestSkipJudges:
    """Tests that skip_judges=True suppresses automatic judge evaluation."""

    async def test_invoke_skips_run_judges_when_skip_judges_true(
        self, mock_ld_client: MagicMock
    ) -> None:
        main_variation = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "Be helpful.",
            "judgeConfiguration": {
                "judges": [{"key": "judge-key", "samplingRate": 1.0}]
            },
            "_ldMeta": {
                "enabled": True,
                "variationKey": "v1",
                "version": 1,
                "mode": "messages",
            },
        }
        judge_variation = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "Judge.",
            "evaluationMetricKey": "judge-metric",
            "_ldMeta": {
                "enabled": True,
                "variationKey": "j1",
                "version": 1,
                "mode": "judge",
            },
        }

        call_count = [0]

        async def multi_variation(*args: object, **kwargs: object) -> dict:
            call_count[0] += 1
            return judge_variation if call_count[0] > 1 else main_variation

        mock_ld_client.variation = AsyncMock(side_effect=multi_variation)

        async def recording_fn(
            cfg, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            return {"output": "Hello", "usage": {"input_tokens": 1, "output_tokens": 1}}

        h = ProviderHandler(fn=recording_fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]
        result = await config(key="flag", handler=h, skip_judges=True).invoke(
            "q", CONTEXT
        )

        # Two variation calls: main config + judge task resolution (build_judge_tasks fetches)
        assert mock_ld_client.variation.call_count == 2
        assert result.judge_results is None
        assert result.judge_tasks is not None
        assert len(result.judge_tasks) == 1
        assert result.judge_tasks[0].config_key == "judge-key"
        assert result.judge_tasks[0].evaluation_metric_key == "judge-metric"

    async def test_invoke_runs_judges_by_default(
        self, mock_ld_client: MagicMock
    ) -> None:
        """When skip_judges is not set, judges configured via judgeConfiguration run normally."""
        judge_variation = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "You are a judge.",
            "_ldMeta": {
                "enabled": True,
                "variationKey": "j1",
                "version": 1,
                "mode": "judge",
            },
        }

        call_count = [0]

        async def multi_variation(*args: object, **kwargs: object) -> dict:
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "model": {"name": "gpt-4"},
                    "provider": {"name": "TestProvider"},
                    "instructions": "Be helpful.",
                    "judgeConfiguration": {
                        "judges": [{"key": "judge-key", "samplingRate": 1.0}]
                    },
                    "_ldMeta": {
                        "enabled": True,
                        "variationKey": "v1",
                        "version": 1,
                        "mode": "messages",
                    },
                }
            return judge_variation

        mock_ld_client.variation = AsyncMock(side_effect=multi_variation)

        async def judge_fn(
            cfg, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            return {
                "output": '{"score": 0.9, "reasoning": "good"}',
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        h = ProviderHandler(fn=judge_fn, provides_for=("TestProvider", "messages"))  # type: ignore[arg-type]

        import random

        with __import__("unittest.mock", fromlist=["patch"]).patch.object(
            random, "random", return_value=0.0
        ):
            result = await config(key="flag", handler=h).invoke("q", CONTEXT)

        # Two variation calls: main config + judge config
        assert mock_ld_client.variation.call_count == 2
        assert result.judge_results is not None
        judge = result.judge_results["judge-key"]
        assert isinstance(judge, JudgeResult)
        # Same attribute reads as examples/conversation.py — dicts would print None.
        assert getattr(judge, "score", None) == 0.9
        assert getattr(judge, "response", None) == "good"


# ---------------------------------------------------------------------------
# invoke() judge_tasks (skip_judges=True)
# ---------------------------------------------------------------------------


class TestInvokeJudgeTasks:
    """Tests for JudgeTasks returned by invoke() when skip_judges=True."""

    def _make_multi_variation(
        self, main_variation: dict, judge_variation: dict
    ) -> AsyncMock:
        call_count = [0]

        async def multi(*args: object, **kwargs: object) -> dict:
            call_count[0] += 1
            return judge_variation if call_count[0] > 1 else main_variation

        return AsyncMock(side_effect=multi)

    def _main_variation_with_judge(self, judge_key: str = "judge-flag") -> dict:
        return {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "Be helpful.",
            "judgeConfiguration": {"judges": [{"key": judge_key, "samplingRate": 1.0}]},
            "_ldMeta": {
                "enabled": True,
                "variationKey": "v1",
                "version": 1,
                "mode": "messages",
            },
        }

    def _judge_variation(self, provider: str = "TestProvider") -> dict:
        return {
            "model": {"name": "gpt-4"},
            "provider": {"name": provider},
            "instructions": "You are a judge.",
            "evaluationMetricKey": "judge-score",
            "_ldMeta": {
                "enabled": True,
                "variationKey": "j1",
                "version": 1,
                "mode": "judge",
            },
        }

    async def test_returns_judge_task_with_correct_shape(
        self, mock_ld_client: MagicMock
    ) -> None:
        mock_ld_client.variation = self._make_multi_variation(
            self._main_variation_with_judge(), self._judge_variation()
        )
        h = ProviderHandler(
            fn=_make_handler()._fn, provides_for=("TestProvider", "messages")
        )
        result = await config(key="flag", handler=h, skip_judges=True).invoke(
            "q", CONTEXT
        )

        assert result.judge_results is None
        assert result.judge_tasks is not None
        assert len(result.judge_tasks) == 1
        task = result.judge_tasks[0]
        assert task.config_key == "judge-flag"
        assert task.evaluation_metric_key == "judge-score"
        assert task.judge_provider == "TestProvider"
        assert task.judge_mode == "messages"
        assert task.collapse_messages is False
        assert task.parent_track_data is not None
        assert task.parent_track_data.get("configKey") == "flag"

    async def test_sets_collapse_messages_for_agent_fallback(
        self, mock_ld_client: MagicMock
    ) -> None:
        # Main config: TestProvider agent mode (so agentHandler can serve it).
        # Judge config: TestProvider messages mode.
        # Only agent handler registered → collapseMessages=True.
        agent_main = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "hi",
            "judgeConfiguration": {
                "judges": [{"key": "judge-flag", "samplingRate": 1.0}]
            },
            "_ldMeta": {
                "enabled": True,
                "variationKey": "v1",
                "version": 1,
                "mode": "agent",
            },
        }
        judge_msgs = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "messages": [{"role": "system", "content": "Judge."}],
            "_ldMeta": {
                "enabled": True,
                "variationKey": "j1",
                "version": 1,
                "mode": "messages",
            },
        }
        mock_ld_client.variation = self._make_multi_variation(agent_main, judge_msgs)

        async def agent_fn(
            cfg, user_input, tool_handlers, variables, history=None
        ) -> dict:  # type: ignore[override]
            return {"output": "ok", "usage": {}}

        h = ProviderHandler(fn=agent_fn, provides_for=("TestProvider", "agent"))  # type: ignore[arg-type]
        result = await config(key="flag", handler=h, skip_judges=True).invoke(
            "q", CONTEXT
        )

        assert result.judge_tasks is not None
        assert len(result.judge_tasks) == 1
        assert result.judge_tasks[0].collapse_messages is True

    async def test_excludes_task_when_no_handler_matches_judge_provider(
        self, mock_ld_client: MagicMock
    ) -> None:
        # Judge config uses Anthropic but only TestProvider handler is registered.
        mock_ld_client.variation = self._make_multi_variation(
            self._main_variation_with_judge(),
            self._judge_variation(provider="Anthropic"),
        )
        h = ProviderHandler(
            fn=_make_handler()._fn, provides_for=("TestProvider", "messages")
        )
        result = await config(key="flag", handler=h, skip_judges=True).invoke(
            "q", CONTEXT
        )

        assert result.judge_tasks == []

    async def test_returns_empty_judge_tasks_when_no_judge_configuration(
        self, mock_ld_client: MagicMock
    ) -> None:
        no_judge_variation = {
            "model": {"name": "gpt-4"},
            "provider": {"name": "TestProvider"},
            "instructions": "Be helpful.",
            "_ldMeta": {
                "enabled": True,
                "variationKey": "v1",
                "version": 1,
                "mode": "messages",
            },
        }
        mock_ld_client.variation = AsyncMock(return_value=no_judge_variation)

        h = ProviderHandler(
            fn=_make_handler()._fn, provides_for=("TestProvider", "messages")
        )
        result = await config(key="flag", handler=h, skip_judges=True).invoke(
            "q", CONTEXT
        )

        assert result.judge_tasks == []
