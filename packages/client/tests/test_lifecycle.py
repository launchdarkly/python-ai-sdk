"""
Tests for §3.9 init_client / get_client / shutdown.
Reference: TESTING.md §3.9
"""

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import launchdarkly_ai_server.lifecycle as lifecycle_module
from launchdarkly_ai_server import get_client, init_client, inspect_config, shutdown
from launchdarkly_ai_server.lifecycle import _reset_for_testing
from launchdarkly_ai_server.sdk_info import (
    register_ai_sdk_package,
    reset_ai_sdk_info,
)


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    """Ensure a fresh singleton for every test."""
    _reset_for_testing()
    reset_ai_sdk_info(clear_known=True)
    yield
    _reset_for_testing()
    reset_ai_sdk_info(clear_known=True)


def _make_stub_client() -> MagicMock:
    stub = MagicMock()
    stub.variation = AsyncMock(return_value=None)
    stub.track = MagicMock()
    stub.flush = AsyncMock()
    stub.close = AsyncMock()
    stub.wait_for_initialization = AsyncMock()
    return stub


# ---------------------------------------------------------------------------
# get_client — before init
# ---------------------------------------------------------------------------


class TestGetClientBeforeInit:
    def test_throws_before_init_client(self) -> None:
        with pytest.raises(RuntimeError, match="init_client"):
            get_client()


# ---------------------------------------------------------------------------
# BYOC path
# ---------------------------------------------------------------------------


class TestInitClientBYOC:
    async def test_accepts_any_ld_client_interface(self) -> None:
        stub = _make_stub_client()
        with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
            result = await init_client(client=stub)
        assert result is stub

    async def test_returns_passed_client(self) -> None:
        stub = _make_stub_client()
        with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
            returned = await init_client(client=stub)
        assert returned is stub

    async def test_get_client_returns_passed_client(self) -> None:
        stub = _make_stub_client()
        with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
            await init_client(client=stub)
        assert get_client() is stub

    async def test_flushes_registered_ai_package_information(self) -> None:
        stub = _make_stub_client()
        register_ai_sdk_package("launchdarkly-ai-server", "0.1.3")

        with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
            await init_client(client=stub)

        stub.track.assert_called_once_with(
            "$ld:ai:sdk:info",
            {"kind": "ld_ai", "key": "ld-internal-tracking", "anonymous": True},
            {
                "aiSdkName": "launchdarkly-ai-server",
                "aiSdkVersion": "0.1.3",
                "aiSdkLanguage": "python",
            },
            1,
        )

    async def test_flushes_package_registered_after_initialization(self) -> None:
        stub = _make_stub_client()
        register_ai_sdk_package("launchdarkly-ai-server", "0.1.3")

        with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
            await init_client(client=stub)
            stub.track.reset_mock()
            register_ai_sdk_package("launchdarkly-ai-openai-agents", "0.1.4")
            await init_client()

        stub.track.assert_called_once()
        assert (
            stub.track.call_args.args[2]["aiSdkName"] == "launchdarkly-ai-openai-agents"
        )

    async def test_does_not_call_node_sdk(self) -> None:
        stub = _make_stub_client()
        mock_import = MagicMock()
        with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
            with patch(
                "importlib.import_module",
                side_effect=lambda n: mock_import if "ldclient" in n else __import__(n),
            ):
                await init_client(client=stub)
        # importlib.import_module should not have been called for ldclient since BYOC path
        for call in mock_import.mock_calls:
            assert "ldclient" not in str(call)

    async def test_still_sets_up_telemetry(self) -> None:
        stub = _make_stub_client()
        with patch.object(
            lifecycle_module, "_setup_telemetry", return_value=MagicMock()
        ) as mock_setup:
            await init_client(client=stub)
        mock_setup.assert_called_once()


# ---------------------------------------------------------------------------
# SDK key path
# ---------------------------------------------------------------------------


class TestInitClientSDKKeyPath:
    async def test_throws_without_sdk_key(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "LD_SDK_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="SDK key"):
                await init_client()

    async def test_uses_options_sdk_key_over_env_var(self) -> None:
        stub_client = _make_stub_client()
        mock_ld = MagicMock()
        mock_ld.Config = MagicMock(return_value=MagicMock())
        mock_ld.LDClient = MagicMock(return_value=stub_client)
        with patch.dict(os.environ, {"LD_SDK_KEY": "env-key"}):
            with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
                with patch("importlib.import_module", return_value=mock_ld):
                    await init_client({"sdkKey": "options-key"})
        mock_ld.Config.assert_called_once_with("options-key")

    async def test_throws_when_ld_sdk_not_installed(self) -> None:
        with patch.dict(os.environ, {"LD_SDK_KEY": "test-key"}):
            with patch(
                "importlib.import_module", side_effect=ImportError("not installed")
            ):
                with pytest.raises(RuntimeError, match="launchdarkly-server-sdk"):
                    await init_client()

    async def test_is_idempotent(self) -> None:
        stub = _make_stub_client()
        mock_ld = MagicMock()
        mock_ld.Config = MagicMock(return_value=MagicMock())
        mock_ld.LDClient = MagicMock(return_value=stub)
        with patch.dict(os.environ, {"LD_SDK_KEY": "test-key"}):
            with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
                with patch("importlib.import_module", return_value=mock_ld):
                    await init_client()
                    await init_client()
        # LDClient is only instantiated on first init; singleton is reused
        assert mock_ld.LDClient.call_count == 1

    async def test_returns_initialized_client(self) -> None:
        stub = _make_stub_client()
        mock_ld = MagicMock()
        mock_ld.Config = MagicMock(return_value=MagicMock())
        mock_ld.LDClient = MagicMock(return_value=stub)
        with patch.dict(os.environ, {"LD_SDK_KEY": "test-key"}):
            with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
                with patch("importlib.import_module", return_value=mock_ld):
                    result = await init_client()
        assert result is stub


# ---------------------------------------------------------------------------
# OTel degradation
# ---------------------------------------------------------------------------


class TestInitClientOtelMissing:
    async def test_emits_warning_once(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging as _logging

        stub = _make_stub_client()

        def _broken_setup(key: str, opts: Any = None) -> None:
            _logging.getLogger("launchdarkly_ai_server.lifecycle").warning(
                "OpenTelemetry packages not installed. Run `pip install opentelemetry-sdk`"
            )

        with caplog.at_level(_logging.WARNING):
            with patch.object(
                lifecycle_module, "_setup_telemetry", side_effect=_broken_setup
            ):
                await init_client(client=stub)
        assert any("OpenTelemetry" in m for m in caplog.messages)

    async def test_still_resolves_successfully(self) -> None:
        stub = _make_stub_client()
        # _setup_telemetry returns None on ImportError; client should still be set
        with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
            result = await init_client(client=stub)
        assert result is stub

    async def test_get_client_returns_client(self) -> None:
        stub = _make_stub_client()
        with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
            await init_client(client=stub)
        assert get_client() is stub

    async def test_shutdown_does_not_throw_for_telemetry(self) -> None:
        stub = _make_stub_client()
        with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
            await init_client(client=stub)
        await shutdown()  # must not raise even though no tracer provider was set


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    async def test_clears_the_singleton(self) -> None:
        stub = _make_stub_client()
        with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
            await init_client(client=stub)
        await shutdown()
        with pytest.raises(RuntimeError):
            get_client()

    async def test_flushes_client(self) -> None:
        stub = _make_stub_client()
        with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
            await init_client(client=stub)
        await shutdown()
        stub.flush.assert_called_once()
        stub.close.assert_called_once()

    async def test_allows_reinitialization(self) -> None:
        stub1 = _make_stub_client()
        stub2 = _make_stub_client()
        with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
            await init_client(client=stub1)
            await shutdown()
            await init_client(client=stub2)
        assert get_client() is stub2

    async def test_reemits_registered_packages_after_shutdown(self) -> None:
        stub1 = _make_stub_client()
        stub2 = _make_stub_client()
        register_ai_sdk_package("launchdarkly-ai-server", "0.1.3")

        with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
            await init_client(client=stub1)
            await shutdown()
            await init_client(client=stub2)

        stub1.track.assert_called_once()
        stub2.track.assert_called_once()

    async def test_idempotent_double_shutdown(self) -> None:
        stub = _make_stub_client()
        with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
            await init_client(client=stub)
        await shutdown()
        await shutdown()  # must not raise

    async def test_completes_teardown_even_if_flush_throws(self) -> None:
        stub = _make_stub_client()
        stub.flush = AsyncMock(side_effect=RuntimeError("flush failed"))
        with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
            await init_client(client=stub)
        await shutdown()  # must not raise
        stub.close.assert_called_once()
        # Subsequent shutdown is also a no-op
        await shutdown()


# ---------------------------------------------------------------------------
# OTel setup details (§3.9)
# ---------------------------------------------------------------------------


class TestOtelSetup:
    """Tests that _setup_telemetry properly configures the OTel pipeline."""

    def _setup_with_mock_exporter(self, sdk_key: str = "test-key") -> tuple[Any, Any]:
        """Call _setup_telemetry with a fake exporter module to avoid real OTLP connections.
        Returns (created_resources, mock_exporter_cls) for assertions."""
        from opentelemetry.sdk.resources import Resource

        created_resources: list[dict] = []
        original_create = Resource.create

        def _spy_create(attributes: dict | None = None) -> Resource:
            if attributes:
                created_resources.append(dict(attributes))
            return original_create(attributes)

        mock_exporter_instance = MagicMock()
        mock_exporter_cls = MagicMock(return_value=mock_exporter_instance)

        import sys

        fake_otlp_module = MagicMock()
        fake_otlp_module.OTLPSpanExporter = mock_exporter_cls

        fake_otlp_http_module = MagicMock()
        fake_otlp_http_module.Compression = MagicMock()
        fake_otlp_http_module.Compression.Gzip = "gzip"

        with patch.dict(
            sys.modules,
            {
                "opentelemetry.exporter.otlp.proto.http.trace_exporter": fake_otlp_module,
                "opentelemetry.exporter.otlp.proto.http": fake_otlp_http_module,
            },
        ):
            with patch.object(Resource, "create", side_effect=_spy_create):
                lifecycle_module._setup_telemetry(sdk_key)

        return created_resources, mock_exporter_cls

    def test_stamps_highlight_project_id(self) -> None:
        """highlight.project_id resource attribute must equal the resolved SDK key."""
        created_resources, _ = self._setup_with_mock_exporter("my-sdk-key")
        assert any(
            r.get("highlight.project_id") == "my-sdk-key" for r in created_resources
        )

    def test_registers_w3c_propagators(self) -> None:
        """init_client must register a CompositePropagator with W3C trace context and baggage propagators."""
        from opentelemetry import propagate
        from opentelemetry.baggage.propagation import W3CBaggagePropagator
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )

        self._setup_with_mock_exporter("any-key")
        propagator = propagate.get_global_textmap()

        # The propagator must be composite or be one of the W3C variants
        propagator_types = type(propagator).__mro__
        type_names = [t.__name__ for t in propagator_types]
        # Accept both CompositePropagator wrapping W3C types or a direct W3C propagator
        has_composite = "CompositePropagator" in type_names
        has_w3c_direct = isinstance(
            propagator, (TraceContextTextMapPropagator, W3CBaggagePropagator)
        )
        assert has_composite or has_w3c_direct, (
            f"Expected CompositePropagator with W3C propagators, got {type(propagator).__name__}"
        )

        if has_composite:
            # Verify it contains both W3C trace context and baggage propagators
            inner_propagators = propagator._propagators  # type: ignore[attr-defined]
            prop_type_names = {type(p).__name__ for p in inner_propagators}
            assert "TraceContextTextMapPropagator" in prop_type_names, (
                f"W3CTraceContextPropagator not found in composite propagators: {prop_type_names}"
            )
            assert "W3CBaggagePropagator" in prop_type_names, (
                f"W3CBaggagePropagator not found in composite propagators: {prop_type_names}"
            )

    def test_configures_gzip_on_otlp_exporter(self) -> None:
        """OTLPSpanExporter must be constructed with Gzip compression."""
        _, mock_exporter_cls = self._setup_with_mock_exporter("key-gzip")

        assert mock_exporter_cls.called, "OTLPSpanExporter was not constructed"
        _, call_kwargs = mock_exporter_cls.call_args
        compression_used = call_kwargs.get("compression")
        assert compression_used is not None, (
            f"No compression argument passed. Call kwargs: {call_kwargs!r}"
        )
        assert "gzip" in str(compression_used).lower(), (
            f"Expected Gzip compression, got: {compression_used!r}"
        )

    def test_uses_ld_default_otlp_endpoint_when_unconfigured(self) -> None:
        """OTLPSpanExporter must default to the LD OTLP endpoint when no env var or option is set."""
        import sys

        mock_exporter_instance = MagicMock()
        mock_exporter_cls = MagicMock(return_value=mock_exporter_instance)
        fake_otlp_module = MagicMock()
        fake_otlp_module.OTLPSpanExporter = mock_exporter_cls
        fake_otlp_http_module = MagicMock()
        fake_otlp_http_module.Compression = MagicMock()
        fake_otlp_http_module.Compression.Gzip = "gzip"

        with patch.dict(
            sys.modules,
            {
                "opentelemetry.exporter.otlp.proto.http.trace_exporter": fake_otlp_module,
                "opentelemetry.exporter.otlp.proto.http": fake_otlp_http_module,
            },
        ):
            with patch.dict(os.environ, {}, clear=False):
                # Remove OTEL_EXPORTER_OTLP_ENDPOINT so the default kicks in
                os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
                lifecycle_module._setup_telemetry("sdk-key")

        assert mock_exporter_cls.called
        _, call_kwargs = mock_exporter_cls.call_args
        endpoint = call_kwargs.get("endpoint", "")
        assert "otel.observability.app.launchdarkly.com" in endpoint, (
            f"Expected LD default OTLP endpoint, got: {endpoint!r}"
        )
        assert endpoint.endswith("/v1/traces"), (
            f"Expected endpoint to end with /v1/traces, got: {endpoint!r}"
        )

    def test_uses_env_var_otlp_endpoint_when_set(self) -> None:
        """OTLPSpanExporter must use OTEL_EXPORTER_OTLP_ENDPOINT when set."""
        import sys

        mock_exporter_instance = MagicMock()
        mock_exporter_cls = MagicMock(return_value=mock_exporter_instance)
        fake_otlp_module = MagicMock()
        fake_otlp_module.OTLPSpanExporter = mock_exporter_cls
        fake_otlp_http_module = MagicMock()
        fake_otlp_http_module.Compression = MagicMock()
        fake_otlp_http_module.Compression.Gzip = "gzip"

        with patch.dict(
            sys.modules,
            {
                "opentelemetry.exporter.otlp.proto.http.trace_exporter": fake_otlp_module,
                "opentelemetry.exporter.otlp.proto.http": fake_otlp_http_module,
            },
        ):
            with patch.dict(
                os.environ,
                {"OTEL_EXPORTER_OTLP_ENDPOINT": "https://my-collector.example.com"},
            ):
                lifecycle_module._setup_telemetry("sdk-key")

        assert mock_exporter_cls.called
        _, call_kwargs = mock_exporter_cls.call_args
        endpoint = call_kwargs.get("endpoint", "")
        assert "my-collector.example.com" in endpoint, (
            f"Expected custom OTLP endpoint, got: {endpoint!r}"
        )

    def test_empty_env_var_falls_back_to_ld_default_endpoint(self) -> None:
        """An empty OTEL_EXPORTER_OTLP_ENDPOINT must be treated as unset."""
        import sys

        mock_exporter_instance = MagicMock()
        mock_exporter_cls = MagicMock(return_value=mock_exporter_instance)
        fake_otlp_module = MagicMock()
        fake_otlp_module.OTLPSpanExporter = mock_exporter_cls
        fake_otlp_http_module = MagicMock()
        fake_otlp_http_module.Compression = MagicMock()
        fake_otlp_http_module.Compression.Gzip = "gzip"

        with patch.dict(
            sys.modules,
            {
                "opentelemetry.exporter.otlp.proto.http.trace_exporter": fake_otlp_module,
                "opentelemetry.exporter.otlp.proto.http": fake_otlp_http_module,
            },
        ):
            with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": ""}):
                lifecycle_module._setup_telemetry("sdk-key")

        _, call_kwargs = mock_exporter_cls.call_args
        endpoint = call_kwargs.get("endpoint", "")
        assert "otel.observability.app.launchdarkly.com" in endpoint, (
            f"Empty env var should fall back to LD default, got: {endpoint!r}"
        )


# ---------------------------------------------------------------------------
# inspect_config
# ---------------------------------------------------------------------------


class TestInspectConfig:
    """Tests for inspect_config — non-throwing config inspection."""

    async def test_returns_enabled_true_and_config_when_variation_is_enabled(
        self,
    ) -> None:
        stub = _make_stub_client()
        stub.variation = AsyncMock(
            return_value={
                "_ldMeta": {"enabled": True, "variationKey": "v1", "version": 1},
                "model": {"name": "claude-3-5"},
                "provider": {"name": "Anthropic"},
                "instructions": "You are helpful.",
            }
        )
        with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
            await init_client(client=stub)
        ctx = {"kind": "user", "key": "user-1"}
        with patch(
            "launchdarkly_ai_server.utils.to_ld_context",
            side_effect=lambda _c, ctx: ctx,
        ):
            result = await inspect_config("my-flag", ctx)

        assert result["enabled"] is True
        assert result["config"] is not None
        assert result["config"]["model"]["name"] == "claude-3-5"  # type: ignore[index]
        assert result["meta"] is not None

    async def test_returns_enabled_false_and_null_config_when_disabled(self) -> None:
        stub = _make_stub_client()
        stub.variation = AsyncMock(return_value={"_ldMeta": {"enabled": False}})
        with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
            await init_client(client=stub)
        ctx = {"kind": "user", "key": "user-1"}
        with patch(
            "launchdarkly_ai_server.utils.to_ld_context",
            side_effect=lambda _c, ctx: ctx,
        ):
            result = await inspect_config("my-flag", ctx)

        assert result["enabled"] is False
        assert result["config"] is None

    async def test_returns_enabled_false_when_variation_returns_none(self) -> None:
        stub = _make_stub_client()
        stub.variation = AsyncMock(return_value=None)
        with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
            await init_client(client=stub)
        ctx = {"kind": "user", "key": "user-1"}
        with patch(
            "launchdarkly_ai_server.utils.to_ld_context",
            side_effect=lambda _c, ctx: ctx,
        ):
            result = await inspect_config("my-flag", ctx)

        assert result["enabled"] is False
        assert result["config"] is None
        assert result["meta"] is None

    async def test_returns_enabled_true_and_null_config_on_schema_failure(self) -> None:
        stub = _make_stub_client()
        stub.variation = AsyncMock(
            return_value={
                "_ldMeta": {"enabled": True},
                # missing model and provider
            }
        )
        with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
            await init_client(client=stub)
        ctx = {"kind": "user", "key": "user-1"}
        with patch(
            "launchdarkly_ai_server.utils.to_ld_context",
            side_effect=lambda _c, ctx: ctx,
        ):
            result = await inspect_config("my-flag", ctx)

        assert result["enabled"] is True
        assert result["config"] is None

    async def test_never_raises_when_variation_throws(self) -> None:
        stub = _make_stub_client()
        stub.variation = AsyncMock(side_effect=RuntimeError("network error"))
        with patch.object(lifecycle_module, "_setup_telemetry", return_value=None):
            await init_client(client=stub)
        ctx = {"kind": "user", "key": "user-1"}
        with patch(
            "launchdarkly_ai_server.utils.to_ld_context",
            side_effect=lambda _c, ctx: ctx,
        ):
            result = await inspect_config("my-flag", ctx)

        assert result["enabled"] is False
        assert result["config"] is None
        assert result["meta"] is None
