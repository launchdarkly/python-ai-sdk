from __future__ import annotations

import importlib
import inspect
import logging
import os
from typing import Any

from . import skills
from .types import InitClientOptions

logger = logging.getLogger(__name__)

_LD_DEFAULT_OTLP_ENDPOINT = "https://otel.observability.app.launchdarkly.com"


def _env(name: str) -> str | None:
    """Read an env var, treating blank/whitespace-only values as unset."""
    value = os.environ.get(name, "").strip()
    return value if value else None


_client: Any = None
_tracer_provider: Any = None


def get_client() -> Any:
    """
    Returns the singleton LaunchDarkly client.
    Raises ``RuntimeError`` if ``init_client`` has not been called yet.
    """
    if _client is None:
        raise RuntimeError(
            "LaunchDarkly client not initialized. Call init_client() first."
        )
    return _client


def _setup_telemetry(sdk_key: str, options: InitClientOptions | None = None) -> Any:
    """
    Attempts to set up OpenTelemetry. Returns the tracer provider or None
    if OTel packages are not installed (graceful degradation).

    This function:
    - Stamps ``service.name``, ``highlight.project_id``, and (when set)
      ``deployment.environment`` resource attributes, mirroring the TS SDK.
    - Registers W3C trace context and baggage propagators.
    - Configures GZIP compression on the OTLP exporter.
    """
    global _tracer_provider

    opts = options or {}

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        try:
            from opentelemetry.exporter.otlp.proto.http import (
                Compression as CompressionAlgorithm,
            )
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            otlp_endpoint = (
                opts.get("otlpEndpoint")
                or _env("OTEL_EXPORTER_OTLP_ENDPOINT")
                or _LD_DEFAULT_OTLP_ENDPOINT
            )
            exporter: Any = OTLPSpanExporter(
                endpoint=f"{otlp_endpoint.rstrip('/')}/v1/traces",
                compression=CompressionAlgorithm.Gzip,
            )
        except ImportError:
            exporter = None

        resource_attrs: dict[str, str] = {
            "service.name": opts.get("serviceName")
            or os.environ.get("LD_SERVICE_NAME", "python-sdk"),
            "highlight.project_id": sdk_key,
        }
        environment = opts.get("environment") or os.environ.get("LD_ENVIRONMENT")
        if environment:
            resource_attrs["deployment.environment"] = environment

        resource = Resource.create(resource_attrs)
        provider = TracerProvider(resource=resource)
        from .conversation import ConversationIdSpanProcessor

        provider.add_span_processor(ConversationIdSpanProcessor())
        if exporter:
            provider.add_span_processor(BatchSpanProcessor(exporter))

        try:
            from opentelemetry import propagate
            from opentelemetry.baggage.propagation import W3CBaggagePropagator
            from opentelemetry.propagators.composite import CompositePropagator
            from opentelemetry.trace.propagation.tracecontext import (
                TraceContextTextMapPropagator,
            )

            propagate.set_global_textmap(
                CompositePropagator(
                    [TraceContextTextMapPropagator(), W3CBaggagePropagator()]
                )
            )
        except ImportError:
            pass

        trace.set_tracer_provider(provider)
        _tracer_provider = provider
        return provider

    except ImportError:
        logger.warning(
            "OpenTelemetry packages not installed. Run `pip install opentelemetry-sdk` "
            "to enable telemetry."
        )
        return None


async def init_client(
    options: InitClientOptions | None = None,
    client: Any = None,
) -> Any:
    """
    Initializes the singleton LaunchDarkly client.

    - Pass *client* directly (BYOC) to skip the LaunchDarkly Python SDK path.
    - Otherwise, reads ``LD_SDK_KEY`` from env or ``options['sdkKey']``.
    - ``options['skillStore']`` configures the store the Agent Skills accessors
      read from. Absent by default, in which case they raise an actionable error.

    This function is idempotent for the client singleton: a second call returns
    the existing client without re-initializing, and every option is ignored —
    **except** ``skillStore``, which is applied on every successful call. That
    asymmetry is deliberate, and it is what lets a client that was lazily
    auto-initialized, or initialized without a store, be given one afterwards.
    A ``skillStore`` of ``None`` (or absent) never clears an already-configured
    store; use ``shutdown()`` for that. The store is installed only once
    initialization has succeeded: a call that raises leaves no global state
    behind, so a failed init cannot leave the skill accessors working against a
    store the application believes was never installed.

    Returns the initialized ``LDClientInterface`` instance.
    """
    opts = options or {}

    ld_client = await _resolve_client(opts, client)

    # The single success point: every path that raises returns before here, so
    # "installed only on success" is one statement rather than a copy per exit.
    skill_store = opts.get("skillStore")
    if skill_store is not None:
        skills._set_store(skill_store)
    return ld_client


async def _resolve_client(opts: InitClientOptions, client: Any) -> Any:
    """
    Returns the singleton client, initializing it on first call.

    Split from ``init_client`` so that function has exactly one success point to
    hang the ``skillStore`` carve-out on.
    """
    global _client

    # Idempotent — if already initialized, return the existing client
    if _client is not None:
        return _client

    # BYOC path — pre-initialized client
    if client is not None:
        _client = client
        _setup_telemetry(opts.get("sdkKey", "byoc"), opts)
        return _client

    # Resolve SDK key
    sdk_key: str | None = opts.get("sdkKey") or os.environ.get("LD_SDK_KEY")
    if not sdk_key:
        raise RuntimeError(
            "No LaunchDarkly SDK key provided. Set LD_SDK_KEY env var or pass sdkKey in options."
        )

    # Load LD SDK dynamically (optional peer dep)
    try:
        ld_module = importlib.import_module("ldclient")
    except ImportError:
        try:
            ld_module = importlib.import_module("launchdarkly_server_sdk")
        except ImportError:
            raise RuntimeError(
                "LaunchDarkly server SDK not installed. "
                "Run `pip install launchdarkly-server-sdk` or pass a pre-initialized client."
            ) from None

    # Initialize LD client — Config wraps the SDK key and URI overrides; LDClient takes a Config.
    config_cls = getattr(ld_module, "Config", None)
    client_cls = getattr(ld_module, "LDClient", None)
    if config_cls is None or client_cls is None:
        raise RuntimeError(
            "Unexpected LaunchDarkly SDK structure; cannot initialize client."
        )

    config_kwargs: dict[str, str] = {}
    base_uri = opts.get("baseUri") or os.environ.get("LD_BASE_URI")
    stream_uri = opts.get("streamUri") or os.environ.get("LD_STREAM_URI")
    events_uri = opts.get("eventsUri") or os.environ.get("LD_EVENTS_URI")
    if base_uri:
        config_kwargs["base_uri"] = base_uri
    if stream_uri:
        config_kwargs["stream_uri"] = stream_uri
    if events_uri:
        config_kwargs["events_uri"] = events_uri

    ld_config = config_cls(sdk_key, **config_kwargs)
    # start_wait caps the blocking init time; matches the TS SDK's 10 s timeout.
    ld_client = client_cls(ld_config, start_wait=10)

    _client = ld_client
    _setup_telemetry(sdk_key, opts)
    return _client


async def shutdown() -> None:
    """
    Shuts down the singleton client. Idempotent — safe to call multiple times
    even if the client was never initialized or already shut down.

    Also clears the configured skill store (and telemetry emitter): after a
    shutdown, re-pass ``skillStore`` to the next ``init_client`` if the skill
    accessors should keep working.
    """
    global _client, _tracer_provider

    local_client = _client
    local_provider = _tracer_provider

    skills._clear_state()

    # Null the singleton before any awaits so a second call is a no-op
    _client = None
    _tracer_provider = None

    if local_provider is not None:
        try:
            local_provider.shutdown()
        except Exception:
            pass

    if local_client is not None:
        try:
            flush_result = local_client.flush()
            if inspect.isawaitable(flush_result):
                await flush_result
        except Exception:
            pass
        try:
            close_result = local_client.close()
            if inspect.isawaitable(close_result):
                await close_result
        except Exception:
            pass


def _set_client_for_testing(c: Any) -> None:
    """Test helper — inject a mock client without going through init_client."""
    global _client
    _client = c


def _reset_for_testing() -> None:
    """Test helper — clear all singleton state."""
    global _client, _tracer_provider
    _client = None
    _tracer_provider = None
    skills._clear_state()


async def inspect_config(
    config_key: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Reads an AI Config variation without invoking the model. Use this to
    inspect the current config state (enabled/disabled, model name, provider,
    etc.) for health checks, logging, or any purpose that doesn't need to
    actually run the AI provider.

    Unlike ``config().invoke()``, this function:

    - Never raises — returns ``{"enabled": False, "config": None, "meta": None}``
      on any error (unreachable LD, bad key, unparseable config, etc.)
    - Does not emit any LaunchDarkly telemetry events
    - Does not call any AI provider

    Returns a dict with keys:
    - ``enabled`` (bool): whether the flag variation is active
    - ``config`` (dict | None): the parsed AI config, or None when disabled/invalid
    - ``meta`` (dict | None): the variation metadata, or None when unreachable
    """
    from .types_validation import parse_ai_config
    from .utils import to_ld_context  # late import avoids circular dependency

    try:
        await init_client()
        client = get_client()
        ld_context = to_ld_context(client, context)
        variation_result = client.variation(config_key, ld_context, None)
        raw = (
            await variation_result
            if inspect.isawaitable(variation_result)
            else variation_result
        )

        if raw is None:
            return {"enabled": False, "config": None, "meta": None}

        ld_meta: dict[str, Any] = (
            raw.get("_ldMeta", {}) if isinstance(raw, dict) else {}
        )
        enabled = bool(ld_meta.get("enabled", False))
        meta: dict[str, Any] | None = ld_meta if ld_meta else None

        if not enabled:
            return {"enabled": False, "config": None, "meta": meta}

        config_raw = (
            {k: v for k, v in raw.items() if k != "_ldMeta"}
            if isinstance(raw, dict)
            else raw
        )
        result = parse_ai_config(config_raw)
        if not result.success:
            return {"enabled": True, "config": None, "meta": meta}

        return {"enabled": True, "config": result.data, "meta": meta}

    except Exception:
        return {"enabled": False, "config": None, "meta": None}


async def extract_variation(
    config_key: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Fetches an AI config variation from the LD client and parses it.
    Returns ``{"config": AiConfigRep, "meta": VariationMeta}``.
    Raises on disabled or invalid variations.

    Lazily initializes the LD client when ``LD_SDK_KEY`` is set, matching
    the TypeScript SDK's behaviour where any AI call auto-initializes.
    """
    from .types_validation import parse_ai_config
    from .utils import to_ld_context  # late import avoids circular dependency

    await init_client()
    client = get_client()
    # The real ldclient.variation() expects an ldclient.Context, not a plain dict.
    # Convert when running against the real SDK; test mocks accept dicts directly.
    ld_context = to_ld_context(client, context)
    # variation() is synchronous in the real Python LD SDK; AsyncMock in tests.
    variation_result = client.variation(config_key, ld_context, None)
    raw = (
        await variation_result
        if inspect.isawaitable(variation_result)
        else variation_result
    )

    if raw is None:
        raise RuntimeError(
            f"Variation '{config_key}' returned None (flag may be disabled)."
        )

    if isinstance(raw, dict) and raw.get("_ldMeta", {}).get("enabled") is False:
        raise RuntimeError(f"Variation '{config_key}' is disabled.")

    # Extract meta from _ldMeta if present
    ld_meta = raw.get("_ldMeta", {}) if isinstance(raw, dict) else {}
    meta = {
        "enabled": ld_meta.get("enabled", True),
        "variationKey": ld_meta.get("variationKey", ""),
        "version": ld_meta.get("version", 1),
        "mode": ld_meta.get("mode"),
    }

    # Strip _ldMeta for config parsing
    config_raw = (
        {k: v for k, v in raw.items() if k != "_ldMeta"}
        if isinstance(raw, dict)
        else raw
    )

    result = parse_ai_config(config_raw)
    if not result.success:
        raise RuntimeError(
            f"Invalid AI config variation for '{config_key}': {result.error['message']}"
        )

    return {"config": result.data, "meta": meta}
