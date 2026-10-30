from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import inspect
import logging
import os
import threading
from typing import Any

from .sdk_info import flush_ai_sdk_info, reset_ai_sdk_info
from .sdk_usage import flush_sdk_usage, report_usage, reset_sdk_usage
from .shutdown_hooks import run_shutdown_hooks
from .types import InitClientOptions, LDClientInterface, SDKOptions
from .utils import model_stamps_from_meta

logger = logging.getLogger(__name__)

_LD_DEFAULT_OTLP_ENDPOINT = "https://otel.observability.app.launchdarkly.com"

_INIT_CLIENT_OPTIONS = frozenset(
    {
        "sdkKey",
        "baseUri",
        "streamUri",
        "eventsUri",
        "otlpEndpoint",
        "serviceName",
        "environment",
    }
)
"""Every ``init_client`` option key that is read. Any other key is warned about."""


def _env(name: str) -> str | None:
    """Read an env var, treating blank/whitespace-only values as unset."""
    value = os.environ.get(name, "").strip()
    return value if value else None


class _InitAttempt:
    """One build of the LaunchDarkly client, shared by every call that arrives while it runs.

    ``result`` is a thread future, not an asyncio one, so callers on any event
    loop or thread can wait on it. ``waiters`` counts the calls still waiting.
    """

    def __init__(self) -> None:
        self.result: concurrent.futures.Future[LDClientInterface] = (
            concurrent.futures.Future()
        )
        self.waiters = 0


# Guards ``_client``, ``_attempt`` and the telemetry globals: the attempt thread
# adopts its client, and callers may arrive from several threads.
_state_lock = threading.Lock()
_client: LDClientInterface | None = None
# The in-flight attempt, cleared when it ends either way, so a failure is not
# cached. ``shutdown()`` and a passed client clear it too, which abandons it.
_attempt: _InitAttempt | None = None
_tracer_provider: Any = None
# True only when *this* SDK's call to trace.set_tracer_provider actually took
# effect. "We built a provider" is not the same as "we own the global": the set
# is once-guarded, so when another library registered first ours is refused and
# the global stays theirs. Only the owner may release it on shutdown.
_owns_otel_globals: bool = False
# Process-wide behavior defaults. Separate from ``_client``: ``init_client`` only
# builds the LaunchDarkly client and telemetry, and ``configure`` can run before
# that exists. ``_ai_invoked`` latches on the first handler call so a later
# ``configure`` cannot change what an in-flight process already started using.
_sdk_options: SDKOptions = SDKOptions()
_ai_invoked: bool = False


def configure(*, options: SDKOptions) -> None:
    """Sets process-wide SDK behavior defaults.

    Call once from process startup, before the first AI invocation, with an
    :class:`SDKOptions` value the process owns. Do not build it from request
    data. Handlers constructed without an explicit ``capture_content`` read the
    configured value when they run, including handlers built before this call.

    A later call may replace the options until the first AI invocation. After
    that, only an identical ``options`` value is accepted. ``shutdown()``
    restores the built-in defaults and allows configuration again.

    ``init_client`` is unchanged: it only initializes the LaunchDarkly client
    and telemetry.
    """
    global _sdk_options
    if not isinstance(options, SDKOptions):
        raise TypeError("configure() options must be an SDKOptions")
    if _ai_invoked and options != _sdk_options:
        raise RuntimeError(
            "SDK options cannot change after the first AI invocation. "
            "shutdown() restores the built-in defaults."
        )
    _sdk_options = options


def effective_capture_content(override: bool | None) -> bool:
    """Resolves content capture for one invocation.

    An explicit ``True`` or ``False`` — from a convenience wrapper, then from
    the handler factory — wins. ``None`` reads the configured default, which
    itself defaults to ``True``. An explicit ``False`` removes conversation
    content everywhere, including on offline evaluation spans.
    """
    if override is not None:
        return override
    return _sdk_options.capture_content


def note_ai_invocation() -> None:
    """Latches the process so ``configure`` can no longer change options."""
    global _ai_invoked
    _ai_invoked = True


def get_client() -> LDClientInterface:
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
    - Stamps ``service.name``, ``launchdarkly.project_id``, and (when set)
      ``deployment.environment`` resource attributes, mirroring the TS SDK.
    - Registers W3C trace context and baggage propagators.
    - Configures GZIP compression on the OTLP exporter.
    """
    global _tracer_provider, _owns_otel_globals

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
            "launchdarkly.project_id": sdk_key,
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
        # The set is refused, with a warning from OTel, when another library got
        # there first. Record whether it actually took: shutdown must not release
        # a global it never owned, and the caller's telemetry options are moot if
        # someone else's provider is the one handing out tracers.
        _owns_otel_globals = trace.get_tracer_provider() is provider
        if not _owns_otel_globals:
            logger.warning(
                "An OpenTelemetry tracer provider was already registered by "
                "something else in this process, so LaunchDarkly's telemetry "
                "configuration is not in effect; spans will go wherever that "
                "provider sends them."
            )
        _tracer_provider = provider
        return provider

    except ImportError:
        logger.warning(
            "OpenTelemetry packages not installed. Run "
            '`pip install "launchdarkly-ai-server[otel]"` to enable telemetry.'
        )
        return None


async def init_client(
    options: InitClientOptions | None = None,
    client: LDClientInterface | None = None,
) -> LDClientInterface:
    """
    Initializes the singleton LaunchDarkly client.

    - Pass *client* directly (BYOC) to skip the LaunchDarkly Python SDK path.
    - Otherwise, reads ``LD_SDK_KEY`` from env or ``options['sdkKey']``.

    Options (each overrides the environment variable in parentheses):

    - ``sdkKey`` (``LD_SDK_KEY``): the server-side SDK key.
    - ``baseUri`` (``LD_BASE_URI``), ``streamUri`` (``LD_STREAM_URI``),
      ``eventsUri`` (``LD_EVENTS_URI``): LaunchDarkly service URIs, passed to
      the SDK config. Not used on the BYOC path.
    - ``otlpEndpoint`` (``OTEL_EXPORTER_OTLP_ENDPOINT``): where spans are
      exported.
    - ``serviceName`` (``LD_SERVICE_NAME``): the ``service.name`` resource
      attribute.
    - ``environment`` (``LD_ENVIRONMENT``): the ``deployment.environment``
      resource attribute.

    Idempotent: later calls return the existing client and ignore every option.
    Calls made while the first one is still connecting wait for that same
    attempt, from any event loop or thread, and ignore their own options.
    An option not listed above is logged as a warning and ignored.

    Returns the initialized ``LDClientInterface`` instance. Raises
    ``RuntimeError`` when the LaunchDarkly client it builds has not initialized
    within 10 seconds or its SDK key was rejected, in every call waiting on that
    attempt; a later call retries. A client passed as *client* is not checked.
    """
    global _client, _attempt

    opts = options or {}
    unknown = sorted(str(key) for key in opts if key not in _INIT_CLIENT_OPTIONS)
    if unknown:
        logger.warning(
            "Ignoring unrecognized init_client option(s): %s", ", ".join(unknown)
        )

    # Idempotent — if already initialized, return the existing client
    if _client is not None:
        flush_ai_sdk_info(_client)
        flush_sdk_usage(_client)
        return _client

    # BYOC path — pre-initialized client
    if client is not None:
        with _state_lock:
            if _client is None:
                # Adopted only after telemetry setup succeeds. ``_client`` is the
                # idempotency guard above, so assigning it first meant a setup that
                # raised (a malformed OTEL_EXPORTER_OTLP_TIMEOUT does) left it set:
                # the next call returned it as a silent success with no telemetry,
                # hiding the config error. The caller owns this client, so it is
                # not closed.
                _setup_telemetry(opts.get("sdkKey", "byoc"), opts)
                _client = client
                # An attempt still connecting is abandoned: it closes the client
                # it builds, and the calls waiting on it get this one.
                _attempt = None
            adopted = _client
        flush_ai_sdk_info(adopted)
        flush_sdk_usage(adopted)
        return adopted

    with _state_lock:
        if _client is None and _attempt is None:
            _attempt = _start_attempt(opts)
        attempt = _attempt
        if attempt is not None:
            attempt.waiters += 1
        existing = _client
    if attempt is None:
        # Adopted by another call between the check above and the lock.
        assert existing is not None
        flush_ai_sdk_info(existing)
        flush_sdk_usage(existing)
        return existing

    try:
        ld_client = await _wait_for_attempt(attempt.result)
    finally:
        with _state_lock:
            attempt.waiters -= 1
    flush_ai_sdk_info(ld_client)
    flush_sdk_usage(ld_client)
    return ld_client


def _start_attempt(opts: InitClientOptions) -> _InitAttempt:
    """Starts building the LaunchDarkly client. The caller holds ``_state_lock``.

    Raises, with nothing started, when there is no SDK key or no SDK to build
    the client with.
    """
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
                'Run `pip install "launchdarkly-ai-server[ld]"` (installs launchdarkly-server-sdk) '
                "or pass a pre-initialized client."
            ) from None

    # Initialize LD client — Config wraps the SDK key and URI overrides; LDClient takes a Config.
    config_cls = getattr(ld_module, "Config", None)
    client_cls = getattr(ld_module, "LDClient", None)
    if config_cls is None or client_cls is None:
        raise RuntimeError(
            "Unexpected LaunchDarkly SDK structure; cannot initialize client."
        )

    config_kwargs: dict[str, str] = {}
    base_uri = opts.get("baseUri") or _env("LD_BASE_URI")
    stream_uri = opts.get("streamUri") or _env("LD_STREAM_URI")
    events_uri = opts.get("eventsUri") or _env("LD_EVENTS_URI")
    if base_uri:
        config_kwargs["base_uri"] = base_uri
    if stream_uri:
        config_kwargs["stream_uri"] = stream_uri
    if events_uri:
        config_kwargs["events_uri"] = events_uri

    ld_config = config_cls(sdk_key, **config_kwargs)
    attempt = _InitAttempt()
    # Its own thread, not asyncio.to_thread: the build blocks for up to
    # start_wait, and holding a default-executor thread per caller for that
    # long starved every other to_thread user. Not tied to any event loop
    # either, so the attempt outlives the loop of the call that started it.
    threading.Thread(
        target=_run_attempt,
        args=(attempt, client_cls, ld_config, sdk_key, opts),
        name="launchdarkly-ai-init-client",
        daemon=True,
    ).start()
    return attempt


def _run_attempt(
    attempt: _InitAttempt,
    client_cls: Any,
    ld_config: Any,
    sdk_key: str,
    opts: InitClientOptions,
) -> None:
    """Builds and adopts the client on the attempt's thread, then settles ``result``."""
    global _attempt

    ld_client: Any = None
    try:
        # start_wait caps the blocking init time; matches the TS SDK's 10 s timeout.
        ld_client = client_cls(ld_config, start_wait=10)
        adopted = _adopt(attempt, ld_client, sdk_key, opts)
    except BaseException as exc:
        # A failed attempt is not cached: drop it before any caller sees the
        # failure, so one that retries at once starts afresh. Close the client
        # too, or each retry leaves a streaming connection open.
        with _state_lock:
            if _attempt is attempt:
                _attempt = None
        if ld_client is not None:
            _close_unadopted(ld_client)
        attempt.result.set_exception(exc)
        return
    if adopted is not ld_client:
        _close_unadopted(ld_client)
    attempt.result.set_result(adopted)


def _adopt(
    attempt: _InitAttempt,
    ld_client: Any,
    sdk_key: str,
    opts: InitClientOptions,
) -> LDClientInterface:
    """Makes the attempt's client the singleton, or says what to return instead."""
    global _client, _attempt

    # The constructor returns normally both when start_wait runs out and when
    # the service rejects the SDK key; it only logs. Adopting that client would
    # serve every AI Config disabled with no error, so fail here.
    if not ld_client.is_initialized():
        raise RuntimeError(
            "LaunchDarkly client failed to initialize: it did not connect within "
            "10 seconds, or the SDK key was rejected. Check LD_SDK_KEY (or the "
            "sdkKey option) and that LaunchDarkly is reachable; the LaunchDarkly "
            "SDK's log has the cause."
        )

    with _state_lock:
        if _attempt is not attempt or attempt.waiters == 0:
            # shutdown() or a passed client replaced this attempt, or every call
            # waiting on it was cancelled. Adopting the client would clobber
            # what came after, or keep a connection open that nobody asked for.
            if _attempt is attempt:
                _attempt = None
            if _client is not None:
                return _client
            raise RuntimeError(
                "init_client was abandoned: shutdown() ran, or every call "
                "waiting on it was cancelled, before the client connected."
            )
        # As on the BYOC path: only a fully set-up client becomes the singleton.
        _setup_telemetry(sdk_key, opts)
        adopted: LDClientInterface = ld_client
        _client = adopted
        _attempt = None
        return adopted


async def _wait_for_attempt(
    result: concurrent.futures.Future[LDClientInterface],
) -> LDClientInterface:
    """Waits on the caller's event loop for an attempt other loops may share.

    The outcome is copied into a future of this loop alone, so cancelling this
    wait leaves the attempt running for the other callers.
    ``asyncio.wrap_future`` would cancel the shared future instead.
    """
    loop = asyncio.get_running_loop()
    waiter: asyncio.Future[LDClientInterface] = loop.create_future()

    def copy() -> None:
        if waiter.done():  # this wait was cancelled
            return
        exc = result.exception()
        if exc is not None:
            waiter.set_exception(exc)
        else:
            waiter.set_result(result.result())

    def wake(_: concurrent.futures.Future[LDClientInterface]) -> None:
        try:
            loop.call_soon_threadsafe(copy)
        except RuntimeError:
            pass  # this caller's loop has closed, so nothing waits on it

    result.add_done_callback(wake)
    return await waiter


def _close_unadopted(ld_client: Any) -> None:
    """Closes, on the attempt thread, a client ``init_client`` built and will not adopt.

    An error from ``close`` is swallowed so it cannot replace the reason the
    client is being discarded.
    """
    try:
        close_result = ld_client.close()
        if inspect.isawaitable(close_result):
            asyncio.run(_await(close_result))
    except Exception:
        pass


async def _await(awaitable: Any) -> None:
    await awaitable


def _release_otel_globals() -> None:
    """
    Releases the process-global tracer provider that ``_setup_telemetry``
    installed, so a later ``init_client`` can install its own.

    ``trace.set_tracer_provider`` is guarded by a ``Once``: a second call logs
    "Overriding of current TracerProvider is not allowed" and keeps the provider
    already in place. Without this, an init/shutdown/init cycle would leave every
    span routed to the provider that was already shut down, and export nothing.

    opentelemetry-python exposes no public API to unset it, so this reaches for
    the module globals — both the slot and the ``Once`` that guards it, since
    clearing the slot alone leaves the guard tripped and the next set a no-op.

    Callers must gate this on ``_owns_otel_globals``. Having built a provider is
    not enough: when another library registered first, our set was refused and
    the global is still theirs, so releasing it here would tear down the host
    application's tracing and leave the global a no-op proxy.

    The global text map propagator needs no equivalent: ``set_global_textmap``
    is a plain assignment with no ``Once``, so the next setup overwrites it.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.util._once import Once

        trace._TRACER_PROVIDER = None
        trace._TRACER_PROVIDER_SET_ONCE = Once()
    except Exception:  # pragma: no cover - defensive, OTel absent or restructured
        logger.debug("Could not release the global OTel tracer provider", exc_info=True)


def _sdk_processor_is_active() -> bool:
    """Whether the active tracer provider carries ``ConversationIdSpanProcessor``.

    Only then does ``count_sdk_spans`` see a handler's spans. Under a foreign
    provider the count stays 0 even when the handler traced itself, so a
    caller that falls back to writing spans on a 0 count would write them twice.
    """
    if _owns_otel_globals:
        return True
    try:
        from opentelemetry import trace
    except ImportError:
        return False
    from .conversation import ConversationIdSpanProcessor

    # ponytail: reads the SDK TracerProvider's private processor list; a provider
    # without it (proxy, vendor wrapper) reads as foreign, which only skips the fallback.
    multi = getattr(trace.get_tracer_provider(), "_active_span_processor", None)
    processors = getattr(multi, "_span_processors", ())
    return any(isinstance(p, ConversationIdSpanProcessor) for p in processors)


def _flush_tracer_provider() -> None:
    """Export spans queued on the provider ``init_client`` registered.

    A provider without ``force_flush`` (the API proxy) is a no-op.
    """
    try:
        from opentelemetry import trace
    except ImportError:
        return
    flush = getattr(trace.get_tracer_provider(), "force_flush", None)
    if flush is None:
        return
    try:
        flush()
    except Exception:
        logger.debug("Could not flush OpenTelemetry spans", exc_info=True)


async def shutdown() -> None:
    """
    Shuts down the singleton client. Idempotent — safe to call multiple times
    even if the client was never initialized or already shut down.

    Also runs the registered shutdown hooks, which clear feature state such as
    the configured skill store, and restores process-wide SDK defaults from
    ``configure``.

    When telemetry was running, this also releases the process-global tracer
    provider so a later ``init_client`` can install its own — see
    ``_release_otel_globals``.
    """
    global _client, _attempt, _tracer_provider, _owns_otel_globals
    global _sdk_options, _ai_invoked

    run_shutdown_hooks()

    # Null the singleton before any awaits so a second call is a no-op.
    # Dropping the attempt abandons one still connecting: it closes the client
    # it builds instead of adopting it after this shutdown.
    # Options reset even when no client was initialized: configure() does not
    # require one, and shutdown is what puts the process back to built-in defaults.
    with _state_lock:
        local_client = _client
        local_provider = _tracer_provider
        owned_globals = _owns_otel_globals
        _client = None
        _attempt = None
        _tracer_provider = None
        _owns_otel_globals = False
    _sdk_options = SDKOptions()
    _ai_invoked = False
    reset_ai_sdk_info()
    reset_sdk_usage()

    if local_provider is not None:
        # Shut the provider down either way — we built it, and it owns an
        # exporter and a batch timer — but only release the global registration
        # when it was ours to take.
        try:
            local_provider.shutdown()
        except Exception:
            pass
        if owned_globals:
            _release_otel_globals()

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
    global _client, _attempt, _tracer_provider, _owns_otel_globals
    global _sdk_options, _ai_invoked
    with _state_lock:
        owned_globals = _owns_otel_globals
        _client = None
        _attempt = None
        _tracer_provider = None
        _owns_otel_globals = False
    _sdk_options = SDKOptions()
    _ai_invoked = False
    # A failed reset should fail the test rather than leak state into the next one.
    run_shutdown_hooks(raise_errors=True)
    # Mirrors shutdown(): without this a suite that inits more than once leaves
    # every later span on the first test's provider.
    if owned_globals:
        _release_otel_globals()


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
    - Does not emit generation, duration, or token tracking events
    - Does not call any AI provider

    Returns a dict with keys:
    - ``enabled`` (bool): whether the flag variation is active
    - ``config`` (dict | None): the parsed AI config, or None when disabled/invalid
    - ``meta`` (dict | None): the variation metadata, or None when unreachable
    """
    report_usage("client.inspectConfig")
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
    Returns ``{"config": AiConfigRep, "meta": VariationMeta, "value": Any}``.
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

    # Only an explicit ``_ldMeta.enabled`` counts as enabled (as in JS and Go).
    ld_meta = (raw.get("_ldMeta") or {}) if isinstance(raw, dict) else {}
    if not isinstance(ld_meta, dict) or not ld_meta.get("enabled"):
        raise RuntimeError(f"Variation '{config_key}' is disabled.")

    meta = {
        "enabled": True,
        "variationKey": ld_meta.get("variationKey", ""),
        "version": ld_meta.get("version", 1),
        "mode": ld_meta.get("mode"),
        # Pinned model-config identity; keys are omitted when absent so that
        # tracking payloads never carry ``None`` values.
        **model_stamps_from_meta(ld_meta),
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

    # ``value`` is the variation exactly as the LD SDK returned it, ``_ldMeta``
    # included. It stays off ``meta``, which surfaces publicly as
    # ``GraphNode.meta`` and ``JudgeTask.judge_meta``.
    return {"config": result.data, "meta": meta, "value": raw}
