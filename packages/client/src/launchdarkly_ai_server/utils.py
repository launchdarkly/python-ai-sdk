from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from .types import (
    AiConfigRep,
    GraphNode,
    InputTokenDetails,
    ProviderHandler,
    UsageDict,
    VariationMeta,
    _HandlerFn,
    _StreamFn,
)


def create_handler(
    provides_for: tuple[str, Literal["agent", "messages"]],
    fn: _HandlerFn,
    stream_fn: _StreamFn | None = None,
    capture_content: bool = False,
) -> ProviderHandler:
    """
    Wraps a plain async callable in a :class:`ProviderHandler` with the given
    ``provides_for`` metadata and optional streaming implementation.
    """
    return ProviderHandler(
        fn=fn,
        provides_for=provides_for,
        stream_fn=stream_fn,
        capture_content=capture_content,
    )


def collapse_messages_to_instructions(config: AiConfigRep) -> AiConfigRep:
    """
    When only an agent handler is available for a messages-mode config, collapse
    all messages into a single ``instructions`` string so the agent handler receives
    a well-formed prompt without requiring a separate messages client to be
    registered. The original config is returned unchanged when ``instructions`` is
    already present or there are no messages.
    """
    messages = config.get("messages") or []
    if not messages or config.get("instructions"):
        return config
    instructions = "\n\n".join(m.get("content", "") for m in messages)
    return {**config, "instructions": instructions, "messages": []}


def normalize_mode(mode: str | None) -> Literal["agent", "messages"]:
    """
    Maps a variation mode string to the two handler mode values.
    ``'agent'`` → ``'agent'``; everything else → ``'messages'``.
    """
    return "agent" if mode == "agent" else "messages"


def number_or_zero(value: Any) -> int:
    """Coerces a provider-reported token count to a finite number, defaulting to 0.

    Provider SDKs report usage as loosely typed bags where a field may be absent, ``None``, or (in
    streaming paths) a partially populated value. Every count that reaches a span or a usage total
    should pass through here.

    An emitted 0 is bad. An emitted ``NaN`` is worse, because the metric guard tests whether the
    total is greater than zero, and that test is false for ``NaN``, so the metric is dropped
    silently rather than reported low.

    Replaces the bare ``int(...)`` this module used to do, which raised on ``None``.
    """
    if value is None or isinstance(value, bool):
        return 0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return 0
    return int(parsed)


#: Provider usage field names, tried in order; the first row whose input *and* output keys are both
#: present wins. Cache fields list every spelling we accept, because providers disagree (Anthropic
#: ``cache_creation_input_tokens``, Bedrock Converse ``cacheWriteInputTokens``) and an unlisted
#: spelling is silently dropped from the total.
#:
#: Contract for handlers: this fold is provider-blind, so it always adds the cache fields on top of
#: the input field. A handler must therefore return *either* raw usage with the cache fields intact
#: (Anthropic-style reporting, where cache is genuinely additional), *or* an input figure that
#: already includes cache with the cache fields omitted (OpenAI- and LangChain-style reporting).
#: Returning a pre-folded input *alongside* the cache fields double-counts the cached portion.
_USAGE_KEY_PAIRS: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = [
    (
        "input_tokens",
        "output_tokens",
        ("cache_read_input_tokens",),
        ("cache_creation_input_tokens",),
    ),
    (
        "inputTokens",
        "outputTokens",
        ("cacheReadInputTokens",),
        ("cacheCreationInputTokens", "cacheWriteInputTokens"),
    ),
    ("input", "output", (), ()),
]


def _read_cache_field(usage: dict[str, Any], keys: tuple[str, ...]) -> int:
    """Sums every accepted spelling of a cache field, so an alias never silently reads as 0."""
    return sum(number_or_zero(usage.get(key)) for key in keys)


def parse_usage(usage: dict[str, Any]) -> dict[str, Any]:
    """Normalises token counts from three possible key-pair shapes, folding cache tokens in.

    Always computes ``total`` from ``input + output``; never trusts a ``total`` field from the raw
    object. A provider-supplied total can include tokens that appear in neither input nor output,
    which would make the figure derivable on five handlers and not on the sixth.

    ``input`` is the inclusive total: every accepted cache spelling is added on top of the reported
    input figure. See the contract note on ``_USAGE_KEY_PAIRS``.

    When the matched row defines cache keys and at least one of them is present, the result also
    carries ``input_details`` with ``uncached``, ``cache_read`` and ``cache_creation``.

    An unrecognised bag returns all zeros rather than raising.
    """
    for input_key, output_key, cache_read_keys, cache_creation_keys in _USAGE_KEY_PAIRS:
        if input_key in usage and output_key in usage:
            cache_read = _read_cache_field(usage, cache_read_keys)
            cache_creation = _read_cache_field(usage, cache_creation_keys)
            uncached = number_or_zero(usage[input_key])
            inp = uncached + cache_read + cache_creation
            out = number_or_zero(usage[output_key])
            result: dict[str, Any] = {"input": inp, "output": out, "total": inp + out}
            has_details = bool(cache_read_keys or cache_creation_keys) and any(
                key in usage for key in (*cache_read_keys, *cache_creation_keys)
            )
            if has_details:
                result["input_details"] = {
                    "uncached": uncached,
                    "cache_read": cache_read,
                    "cache_creation": cache_creation,
                }
            return result
    return {"input": 0, "output": 0, "total": 0}


def to_usage_dict(usage: dict[str, Any]) -> UsageDict:
    """Builds the public :class:`UsageDict` from a :func:`parse_usage` result.

    Shared because ``invoke`` and the judge runner both need it and both used to build the dataclass
    by hand from three keys, which silently dropped the cache breakdown the moment ``parse_usage``
    started reporting one. A caller reading ``input_details`` off a blocking call got ``None`` while
    the streaming path handed back the nested dict, so the two paths disagreed about the same run.
    """
    details = usage.get("input_details")
    return UsageDict(
        input=usage.get("input", 0),
        output=usage.get("output", 0),
        total=usage.get("total", 0),
        input_details=(
            InputTokenDetails(
                uncached=details.get("uncached", 0),
                cache_read=details.get("cache_read", 0),
                cache_creation=details.get("cache_creation", 0),
            )
            if isinstance(details, dict)
            else None
        ),
    )


@dataclass
class SpanUsage:
    """The provider-neutral token counts a span reports, after the caller applied its cache rule.

    ``input`` is the *inclusive* total: whether cache tokens were already counted inside the
    provider's input figure (OpenAI, LangChain) or reported alongside it and folded in by the caller
    (Anthropic), by the time a value reaches this type the folding is done.

    That is the whole point of the type existing. See :func:`add_cached_tokens_to_input`.
    """

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_creation: int = 0


def add_cached_tokens_to_input(raw_usage: dict[str, Any]) -> SpanUsage:
    """Adds a provider's cached-token counts into its input total.

    Providers describe the same call two different ways. Anthropic reports cache reads and cache
    writes as buckets *beside* ``input_tokens``, so ``input_tokens`` counts only the new tokens: a
    turn that read 19,971 tokens from cache and wrote 3,580 more reports ``input_tokens: 3``, when
    the model actually processed 23,554. OpenAI and LangChain report one input figure that already
    contains the cached tokens, with a subset breakdown alongside.

    This is for the first kind. Applying it to the second would count the cached tokens twice, which
    is why the rule lives at the call site and not inside :func:`set_usage_span_attributes`.

    Not named for Anthropic: Bedrock Converse reports the same way, and the shape is the reason it
    applies, not the vendor.
    """
    cache_read = number_or_zero(raw_usage.get("cache_read_input_tokens"))
    cache_creation = number_or_zero(raw_usage.get("cache_creation_input_tokens"))
    return SpanUsage(
        input=number_or_zero(raw_usage.get("input_tokens"))
        + cache_read
        + cache_creation,
        output=number_or_zero(raw_usage.get("output_tokens")),
        cache_read=cache_read,
        cache_creation=cache_creation,
    )


@dataclass
class RunUsage:
    """A run's accumulated token spend, for the ``invoke_agent`` root.

    The root is the only span carrying ``launchdarkly.*`` and the ``feature_flag`` event, so it is
    the span a config-scoped query finds. Without a run total on it, that query returns nothing:
    summing the children requires having already found them.

    Provider-blind on purpose. It sums four numbers and remembers whether anything was added. The
    cache-folding rule that differs per provider has already been applied by the time a
    :class:`SpanUsage` exists, so each handler maps its own bag first and this stays shared.

    The two Anthropic handlers deliberately do not use it. Their run total has to stay in Anthropic's
    own field names with the cache buckets *unfolded*, because it is also the handler's return value
    and :func:`parse_usage` folds it exactly once; handing back a :class:`SpanUsage` there would
    count the cache twice.
    """

    total: SpanUsage = field(default_factory=SpanUsage)
    _turns: int = 0

    @property
    def reported(self) -> bool:
        """Whether any turn reported usage at all.

        The failure path needs this to tell "no call completed" from "a call completed and reported
        zero". Only the second may be written to a span: all-zero attributes assert the run cost
        nothing, which a run whose first call died mid-flight cannot claim, whereas an absent
        attribute correctly says "unknown".

        Counted rather than derived by testing the total for zero, so a provider reporting a
        genuinely empty bag stays distinguishable.
        """
        return self._turns > 0

    def add(self, turn: SpanUsage | None) -> None:
        """Adds one turn. ``None`` is a no-op and does not count as reported."""
        if turn is None:
            return
        self._turns += 1
        self.total.input += turn.input
        self.total.output += turn.output
        self.total.cache_read += turn.cache_read
        self.total.cache_creation += turn.cache_creation


def create_run_usage() -> RunUsage:
    """A fresh run accumulator. Present so call sites read the same as the TypeScript SDK's."""
    return RunUsage()


def lang_chain_span_usage(usage: dict[str, Any] | None) -> SpanUsage | None:
    """One LangChain ``usage_metadata`` bag as :class:`SpanUsage`, or ``None`` when it reports nothing.

    LangChain already includes cached tokens in ``input_tokens``, so nothing is folded here. Shared
    because both LangChain handlers read the identical shape, and because ``None`` for an empty bag
    is what keeps a turn the provider said nothing about from registering as reported: the callback
    path hands over ``{}`` rather than nothing when a provider omits usage entirely.
    """
    if not usage:
        return None
    if usage.get("input_tokens") is None and usage.get("output_tokens") is None:
        return None
    details = usage.get("input_token_details") or {}
    return SpanUsage(
        input=number_or_zero(usage.get("input_tokens")),
        output=number_or_zero(usage.get("output_tokens")),
        cache_read=number_or_zero(details.get("cache_read")),
        cache_creation=number_or_zero(details.get("cache_creation")),
    )


def set_usage_span_attributes(span: Any, usage: SpanUsage) -> None:
    """Writes the OpenTelemetry ``gen_ai.usage.*`` token attributes onto a span.

    Always writes all seven attributes, every time, including zeros. Consumers group and aggregate
    on the complete set, and an *absent* attribute drops a span from those queries entirely, which
    reads as "this call had no cached tokens" when it actually means "this handler forgot to say". A
    provider with no cache-creation concept still reports 0.

    ``total`` is always ``input + output``. Providers that report their own total are deliberately
    not trusted here.

    This helper does **not** know each provider's cache accounting, and must not learn it. Anthropic
    reports cache tokens alongside input, so its callers fold them in; OpenAI and LangChain already
    count cache tokens inside input, so their callers pass the reported figure through untouched.
    Both arrive here as an inclusive ``input``. Centralising that rule would silently double-count
    for two providers out of three.

    The last two are OpenLLMetry aliases for the same two numbers. They belong here rather than
    beside the completion text so they cannot disagree with the canonical attributes above:
    computed at a call site off the raw input field, the alias undercounted Anthropic by every
    cached token, because on Anthropic that field excludes the cache. One writer, one number.
    """
    inp = number_or_zero(usage.input)
    out = number_or_zero(usage.output)
    span.set_attribute("gen_ai.usage.input_tokens", inp)
    span.set_attribute("gen_ai.usage.output_tokens", out)
    span.set_attribute("gen_ai.usage.total_tokens", inp + out)
    span.set_attribute(
        "gen_ai.usage.cache_read.input_tokens", number_or_zero(usage.cache_read)
    )
    span.set_attribute(
        "gen_ai.usage.cache_creation.input_tokens", number_or_zero(usage.cache_creation)
    )
    span.set_attribute("gen_ai.usage.prompt_tokens", inp)
    span.set_attribute("gen_ai.usage.completion_tokens", out)


def set_model_identity_attributes(
    span: Any,
    provider_name: str,
    request_model: str,
    legacy_system: str | None = None,
) -> None:
    """Writes the model identity attributes that every LLM span carries.

    Both spellings of the provider key are emitted on purpose. ``gen_ai.system`` is the pre-1.37
    semconv name and is what handlers shipped before the span hierarchy landed;
    ``gen_ai.provider.name`` is the current name. Emitting only the new key would silently break
    dashboards written against the old one, and emitting only the old one leaves us off-spec, so
    both go out until the next major.

    ``legacy_system`` exists because the two keys do not always want the same value. The LangChain
    handlers ship ``gen_ai.system = 'langchain'``, but ``gen_ai.provider.name`` means *who served
    the model* and its semconv enum has no ``langchain`` member, so those handlers pass the real
    provider for the new key and keep the framework name on the old one.
    """
    span.set_attribute(
        "gen_ai.system", provider_name if legacy_system is None else legacy_system
    )
    span.set_attribute("gen_ai.provider.name", provider_name)
    span.set_attribute("gen_ai.request.model", request_model)


def end_span_once(
    span: Any, tracker: set[int], abandoned: bool = False, cancelled: bool = False
) -> None:
    """Ends a span exactly once, even when the caller cannot know whether an earlier path ended it.

    The streaming handlers need this: a consumer that breaks out of ``async for``, or throws inside
    the loop body, makes the generator run its ``finally`` without ever entering ``except``, so the
    cleanup path and the success path can both reach the same span. Ending twice is silently ignored
    by the OTel SDK but recorded as a diagnostic error, and would also hide a genuine leak, so the
    guard is explicit.

    *tracker* holds ``id(span)`` rather than the span itself, because an OTel span is not guaranteed
    hashable across implementations.

    *cancelled* wins over *abandoned*, because both can be true of the same unwind and only one of
    them is the reason. A consumer that stops reading abandoned the stream. A ``CancelledError``
    means something cancelled the run, usually a timeout, and the consumer never chose anything. The
    blocking paths report that second case as ``launchdarkly.run.cancelled``, so the streaming paths
    say the same thing rather than calling a timed-out run an abandoned one.

    An abandoned span is marked with ``launchdarkly.stream.abandoned`` and deliberately left at
    ``UNSET`` rather than ``ERROR``. Stopping early is a normal thing for a consumer to do, such as
    rendering enough of a response and moving on, and nothing failed. Marking it ``ERROR`` would
    also put the trace at odds with LaunchDarkly's own metrics, which record neither a success nor
    an error for an abandoned stream: two dashboards would disagree about the same run. The
    attribute keeps abandonment findable without asserting a failure.

    Ending the span is not always enough. Two handlers also hold a vendor generator or run that must
    be closed or cancelled in the same ``finally``. See TELEMETRY-CONTRACT.md section 6.

    A ``None`` span is a no-op, matching every other helper in this family. Handlers hold ``None``
    whenever the OpenTelemetry SDK is absent, and a cleanup path in a ``finally`` is the last place
    that should have to remember it.
    """
    if span is None:
        return
    key = id(span)
    if key in tracker:
        return
    tracker.add(key)
    if cancelled:
        span.set_attribute("launchdarkly.run.cancelled", True)
    elif abandoned:
        span.set_attribute("launchdarkly.stream.abandoned", True)
    span.end()


def end_unfinished_spans(*spans: Any) -> None:
    """Ends every span still open, for an exception no ``except Exception`` can catch.

    ``asyncio.CancelledError`` inherits from ``BaseException``, deliberately, so a timeout or a
    ``task.cancel()`` walks straight past every ``except Exception`` a handler writes. The blocking
    paths ended their spans only from those clauses, so a cancelled run exported nothing at all: not a
    wrong attribute, no span. The root carries the ``feature_flag`` event and every ``launchdarkly.*``
    attribute, so a stranded root means the whole run never reaches AI Config Monitoring.

    Call this from a ``finally``, not from an ``except``. The point is the paths an ``except`` cannot
    see.

    Spans are left at ``UNSET`` and marked ``launchdarkly.run.cancelled``. Nothing failed: the caller
    went away. That is the same reasoning :func:`end_span_once` applies to an abandoned stream, and it
    keeps the trace agreeing with LaunchDarkly's own metrics, which record neither a success nor an
    error for a run that never finished.

    A span another path already ended is skipped. ``is_recording()`` is ``False`` once ``end()`` has
    run, so this is a no-op on the success path and cannot end a span twice. Setting an attribute after
    ``end()`` logs a warning, which is why the check comes first. A ``None`` span is a no-op, matching
    every other helper in this family.

    This has no TypeScript counterpart, and needs none. An aborted request there rejects its promise
    and the ``catch`` catches it. Only Python routes cancellation around a handler's guards.
    """
    for span in spans:
        if span is None or not span.is_recording():
            continue
        span.set_attribute("launchdarkly.run.cancelled", True)
        span.end()


def parse_template(template: str, variables: dict[str, Any]) -> str:
    """
    Replaces ``{{variable}}`` placeholders in *template* with values from
    *variables*. Supports dot-notation for nested access (e.g. ``{{user.name}}``).
    Unrecognised placeholders are left as-is.
    """

    def _resolve(match: re.Match[str]) -> str:
        path = match.group(1)
        value: Any = variables
        for key in path.split("."):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return match.group(0)  # leave placeholder unchanged
        if value is None:
            return match.group(0)
        return str(value)

    return re.sub(r"\{\{([\w.]+)\}\}", _resolve, template)


def to_ld_context(client: Any, context: dict[str, Any]) -> Any:
    """
    Convert a plain-dict ``LDContext`` to an ``ldclient.Context`` for the real
    SDK. Only converts when the client is an actual ``ldclient.LDClient``
    instance — mock clients used in tests receive the dict unchanged.
    """
    try:
        import ldclient as _ld

        if isinstance(client, _ld.LDClient):
            return _ld.Context.from_dict(context)
    except Exception:
        pass
    return context


def select_handler(
    config: AiConfigRep,
    meta: VariationMeta,
    handlers: list[ProviderHandler],
    *,
    strict: bool = True,
) -> ProviderHandler:
    """
    Selects a handler from *handlers* based on the provider and mode in *config*/*meta*.

    Resolution order:
    1. Exact ``(provider, mode)`` match.
    2. Wildcard ``("*", mode)`` match — for multi-provider adapters like LangChain.
    3. (Non-strict only) Provider-only match, then single-handler fallback.

    When *strict* is ``True`` (default, used by ``config()``), steps 1–2 are tried
    and a descriptive error is raised if neither matches.

    When *strict* is ``False`` (used by ``graph()``), resolution falls back
    progressively: exact match → wildcard match → provider-only match → single-handler fallback.
    """
    provider = (
        (config.get("provider") or {}).get("name") if isinstance(config, dict) else None
    )
    if not provider:
        raise ValueError("Provider not found")

    mode = normalize_mode(meta.get("mode") if isinstance(meta, dict) else None)

    exact = next((h for h in handlers if h.provides_for == (provider, mode)), None)
    if exact:
        return exact

    wildcard = next(
        (
            h
            for h in handlers
            if h.provides_for and h.provides_for[0] == "*" and h.provides_for[1] == mode
        ),
        None,
    )
    if wildcard:
        return wildcard

    if strict:
        has_coverage = any(
            h.provides_for
            and (h.provides_for[0] == provider or h.provides_for[0] == "*")
            for h in handlers
        )
        if not has_coverage:
            raise ValueError(f"Handler for provider {provider} not found")
        raise ValueError(f"Handler for provider {provider} with mode {mode} not found")

    by_provider = next(
        (h for h in handlers if h.provides_for and h.provides_for[0] == provider),
        None,
    )
    if by_provider:
        return by_provider

    if len(handlers) == 1:
        return handlers[0]

    raise ValueError(f"Handler for provider {provider} not found")


def make_track_data(node: GraphNode, graph_key: str, run_id: str) -> dict[str, Any]:
    """
    Builds the standard tracking payload for a graph node event.
    Shared by all native graph adapters (openai-agents, claude-agents, langchain-agents).
    """
    meta = node.meta if isinstance(node.meta, dict) else {}
    config = node.config if isinstance(node.config, dict) else {}
    return {
        "runId": run_id,
        "configKey": node.key,
        "variationKey": meta.get("variationKey", ""),
        "version": meta.get("version", 1),
        "modelName": config.get("model", {}).get("name", ""),
        "providerName": config.get("provider", {}).get("name", ""),
        "graphKey": graph_key,
    }


def _usable_context_key(value: Any) -> str | None:
    return value if isinstance(value, str) and value != "" else None


def _escape_canonical_part(value: str) -> str:
    return value.replace("%", "%25").replace(":", "%3A")


def _compact_context_keys_json(keys: dict[str, str]) -> str:
    """Compact JSON of per-kind keys in lexicographic kind order."""
    parts = [
        f"{json.dumps(kind, ensure_ascii=False)}:{json.dumps(keys[kind], ensure_ascii=False)}"
        for kind in sorted(keys)
    ]
    return "{" + ",".join(parts) + "}"


def _context_identity_from_ld_context(
    ld_context: Any,
) -> tuple[str, dict[str, str]] | None:
    """Canonical key plus per-kind map, or None when there is no usable identity.

    Never raises.
    """
    try:
        if not isinstance(ld_context, dict):
            return None

        if ld_context.get("kind") == "multi":
            raw: dict[str, str] = {}
            for kind, value in ld_context.items():
                if kind in ("kind", "_meta") or not isinstance(value, dict):
                    continue
                key = _usable_context_key(value.get("key"))
                if key is not None:
                    raw[kind] = key
            kinds = sorted(raw)
            if not kinds:
                return None
            keys = {kind: raw[kind] for kind in kinds}
            canonical = ":".join(
                f"{_escape_canonical_part(kind)}:{_escape_canonical_part(raw[kind])}"
                for kind in kinds
            )
            return canonical, keys

        key = _usable_context_key(ld_context.get("key"))
        if key is None:
            return None
        kind_value = ld_context.get("kind")
        kind = kind_value if isinstance(kind_value, str) and kind_value else "user"
        keys = {kind: key}
        canonical = (
            _escape_canonical_part(key)
            if kind == "user"
            else f"{_escape_canonical_part(kind)}:{_escape_canonical_part(key)}"
        )
        return canonical, keys
    except Exception:
        return None


def set_ld_span_attributes(span: Any, variables: dict[str, Any] | None) -> None:
    """
    Sets LaunchDarkly config-identifying attributes on an OTel span and emits
    the ``feature_flag`` span event required by the AI Config Monitoring Traces
    tab.

    Reads the ``__ld`` entry injected into *variables* by
    ``execute_and_track`` / ``execute_and_stream``, so handlers never need to
    receive ``TrackData`` directly. Context identity is read from
    ``variables.ldContext``, never from ``TrackData``.

    Span attributes (LLM dashboard discovery and custom queries):

    * ``launchdarkly.operation.type`` = ``'gen_ai'``
    * ``launchdarkly.config.key``     = configKey
    * ``launchdarkly.variation.key``  = variationKey
    * ``launchdarkly.run.id``         = runId
    * ``launchdarkly.graph.key``      = graphKey  (only when present)
    * ``context.contextKeys.<kind>``  = raw per-kind key (when ldContext has identity)

    Span event (required for AI Config Monitoring Traces tab correlation):
    ``name='feature_flag'`` with ``feature_flag.key``,
    ``feature_flag.provider.name``, ``feature_flag.set.id`` (when
    ``LD_ENVIRONMENT_ID`` is set or the TS SDK auto-resolved it),
    ``feature_flag.context.id``, and ``feature_flag.contextKeys`` (when
    ``ldContext`` has a usable identity).
    """
    span.set_attribute("launchdarkly.operation.type", "gen_ai")
    if not variables:
        return
    ld = variables.get("__ld")
    if not ld:
        return
    span.set_attribute("launchdarkly.config.key", ld.get("configKey", ""))
    span.set_attribute("launchdarkly.variation.key", ld.get("variationKey", ""))
    span.set_attribute("launchdarkly.run.id", ld.get("runId", ""))
    if ld.get("graphKey"):
        span.set_attribute("launchdarkly.graph.key", ld["graphKey"])

    feature_flag_attrs: dict[str, str] = {
        "feature_flag.key": ld.get("configKey", ""),
        "feature_flag.provider.name": "LaunchDarkly",
    }
    if ld.get("environmentId"):
        feature_flag_attrs["feature_flag.set.id"] = ld["environmentId"]

    identity = _context_identity_from_ld_context(variables.get("ldContext"))
    if identity is not None:
        canonical, keys = identity
        feature_flag_attrs["feature_flag.context.id"] = canonical
        feature_flag_attrs["feature_flag.contextKeys"] = _compact_context_keys_json(
            keys
        )
        for kind, key in keys.items():
            span.set_attribute(f"context.contextKeys.{kind}", key)

    span.add_event("feature_flag", feature_flag_attrs)


def set_openllmetry_prompt(span: Any, messages: list[dict[str, str]]) -> None:
    """Set OpenLLMetry-style indexed prompt attributes on a span.

    Gonfalon's LLM Summary tab reads ``gen_ai.prompt.N.role`` / ``.content``
    (attribute-based, takes precedence over span events).
    """
    for i, msg in enumerate(messages):
        span.set_attribute(f"gen_ai.prompt.{i}.role", msg["role"])
        span.set_attribute(f"gen_ai.prompt.{i}.content", msg["content"])


def set_openllmetry_completion(
    span: Any,
    completion: str,
    usage: dict[str, int],
) -> None:
    """Set OpenLLMetry-style indexed completion attributes and token usage aliases.

    Gonfalon reads ``gen_ai.completion.0.role`` / ``.content`` and prefers
    ``gen_ai.usage.prompt_tokens`` / ``completion_tokens``.
    """
    span.set_attribute("gen_ai.completion.0.role", "assistant")
    span.set_attribute("gen_ai.completion.0.content", completion)
    span.set_attribute("gen_ai.usage.prompt_tokens", usage.get("input_tokens", 0))
    span.set_attribute("gen_ai.usage.completion_tokens", usage.get("output_tokens", 0))


def parse_json_with_possible_fences(raw_text: str) -> Any | None:
    """
    Parses a JSON string that may be wrapped in markdown code fences
    (` ```json ` or bare ` ``` `). Also handles text with a preamble before
    the fence block. Returns ``None`` if the text cannot be parsed as valid
    JSON after fence removal.
    """
    trimmed = raw_text.strip()

    try:
        return json.loads(trimmed)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strip leading/trailing fences (text starts with the fence)
    stripped = re.sub(r"^```json\r?\n", "", trimmed)
    stripped = re.sub(r"^```\r?\n", "", stripped)
    stripped = re.sub(r"\r?\n```$", "", stripped)
    stripped = stripped.strip()

    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass

    # Find a code fence block anywhere in the text (preamble before the fence)
    fence_match = re.search(r"```(?:json)?\r?\n([\s\S]*?)\r?\n```", trimmed)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    return None
