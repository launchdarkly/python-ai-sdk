from __future__ import annotations

import json
import re
from typing import Any, Literal

from .types import (
    AiConfigRep,
    GraphNode,
    ProviderHandler,
    VariationMeta,
    _HandlerFn,
    _StreamFn,
)


def create_handler(
    provides_for: tuple[str, Literal["agent", "messages"]],
    fn: _HandlerFn,
    stream_fn: _StreamFn | None = None,
) -> ProviderHandler:
    """
    Wraps a plain async callable in a :class:`ProviderHandler` with the given
    ``provides_for`` metadata and optional streaming implementation.
    """
    return ProviderHandler(fn=fn, provides_for=provides_for, stream_fn=stream_fn)


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


_USAGE_KEY_PAIRS = [
    ("input_tokens", "output_tokens"),
    ("inputTokens", "outputTokens"),
    ("input", "output"),
]


def parse_usage(usage: dict[str, Any]) -> dict[str, int]:
    """
    Normalises token counts from three possible key-pair shapes.
    Always computes ``total`` from ``input + output``; never trusts a ``total``
    field from the raw object.
    """
    for input_key, output_key in _USAGE_KEY_PAIRS:
        if input_key in usage and output_key in usage:
            inp = int(usage[input_key])
            out = int(usage[output_key])
            return {"input": inp, "output": out, "total": inp + out}
    return {"input": 0, "output": 0, "total": 0}


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


def set_ld_span_attributes(span: Any, variables: dict[str, Any] | None) -> None:
    """
    Sets LaunchDarkly config-identifying attributes on an OTel span and emits
    the ``feature_flag`` span event required by the AI Config Monitoring Traces
    tab.

    Reads the ``__ld`` entry injected into *variables* by
    ``execute_and_track`` / ``execute_and_stream``, so handlers never need to
    receive ``TrackData`` directly.

    Span attributes (LLM dashboard discovery and custom queries):

    * ``launchdarkly.operation.type`` = ``'gen_ai'``
    * ``launchdarkly.config.key``     = configKey
    * ``launchdarkly.variation.key``  = variationKey
    * ``launchdarkly.run.id``         = runId
    * ``launchdarkly.graph.key``      = graphKey  (only when present)

    Span event (required for AI Config Monitoring Traces tab correlation):
    ``name='feature_flag'`` with ``feature_flag.key``,
    ``feature_flag.provider.name``, and ``feature_flag.set.id`` (when
    ``LD_ENVIRONMENT_ID`` is set or the TS SDK auto-resolved it).
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
