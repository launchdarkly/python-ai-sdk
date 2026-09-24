"""
Shared filtering for ``model.parameters`` before it reaches a provider SDK call.

``model.parameters`` is a free-form dict the LaunchDarkly UI writes, offering keys that make
sense across providers (``temperature``, ``top_p``, ``max_tokens``, ``tool_choice``, ...). No
single provider SDK accepts all of them: passing a key a plain method or a dataclass constructor
does not declare raises ``TypeError`` before any request is made, and passing an unknown key to a
pydantic model is merely a landmine for the day a stricter model config removes that tolerance.

Every handler therefore filters ``model_parameters(config)`` down to the keys the specific
provider entry point actually accepts before merging it into its own call kwargs. This module is
the one place that derives an accept-set from a live provider type (never a hand-maintained list,
which would drift the moment an SDK adds or removes a parameter) and the one place that strips
transport-only keys regardless of what the accept-set says.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Mapping
from typing import Any

#: Keys that are never forwarded to a provider call, even when the callable's own signature
#: happens to accept them. These are transport or escape-hatch concerns (request timeouts, raw
#: HTTP overrides), not model settings, and forwarding one from ``model.parameters`` would let a
#: config silently rewrite the transport for every call under it.
_TRANSPORT_PARAMETER_KEYS = frozenset({"timeout"})
_TRANSPORT_PARAMETER_PREFIX = "extra_"


def is_transport_parameter(key: str) -> bool:
    """Whether *key* is a transport/escape-hatch parameter that must never be forwarded.

    Matches the exact names ``timeout`` and any key prefixed with ``extra_`` (``extra_headers``,
    ``extra_query``, ``extra_body``, ``extra_args``, and any future ``extra_*`` addition).
    """
    return key in _TRANSPORT_PARAMETER_KEYS or key.startswith(
        _TRANSPORT_PARAMETER_PREFIX
    )


def strip_transport_parameters(params: Mapping[str, Any]) -> dict[str, Any]:
    """Returns a copy of *params* with every transport/escape-hatch key removed.

    Always applied before the accept-set filter below, so a provider call signature that happens
    to declare ``timeout`` or ``extra_body`` can never let a config value reach it.
    """
    return {k: v for k, v in params.items() if not is_transport_parameter(k)}


def accepted_parameter_keys_from_signature(fn: Callable[..., Any]) -> frozenset[str]:
    """Derives the accepted keyword-argument names from a plain callable's signature.

    For the provider methods this SDK forwards to directly (``AsyncAnthropic.messages.create``,
    ``AsyncOpenAI.responses.create``, and their streaming counterparts), the accepted keys are
    exactly the callable's parameter names, read once via :func:`inspect.signature` rather than
    hand-maintained, so a parameter the SDK adds or removes is picked up automatically.

    Raises ``ValueError`` if the signature has a ``**kwargs`` catch-all: that shape has no
    derivable accept-set, and treating it as "accepts everything" would defeat the point of this
    module. Callers should surface which callable that was rather than silently letting everything
    through.
    """
    signature = inspect.signature(fn)
    accepted: set[str] = set()
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            raise ValueError(
                f"{fn!r} accepts **{name}; no accept-set can be derived from its signature"
            )
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        accepted.add(name)
    return frozenset(accepted)


def accepted_parameter_keys_from_dataclass(cls: type) -> frozenset[str]:
    """Derives the accepted keys from a dataclass's field names.

    Used for ``ClaudeAgentOptions`` (claude-agents) and ``ModelSettings`` (openai-agents), both of
    which are plain dataclasses with no ``**kwargs`` catch-all.
    """
    return frozenset(f.name for f in dataclasses.fields(cls))


def _alias_strings(field: Any) -> list[str]:
    """Every string alias a pydantic ``FieldInfo``-like object declares.

    Duck-typed against ``.alias`` and ``.validation_alias`` rather than importing pydantic, so a
    lightweight test double built the same shape works identically to a real ``FieldInfo``.
    ``validation_alias`` may itself be a plain string, or an ``AliasChoices``/``AliasPath`` whose
    ``.choices`` holds the individual alias strings; anything else is ignored.
    """
    aliases: list[str] = []
    alias = getattr(field, "alias", None)
    if isinstance(alias, str):
        aliases.append(alias)
    validation_alias = getattr(field, "validation_alias", None)
    if isinstance(validation_alias, str):
        aliases.append(validation_alias)
    else:
        choices = getattr(validation_alias, "choices", None)
        if choices:
            aliases.extend(choice for choice in choices if isinstance(choice, str))
    return aliases


def accepted_parameter_keys_from_pydantic_model(cls: Any) -> frozenset[str]:
    """Derives the accepted keys from a pydantic model's fields and their aliases.

    Used for the LangChain chat model classes (``ChatOpenAI``, ``ChatAnthropic``,
    ``ChatBedrockConverse``), which populate by alias (``populate_by_name=True``,
    ``validate_by_alias=True``): a constructor kwarg may name either the field itself or one of
    its aliases, so both must be in the accept-set for a value to actually land.

    Reads ``cls.model_fields`` (pydantic v2) rather than importing pydantic, so this also accepts
    any test double exposing the same shape. Raises ``ValueError`` if *cls* has no usable
    ``model_fields`` mapping.
    """
    fields = getattr(cls, "model_fields", None)
    if not isinstance(fields, Mapping):
        raise ValueError(
            f"{cls!r} has no usable model_fields; cannot derive an accept-set from it"
        )
    accepted: set[str] = set()
    for name, field in fields.items():
        accepted.add(name)
        accepted.update(_alias_strings(field))
    return frozenset(accepted)


def filter_forwardable_parameters(
    params: Mapping[str, Any],
    accepted_keys: frozenset[str],
    excluded_keys: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Returns the subset of *params* that is safe to forward to a provider call.

    Three passes, in order: transport/escape-hatch keys are stripped unconditionally (see
    :func:`strip_transport_parameters`), then *excluded_keys* are stripped, then whatever remains is
    filtered down to *accepted_keys*.

    *excluded_keys* is for a key the provider's own signature accepts but that would break the
    handler if a config set it, because the handler itself decides that behaviour (for example
    ``stream``, which the handler picks by calling a blocking or a streaming method, not by a
    kwarg). This is a narrower, opt-in exclusion than the handler-owned keys a call site pops after
    this filter runs (``model``, ``messages``/``input``, ``tools``, ...): those are always removed,
    this is provider-specific and each call site decides its own set. The rule for both this and
    the handler-owned set is the same: exclude only what would break the handler, forward
    everything else the provider accepts, even keys that are not strictly generation settings.
    A key the UI offers but the specific provider entry point does not declare is dropped rather
    than raising, matching this SDK's behaviour before ``model.parameters`` forwarding existed:
    an unrecognised tuning value is silently ignored, not a hard failure.
    """
    transport_free = strip_transport_parameters(params)
    return {
        k: v
        for k, v in transport_free.items()
        if k not in excluded_keys and k in accepted_keys
    }
