"""Derives span-safe identity from an ``LDContext`` dict.

Ported from the observability browser SDK's LaunchDarkly integration
(``sdk/highlight-run/src/integrations/launchdarkly/index.ts``), and mirrored by
``js-ai-sdk``'s ``packages/client/src/context.ts``, so every LaunchDarkly
emitter produces byte-identical canonical keys.

Its own module because it is pure: no OTel, no LD client, no I/O. That is also
why it does not go through ``ldclient.Context`` — ``ldclient`` is an optional
import here (see ``utils.to_ld_context``), so relying on it would make the
attribute silently absent for anyone using a custom client.
"""

from __future__ import annotations

from typing import Any


def _encode_key(key: str) -> str:
    """Escapes the two characters ambiguous inside a canonical key: ``%`` and ``:``.

    ``%`` is replaced first so an escape sequence is never double-escaped.
    """
    if "%" in key or ":" in key:
        return key.replace("%", "%25").replace(":", "%3A")
    return key


def _multi_kind_pairs(context: dict[str, Any]) -> list[tuple[str, str]]:
    """``(kind, key)`` pairs of a multi-kind context, sorted by kind.

    Skips any kind whose sub-context has no usable string key. Both public
    functions go through this, so the canonical key and the per-kind map can
    never disagree about which kinds are present.
    """
    pairs: list[tuple[str, str]] = []
    for kind in sorted(context):
        if kind == "kind":
            continue
        sub = context.get(kind)
        key = sub.get("key") if isinstance(sub, dict) else None
        if isinstance(key, str) and key:
            pairs.append((kind, key))
    return pairs


def get_context_keys(context: dict[str, Any]) -> dict[str, str]:
    """The per-kind keys of *context*, as ``{<kind>: <key>}``.

    Keys are raw — only the canonical key is escaped. A legacy user (no
    ``kind``) reports as kind ``user``, matching every other LaunchDarkly
    integration.
    """
    if context.get("kind") == "multi":
        return dict(_multi_kind_pairs(context))
    key = context.get("key")
    if not isinstance(key, str) or not key:
        return {}
    kind = context.get("kind")
    return {kind if isinstance(kind, str) and kind else "user": key}


def get_canonical_key(context: dict[str, Any]) -> str:
    """The canonical key of *context*.

    The same value the Go SDK's ``ldotel`` hook puts on
    ``feature_flag.context.id`` via ``Context().FullyQualifiedKey()``. Stable
    and consistent, not for presentation: it is what links a span to a context
    instance.
    """
    if context.get("kind") == "multi":
        return ":".join(
            f"{kind}:{_encode_key(key)}" for kind, key in _multi_kind_pairs(context)
        )
    key = context.get("key")
    if not isinstance(key, str) or not key:
        return ""
    kind = context.get("kind")
    # A legacy user (no kind) and an explicit `user` kind both canonicalise to
    # the bare key, with no `user:` prefix.
    if not isinstance(kind, str) or not kind or kind == "user":
        return key
    return f"{kind}:{_encode_key(key)}"


def context_identity(context: Any) -> tuple[str, dict[str, str]] | None:
    """The canonical key and per-kind keys of *context*, or ``None``.

    ``None`` whenever there is no usable identity. Never raises: this runs on
    the emit path of every run, and a malformed context must degrade to
    emitting nothing rather than break the caller's AI call.
    """
    if not isinstance(context, dict):
        return None
    try:
        context_keys = get_context_keys(context)
        canonical_key = get_canonical_key(context)
    except Exception:
        return None
    if not canonical_key or not context_keys:
        return None
    return canonical_key, context_keys
