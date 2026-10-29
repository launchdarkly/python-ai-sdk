"""
Agent Skills — the skills an AI Config pins, read when they are needed.

``config_skills(key, context)`` builds a ``SkillSource`` that evaluates one AI
Config for one context on every read. Passed to ``watch_skills``, it follows
the config: a version bump replaces the file on disk, and a skill the config
stops referencing is pruned. A list from ``skill_refs`` is fixed when it is
built, so a watcher fed one never sees the config change.

Every way an evaluation can fail raises rather than answering ``[]``, because
``[]`` is the answer that prunes every managed skill.
"""

from __future__ import annotations

import inspect
from typing import Any

from .lifecycle import get_client
from .sdk_usage import report_usage
from .skills import _skill_refs
from .types import LDContext, SkillReference, SkillSource
from .types_validation import parse_ai_config
from .utils import to_ld_context


def config_skills(key: str, context: LDContext) -> SkillSource:
    """
    A ``SkillSource`` for the skills AI Config *key* pins for *context*.

    The documented form for keeping a config's skills on disk with
    ``watch_skills``::

        report, watcher = await watch_skills(
            config_skills("support-agent", context), "/etc/agent/skills"
        )

    Each read evaluates the config afresh with the initialized client's
    ``variation``, which records an evaluation event, as ``inspect_config``
    does: once per reconcile under a watcher, bounded by its debounce. A read
    sends no AI telemetry.

    A read answers ``[]`` only when LaunchDarkly says there are no skills: the
    variation is disabled, or the config has no ``skills`` field. It raises
    ``RuntimeError`` naming *key* and the cause when no client is initialized
    (it never initializes one), when ``variation`` raises or answers its
    default, which is how a missing flag and a client that is not ready yet
    both arrive, and when the variation or its ``skills`` field is malformed.
    ``write_skills`` treats that as an unavailable retrieval and changes
    nothing on disk.

    For several configs, or a client brought to ``init_client`` without a
    ``flag_tracker``, build a ``SkillSource`` by hand.
    """
    report_usage("client.configSkills")

    async def refs() -> list[SkillReference]:
        return await _read_config_skills(key, context)

    return SkillSource(refs, config_keys=(key,))


def _failed(key: str, cause: str) -> RuntimeError:
    return RuntimeError(
        f"config_skills could not read the skills of AI Config '{key}': {cause}"
    )


async def _read_config_skills(key: str, context: LDContext) -> list[SkillReference]:
    try:
        client = get_client()
    except RuntimeError as exc:
        raise _failed(
            key, "no AI client is initialized; call init_client() first"
        ) from exc

    try:
        result = client.variation(key, to_ld_context(client, context), None)
        value: Any = await result if inspect.isawaitable(result) else result
    except Exception as exc:
        raise _failed(key, f"variation raised {type(exc).__name__}: {exc}") from exc

    # The default must stay distinguishable from every real answer: anything
    # that reads as disabled here would prune every skill.
    if value is None:
        raise _failed(
            key,
            "the evaluation returned its default, so the config was not found, "
            "the client is not ready, or evaluation failed",
        )
    if not isinstance(value, dict):
        raise _failed(
            key, f"the evaluation returned {type(value).__name__}, not an object"
        )
    meta = value.get("_ldMeta")
    if not isinstance(meta, dict) or not isinstance(meta.get("enabled"), bool):
        raise _failed(
            key, "the variation has no _ldMeta object carrying a boolean enabled"
        )

    if not meta["enabled"]:
        # A disabled variation is an answer from LaunchDarkly: its skills leave.
        return []

    parsed = parse_ai_config({k: v for k, v in value.items() if k != "_ldMeta"})
    if not parsed.success:
        raise _failed(key, f"the config failed to parse: {parsed.error.get('message')}")
    try:
        return _skill_refs(parsed.data)
    except ValueError as exc:
        raise _failed(key, str(exc)) from exc
