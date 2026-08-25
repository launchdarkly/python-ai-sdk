from __future__ import annotations

import re
from typing import Any, TypeGuard

from .types import ParseFailure, ParseResult, ParseSuccess

_VALID_ROLES = {"user", "assistant", "system"}

SKILL_KEY_GRAMMAR = "^[a-z0-9][a-z0-9-]*$"
"""
The skill key grammar, as a string, so every message that has to explain a
rejection quotes the rule rather than restating it. Tightening the pattern below
then cannot leave an error message describing the old grammar.
"""

_SKILL_KEY_PATTERN = re.compile(r"\A[a-z0-9][a-z0-9-]*\Z")
"""
``SKILL_KEY_GRAMMAR``, anchored with ``\\A``/``\\Z`` rather than ``^``/``$``
because ``$`` also matches immediately before a trailing newline, which would
let ``"pdf-extraction\\n"`` through as a directory name.
"""

SKILL_KEY_MAX_LENGTH = 256
"""Longest key the data model permits. Note that no mainstream filesystem allows
a 256-byte path component, so ``write_skills`` applies a tighter bound of its own."""


def _is_object(v: Any) -> bool:
    return isinstance(v, dict)


def skill_key_rejection_reason(key: Any) -> str | None:
    """
    Why *key* is not a valid skill key, or ``None`` when it is.

    The canonical explanation, so the config parser, the filesystem layer and
    the reference projection all reject a key for the same stated reason.
    ``is_valid_skill_key`` is this predicate with the reason discarded.
    """
    if not isinstance(key, str):
        return "must be a string"
    if len(key) > SKILL_KEY_MAX_LENGTH:
        return f"must be at most {SKILL_KEY_MAX_LENGTH} characters"
    if _SKILL_KEY_PATTERN.match(key) is None:
        return f"must match {SKILL_KEY_GRAMMAR}"
    return None


def is_valid_skill_key(key: Any) -> TypeGuard[str]:
    """Skill keys are untrusted input everywhere they appear — validate every time."""
    return isinstance(key, str) and skill_key_rejection_reason(key) is None


def is_valid_skill_version(version: Any) -> TypeGuard[int]:
    """Skill versions are integers >= 1. ``bool`` is not an acceptable integer."""
    return isinstance(version, int) and not isinstance(version, bool) and version >= 1


def _parse_tool(raw: Any, key: str) -> str | None:
    """Returns an error message string or ``None`` on success."""
    if not _is_object(raw):
        return f"tools.{key} must be an object"
    if not isinstance(raw.get("name"), str):
        return f"tools.{key}.name must be a string"
    if raw.get("type") != "function":
        return f'tools.{key}.type must be "function"'
    if not _is_object(raw.get("parameters")):
        return f"tools.{key}.parameters must be an object"
    return None


def _parse_skills(raw: Any) -> str | None:
    """
    Validates the optional ``skills`` array. Returns an error message or ``None``.

    Fail closed: a malformed reference makes the whole config malformed, because
    an SDK that silently dropped a bad reference would materialize a partial
    skill set without telling anyone.
    """
    if not isinstance(raw, list):
        return "skills must be an array of {key, version} objects"

    for index, entry in enumerate(raw):
        if not _is_object(entry):
            return f"skills[{index}] must be an object with key and version"
        key_rejection = skill_key_rejection_reason(entry.get("key"))
        if key_rejection is not None:
            return f"skills[{index}].key {key_rejection}"
        if not is_valid_skill_version(entry.get("version")):
            return f"skills[{index}].version must be an integer >= 1"
    return None


def parse_ai_config(raw: Any) -> ParseResult:
    """
    Validates a raw LaunchDarkly flag variation as an ``AiConfigRep``.

    Returns ``ParseSuccess`` when valid, ``ParseFailure`` otherwise.
    """
    if not _is_object(raw):
        return ParseFailure(
            success=False, error={"message": "Config must be an object"}
        )

    model = raw.get("model")
    if not _is_object(model) or not isinstance(model.get("name"), str):
        return ParseFailure(
            success=False,
            error={"message": "model.name is required and must be a string"},
        )

    provider = raw.get("provider")
    if not _is_object(provider) or not isinstance(provider.get("name"), str):
        return ParseFailure(
            success=False,
            error={"message": "provider.name is required and must be a string"},
        )

    has_instructions = isinstance(raw.get("instructions"), str)
    messages = raw.get("messages")
    has_messages = isinstance(messages, list) and len(messages) > 0

    if not has_instructions and not has_messages:
        return ParseFailure(
            success=False,
            error={
                "message": "AiConfigRep must have either instructions or a non-empty messages array"
            },
        )

    if isinstance(messages, list):
        for msg in messages:
            if not _is_object(msg) or msg.get("role") not in _VALID_ROLES:
                role = msg.get("role") if _is_object(msg) else msg
                return ParseFailure(
                    success=False,
                    error={"message": f"Invalid message role: {role}"},
                )

    tools = raw.get("tools")
    if tools is not None:
        if not _is_object(tools):
            return ParseFailure(
                success=False, error={"message": "tools must be an object"}
            )
        for k, v in tools.items():
            err = _parse_tool(v, k)
            if err:
                return ParseFailure(success=False, error={"message": err})

    output_format = raw.get("outputFormat")
    if output_format is not None and not _is_object(output_format):
        return ParseFailure(
            success=False,
            error={"message": "outputFormat must be an object (JSON Schema)"},
        )

    skills = raw.get("skills")
    if skills is not None:
        err = _parse_skills(skills)
        if err:
            return ParseFailure(success=False, error={"message": err})

    return ParseSuccess(success=True, data=raw)
