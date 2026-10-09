from __future__ import annotations

import re
from typing import Any, TypeGuard

from .types import ParseFailure, ParseResult, ParseSuccess

_VALID_ROLES = {"user", "assistant", "system"}

SKILL_KEY_GRAMMAR = "^[a-z0-9][a-z0-9-]*$"
"""The skill key grammar, as quoted in rejection messages."""

_SKILL_KEY_PATTERN = re.compile(r"\A[a-z0-9][a-z0-9-]*\Z")
"""
``SKILL_KEY_GRAMMAR`` anchored with ``\\A``/``\\Z``: ``$`` would also match before
a trailing newline and let ``"pdf-extraction\\n"`` through as a directory name.
"""

SKILL_KEY_MAX_LENGTH = 256
"""Longest permitted skill key. ``write_skills`` applies a tighter bound, since
most filesystems cap a path component below 256 bytes."""


def _is_object(v: Any) -> bool:
    return isinstance(v, dict)


def skill_key_rejection_reason(key: Any) -> str | None:
    """
    Why *key* is not a valid skill key, or ``None`` when it is.

    Shared by every caller that validates keys, so rejections read the same.
    """
    if not isinstance(key, str):
        return "must be a string"
    if len(key) > SKILL_KEY_MAX_LENGTH:
        return f"must be at most {SKILL_KEY_MAX_LENGTH} characters"
    if _SKILL_KEY_PATTERN.match(key) is None:
        return f"must match {SKILL_KEY_GRAMMAR}"
    return None


def is_valid_skill_key(key: Any) -> TypeGuard[str]:
    """Whether *key* is a valid skill key (see ``skill_key_rejection_reason``)."""
    return isinstance(key, str) and skill_key_rejection_reason(key) is None


def is_valid_skill_version(version: Any) -> TypeGuard[int]:
    """Whether *version* is a valid skill version: an ``int`` >= 1 (not ``bool``)."""
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


def skills_field_rejection_reason(raw: Any) -> str | None:
    """
    Why a present ``skills`` field is malformed, or ``None`` when it is valid.

    Used by ``skill_refs``, not by ``parse_ai_config``: Agent Skills is
    experimental, and a malformed field must not fail a core config call.
    One malformed reference rejects the whole field, never a partial list.
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

    # ``skills`` is passed through unvalidated. Agent Skills is experimental, so
    # a malformed field must not fail a core config call (TESTING.md §0.3);
    # ``skill_refs`` rejects it where the references are used.

    return ParseSuccess(success=True, data=raw)
