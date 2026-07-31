from __future__ import annotations

from typing import Any

from .types import ParseFailure, ParseResult, ParseSuccess

_VALID_ROLES = {"user", "assistant", "system"}


def _is_object(v: Any) -> bool:
    return isinstance(v, dict)


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

    return ParseSuccess(success=True, data=raw)
