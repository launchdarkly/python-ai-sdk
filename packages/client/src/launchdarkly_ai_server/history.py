"""Shared conversation-history composition and multimodal content helpers.

Mirrors the TypeScript ``history`` module so every handler composes runtime
``history`` the same way (TESTING.md §1.11) and maps LaunchDarkly-canonical
content blocks to each provider's native shape (Appendix A.7).

History messages are plain dicts: ``{"role": ..., "content": ...}`` where
``content`` is either a string or a list of content-block dicts:

    {"type": "text", "text": str}
    {"type": "image", "source": {"type": "base64", "media_type": str, "data": str}}
    {"type": "image", "source": {"type": "url", "url": str}}
"""

from __future__ import annotations

from typing import Any

MessageContent = str | list[dict[str, Any]]


def compose_history(
    *,
    history: list[dict[str, Any]],
    user_input: str | None = None,
    config_messages: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Composes the ordered conversation turns for a handler with runtime history.

    Order: ``[config conversation messages] -> [history] -> [user_input?]``.

    - System-role history messages are dropped (system belongs on the provider
      system prompt, derived separately by each handler).
    - A non-empty ``user_input`` is always appended as a final user text turn,
      even when history already ends with a user turn (image-only history + a
      separate question).
    - An empty/missing ``user_input`` appends nothing, so history that already
      carries the full (possibly multimodal) user turn is sent as-is.

    Returns a list of ``{"role": "user"|"assistant", "content": ...}`` dicts.
    Callers only take this structured path when ``history`` is non-empty; with no
    history they keep their single-string prompt behaviour, so an empty history
    stays identical to passing none.
    """
    turns: list[dict[str, Any]] = list(config_messages or [])

    for message in history:
        role = message.get("role")
        if role == "system":
            continue
        turns.append({"role": role, "content": message.get("content")})

    if user_input:
        turns.append({"role": "user", "content": user_input})

    return turns


def is_content_blocks(content: MessageContent) -> bool:
    """True when content is the multimodal block-array shape."""
    return isinstance(content, list)


def has_multimodal_content(content: MessageContent) -> bool:
    """True when a message carries any non-text (e.g. image) content block."""
    if not isinstance(content, list):
        return False
    return any(block.get("type") != "text" for block in content)


def any_multimodal(turns: list[dict[str, Any]]) -> bool:
    """True when any turn in the list carries multimodal content."""
    return any(has_multimodal_content(turn.get("content", "")) for turn in turns)


def content_to_text(content: MessageContent) -> str:
    """Flattens content to plain text: a string passes through; a block array
    contributes only its text blocks."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") for block in content if block.get("type") == "text"
    )


def image_block_to_url(block: dict[str, Any]) -> str:
    """Builds a ``data:<media_type>;base64,<data>`` URL for a base64 image block,
    or returns the URL directly for a URL-sourced block.

    This is the form OpenAI and LangChain expect (``image_url``); Anthropic keeps
    ``media_type`` + ``data`` split, so its handlers read ``block["source"]``
    directly instead.
    """
    source = block.get("source", {})
    if source.get("type") == "url":
        return str(source.get("url", ""))
    return f"data:{source.get('media_type', '')};base64,{source.get('data', '')}"
