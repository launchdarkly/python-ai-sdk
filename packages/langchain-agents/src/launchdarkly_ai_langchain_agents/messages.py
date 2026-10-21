"""Maps LaunchDarkly-canonical conversation turns onto LangChain messages.

Images travel as ``image_url`` content parts with a data or remote URL — the
standard multimodal shape every LangChain chat model accepts — rather than the
LaunchDarkly-canonical ``{"type": "image", "source": ...}`` block, which no
LangChain provider understands (TESTING.md Appendix A.7).
"""

from __future__ import annotations

from typing import Any

from launchdarkly_ai_server import content_to_text, image_block_to_url


def to_content_parts(content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Maps one canonical message's content into LangChain user content parts."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    parts: list[dict[str, Any]] = []
    for block in content:
        if block.get("type") == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        else:
            parts.append(
                {"type": "image_url", "image_url": {"url": image_block_to_url(block)}}
            )
    return parts


def to_lang_chain_messages(turns: list[dict[str, Any]]) -> list[Any]:
    """Turns composed canonical turns into LangChain messages.

    A string user turn stays a string-content ``HumanMessage``, so text-only
    callers see exactly the message they saw before history existed. Assistant
    turns are flattened to text: an ``AIMessage`` carries the model's own prior
    reply, which has no image to preserve.
    """
    import importlib

    msgs_mod = importlib.import_module("langchain_core.messages")
    HumanMessage = msgs_mod.HumanMessage
    AIMessage = msgs_mod.AIMessage

    messages: list[Any] = []
    for turn in turns:
        content = turn.get("content") or ""
        if turn.get("role") == "assistant":
            messages.append(AIMessage(content_to_text(content)))
        elif isinstance(content, str):
            messages.append(HumanMessage(content))
        else:
            messages.append(HumanMessage(to_content_parts(content)))
    return messages
