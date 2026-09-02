"""LaunchDarkly AI SDK - integration for Claude messages."""

__version__ = "0.1.4"  # x-release-please-version

from launchdarkly_ai_server import register_ai_sdk_package

from .handler import claude_messages, create_claude_messages_handler

__all__ = ["claude_messages", "create_claude_messages_handler"]

register_ai_sdk_package("launchdarkly-ai-claude-messages", __version__)
