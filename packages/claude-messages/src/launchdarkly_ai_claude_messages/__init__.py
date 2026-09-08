"""LaunchDarkly AI SDK - integration for Claude messages.

See https://launchdarkly.com/docs for usage.
"""

__version__ = "0.2.1"  # x-release-please-version

from launchdarkly_ai_server import register_ai_sdk_package

from .handler import claude_messages, create_claude_messages_handler

__all__ = ["claude_messages", "create_claude_messages_handler"]

register_ai_sdk_package("launchdarkly-ai-claude-messages", __version__)
