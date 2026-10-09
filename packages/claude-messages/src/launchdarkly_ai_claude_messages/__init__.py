"""LaunchDarkly AI SDK - integration for Claude messages.

See https://launchdarkly.com/docs for usage.
"""

from launchdarkly_ai_server import register_ai_sdk_package

from ._version import PACKAGE_NAME, __version__
from .handler import claude_messages, create_claude_messages_handler

__all__ = ["claude_messages", "create_claude_messages_handler"]

register_ai_sdk_package(PACKAGE_NAME, __version__)
