"""LaunchDarkly AI SDK integration for Amazon Bedrock Converse."""

from launchdarkly_ai_server import register_ai_sdk_package

from ._version import PACKAGE_NAME, __version__
from .handler import bedrock_messages, create_bedrock_messages_handler

__all__ = ["bedrock_messages", "create_bedrock_messages_handler"]

register_ai_sdk_package(PACKAGE_NAME, __version__)
