"""LaunchDarkly AI SDK integration for Amazon Bedrock Converse."""

__version__ = "0.0.1"  # x-release-please-version

from .handler import bedrock_messages, create_bedrock_messages_handler

__all__ = ["bedrock_messages", "create_bedrock_messages_handler"]
