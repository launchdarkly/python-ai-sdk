"""LaunchDarkly AI SDK - integration for OpenAI messages."""

__version__ = "0.1.4"  # x-release-please-version

from launchdarkly_ai_server import register_ai_sdk_package

from .handler import create_openai_messages_handler, openai_messages

__all__ = ["create_openai_messages_handler", "openai_messages"]

register_ai_sdk_package("launchdarkly-ai-openai-messages", __version__)
