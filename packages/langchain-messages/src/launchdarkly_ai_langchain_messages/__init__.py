"""LaunchDarkly AI SDK - integration for LangChain messages."""

__version__ = "0.2.0"  # x-release-please-version

from launchdarkly_ai_server import register_ai_sdk_package

from .handler import create_langchain_messages_handler, langchain_messages

__all__ = ["create_langchain_messages_handler", "langchain_messages"]

register_ai_sdk_package("launchdarkly-ai-langchain-messages", __version__)
