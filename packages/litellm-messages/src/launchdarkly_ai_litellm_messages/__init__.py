"""LaunchDarkly AI SDK integration for LiteLLM chat completions."""

__version__ = "0.1.0"  # x-release-please-version

from launchdarkly_ai_server import register_ai_sdk_package

from .handler import create_litellm_messages_handler, litellm_messages

__all__ = ["create_litellm_messages_handler", "litellm_messages"]

register_ai_sdk_package("launchdarkly-ai-litellm-messages", __version__)
