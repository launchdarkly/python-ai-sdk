"""LaunchDarkly AI SDK integration for Vercel AI SDK messages."""

__version__ = "0.1.0"  # x-release-please-version

from launchdarkly_ai_server import register_ai_sdk_package

from .handler import create_vercel_messages_handler, vercel_messages

__all__ = ["create_vercel_messages_handler", "vercel_messages"]

register_ai_sdk_package("launchdarkly-ai-vercel-messages", __version__)
