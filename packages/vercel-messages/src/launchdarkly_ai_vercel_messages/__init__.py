"""LaunchDarkly AI SDK integration for Vercel AI SDK messages."""

from launchdarkly_ai_server import register_ai_sdk_package

from ._version import PACKAGE_NAME, __version__
from .handler import create_vercel_messages_handler, vercel_messages

__all__ = ["create_vercel_messages_handler", "vercel_messages"]

register_ai_sdk_package(PACKAGE_NAME, __version__)
