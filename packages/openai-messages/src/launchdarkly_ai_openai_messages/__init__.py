"""LaunchDarkly AI SDK - integration for OpenAI messages.

See https://launchdarkly.com/docs for usage.
"""

from launchdarkly_ai_server import register_ai_sdk_package

from ._version import PACKAGE_NAME, __version__
from .handler import create_openai_messages_handler, openai_messages

__all__ = ["create_openai_messages_handler", "openai_messages"]

register_ai_sdk_package(PACKAGE_NAME, __version__)
