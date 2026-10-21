"""LaunchDarkly AI SDK integration for LiteLLM chat completions."""

from launchdarkly_ai_server import register_ai_sdk_package

from ._version import PACKAGE_NAME, __version__
from .handler import create_litellm_messages_handler, litellm_messages

__all__ = ["create_litellm_messages_handler", "litellm_messages"]

register_ai_sdk_package(PACKAGE_NAME, __version__)
