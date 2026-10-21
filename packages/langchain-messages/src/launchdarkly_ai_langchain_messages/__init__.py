"""LaunchDarkly AI SDK - integration for LangChain messages.

See https://launchdarkly.com/docs for usage.
"""

from launchdarkly_ai_server import register_ai_sdk_package

from ._version import PACKAGE_NAME, __version__
from .handler import create_langchain_messages_handler, langchain_messages

__all__ = ["create_langchain_messages_handler", "langchain_messages"]

register_ai_sdk_package(PACKAGE_NAME, __version__)
