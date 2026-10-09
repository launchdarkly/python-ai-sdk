"""LaunchDarkly AI SDK - integration for LangChain agents.

See https://launchdarkly.com/docs for usage.
"""

from launchdarkly_ai_server import register_ai_sdk_package

from ._version import PACKAGE_NAME, __version__
from .graph import langchain_graph
from .handler import create_langchain_agents_handler, langchain_agents
from .native_graph import to_lang_graph

__all__ = [
    "create_langchain_agents_handler",
    "langchain_agents",
    "langchain_graph",
    "to_lang_graph",
]

register_ai_sdk_package(PACKAGE_NAME, __version__)
