"""LaunchDarkly AI SDK - integration for LangChain agents."""

__version__ = "0.2.0"  # x-release-please-version

from launchdarkly_ai_server import register_ai_sdk_package

from .graph import langchain_graph
from .handler import create_langchain_agents_handler, langchain_agents
from .native_graph import to_lang_graph

__all__ = [
    "create_langchain_agents_handler",
    "langchain_agents",
    "langchain_graph",
    "to_lang_graph",
]

register_ai_sdk_package("launchdarkly-ai-langchain-agents", __version__)
