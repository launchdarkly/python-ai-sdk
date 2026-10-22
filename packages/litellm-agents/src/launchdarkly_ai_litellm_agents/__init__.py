"""LaunchDarkly AI SDK integration for LiteLLM-backed OpenAI Agents."""

from launchdarkly_ai_server import register_ai_sdk_package

from ._version import PACKAGE_NAME, __version__
from .graph import litellm_graph
from .handler import create_litellm_agents_handler, litellm_agents
from .native_graph import to_litellm_agents

__all__ = [
    "create_litellm_agents_handler",
    "litellm_agents",
    "litellm_graph",
    "to_litellm_agents",
]

register_ai_sdk_package(PACKAGE_NAME, __version__)
