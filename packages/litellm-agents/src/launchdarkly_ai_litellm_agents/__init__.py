"""LaunchDarkly AI SDK integration for LiteLLM-backed OpenAI Agents."""

__version__ = "0.1.0"  # x-release-please-version

from launchdarkly_ai_server import register_ai_sdk_package

from .graph import litellm_graph
from .handler import create_litellm_agents_handler, litellm_agents
from .native_graph import to_litellm_agents

__all__ = [
    "create_litellm_agents_handler",
    "litellm_agents",
    "litellm_graph",
    "to_litellm_agents",
]

register_ai_sdk_package("launchdarkly-ai-litellm-agents", __version__)
