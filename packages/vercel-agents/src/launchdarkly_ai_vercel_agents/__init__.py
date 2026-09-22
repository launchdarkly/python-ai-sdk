"""LaunchDarkly AI SDK integration for native Vercel AI SDK agents."""

__version__ = "0.1.0"  # x-release-please-version

from launchdarkly_ai_server import register_ai_sdk_package

from .graph import vercel_graph
from .handler import create_vercel_agents_handler, vercel_agents
from .native_graph import to_vercel_agents

__all__ = [
    "create_vercel_agents_handler",
    "to_vercel_agents",
    "vercel_agents",
    "vercel_graph",
]

register_ai_sdk_package("launchdarkly-ai-vercel-agents", __version__)
