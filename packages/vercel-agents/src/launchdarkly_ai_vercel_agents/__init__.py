"""LaunchDarkly AI SDK integration for native Vercel AI SDK agents."""

from launchdarkly_ai_server import register_ai_sdk_package

from ._version import PACKAGE_NAME, __version__
from .graph import vercel_graph
from .handler import create_vercel_agents_handler, vercel_agents
from .native_graph import VercelNativeGraphOptions, to_vercel_agents

__all__ = [
    "VercelNativeGraphOptions",
    "create_vercel_agents_handler",
    "to_vercel_agents",
    "vercel_agents",
    "vercel_graph",
]

register_ai_sdk_package(PACKAGE_NAME, __version__)
