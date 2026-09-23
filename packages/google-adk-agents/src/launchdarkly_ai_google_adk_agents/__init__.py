"""LaunchDarkly AI SDK integration for Google ADK agents.

See https://launchdarkly.com/docs for usage.
"""

__version__ = "0.1.0"  # x-release-please-version

from launchdarkly_ai_server import register_ai_sdk_package

from .graph import google_adk_graph
from .handler import (
    create_google_adk_agents_handler,
    google_adk_agents,
    history_contents,
)
from .native_graph import to_adk_agents

__all__ = [
    "create_google_adk_agents_handler",
    "google_adk_agents",
    "google_adk_graph",
    "history_contents",
    "to_adk_agents",
]

register_ai_sdk_package("launchdarkly-ai-google-adk-agents", __version__)
