"""LaunchDarkly AI SDK integration for Google ADK agents.

See https://launchdarkly.com/docs for usage.
"""

from launchdarkly_ai_server import register_ai_sdk_package

from ._version import PACKAGE_NAME, __version__
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

register_ai_sdk_package(PACKAGE_NAME, __version__)
