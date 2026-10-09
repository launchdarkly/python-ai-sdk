"""LaunchDarkly AI SDK - integration for OpenAI agents.

See https://launchdarkly.com/docs for usage.
"""

from launchdarkly_ai_server import register_ai_sdk_package

from . import native_graph  # noqa: F401
from ._version import PACKAGE_NAME, __version__
from .graph import openai_graph
from .handler import create_openai_agent_handler, openai_agents
from .native_graph import to_openai_agents
from .utils import build_output_type

__all__ = [
    "build_output_type",
    "create_openai_agent_handler",
    "openai_agents",
    "openai_graph",
    "to_openai_agents",
]

register_ai_sdk_package(PACKAGE_NAME, __version__)
