"""LaunchDarkly AI SDK integration for Amazon Bedrock Strands agents."""

from launchdarkly_ai_server import register_ai_sdk_package

from ._version import PACKAGE_NAME, __version__
from .graph import bedrock_graph
from .handler import bedrock_agents, create_bedrock_agents_handler

__all__ = ["bedrock_agents", "bedrock_graph", "create_bedrock_agents_handler"]

register_ai_sdk_package(PACKAGE_NAME, __version__)
