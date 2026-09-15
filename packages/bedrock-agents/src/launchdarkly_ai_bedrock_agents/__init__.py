"""LaunchDarkly AI SDK integration for Amazon Bedrock Strands agents."""

__version__ = "0.0.1"  # x-release-please-version

from .graph import bedrock_graph
from .handler import bedrock_agents, create_bedrock_agents_handler

__all__ = ["bedrock_agents", "bedrock_graph", "create_bedrock_agents_handler"]
