"""LaunchDarkly AI SDK - integration for OpenCode agents.

See https://launchdarkly.com/docs for usage.
"""

from launchdarkly_ai_server import register_ai_sdk_package

from ._version import PACKAGE_NAME, __version__
from .handler import (
    OpenCodeSessionError,
    create_opencode_handler,
    create_opencode_handlers,
    opencode_agents,
)

__all__ = [
    "OpenCodeSessionError",
    "create_opencode_handler",
    "create_opencode_handlers",
    "opencode_agents",
]

register_ai_sdk_package(PACKAGE_NAME, __version__)
