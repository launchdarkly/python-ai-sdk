"""LaunchDarkly AI SDK - convenience package for Python.

See https://launchdarkly.com/docs for usage.
"""

__version__ = "0.1.6"  # x-release-please-version

from launchdarkly_ai_server import *  # noqa: F403
from launchdarkly_ai_server import register_ai_sdk_package

register_ai_sdk_package("launchdarkly-ai-python", __version__)
