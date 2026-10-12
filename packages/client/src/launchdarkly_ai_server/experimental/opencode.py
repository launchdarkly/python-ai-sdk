"""Experimental OpenCode handler.

Import ``create_opencode_handler`` from this module. It is not part of the
package root. The names may change in a minor release.
"""

from ._opencode import OpenCodeSessionError, create_opencode_handler

__all__ = [
    "OpenCodeSessionError",
    "create_opencode_handler",
]
