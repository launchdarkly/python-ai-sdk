"""Experimental reader for the richer stream.

``config().stream()`` stays ``chunk`` then ``done``. Call :func:`stream` to
read the AG-UI text and tool events for the same run. This does not select a
product default, and there is no protocol field on ``configure``.
"""

from ._rich_stream import stream

__all__ = ["stream"]
