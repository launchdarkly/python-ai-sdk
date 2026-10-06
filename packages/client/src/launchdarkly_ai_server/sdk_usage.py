"""Records which public AI SDK helpers an application calls.

One ``$ld:ai:sdk:usage`` event per helper per client. A call made before a
client exists is held and sent on the next init, on the same path as sdk-info.
A call made from inside another helper does not report.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from .sdk_info import SDK_INFO_CONTEXT

SDK_USAGE_EVENT = "$ld:ai:sdk:usage"
_SDK_USAGE_LANGUAGE = "python"
_SDK_NAME = "launchdarkly-ai-server"

_pending: set[str] = set()
_reported: set[str] = set()
_depth: ContextVar[int] = ContextVar("ld_ai_sdk_usage_depth", default=0)


def _peek_client() -> Any:
    from . import lifecycle

    return lifecycle._client


def _version() -> str:
    from . import __version__

    return __version__


def _deliver(client: Any, helper: str) -> None:
    from .utils import to_ld_context

    try:
        client.track(
            SDK_USAGE_EVENT,
            to_ld_context(client, SDK_INFO_CONTEXT),
            {
                "aiSdkName": _SDK_NAME,
                "aiSdkVersion": _version(),
                "aiSdkLanguage": _SDK_USAGE_LANGUAGE,
                "helper": helper,
            },
            1,
        )
    except Exception:
        # A throw from track, or from building the payload, must not fail the helper.
        return


def report_usage(helper: str) -> None:
    """Record a public helper.

    Sends immediately when a client exists, otherwise holds the helper until
    :func:`flush_sdk_usage`. No-ops when already reported or when called from
    inside :func:`within_sdk`.
    """
    try:
        if _depth.get() > 0:
            return
        if helper in _reported or helper in _pending:
            return
        client = _peek_client()
        if client is None:
            _pending.add(helper)
            return
        _deliver(client, helper)
        _reported.add(helper)
    except Exception:
        _pending.discard(helper)
        _reported.add(helper)


@contextmanager
def within_sdk() -> Iterator[None]:
    """Mark the body as an internal SDK call so nested helpers do not report."""
    token = _depth.set(_depth.get() + 1)
    try:
        yield
    finally:
        _depth.reset(token)


def call_within_sdk(fn: Callable[[], Any]) -> Any:
    """Call ``fn`` inside the SDK scope.

    A wrapper reports itself, then uses this so the helpers it calls do not
    report. When ``fn`` returns an awaitable, the scope stays active until that
    awaitable finishes: an async function does not run until it is awaited, and
    the scope has to cover that run.
    """
    with within_sdk():
        result = fn()
    if isinstance(result, Awaitable):

        async def _drive() -> Any:
            with within_sdk():
                return await result

        return _drive()
    return result


def flush_sdk_usage(client: Any) -> None:
    """Send every helper that was recorded before a client existed."""
    if not _pending:
        return
    held = list(_pending)
    _pending.clear()
    for helper in held:
        _deliver(client, helper)
        _reported.add(helper)


def reset_sdk_usage() -> None:
    """Drop the reported set so the next client hears each helper again."""
    _reported.clear()
    _pending.clear()
