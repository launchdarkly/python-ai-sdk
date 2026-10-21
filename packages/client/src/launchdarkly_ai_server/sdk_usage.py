"""Records which public AI SDK helpers an application calls.

One ``$ld:ai:sdk:usage`` event per helper per client. A call made before a
client exists is held and sent on the next init, on the same path as sdk-info.

Each event names the core client (``aiSdkName`` / ``aiSdkVersion``) and the
package the helper belongs to (``helperPackageName`` / ``helperPackageVersion``).
Handler packages are versioned separately from the core, and sdk-info sends
every package on one shared context, so only this event ties a helper call to
the handler release that ran it.

Only a call from outside the SDK reports. Each reporting helper has a
non-reporting internal version, and SDK code calls that version instead, so
this module never needs to know who the caller is.
"""

from __future__ import annotations

import threading
from typing import Any

from .sdk_info import SDK_INFO_CONTEXT

SDK_USAGE_EVENT = "$ld:ai:sdk:usage"
_SDK_USAGE_LANGUAGE = "python"
_SDK_NAME = "launchdarkly-ai-server"

# A helper's package: (name, version), or None for the core client.
_Package = tuple[str, str] | None

# Held helpers keep their package so the flushed event is complete.
_pending: dict[str, _Package] = {}
_reported: set[str] = set()
# Guards _pending and _reported. A helper is marked reported under the lock
# before it is delivered, so concurrent first calls send it once. The client is
# read under the lock too: init sets the client and then flushes, and the flush
# waits here, so a helper held just as init runs is still sent by that flush.
# Delivery runs outside the lock so a slow track() does not block other callers.
_lock = threading.Lock()


def _peek_client() -> Any:
    from . import lifecycle

    return lifecycle._client


def _version() -> str:
    from . import __version__

    return __version__


def _deliver(client: Any, helper: str, package: _Package) -> None:
    from .utils import to_ld_context

    try:
        core_version = _version()
        package_name, package_version = package or (_SDK_NAME, core_version)
        client.track(
            SDK_USAGE_EVENT,
            to_ld_context(client, SDK_INFO_CONTEXT),
            {
                "aiSdkName": _SDK_NAME,
                "aiSdkVersion": core_version,
                "aiSdkLanguage": _SDK_USAGE_LANGUAGE,
                "helper": helper,
                "helperPackageName": package_name,
                "helperPackageVersion": package_version,
            },
            1,
        )
    except Exception:
        # A throw from track, or from building the payload, must not fail the helper.
        return


def report_usage(
    helper: str,
    package_name: str | None = None,
    package_version: str | None = None,
) -> None:
    """Record a public helper.

    A handler package passes the same *package_name* and *package_version* it
    gives :func:`register_ai_sdk_package`. A core client helper passes neither,
    and the event names the core client. Pass both or neither; passing only one
    reports the core client.

    Sends immediately when a client exists, otherwise holds the helper until
    :func:`flush_sdk_usage`. No-ops when already reported or held; the package
    is not part of that key. Never raises.
    """
    try:
        package: _Package = None
        if package_name is not None and package_version is not None:
            package = (package_name, package_version)
        with _lock:
            client = _peek_client()
            if helper in _reported or helper in _pending:
                return
            if client is None:
                _pending[helper] = package
                return
            _reported.add(helper)
        _deliver(client, helper, package)
    except Exception:
        with _lock:
            _pending.pop(helper, None)
            _reported.add(helper)


def flush_sdk_usage(client: Any) -> None:
    """Send every helper that was recorded before a client existed."""
    with _lock:
        if not _pending:
            return
        held = list(_pending.items())
        _pending.clear()
        _reported.update(helper for helper, _ in held)
    for helper, package in held:
        _deliver(client, helper, package)


def reset_sdk_usage() -> None:
    """Drop the reported set so the next client hears each helper again."""
    with _lock:
        _reported.clear()
        _pending.clear()
