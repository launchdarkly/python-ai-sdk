from __future__ import annotations

from typing import Any

from .types import LDContext

SDK_INFO_EVENT = "$ld:ai:sdk:info"

SDK_INFO_CONTEXT: LDContext = {
    "kind": "ld_ai",
    "key": "ld-internal-tracking",
    "anonymous": True,
}

_SDK_INFO_LANGUAGE = "python"

_known: dict[str, tuple[str, str]] = {}
_reported: set[str] = set()


def register_ai_sdk_package(name: str, version: str) -> None:
    """Record a LaunchDarkly AI package so it can report on the next flush."""
    package_id = f"{name}@{version}"
    if package_id in _known:
        return
    _known[package_id] = (name, version)


def flush_ai_sdk_info(client: Any) -> None:
    """Emit ``$ld:ai:sdk:info`` once per unreported registered package."""
    if len(_known) == len(_reported):
        return

    from .utils import to_ld_context

    context = to_ld_context(client, SDK_INFO_CONTEXT)
    for package_id, (name, version) in list(_known.items()):
        if package_id in _reported:
            continue
        try:
            client.track(
                SDK_INFO_EVENT,
                context,
                {
                    "aiSdkName": name,
                    "aiSdkVersion": version,
                    "aiSdkLanguage": _SDK_INFO_LANGUAGE,
                },
                1,
            )
        except Exception:
            pass
        _reported.add(package_id)


def reset_ai_sdk_info(*, clear_known: bool = False) -> None:
    """Mark packages unreported. Optionally drop the registered set (tests)."""
    _reported.clear()
    if clear_known:
        _known.clear()
