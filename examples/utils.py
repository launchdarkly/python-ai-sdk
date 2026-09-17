"""Shared helpers for examples."""

from __future__ import annotations

import dataclasses
import json
import random
import string
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def new_context() -> dict[str, Any]:
    """Returns a random LaunchDarkly user context."""
    key = "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
    return {"kind": "user", "key": key}


def new_multi_context() -> dict[str, Any]:
    """Returns a unique multi-context that exercises canonical-key escaping."""
    return {
        "kind": "multi",
        "organization": {"key": "example-org:west%region"},
        "user": {"key": f"example-user-{uuid4().hex[:8]}"},
    }


def new_conversation_id(label: str) -> str:
    """A fresh conversation id per run.

    A constant would collapse every run — by every developer, and every CI pass — into one
    ever-growing conversation in LaunchDarkly's view: a misleading demo of the very feature it is
    demonstrating. An id should be stable across the turns of one conversation and distinct across
    conversations.
    """
    return f"{label}-{uuid4().hex[:8]}"


def _default_encoder(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return str(obj)


def write_output(data: Any) -> None:
    """Serialises *data* to a timestamped JSON file under output/."""
    out_dir = Path(__file__).parent.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        datetime.now(tz=UTC).isoformat().replace(":", "-").replace(".", "-") + ".json"
    )
    path = out_dir / filename
    path.write_text(
        json.dumps(data, indent=2, default=_default_encoder), encoding="utf-8"
    )
    print(f"Output written to output/{filename}")
