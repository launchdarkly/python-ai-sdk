"""Shared helpers for examples."""

from __future__ import annotations

import dataclasses
import json
import random
import string
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def new_context() -> dict[str, Any]:
    """Returns a random LaunchDarkly user context."""
    key = "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
    return {"kind": "user", "key": key}


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
