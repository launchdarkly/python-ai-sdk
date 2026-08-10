"""Shared helpers for examples."""

from __future__ import annotations

import base64
import dataclasses
import json
import random
import string
import struct
import zlib
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


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def solid_color_png_base64(rgb: tuple[int, int, int], size: int = 64) -> str:
    """Encodes a solid-colour PNG as base64 for multimodal examples.

    Generating the image avoids committing a binary fixture, and the colour is
    the only thing the model can report back — which makes it a usable signal
    for whether the image actually reached the provider.
    """
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")
