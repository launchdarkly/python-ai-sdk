"""
buildOutputType utility — converts an LD AI config outputFormat into the
JsonSchemaDefinition envelope that the OpenAI Agents SDK expects.
"""

from __future__ import annotations

from typing import Any


def build_output_type(output_format: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Converts the ``outputFormat`` from an LD AI config into the schema envelope
    the OpenAI Agents SDK expects as ``output_type`` on an ``Agent``.

    * Returns ``None`` when *output_format* is absent or empty.
    * Ensures ``type: 'object'`` at the schema root.
    * Populates ``required`` from all property keys (OpenAI Structured Outputs
      requires every property in ``required`` when ``additionalProperties: false``).

    Note: The Python Agents SDK requires a concrete Python type for
    ``output_type``, not a raw JSON schema dict. Callers should not pass the
    result of this function directly to ``Agent(output_type=...)``. Use it to
    inspect or log the expected schema shape, or extend with dynamic Pydantic
    model generation when structured output is needed at runtime.
    """
    if not output_format:
        return None

    properties = output_format.get("properties")
    if isinstance(properties, dict):
        required: list[str] = list(properties.keys())
    else:
        required = list(output_format.get("required") or [])

    schema: dict[str, Any] = {
        **output_format,
        "type": "object",
        **({"required": required} if required else {}),
    }

    return {
        "type": "json_schema",
        "name": "output",
        "strict": False,
        "schema": schema,
    }
