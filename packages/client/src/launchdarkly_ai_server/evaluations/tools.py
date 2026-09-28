"""Tools an evaluation run gives to its handler."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..types import NativeTool
from .api import (
    EvaluationsError,
    LDApiClient,
    LDApiError,
    require_mapping,
    segment,
)

ToolImplementation = Callable[..., Any] | NativeTool


@dataclass
class Tool:
    """A tool a run gives to its handler. Pass a list of these to ``run``.

    Construct one to define a tool in code. ``source`` is then ``"inline"`` and
    ``version`` is ``None``. Neither is a constructor argument, so a
    constructed tool is always inline.

    Call ``evals.tools.get(key, implementation=...)`` instead to use a tool
    from the LaunchDarkly tool library. That returns a tool with ``source``
    ``"library"`` and the version it pinned.

    ``implementation`` is the function the handler calls. A key must be
    lowercase. A :class:`~launchdarkly_ai_server.NativeTool` is valid only for
    a library tool, because the provider supplies its schema.
    """

    key: str
    implementation: ToolImplementation
    schema: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    source: Literal["library", "inline"] = field(default="inline", init=False)
    version: int | None = field(default=None, init=False)

    @classmethod
    def _library(
        cls,
        key: str,
        implementation: ToolImplementation,
        *,
        version: int,
        schema: dict[str, Any],
        description: str,
    ) -> Tool:
        """Build a library tool. Used by ``ToolsClient.get``."""
        tool = cls(
            key=key,
            implementation=implementation,
            schema=schema,
            description=description,
        )
        tool.source = "library"
        tool.version = version
        return tool

    def to_create_wire(self) -> dict[str, Any]:
        """The entry this tool contributes to the evaluation-create body."""
        if self.source == "inline":
            return {
                "key": self.key,
                "schema": self.schema,
                "description": self.description,
                "source": "inline",
            }
        return {"key": self.key, "version": self.version, "source": "library"}


def validate_tool_key(key: str) -> None:
    """Validate a tool key. Raises ``EvaluationsError``."""
    if not isinstance(key, str) or not key.strip():
        raise EvaluationsError("tool keys must not be blank")
    # Keys are lowercase.
    if key != key.lower():
        raise EvaluationsError(
            f"Tool key {key!r} must not use uppercase letters. Use "
            f"{key.lower()!r} instead."
        )


def validate_tool_implementation(key: str, implementation: Any) -> None:
    """Validate a tool implementation. Raises ``EvaluationsError``."""
    if not callable(implementation) and not isinstance(implementation, NativeTool):
        raise EvaluationsError(
            f"Tool {key!r} implementation must be callable or a NativeTool, got "
            f"{type(implementation).__name__}"
        )


def _validate_inline_tool(tool: Tool) -> None:
    """Validate one inline tool. Raises ``EvaluationsError``."""
    key = tool.key
    if isinstance(tool.implementation, NativeTool):
        raise EvaluationsError(
            f"Inline tool {key!r} must not use a NativeTool. The provider "
            "supplies the schema of a native tool. Call evals.tools.get() for "
            "a library tool, or give this tool a function."
        )
    if not isinstance(tool.schema, Mapping):
        raise EvaluationsError(
            f"Inline tool {key!r} schema must be a JSON object, got "
            f"{type(tool.schema).__name__}"
        )
    try:
        # allow_nan=False rejects NaN and Infinity, which are not valid JSON.
        json.dumps(dict(tool.schema), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise EvaluationsError(
            f"Inline tool {key!r} schema must be JSON-serializable: {error}"
        ) from error


def validate_tools(tools: Sequence[Tool]) -> None:
    """Validate the tools list. Raises ``EvaluationsError``.

    Issues no requests.
    """
    keys_by_identity: dict[str, str] = {}
    for tool in tools:
        if not isinstance(tool, Tool):
            raise EvaluationsError(
                "each entry in tools must be a Tool. Construct one for an "
                "inline tool, or call evals.tools.get() for a library tool, "
                f"got {type(tool).__name__}"
            )
        key = tool.key
        validate_tool_key(key)
        validate_tool_implementation(key, tool.implementation)
        if not isinstance(tool.description, str):
            raise EvaluationsError(
                f"Tool {key!r} description must be a string, got "
                f"{type(tool.description).__name__}"
            )
        if tool.source == "inline":
            _validate_inline_tool(tool)
        # One key names one tool. Keys are compared case-insensitively.
        identity = key.strip().lower()
        collision = keys_by_identity.get(identity)
        if collision is not None:
            if collision == key:
                raise EvaluationsError(f"Tool {key!r} appears more than once in tools.")
            raise EvaluationsError(
                f"Tool {key!r} collides with {collision!r}. Two tools in one "
                "run must not have keys that differ only by case."
            )
        keys_by_identity[identity] = key


def tool_handlers(tools: Sequence[Tool]) -> dict[str, ToolImplementation]:
    """Return the tools list as ``{key: executable}``."""
    return {tool.key: tool.implementation for tool in tools}


def handler_config_tools(tools: Sequence[Tool]) -> dict[str, dict[str, Any]]:
    """Return the ``tools`` entry of a handler config."""
    return {
        tool.key: {"description": tool.description, "parameters": tool.schema}
        for tool in tools
    }


def create_wire_tools(tools: Sequence[Tool]) -> list[dict[str, Any]]:
    """Return the ``tools`` array of an evaluation-create body."""
    return [tool.to_create_wire() for tool in tools]


class ToolsClient:
    """Reads tools from the LaunchDarkly tool library."""

    def __init__(self, api_client: LDApiClient, project_key: str) -> None:
        self._api = api_client
        self._project_key = project_key

    def get(self, key: str, *, implementation: ToolImplementation) -> Tool:
        """Return the library tool ``key``, paired with ``implementation``.

        Reads the tool now and pins the version it returns. Raises
        ``EvaluationsError`` when the tool does not exist in the project.
        """
        validate_tool_key(key)
        validate_tool_implementation(key, implementation)
        path = f"projects/{segment(self._project_key)}/ai-tools/{segment(key)}"
        try:
            raw = require_mapping(self._api.get(path), description=f"tool {key!r}")
        except LDApiError as error:
            if error.status == 404:
                raise EvaluationsError(
                    f"LaunchDarkly AI tool {key!r} was not found in project "
                    f"{self._project_key!r}"
                ) from error
            raise
        version = raw.get("version")
        if not isinstance(version, int):
            raise EvaluationsError(
                f"LaunchDarkly AI tool {key!r} has no integer version"
            )
        schema = raw.get("schema")
        if not isinstance(schema, Mapping):
            schema = {}
        return Tool._library(
            key,
            implementation,
            version=version,
            schema=dict(schema),
            description=str(raw.get("description") or ""),
        )
