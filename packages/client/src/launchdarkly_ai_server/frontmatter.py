"""
``SKILL.md`` frontmatter parsing.

A lazy convenience for ``Skill.frontmatter()``, and deliberately nothing more:
this is **never** part of the integrity path. It lives in its own module rather than in ``types.py`` because
``types.py`` is the package's shared declarative type surface — imported by every
handler package — and a YAML loading strategy has no business riding along with
``LDContext``.

Parsing is bounded on every axis a hostile document could exploit: the block is
at most 8 KB, nesting at most 10 levels deep, alias/anchor resolution is disabled
outright, and only a safe loader is used so no object can be constructed. Every
failure degrades to ``None``; nothing here raises.
"""

from __future__ import annotations

from typing import Any

_FRONTMATTER_MAX_BYTES = 8 * 1024
"""Upper bound on the leading frontmatter block handed to the YAML parser."""

_FRONTMATTER_MAX_DEPTH = 10
"""Upper bound on frontmatter nesting depth."""


def extract_block(content: str) -> str | None:
    """
    Returns the body of the leading ``---`` block, or ``None``.

    Delimiters are anchored at column 0 and compared with ``rstrip()``, not
    ``strip()``. Every convention this format follows (Jekyll, gray-matter,
    python-frontmatter, the agentskills.io ``SKILL.md`` layout) anchors them
    that way, and YAML itself only recognises a document marker at column 0.
    Stripping the *left* side would let an indented ``---`` or ``...`` — which
    is ordinary text inside a block scalar — terminate the block early and
    return a truncated mapping the caller could not distinguish from the real
    one. ``rstrip()`` still tolerates a trailing ``\r`` (CRLF files) and
    trailing spaces.

    Scanned newline by newline rather than over ``rest.split("\\n")``: splitting
    materializes every line of a document that may be 64 KB to find a delimiter
    that can only matter in the first 8 KB. The offset bail below is what makes
    that bound explicit — past it, any block found would be rejected as oversize
    anyway, so a document with no delimiter at all stops costing more to reject
    the larger it gets.
    """
    first_newline = content.find("\n")
    if first_newline == -1 or content[:first_newline].rstrip() != "---":
        return None

    rest = content[first_newline + 1 :]
    offset = 0
    while True:
        newline = rest.find("\n", offset)
        end = len(rest) if newline == -1 else newline
        if rest[offset:end].rstrip() in ("---", "..."):
            block = rest[:offset]
            try:
                oversize = len(block.encode("utf-8")) > _FRONTMATTER_MAX_BYTES
            except UnicodeEncodeError:
                # Unpaired surrogates: not authentic content, and not something
                # this convenience accessor may raise over.
                return None
            return None if oversize else block
        if newline == -1:
            return None  # unterminated block
        offset = newline + 1
        # A block ending at this offset is `offset` characters long, and UTF-8 is
        # never shorter than one byte per character, so every delimiter from here
        # on yields a block the size check would reject.
        if offset > _FRONTMATTER_MAX_BYTES:
            return None


def parse_block(block: str) -> dict[str, Any] | None:
    """
    Safe, bounded YAML parse of an already-size-checked frontmatter block.

    Everything — the import, the loader subclass, and the parse — sits inside
    one guard. ``import yaml`` succeeding does not prove PyYAML is what was
    imported: a shadowing ``yaml.py``, a partial install, or an unrelated module
    of the same name would make ``yaml.SafeLoader`` an ``AttributeError`` at
    class-creation time. This accessor is documented to return ``None`` rather
    than raise, so every one of those degrades to ``None``.
    """
    try:
        # Imported here, not at module scope: pyyaml is a development-only
        # dependency and must never become a runtime one.
        import yaml  # type: ignore[import-untyped]

        class _BoundedSafeLoader(yaml.SafeLoader):  # type: ignore[misc]
            """
            ``SafeLoader`` that refuses aliases and bounds nesting depth.

            Aliases are disabled outright rather than counted: PyYAML resolves
            them as shared references, so the classic billion-laughs document
            parses in about a millisecond and no size or depth bound would
            reject it. Making the presence of an alias itself disqualifying is
            also what keeps the Python and TypeScript implementations in
            agreement on the same input.
            """

            _depth = 0

            def compose_node(self, parent: Any, index: Any) -> Any:
                if self.check_event(yaml.AliasEvent):
                    raise yaml.YAMLError("alias nodes are not permitted in frontmatter")
                self._depth += 1
                try:
                    if self._depth > _FRONTMATTER_MAX_DEPTH:
                        raise yaml.YAMLError("frontmatter nesting is too deep")
                    return super().compose_node(parent, index)
                finally:
                    self._depth -= 1

        parsed = yaml.load(block, Loader=_BoundedSafeLoader)
    except Exception:
        return None

    return parsed if isinstance(parsed, dict) else None
