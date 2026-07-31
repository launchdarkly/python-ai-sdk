"""Tool functions used by example handlers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PREFS_PATH = Path(__file__).parent / "user_preferences.json"

_llms_txt_cache: str | None = None
_LLMS_TXT_URL = "https://launchdarkly.com/docs/llms.txt"


async def get_preferences(args: dict[str, Any]) -> str:
    """Return stored preferences for a user from user_preferences.json."""
    user_id: str = args.get("user_id", "")
    try:
        data: dict[str, Any] = json.loads(_PREFS_PATH.read_text(encoding="utf-8"))
        prefs = data.get(user_id)
        if prefs is None:
            return f'No preferences found for user "{user_id}".'
        return json.dumps(prefs)
    except Exception as exc:
        return f"Error reading preferences: {exc}"


async def web_search(args: dict[str, Any]) -> str:
    """Stub web search — wire up a real provider to use."""
    query: str = args.get("query", "")
    return f"Search results for '{query}': (stub – wire up a real search provider)"


async def _get_llms_txt() -> str:
    global _llms_txt_cache
    if _llms_txt_cache:
        return _llms_txt_cache
    import httpx

    async with httpx.AsyncClient(
        headers={"User-Agent": "LD-Docs-Agent/1.0"}, timeout=10, follow_redirects=True
    ) as client:
        resp = await client.get(_LLMS_TXT_URL)
        resp.raise_for_status()
        _llms_txt_cache = resp.text
    return _llms_txt_cache


async def search_ld_documentation(args: dict[str, Any]) -> str:
    """Search the LaunchDarkly docs index (llms.txt) for matching pages."""
    query: str = args.get("query", "")
    try:
        index = await _get_llms_txt()
        terms = [t for t in query.lower().split() if t]
        results = [
            line
            for line in index.splitlines()
            if line.startswith("- [") and all(t in line.lower() for t in terms)
        ][:10]
        if not results:
            return f'No documentation results found for "{query}".'
        return "\n".join(results)
    except Exception as exc:
        return f"Search error: {exc}"


async def fetch_launchdarkly_documentation(args: dict[str, Any]) -> str:
    """Fetch a LaunchDarkly documentation page (appends .md for clean Markdown)."""
    import httpx

    url: str = args.get("url", "").rstrip("/")
    md_url = url if url.endswith(".md") else f"{url}.md"
    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": "LD-Docs-Agent/1.0"},
            timeout=10,
            follow_redirects=True,
        ) as client:
            resp = await client.get(
                md_url, headers={"Accept": "text/markdown,text/plain,*/*"}
            )
            if resp.status_code >= 400:
                resp = await client.get(url)
                if resp.status_code >= 400:
                    return f'Failed to fetch "{url}": HTTP {resp.status_code}'
                import re

                plain = re.sub(r"<[^>]+>", " ", resp.text)
                plain = re.sub(r"\s+", " ", plain).strip()[:8000]
                return plain

            text = resp.text
        return text[:10000] + "\n\n[…content truncated…]" if len(text) > 10000 else text
    except Exception as exc:
        return f"Fetch error: {exc}"
