"""One OpenCode session per call, over the stable HTTP API.

``GET /event`` is opened before ``POST /session/{id}/message``. The message
route is the stable API. ``/api/session/{id}/prompt`` is not used: that preview
route rejects the amazon-bedrock provider.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

DRAIN_SECONDS = 0.05

ModelResolver = Callable[[dict[str, Any]], tuple[str, str]]


class OpenCodeHTTPError(RuntimeError):
    """The OpenCode server rejected a session or message call."""


def default_model(config: dict[str, Any]) -> tuple[str, str]:
    """Map an AI Config model onto an OpenCode provider id and model id.

    A ``provider/model`` name is split. Otherwise Bedrock and Gemini use
    OpenCode's provider ids, and any other provider name is lowercased.
    """
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    name = model.get("name") if isinstance(model, dict) else ""
    name = name if isinstance(name, str) else ""
    provider = (
        config.get("provider") if isinstance(config.get("provider"), dict) else {}
    )
    provider_name = provider.get("name") if isinstance(provider, dict) else ""
    provider_name = provider_name if isinstance(provider_name, str) else ""
    if "/" in name:
        provider_id, model_id = name.split("/", 1)
        return provider_id, model_id
    mapped = {"Bedrock": "amazon-bedrock", "Gemini": "google"}
    return mapped.get(provider_name, provider_name.lower()), name


async def session_events(
    *,
    base_url: str,
    prompt: str,
    system: str | None,
    provider_id: str,
    model_id: str,
    agent: str | None,
    session_ids: list[str],
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Open one session and yield its bus events.

    The event stream is opened before the message post. The post runs
    concurrently with the read. ``session.idle`` ends the turn. A returned
    post is followed by a brief drain. Leaving early cancels the post and
    aborts the session.
    """
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=None)
    base = base_url.rstrip("/")
    post: asyncio.Task[httpx.Response] | None = None
    session_id: str | None = None
    succeeded = False
    try:
        async with http.stream("GET", f"{base}/event") as sse:
            sse.raise_for_status()
            created = await http.post(f"{base}/session")
            if created.status_code >= 400:
                raise OpenCodeHTTPError(f"POST /session -> {created.status_code}")
            session_id = _session_id(created.json())
            session_ids.append(session_id)
            body: dict[str, Any] = {
                "parts": [{"type": "text", "text": prompt}],
                "model": {"providerID": provider_id, "modelID": model_id},
            }
            if system:
                body["system"] = system
            if agent:
                body["agent"] = agent
            post = asyncio.create_task(
                http.post(f"{base}/session/{session_id}/message", json=body)
            )
            async for event in _read_until_done(sse, post, session_id):
                yield event
            response = await post
            if response.status_code >= 400:
                raise OpenCodeHTTPError(
                    f"POST /session/{session_id}/message -> {response.status_code}"
                )
            succeeded = True
    finally:
        if not succeeded and session_id is not None:
            if post is not None and not post.done():
                post.cancel()
            try:
                await http.post(f"{base}/session/{session_id}/abort")
            except Exception:
                pass
        if owns_client:
            await http.aclose()


async def _read_until_done(
    response: httpx.Response,
    post: asyncio.Task[httpx.Response],
    session_id: str,
) -> AsyncIterator[dict[str, Any]]:
    lines = response.aiter_lines()
    while True:
        timeout = DRAIN_SECONDS if post.done() else None
        try:
            if timeout is None:
                line = await lines.__anext__()
            else:
                line = await asyncio.wait_for(lines.__anext__(), timeout)
        except StopAsyncIteration:
            return
        except TimeoutError:
            return
        event = _parse_frame(line)
        if event is None:
            continue
        yield event
        if event.get("type") != "session.idle":
            continue
        owner = _frame_session(event)
        if owner is None or owner == session_id:
            return


def _session_id(body: object) -> str:
    if isinstance(body, dict):
        raw = body.get("id")
        if isinstance(raw, str) and raw:
            return raw
        data = body.get("data")
        if isinstance(data, dict):
            nested = data.get("id")
            if isinstance(nested, str) and nested:
                return nested
    raise OpenCodeHTTPError("session create returned no id")


def _parse_frame(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if text.startswith("data:"):
        text = text[5:].lstrip()
    if not text or text[0] != "{":
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _frame_session(event: dict[str, Any]) -> str | None:
    properties = event.get("properties")
    if not isinstance(properties, dict):
        return None
    raw = properties.get("sessionID")
    return raw if isinstance(raw, str) and raw else None
