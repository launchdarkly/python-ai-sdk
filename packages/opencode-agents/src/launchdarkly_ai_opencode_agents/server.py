"""One OpenCode session per call, over the stable HTTP API.

``GET /event`` is opened before ``POST /session/{id}/message``. The message
route is the stable API. ``/api/session/{id}/prompt`` is not used: that preview
route rejects the amazon-bedrock provider. The base URL is the loopback server
this process just started. It is not a caller argument.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from typing import Any

import httpx

# After the message POST returns, keep reading until this session's
# ``session.idle``. A quiet bus ends the turn only after this fallback, so a
# late ``message.updated`` is not dropped the way a 50 ms drain dropped it.
DRAIN_FALLBACK_SECONDS = 5.0
# Session create and abort. The message POST blocks for the whole turn, so
# its timeout is longer, and still finite.
HTTP_TIMEOUT_SECONDS = 30.0
MESSAGE_TIMEOUT_SECONDS = 600.0

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
    password: str,
    prompt: str,
    system: str | None,
    provider_id: str,
    model_id: str,
    agent: str | None,
    session_ids: list[str],
) -> AsyncGenerator[dict[str, Any], None]:
    """Open one session and yield its bus events.

    The handler owns the HTTP client. ``base_url`` is the ``opencode serve``
    this call started. ``password`` is the secret that start set on the child,
    sent as basic auth with username ``opencode``. Callers do not supply a
    client or a transport. The event stream is opened before the message post.
    The post runs concurrently with the read. This session's ``session.idle`` ends the
    turn. A failed message POST raises without waiting for another bus
    frame. Leaving early cancels the post and aborts the session.
    """
    http = httpx.AsyncClient(
        auth=("opencode", password),
        timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS, read=MESSAGE_TIMEOUT_SECONDS),
    )
    base = base_url.rstrip("/")
    post: asyncio.Task[httpx.Response] | None = None
    session_id: str | None = None
    succeeded = False
    try:
        async with http.stream(
            "GET",
            f"{base}/event",
            timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS, read=None),
        ) as sse:
            sse.raise_for_status()
            created = await http.post(f"{base}/session", timeout=HTTP_TIMEOUT_SECONDS)
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
                http.post(
                    f"{base}/session/{session_id}/message",
                    json=body,
                    timeout=MESSAGE_TIMEOUT_SECONDS,
                )
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
                    await post
                except BaseException:
                    pass
            try:
                await http.post(
                    f"{base}/session/{session_id}/abort",
                    timeout=HTTP_TIMEOUT_SECONDS,
                )
            except Exception:
                pass
        await http.aclose()


async def _read_until_done(
    response: httpx.Response,
    post: asyncio.Task[httpx.Response],
    session_id: str,
) -> AsyncGenerator[dict[str, Any], None]:
    """Read the bus until this session is idle, or the message POST fails.

    The next SSE line is raced against the post task. A status of 400 or
    higher raises immediately, even when ``/event`` stays open. After the
    post returns successfully, reading continues until this session's
    ``session.idle`` or ``DRAIN_FALLBACK_SECONDS`` of silence. An idle event
    with no session id does not end the turn.
    """
    lines = response.aiter_lines()
    line_task: asyncio.Task[str | None] | None = None
    try:
        while True:
            if post.done():
                _raise_if_message_failed(post, session_id)
            if line_task is None:
                line_task = asyncio.create_task(_next_line(lines))
            waiting: set[asyncio.Task[Any]] = {line_task}
            if not post.done():
                waiting.add(post)
            finished, _pending = await asyncio.wait(
                waiting,
                timeout=DRAIN_FALLBACK_SECONDS if post.done() else None,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not finished:
                return
            if post in finished:
                _raise_if_message_failed(post, session_id)
            if line_task not in finished:
                continue
            line = line_task.result()
            line_task = None
            if line is None:
                return
            event = _parse_frame(line)
            if event is None:
                continue
            yield event
            if (
                event.get("type") == "session.idle"
                and _frame_session(event) == session_id
            ):
                return
    finally:
        # Cancel and await before the caller closes GET /event. A bare cancel
        # leaves this read running into the closed stream. The quiet-bus
        # return and a failed message POST both leave it pending.
        if line_task is not None and not line_task.done():
            line_task.cancel()
        if line_task is not None:
            try:
                await line_task
            except asyncio.CancelledError:
                pass


async def _next_line(lines: Any) -> str | None:
    try:
        line = await lines.__anext__()
    except StopAsyncIteration:
        return None
    return line if isinstance(line, str) else None


def _raise_if_message_failed(
    post: asyncio.Task[httpx.Response], session_id: str
) -> None:
    response = post.result()
    if response.status_code >= 400:
        raise OpenCodeHTTPError(
            f"POST /session/{session_id}/message -> {response.status_code}"
        )


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
