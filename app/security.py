"""API-key authentication for every route, HTTP and WebSocket alike.

Off unless `TTS_API_KEYS` is set, so a loopback-only deployment and the test
suite behave exactly as before. Once set, a request must present one of the
keys, in order of preference:

  * `Authorization: Bearer <key>` - what server-side agents should send,
  * `X-API-Key: <key>`,
  * `?api_key=<key>` - only for browsers, whose WebSocket constructor and
    `<audio src>` cannot set headers. It ends up in proxy logs, which is why
    it is last; uvicorn's own access log is off in `run.sh` for this reason.

It is a raw ASGI middleware rather than a FastAPI dependency so the WebSocket
check happens before the handshake is accepted: an unauthenticated socket is
refused with an HTTP 403 and never reaches `Session`, never counts against
`max_sessions`, and never costs the engine anything.

`/health` and `/ready` stay open because orchestrators probe them without
credentials, and they reveal nothing beyond "up" or "not yet".
"""

from __future__ import annotations

import hmac
import logging
from typing import Iterable
from urllib.parse import parse_qs

from app.metrics import METRICS

logger = logging.getLogger("tts.security")

#: Reachable without a key. Exact paths only.
PUBLIC_PATHS = frozenset({"/health", "/ready"})


def _header(scope, name: bytes) -> str | None:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            return value.decode("latin-1")
    return None


def presented_key(scope) -> str | None:
    auth = _header(scope, b"authorization")
    if auth and auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    key = _header(scope, b"x-api-key")
    if key:
        return key.strip()
    query = scope.get("query_string", b"").decode("latin-1")
    values = parse_qs(query).get("api_key")
    return values[0] if values else None


def key_is_valid(presented: str | None, keys: Iterable[str]) -> bool:
    if not presented:
        return False
    # Compare against every key so the time taken does not say which matched.
    ok = False
    for key in keys:
        ok |= hmac.compare_digest(presented.encode(), key.encode())
    return ok


class APIKeyMiddleware:
    def __init__(self, app, keys: Iterable[str]) -> None:
        self.app = app
        self.keys = tuple(keys)

    async def __call__(self, scope, receive, send):
        if (
            not self.keys
            or scope["type"] not in ("http", "websocket")
            or scope.get("path") in PUBLIC_PATHS
            or (scope["type"] == "http" and scope.get("method") == "OPTIONS")  # CORS preflight
            or key_is_valid(presented_key(scope), self.keys)
        ):
            await self.app(scope, receive, send)
            return

        METRICS.auth_failures.inc()
        client = scope.get("client") or ("?", 0)
        logger.warning("refused %s %s from %s: missing or bad API key", scope["type"], scope.get("path"), client[0])
        if scope["type"] == "websocket":
            # Closing before accept makes the server answer the upgrade with 403.
            await send({"type": "websocket.close", "code": 1008, "reason": "unauthorized"})
            return
        body = b'{"error":"missing or invalid API key","type":"Unauthorized"}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", b"Bearer"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
