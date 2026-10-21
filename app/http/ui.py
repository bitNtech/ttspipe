"""`GET /` -- a single static page for driving the server by hand.

Served by the app itself, from the same origin, on purpose. The page opens a
WebSocket to `/v1/tts/live` and POSTs clips to `/v1/voice`; from any other
origin both would need CORS, and a page served over https could not reach a
local `ws://` at all. Same-origin sidesteps every one of those.

One file, no build step, no CDN: `static/index.html` is the whole thing,
including its MessagePack decoder, so it works on a boxed-in GPU host with no
outbound network. If the file is missing the route says so plainly instead of
500ing -- deleting it should cost you the test page, not the server.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse, Response

from app.config import REPO_ROOT
from app.http.common import error

router = APIRouter()

_INDEX = REPO_ROOT / "static" / "index.html"


@router.get("/", include_in_schema=False)
@router.get("/ui", include_in_schema=False)
async def index() -> Response:
    if not _INDEX.is_file():
        return error(404, f"the test page is not installed at {_INDEX}", "NotFound")
    # no-store: the page is edited in place during development, and a cached
    # copy of it silently testing an older protocol is a miserable hour.
    return FileResponse(
        _INDEX, media_type="text/html", headers={"Cache-Control": "no-store"}
    )
