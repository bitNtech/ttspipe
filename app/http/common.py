"""Shared plumbing for the HTTP routes: one error shape, one body reader.

Both `/v1/tts/clone` and `/v1/voice` take the same thing -- an audio clip and
its exact transcript -- from the same two kinds of client, so they read their
bodies and report their failures identically rather than each inventing its
own conventions.

Two body encodings are supported on every route that takes a clip:

  * `multipart/form-data` -- the clip is a file part. This is what `curl -F`
    and a browser `<form>`/`FormData` produce, and it does not inflate a
    megabyte of audio by a third the way base64 does.
  * `application/json` -- the clip is base64. This is the same shape as the
    WebSocket protocol's `references[].audio`, so code that already talks to
    `/v1/tts/live` ports across unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.engine.base import EngineError
from app.metrics import METRICS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.synthesis import OneShotSynthesis

logger = logging.getLogger("tts.http")

#: Longest text a one-shot request may carry. Generation is serial on one
#: model worker, so an unbounded body is a way to occupy it for minutes.
MAX_TEXT_CHARS = 8000

#: A reference clip is ~10s of speech -- a few hundred KB. This is two orders
#: of magnitude above that: high enough never to reject a legitimate clip
#: (including an uncompressed stereo one), low enough that a mistaken upload
#: fails fast instead of being read into memory in full.
MAX_CLIP_BYTES = 32 * 1024 * 1024

#: Form parts whose value is the empty string are dropped before validation:
#: an HTML form posts every field it has, including the ones left blank, and
#: `top_p=""` should mean "unset", not "invalid float". Routes name the fields
#: that are exempt, because blank there is a real error with its own message.
_KEEP_EMPTY: tuple[str, ...] = ("text", "reference_text")


class UnsupportedMediaType(ValueError):
    """The body encoding is not one of the two these routes speak."""


def error(status: int, message: str, error_type: str = "BadRequest") -> JSONResponse:
    """The same body shape `app.main`'s exception handler returns."""
    return JSONResponse(status_code=status, content={"error": message, "type": error_type})


#: Where `read_body` files away each upload's filename and declared media
#: type: one reserved key, so a route pops it once instead of having to sieve
#: synthesized `<field>_filename` entries out of the caller's real fields.
UPLOADS_KEY = "_uploads"


async def read_body(request: Request) -> dict[str, Any]:
    """Normalise either supported encoding into one flat dict of fields.

    File parts come back as `bytes`; everything else as the string the client
    sent, which pydantic coerces on the way into `TTSParams`. Upload metadata
    lands under `UPLOADS_KEY`.
    """
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()

    if content_type == "application/json":
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return dict(payload)

    if content_type in ("multipart/form-data", "application/x-www-form-urlencoded"):
        form = await request.form()
        fields: dict[str, Any] = {}
        uploads: dict[str, dict[str, str]] = {}
        for key in form:
            value = form[key]
            if hasattr(value, "read"):  # an UploadFile part
                fields[key] = await value.read()
                uploads[key] = {
                    "filename": getattr(value, "filename", "") or "",
                    "content_type": getattr(value, "content_type", "") or "",
                }
            elif value == "" and key not in _KEEP_EMPTY:
                continue
            else:
                fields[key] = value
        if uploads:
            fields[UPLOADS_KEY] = uploads
        return fields

    raise UnsupportedMediaType(
        "unsupported Content-Type "
        f"{content_type or '(none)'}; use multipart/form-data or application/json"
    )


def coerce_audio(value: Any, field: str) -> bytes | None:
    """A clip as `bytes`, from raw bytes or a base64 string. None if absent.

    Raises `ValueError` for a value that was clearly meant to be audio but
    cannot be -- a wrong-typed JSON field, bad base64, or something far too
    large to be a reference clip.
    """
    if value is None or value == "":
        return None
    if isinstance(value, str):
        import base64

        try:
            value = base64.b64decode(value, validate=True)
        except Exception as exc:
            raise ValueError(f"{field!r} must be raw bytes or base64") from exc
    if not isinstance(value, (bytes, bytearray)):
        raise ValueError(f"{field!r} must be a file part or a base64 string")
    if len(value) > MAX_CLIP_BYTES:
        raise ValueError(
            f"{field!r} is {len(value) / 1024 / 1024:.1f} MiB; the limit is "
            f"{MAX_CLIP_BYTES // 1024 // 1024} MiB. A reference clip should be ~10s of speech."
        )
    return bytes(value)


def engine_or_error(request: Request):
    """-> (engine, None) when it can serve now, else (None, error response)."""
    engine = request.app.state.engine
    load_error = getattr(request.app.state, "engine_load_error", None)
    if load_error is not None:
        return None, error(503, f"engine unavailable: {load_error}", "EngineError")
    if engine is None or not engine.ready:
        return None, error(503, "engine is still loading; poll /ready", "EngineNotReady")
    return engine, None


async def stream_synthesis(
    synthesis: "OneShotSynthesis", label: str, headers: dict[str, str] | None = None
) -> Response:
    """The response for a prepared one-shot synthesis, streamed as it decodes.

    The first payload is primed here, while a status line can still be
    changed: everything expensive that can fail cleanly - reference encoding,
    model readiness, the first generation - fails during this await and
    becomes a real HTTP error. After that the status line is gone, and a
    failure can only truncate the body (and is logged, so it is visible).
    """
    request_id = synthesis.request_id
    body = synthesis.stream()
    try:
        first = await anext(body, None)
    except EngineError as exc:
        await body.aclose()
        METRICS.http_requests.inc(status="500")
        logger.warning("%s %s: synthesis failed before first byte: %s", label, request_id, exc)
        return error(500, str(exc), type(exc).__name__)
    METRICS.http_requests.inc(status="200")

    async def stream_body():
        try:
            if first is not None:
                yield first
            async for piece in body:
                yield piece
        except EngineError as exc:
            logger.error("%s %s: synthesis failed mid-stream: %s", label, request_id, exc)
        finally:
            await body.aclose()  # cancels the engine job if the client vanished

    out = {
        "X-Request-Id": request_id,
        "X-Audio-Format": synthesis.format,
        "X-Sample-Rate": str(synthesis.sample_rate),
        "X-Seed": str(synthesis.seed),
        **(headers or {}),
    }
    if synthesis.warning:
        out["X-Warning"] = synthesis.warning
        logger.info("%s %s: %s", label, request_id, synthesis.warning)
    logger.info(
        "%s %s: %d segment(s), format=%s rate=%d seed=%d",
        label, request_id, len(synthesis.segments), synthesis.format,
        synthesis.sample_rate, synthesis.seed,
    )
    return StreamingResponse(stream_body(), media_type=synthesis.media_type, headers=out)
