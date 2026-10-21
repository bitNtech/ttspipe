"""`/v1/voice` -- set the one voice every WebSocket turn defaults to.

Set it once here and every `start` frame that names no voice of its own is
conditioned on it, until it is replaced or cleared. That is the whole point:
a conversational client picks a voice at the top of a session and then just
streams text, instead of re-uploading a clip on every turn or teaching every
client how to carry one.

`POST /v1/voice` takes the same pair as `/v1/tts/clone` -- `reference_audio`
and its **exact** `reference_text` -- under the same two body encodings, and
enforces the same rule: a transcript that does not match the clip is not an
error, it is a quietly worse voice, so a missing one is refused outright
rather than accepted.

Setting a voice does not synthesize anything, so it returns immediately - but
it does schedule a warm-up in the background. A reference clip costs a codec
encode *and*, because it is carried as VQ codes in the system prompt, a
prefill of ~220 tokens; leaving both to the first request that speaks in this
voice put 17s of latency on it. `warm_active_voice` pays them here instead,
without making the caller wait (see `app/voice.py`).
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from app.http.common import (
    UPLOADS_KEY,
    UnsupportedMediaType,
    coerce_audio,
    error,
    read_body,
)
from app.voice import VoiceStore, warm_active_voice

logger = logging.getLogger("tts.http")
router = APIRouter()

#: Strong references to in-flight warm-ups; asyncio only holds weak ones, so
#: without this the task can be collected mid-prefill.
_warm_tasks: set[asyncio.Task] = set()

_TRANSCRIPT_REQUIRED = (
    "'reference_text' is required and must be the exact, verbatim transcript "
    "of 'reference_audio' -- not a description of it. Cloning conditions on "
    "the audio and its transcript as a matched pair; a wrong transcript "
    "degrades the voice silently."
)


def _store(request: Request) -> VoiceStore | None:
    return getattr(request.app.state, "voices", None)


@router.get("/v1/voice")
async def get_active_voice(request: Request) -> Response:
    """What the WebSocket will use right now. 404 when nothing is set.

    Metadata only -- the clip itself is at `/v1/voice/audio`, so polling this
    from a UI stays cheap however large the reference is.
    """
    store = _store(request)
    voice = store.get() if store is not None else None
    if voice is None:
        return error(404, "no active voice is set", "NotFound")
    return JSONResponse(status_code=200, content=voice.as_dict())


@router.get("/v1/voice/audio")
async def get_active_voice_audio(request: Request) -> Response:
    """The stored clip itself, so a UI can play back what it actually set."""
    store = _store(request)
    voice = store.get() if store is not None else None
    if voice is None:
        return error(404, "no active voice is set", "NotFound")
    return Response(
        content=voice.audio,
        media_type=voice.content_type or "application/octet-stream",
        headers={
            "X-Voice-Id": voice.voice_id,
            "Content-Disposition": f'inline; filename="{voice.filename}"',
        },
    )


@router.post("/v1/voice")
async def set_active_voice(request: Request) -> Response:
    """Replace the active voice. Takes effect on the next `start` frame.

    Sessions already mid-turn keep the voice they started with: their
    `references` were resolved at `start` and the engine has the encoded
    prompt, so a voice change never swaps timbre partway through a sentence.
    """
    store = _store(request)
    if store is None:  # pragma: no cover - always wired up in app.main
        return error(503, "voice store is unavailable", "Unavailable")

    try:
        fields = await read_body(request)
    except UnsupportedMediaType as exc:
        return error(415, str(exc), "UnsupportedMediaType")
    except Exception as exc:
        return error(400, f"could not parse request body: {exc}")

    uploads = fields.pop(UPLOADS_KEY, {})
    reference_text = str(fields.get("reference_text") or "").strip()
    try:
        audio = coerce_audio(fields.get("reference_audio"), "reference_audio")
    except ValueError as exc:
        return error(400, str(exc))

    if not audio:
        return error(
            400,
            "'reference_audio' is required: a ~10s clip of the voice to clone "
            "(a file part under multipart/form-data, or base64 under JSON)",
        )
    if not reference_text:
        return error(400, _TRANSCRIPT_REQUIRED)

    upload = uploads.get("reference_audio", {})
    try:
        voice = store.set(
            audio=audio,
            text=reference_text,
            filename=str(fields.get("filename") or upload.get("filename") or ""),
            content_type=str(upload.get("content_type") or ""),
        )
    except ValueError as exc:
        return error(400, str(exc))

    # Fire-and-forget: the response must not wait on a GPU prefill, and a
    # warm-up that fails only costs the next turn the latency it would have
    # paid anyway.
    task = asyncio.create_task(warm_active_voice(request.app.state, voice))
    _warm_tasks.add(task)
    task.add_done_callback(_warm_tasks.discard)

    return JSONResponse(status_code=200, content=voice.as_dict())


@router.delete("/v1/voice")
async def clear_active_voice(request: Request) -> Response:
    """Drop the active voice; sessions fall back to the model's default.

    Idempotent, and deliberately a 200 either way: "there is no active voice"
    is the caller's goal, and whether one happened to be set is reported in
    the body rather than as a failure.
    """
    store = _store(request)
    if store is None:  # pragma: no cover - always wired up in app.main
        return error(503, "voice store is unavailable", "Unavailable")
    return JSONResponse(status_code=200, content={"cleared": store.clear()})
