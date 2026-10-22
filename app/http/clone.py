"""`POST /v1/tts/clone` -- speak arbitrary text in the voice of an uploaded clip.

The WebSocket API can already clone a voice (`references` on the `start`
frame), but it asks the caller to speak msgpack, hold a connection open and
reassemble audio frames. The common case does not need any of that: here is a
clip, here is exactly what is said in it, here is what I want said instead,
give me back a file. That is this route.

Either body encoding works -- `multipart/form-data` with the clip as a file
part, or `application/json` with it base64 in `reference_audio`. See
`app/http/common.py`.

This route always clones the clip attached to *this* request. It deliberately
ignores the server's active voice (`POST /v1/voice`), because a route whose
whole job is "clone this" should not sometimes clone something else.

**The transcript is not optional and it is not a description.**
`reference_text` must be, verbatim, the words spoken in the clip. In-context
cloning conditions generation on (prompt codes, prompt text) as a matched
pair: the model infers the speaker's timbre and delivery from the alignment
between the two. A transcript that does not match what the audio says
mis-teaches that alignment, and the failure is not an error -- it is a worse
voice, mispronunciations, or invented words. So an empty transcript is
rejected here rather than quietly accepted.

The response body streams: the first bytes leave as soon as the first segment
has decoded, not after the last. A failure *before* the first byte becomes a
real HTTP status (the body is primed inside the handler, before the response
is committed); after that the status line is already sent, so a failure can
only truncate the stream and be logged.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import Response
from pydantic import ValidationError

from app.config import Settings
from app.engine.base import EngineNotReady
from app.http.common import (
    UPLOADS_KEY,
    UnsupportedMediaType,
    coerce_audio,
    engine_or_error,
    error,
    read_body,
    stream_synthesis,
)
from app.protocol import ReferenceAudio, TTSParams
from app.synthesis import OneShotSynthesis

router = APIRouter()

#: Set by the caller, never inferred from the body. `references` and
#: `reference_id` would make "which voice did I get?" ambiguous on a route
#: that exists to clone an upload; `use_active_voice` has nothing to switch
#: off here, since this route never consults the active voice at all.
_RESERVED_FIELDS = ("references", "reference_id", "use_active_voice")


@router.post("/v1/tts/clone")
async def clone(request: Request) -> Response:
    engine, problem = engine_or_error(request)
    if problem is not None:
        return problem
    settings: Settings = request.app.state.settings

    try:
        fields = await read_body(request)
    except UnsupportedMediaType as exc:
        return error(415, str(exc), "UnsupportedMediaType")
    except Exception as exc:  # malformed JSON, truncated multipart, ...
        return error(400, f"could not parse request body: {exc}")

    fields.pop(UPLOADS_KEY, None)

    for name in _RESERVED_FIELDS:
        if name in fields:
            return error(
                400,
                f"{name!r} is not accepted here: /v1/tts/clone always clones the "
                "clip you upload. Use the WebSocket API's 'start' frame for "
                "stored voices.",
            )

    text = str(fields.pop("text", "") or "").strip()
    reference_text = str(fields.pop("reference_text", "") or "").strip()
    try:
        reference_audio = coerce_audio(fields.pop("reference_audio", None), "reference_audio")
    except ValueError as exc:
        return error(400, str(exc))

    if not text:
        return error(400, "'text' is required: the words to speak in the cloned voice")
    if not reference_audio:
        return error(
            400,
            "'reference_audio' is required: a ~10s clip of the voice to clone "
            "(a file part under multipart/form-data, or base64 under JSON)",
        )
    if not reference_text:
        return error(
            400,
            "'reference_text' is required and must be the exact, verbatim "
            "transcript of 'reference_audio' -- not a description of it. "
            "Cloning conditions on the audio and its transcript as a matched "
            "pair; a wrong transcript degrades the voice silently.",
        )

    try:
        reference = ReferenceAudio(audio=reference_audio, text=reference_text)
        params = TTSParams(text="", references=[reference], **fields)
    except ValidationError as exc:
        return error(422, str(exc.errors(include_url=False)), "ValidationError")

    request_id = uuid.uuid4().hex
    try:
        synthesis = OneShotSynthesis(
            engine=engine,
            settings=settings,
            params=params,
            text=text,
            request_id=request_id,
        )
    except EngineNotReady as exc:
        return error(503, str(exc), "EngineNotReady")

    return await stream_synthesis(synthesis, "clone")
