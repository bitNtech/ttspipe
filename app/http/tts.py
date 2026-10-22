"""Plain text-to-speech over HTTP, an OpenAI-compatible speech route, voice
listing and a capabilities document.

    POST /v1/tts            text in, audio streamed out, in the server's voice
    POST /v1/audio/speech   the same, shaped like OpenAI's endpoint, so an agent
                            framework's stock OpenAI TTS client (LiveKit,
                            Pipecat, the openai SDK) works with a base-URL change
    GET  /v1/voices         every voice a request can name
    GET  /v1/info           what this server can do, for clients to configure from

The WebSocket (`/v1/tts/live`) stays the low-latency path for a live
conversation - text can arrive before it is finished and be interrupted.
These routes are for when the whole text is known up front: a greeting, a
hold message, a notification. They still stream: the first bytes leave when
the first segment decodes, not when the last does.

Voices resolve exactly as a WebSocket `start` frame's do (`app.voice.resolve_voice`):
an explicit `reference_id` or inline `references` wins, otherwise the active
voice (`POST /v1/voice`), otherwise the model's own.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from app.audio import SUPPORTED_SAMPLE_RATES
from app.config import REPO_ROOT, Settings
from app.engine.base import EngineNotReady
from app.http.common import (
    MAX_TEXT_CHARS,
    UPLOADS_KEY,
    UnsupportedMediaType,
    engine_or_error,
    error,
    read_body,
    stream_synthesis,
)
from app.protocol import AUDIO_FORMATS, TTSParams
from app.synthesis import OneShotSynthesis
from app.voice import ACTIVE_VOICE_ID, named_voices, resolve_voice

router = APIRouter()

REFERENCES_DIR = REPO_ROOT / "references"

#: OpenAI's voice names. A client written against OpenAI sends one of these
#: and means "a voice"; here that is the active voice.
_OPENAI_VOICES = frozenset(
    {"alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"}
)
#: `voice` values that ask for the model's own, uncloned voice.
_MODEL_VOICE = frozenset({"model", "none"})


def _check_text(text: str, field: str) -> Response | None:
    if not text:
        return error(400, f"{field!r} is required: the text to speak")
    if len(text) > MAX_TEXT_CHARS:
        return error(
            413,
            f"{field!r} is {len(text)} characters; the limit is {MAX_TEXT_CHARS}. "
            "Split it, or stream it over /v1/tts/live.",
            "PayloadTooLarge",
        )
    return None


def _check_reference(params: TTSParams) -> Response | None:
    """An unknown voice is the caller's mistake: say so with a 404 up front
    rather than as a 500 from deep inside the engine."""
    rid = params.reference_id
    if rid and rid != ACTIVE_VOICE_ID and not (REFERENCES_DIR / rid).is_dir():
        known = [v["id"] for v in named_voices(REFERENCES_DIR)]
        return error(404, f"unknown voice {rid!r}; known: {known or 'none'}", "UnknownVoice")
    if rid == ACTIVE_VOICE_ID and not (REFERENCES_DIR / ACTIVE_VOICE_ID).is_dir():
        return error(404, "no active voice is set; POST /v1/voice first", "UnknownVoice")
    return None


async def _synthesize(
    request: Request, params: TTSParams, text: str, label: str, headers: dict | None = None
) -> Response:
    engine, problem = engine_or_error(request)
    if problem is not None:
        return problem
    settings: Settings = request.app.state.settings
    params, voice_id = resolve_voice(params, getattr(request.app.state, "voices", None))
    try:
        synthesis = OneShotSynthesis(
            engine=engine, settings=settings, params=params, text=text,
            request_id=uuid.uuid4().hex,
        )
    except EngineNotReady as exc:
        return error(503, str(exc), "EngineNotReady")
    return await stream_synthesis(
        synthesis, label, {"X-Voice-Id": voice_id or "model", **(headers or {})}
    )


@router.post("/v1/tts")
async def tts(request: Request) -> Response:
    """`text` plus any `start`-frame request field: format, sample_rate,
    reference_id, use_active_voice, seed, temperature, top_p, ..."""
    try:
        fields = await read_body(request)
    except UnsupportedMediaType as exc:
        return error(415, str(exc), "UnsupportedMediaType")
    except Exception as exc:  # malformed JSON, truncated multipart, ...
        return error(400, f"could not parse request body: {exc}")
    fields.pop(UPLOADS_KEY, None)

    text = str(fields.pop("text", "") or "").strip()
    problem = _check_text(text, "text")
    if problem is not None:
        return problem
    try:
        params = TTSParams(**fields)
    except ValidationError as exc:
        return error(422, str(exc.errors(include_url=False)), "ValidationError")
    problem = _check_reference(params)
    if problem is not None:
        return problem
    return await _synthesize(request, params, text, "tts")


@router.post("/v1/audio/speech")
async def openai_speech(request: Request) -> Response:
    """OpenAI's `POST /v1/audio/speech`: {model, input, voice, response_format,
    speed, instructions}. `model` and `instructions` are accepted and ignored
    (there is one model; style goes inline as tags like `[whispers]`).

    `voice` is an OpenAI voice name (-> the active voice), "model" (the
    model's own voice) or the id of a voice under references/. `pcm` is 24 kHz
    as OpenAI defines it; other formats keep the native rate unless the
    non-standard `sample_rate` field says otherwise.
    """
    try:
        body = await request.json()
    except Exception as exc:
        return error(400, f"body must be JSON: {exc}")
    if not isinstance(body, dict):
        return error(400, "body must be a JSON object")

    text = str(body.get("input") or "").strip()
    problem = _check_text(text, "input")
    if problem is not None:
        return problem

    fmt = str(body.get("response_format") or "mp3").lower()
    if fmt not in ("mp3", "wav", "flac", "pcm"):
        return error(
            400,
            f"response_format {fmt!r} is not supported here; use mp3, wav, flac or pcm",
        )
    fields: dict = {"format": fmt}
    if fmt == "pcm":
        fields["sample_rate"] = 24000  # what OpenAI clients assume of raw pcm
    for name in ("sample_rate", "seed", "temperature", "top_p", "language"):
        if body.get(name) is not None:
            fields[name] = body[name]

    voice = str(body.get("voice") or "").strip()
    if voice.lower() in _MODEL_VOICE:
        fields["use_active_voice"] = False
    elif voice and voice.lower() not in _OPENAI_VOICES and voice not in ("active", "default"):
        fields["reference_id"] = voice

    try:
        params = TTSParams(**fields)
    except ValidationError as exc:
        return error(422, str(exc.errors(include_url=False)), "ValidationError")
    problem = _check_reference(params)
    if problem is not None:
        return problem

    headers = {}
    speed = body.get("speed")
    if speed not in (None, 1, 1.0):
        headers["X-Warning"] = "speed is not supported and was ignored"
    return await _synthesize(request, params, text, "speech", headers)


@router.get("/v1/voices")
async def voices(request: Request) -> JSONResponse:
    store = getattr(request.app.state, "voices", None)
    active = store.get() if store is not None else None
    return JSONResponse(
        {
            # What a request that names no voice gets.
            "default": "active" if active is not None else "model",
            "active": active.as_dict() if active is not None else None,
            # Name one of these as `reference_id` (or `voice` on /v1/audio/speech).
            "voices": named_voices(REFERENCES_DIR),
        }
    )


@router.get("/v1/info")
async def info(request: Request) -> JSONResponse:
    settings: Settings = request.app.state.settings
    engine = request.app.state.engine
    status = engine.status() if engine is not None else None
    return JSONResponse(
        {
            "engine": status.backend if status else None,
            "model": status.model_repo if status else None,
            "device": status.device if status else None,
            "ready": bool(engine is not None and engine.ready),
            "native_sample_rate": getattr(engine, "sample_rate", 0) or None,
            "sample_rates": list(SUPPORTED_SAMPLE_RATES),
            "formats": list(AUDIO_FORMATS),
            "defaults": {
                "format": settings.default_format,
                "sample_rate": settings.default_sample_rate or None,
                "chunk_length": settings.chunk_length,
            },
            "limits": {
                "max_sessions": settings.max_sessions,
                "max_text_chars": MAX_TEXT_CHARS,
                "session_idle_timeout_s": settings.session_idle_timeout_s,
            },
            "auth_required": bool(settings.api_keys),
            "routes": {
                "stream": "/v1/tts/live",
                "tts": "/v1/tts",
                "openai_speech": "/v1/audio/speech",
                "clone": "/v1/tts/clone",
                "voice": "/v1/voice",
                "voices": "/v1/voices",
            },
            "websocket_events": {
                "client": ["start", "text", "flush", "stop", "cancel", "ping"],
                "server": ["started", "audio", "log", "finish", "pong"],
            },
        }
    )
