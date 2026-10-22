"""WebSocket wire protocol.

Frames are MessagePack maps carrying an ``event`` discriminator, mirroring the
shape of the official Fish Audio streaming API so existing clients port over
with little more than a URL change. JSON text frames are accepted too, which
makes the server reachable from a browser console or `websocat` without a
msgpack codec.

Client -> server
    {"event": "start",  "request": {...}, "debug": false, "context_id": str|None}
    {"event": "text",   "text": "a piece of the utterance "}
    {"event": "flush"}                  # speak whatever is buffered, right now
    {"event": "stop"}                   # no more text; drain, then finish
    {"event": "cancel"}                 # barge-in; abandon audio immediately
    {"event": "ping"}

Server -> client
    {"event": "started", "session_id": str, "sample_rate": int,
     "format": str, "seed": int, "voice_id": str|None}
    {"event": "audio",  "audio": <bytes>, "time": <float ms of audio so far>}
    {"event": "log",    "message": str}
    {"event": "finish", "reason": "stop"|"cancel"|"error", "error": str|None}
    {"event": "pong"}

When the `start` frame carried a `context_id`, every `started`, `audio` and
`finish` frame of that turn carries it back, so an agent that has moved on to
a new turn can drop stale audio by tag instead of by timing. Frames of a turn
started without one are unchanged, byte for byte.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Annotated, Any, Literal

import ormsgpack
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.audio import SUPPORTED_SAMPLE_RATES

AUDIO_FORMATS = ("wav", "pcm", "mp3", "flac", "mulaw")

_REFERENCE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


class ProtocolError(ValueError):
    """Raised when a client frame cannot be understood. Closes the session."""


class ReferenceAudio(BaseModel):
    """A voice-cloning prompt: a short clip plus its exact transcript."""

    audio: bytes
    text: str

    @model_validator(mode="before")
    @classmethod
    def _decode_base64(cls, values: Any) -> Any:
        # JSON clients cannot send raw bytes, so accept base64 for `audio`.
        if isinstance(values, dict):
            audio = values.get("audio")
            if isinstance(audio, str):
                try:
                    values = {**values, "audio": base64.b64decode(audio, validate=True)}
                except Exception as exc:
                    raise ValueError("reference audio must be bytes or base64") from exc
        return values


class TTSParams(BaseModel):
    """Inference knobs. Bounds match what S2-Pro actually tolerates."""

    model_config = {"extra": "forbid"}

    text: str = ""
    language: str | None = None
    format: Literal["wav", "pcm", "mp3", "flac", "mulaw"] = "wav"
    # Output rate. None means the engine's native rate (44.1 kHz for S2-Pro),
    # which is also the cheapest: no conversion at all. 16000 or 24000 is what
    # most agent stacks consume, 8000 with `mulaw` is telephony.
    sample_rate: int | None = None

    references: list[ReferenceAudio] = Field(default_factory=list)
    reference_id: str | None = None
    # When this frame names no voice of its own, fall back to whatever
    # `POST /v1/voice` last set. Opt out to get the model's own default voice
    # without having to clear the server-wide one.
    use_active_voice: bool = True

    seed: int | None = None
    normalize: bool = True
    # These use Field, not conint. `Annotated[int, conint(...)]` looks like it
    # constrains the value and does nothing at all under pydantic v2 - conint()
    # builds a *type*, so as Annotated metadata it is silently ignored. Every
    # bound below was unenforced: `max_new_tokens=-1` produced an empty audio
    # stream and a clean `finish`, and a negative `chunk_length` reached the
    # fork's byte splitter.
    chunk_length: Annotated[int, Field(ge=100, le=300)] = 200
    # 0 means "as many frames as the context still has room for", which is
    # what `generate_long` meant by it too. It is the default because the old
    # 1024 was an invisible ceiling: a segment that reached it stopped
    # mid-word and looked from the outside exactly like one that had finished.
    # The context is the real bound, and `/status` counts it if it ever binds
    # (`prefix_cache.truncations`).
    max_new_tokens: Annotated[int, Field(ge=0, le=8192)] = 0
    top_p: Annotated[float, Field(ge=0.1, le=1.0)] = 0.8
    top_k: Annotated[int, Field(ge=1, le=1000)] = 30
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0)] = 1.1
    temperature: Annotated[float, Field(ge=0.1, le=1.0)] = 0.8

    @model_validator(mode="after")
    def _check_sample_rate(self) -> "TTSParams":
        if self.sample_rate is not None and self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {list(SUPPORTED_SAMPLE_RATES)}, got {self.sample_rate}"
            )
        return self

    @model_validator(mode="after")
    def _normalise_reference_id(self) -> "TTSParams":
        rid = (self.reference_id or "").strip()
        # It names a directory under references/, so it must not be able to
        # name anything else: "../.." would otherwise walk out of it.
        if rid and not _REFERENCE_ID.fullmatch(rid):
            raise ValueError(
                "reference_id must be 1-64 letters, digits, '_' or '-', "
                f"got {rid[:40]!r}"
            )
        object.__setattr__(self, "reference_id", rid or None)
        return self


class StartEvent(BaseModel):
    event: Literal["start"] = "start"
    request: TTSParams = Field(default_factory=TTSParams)
    debug: bool = False
    # Opaque tag echoed on this turn's server frames. See the module docstring.
    context_id: Annotated[str, Field(max_length=128)] | None = None


class TextEvent(BaseModel):
    event: Literal["text"] = "text"
    text: str = ""


class FlushEvent(BaseModel):
    event: Literal["flush"] = "flush"


class StopEvent(BaseModel):
    event: Literal["stop"] = "stop"


class CancelEvent(BaseModel):
    event: Literal["cancel"] = "cancel"


class PingEvent(BaseModel):
    event: Literal["ping"] = "ping"


ClientEvent = StartEvent | TextEvent | FlushEvent | StopEvent | CancelEvent | PingEvent

_CLIENT_EVENTS: dict[str, type[BaseModel]] = {
    "start": StartEvent,
    "text": TextEvent,
    "flush": FlushEvent,
    "stop": StopEvent,
    "cancel": CancelEvent,
    "ping": PingEvent,
}


def with_server_defaults(params: TTSParams, **defaults) -> TTSParams:
    """Fill in only the fields the client left alone.

    `TTSParams` carries its own defaults, which makes a server-wide default
    impossible to express: once validation runs there is no way to tell
    "chunk_length: 200 because the operator configured it" from "200 because
    nobody said". `model_fields_set` is that distinction - it holds exactly the
    keys present in the inbound frame - so a request that names a field keeps
    it and one that does not inherits the server's.

    This is what makes `TTS_CHUNK_LENGTH` and `TTS_DEFAULT_FORMAT` mean
    something. Both used to be settings nothing read.
    """
    update = {
        key: value
        for key, value in defaults.items()
        if value is not None and key not in params.model_fields_set
    }
    return params.model_copy(update=update) if update else params


def decode_client_frame(raw: bytes | str) -> ClientEvent:
    """Turn one inbound frame into a typed event, or raise ProtocolError."""
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"frame is not valid JSON: {exc}") from exc
    else:
        try:
            payload = ormsgpack.unpackb(raw)
        except Exception:
            # Some clients send msgpack-shaped JSON over binary frames.
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                raise ProtocolError("frame is neither msgpack nor JSON") from exc

    if not isinstance(payload, dict):
        raise ProtocolError("frame must be a map")

    name = payload.get("event")
    if not isinstance(name, str):
        raise ProtocolError("frame is missing a string 'event' field")

    model = _CLIENT_EVENTS.get(name)
    if model is None:
        raise ProtocolError(
            f"unknown event {name!r}; expected one of {sorted(_CLIENT_EVENTS)}"
        )

    try:
        return model.model_validate(payload)  # type: ignore[return-value]
    except ValidationError as exc:
        raise ProtocolError(f"invalid {name!r} frame: {exc.errors(include_url=False)}") from exc


def _pack(payload: dict) -> bytes:
    return ormsgpack.packb(payload)


def _tagged(payload: dict, context_id: str | None) -> dict:
    if context_id is not None:
        payload["context_id"] = context_id
    return payload


def started_frame(
    session_id: str,
    sample_rate: int,
    fmt: str,
    seed: int,
    voice_id: str | None = None,
    context_id: str | None = None,
) -> bytes:
    """`voice_id` answers "whose voice am I about to hear?" without guesswork.

    A client that sends no `references` cannot otherwise tell whether it got
    the server's active voice, a stale one, or the model's default -- and the
    difference is audible rather than an error. `None` means no cloning.
    """
    return _pack(
        _tagged(
            {
                "event": "started",
                "session_id": session_id,
                "sample_rate": sample_rate,
                "format": fmt,
                "seed": seed,
                "voice_id": voice_id,
            },
            context_id,
        )
    )


def audio_frame(audio: bytes, elapsed_ms: float, context_id: str | None = None) -> bytes:
    return _pack(_tagged({"event": "audio", "audio": audio, "time": elapsed_ms}, context_id))


def log_frame(message: str) -> bytes:
    return _pack({"event": "log", "message": message})


def finish_frame(reason: str, error: str | None = None, context_id: str | None = None) -> bytes:
    return _pack(_tagged({"event": "finish", "reason": reason, "error": error}, context_id))


def pong_frame() -> bytes:
    return _pack({"event": "pong"})
