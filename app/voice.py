"""The active cloning voice: set once over HTTP, used by every WebSocket turn.

`POST /v1/tts/clone` clones the clip attached to that one request, which is
right for "say this, in this voice, once". It is the wrong shape for a
conversational client, which wants to pick a voice at the start of a session
and then just stream text at the server for as long as it likes.

So the server holds exactly one *active* voice. Set it with `POST /v1/voice`;
every `start` frame that does not name a voice of its own is then conditioned
on it, until it is replaced or cleared. One voice, not a table of them, because
"which voice am I hearing?" should have exactly one answer per server and
because the per-session alternative already exists: a client that wants its own
voice sends `references` on its own `start` frame, and that always wins.

It is written through to disk under `references/_active/` so a restart does not
silently drop back to the default voice -- which, given the failure mode is
"the wrong voice comes out" rather than an error, is worth the two file writes.
That location is deliberate: it is the same layout `reference_id` reads, so the
active voice is simultaneously addressable as `reference_id: "_active"` with no
extra machinery.

Storage is deliberately tiny and synchronous. The clip is a few hundred KB, it
is written once per voice change (not per request), and every reader is on the
event loop, so a lock and a file write cost far less than the coordination an
async store would need.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("tts.voice")

#: The reference_id the active voice is simultaneously reachable under.
ACTIVE_VOICE_ID = "_active"

_META_FILE = "meta.json"
_CLIP_STEM = "clip"
_DEFAULT_SUFFIX = ".wav"

#: Clients are unreliable about the media type of an audio upload -- `curl -F`
#: labels a .wav `application/octet-stream`, and a JSON caller sends no type at
#: all -- but `/v1/voice/audio` needs a real one or a browser `<audio>` element
#: refuses to play the clip back. So the extension wins whenever it is
#: recognised, and the client's guess is only a fallback.
_MEDIA_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".aac": "audio/aac",
    ".webm": "audio/webm",
}

#: Mirrors app/engine/references.py: prompt tokens scale with clip length and
#: are replayed on every request that uses the voice, so a long clip is a
#: permanent latency tax rather than a one-off cost.
_WARN_CLIP_SECONDS = 20.0


@dataclass(frozen=True, slots=True)
class Voice:
    """One reference clip and its exact transcript, plus what the UI shows."""

    voice_id: str
    audio: bytes
    text: str
    filename: str
    content_type: str
    created_at: float
    duration_s: float | None = None

    def as_dict(self) -> dict:
        """Metadata only -- never the audio bytes, which are not JSON-safe."""
        return {
            "voice_id": self.voice_id,
            "text": self.text,
            "filename": self.filename,
            "content_type": self.content_type,
            "created_at": self.created_at,
            "duration_s": self.duration_s,
            "size_bytes": len(self.audio),
            "reference_id": ACTIVE_VOICE_ID,
        }


def _probe_duration(audio: bytes) -> float | None:
    """Clip length in seconds, or None if we cannot tell cheaply.

    Best-effort by design: this is for the operator's benefit (a warning, a
    number in the UI), never for correctness, so an unreadable container is
    not a reason to refuse a voice the engine may well handle.
    """
    try:
        import soundfile as sf

        return float(sf.info(io.BytesIO(audio)).duration)
    except Exception:
        return None


class VoiceStore:
    """Holds the active voice. Thread-safe, and persistent across restarts."""

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)
        self._lock = threading.Lock()
        self._voice: Voice | None = None
        self._load()

    # --- public API ---

    def get(self) -> Voice | None:
        with self._lock:
            return self._voice

    def set(self, audio: bytes, text: str, filename: str = "", content_type: str = "") -> Voice:
        """Replace the active voice. The transcript must be the clip's exact words."""
        if not audio:
            raise ValueError("reference audio is empty")
        if not text.strip():
            raise ValueError("reference text is empty")

        suffix = Path(filename or "").suffix.lower() or _DEFAULT_SUFFIX
        media_type = _MEDIA_TYPES.get(suffix) or content_type or "application/octet-stream"
        duration = _probe_duration(audio)
        if duration is not None and duration > _WARN_CLIP_SECONDS:
            logger.warning(
                "active voice clip is %.1fs; its prompt tokens are replayed on "
                "every request that uses it, so ~10s clips stay much faster",
                duration,
            )

        voice = Voice(
            voice_id=hashlib.sha256(audio).hexdigest()[:16],
            audio=audio,
            text=text.strip(),
            filename=filename or f"{_CLIP_STEM}{suffix}",
            content_type=media_type,
            created_at=time.time(),
            duration_s=duration,
        )

        with self._lock:
            self._voice = voice
            self._persist(voice, suffix)
        logger.info(
            "active voice set: %s (%s, %s)",
            voice.voice_id,
            voice.filename,
            f"{duration:.1f}s" if duration is not None else "duration unknown",
        )
        return voice

    def clear(self) -> bool:
        """Drop the active voice. Returns whether there was one. Idempotent."""
        with self._lock:
            had = self._voice is not None
            self._voice = None
            if self._dir.exists():
                shutil.rmtree(self._dir, ignore_errors=True)
        if had:
            logger.info("active voice cleared")
        return had

    # --- persistence ---

    def _persist(self, voice: Voice, suffix: str) -> None:
        """Write `clip<ext>` + `clip.lab` + meta.json, replacing any previous voice.

        The on-disk stem is always `clip`, whatever the upload was called,
        because `reference_id` pairs audio with its transcript by stem;
        `voice.filename` keeps the original name for display. A failure here
        must not lose the voice.

        The in-memory voice is already live at this point, so a read-only or
        full disk degrades to "works until restart" rather than rejecting a
        request the server could otherwise serve.
        """
        try:
            if self._dir.exists():
                shutil.rmtree(self._dir, ignore_errors=True)
            self._dir.mkdir(parents=True, exist_ok=True)

            clip_path = self._dir / f"{_CLIP_STEM}{suffix}"
            clip_path.write_bytes(voice.audio)
            # The .lab sibling is what app/engine/references.py reads when the
            # same clip is addressed as reference_id="_active".
            clip_path.with_suffix(".lab").write_text(voice.text, encoding="utf-8")
            (self._dir / _META_FILE).write_text(
                json.dumps({**voice.as_dict(), "audio_file": clip_path.name}, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("could not persist the active voice to %s: %s", self._dir, exc)

    def _load(self) -> None:
        meta_path = self._dir / _META_FILE
        if not meta_path.is_file():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            audio = (self._dir / meta["audio_file"]).read_bytes()
            self._voice = Voice(
                voice_id=meta.get("voice_id") or hashlib.sha256(audio).hexdigest()[:16],
                audio=audio,
                text=meta.get("text", ""),
                filename=meta.get("filename", meta["audio_file"]),
                content_type=meta.get("content_type", "application/octet-stream"),
                created_at=float(meta.get("created_at", 0.0)),
                duration_s=meta.get("duration_s"),
            )
            logger.info("restored active voice %s from %s", self._voice.voice_id, self._dir)
        except (OSError, KeyError, ValueError) as exc:
            # A half-written or hand-edited directory should not stop the
            # server booting; it just means there is no active voice.
            logger.warning("ignoring unreadable active voice in %s: %s", self._dir, exc)
            self._voice = None


async def warm_active_voice(app_state, voice=None) -> int:
    """Prefill the active voice's prompt into the engine's KV cache.

    Setting a voice used to be pure bookkeeping, with the whole cost of the
    reference clip - the codec encode *and* a ~220-token prompt prefill -
    landing on the first turn that spoke in it. On a 4 GB card that first turn
    measured 17s to first audio against 0.9s for every one after it. Paying it
    here instead means the user-visible cost of changing voice is the change
    itself, not the next thing they say.

    Best-effort and fire-and-forget by design: if the engine is not up yet, or
    the warm-up fails, the next synthesis simply prefills as it always would.
    """
    import logging

    from app.protocol import ReferenceAudio, TTSParams

    logger = logging.getLogger("tts.voice")
    engine = getattr(app_state, "engine", None)
    if engine is None or not engine.ready:
        return 0

    store = getattr(app_state, "voices", None)
    voice = voice or (store.get() if store is not None else None)
    if voice is None:
        return 0

    try:
        params = TTSParams(references=[ReferenceAudio(audio=voice.audio, text=voice.text)])
        return await engine.warm_voice(params)
    except Exception:  # noqa: BLE001 - never let a warm-up break the request that triggered it
        logger.warning("could not warm the active voice prefix", exc_info=True)
        return 0
