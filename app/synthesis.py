"""One request in, one utterance out -- the request/response sibling of `Session`.

`Session` is built around a client that does not yet know what it wants to
say: text arrives in fragments, segments are cut and synthesized as they
become speakable, and the turn ends whenever the client says so. All of that
machinery (a segment queue, a background worker task, barge-in, turn state)
exists to serve incremental *input*.

An HTTP caller has the whole utterance up front. It still wants incremental
*output* -- the first bytes as soon as the first segment decodes, rather than
after the last -- but none of the rest. So this module keeps the part that
matters (segment -> engine -> encoder -> bytes, streamed) and drops the part
that does not, rather than contorting `Session` into a shape it was not
designed for.

The two paths share everything below the seam: the same `Segmenter`, the same
`StreamEncoder`s from `app.audio`, and the same single-GPU-worker engine, so
audio produced here is byte-identical to audio produced over the WebSocket
for the same text, voice and seed.

Error handling differs in one unavoidable way. Over a WebSocket a failure
mid-utterance is just a `finish(reason="error")` frame; over HTTP the status
line is long gone by then. `stream()` therefore lets `EngineError` propagate
and leaves the decision to the caller, which can turn a failure that happens
*before* the first byte into a real HTTP error status (see
`app/http/router.py`) and can only truncate one that happens after.
"""

from __future__ import annotations

import logging
import random
from contextlib import suppress
from typing import TYPE_CHECKING, AsyncIterator

from app.audio import StreamEncoder, resolve_format
from app.config import Settings
from app.engine.base import (
    CancelToken,
    EngineNotReady,
    SynthesisRequest,
    TTSEngine,
)
from app.segmenter import Segmenter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.protocol import TTSParams

logger = logging.getLogger("tts.synthesis")

#: What each wire format should be served as over HTTP. `pcm` is headerless
#: int16 with no registered media type; the sample rate travels in a response
#: header instead, since the bytes themselves carry no way to say it.
MEDIA_TYPES = {
    "wav": "audio/wav",
    "pcm": "application/octet-stream",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
}


def draw_seed() -> int:
    """A fresh 31-bit seed, pinned for one utterance so its voice is stable."""
    return random.getrandbits(31)


class OneShotSynthesis:
    """A prepared utterance: everything decided up front, bytes on demand.

    Construction is cheap and does no inference, but it does resolve the
    output format and pin the seed -- so a caller can write honest response
    headers (`Content-Type`, sample rate, the seed actually used) *before*
    committing to a response body it has not generated yet.
    """

    def __init__(
        self,
        engine: TTSEngine,
        settings: Settings,
        params: "TTSParams",
        text: str,
        request_id: str = "",
    ) -> None:
        if not engine.ready:
            raise EngineNotReady("engine has not finished loading")

        self.engine = engine
        self.settings = settings
        self.request_id = request_id
        self.sample_rate = engine.sample_rate

        from app.protocol import with_server_defaults

        params = with_server_defaults(
            params,
            chunk_length=settings.chunk_length,
            format=settings.default_format,
        )
        self.seed = params.seed if params.seed is not None else draw_seed()
        self.params: "TTSParams" = params.model_copy(update={"seed": self.seed})

        # Same cuts the WebSocket path would make, just all at once: the
        # segmenter is fed the whole text and then closed.
        segmenter = Segmenter(settings.min_segment_chars, settings.max_segment_chars)
        self.segments: list[str] = segmenter.push(text) + segmenter.close()

        fmt, encoder, warning = resolve_format(params.format, self.sample_rate)
        self.format: str = fmt
        self.warning: str | None = warning
        self._encoder: StreamEncoder | None = encoder
        self._cancel = CancelToken()

    @property
    def media_type(self) -> str:
        return MEDIA_TYPES.get(self.format, "application/octet-stream")

    def cancel(self) -> None:
        """Abandon in-flight generation. Safe from any task, any time."""
        self._cancel.cancel()

    async def stream(self) -> AsyncIterator[bytes]:
        """Yield the response body incrementally. Raises `EngineError` as-is.

        Exactly one encoder lifetime per instance, so this is single-use.
        Closing the generator early (a disconnected client, `anext` abandoned)
        cancels the engine job through the `finally` below rather than leaving
        the GPU working for nobody.
        """
        encoder = self._encoder
        if encoder is None:
            raise RuntimeError("OneShotSynthesis.stream() is single-use")
        self._encoder = None

        header_sent = False
        try:
            for segment in self.segments:
                request = SynthesisRequest(
                    text=segment, params=self.params, session_id=self.request_id
                )
                async for chunk in self.engine.synthesize(request, self._cancel):
                    if self._cancel.cancelled:
                        break
                    payload = b""
                    if not header_sent:
                        payload += encoder.header() or b""
                        header_sent = True
                    payload += encoder.encode(chunk)
                    if payload:
                        yield payload
                if self._cancel.cancelled:
                    break

            # An utterance that produced no audio at all (empty text, or a
            # cancel that landed before the first chunk) still owes the client
            # a well-formed container rather than zero bytes.
            tail = b"" if header_sent else (encoder.header() or b"")
            tail += encoder.finalize()
            if tail:
                yield tail
        finally:
            self._cancel.cancel()
            # Idempotent on every encoder; this is the abandoned-generator
            # path, where the normal finalize() above never ran and libsndfile
            # would otherwise keep its handle open.
            with suppress(Exception):
                encoder.finalize()
