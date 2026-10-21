"""Turn a stream of `AudioChunk`s into bytes on the wire, one chunk at a time.

Nothing here ever buffers a whole utterance: each `encode()` call is handed
exactly one decoded segment and must return only the bytes that segment adds.
That is what keeps time-to-first-audio low and memory flat regardless of how
long the client lets the session run.

`pcm` and `wav` are hand-rolled because the format is trivial and we want
zero surprises. `mp3` and `flac` are delegated to `soundfile` (libsndfile),
which happily accepts incremental `write()` calls against a non-seekable
buffer -- with one catch: on `close()` it patches a handful of bytes at the
*start* of the stream in place (FLAC rewrites the STREAMINFO block's
min/max-blocksize and total-sample-count fields; MP3 fills in a VBR "Xing"
tag in the first frame). Those patches land behind data we may have already
put on the wire, so we cannot deliver them.

We treat this exactly like the WAV encoder treats an unknown length: ship a
header with the "I don't know yet" values libsndfile would otherwise fill in
later, and never touch it again.

  - FLAC: `total_samples = 0` in STREAMINFO is spec-legal for "unknown", and
    `min/max_blocksize`/`min/max_framesize = 0` mean "not constant, see each
    frame header" -- both true for a live stream.
  - MP3: a missing Xing tag just means a VBR-unaware decoder can't seek or
    show a duration upfront; every audio frame after it decodes normally.

This was verified empirically (see tests/test_audio.py): once real audio
frames have been appended, `close()` only ever rewrites bytes inside the
fixed-size header region -- everything after that offset is a pure,
never-revised append. If `soundfile` is missing, or the installed libsndfile
build lacks an encoder for the requested format, `create_encoder` raises
`UnsupportedFormatError` instead of producing a silently-broken stream;
`resolve_format` is the graceful-degradation wrapper session.py uses.
"""

from __future__ import annotations

import io
import struct
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.engine.base import AudioChunk

try:
    import soundfile as _sf
except ImportError:  # pragma: no cover - exercised via UnsupportedFormatError tests
    _sf = None

_SF_FORMAT = {"mp3": "MP3", "flac": "FLAC"}


class UnsupportedFormatError(RuntimeError):
    """The requested wire format has no usable streaming encoder here."""


def float_to_pcm16(samples: np.ndarray) -> bytes:
    """float32 mono in [-1, 1] -> little-endian int16 bytes. Clamps, never wraps."""
    if samples.size == 0:
        return b""
    # NaN would cast to an unpredictable int16; map it to silence rather than
    # noise. The engine raises on non-finite output, so this is belt-and-braces.
    clamped = np.clip(np.nan_to_num(samples, copy=True), -1.0, 1.0)
    ints = np.round(clamped * 32767.0).astype("<i2", copy=False)
    return ints.tobytes()


class StreamEncoder:
    """One encoder instance per utterance. Call order: header(), encode()*, finalize()."""

    def header(self) -> bytes | None:
        """Bytes to send before any audio, or None if the format has none."""
        raise NotImplementedError

    def encode(self, chunk: "AudioChunk") -> bytes:
        """Bytes this segment adds to the stream."""
        raise NotImplementedError

    def finalize(self) -> bytes:
        """Bytes to send once, after the last chunk. May be empty."""
        raise NotImplementedError


class PCMEncoder(StreamEncoder):
    """Headerless raw int16 samples. The simplest possible thing to play back."""

    def header(self) -> bytes | None:
        return None

    def encode(self, chunk: "AudioChunk") -> bytes:
        return float_to_pcm16(chunk.samples)

    def finalize(self) -> bytes:
        return b""


class WavEncoder(StreamEncoder):
    """RIFF/WAVE with the standard "streaming, size unknown" placeholders.

    Every field a real WAV needs to know the file size up front (the RIFF
    chunk size, the data subchunk size) is set to 0xFFFFFFFF, which players
    that support live/piped WAV (ffplay, vlc, most browsers) accept as "read
    until EOF" rather than a fixed byte count.
    """

    def __init__(self, sample_rate: int, channels: int = 1, bits_per_sample: int = 16):
        self.sample_rate = sample_rate
        self.channels = channels
        self.bits_per_sample = bits_per_sample

    def header(self) -> bytes:
        block_align = self.channels * self.bits_per_sample // 8
        byte_rate = self.sample_rate * block_align
        unknown = 0xFFFFFFFF
        fmt_chunk = struct.pack(
            "<IHHIIHH",
            16,  # fmt chunk size
            1,  # PCM
            self.channels,
            self.sample_rate,
            byte_rate,
            block_align,
            self.bits_per_sample,
        )
        return b"RIFF" + struct.pack("<I", unknown) + b"WAVE" + b"fmt " + fmt_chunk + b"data" + struct.pack("<I", unknown)

    def encode(self, chunk: "AudioChunk") -> bytes:
        return float_to_pcm16(chunk.samples)

    def finalize(self) -> bytes:
        return b""


class _SoundfileEncoder(StreamEncoder):
    """Incremental mp3/flac via libsndfile. See module docstring for the caveat."""

    def __init__(self, fmt: str, sample_rate: int, channels: int = 1):
        if _sf is None:
            raise UnsupportedFormatError(f"{fmt}: 'soundfile' is not installed")
        sf_format = _SF_FORMAT.get(fmt)
        if sf_format is None:
            raise UnsupportedFormatError(f"{fmt}: not a soundfile-backed format")
        try:
            available = sf_format in _sf.available_formats()
        except Exception as exc:  # pragma: no cover - defensive
            raise UnsupportedFormatError(f"{fmt}: could not query libsndfile: {exc}") from exc
        if not available:
            raise UnsupportedFormatError(f"{fmt}: not supported by the installed libsndfile")

        self._buf = io.BytesIO()
        try:
            self._sf = _sf.SoundFile(
                self._buf, mode="w", samplerate=sample_rate, channels=channels, format=sf_format
            )
        except Exception as exc:
            raise UnsupportedFormatError(f"{fmt}: {exc}") from exc
        self._read_pos = 0
        self._closed = False

    def header(self) -> bytes | None:
        # libsndfile doesn't emit any bytes until the first write() call.
        return None

    def _drain(self) -> bytes:
        self._buf.seek(self._read_pos)
        tail = self._buf.read()
        self._read_pos += len(tail)
        return tail

    def encode(self, chunk: "AudioChunk") -> bytes:
        if chunk.samples.size:
            self._sf.write(chunk.samples)
            self._sf.flush()
        return self._drain()

    def finalize(self) -> bytes:
        if not self._closed:
            self._sf.close()
            self._closed = True
        # close() may patch header bytes behind self._read_pos in place; we
        # never re-read that span, so draining here only ever yields the
        # genuinely new trailing bytes flushed at close.
        return self._drain()


def create_encoder(fmt: str, sample_rate: int, channels: int = 1) -> StreamEncoder:
    """Build a `StreamEncoder` for `fmt`, or raise `UnsupportedFormatError`."""
    if fmt == "pcm":
        return PCMEncoder()
    if fmt == "wav":
        return WavEncoder(sample_rate, channels=channels)
    if fmt in _SF_FORMAT:
        return _SoundfileEncoder(fmt, sample_rate, channels=channels)
    raise UnsupportedFormatError(f"unknown audio format {fmt!r}")


def resolve_format(
    fmt: str, sample_rate: int, channels: int = 1
) -> tuple[str, StreamEncoder, str | None]:
    """Build an encoder for `fmt`, degrading to wav (with a reason) if unavailable.

    Returns (actual_format, encoder, warning_or_None). Callers that already
    told the client which format they'd get (the `started` frame) must call
    this before sending it, so the reply is honest.
    """
    try:
        return fmt, create_encoder(fmt, sample_rate, channels=channels), None
    except UnsupportedFormatError as exc:
        warning = f"format {fmt!r} unavailable ({exc}); falling back to wav"
        return "wav", create_encoder("wav", sample_rate, channels=channels), warning
