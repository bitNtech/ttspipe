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

#: Output rates a client may ask for. The engine produces its codec's native
#: rate (44.1 kHz for S2-Pro); anything else is converted on the way out by
#: `ResamplingEncoder`. Voice-agent stacks overwhelmingly want 16 or 24 kHz,
#: and telephony wants 8 kHz.
SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)


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


def float_to_mulaw(samples: np.ndarray) -> bytes:
    """float32 mono in [-1, 1] -> G.711 mu-law bytes, one per sample.

    The segment-and-mantissa form of ITU-T G.711, on 14-bit magnitudes, so the
    bytes match what a telephony stack (Twilio, Asterisk, `audioop.lin2ulaw`)
    produces from the same int16 samples.
    """
    if samples.size == 0:
        return b""
    clamped = np.clip(np.nan_to_num(samples, copy=True), -1.0, 1.0)
    pcm = np.round(clamped * 32767.0).astype(np.int32) >> 2  # 14-bit, arithmetic
    mask = np.where(pcm < 0, 0x7F, 0xFF)
    magnitude = np.minimum(np.abs(pcm), 8159) + 33  # clip, then bias
    segment = np.maximum(np.floor(np.log2(magnitude)).astype(np.int32) - 5, 0)
    ulaw = (segment << 4) | ((magnitude >> (segment + 1)) & 0x0F)
    ulaw = np.where(segment > 7, 0x7F, ulaw) ^ mask
    return ulaw.astype(np.uint8).tobytes()


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


class MulawEncoder(StreamEncoder):
    """Headerless G.711 mu-law, 8 bits a sample. What phone networks carry.

    Meant to be paired with `sample_rate: 8000`; at any other rate it is still
    valid mu-law, just not something a telephony provider will accept.
    """

    def header(self) -> bytes | None:
        return None

    def encode(self, chunk: "AudioChunk") -> bytes:
        return float_to_mulaw(chunk.samples)

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


class ResamplingEncoder(StreamEncoder):
    """Convert each chunk to `out_rate`, then hand it to the real encoder.

    Uses soxr's streaming resampler, which carries its filter state from one
    chunk to the next, so chunk boundaries are seamless - resampling each chunk
    on its own would put a click at every one. The price is the filter's group
    delay: a few milliseconds of every chunk's tail are held back and leave
    with the next one, and `finalize()` flushes the last of them. That delay
    never accumulates; it is the same few milliseconds for the whole stream.
    """

    def __init__(self, inner: StreamEncoder, in_rate: int, out_rate: int) -> None:
        import soxr

        self._inner = inner
        self._in_rate = in_rate
        self._out_rate = out_rate
        self._stream = soxr.ResampleStream(in_rate, out_rate, 1, dtype="float32", quality="HQ")
        self._done = False

    def _chunk(self, samples: np.ndarray) -> "AudioChunk":
        from app.engine.base import AudioChunk

        return AudioChunk(samples=samples, sample_rate=self._out_rate)

    def header(self) -> bytes | None:
        return self._inner.header()

    def encode(self, chunk: "AudioChunk") -> bytes:
        if chunk.samples.size == 0:
            return b""
        samples = np.ascontiguousarray(chunk.samples, dtype=np.float32)
        return self._inner.encode(self._chunk(self._stream.resample_chunk(samples)))

    def finalize(self) -> bytes:
        if self._done:
            return self._inner.finalize()
        self._done = True
        tail = self._stream.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
        out = self._inner.encode(self._chunk(tail)) if tail.size else b""
        return out + self._inner.finalize()


def create_encoder(
    fmt: str, sample_rate: int, channels: int = 1, out_rate: int | None = None
) -> StreamEncoder:
    """Build a `StreamEncoder` for `fmt`, or raise `UnsupportedFormatError`.

    `sample_rate` is what the engine produces; `out_rate`, when given and
    different, is what the client receives.
    """
    rate = out_rate or sample_rate
    if out_rate and out_rate not in SUPPORTED_SAMPLE_RATES:
        raise UnsupportedFormatError(
            f"sample rate {out_rate} is not one of {list(SUPPORTED_SAMPLE_RATES)}"
        )
    if fmt == "pcm":
        inner: StreamEncoder = PCMEncoder()
    elif fmt == "mulaw":
        inner = MulawEncoder()
    elif fmt == "wav":
        inner = WavEncoder(rate, channels=channels)
    elif fmt in _SF_FORMAT:
        inner = _SoundfileEncoder(fmt, rate, channels=channels)
    else:
        raise UnsupportedFormatError(f"unknown audio format {fmt!r}")
    if rate != sample_rate:
        return ResamplingEncoder(inner, sample_rate, rate)
    return inner


def resolve_format(
    fmt: str, sample_rate: int, channels: int = 1, out_rate: int | None = None
) -> tuple[str, StreamEncoder, str | None]:
    """Build an encoder for `fmt`, degrading to wav (with a reason) if unavailable.

    Returns (actual_format, encoder, warning_or_None). Callers that already
    told the client which format they'd get (the `started` frame) must call
    this before sending it, so the reply is honest. The output rate is never
    degraded: `TTSParams` only admits rates `create_encoder` accepts.
    """
    try:
        return fmt, create_encoder(fmt, sample_rate, channels, out_rate), None
    except UnsupportedFormatError as exc:
        warning = f"format {fmt!r} unavailable ({exc}); falling back to wav"
        return "wav", create_encoder("wav", sample_rate, channels, out_rate), warning
