"""Coverage for the streaming audio encoders in `app.audio`.

Confirms the wav/pcm path is byte-exact and never buffers, that mp3/flac
round-trip through a real decoder (soundfile), and that the header-patch
behaviour the module docstring describes (why it's safe to stream those
formats at all) is what libsndfile actually does here.
"""

from __future__ import annotations

import io
import struct
import wave

import numpy as np
import pytest
import soundfile as sf

from app.audio import (
    PCMEncoder,
    UnsupportedFormatError,
    WavEncoder,
    create_encoder,
    float_to_pcm16,
    resolve_format,
)
from app.engine.base import AudioChunk

SAMPLE_RATE = 24000


def _chunk(samples: np.ndarray) -> AudioChunk:
    return AudioChunk(samples=samples.astype(np.float32), sample_rate=SAMPLE_RATE)


def _tone(n: int, freq: float = 220.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _read_all(buf: io.BytesIO) -> tuple[np.ndarray, int]:
    """Read a whole streaming file that declares an unknown length.

    `soundfile.read()` trusts the frame count in the header and, for our
    "unknown length" FLAC/WAV streams, that count is a sentinel (0 samples
    known -> soundfile treats it as unbounded and tries to preallocate an
    enormous array). A real streaming player reads in a loop until EOF
    instead of trusting the header's count; this helper does the same.
    """
    with sf.SoundFile(buf, mode="r") as f:
        blocks = []
        while True:
            try:
                block = f.read(512, dtype="float32")
            except sf.LibsndfileError:
                # libsndfile's own EOF bookkeeping for an unknown-length
                # stream can trip an internal seek past end-of-buffer on the
                # final short read; treat that the same as a clean EOF.
                break
            if len(block) == 0:
                break
            blocks.append(block)
        rate = f.samplerate
    return (np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)), rate


# -- float -> int16 conversion --


def test_float_to_pcm16_maps_full_range_without_wraparound():
    samples = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    ints = np.frombuffer(float_to_pcm16(samples), dtype="<i2")
    assert ints[0] == -32767
    assert ints[1] == 0
    assert ints[2] == 32767


def test_float_to_pcm16_clamps_out_of_range_instead_of_wrapping():
    samples = np.array([-5.0, 5.0], dtype=np.float32)
    ints = np.frombuffer(float_to_pcm16(samples), dtype="<i2")
    # a naive cast without clamping would wrap around to garbage/opposite sign
    assert ints[0] == -32767
    assert ints[1] == 32767
    assert ints[0] < 0 and ints[1] > 0


def test_float_to_pcm16_empty_is_empty():
    assert float_to_pcm16(np.zeros(0, dtype=np.float32)) == b""


# -- pcm --


def test_pcm_encoder_headerless_and_correct_length():
    enc = PCMEncoder()
    assert enc.header() is None
    samples = _tone(100)
    data = enc.encode(_chunk(samples))
    assert len(data) == 100 * 2  # int16 = 2 bytes/sample
    assert enc.finalize() == b""


# -- wav --


def test_wav_header_has_streaming_unknown_length_placeholders():
    enc = WavEncoder(sample_rate=SAMPLE_RATE)
    header = enc.header()
    assert header[:4] == b"RIFF"
    riff_size = struct.unpack("<I", header[4:8])[0]
    assert riff_size == 0xFFFFFFFF
    assert header[8:12] == b"WAVE"
    assert header[12:16] == b"fmt "
    data_size = struct.unpack("<I", header[-4:])[0]
    assert data_size == 0xFFFFFFFF
    assert header[-8:-4] == b"data"


def test_wav_roundtrip_is_readable_with_correct_sample_count_and_values():
    enc = WavEncoder(sample_rate=SAMPLE_RATE)
    samples1 = _tone(240, freq=220.0)
    samples2 = _tone(360, freq=440.0)

    out = io.BytesIO()
    out.write(enc.header())
    out.write(enc.encode(_chunk(samples1)))
    out.write(enc.encode(_chunk(samples2)))
    out.write(enc.finalize())
    out.seek(0)

    # streaming WAV with 0xFFFFFFFF placeholders is exactly what a live
    # decoder is expected to accept; soundfile (libsndfile) reads it fine.
    read_samples, read_rate = sf.read(out, dtype="float32")
    assert read_rate == SAMPLE_RATE
    assert len(read_samples) == 240 + 360

    expected = np.concatenate([samples1, samples2])
    expected_int16 = np.frombuffer(float_to_pcm16(expected), dtype="<i2").astype(np.float32) / 32767.0
    got_int16 = np.round(read_samples * 32767.0) / 32767.0
    assert np.allclose(got_int16, expected_int16, atol=1e-4)


def test_wav_never_buffers_whole_utterance_bytes_are_incremental():
    enc = WavEncoder(sample_rate=SAMPLE_RATE)
    chunk_bytes = enc.encode(_chunk(_tone(1000)))
    # exactly this chunk's bytes, nothing more, nothing accumulated internally
    assert len(chunk_bytes) == 1000 * 2
    chunk_bytes_2 = enc.encode(_chunk(_tone(500)))
    assert len(chunk_bytes_2) == 500 * 2


def test_stdlib_wave_module_can_at_least_open_the_stream():
    # a stricter, independent reader than soundfile: confirms the header is
    # well-formed WAV, not just something libsndfile happens to tolerate.
    enc = WavEncoder(sample_rate=SAMPLE_RATE)
    out = io.BytesIO()
    out.write(enc.header())
    out.write(enc.encode(_chunk(_tone(480))))
    out.write(enc.finalize())
    out.seek(0)
    with wave.open(out, "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE
        assert wf.getsampwidth() == 2
        assert wf.getnchannels() == 1


# -- mp3 / flac (soundfile-backed, incremental) --


@pytest.mark.parametrize("fmt", ["flac", "mp3"])
def test_compressed_formats_stream_incrementally_and_decode(fmt):
    enc = create_encoder(fmt, SAMPLE_RATE)
    assert enc.header() is None  # libsndfile emits nothing before the first write

    out = io.BytesIO()
    sizes = []
    for _ in range(4):
        out.write(enc.encode(_chunk(_tone(2400))))  # 100ms bursts
        sizes.append(out.tell())
    out.write(enc.finalize())
    total_size = out.tell()

    # genuinely incremental: each call added bytes, not just at the end
    assert all(b > 0 for b in sizes)
    assert sizes == sorted(sizes)
    assert total_size >= sizes[-1]

    out.seek(0)
    read_samples, read_rate = _read_all(out)
    assert read_rate == SAMPLE_RATE
    # Lossless FLAC round-trips almost exactly. Lossy MP3 has real encoder
    # delay/padding, and without a Xing tag (deliberately not patched in --
    # see the module docstring) a decoder can't trim it precisely, so it
    # needs a much more generous tolerance.
    slack = 1024 if fmt == "flac" else 8192
    assert abs(len(read_samples) - 4 * 2400) < slack


@pytest.mark.parametrize("fmt", ["flac", "mp3"])
def test_compressed_format_header_patch_is_confined_and_never_redelivered(fmt):
    """The empirical claim in app.audio's module docstring, checked directly.

    libsndfile patches a handful of header bytes in place at close() (FLAC's
    STREAMINFO sample count, MP3's Xing tag). We already handed those exact
    byte positions to a caller as part of an earlier `.encode()` call, with
    placeholder ("unknown") values -- so the encoder is only safe to stream
    if (a) that patch stays confined to the small fixed-size header region,
    and (b) we never re-drain that region, i.e. the caller keeps whatever
    placeholder values it already has, same as the WAV encoder's 0xFFFFFFFF.
    """
    enc = create_encoder(fmt, SAMPLE_RATE)
    delivered_before_close = b""
    for _ in range(3):
        delivered_before_close += enc.encode(_chunk(_tone(2400)))
    already_delivered_len = len(delivered_before_close)

    tail = enc.finalize()
    final_buffer = enc._buf.getvalue()  # noqa: SLF001 -- whitebox check by design

    # (b): finalize()'s return value is exactly whatever is new since the
    # last encode() -- it never re-hands out bytes from [0, already_delivered_len).
    assert tail == final_buffer[already_delivered_len:]

    # (a): the patch is real -- close() rewrites a few early bytes in place
    # (this is *why* streaming these formats is only safe with placeholder
    # header values) -- but it stays confined to a small header prefix; the
    # already-delivered bytes past that point are untouched.
    header_region = 256
    assert delivered_before_close[:header_region] != final_buffer[:header_region]
    assert (
        delivered_before_close[header_region:already_delivered_len]
        == final_buffer[header_region:already_delivered_len]
    )


def test_create_encoder_unknown_format_raises():
    with pytest.raises(UnsupportedFormatError):
        create_encoder("ogg", SAMPLE_RATE)


def test_resolve_format_passes_through_supported_format():
    fmt, enc, warning = resolve_format("wav", SAMPLE_RATE)
    assert fmt == "wav"
    assert warning is None
    assert isinstance(enc, WavEncoder)


def test_resolve_format_degrades_to_wav_and_reports_when_unsupported():
    fmt, enc, warning = resolve_format("ogg", SAMPLE_RATE)
    assert fmt == "wav"
    assert isinstance(enc, WavEncoder)
    assert warning is not None
    assert "ogg" in warning
