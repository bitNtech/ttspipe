"""Audio helpers: stateful low-delay resampler, PCM16, WAV, levels (ARCHITECTURE.md §3)."""
from __future__ import annotations

import io
import math
import wave

import numpy as np

OUT_RATES = (16000, 22050, 24000, 48000)


class StreamResampler:
    """Kaiser-windowed sinc resampler with carried state.

    Delay is `half` input samples (16 taps per side at the lower rate, < 1 ms). Output is aligned with
    the input (no added group delay) and frame-exact: after flush() the total output length equals
    round(total_in * out_rate / in_rate).
    """

    def __init__(self, in_rate: int, out_rate: int, taps_per_side: int = 16, beta: float = 8.0):
        self.in_rate, self.out_rate = int(in_rate), int(out_rate)
        g = math.gcd(self.in_rate, self.out_rate)
        self.up, self.down = self.out_rate // g, self.in_rate // g  # t_n = n * down / up (input samples)
        r = self.out_rate / self.in_rate
        self.fc = min(1.0, r) * 0.97
        self.half = int(math.ceil(taps_per_side / min(1.0, r)))
        self.beta = beta
        # kernel table: one row per output phase (t_n fractional part = phase / up)
        k = np.arange(-self.half + 1, self.half + 1)
        frac = np.arange(self.up)[:, None] / self.up
        x = frac - k[None, :]
        w = np.clip(1.0 - (x / self.half) ** 2, 0.0, None)
        self.table = (self.fc * np.sinc(self.fc * x) * (np.i0(beta * np.sqrt(w)) / float(np.i0(beta))))
        self.taps = k
        self.buf = np.zeros(self.half, dtype=np.float64)  # virtual zero history
        self.offset = -self.half  # absolute input index of buf[0]
        self.n_out = 0  # next output sample index
        self.n_in = 0
        self.flushed = False

    @property
    def passthrough(self) -> bool:
        return self.in_rate == self.out_rate

    def _emit(self, limit_n: int) -> np.ndarray:
        """Produce outputs n in [self.n_out, limit_n)."""
        if limit_n <= self.n_out:
            return np.zeros(0, dtype=np.float32)
        n = np.arange(self.n_out, limit_n, dtype=np.int64)
        num = n * self.down
        base = num // self.up  # floor(t)
        phase = num % self.up
        idx = (base - self.offset)[:, None] + self.taps[None, :]
        out = np.einsum("ij,ij->i", self.buf[idx], self.table[phase])
        self.n_out = limit_n
        # drop input no longer needed
        keep_from = int(((self.n_out * self.down) // self.up) - self.half - self.offset)
        if keep_from > 0:
            self.buf = self.buf[keep_from:]
            self.offset += keep_from
        return out.astype(np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if self.passthrough:
            self.n_in += len(x)
            return x
        self.buf = np.concatenate([self.buf, x.astype(np.float64)])
        self.n_in += len(x)
        avail_end = self.offset + len(self.buf)  # exclusive absolute input index
        # output n needs inputs up to floor(t_n) + half  < avail_end
        max_t = avail_end - 1 - self.half
        if max_t < 0:
            return np.zeros(0, dtype=np.float32)
        limit = (max_t * self.up) // self.down + 1  # n with floor(n*down/up) <= max_t
        return self._emit(int(limit))

    def flush(self) -> np.ndarray:
        if self.passthrough or self.flushed:
            self.flushed = True
            return np.zeros(0, dtype=np.float32)
        self.flushed = True
        self.buf = np.concatenate([self.buf, np.zeros(self.half + 1)])
        total = int(round(self.n_in * self.up / self.down))
        return self._emit(total)


def to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def from_pcm16(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0


def wav_bytes(x: np.ndarray, rate: int) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(to_pcm16(x))
    return bio.getvalue()


def read_wav(path_or_bytes) -> tuple[np.ndarray, int]:
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else str(path_or_bytes)
    with wave.open(src, "rb") as w:
        rate, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, rate


def resample_all(x: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    r = StreamResampler(in_rate, out_rate)
    return np.concatenate([r.process(x), r.flush()])


def db(x: float) -> float:
    return 20.0 * math.log10(max(x, 1e-9))


def levels(x: np.ndarray, pairs: int = 8) -> dict:
    if len(x) == 0:
        return {"rms": 0.0, "peak": 0.0, "rms_db": -120.0, "wave": [0.0] * (2 * pairs)}
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(x)))
    wave_ = []
    for seg in np.array_split(x, pairs):
        if len(seg):
            wave_ += [round(float(seg.min()), 3), round(float(seg.max()), 3)]
        else:
            wave_ += [0.0, 0.0]
    return {"rms": round(rms, 4), "peak": round(peak, 4), "rms_db": round(db(rms), 1), "wave": wave_}


def frame_samples(rate: int, frame_ms: int) -> int:
    return int(rate * frame_ms // 1000)
