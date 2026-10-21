"""Time-to-first-audio and per-frame cost, in-process, on the real model.

Loads the worker exactly as the server does, sets the active voice, then runs
turns straight through `GpuWorker._handle` so nothing but the GPU is measured:

- ``repeat``: the same sentence twice; the second has its whole prompt cached.
- ``cold``:   a sentence the cache has never seen (only the voice prefix is warm).
- ``hinted``: a sentence whose prompt a hint prefilled first, as the session does
  while an LLM is still streaming text in.

Needs the GPU to itself - stop the server first.

    python scripts/bench_latency.py
    python scripts/bench_latency.py --turns 5 --wav out
"""

from __future__ import annotations

import argparse
import queue
import statistics
import sys
import time
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SENTENCES = [
    "Hello, thanks for calling. How can I help you today?",
    "Sure, I can check the status of your order right now.",
    "Your package left the warehouse this morning and should arrive tomorrow.",
    "Is there anything else I can help you with?",
    "Great, have a wonderful day and thank you for your patience.",
]


def _run(worker, text, params):
    from app.engine.base import CancelToken
    from app.engine.worker import ChunkMessage, DoneMessage, ErrorMessage, _Job

    out: queue.Queue = queue.Queue()
    job = _Job(text=text, params=params, cancel=CancelToken(), out=out)
    t0 = time.perf_counter()
    worker._handle(job)
    chunks, first = [], None
    while True:
        msg = out.get_nowait()
        if isinstance(msg, ChunkMessage):
            chunks.append(msg.chunk.samples)
        elif isinstance(msg, ErrorMessage):
            raise msg.error
        elif isinstance(msg, DoneMessage):
            m = msg.metrics
            first = m.ttfa_ms
            wall = (m.finished_at - t0) * 1000.0
            return first, wall, m, chunks


def _write_wav(path, chunks, rate):
    import numpy as np

    pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    data = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=len(SENTENCES))
    parser.add_argument("--wav", type=Path, default=None, help="directory to save turns")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from app.config import get_settings
    from app.engine.base import CancelToken
    from app.engine.factory import _ensure_vendor_on_path
    from app.engine.worker import GpuWorker
    from app.protocol import TTSParams

    settings = get_settings()
    _ensure_vendor_on_path(settings)
    import torch

    worker = GpuWorker(settings)
    t = time.perf_counter()
    worker._load()
    print(f"load {time.perf_counter() - t:.1f}s  vram {torch.cuda.memory_allocated() / 2**20:.0f} MiB")

    params = TTSParams(reference_id="_active", seed=args.seed)
    warm = worker._streamer.warm(
        *reversed(worker._references.resolve([], "_active")), CancelToken()
    )
    print(f"voice prefix {warm} tokens")
    # One throwaway turn: first-call allocations, cuBLAS handles, kernel builds.
    _run(worker, "Warming up.", params)

    if args.wav:
        args.wav.mkdir(parents=True, exist_ok=True)

    rows = {"cold": [], "repeat": [], "hinted": []}
    frame_ms = []
    for i, text in enumerate(SENTENCES[: args.turns]):
        ttfa, wall, m, chunks = _run(worker, text, params)
        stats = worker._streamer.stats
        rows["cold"].append((ttfa, stats.last_prefilled_tokens))
        frame_ms.append(wall / max(1, m.token_frames))
        rtf = (wall / 1000.0) / max(1e-9, m.audio_seconds)
        print(
            f"cold   ttfa {ttfa:6.0f} ms  N={stats.last_prefilled_tokens:3d}  "
            f"frames {m.token_frames:3d}  {wall / max(1, m.token_frames):5.1f} ms/frame  rtf {rtf:.2f}"
        )
        if args.wav:
            _write_wav(args.wav / f"turn{i}.wav", chunks, worker.sample_rate)

        ttfa, *_ = _run(worker, text, params)
        rows["repeat"].append((ttfa, worker._streamer.stats.last_prefilled_tokens))
        print(f"repeat ttfa {ttfa:6.0f} ms  N={worker._streamer.stats.last_prefilled_tokens:3d}")

        # What a streaming client gets: a hint had the idle gap to prefill.
        other = SENTENCES[(i + 1) % len(SENTENCES)]
        _run(worker, other, TTSParams(reference_id="_active", seed=args.seed, max_new_tokens=1))
        prompt_tokens, prompt_texts = worker._references.resolve([], "_active")
        worker._streamer.hint(text, prompt_texts, prompt_tokens, CancelToken())
        ttfa, *_ = _run(worker, text, params)
        rows["hinted"].append((ttfa, worker._streamer.stats.last_prefilled_tokens))
        print(f"hinted ttfa {ttfa:6.0f} ms  N={worker._streamer.stats.last_prefilled_tokens:3d}")

    med = statistics.median
    print("\nmedian ttfa_ms: " + "  ".join(f"{k} {med(r[0] for r in v):.0f}" for k, v in rows.items()))
    print(f"median ms/frame (whole turn incl. codec): {med(frame_ms):.1f}  (realtime = 46.4)")


if __name__ == "__main__":
    main()
