"""Measure whether a live turn can actually be played without breaking up.

`examples/client.py` reports time-to-first-audio and a realtime factor. Neither
says what a listener hears, because both are averages over a whole turn and
what breaks a voice is a single moment where the next sample has not arrived
yet. This asks the question directly:

  * **min_preroll_ms** - the smallest amount a client could buffer before
    starting playback and still never run dry. It is the maximum, over every
    audio frame, of "wall time this frame arrived" minus "audio playable before
    it", which is exactly the deficit a playout clock would hit there. Below it,
    playback breaks; at or above it, playback is continuous.
  * **stall** figures for a given pre-roll - how many times and for how long a
    player would have run dry, so a proposed buffer setting can be checked
    rather than guessed.
  * **seam** figures - the sample-to-sample discontinuity at every frame
    boundary against the distribution inside the frames, which is what a click
    at a block or segment boundary looks like numerically.

It talks the same protocol `examples/client.py` does and asks for `pcm`, since
that is the format whose bytes are samples and can be analysed without a
decoder.

    python scripts/stream_probe.py --text "..." --trickle
    python scripts/stream_probe.py --text "..." --preroll 600 --json out.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import ormsgpack
import websockets


@dataclass
class Frame:
    """One `audio` event: when it landed and how much speech it carried."""

    at: float  # seconds since the turn started
    samples: np.ndarray
    reported_ms: float  # the server's cumulative `time` field


@dataclass
class Turn:
    started_at: float
    frames: list[Frame] = field(default_factory=list)
    finished_at: float | None = None
    finish_reason: str | None = None
    sample_rate: int = 44100
    logs: list[str] = field(default_factory=list)


def _pack(payload: dict) -> bytes:
    return ormsgpack.packb(payload)


def _unpack(raw: bytes | str) -> dict:
    if isinstance(raw, (bytes, bytearray)):
        return ormsgpack.unpackb(raw)
    return json.loads(raw)


async def run_turn(args: argparse.Namespace) -> Turn:
    request: dict = {"format": "pcm", "temperature": args.temperature}
    if args.seed is not None:
        request["seed"] = args.seed
    if args.ref_audio:
        request["references"] = [
            {
                "audio": base64.b64encode(Path(args.ref_audio).read_bytes()).decode("ascii"),
                "text": args.ref_text or "",
            }
        ]

    async with websockets.connect(args.url, max_size=None) as ws:
        turn = Turn(started_at=time.perf_counter())
        await ws.send(_pack({"event": "start", "request": request}))

        async def sender() -> None:
            if args.trickle:
                # The shape a voice agent's LLM produces: a word at a time.
                words = args.text.split(" ")
                for i, word in enumerate(words):
                    await ws.send(
                        _pack(
                            {
                                "event": "text",
                                "text": word + (" " if i < len(words) - 1 else ""),
                            }
                        )
                    )
                    await asyncio.sleep(args.word_interval)
            else:
                await ws.send(_pack({"event": "text", "text": args.text}))
            await ws.send(_pack({"event": "stop"}))

        send_task = asyncio.create_task(sender())
        try:
            async for raw in ws:
                frame = _unpack(raw)
                event = frame.get("event")
                if event == "started":
                    turn.sample_rate = frame["sample_rate"]
                elif event == "audio":
                    turn.frames.append(
                        Frame(
                            at=time.perf_counter() - turn.started_at,
                            samples=np.frombuffer(frame["audio"], dtype="<i2").astype(
                                np.float32
                            )
                            / 32768.0,
                            reported_ms=float(frame["time"]),
                        )
                    )
                elif event == "log":
                    turn.logs.append(frame["message"])
                elif event == "finish":
                    turn.finished_at = time.perf_counter() - turn.started_at
                    turn.finish_reason = frame["reason"]
                    break
        finally:
            send_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await send_task
    return turn


def playout(turn: Turn, preroll_s: float) -> dict:
    """Simulate a player that starts `preroll_s` after the first audio arrives.

    The player consumes samples at exactly realtime from a queue the server
    fills in bursts. Audio position `d` (seconds into the utterance) can only
    be heard once the frame carrying it has landed, so a stall happens whenever
    the playout clock reaches a position whose frame is still in flight. The
    stall's length is how long the player then waits, and once it resumes the
    clock has slipped by that much - which is why this carries `slip` forward
    rather than measuring each frame against the original start.
    """
    rate = turn.sample_rate
    if not turn.frames:
        return {"stalls": 0, "stall_ms": 0.0, "min_preroll_ms": 0.0}

    start = turn.frames[0].at + preroll_s
    played = 0.0  # seconds of audio consumed by the playout clock
    slip = 0.0  # extra wall time spent stalled
    stalls = 0
    stall_ms = 0.0
    worst = 0.0
    deficit_at = 0.0

    available = 0.0  # seconds of audio handed over so far
    for frame in turn.frames:
        # Everything already in the queue can play before this frame lands.
        wall_when_queue_empties = start + slip + available
        deficit = frame.at - wall_when_queue_empties
        if deficit > worst:
            worst = deficit
            deficit_at = available
        # A player is not stalled by a nanosecond. The tolerance also keeps
        # `min_preroll_ms` a real boundary rather than one floating-point
        # rounding away from failing its own definition.
        if deficit > 1e-9:
            stalls += 1
            stall_ms += deficit * 1000.0
            slip += deficit
        available += len(frame.samples) / rate
        played = available

    # The same walk with preroll removed gives the smallest pre-roll that would
    # have left no deficit anywhere.
    need = 0.0
    avail = 0.0
    for frame in turn.frames:
        need = max(need, frame.at - turn.frames[0].at - avail)
        avail += len(frame.samples) / rate

    return {
        "stalls": stalls,
        "stall_ms": round(stall_ms, 1),
        "worst_stall_ms": round(worst * 1000.0, 1),
        "worst_stall_at_s": round(deficit_at, 2),
        # Rounded *up*: this number's whole meaning is "a buffer this deep
        # plays through", and rounding to nearest would hand back a value a
        # fraction of a millisecond short of its own definition.
        "min_preroll_ms": math.ceil(need * 10000.0) / 10.0,
        "audio_s": round(played, 3),
    }


def seams(turn: Turn) -> dict:
    """How discontinuous the waveform is where two frames meet.

    A click at a block boundary is a sample-to-sample jump much larger than the
    signal's own steps. Comparing the jumps at frame joins against the
    distribution of jumps inside frames is what makes "much larger" a number.
    """
    if len(turn.frames) < 2:
        return {}
    audio = np.concatenate([f.samples for f in turn.frames])
    deltas = np.abs(np.diff(audio))
    if deltas.size == 0:
        return {}

    joins = []
    at = 0
    for frame in turn.frames[:-1]:
        at += len(frame.samples)
        if 0 < at < len(audio):
            joins.append(abs(float(audio[at] - audio[at - 1])))

    p9999 = float(np.percentile(deltas, 99.99))
    return {
        "frames": len(turn.frames),
        "join_max": round(max(joins), 5) if joins else 0.0,
        "join_mean": round(statistics.fmean(joins), 5) if joins else 0.0,
        "inside_p99_99": round(p9999, 5),
        # >1 means the worst join is sharper than anything the signal itself
        # does; that is what a click sounds like.
        "join_max_over_p99_99": round(max(joins) / p9999, 2) if joins and p9999 else 0.0,
        "peak": round(float(np.abs(audio).max()), 4),
    }


def silence_runs(turn: Turn, floor: float = 1e-4, min_ms: float = 30.0) -> dict:
    """Runs of near-silence inside the delivered audio, longest first."""
    if not turn.frames:
        return {}
    audio = np.concatenate([f.samples for f in turn.frames])
    quiet = np.abs(audio) < floor
    runs: list[tuple[float, float]] = []
    start = None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            runs.append((start / turn.sample_rate, (i - start) / turn.sample_rate))
            start = None
    if start is not None:
        runs.append((start / turn.sample_rate, (len(quiet) - start) / turn.sample_rate))
    runs = [r for r in runs if r[1] * 1000.0 >= min_ms]
    runs.sort(key=lambda r: -r[1])
    return {
        "count": len(runs),
        "longest_ms": round(runs[0][1] * 1000.0, 1) if runs else 0.0,
        "longest_at_s": round(runs[0][0], 2) if runs else 0.0,
    }


def summarise(turn: Turn, prerolls: list[float]) -> dict:
    rate = turn.sample_rate
    audio_s = sum(len(f.samples) for f in turn.frames) / rate if turn.frames else 0.0
    ttfa_ms = turn.frames[0].at * 1000.0 if turn.frames else None
    wall_s = turn.frames[-1].at if turn.frames else 0.0
    gaps = [
        round((b.at - a.at) * 1000.0, 1) for a, b in zip(turn.frames, turn.frames[1:])
    ]
    block_ms = [round(len(f.samples) / rate * 1000.0, 1) for f in turn.frames]

    return {
        "sample_rate": rate,
        "finish_reason": turn.finish_reason,
        "frames": len(turn.frames),
        "audio_s": round(audio_s, 3),
        "ttfa_ms": round(ttfa_ms, 1) if ttfa_ms is not None else None,
        "wall_to_last_audio_s": round(wall_s, 3),
        # Below 1.0 the server produces audio faster than it plays; above it,
        # a client must buffer and the buffer must grow with the utterance.
        "rtf": round(wall_s / audio_s, 3) if audio_s else None,
        "block_ms": {
            "first": block_ms[0] if block_ms else None,
            "median": round(statistics.median(block_ms), 1) if block_ms else None,
            "max": max(block_ms) if block_ms else None,
        },
        "arrival_gap_ms": {
            "median": round(statistics.median(gaps), 1) if gaps else None,
            "max": max(gaps) if gaps else None,
        },
        # One row per frame: when it landed, how much speech it carried, and
        # how long the client had been waiting for it. The row where `wait`
        # jumps is where the turn actually breaks.
        "timeline": [
            {
                "at_s": round(f.at, 3),
                "wait_ms": round((f.at - (turn.frames[i - 1].at if i else 0.0)) * 1000.0, 1),
                "ms": round(len(f.samples) / rate * 1000.0, 1),
            }
            for i, f in enumerate(turn.frames)
        ],
        "playout": {f"preroll_{int(p * 1000)}ms": playout(turn, p) for p in prerolls},
        "seams": seams(turn),
        "silence": silence_runs(turn),
        "logs": turn.logs,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--url", default="ws://127.0.0.1:8080/v1/tts/live")
    p.add_argument("--text", required=True)
    p.add_argument("--trickle", action="store_true", help="send text a word at a time")
    p.add_argument("--word-interval", type=float, default=0.06)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--ref-audio")
    p.add_argument("--ref-text")
    p.add_argument(
        "--preroll",
        type=float,
        nargs="*",
        default=[0.06, 0.3, 1.0],
        metavar="MS",
        help="pre-roll sizes in ms to simulate (default: 60 300 1000)",
    )
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--json", type=Path, help="also write the full report here")
    p.add_argument("--wav", type=Path, help="write the received audio for listening")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prerolls = [p / 1000.0 if p > 5 else p for p in args.preroll]

    reports = []
    for run in range(args.repeat):
        turn = asyncio.run(run_turn(args))
        report = summarise(turn, prerolls)
        reports.append(report)
        print(f"--- run {run + 1}/{args.repeat} ---")
        print(json.dumps(report, indent=2))
        if args.wav and turn.frames:
            import wave

            audio = np.concatenate([f.samples for f in turn.frames])
            with wave.open(str(args.wav), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(turn.sample_rate)
                w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
            print(f"wrote {args.wav}")

    if args.json:
        args.json.write_text(json.dumps(reports, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
