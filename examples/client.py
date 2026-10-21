"""Reference client for the `/v1/tts/live` WebSocket protocol.

This is meant to double as living documentation: every frame it sends or
reads is exactly what `app.protocol` defines, nothing more. Point it at a
`TTS_ENGINE=fake` server for a GPU-free smoke test, or at a real one.

Examples:
    python examples/client.py --text "Hello there, friend." --out out.wav
    python examples/client.py --text "Some words to trickle in" \\
        --stream-text --out out.wav
    python examples/client.py --text "This is a longer story that we will \\
        interrupt partway through." --cancel-after 0.4 --out out.wav
    python examples/client.py --text "Say it in my voice." \\
        --ref-audio prompt.wav --ref-text "This is the reference clip." \\
        --format mp3 --out clone.mp3
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import time
from pathlib import Path

import ormsgpack
import websockets


def _pack(payload: dict) -> bytes:
    return ormsgpack.packb(payload)


def _unpack(raw: bytes | str) -> dict:
    if isinstance(raw, (bytes, bytearray)):
        return ormsgpack.unpackb(raw)
    return json.loads(raw)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="ws://127.0.0.1:8080/v1/tts/live", help="WebSocket URL")
    p.add_argument("--text", required=True, help="Text to synthesize")
    p.add_argument("--ref-audio", help="Path to a reference/prompt clip for voice cloning")
    p.add_argument("--ref-text", help="Exact transcript of --ref-audio")
    p.add_argument("--format", default="wav", choices=("wav", "pcm", "mp3", "flac"))
    p.add_argument("--out", default="out.wav", help="Where to write the received audio")
    p.add_argument(
        "--stream-text",
        action="store_true",
        help="Feed --text to the server word by word instead of in one 'text' frame, "
        "to demonstrate streaming input",
    )
    p.add_argument(
        "--cancel-after",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Send 'cancel' this many seconds after the first 'text' frame, then keep "
        "talking in the same session, to demonstrate barge-in",
    )
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


async def run(args: argparse.Namespace) -> None:
    request: dict = {"format": args.format}
    if args.seed is not None:
        request["seed"] = args.seed
    if args.ref_audio:
        ref_bytes = Path(args.ref_audio).read_bytes()
        request["references"] = [
            {"audio": base64.b64encode(ref_bytes).decode("ascii"), "text": args.ref_text or ""}
        ]

    audio_parts: list[bytes] = []
    t_connect = time.monotonic()
    t_first_audio: float | None = None
    t_last_audio: float | None = None
    total_audio_ms = 0.0

    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(_pack({"event": "start", "request": request}))

        async def sender() -> None:
            if args.stream_text:
                words = args.text.split(" ")
                for i, word in enumerate(words):
                    piece = word + (" " if i < len(words) - 1 else "")
                    await ws.send(_pack({"event": "text", "text": piece}))
                    await asyncio.sleep(0.03)
            else:
                await ws.send(_pack({"event": "text", "text": args.text}))

            if args.cancel_after is not None:
                await asyncio.sleep(args.cancel_after)
                print(f"[client] sending cancel after {args.cancel_after}s")
                await ws.send(_pack({"event": "cancel"}))
                await ws.send(_pack({"event": "text", "text": "Continuing right after the interruption."}))

            await ws.send(_pack({"event": "stop"}))

        send_task = asyncio.create_task(sender())
        try:
            async for raw in ws:
                frame = _unpack(raw)
                event = frame.get("event")
                if event == "started":
                    print(
                        f"[server] started: session={frame['session_id']} "
                        f"format={frame['format']} sample_rate={frame['sample_rate']} "
                        f"seed={frame['seed']}"
                    )
                elif event == "audio":
                    now = time.monotonic()
                    if t_first_audio is None:
                        t_first_audio = now
                    t_last_audio = now
                    audio_parts.append(frame["audio"])
                    total_audio_ms = frame["time"]
                elif event == "log":
                    print(f"[server] log: {frame['message']}")
                elif event == "finish":
                    print(f"[server] finish: reason={frame['reason']} error={frame.get('error')}")
                    if frame["reason"] in ("stop", "error"):
                        break
                    # reason == "cancel": the session is still active, keep reading.
                elif event == "pong":
                    print("[server] pong")
        finally:
            send_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await send_task

    out_path = Path(args.out)
    out_path.write_bytes(b"".join(audio_parts))

    audio_s = total_audio_ms / 1000.0
    ttfa_ms = (t_first_audio - t_connect) * 1000.0 if t_first_audio else None
    gen_wall_s = (t_last_audio - t_connect) if t_last_audio else 0.0
    rtf = (gen_wall_s / audio_s) if audio_s > 0 else float("nan")

    print()
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
    print(f"time-to-first-audio: {ttfa_ms:.1f} ms" if ttfa_ms is not None else "time-to-first-audio: n/a (no audio received)")
    print(f"total audio duration: {audio_s:.3f} s")
    print(f"generation wall time: {gen_wall_s:.3f} s")
    print(f"realtime factor (wall/audio, lower is faster than realtime): {rtf:.3f}")


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
