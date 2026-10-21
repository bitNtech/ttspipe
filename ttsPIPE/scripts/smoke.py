"""End-to-end smoke test (ARCHITECTURE.md §15).

Speaks every /api/scripts text through /ws/tts with the builtin engine (tts/tone) and with the default
engine (the default config's `active_profile` TTS: fish_s2, or balanced on a CPU box), prints TTFA per script, and checks that a stop mid-utterance returns
a correct heard_text. Exit code 0 = all checks passed.

    python scripts/smoke.py                 # starts a private server on a free port
    python scripts/smoke.py --url http://127.0.0.1:8200   # use a running server (config is restored)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def http(url: str, method: str = "GET", body=None, timeout: float = 30):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_server() -> tuple[subprocess.Popen, str]:
    port = free_port()
    env = dict(os.environ, TTSPIPE_PORT=str(port), TTSPIPE_DATA_DIR=tempfile.mkdtemp(prefix="ttspipe-smoke-"),
               TTSPIPE_LOG_LEVEL="WARNING", PYTHONUTF8="1")
    env.setdefault("TTSPIPE_MODELS_DIR", str(ROOT / "data" / "models"))  # reuse downloaded models
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port),
                             "--log-level", "warning"], cwd=ROOT / "backend", env=env)
    url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            http(url + "/api/health", timeout=1)
            return proc, url
        except Exception:
            time.sleep(0.1)
    proc.kill()
    raise SystemExit("server did not start")


def wait_ready(url: str, timeout: float = 600) -> dict:
    t_end = time.time() + timeout
    while time.time() < t_end:
        st = http(url + "/api/pipeline")["stages"]["tts"]
        if st["status"] != "loading":
            return st
        time.sleep(0.3)
    raise SystemExit("tts stage still loading")


async def speak(ws_url: str, text: str, *, llm: bool, stop_after_words: int | None = None) -> dict:
    import websockets

    out = {"frames": 0, "events": [], "end": None, "interrupted": None, "progress": None}
    async with websockets.connect(ws_url, max_size=None) as ws:
        while True:  # session.started
            m = json.loads(await ws.recv())
            if m["type"] == "session.started":
                break
        uid = f"smoke{int(time.time() * 1000) % 100000}"
        t0 = time.perf_counter()
        if llm:
            for i in range(0, len(text), 3):
                await ws.send(json.dumps({"type": "text", "delta": text[i:i + 3], "utterance_id": uid}))
                await asyncio.sleep(0.015)
            await ws.send(json.dumps({"type": "end_utterance", "utterance_id": uid}))
        else:
            await ws.send(json.dumps({"type": "speak", "text": text, "utterance_id": uid}))
        stopped = False
        first_frame = None
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=300)
            if isinstance(msg, bytes):
                out["frames"] += 1
                if first_frame is None:
                    first_frame = (time.perf_counter() - t0) * 1000
                continue
            ev = json.loads(msg)
            out["events"].append(ev["type"])
            d = ev["data"]
            if ev["type"] == "playback.progress":
                out["progress"] = d
                if stop_after_words is not None and not stopped and d["word_index"] >= stop_after_words:
                    stopped = True
                    await ws.send(json.dumps({"type": "stop"}))
            elif ev["type"] == "utterance.interrupted":
                out["interrupted"] = d
            elif ev["type"] == "utterance.end":
                out["end"] = d
                out["client_first_frame_ms"] = first_frame
                return out


def check_heard(text_words: list[str], heard: str, unplayed: str, last_progress: str | None) -> list[str]:
    errs = []
    hw = heard.split()
    if hw != text_words[:len(hw)]:
        errs.append(f"heard_text is not a word prefix: {heard!r}")
    if not hw:
        errs.append("heard_text is empty after words were played")
    if len(hw) >= len(text_words):
        errs.append("heard_text covers the whole utterance although it was stopped")
    if (heard + " " + unplayed).split() != text_words:
        errs.append("heard_text + unplayed_text != utterance text")
    if last_progress and not heard.startswith(last_progress):
        errs.append(f"heard_text {heard!r} does not extend last progress {last_progress!r}")
    return errs


async def run_engine(url: str, impl: str, voice: str | None, scripts: list[dict],
                     params: dict | None = None) -> tuple[list[dict], list[str]]:
    body = {"impl": impl, "params": params or {}}
    http(url + "/api/pipeline/stages/tts", "PUT", body)
    st = wait_ready(url)
    if st["impl"] != impl:
        return [], [f"{impl}: not loaded ({st.get('error')}) — slot runs {st['impl']}"]
    if voice:
        http(url + "/api/pipeline/voice", "PUT", {"impl": impl, "voice_id": voice})
    ws_url = url.replace("http", "ws") + "/ws/tts?label=smoke"
    rows, errors = [], []
    # warm-up (first call pays lazy init)
    await speak(ws_url, "Warm up.", llm=False)
    for s in scripts:
        r = await speak(ws_url, s["text"], llm=True)
        lat = r["end"]["latency"]
        audio_ms = lat.get("audio_ms") or 0
        rows.append({"engine": impl, "script": s["id"], "ttfa_ms": lat.get("ttfa_ms"), "ttfa_end_ms": lat.get("ttfa_end_ms"),
                     "first_synth_ms": lat.get("first_synth_ms"), "rtf": lat.get("rtf"), "audio_s": round(audio_ms / 1000, 2),
                     "frames": r["frames"], "chunks": len(r["end"]["chunks"])})
        if r["frames"] == 0 or not audio_ms:
            errors.append(f"{impl}/{s['id']}: no audio")
        if "underrun" in r["events"]:
            rows[-1]["underrun"] = True
    long = next((s for s in scripts if s["id"] == "long"), scripts[-1])
    r = await speak(ws_url, long["text"], llm=False, stop_after_words=4)
    if not r["interrupted"]:
        errors.append(f"{impl}: stop did not produce utterance.interrupted")
    else:
        d = r["interrupted"]
        # heard_text is in ingested text (markdown stripped); the long script is plain text
        errs = check_heard(long["text"].split(), d["heard_text"], d["unplayed_text"], (r["progress"] or {}).get("heard_text"))
        errors += [f"{impl}/stop: {e}" for e in errs]
        print(f"  stop test [{impl}]: heard {len(d['heard_text'].split())} words → {d['heard_text']!r}")
    return rows, errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="running server; default starts a private one")
    ap.add_argument("--engines", default="tone,default", help="comma list: tone, default, or tts impl names")
    ap.add_argument("--profile", help="profile whose TTS is the default engine (default: the active profile)")
    args = ap.parse_args()
    proc = None
    url = args.url
    if not url:
        proc, url = start_server()
    original = http(url + "/api/pipeline")["config"]
    try:
        scripts = [s for s in http(url + "/api/scripts") if s["id"] != "llm_markdown"] + \
                  [s for s in http(url + "/api/scripts") if s["id"] == "llm_markdown"]
        prof_name = args.profile or http(url + "/api/pipeline").get("active_profile") or "balanced"
        balanced = http(url + f"/api/profiles/{prof_name}")["config"]
        default_impl = ((balanced.get("stages") or {}).get("tts") or {}).get("impl") or "tone"
        default_params = ((balanced.get("stages") or {}).get("tts") or {}).get("params") or {}
        default_voice = (balanced.get("voice") or {}).get("voice_id")
        all_rows, all_errors = [], []
        for e in args.engines.split(","):
            impl, voice = ("tone", "tone-f") if e == "tone" else (default_impl, default_voice) if e == "default" else (e, None)
            if e == "default" and impl == "tone":
                print(f"default engine is tone ({prof_name} profile) — skipping duplicate run")
                continue
            print(f"== engine {impl} ==")
            rows, errs = asyncio.run(run_engine(url, impl, voice, scripts, default_params if e == "default" else None))
            all_rows += rows
            all_errors += errs
        hdr = ["engine", "script", "ttfa_ms", "ttfa_end_ms", "first_synth_ms", "rtf", "audio_s", "chunks", "frames"]
        print("\n" + " | ".join(f"{h:>14}" for h in hdr))
        for r in all_rows:
            print(" | ".join(f"{str(r.get(h)):>14}" for h in hdr) + ("  UNDERRUN" if r.get("underrun") else ""))
        by_engine: dict[str, list[float]] = {}
        for r in all_rows:
            if r["ttfa_ms"] is not None:
                by_engine.setdefault(r["engine"], []).append(r["ttfa_ms"])
        for eng, v in by_engine.items():
            v.sort()
            print(f"TTFA {eng}: p50 {v[len(v) // 2]:.0f} ms, max {v[-1]:.0f} ms over {len(v)} scripts")
        if all_errors:
            print("\nFAILED:\n  " + "\n  ".join(all_errors))
            return 1
        print("\nsmoke OK")
        return 0
    finally:
        try:
            http(url + "/api/pipeline", "PUT", original)
        except Exception:
            pass
        if proc:
            proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
