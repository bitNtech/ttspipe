"""End-to-end TTFA per builtin profile through the real pipeline (private server, quiet box).

    .venv/Scripts/python.exe scripts/research/profile_ttfa.py [profile ...]

The phrase cache is disabled. Every sample script is streamed as LLM deltas (3 chars / 15 ms) after one warm-up; TTFA is measured
from the first delta to the first audio frame sent (server-side `utterance.end.latency`).
"""
import asyncio
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smoke import http, speak, start_server, wait_ready  # noqa: E402

PROFILES = sys.argv[1:] or ["lowest_latency", "balanced", "high_quality", "english_fast", "minimal"]


async def run(url, scripts):
    ws = url.replace("http", "ws") + "/ws/tts?label=profile"
    await speak(ws, "Warm up. வணக்கம்.", llm=False)
    rows = []
    for s in scripts:
        for _ in range(3):
            r = await speak(ws, s["text"], llm=True)
            lat = r["end"]["latency"]
            rows.append({"script": s["id"], "ttfa": lat["ttfa_ms"], "ttfa_end": lat["ttfa_end_ms"],
                         "first_synth": lat["first_synth_ms"], "rtf": lat["rtf"], "underrun": "underrun" in r["events"]})
    return rows


def main():
    proc, url = start_server()
    out = {}
    try:
        scripts = [s for s in http(url + "/api/scripts") if s["id"] != "llm_markdown"]
        for p in PROFILES:
            cfg = http(url + f"/api/profiles/{p}/apply", "POST")["config"]
            cfg["cache"] = {"enabled": False, "max_mb": 64}  # measure synthesis, not phrase-cache hits
            http(url + "/api/pipeline", "PUT", cfg)
            st = wait_ready(url)
            rows = asyncio.run(run(url, scripts))
            t = sorted(r["ttfa"] for r in rows)
            out[p] = {"tts": st["impl"], "fallback": st["fallback"], "n": len(t), "ttfa_p50": statistics.median(t),
                      "ttfa_p95": t[int(0.95 * (len(t) - 1))], "ttfa_end_p50": statistics.median(r["ttfa_end"] for r in rows),
                      "first_synth_p50": statistics.median(r["first_synth"] for r in rows),
                      "rtf_p50": statistics.median(r["rtf"] for r in rows if r["rtf"] is not None),
                      "underruns": sum(r["underrun"] for r in rows),
                      "per_script": {s["id"]: statistics.median(r["ttfa"] for r in rows if r["script"] == s["id"]) for s in scripts}}
            print(p, json.dumps(out[p]), flush=True)
    finally:
        proc.terminate()
    dest = Path(__file__).resolve().parents[2] / "data" / "research_results" / "profile_ttfa.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
