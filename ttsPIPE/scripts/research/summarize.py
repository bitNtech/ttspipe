"""Print markdown tables from data/research_results/*.json. Usage: summarize.py [name-substring]"""
import json
import sys
from pathlib import Path

R = Path("C:/tts/ttsPIPE/data/research_results")
filt = sys.argv[1] if len(sys.argv) > 1 else ""
for f in sorted(R.glob("*.json")):
    if filt not in f.stem:
        continue
    d = json.loads(f.read_text(encoding="utf-8"))
    print(f"\n#### {f.stem}  (load {d['load_s']} s, RSS after load {d['rss_after_load_mb']} MB, peak RSS {d['rss_peak_mb']} MB, {d['device']})\n")
    print("| text | audio s | synth median s | synth min s | RTF median | TTFA median ms | TTFA min ms | TTFA kind |")
    print("|---|---|---|---|---|---|---|---|")
    for k, v in d["texts"].items():
        if "error" in v:
            print(f"| {k} | ERROR: {v['error']} |||||||")
        else:
            print(f"| {k} | {v['audio_s']} | {v['synth_med_s']} | {v['synth_min_s']} | {v['rtf_med']} | {v['ttfa_med_ms']} | {v['ttfa_min_ms']} | {v['ttfa_kind']} |")
