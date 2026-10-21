"""Shared CPU benchmark harness for ttsPIPE engine research.

Each engine script defines load() -> ctx and synth(ctx, text, lang) -> (np.float32 audio, sr),
optionally stream(ctx, text, lang) -> iterator of float32 chunks (true incremental audio),
then calls run(). Results -> data/research_results/<engine>.json, WAVs -> data/research_wavs/<engine>/.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # target is CPU only; must precede any torch import
os.environ.setdefault("HF_HOME", "C:/tts/ttsPIPE/data/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

ROOT = Path("C:/tts/ttsPIPE")
TEXTS = {
    "ta": ("ta", "வணக்கம், இன்று வானிலை மிகவும் நன்றாக இருக்கிறது. நாளை மழை பெய்யக்கூடும்."),
    "tanglish_script": ("ta", "நாளைக்கு meeting இருக்கு, please time-க்கு வந்துடுங்க."),
    "tanglish_roman": ("ta-latn", "Naalaikku meeting irukku, please time ku vandhudunga."),
    "en": ("en", "Hello! Your order has shipped and should arrive by Thursday afternoon."),
    "short_ta": ("ta", "வணக்கம், எப்படி இருக்கீங்க?"),
    "short_en": ("en", "Hello, how are you?"),
}


def rss_mb():
    mi = psutil.Process().memory_info()
    return mi.rss / 2**20, getattr(mi, "peak_wset", mi.rss) / 2**20  # peak_wset = Windows peak RSS


def set_threads(n):
    import sys

    if "torch" not in sys.modules:  # don't import torch for ONNX/C engines (would inflate RSS by ~180 MB)
        return "no torch (onnxruntime/C engine) device=cpu"
    try:
        import torch

        torch.set_num_threads(n)
        assert not torch.cuda.is_available(), "CUDA visible - CPU-only run required"
        return f"torch {torch.__version__} device=cpu threads={torch.get_num_threads()}"
    except ImportError:
        return "no torch"


def run(engine, load, synth, stream=None, texts=None, runs=5, extra=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--runs", type=int, default=runs)
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", default="", help="comma list of text keys")
    a = ap.parse_args()
    os.environ["TTSPIPE_THREADS"] = str(a.threads)
    name = engine + (f"-{a.tag}" if a.tag else "")
    wav_dir = ROOT / "data/research_wavs" / engine
    wav_dir.mkdir(parents=True, exist_ok=True)
    texts = texts or TEXTS
    if a.only:
        texts = {k: v for k, v in texts.items() if k in a.only.split(",")}

    rss0, _ = rss_mb()
    t0 = time.perf_counter()
    ctx = load()
    load_s = time.perf_counter() - t0
    dev = set_threads(a.threads)  # after load: engines import torch inside load()
    rss_loaded, _ = rss_mb()
    print(f"[{name}] {dev} load={load_s:.2f}s rss={rss_loaded:.0f}MB", flush=True)

    res = {"engine": name, "device": dev, "threads": a.threads, "cpu": platform.processor(),
           "box": "i5-12450H proxy (target: 8-core Ryzen, 14GB, no GPU)", "load_s": round(load_s, 3),
           "rss_before_load_mb": round(rss0), "rss_after_load_mb": round(rss_loaded), "texts": {}}
    for key, (lang, text) in texts.items():
        try:
            audio, sr = synth(ctx, text, lang)  # warm-up
            times, ttfas = [], []
            for _ in range(a.runs):
                t = time.perf_counter()
                audio, sr = synth(ctx, text, lang)
                times.append(time.perf_counter() - t)
                if stream:
                    t = time.perf_counter()
                    for _chunk in stream(ctx, text, lang):
                        ttfas.append(time.perf_counter() - t)
                        break
            audio = np.asarray(audio, dtype=np.float32).squeeze()
            dur = len(audio) / sr
            sf.write(wav_dir / f"{key}{('-' + a.tag) if a.tag else ''}.wav", audio, sr)
            med = statistics.median(times)
            r = {"text": text, "sr": sr, "audio_s": round(dur, 3), "synth_med_s": round(med, 4),
                 "synth_min_s": round(min(times), 4), "rtf_med": round(med / dur, 4) if dur else None,
                 "ttfa_med_ms": round(1000 * (statistics.median(ttfas) if ttfas else med), 1),
                 "ttfa_min_ms": round(1000 * (min(ttfas) if ttfas else min(times)), 1),
                 "ttfa_kind": "stream-first-chunk" if ttfas else "full-synth",
                 "peak_abs": round(float(np.abs(audio).max()), 3) if len(audio) else 0,
                 "rms_dbfs": round(float(20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-9)), 1) if len(audio) else None}
        except Exception as e:  # record and continue: report must show exact errors
            r = {"text": text, "error": f"{type(e).__name__}: {e}"}
        res["texts"][key] = r
        print(f"[{name}] {key}: {r}", flush=True)
    res["rss_peak_mb"] = round(rss_mb()[1])
    if extra:
        res.update(extra(ctx) or {})
    out = ROOT / "data/research_results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}-t{a.threads}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{name}] peak RSS {res['rss_peak_mb']} MB -> {out / f'{name}-t{a.threads}.json'}")
    return res
