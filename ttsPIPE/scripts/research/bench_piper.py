"""Piper (piper-tts, GPL-3.0 runtime) VITS ONNX voices. Streams one chunk per sentence.
Usage: bench_piper.py --tag <voice-name> (voice file under data/models/piper/<voice>.onnx)
"""
import os
import sys

from bench_common import ROOT, TEXTS, run

VOICE = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--tag"), "ta_IN-rasa_female-medium")


def load():
    import numpy as np
    import onnxruntime as ort
    from piper import PiperVoice

    path = ROOT / f"data/models/piper/{VOICE}.onnx"
    v = PiperVoice.load(path)
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ["TTSPIPE_THREADS"])
    v.session = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    return v, np


def stream(ctx, text, lang):
    v, _ = ctx
    for chunk in v.synthesize(text):  # one AudioChunk per sentence
        yield chunk.audio_float_array


def synth(ctx, text, lang):
    v, np = ctx
    return np.concatenate(list(stream(ctx, text, lang))), v.config.sample_rate


def extra(ctx):
    v, _ = ctx
    return {f"phonemes_{k}": ["".join(s) for s in v.phonemize(t)] for k, (_, t) in TEXTS.items()}


if __name__ == "__main__":
    run("piper", load, synth, stream=stream, extra=extra)
