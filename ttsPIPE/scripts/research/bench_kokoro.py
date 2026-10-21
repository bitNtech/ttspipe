"""Kokoro-82M via kokoro-onnx (onnxruntime). --tag fp32|int8|fp16 picks the model file; KOKORO_VOICE env picks voice.
Tamil texts are phonemized with espeak-ng 'ta' (experiment only: Kokoro has no Tamil training data).
stream = app-side sentence split (kokoro-onnx itself only splits at 510 phonemes).
"""
import os
import re
import sys

from bench_common import ROOT, TEXTS, run

VARIANT = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--tag"), "fp32")
FILE = {"fp32": "kokoro-v1.0.onnx", "int8": "kokoro-v1.0.int8.onnx", "fp16": "kokoro-v1.0.fp16.onnx"}[VARIANT]
VOICE = os.environ.get("KOKORO_VOICE", "af_heart")
LANG = {"ta": "ta", "ta-latn": "en-us", "en": "en-us"}


def load():
    import numpy as np
    import onnxruntime as ort
    from kokoro_onnx import Kokoro

    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ["TTSPIPE_THREADS"])
    sess = ort.InferenceSession(str(ROOT / "data/models/kokoro" / FILE), sess_options=so, providers=["CPUExecutionProvider"])
    return Kokoro.from_session(sess, str(ROOT / "data/models/kokoro/voices-v1.0.bin")), np


def synth(ctx, text, lang):
    k, _ = ctx
    return k.create(text, voice=VOICE, lang=LANG[lang])


def stream(ctx, text, lang):
    k, _ = ctx
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        if sent:
            yield k.create(sent, voice=VOICE, lang=LANG[lang])[0]


def extra(ctx):
    k, _ = ctx
    out = {"voice": VOICE, "model_file": FILE, "n_voices": len(k.get_voices()), "voices": k.get_voices()}
    for key, (lang, t) in TEXTS.items():
        ph = k.tokenizer.phonemize(t, LANG[lang])
        out[f"phonemes_{key}"] = ph
        out[f"oov_{key}"] = sorted({c for c in ph if c not in k.tokenizer.vocab})
    return out


if __name__ == "__main__":
    run("kokoro", load, synth, stream=stream, extra=extra)
