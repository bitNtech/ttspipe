"""espeak-ng (GPL-3.0) formant synth via the espeak-ng.dll bundled in the espeakng-loader wheel (ctypes).
In a product, run espeak-ng.exe as a subprocess to keep GPL code out of process. Streams via synth callback.
ESPEAK_VOICE env overrides voice; default ta for ta / ta-latn, en-us for en.
"""
import ctypes as C
import os

from bench_common import run

VOICE = {"ta": "ta", "ta-latn": os.environ.get("ESPEAK_ROMAN_VOICE", "ta"), "en": "en-us"}


def load():
    import espeakng_loader as L
    import numpy as np

    lib = C.CDLL(L.get_library_path())
    sr = lib.espeak_Initialize(2, 200, L.get_data_path().encode(), 0)  # AUDIO_OUTPUT_SYNCHRONOUS, 200 ms buffers
    assert sr > 0, "espeak_Initialize failed"
    CB = C.CFUNCTYPE(C.c_int, C.POINTER(C.c_short), C.c_int, C.c_void_p)
    state = {"chunks": [], "on_chunk": None}

    def cb(wav, n, events):
        if wav and n > 0:
            a = np.ctypeslib.as_array(wav, shape=(n,)).astype(np.float32) / 32768.0
            state["chunks"].append(a)
            if state["on_chunk"]:
                state["on_chunk"]()
        return 0

    state["cb"] = CB(cb)  # keep a reference alive
    lib.espeak_SetSynthCallback(state["cb"])
    return lib, sr, state, np


def _speak(ctx, text, lang):
    lib, sr, state, np = ctx
    lib.espeak_SetVoiceByName(VOICE[lang].encode())
    b = text.encode("utf-8")
    state["chunks"] = []
    rc = lib.espeak_Synth(b, len(b) + 1, 0, 1, 0, 1, None, None)  # POS_CHARACTER, espeakCHARS_UTF8
    assert rc == 0, f"espeak_Synth rc={rc}"
    lib.espeak_Synchronize()


def synth(ctx, text, lang):
    lib, sr, state, np = ctx
    _speak(ctx, text, lang)
    return np.concatenate(state["chunks"]), sr


def extra(ctx):
    """True TTFA: time from espeak_Synth call to first audio callback (median of 5)."""
    import statistics
    import time

    from bench_common import TEXTS

    lib, sr, state, np = ctx
    out = {}
    for k, (lang, t) in TEXTS.items():
        ds = []
        for _ in range(5):
            first = {}
            state["on_chunk"] = lambda: first.setdefault("t", time.perf_counter())
            t0 = time.perf_counter()
            _speak(ctx, t, lang)
            ds.append(1000 * (first["t"] - t0))
        state["on_chunk"] = None
        out[f"ttfa_first_callback_ms_{k}"] = round(statistics.median(ds), 2)
    return out


if __name__ == "__main__":
    run("espeak-ng", load, synth, extra=extra)
