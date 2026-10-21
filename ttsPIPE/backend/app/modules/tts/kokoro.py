"""tts/kokoro: Kokoro-82M via kokoro-onnx. English, plus Tamil through espeak-ng phonemes. See docs/research/kokoro.md."""
from __future__ import annotations

import numpy as np

from app.core.base import AudioPiece, ModelFile, ModuleInfo, TTSModule, VoiceInfo, schema
from app.core.registry import register
from app.core.textutil import has_tamil

REL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
FILES = {"fp32": ("kokoro-v1.0.onnx", 326), "int8": ("kokoro-v1.0.int8.onnx", 114)}
VOICES = [("af_heart", "Heart", "female"), ("af_bella", "Bella", "female"), ("af_nicole", "Nicole", "female"),
          ("am_michael", "Michael", "male"), ("am_adam", "Adam", "male"), ("bf_emma", "Emma", "female"),
          ("bm_george", "George", "male")]
# Kokoro has no Tamil training data; its Hindi voices carry the closest accent and phoneme set
TA_VOICES = [("hf_alpha", "Alpha", "female"), ("hf_beta", "Beta", "female"), ("hm_omega", "Omega", "male"),
             ("hm_psi", "Psi", "male")]
# espeak-ng 'ta' symbols outside Kokoro's vocabulary (dropped silently) or only seen in English,
# mapped to the ones the Hindi voices were trained on
TA_PHONES = str.maketrans({"ʉ": "ʊ", "ɭ": "l", "ɹ": "ɾ"})


def ta_phonemes(text: str) -> str:
    import phonemizer
    from kokoro_onnx.tokenizer import _espeak_lock

    with _espeak_lock:
        ph = phonemizer.phonemize(text, "ta", preserve_punctuation=True, with_stress=True)
    return ph.translate(TA_PHONES).strip()


@register
class KokoroTTS(TTSModule):
    info = ModuleInfo(
        id="tts/kokoro", kind="tts", name="Kokoro-82M (ONNX)",
        description="Best-quality small English voice. Tamil is spoken through espeak-ng phonemes with the Hindi "
                    "voices (hf_*/hm_*): clear but accented, as Kokoro has no Tamil training data. "
                    "Whole chunk per call (non-streaming); the chunker keeps chunks short.",
        model_id="hexgrad/Kokoro-82M", size_mb=354, ram_mb=600, license="Apache-2.0 weights; GPL-3.0 phonemizer/espeak-ng runtime",
        streaming=False, native_rate=24000, requires=["kokoro_onnx", "onnxruntime"], family="kokoro", languages=["en", "ta"],
        tags=["gpl-runtime"], voices=[VoiceInfo(v, f"{n} ({'UK' if v[0] == 'b' else 'US'})", "en", g) for v, n, g in VOICES]
        + [VoiceInfo(v, f"{n} (Tamil)", "ta", g) for v, n, g in TA_VOICES],
        params_schema=schema(
            variant=("string", "fp32", {"enum": list(FILES), "description": "int8 is smaller; see research for speed"}),
            threads=("integer", 4, {"min": 1, "max": 16}),
        ),
    )

    def model_files(self) -> list[ModelFile]:
        name, size = FILES[self.params["variant"]]
        return [ModelFile(name, f"{REL}/{name}", f"tts/kokoro/{name}", size),
                ModelFile("voices-v1.0.bin", f"{REL}/voices-v1.0.bin", "tts/kokoro/voices-v1.0.bin", 28)]

    def load(self) -> None:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        name, _ = FILES[self.params["variant"]]
        path = self.models_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — POST /api/models/tts/kokoro/download or ./run.sh models kokoro")
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(self.params["threads"])
        sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        self.k = Kokoro.from_session(sess, str(self.models_dir / "voices-v1.0.bin"))
        self.k.create("ok.", voice="af_heart")  # warm-up
        self.loaded = True

    def unload(self) -> None:
        self.k = None
        self.loaded = False

    def synthesize(self, state, chunk, plan, voice):
        tamil = has_tamil(chunk.tts_text)
        known = {v for v, _, _ in VOICES + TA_VOICES}
        vid = voice.voice_id if voice.voice_id in known else ("hf_alpha" if tamil else "af_heart")
        speed = float(np.clip(voice.speed * plan.speed, 0.5, 2.0))
        en = "en-gb" if vid.startswith("b") else "en-us"
        if tamil:  # per segment, so English words in a Tamil chunk keep English phonemes
            ph = " ".join(ta_phonemes(t) if has_tamil(t) else self.k.tokenizer.phonemize(t, en)
                          for t in (s.tts_text for s in chunk.segments or [chunk]) if t.strip())
            audio, rate = self.k.create(ph, voice=vid, speed=speed, is_phonemes=True)
        else:
            audio, rate = self.k.create(chunk.tts_text, voice=vid, speed=speed, lang=en)
        yield AudioPiece(pcm=np.asarray(audio, np.float32), sample_rate=rate, is_last=True)
