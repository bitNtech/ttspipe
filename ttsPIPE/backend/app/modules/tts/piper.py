"""tts/piper: Piper VITS voices (ONNX) — Tamil (AI4Bharat Rasa) and English. See docs/research/piper.md."""
from __future__ import annotations

import numpy as np

from app.core.base import AudioPiece, ModelFile, ModuleInfo, TTSModule, VoiceInfo, schema
from app.core.registry import register

HF = "https://huggingface.co"
# voice id → (repo path prefix, language, gender, licence, size MB)
VOICES = {
    "ta_IN-rasa_female-medium": (f"{HF}/tinisoft/piper-ta_IN-rasa_female-medium/resolve/main", "ta", "female", "CC-BY-4.0", 64),
    "ta_IN-rasa_male-medium": (f"{HF}/tinisoft/piper-ta_IN-rasa_male-medium/resolve/main", "ta", "male", "CC-BY-4.0", 64),
    "en_US-ljspeech-medium": (f"{HF}/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/medium", "en", "female", "MIT model, public-domain data", 64),
    "en_US-kristin-medium": (f"{HF}/rhasspy/piper-voices/resolve/main/en/en_US/kristin/medium", "en", "female", "MIT model, public-domain data", 64),
    "en_US-lessac-medium": (f"{HF}/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium", "en", "female", "MIT model, Blizzard-2013 data (non-commercial)", 64),
}


@register
class PiperTTS(TTSModule):
    info = ModuleInfo(
        id="tts/piper", kind="tts", name="Piper (VITS, ONNX)",
        description="Fast CPU VITS voices. Tamil: AI4Bharat Rasa voices (eSpeak 'ta' phonemes; English words inside "
                    "Tamil text are phonemized with English rules, so Tanglish works in one voice). One audio piece per sentence.",
        model_id="tinisoft/piper-ta_IN-rasa_female-medium", size_mb=192, ram_mb=330,
        license="GPL-3.0 runtime (piper-tts) + per-voice model licence", streaming=True, native_rate=22050,
        requires=["piper", "onnxruntime"], family="piper", languages=["ta", "en", "ta-Latn"], tags=["gpl-runtime"],
        voices=[VoiceInfo(v, v.replace("-medium", "").replace("_", " "), lang, g) for v, (_, lang, g, _, _) in VOICES.items()],
        params_schema=schema(
            voices=("array", ["ta_IN-rasa_female-medium", "ta_IN-rasa_male-medium", "en_US-ljspeech-medium"],
                    {"items": {"type": "string"}, "description": f"Voices to download and load ({', '.join(VOICES)})"}),
            threads=("integer", 4, {"min": 1, "max": 16, "description": "onnxruntime intra-op threads (measured best on the proxy box: 2-4; 8 is slower)"}),
            noise_scale=("number", 0.667, {"min": 0.0, "max": 1.5, "advanced": True}),
            noise_w=("number", 0.8, {"min": 0.0, "max": 1.5, "advanced": True}),
            sentence_silence_ms=("integer", 0, {"min": 0, "max": 1000, "unit": "ms", "description": "Extra silence between sentences inside a chunk"}),
        ),
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.loaded_voices: dict = {}

    def _voices(self) -> list[str]:
        return [v for v in self.params["voices"] if v in VOICES]

    def model_files(self) -> list[ModelFile]:
        out = []
        for v in self._voices():
            base, _, _, _, size = VOICES[v]
            out.append(ModelFile(f"{v}.onnx", f"{base}/{v}.onnx", f"tts/piper/{v}.onnx", size))
            out.append(ModelFile(f"{v}.onnx.json", f"{base}/{v}.onnx.json", f"tts/piper/{v}.onnx.json", 0.01))
        return out

    def load(self) -> None:
        import onnxruntime as ort
        from piper import PiperVoice

        self.loaded_voices = {}
        for v in self._voices():
            path = self.models_dir / f"{v}.onnx"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing — POST /api/models/tts/piper/download or ./run.sh models piper")
            voice = PiperVoice.load(path)
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(self.params["threads"])
            so.inter_op_num_threads = 1
            voice.session = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
            self.loaded_voices[v] = voice
        if not self.loaded_voices:
            raise ValueError("no Piper voices configured")
        # warm up: first inference allocates arenas (otherwise the first utterance pays ~100 ms)
        for voice in self.loaded_voices.values():
            for _ in voice.synthesize("ok."):
                pass
        self.loaded = True

    def unload(self) -> None:
        self.loaded_voices = {}
        self.loaded = False

    def _pick(self, voice_id: str, lang: str):
        if voice_id in self.loaded_voices:
            return self.loaded_voices[voice_id]
        want = "ta" if lang in ("ta", "ta-Latn") else "en"
        for v, inst in self.loaded_voices.items():
            if VOICES[v][1] == want:
                return inst
        return next(iter(self.loaded_voices.values()))

    def synthesize(self, state, chunk, plan, voice):
        from piper import SynthesisConfig

        pv = self._pick(voice.voice_id, chunk.lang)
        speed = max(0.3, voice.speed * plan.speed)
        cfg = SynthesisConfig(length_scale=pv.config.length_scale / speed, noise_scale=self.params["noise_scale"],
                              noise_w_scale=self.params["noise_w"], normalize_audio=True)
        gap = np.zeros(int(pv.config.sample_rate * self.params["sentence_silence_ms"] / 1000), np.float32)
        for ch in pv.synthesize(chunk.tts_text, cfg):  # one AudioChunk per sentence, yielded as soon as it exists
            pcm = ch.audio_float_array.astype(np.float32, copy=False)
            yield AudioPiece(pcm=np.concatenate([pcm, gap]) if len(gap) else pcm, sample_rate=pv.config.sample_rate)
        yield AudioPiece(pcm=np.zeros(0, np.float32), sample_rate=pv.config.sample_rate, is_last=True)
