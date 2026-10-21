"""tts/indic_mio: SPRINGLab/Indic-Mio (Qwen3-0.6B + MioCodec, 22 Indian languages + English) on the GPU.

Same wire protocol as tts/fish_s2, but against its own copy of the parent repo's server started with TTS_ENGINE=mio
(app/engine/mio.py), on TTSPIPE_MIO_URL. Mio has no voice of its own: a voice is a reference clip's speaker embedding.
The built-in Tamil voices are therefore clips made once from the Piper Rasa voices (CC-BY-4.0) and cached under
data/models/tts/indic_mio/; "clone:<id>" uses the cloned-voice library like Fish does (the transcript is ignored).
The GPU holds one of Fish S2-Pro / Indic-Mio at a time on a 4 GB card.
"""
from __future__ import annotations

import numpy as np

from app import settings
from app.core import clones
from app.core.audio import wav_bytes
from app.core.base import ModuleInfo, VoiceInfo, schema
from app.core.fish_server import MIO_SERVER
from app.core.registry import register
from app.modules.tts.fish_s2 import FishS2TTS

# builtin voice id → (Piper voice that makes its reference clip, gender)
BUILTIN = {
    "builtin_ta_female": ("ta_IN-rasa_female-medium", "female"),
    "builtin_ta_male": ("ta_IN-rasa_male-medium", "male"),
}
REF_TEXT = "வணக்கம், இது என்னுடைய குரல். நான் தமிழில் தெளிவாகப் பேசுகிறேன். இன்று வானிலை மிகவும் அருமையாக இருக்கிறது."
FISH_DEFAULT = "builtin_fish_default"  # Fish S2-Pro's own default voice, rendered once into a clip
STATIC_VOICES = [VoiceInfo(FISH_DEFAULT, "Fish S2-Pro default voice (built-in)", "multi")] + [VoiceInfo(v, f"Tamil {g} (built-in)", "ta", g) for v, (_, g) in BUILTIN.items()] + [
    VoiceInfo("server_active", "Server active voice (set on the server)", "multi")]


def builtin_clip(voice_id: str) -> bytes:
    """Reference WAV for a built-in voice, synthesised with Piper on first use and kept on disk."""
    path = settings.current.data_dir / "models" / "tts" / "indic_mio" / f"{voice_id}.wav"
    if not path.exists() and voice_id == FISH_DEFAULT:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_fish_default_clip())
    elif not path.exists():
        from piper import PiperVoice

        piper_voice, _ = BUILTIN[voice_id]

        model = settings.current.data_dir / "models" / "tts" / "piper" / f"{piper_voice}.onnx"
        if not model.exists():
            raise FileNotFoundError(f"{model} missing: download the Piper voice (tts/piper) first; it makes the "
                                    f"reference clip for {voice_id}")
        parts = list(PiperVoice.load(model).synthesize(REF_TEXT))
        pcm = np.concatenate([c.audio_float_array for c in parts]).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(wav_bytes(pcm, parts[0].sample_rate))
    return path.read_bytes()


def _fish_default_clip() -> bytes:
    """Ask the Fish server (started if needed) to say REF_TEXT in its own default voice."""
    from app.core.base import Chunk, ProsodyPlan, Voice, Word
    from app.modules.tts.fish_s2 import FishS2TTS

    fish = FishS2TTS({})
    fish.load()
    c = Chunk("ref.0", "ref", 0, [Word(REF_TEXT, REF_TEXT)], [], tts_text=REF_TEXT)
    pieces = list(fish.synthesize(None, c, ProsodyPlan(), Voice(impl="fish_s2", voice_id="model_default")))
    fish._close()
    pcm = np.concatenate([p.pcm for p in pieces if len(p.pcm)])
    return wav_bytes(pcm, pieces[0].sample_rate)


@register
class IndicMioTTS(FishS2TTS):
    server = MIO_SERVER
    info = ModuleInfo(
        id="tts/indic_mio", kind="tts", name="Indic-Mio (GPU)",
        description="0.6B Qwen3 + MioCodec TTS from SPRINGLab: Tamil, the other Indian languages and English, "
                    "code-mixed text, voice cloning from a short clip. Runs on the GPU in its own server process "
                    "(../, own .venv), started on demand. Generates a whole chunk before audio starts, so latency "
                    "is the chunk's generation time (RTF well under 1).",
        model_id="SPRINGLab/Indic-Mio", size_mb=1200, ram_mb=2500, license="Apache-2.0", streaming=True,
        native_rate=24000, requires=["msgpack", "websockets", "piper"], family="mio",
        languages=["ta", "en", "ta-Latn", "multi"], tags=["remote", "gpu", "cloning", "no-speed"],
        voices=STATIC_VOICES,
        params_schema=schema(
            autostart=("boolean", True, {"description": "Start the Indic-Mio server when this engine loads and it is not running"}),
            load_timeout_s=("integer", 300, {"min": 10, "max": 1800, "unit": "s", "description": "How long load waits for the model to come up"}),
            temperature=("number", 0.9, {"min": 0.1, "max": 1.0, "description": "Sampling temperature (lower = steadier)"}),
            top_p=("number", 0.9, {"min": 0.1, "max": 1.0}),
            seed=("integer", 42, {"min": -1, "description": "Fixed seed keeps the voice consistent across chunks (-1 = random)"}),
            timeout_s=("number", 60.0, {"min": 5, "max": 600, "unit": "s", "advanced": True, "description": "Max wait for the next audio block"}),
        ),
    )

    @classmethod
    def list_voices(cls) -> list[VoiceInfo]:
        return STATIC_VOICES + [VoiceInfo(c["voice_id"], f"{c['name']} (clone)", c.get("language") or "multi")
                                for c in clones.store().list()]

    def request(self, voice_id: str) -> dict:
        p = self.params
        req = {"format": "pcm", "temperature": float(p["temperature"]), "top_p": float(p["top_p"])}
        if int(p["seed"]) >= 0:
            req["seed"] = int(p["seed"])
        cid = clones.clone_id(voice_id)
        if cid:
            audio, text = clones.store().clip(cid)
            req["references"] = [{"audio": audio, "text": text}]
        elif voice_id != "server_active":  # a built-in voice, or any id Mio cannot resolve: Fish's default voice
            vid = voice_id if voice_id in BUILTIN else FISH_DEFAULT
            req["references"] = [{"audio": builtin_clip(vid), "text": REF_TEXT}]
        return req
