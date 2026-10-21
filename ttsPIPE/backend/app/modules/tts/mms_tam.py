"""tts/mms_tam: Meta MMS-TTS Tamil (VITS) via transformers. NON-COMMERCIAL licence. See docs/research/mms-tts-tam.md."""
from __future__ import annotations

import numpy as np

from app.core.base import AudioPiece, ModelFile, ModuleInfo, TTSModule, VoiceInfo, schema
from app.core.registry import register

REPO = "https://huggingface.co/facebook/mms-tts-tam/resolve/main"
FILES = [("model.safetensors", 145), ("config.json", 0.01), ("tokenizer_config.json", 0.01), ("vocab.json", 0.01),
         ("special_tokens_map.json", 0.01)]


@register
class MmsTamTTS(TTSModule):
    info = ModuleInfo(
        id="tts/mms_tam", kind="tts", name="MMS-TTS Tamil (VITS)",
        description="Meta MMS Tamil voice. CC-BY-NC-4.0 (non-commercial). Its vocabulary has Tamil letters only: "
                    "Latin text is dropped, so romanized Tamil must be transliterated (g2p) and English routed elsewhere.",
        model_id="facebook/mms-tts-tam", size_mb=145, ram_mb=520, license="CC-BY-NC-4.0", streaming=False,
        native_rate=16000, requires=["torch", "transformers"], family="mms", languages=["ta"], tags=["non-commercial"],
        voices=[VoiceInfo("default", "MMS Tamil", "ta", "male")],
        params_schema=schema(
            threads=("integer", 4, {"min": 1, "max": 16}),
            noise_scale=("number", 0.667, {"min": 0.0, "max": 1.5, "advanced": True}),
            seed=("integer", 0, {"min": 0, "max": 2**31 - 1, "advanced": True, "description": "VITS sampling seed (deterministic output)"}),
        ),
    )

    def model_files(self) -> list[ModelFile]:
        return [ModelFile(n, f"{REPO}/{n}", f"tts/mms_tam/{n}", s) for n, s in FILES]

    def load(self) -> None:
        import torch
        from transformers import AutoTokenizer, VitsModel

        if not (self.models_dir / "model.safetensors").exists():
            raise FileNotFoundError(f"{self.models_dir} missing model — ./run.sh models mms")
        torch.set_num_threads(int(self.params["threads"]))
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.models_dir)
        self.model = VitsModel.from_pretrained(self.models_dir).eval()
        self.model.noise_scale = self.params["noise_scale"]
        self.loaded = True

    def unload(self) -> None:
        self.model = self.tok = None
        self.loaded = False

    def synthesize(self, state, chunk, plan, voice):
        ids = self.tok(chunk.tts_text, return_tensors="pt")
        rate = self.model.config.sampling_rate
        if ids["input_ids"].shape[1] == 0:  # nothing Tamil in the text
            yield AudioPiece(pcm=np.zeros(0, np.float32), sample_rate=rate, is_last=True)
            return
        self.model.speaking_rate = float(np.clip(voice.speed * plan.speed, 0.5, 2.0))
        self.torch.manual_seed(self.params["seed"])
        with self.torch.inference_mode():
            wav = self.model(**ids).waveform[0].numpy().astype(np.float32)
        yield AudioPiece(pcm=wav, sample_rate=rate, is_last=True)
