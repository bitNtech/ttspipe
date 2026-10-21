"""ai4bharat/IndicF5 (MIT; F5-TTS flow matching + Vocos 24 kHz). Run in .venv-research-indic.
Weights: sha256-identical ungated mirror rsolanki1822/IndicF5-mirror (upstream gated click-through).
Needs a reference clip + its transcript. INDICF5_REF=pan (shipped PAN_F_HAPPY prompt, cross-lingual)
or ta (short Tamil clip; see REFS). INDICF5_NFE = flow-matching steps (default 32).
Whole-utterance generation (no incremental audio); TTFA == full synth unless the app splits sentences.
"""
import os

from bench_common import ROOT, run

M = ROOT / "data/models/indicf5"
NFE = int(os.environ.get("INDICF5_NFE", "32"))
REFS = {
    "pan": (M / "prompts/PAN_F_HAPPY_00001.wav",
            "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"),
    # Tamil ref = Piper ta_IN-rasa_female output (same Rasa speaker family IndicF5 was trained on). Synthetic, documented.
    "ta": (ROOT / "data/research_wavs/piper/short_ta-ta_IN-rasa_female-medium.wav", "வணக்கம், எப்படி இருக்கீங்க?"),
}
REF = os.environ.get("INDICF5_REF", "pan")


def load():
    import soundfile as sf
    import torch
    from safetensors.torch import load_file

    import f5_tts.infer.utils_infer as U
    from f5_tts.model import DiT

    def _sf_load(path):  # torchaudio>=2.9 load() needs torchcodec+FFmpeg DLLs; soundfile avoids that
        a, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(a.T.copy()), sr

    U.torchaudio.load = _sf_load
    voc = U.load_vocoder(vocoder_name="vocos", is_local=False, device="cpu")
    model = U.load_model(DiT, dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
                         mel_spec_type="vocos", vocab_file=str(M / "checkpoints/vocab.txt"), device="cpu")
    sd = load_file(str(M / "model.safetensors"))
    ema = {k.split("ema_model.", 1)[1].replace("_orig_mod.", ""): v for k, v in sd.items() if k.startswith("ema_model.")}
    missing, unexpected = model.load_state_dict(ema, strict=False)
    assert not [m for m in missing if "mel_stft" not in m], f"missing keys: {missing[:5]}"
    model.eval()
    ref_audio, ref_text = U.preprocess_ref_audio_text(str(REFS[REF][0]), REFS[REF][1], show_info=lambda *a: None)
    return U, model, voc, ref_audio, ref_text, torch


def synth(ctx, text, lang):
    U, model, voc, ref_audio, ref_text, torch = ctx
    torch.manual_seed(0)
    wav, sr, _ = U.infer_process(ref_audio, ref_text, text, model, voc, mel_spec_type="vocos", nfe_step=NFE,
                                 show_info=lambda *a: None, progress=None, device="cpu")
    return wav, sr


if __name__ == "__main__":
    run("indicf5", load, synth, extra=lambda ctx: {"nfe_step": NFE, "ref": REF, "ref_text": ctx[4]})
