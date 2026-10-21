"""AI4Bharat Indic-TTS Tamil (FastPitch + HiFi-GAN, coqui-tts runtime). Whole-utterance.
INDICTTS_SPK env picks speaker (male/female, see speakers.pth). Checkpoints: github release v1-checkpoints-release/ta.zip.
"""
import os

from bench_common import ROOT, run

D = ROOT / "data/models/indic-tts/ta"
SPK = os.environ.get("INDICTTS_SPK", "female")


def load():
    import torch
    from TTS.utils.synthesizer import Synthesizer

    syn = Synthesizer(
        tts_checkpoint=str(D / "fastpitch/best_model.pth"), tts_config_path=str(D / "fastpitch/config.local.json"),
        tts_speakers_file=str(D / "fastpitch/speakers.pth"),
        vocoder_checkpoint=str(D / "hifigan/best_model.pth"), vocoder_config=str(D / "hifigan/config.json"),
        use_cuda=False,
    )
    return syn, torch


def synth(ctx, text, lang):
    syn, torch = ctx
    with torch.inference_mode():
        wav = syn.tts(text, speaker_name=SPK)
    return wav, syn.output_sample_rate


def extra(ctx):
    syn, _ = ctx
    from bench_common import TEXTS

    tk = syn.tts_model.tokenizer
    out = {"speakers": list(syn.tts_model.speaker_manager.name_to_id) if syn.tts_model.speaker_manager else None}
    for k, (_, t) in TEXTS.items():
        ids = tk.text_to_ids(t)
        out[f"tok_{k}"] = tk.decode(ids)
    return out


if __name__ == "__main__":
    run("indic-tts-fastpitch", load, synth, extra=extra)
