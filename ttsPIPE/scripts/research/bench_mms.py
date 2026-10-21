"""facebook/mms-tts-tam (VITS, CC-BY-NC-4.0) via transformers VitsModel. Whole-utterance (no streaming)."""
from bench_common import ROOT, run

MODEL = ROOT / "data/models/mms-tts-tam"


def load():
    import torch
    from transformers import AutoTokenizer, VitsModel

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = VitsModel.from_pretrained(MODEL).eval()
    return tok, model, torch


def synth(ctx, text, lang):
    tok, model, torch = ctx
    ids = tok(text, return_tensors="pt")
    if ids["input_ids"].shape[1] == 0:
        raise ValueError("tokenizer produced 0 tokens")
    torch.manual_seed(0)
    with torch.inference_mode():
        wav = model(**ids).waveform[0].numpy()
    return wav, model.config.sampling_rate


def extra(ctx):
    tok, model, _ = ctx
    from bench_common import TEXTS

    # What the tokenizer keeps vs drops (Latin coverage check for Tanglish).
    out = {"vocab_size": len(tok.get_vocab()), "latin_in_vocab": sorted(c for c in tok.get_vocab() if c.isascii() and c.isalpha())}
    for k, (_, t) in TEXTS.items():
        ids = tok(t)["input_ids"]
        kept = tok.decode(ids).replace("_", "")  # add_blank interleaves pad id
        out[f"tok_{k}"] = {"n_ids": len(ids), "decoded": kept}
    return out


if __name__ == "__main__":
    run("mms-tts-tam", load, synth, extra=extra)
