"""ai4bharat/indic-parler-tts (Apache-2.0) via parler_tts. Run in .venv-research-indic (transformers==4.46.1 pin).
Weights: sha256-identical ungated mirror RXD03/indic-parler-tts (upstream is click-through gated).
stream = ParlerTTSStreamer (true incremental DAC decode every PLAY_S seconds of audio).
"""
import os
import threading

from bench_common import ROOT, run

MODEL = ROOT / "data/models/indic-parler-tts"
DESC = {
    "ta": "Jaya speaks with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.",
    "ta-latn": "Jaya speaks with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.",
    "en": "Mary speaks with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.",
}
PLAY_S = float(os.environ.get("PARLER_PLAY_S", "0.5"))


def load():
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL, attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)  # google/flan-t5-large tokenizer
    return model, tok, dtok, torch


def _inputs(ctx, text, lang):
    model, tok, dtok, torch = ctx
    d = dtok(DESC[lang], return_tensors="pt")
    p = tok(text, return_tensors="pt")
    return dict(input_ids=d.input_ids, attention_mask=d.attention_mask,
                prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)


def synth(ctx, text, lang):
    model, tok, dtok, torch = ctx
    torch.manual_seed(0)
    with torch.inference_mode():
        out = model.generate(**_inputs(ctx, text, lang))
    return out.cpu().numpy().squeeze(), model.config.sampling_rate


def stream(ctx, text, lang):
    from parler_tts import ParlerTTSStreamer
    from transformers import StoppingCriteria, StoppingCriteriaList

    model, tok, dtok, torch = ctx
    stop = threading.Event()

    class Stop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            return stop.is_set()

    play_steps = int(model.audio_encoder.config.frame_rate * PLAY_S)
    streamer = ParlerTTSStreamer(model, device="cpu", play_steps=play_steps)

    def gen():
        torch.manual_seed(0)
        with torch.inference_mode():
            model.generate(**_inputs(ctx, text, lang), streamer=streamer, stopping_criteria=StoppingCriteriaList([Stop()]))

    th = threading.Thread(target=gen)
    th.start()
    try:
        for chunk in streamer:
            if chunk.shape[0]:
                yield chunk
    finally:
        stop.set()
        th.join()


def extra(ctx):
    model, tok, dtok, _ = ctx
    from bench_common import TEXTS

    vocab = tok.get_vocab()
    out = {"sampling_rate": model.config.sampling_rate, "prompt_vocab": len(vocab), "frame_rate": model.audio_encoder.config.frame_rate}
    for k, (_, t) in TEXTS.items():
        ids = tok(t).input_ids
        out[f"tok_{k}"] = {"n": len(ids), "unk": sum(i == tok.unk_token_id for i in ids), "pieces": tok.convert_ids_to_tokens(ids)}
    return out


if __name__ == "__main__":
    run("indic-parler-tts", load, synth, stream=stream, extra=extra)
