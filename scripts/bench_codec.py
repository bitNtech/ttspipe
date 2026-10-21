"""Where the time in one incremental codec decode actually goes.

`IncrementalDecoder.drain()` is called once per emitted block, and the server's
own per-job timing puts it at ~9% of a turn's wall clock with a fixed cost per
call far larger than the ~30 ms the module docstring assumes. That fixed cost
is what decides how finely audio can be chunked: every extra block on the ramp
buys lower latency and costs one more of it.

This breaks one call into its three parts - the quantizer's `post_module`
re-run, the decoder's convolution stack over the block plus its lookback, and
the device-to-host copy - across the block sizes the emit schedule actually
uses, so the fixed part and the per-frame part can be told apart.

Needs the GPU to itself; stop the server first.

    python scripts/bench_codec.py
    python scripts/bench_codec.py --blocks 1 8 32 --repeats 5
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_codec(settings):
    sys.path.insert(0, str(settings.vendor_dir))
    import torch

    from fish_speech.models.dac.inference import load_model as load_codec

    precision = None
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        precision = torch.float16
    return load_codec(
        config_name="modded_dac_vq",
        checkpoint_path=str(settings.checkpoint_dir / "codec.pth"),
        device=settings.device,
        decode_only=True,
        precision=precision,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--history", type=int, default=200, help="frames already decoded")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from app.config import get_settings

    settings = get_settings()

    import torch

    codec = _load_codec(settings)

    from app.engine.codecstream import IncrementalDecoder

    device = settings.device
    num_codebooks = codec.quantizer.n_codebooks if hasattr(codec.quantizer, "n_codebooks") else 10
    print(f"codec loaded on {device}; num_codebooks={num_codebooks}")

    def make_codes(n: int):
        return torch.randint(0, 1024, (num_codebooks, n), device=device, dtype=torch.long)

    print(
        f"{'block':>6} {'total_ms':>9} {'post_ms':>9} {'conv_ms':>9} {'copy_ms':>9} "
        f"{'ms/frame':>9}"
    )
    rows = []
    for block in args.blocks:
        totals, posts, convs, copies = [], [], [], []
        for _ in range(args.repeats):
            decoder = IncrementalDecoder(
                codec,
                lookback_frames=settings.codec_lookback_frames,
                post_context_frames=settings.codec_post_context_frames,
                use_autocast=False,
            )
            # Prime it so the call under test looks like one in mid-utterance.
            decoder.extend(make_codes(args.history))
            decoder.drain()
            torch.cuda.synchronize()

            decoder.extend(make_codes(block))
            start = decoder.emitted_frames
            end = start + block

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                post_start = max(0, end - decoder._post_context)
                z_post = codec.quantizer.decode_codes(
                    decoder._codes[..., post_start:end].clone()
                )
                torch.cuda.synchronize()
                t1 = time.perf_counter()

                conv_start = max(post_start, start - decoder._lookback)
                context = start - conv_start
                piece = z_post[..., conv_start - post_start : end - post_start]
                audio = codec.decoder(codec.quantizer.upsample(piece))
                waveform = audio[0].squeeze()[context * 2048 :]
                torch.cuda.synchronize()
                t2 = time.perf_counter()

                waveform.float().contiguous().cpu().numpy()
                t3 = time.perf_counter()

            totals.append((t3 - t0) * 1000.0)
            posts.append((t1 - t0) * 1000.0)
            convs.append((t2 - t1) * 1000.0)
            copies.append((t3 - t2) * 1000.0)

        med = statistics.median
        rows.append((block, med(totals), med(posts), med(convs), med(copies)))
        print(
            f"{block:>6} {med(totals):>9.1f} {med(posts):>9.1f} {med(convs):>9.1f} "
            f"{med(copies):>9.1f} {med(totals) / block:>9.1f}"
        )

    if len(rows) >= 2:
        (b0, t0, *_), (b1, t1, *_) = rows[0], rows[-1]
        slope = (t1 - t0) / (b1 - b0)
        print(
            f"\nfixed ~{t0 - slope * b0:.1f} ms per call, ~{slope:.2f} ms per extra frame\n"
            f"(one frame is 46.4 ms of audio, so a call costs "
            f"{(t0 - slope * b0) / 46.4:.2f} frames' worth of playback time)"
        )


if __name__ == "__main__":
    main()
