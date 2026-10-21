#!/usr/bin/env python
"""Environment sanity check for the TTS server.

Verifies, in order:
  1. torch sees a CUDA device with compute capability >= 7.0.
  2. bitsandbytes' CUDA setup works, by round-tripping a tiny Linear4bit
     layer on GPU.
  3. the vendored fish-speech fork (vendor/fish-speech, added to sys.path)
     imports cleanly -- both the text2semantic and DAC codec entry points.
  4. every file the checkpoint needs is present and non-empty.

Exits 0 and prints a short report on success. Exits non-zero with an
actionable message identifying exactly what failed. `--json` switches the
report to a single machine-readable JSON object on stdout (still exit
non-zero on failure).

This script is invoked by run.sh (`./run.sh --preflight-only`), but it only
depends on env vars and the venv it's run under, so it can also be run
directly: `.venv/bin/python scripts/preflight.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: str) -> Path:
    raw = os.environ.get(name, default)
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


VENDOR_DIR = _env_path("TTS_VENDOR_DIR", "vendor/fish-speech")
CHECKPOINT_DIR = _env_path("TTS_CHECKPOINT_DIR", "checkpoints/s2-pro-nf4")
REQUIRED_CHECKPOINT_FILES = (
    "model.pth",
    "codec.pth",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)
MIN_COMPUTE_CAPABILITY = (7, 0)


class Check:
    def __init__(self, report: dict) -> None:
        self.report = report
        self.failures: list[str] = []

    def fail(self, key: str, message: str) -> None:
        self.report[key] = {"ok": False, "detail": message}
        self.failures.append(f"[{key}] {message}")

    def ok(self, key: str, **detail) -> None:
        self.report[key] = {"ok": True, **detail}


def check_torch_cuda(c: Check) -> "object | None":
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        c.fail(
            "torch_import",
            f"failed to import torch: {exc!r}. Was `requirements.txt` "
            "installed into this interpreter (run.sh phase 3)?",
        )
        return None

    if not torch.cuda.is_available():
        c.fail(
            "cuda_available",
            "torch.cuda.is_available() is False. Check `nvidia-smi` works, "
            "the driver matches the CUDA build torch was installed with, and "
            "that torch was installed from the cu128 index, not a CPU wheel.",
        )
        return torch

    try:
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        major, minor = torch.cuda.get_device_capability(idx)
        total_vram_mib = torch.cuda.get_device_properties(idx).total_memory / (1024 * 1024)
    except Exception as exc:
        c.fail("cuda_available", f"CUDA device query failed: {exc!r}")
        return torch

    if (major, minor) < MIN_COMPUTE_CAPABILITY:
        c.fail(
            "compute_capability",
            f"GPU compute capability {major}.{minor} is below the minimum "
            f"{MIN_COMPUTE_CAPABILITY[0]}.{MIN_COMPUTE_CAPABILITY[1]} this "
            "server requires.",
        )
    else:
        c.ok(
            "cuda_available",
            device_name=name,
            compute_capability=f"{major}.{minor}",
            total_vram_mib=round(total_vram_mib, 1),
            torch_version=torch.__version__,
        )
    return torch


def check_bitsandbytes(c: Check, torch_mod) -> None:
    if torch_mod is None or not torch_mod.cuda.is_available():
        c.fail("bitsandbytes", "skipped: no CUDA device available to test against")
        return
    try:
        import bitsandbytes as bnb
    except Exception as exc:
        c.fail(
            "bitsandbytes",
            f"failed to import bitsandbytes: {exc!r}. On sm_75 (Turing) make "
            "sure the installed wheel actually ships a CUDA binary matching "
            "the driver's CUDA runtime -- check `python -m bitsandbytes`.",
        )
        return

    try:
        layer = bnb.nn.Linear4bit(
            64, 64, bias=False, compute_dtype=torch_mod.float16, quant_type="nf4"
        ).to("cuda")
        x = torch_mod.randn(4, 64, dtype=torch_mod.float16, device="cuda")
        with torch_mod.no_grad():
            y = layer(x)
        if y.shape != (4, 64) or not torch_mod.isfinite(y).all():
            raise RuntimeError(f"unexpected output shape/values: {y.shape}")
    except Exception as exc:
        c.fail(
            "bitsandbytes",
            f"Linear4bit round-trip on GPU failed: {exc!r}. bitsandbytes "
            "likely built/shipped a CUDA kernel that doesn't match this "
            "driver/GPU (sm_75, Turing, no bf16/tensor-core NF4 fast path). "
            "Try upgrading bitsandbytes.",
        )
        return

    c.ok("bitsandbytes", version=getattr(bnb, "__version__", "unknown"))


def check_vendor_import(c: Check) -> None:
    if not VENDOR_DIR.is_dir():
        c.fail(
            "vendor_import",
            f"{VENDOR_DIR} does not exist. Run.sh phase 4 (vendor clone) "
            "has not completed.",
        )
        return

    sys.path.insert(0, str(VENDOR_DIR))
    try:
        import fish_speech.models.text2semantic.inference  # noqa: F401
        import fish_speech.models.dac.inference  # noqa: F401
    except Exception as exc:
        c.fail(
            "vendor_import",
            f"importing the fork from {VENDOR_DIR} failed: {exc!r}. This "
            "usually means a runtime dependency is missing from "
            "requirements.txt -- check the exception's own import chain.",
        )
        return

    c.ok("vendor_import", vendor_dir=str(VENDOR_DIR))


def check_checkpoint_files(c: Check) -> None:
    if not CHECKPOINT_DIR.is_dir():
        c.fail(
            "checkpoint_files",
            f"{CHECKPOINT_DIR} does not exist. Run.sh phase 5 (model "
            "download) has not completed.",
        )
        return

    missing = []
    empty = []
    for name in REQUIRED_CHECKPOINT_FILES:
        p = CHECKPOINT_DIR / name
        if not p.exists():
            missing.append(name)
        elif p.stat().st_size == 0:
            empty.append(name)

    if missing or empty:
        detail = []
        if missing:
            detail.append(f"missing: {', '.join(missing)}")
        if empty:
            detail.append(f"empty: {', '.join(empty)}")
        c.fail(
            "checkpoint_files",
            f"{CHECKPOINT_DIR} is incomplete ({'; '.join(detail)}). Re-run "
            "`./run.sh` (or delete the directory) to re-download.",
        )
        return

    c.ok("checkpoint_files", checkpoint_dir=str(CHECKPOINT_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable JSON output")
    args = parser.parse_args()

    report: dict = {}
    c = Check(report)

    engine = os.environ.get("TTS_ENGINE", "fish").strip().lower()
    if engine == "fake":
        c.ok("engine", engine="fake", detail="Running in fake/CPU tone-generator mode; GPU checks bypassed")
    else:
        torch_mod = check_torch_cuda(c)
        check_bitsandbytes(c, torch_mod)
        check_vendor_import(c)
        check_checkpoint_files(c)

    ok = not c.failures

    if args.json:
        print(json.dumps({"ok": ok, "checks": report}, indent=2))
    else:
        for key, val in report.items():
            status = "OK" if val.get("ok") else "FAIL"
            extra = {k: v for k, v in val.items() if k not in ("ok", "detail")}
            print(f"[{status}] {key}" + (f" {extra}" if extra else ""))
            if not val.get("ok"):
                print(f"       {val.get('detail')}")
        print()
        print("PREFLIGHT PASSED" if ok else "PREFLIGHT FAILED")
        if not ok:
            print()
            print("Failures:")
            for f in c.failures:
                print(f"  - {f}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
