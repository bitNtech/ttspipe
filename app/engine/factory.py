"""`build_engine(settings) -> TTSEngine`: validate the environment, then
construct (but do not load) the FishSpeech engine.

Kept import-light on purpose: torch and fish_speech are imported inside the
functions that need them, not at module scope, so `import app.engine.factory`
stays cheap and safe on a machine with no CUDA and no vendored fork checked
out yet - which is exactly the situation the unit tests in
tests/test_engine_unit.py run under.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from app.engine.base import EngineError

if TYPE_CHECKING:
    from app.config import Settings
    from app.engine.base import TTSEngine

_CHECKPOINT_WEIGHT_NAMES = ("model.pth", "model.safetensors", "model.safetensors.index.json")


class EngineConfigError(EngineError):
    """A settings/environment problem caught before any GPU work starts."""


def build_engine(settings: "Settings") -> "TTSEngine":
    if settings.engine == "mio":
        from app.engine.mio import MioEngine

        return MioEngine(settings)
    if settings.engine != "fish":
        raise EngineConfigError(f"TTS_ENGINE={settings.engine!r}; expected fish, mio or fake")
    _ensure_vendor_on_path(settings)
    _validate_checkpoint(settings)
    _validate_precision(settings)

    from app.engine.fish import FishSpeechEngine

    return FishSpeechEngine(settings)


def _ensure_vendor_on_path(settings: "Settings") -> None:
    vendor_dir = settings.vendor_dir
    if not vendor_dir.is_dir():
        raise EngineConfigError(
            f"vendor_dir {vendor_dir} does not exist; check out the fish-speech "
            "fork (scarxity/fish-speech-int4-patch) there before the engine "
            "can load"
        )
    if not (vendor_dir / "fish_speech").is_dir():
        raise EngineConfigError(
            f"{vendor_dir} exists but has no fish_speech/ package inside it; "
            "expected a checkout of the fork's repo root, not just its source dir"
        )

    vendor_str = str(vendor_dir)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)


def _validate_checkpoint(settings: "Settings") -> None:
    ckpt = settings.checkpoint_dir
    if not ckpt.is_dir():
        raise EngineConfigError(
            f"checkpoint_dir {ckpt} does not exist; download {settings.model_repo} into it first"
        )
    if not (ckpt / "config.json").is_file():
        raise EngineConfigError(f"{ckpt} has no config.json; not a valid checkpoint directory")
    if not any((ckpt / name).exists() for name in _CHECKPOINT_WEIGHT_NAMES):
        raise EngineConfigError(
            f"{ckpt} has none of {_CHECKPOINT_WEIGHT_NAMES}; incomplete checkpoint download"
        )
    if not (ckpt / "codec.pth").is_file():
        raise EngineConfigError(f"{ckpt} has no codec.pth (the DAC codec checkpoint)")


def _detect_cuda_capability() -> "tuple[int, int] | None":
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()


def _validate_precision(settings: "Settings", capability: "tuple[int, int] | None" = None) -> None:
    """Refuse bf16 on pre-Ampere GPUs instead of letting it silently run the
    slow emulated path (or fail deep inside a kernel later).

    `capability` is normally auto-detected; unit tests inject it directly so
    the sm_75 refusal is exercised without needing an actual GTX 1650.
    """
    if settings.half or settings.device != "cuda":
        return  # fp16 requested, or not running on a CUDA device at all

    if capability is None:
        capability = _detect_cuda_capability()
    if capability is None:
        return  # no GPU visible yet; the real failure will surface at load time

    major, minor = capability
    if major < 8:
        raise EngineConfigError(
            f"TTS_HALF=false requests bf16, but this GPU is compute capability "
            f"sm_{major}{minor} - bf16 needs sm_80 (Ampere) or newer. Set "
            "TTS_HALF=true (the default) to use fp16 instead, which sm_75 "
            "(Turing / GTX 1650) supports natively."
        )
