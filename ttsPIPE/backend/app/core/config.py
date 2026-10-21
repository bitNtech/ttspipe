"""Config load / deep-merge / validate / persist (ARCHITECTURE.md §11)."""
from __future__ import annotations

import copy
import logging
from pathlib import Path

import yaml

from . import registry
from .audio import OUT_RATES
from .base import KINDS

log = logging.getLogger("ttspipe.config")

REQUIRED_STAGES = ("ingest", "chunker", "tts")
PLAYBACK_KEYS = {
    "lookahead_chunks": (int, 0, 8), "jitter_ms": (int, 0, 1000), "duck_db": (float, 0, 60),
    "duck_ramp_ms": (int, 0, 2000), "duck_release_ms": (int, 0, 60000), "stop_fade_ms": (int, 0, 1000),
    "crossfade_ms": (int, 0, 100), "max_silence_gap_ms": (int, 0, 10000), "fill_underrun_silence": (bool, None, None),
    "level_every_ticks": (int, 1, 50), "filler_enabled": (bool, None, None), "filler_after_ms": (int, 0, 10000),
    "filler_gap_ms": (int, 0, 10000),
}
CHUNKER_KEYS = {
    "first_chunk_min_words": (int, 1, 50), "first_chunk_max_words": (int, 1, 100),
    "first_chunk_max_wait_ms": (int, 0, 10000), "min_chunk_chars": (int, 1, 2000),
    "max_chunk_chars": (int, 10, 5000), "nonstreaming_max_chunk_chars": (int, 10, 5000),
}
VOICE_KEYS = {"impl": (str, None, None), "voice_id": (str, None, None), "speed": (float, 0.5, 2.0),
              "pitch": (float, -12, 12)}


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    tmp.replace(path)


def _atomic(path: tuple) -> bool:
    """Values replaced whole, never key-merged: a stage's params and the shadow TTS spec."""
    return (len(path) == 3 and path[0] == "stages" and path[2] == "params") or path == ("graph", "shadows", "tts")


def deep_merge(base: dict, over: dict, _path: tuple = ()) -> dict:
    """Dicts merge recursively, other values replace. Stage params are atomic, and changing a stage's
    impl without giving params resets them (params of one module are meaningless for another)."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        path = _path + (k,)
        if isinstance(v, dict) and isinstance(out.get(k), dict) and not _atomic(path):
            if len(path) == 2 and path[0] == "stages" and "impl" in v and v["impl"] != out[k].get("impl")                     and "params" not in v:
                out[k] = {**out[k], "params": {}}
            out[k] = deep_merge(out[k], v, path)
        else:
            out[k] = copy.deepcopy(v)
    return out


def diff(base: dict, cur: dict, _path: tuple = ()) -> dict:
    """Minimal override such that deep_merge(base, diff) == cur (keys removed in cur are not tracked)."""
    out = {}
    for k, v in cur.items():
        path = _path + (k,)
        b = base.get(k) if isinstance(base, dict) else None
        if isinstance(v, dict) and isinstance(b, dict) and not _atomic(path):
            d = diff(b, v, path)
            if d:
                out[k] = d
        elif v != b:
            out[k] = copy.deepcopy(v)
    return out


def _check_num(path: str, val, spec, errors: list[str]) -> None:
    typ, lo, hi = spec
    if typ is bool:
        if not isinstance(val, bool):
            errors.append(f"{path}: expected boolean")
        return
    if typ is str:
        if not isinstance(val, str):
            errors.append(f"{path}: expected string")
        return
    if isinstance(val, bool) or not isinstance(val, (int, float)) or (typ is int and not float(val).is_integer()):
        errors.append(f"{path}: expected {typ.__name__}")
        return
    if lo is not None and val < lo:
        errors.append(f"{path}: {val} < min {lo}")
    if hi is not None and val > hi:
        errors.append(f"{path}: {val} > max {hi}")


def validate_params(schema: dict, params: dict, prefix: str) -> list[str]:
    errors = []
    for k, v in (params or {}).items():
        p = f"{prefix}.{k}"
        if k not in schema:
            errors.append(f"{p}: unknown parameter")
            continue
        s = schema[k]
        t = s.get("type")
        ok = {
            "number": isinstance(v, (int, float)) and not isinstance(v, bool),
            "integer": isinstance(v, int) and not isinstance(v, bool),
            "string": isinstance(v, str),
            "boolean": isinstance(v, bool),
            "array": isinstance(v, list),
        }.get(t, True)
        if v is None and s.get("nullable"):
            continue
        if not ok:
            errors.append(f"{p}: expected {t}")
            continue
        if "enum" in s and v not in s["enum"]:
            errors.append(f"{p}: {v!r} not in {s['enum']}")
        if t in ("number", "integer"):
            if "min" in s and v < s["min"]:
                errors.append(f"{p}: {v} < min {s['min']}")
            if "max" in s and v > s["max"]:
                errors.append(f"{p}: {v} > max {s['max']}")
    return errors


def validate(cfg: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for a full effective config."""
    errors: list[str] = []
    warnings: list[str] = []
    audio = cfg.get("audio", {})
    if audio.get("out_rate") not in OUT_RATES:
        errors.append(f"audio.out_rate: {audio.get('out_rate')!r} not in {list(OUT_RATES)}")
    if audio.get("frame_ms") != 20:
        errors.append("audio.frame_ms: only 20 is supported")
    stages = cfg.get("stages", {})
    for kind in KINDS:
        st = stages.get(kind)
        if not isinstance(st, dict):
            errors.append(f"stages.{kind}: missing")
            continue
        if not st.get("enabled", True) and kind in REQUIRED_STAGES:
            errors.append(f"stages.{kind}.enabled: {kind} cannot be disabled")
        mid = f"{kind}/{st.get('impl')}"
        try:
            cls = registry.get(mid)
        except KeyError:
            errors.append(f"stages.{kind}.impl: unknown module {mid}")
            continue
        errors += validate_params(cls.info.params_schema, st.get("params") or {}, f"stages.{kind}.params")
        ok, reason = cls.availability()
        if not ok:
            warnings.append(f"stages.{kind}: {mid} unavailable ({reason}); will fall back to builtin")
    for k, v in (cfg.get("playback") or {}).items():
        if k not in PLAYBACK_KEYS:
            errors.append(f"playback.{k}: unknown key")
        else:
            _check_num(f"playback.{k}", v, PLAYBACK_KEYS[k], errors)
    for k, v in (cfg.get("chunker") or {}).items():
        if k not in CHUNKER_KEYS:
            errors.append(f"chunker.{k}: unknown key")
        else:
            _check_num(f"chunker.{k}", v, CHUNKER_KEYS[k], errors)
    for k, v in (cfg.get("voice") or {}).items():
        if k in VOICE_KEYS:
            _check_num(f"voice.{k}", v, VOICE_KEYS[k], errors)
    voice_impl = (cfg.get("voice") or {}).get("impl")
    tts_impl = stages.get("tts", {}).get("impl")
    if voice_impl and tts_impl and voice_impl != tts_impl and tts_impl != "lang_router":
        warnings.append(f"voice.impl {voice_impl!r} differs from stages.tts.impl {tts_impl!r}; tts voice is used")
    shadow = ((cfg.get("graph") or {}).get("shadows") or {}).get("tts")
    if shadow:
        sid = f"tts/{shadow.get('impl')}"
        try:
            scls = registry.get(sid)
            errors += validate_params(scls.info.params_schema, shadow.get("params") or {}, "graph.shadows.tts.params")
            if shadow.get("impl") == tts_impl and not shadow.get("params"):
                warnings.append("graph.shadows.tts: shadow equals primary engine")
            ok, reason = scls.availability()
            if not ok:
                warnings.append(f"graph.shadows.tts: {sid} unavailable ({reason})")
        except KeyError:
            errors.append(f"graph.shadows.tts.impl: unknown module {sid}")
    cache = cfg.get("cache") or {}
    if "max_mb" in cache:
        _check_num("cache.max_mb", cache["max_mb"], (float, 0, 4096), errors)
    return errors, warnings


def load_effective(defaults_path: Path, override_path: Path) -> tuple[dict, dict, list[str]]:
    """Return (defaults, effective, warnings). Invalid override keys are dropped with a warning."""
    defaults = load_yaml(defaults_path)
    override = load_yaml(override_path)
    warnings: list[str] = []
    eff = deep_merge(defaults, override)
    errors, _ = validate(eff)
    if errors and override:
        # drop offending keys one by one (path prefix match)
        for err in errors:
            path = err.split(":", 1)[0].split(".")
            node = override
            for p in path[:-1]:
                node = node.get(p) if isinstance(node, dict) else None
                if node is None:
                    break
            if isinstance(node, dict) and path[-1] in node:
                node.pop(path[-1])
                warnings.append(f"override dropped {'.'.join(path)} ({err})")
        eff = deep_merge(defaults, override)
        if validate(eff)[0]:
            warnings.append("override ignored entirely: still invalid after dropping keys")
            eff = copy.deepcopy(defaults)
    for w in warnings:
        log.warning(w)
    return defaults, eff, warnings
