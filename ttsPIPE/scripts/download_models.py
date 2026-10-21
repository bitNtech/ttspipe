"""Download model files for ML modules into data/models/<kind>/<impl>/.

    python scripts/download_models.py                  # modules used by the builtin profiles
    python scripts/download_models.py --family=mms     # every module of a family
    python scripts/download_models.py --module tts/kokoro
    python scripts/download_models.py --all
    python scripts/download_models.py --families       # print families the builtin profiles need
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core import registry  # noqa: E402
from app.core.config import deep_merge, load_yaml  # noqa: E402
from app.core.manager import download_file  # noqa: E402
from app.settings import Settings  # noqa: E402


def profile_modules() -> set[str]:
    """Module ids referenced by config/pipeline.yaml and every builtin profile (incl. router sub-engines)."""
    defaults = load_yaml(ROOT / "config" / "pipeline.yaml")
    ids: set[str] = set()
    for p in sorted((ROOT / "config" / "profiles").glob("*.yaml")):
        cfg = deep_merge(defaults, load_yaml(p).get("config") or {})
        for kind, st in cfg["stages"].items():
            ids.add(f"{kind}/{st['impl']}")
            for key in ("ta_impl", "en_impl"):
                if (st.get("params") or {}).get(key):
                    ids.add(f"tts/{st['params'][key]}")
        sh = ((cfg.get("graph") or {}).get("shadows") or {}).get("tts")
        if sh:
            ids.add(f"tts/{sh['impl']}")
            for key in ("ta_impl", "en_impl"):
                if (sh.get("params") or {}).get(key):
                    ids.add(f"tts/{sh['params'][key]}")
    if any(i == "tts/lang_router" for i in ids):
        from app.modules.tts import lang_router  # noqa: F401  default sub-engines
        r = registry.get("tts/lang_router").info.default_params
        ids |= {f"tts/{r['ta_impl']}", f"tts/{r['en_impl']}"}
    return {i for i in ids if _exists(i)}


def _exists(mid: str) -> bool:
    try:
        registry.get(mid)
        return True
    except KeyError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", action="append", default=[])
    ap.add_argument("--module", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--families", action="store_true", help="print the ML families the builtin profiles need")
    args = ap.parse_args()
    mods = registry.all_modules()
    if args.families:
        fams = sorted({registry.get(m).info.family for m in profile_modules()} - {""})
        print("\n".join(fams))
        return 0
    if args.all:
        chosen = [m for m in mods if not m.info.builtin]
    elif args.family or args.module:
        chosen = [m for m in mods if m.info.family in args.family or m.info.id in args.module]
    else:
        wanted = profile_modules()
        chosen = [m for m in mods if m.info.id in wanted and not m.info.builtin]
    s = Settings.from_env()
    failed = 0
    for cls in chosen:
        inst = cls({})
        inst.models_dir = s.models_dir / cls.info.kind / cls.info.impl
        files = inst.model_files()
        if not files:
            print(f"{cls.info.id}: no model files")
            continue
        for f in files:
            dest = s.models_dir / f.path
            if dest.exists():
                print(f"{cls.info.id}: {f.name} present")
                continue
            print(f"{cls.info.id}: downloading {f.name} ({f.size_mb:.0f} MB) …", flush=True)
            state = {"done_mb": 0.0}
            t0 = time.time()
            try:
                download_file(f.url, dest, state, lambda: print(f"  {state['done_mb']:.0f} MB", end="\r", flush=True))
                print(f"  done {state['done_mb']:.0f} MB in {time.time() - t0:.0f}s      ")
            except Exception as e:
                failed += 1
                print(f"  FAILED: {type(e).__name__}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
