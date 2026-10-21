"""Environment check: python, deps, modules, models, ports, server, frontend. Exit 1 on any FAIL."""
from __future__ import annotations

import importlib.util
import json
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

results: list[tuple[str, str, str]] = []


def check(name: str, status: str, detail: str = "") -> None:
    results.append((status, name, detail))


def main() -> int:
    v = sys.version_info
    check("python", "OK" if v >= (3, 11) else "FAIL", sys.version.split()[0])
    for pkg in ("fastapi", "uvicorn", "numpy", "yaml", "psutil", "websockets"):
        check(f"core dep {pkg}", "OK" if importlib.util.find_spec(pkg) else "FAIL", "" if importlib.util.find_spec(pkg) else "run ./run.sh setup")
    try:
        import psutil
        mem = psutil.virtual_memory()
        check("CPU", "OK", f"{psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads")
        check("RAM", "OK" if mem.total > 8e9 else "WARN", f"{mem.total / 1e9:.1f} GB total, {mem.available / 1e9:.1f} GB free")
        du = shutil.disk_usage(ROOT)
        check("disk", "OK" if du.free > 5e9 else "WARN", f"{du.free / 1e9:.0f} GB free")
    except ImportError:
        pass
    try:
        from app.core import registry
        from app.core.config import deep_merge, load_yaml, validate
        from app.settings import Settings
        s = Settings.from_env()
        for cls in registry.all_modules():
            ok, reason = cls.availability()
            inst = cls({})
            inst.models_dir = s.models_dir / cls.info.kind / cls.info.impl
            files = inst.model_files()
            missing = [f.name for f in files if not (s.models_dir / f.path).exists()]
            if cls.info.builtin:
                check(f"module {cls.info.id}", "OK", "builtin")
            elif not ok:
                check(f"module {cls.info.id}", "INFO", f"{reason} → ./run.sh models {cls.info.family}")
            elif missing:
                check(f"module {cls.info.id}", "WARN", f"model files missing: {', '.join(missing)} → ./run.sh models {cls.info.family}")
            else:
                check(f"module {cls.info.id}", "OK", f"{cls.info.license}")
        from app.core.fish_server import SERVER
        fs = SERVER.status()
        if fs["dir"] is None or not fs["venv_present"]:
            check("fish server", "WARN", f"no Fish S2-Pro repo/.venv found ({fs['dir']}); set TTSPIPE_FISH_DIR")
        elif not fs["checkpoint_present"]:
            check("fish server", "WARN", f"checkpoint missing in {fs['dir']}/checkpoints/s2-pro-nf4 (run that repo's ./run.sh)")
        else:
            check("fish server", "OK" if fs["health"] == "ok" else "INFO",
                  f"{fs['url']} {fs['health']}" + ("" if fs["health"] == "ok" else " (starts on demand when tts/fish_s2 loads)"))
        defaults = load_yaml(s.config)
        errors, warnings = validate(deep_merge(defaults, load_yaml(s.data_dir / "pipeline.override.yaml")))
        check("config", "FAIL" if errors else ("WARN" if warnings else "OK"), "; ".join(errors + warnings)[:300])
        for p in sorted(s.profiles_dir.glob("*.yaml")):
            e, w = validate(deep_merge(defaults, load_yaml(p).get("config") or {}))
            check(f"profile {p.stem}", "FAIL" if e else ("WARN" if w else "OK"), "; ".join(e + w)[:200])
        port = s.port
    except Exception as e:  # noqa: BLE001
        check("backend import", "FAIL", f"{type(e).__name__}: {e}")
        port = 8200
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as r:
            h = json.loads(r.read())
        check("server", "OK" if h["status"] == "ok" else "WARN", f"{h['status']} {'; '.join(h.get('reasons', []))}")
    except Exception:
        with socket.socket() as sock:
            busy = sock.connect_ex(("127.0.0.1", port)) == 0
        check("server", "WARN" if busy else "INFO", f"port {port} {'in use by something else' if busy else 'free (server not running)'}")
    node = shutil.which("node")
    check("node", "OK" if node else "FAIL", subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip() if node else "install Node 20+")
    check("frontend deps", "OK" if (ROOT / "frontend" / "node_modules").exists() else "WARN", "" if (ROOT / "frontend" / "node_modules").exists() else "run ./run.sh setup")
    width = max(len(n) for _, n, _ in results)
    for st, name, detail in results:
        print(f"[{st:>4}] {name:<{width}}  {detail}")
    return 1 if any(st == "FAIL" for st, _, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
