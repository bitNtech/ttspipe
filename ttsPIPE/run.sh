#!/usr/bin/env bash
# ttsPIPE single entry point: ./run.sh setup | models | dev | test | stop | doctor
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
RUN_DIR="$ROOT/data/run"
mkdir -p "$RUN_DIR"
export PYTHONUTF8=1

if [[ -x .venv/Scripts/python.exe ]]; then PY="$ROOT/.venv/Scripts/python.exe"; else PY="$ROOT/.venv/bin/python"; fi
UV="$(command -v uv || true)"
[[ -z "$UV" && -x "$HOME/.local/bin/uv" ]] && UV="$HOME/.local/bin/uv"
[[ -z "$UV" && -x "$HOME/.local/bin/uv.exe" ]] && UV="$HOME/.local/bin/uv.exe"

pip_install() {
  if [[ -n "$UV" ]]; then "$UV" pip install --python "$PY" "$@"; else "$PY" -m pip install "$@"; fi
}

kill_pidfile() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  local pid; pid="$(cat "$f")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    command -v taskkill >/dev/null 2>&1 && taskkill //PID "$pid" //T //F >/dev/null 2>&1 || true
  fi
  rm -f "$f"
}

kill_port() {  # Windows: bash PIDs are not Windows PIDs, so also stop whatever listens on the port
  command -v netstat >/dev/null 2>&1 && command -v taskkill >/dev/null 2>&1 || return 0
  for pid in $(netstat -ano 2>/dev/null | awk -v p=":$1" '$2 ~ p"$" && $4 == "LISTENING" {print $5}' | sort -u); do
    taskkill //PID "$pid" //T //F >/dev/null 2>&1 || true
  done
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  setup)
    if [[ ! -x "$PY" ]]; then
      if [[ -n "$UV" ]]; then "$UV" venv --python 3.12 .venv || "$UV" venv .venv; else python3 -m venv .venv || python -m venv .venv; fi
      if [[ -x .venv/Scripts/python.exe ]]; then PY="$ROOT/.venv/Scripts/python.exe"; else PY="$ROOT/.venv/bin/python"; fi
    fi
    pip_install -r backend/requirements-dev.txt
    (cd frontend && npm install --no-audit --no-fund)
    echo "setup done. Next: ./run.sh models (optional ML engines), ./run.sh dev"
    ;;
  models)
    # ./run.sh models [family ...]   default: families used by the builtin profiles
    families=("$@")
    if [[ ${#families[@]} -eq 0 ]]; then
      mapfile -t families < <("$PY" scripts/download_models.py --families)
    fi
    for fam in "${families[@]}"; do
      f="backend/requirements/ml-$fam.txt"
      [[ -f "$f" ]] || { echo "unknown family $fam (have: $(ls backend/requirements | sed 's/ml-//;s/.txt//' | tr '\n' ' '))"; exit 1; }
      echo "== installing $f"
      pip_install -r "$f"
    done
    "$PY" scripts/download_models.py "${families[@]/#/--family=}"
    ;;
  dev)
    "$0" stop >/dev/null 2>&1 || true
    [[ -d frontend/node_modules ]] || (cd frontend && npm install --no-audit --no-fund)
    (cd backend && exec "$PY" -m uvicorn app.main:app --host "${TTSPIPE_HOST:-127.0.0.1}" --port "${TTSPIPE_PORT:-8200}" --reload --reload-dir app) &
    echo $! > "$RUN_DIR/backend.pid"
    (cd frontend && exec npx vite --port 5174 --strictPort) &
    echo $! > "$RUN_DIR/frontend.pid"
    echo "backend  http://127.0.0.1:${TTSPIPE_PORT:-8200}   (API docs /docs)"
    echo "frontend http://127.0.0.1:5174   (mock mode: /?mock=1)"
    trap '"$0" stop; exit 0' INT TERM
    wait
    ;;
  test)
    (cd backend && "$PY" -m pytest "$@")
    (cd frontend && npx tsc -b)
    echo "tests OK (pytest + tsc -b)"
    ;;
  smoke)
    "$PY" scripts/smoke.py "$@"
    ;;
  stop)
    kill_pidfile "$RUN_DIR/backend.pid"
    kill_pidfile "$RUN_DIR/frontend.pid"
    kill_port "${TTSPIPE_PORT:-8200}"
    kill_port 5174
    echo "stopped"
    ;;
  doctor)
    "$PY" scripts/doctor.py "$@"
    ;;
  *)
    echo "usage: ./run.sh setup | models [family...] | dev | test [pytest args] | smoke [--url URL] | stop | doctor"
    exit 1
    ;;
esac
