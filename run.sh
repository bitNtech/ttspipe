#!/usr/bin/env bash
# Single entrypoint: interpreter -> venv -> deps -> vendor -> model -> preflight -> launch.
#
# Every phase is idempotent and individually skippable. Re-running this script
# after the first successful run should be fast (seconds), because each phase
# checks whether its work is already done before doing it.
#
# See `./run.sh --help` for flags. `.env` is sourced first if present.

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. paths, defaults, .env, flags
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Commit on the fork's `main` branch, pinned so a clean checkout is
# reproducible. Override with TTS_VENDOR_REF for testing against a newer
# commit. Resolved with: git ls-remote https://github.com/scarxity/fish-speech-int4-patch.git main
DEFAULT_VENDOR_REF="3fd7da6a65dfeb9f8ad1975813301ed16c4da47a"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"
TTS_HOST="${TTS_HOST:-0.0.0.0}"
TTS_PORT="${TTS_PORT:-8080}"
TTS_CHECKPOINT_DIR="${TTS_CHECKPOINT_DIR:-checkpoints/s2-pro-nf4}"
TTS_VENDOR_DIR="${TTS_VENDOR_DIR:-vendor/fish-speech}"
TTS_MODEL_REPO="${TTS_MODEL_REPO:-scarxity/fish-speech-s2-pro-nf4}"
TTS_VENDOR_REF="${TTS_VENDOR_REF:-$DEFAULT_VENDOR_REF}"
TTS_SKIP_INSTALL="${TTS_SKIP_INSTALL:-0}"
TTS_SKIP_DOWNLOAD="${TTS_SKIP_DOWNLOAD:-0}"
TTS_SKIP_PREFLIGHT="${TTS_SKIP_PREFLIGHT:-0}"
# Dev extras (pytest et al) are installed by default: the test suite is part of
# how you verify a deployment, and requirements-dev.txt already pulls in
# requirements.txt, so this stays a single resolve.
TTS_INSTALL_DEV="${TTS_INSTALL_DEV:-1}"
HF_HOME="${HF_HOME:-$SCRIPT_DIR/.hf-home}"
export HF_HOME

SKIP_INSTALL=0
SKIP_DOWNLOAD=0
SKIP_PREFLIGHT=0
PREFLIGHT_ONLY=0

[[ "$TTS_SKIP_INSTALL" == "1" || "$TTS_SKIP_INSTALL" == "true" ]] && SKIP_INSTALL=1
[[ "$TTS_SKIP_DOWNLOAD" == "1" || "$TTS_SKIP_DOWNLOAD" == "true" ]] && SKIP_DOWNLOAD=1
[[ "$TTS_SKIP_PREFLIGHT" == "1" || "$TTS_SKIP_PREFLIGHT" == "true" ]] && SKIP_PREFLIGHT=1

usage() {
    cat <<EOF
Usage: ./run.sh [--skip-install] [--skip-download] [--skip-preflight] [--preflight-only] [--help]

Bootstraps the venv, dependencies, vendored fish-speech fork and model
checkpoint (each idempotent), runs a GPU/import preflight check, then
launches the server with uvicorn.

Flags:
  --skip-install     Skip the dependency install phase (assumes venv is ready)
  --skip-download    Skip the model download phase (assumes checkpoint is present)
  --skip-preflight   Skip preflight checks (e.g. when using TTS_ENGINE=fake)
  --preflight-only   Run phases 1-6 (through preflight) and exit; do not launch
  --help             Show this help and exit

Environment variables (same names .env / app/config.py use):
  PYTHON_BIN            Interpreter to build the venv with (else probed / uv-installed)
  VENV_DIR               Default: .venv
  TTS_HOST                Default: 0.0.0.0
  TTS_PORT                Default: 8080
  TTS_CHECKPOINT_DIR      Default: checkpoints/s2-pro-nf4
  TTS_VENDOR_DIR          Default: vendor/fish-speech
  TTS_MODEL_REPO          Default: scarxity/fish-speech-s2-pro-nf4
  TTS_VENDOR_REF          Fork commit to pin to. Default: $DEFAULT_VENDOR_REF
  TTS_SKIP_INSTALL        1/true to skip deps install (same as --skip-install)
  TTS_SKIP_DOWNLOAD       1/true to skip model download (same as --skip-download)
  TTS_SKIP_PREFLIGHT      1/true to skip preflight (same as --skip-preflight)
  TTS_INSTALL_DEV         0 to install runtime deps only (no pytest). Default: 1
  TTS_DOWNLOAD_RETRIES    Checkpoint download attempts (resumable). Default: 6
  TTS_DISABLE_XET         1 to bypass HF's Xet backend (slower, more reliable)
  HF_HOME                 Default: $SCRIPT_DIR/.hf-home
EOF
}

for arg in "$@"; do
    case "$arg" in
        --skip-install) SKIP_INSTALL=1 ;;
        --skip-download) SKIP_DOWNLOAD=1 ;;
        --skip-preflight) SKIP_PREFLIGHT=1 ;;
        --preflight-only) PREFLIGHT_ONLY=1 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "run.sh: unknown argument: $arg" >&2; usage; exit 2 ;;
    esac
done

log() { echo "[run.sh] $*" >&2; }
die() { echo "[run.sh] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. interpreter: $PYTHON_BIN, else python3.12/3.11/3.10 on PATH, else uv
# ---------------------------------------------------------------------------

pick_interpreter() {
    if [[ -n "$PYTHON_BIN" ]]; then
        [[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN=$PYTHON_BIN is not executable"
        echo "$PYTHON_BIN"
        return
    fi

    for cand in python3.12 python3.11 python3.10; do
        if command -v "$cand" >/dev/null 2>&1; then
            command -v "$cand"
            return
        fi
    done

    log "no system python3.10-3.12 found on PATH; bootstrapping uv to fetch one"
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh || die "failed to install uv"
        export PATH="$HOME/.local/bin:$PATH"
    fi
    command -v uv >/dev/null 2>&1 || die "uv install reported success but 'uv' is not on PATH"

    uv python install 3.12 || die "uv failed to install python 3.12"

    # `uv python find` prefers a discovered project venv, which on a rerun is
    # the very venv we are about to (re)build -- and on a half-built one that
    # is a circular, possibly broken answer. Take the managed install directly
    # and only fall back to `find`, rejecting anything inside VENV_DIR.
    local py=""
    local cand
    for cand in "$HOME"/.local/share/uv/python/cpython-3.12.*/bin/python3.12; do
        if [[ -x "$cand" ]]; then
            py="$cand"
            break
        fi
    done

    if [[ -z "$py" ]]; then
        py="$(UV_NO_CONFIG=1 uv python find 3.12 2>/dev/null || true)"
        local venv_abs
        venv_abs="$(cd "$SCRIPT_DIR" && printf '%s/%s' "$PWD" "$VENV_DIR")"
        if [[ -n "$py" && "$py" == "$venv_abs"/* ]]; then
            die "uv resolved python 3.12 to the project venv ($py); remove $VENV_DIR and rerun, or set PYTHON_BIN"
        fi
    fi

    [[ -n "$py" && -x "$py" ]] || die "uv could not locate the python 3.12 it just installed"
    echo "$py"
}

if [[ -z "${_RUN_SH_PY:-}" ]]; then
    _RUN_SH_PY="$(pick_interpreter)"
fi
log "using interpreter: $_RUN_SH_PY ($("$_RUN_SH_PY" --version 2>&1))"

PYVER="$("$_RUN_SH_PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PYVER" in
    3.10|3.11|3.12) ;;
    *) die "interpreter $_RUN_SH_PY is python $PYVER; torch 2.8.0 needs 3.10-3.12" ;;
esac

# ---------------------------------------------------------------------------
# 2. venv
# ---------------------------------------------------------------------------

IS_WINDOWS=0
case "${OSTYPE:-}" in msys*|cygwin*|win32) IS_WINDOWS=1 ;; esac

if [[ "$IS_WINDOWS" == "1" ]]; then
    if [[ ! -x "$VENV_DIR/Scripts/python.exe" ]]; then
        log "creating venv at $VENV_DIR"
        "$_RUN_SH_PY" -m venv "$VENV_DIR" || die "python -m venv failed"
    fi
    VENV_BIN="$VENV_DIR/Scripts"
    VENV_PY="$VENV_BIN/python.exe"
else
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        log "creating venv at $VENV_DIR"
        "$_RUN_SH_PY" -m venv "$VENV_DIR" || die "python -m venv failed"
    fi
    VENV_BIN="$VENV_DIR/bin"
    VENV_PY="$VENV_BIN/python"
fi

if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
    log "venv has no pip; bootstrapping via ensurepip"
    "$VENV_PY" -m ensurepip --upgrade || die "ensurepip failed; venv's base interpreter may lack pip entirely"
fi

# ---------------------------------------------------------------------------
# 3. deps (stamp-gated so reruns are fast)
# ---------------------------------------------------------------------------

STAMP_FILE="$VENV_DIR/.deps-stamp"
REQS_HASH="$( { cat requirements.txt requirements-dev.txt 2>/dev/null; echo "dev=$TTS_INSTALL_DEV"; } | sha256sum | cut -d' ' -f1)"

if [[ "$SKIP_INSTALL" == "1" ]]; then
    log "skipping dependency install (--skip-install / TTS_SKIP_INSTALL)"
elif [[ -f "$STAMP_FILE" && "$(cat "$STAMP_FILE" 2>/dev/null)" == "$REQS_HASH" ]]; then
    log "dependencies already match requirements.txt (stamp hit); skipping install"
else
    log "installing dependencies into $VENV_DIR (this can take a while on first run)"
    "$VENV_PY" -m pip install --upgrade pip || die "pip upgrade failed"
    REQ_FILE=requirements.txt
    if [[ "$TTS_INSTALL_DEV" == "1" || "$TTS_INSTALL_DEV" == "true" ]]; then
        # requirements-dev.txt begins with `-r requirements.txt`, so this is
        # one resolve, not two.
        REQ_FILE=requirements-dev.txt
    fi
    "$VENV_PY" -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -r "$REQ_FILE" \
        || die "pip install -r $REQ_FILE failed"
    echo "$REQS_HASH" > "$STAMP_FILE"
    log "dependency install complete"
fi

# ---------------------------------------------------------------------------
# 4. vendor: clone fork at pinned commit (never installed, only sys.path'd)
# ---------------------------------------------------------------------------

if [[ -d "$TTS_VENDOR_DIR/.git" ]]; then
    CURRENT_SHA="$(git -C "$TTS_VENDOR_DIR" rev-parse HEAD 2>/dev/null || echo "")"
    if [[ "$CURRENT_SHA" == "$TTS_VENDOR_REF" ]]; then
        log "vendor fork already at pinned commit $TTS_VENDOR_REF; skipping"
    else
        log "updating vendor fork at $TTS_VENDOR_DIR to $TTS_VENDOR_REF"
        git -C "$TTS_VENDOR_DIR" fetch --depth 1 origin "$TTS_VENDOR_REF" 2>/dev/null \
            || git -C "$TTS_VENDOR_DIR" fetch origin \
            || die "git fetch failed in $TTS_VENDOR_DIR"
        git -C "$TTS_VENDOR_DIR" checkout --detach "$TTS_VENDOR_REF" \
            || die "git checkout $TTS_VENDOR_REF failed in $TTS_VENDOR_DIR"
    fi
else
    log "cloning fork to $TTS_VENDOR_DIR"
    mkdir -p "$(dirname "$TTS_VENDOR_DIR")"
    rm -rf "$TTS_VENDOR_DIR"
    git clone https://github.com/scarxity/fish-speech-int4-patch.git "$TTS_VENDOR_DIR" \
        || die "git clone of fork failed"
    git -C "$TTS_VENDOR_DIR" checkout --detach "$TTS_VENDOR_REF" \
        || die "git checkout $TTS_VENDOR_REF failed after clone"
fi

# ---------------------------------------------------------------------------
# 5. model download
# ---------------------------------------------------------------------------

REQUIRED_CKPT_FILES=(model.pth codec.pth config.json tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja)

ckpt_complete() {
    [[ -d "$TTS_CHECKPOINT_DIR" ]] || return 1
    local f
    for f in "${REQUIRED_CKPT_FILES[@]}"; do
        [[ -s "$TTS_CHECKPOINT_DIR/$f" ]] || return 1
    done
    return 0
}

if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    log "skipping model download (--skip-download / TTS_SKIP_DOWNLOAD)"
elif ckpt_complete; then
    log "checkpoint already complete at $TTS_CHECKPOINT_DIR; skipping download"
else
    log "downloading $TTS_MODEL_REPO to $TTS_CHECKPOINT_DIR (~4.9 GB, be patient)"
    mkdir -p "$TTS_CHECKPOINT_DIR"

    # A multi-gigabyte download over a consumer link will drop connections.
    # snapshot_download resumes what it already has, so a retry is cheap --
    # but each attempt trades throughput for robustness:
    #   1. hf_transfer + Xet   : fastest (measured ~4.4 MB/s here)
    #   2. hf_transfer off     : plain resumable HTTP, far more forgiving
    #   3. Xet off as well     : classic CDN, the most conservative path
    TTS_DOWNLOAD_RETRIES="${TTS_DOWNLOAD_RETRIES:-6}"
    download_ok=0
    for attempt in $(seq 1 "$TTS_DOWNLOAD_RETRIES"); do
        case "$attempt" in
            1)
                if "$VENV_PY" -c "import hf_transfer" >/dev/null 2>&1; then
                    export HF_HUB_ENABLE_HF_TRANSFER=1
                else
                    unset HF_HUB_ENABLE_HF_TRANSFER || true
                fi
                # Xet is the fastest path when it works, but it has proven
                # flaky on some links; TTS_DISABLE_XET=1 skips straight to
                # the classic CDN instead of discovering that by failing.
                if [[ "${TTS_DISABLE_XET:-0}" == "1" ]]; then
                    export HF_HUB_DISABLE_XET=1
                else
                    unset HF_HUB_DISABLE_XET || true
                fi
                ;;
            2)
                log "retrying without hf_transfer (slower, but tolerates flaky links)"
                unset HF_HUB_ENABLE_HF_TRANSFER || true
                ;;
            *)
                log "retrying with the Xet backend disabled (classic CDN)"
                unset HF_HUB_ENABLE_HF_TRANSFER || true
                export HF_HUB_DISABLE_XET=1
                ;;
        esac

        if "$VENV_PY" - "$TTS_MODEL_REPO" "$TTS_CHECKPOINT_DIR" <<'PYINNER'
import sys
from huggingface_hub import snapshot_download

repo, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=repo,
    local_dir=local_dir,
    max_workers=4,
    etag_timeout=30,
)
print("download complete:", local_dir)
PYINNER
        then
            download_ok=1
            break
        fi

        if [[ "$attempt" -lt "$TTS_DOWNLOAD_RETRIES" ]]; then
            backoff=$(( attempt * 10 ))
            log "download attempt $attempt/$TTS_DOWNLOAD_RETRIES failed; resuming in ${backoff}s"
            sleep "$backoff"
        fi
    done

    if [[ "$download_ok" != "1" ]]; then
        die "model download failed after $TTS_DOWNLOAD_RETRIES attempts. The partial download is kept at $TTS_CHECKPOINT_DIR and resumes on the next run."
    fi

    ckpt_complete || die "download reported success but $TTS_CHECKPOINT_DIR is still missing required files"
fi

# ---------------------------------------------------------------------------
# 6. preflight
# ---------------------------------------------------------------------------

if [[ "$SKIP_PREFLIGHT" == "1" ]]; then
    log "skipping preflight checks (--skip-preflight / TTS_SKIP_PREFLIGHT)"
else
    log "running preflight checks"
    TTS_VENDOR_DIR="$TTS_VENDOR_DIR" TTS_CHECKPOINT_DIR="$TTS_CHECKPOINT_DIR" \
        "$VENV_PY" scripts/preflight.py || die "preflight checks failed (see above)"
fi

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    log "--preflight-only: not launching the server"
    exit 0
fi

# ---------------------------------------------------------------------------
# 7. launch
# ---------------------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# Windows python splits PYTHONPATH on ';', not ':'.
case "${OSTYPE:-}" in msys*|cygwin*|win32) PATHSEP=';' ;; *) PATHSEP=':' ;; esac
export PYTHONPATH="$TTS_VENDOR_DIR${PYTHONPATH:+$PATHSEP$PYTHONPATH}"

log "launching uvicorn on $TTS_HOST:$TTS_PORT"
exec "$VENV_BIN/uvicorn" app.main:app --host "$TTS_HOST" --port "$TTS_PORT"
