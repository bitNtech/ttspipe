#!/usr/bin/env bash
# Drain the server before stopping it: refuse new sessions, then wait for the
# open ones to finish (or a deadline), so no caller is cut off mid-sentence.
#
#   scripts/drain.sh [max_wait_seconds]      (default: TTS_DRAIN_TIMEOUT_S or 60)
#
# Reads TTS_PORT and TTS_API_KEYS (first key) from the environment, falling
# back to .env for whichever of them is unset. Exits 0 once idle or at the
# deadline; never blocks a stop forever.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -f .env ]]; then
    # Only fill in what the caller did not set: an explicit TTS_PORT must not
    # be redirected by .env onto some other instance.
    for name in TTS_PORT TTS_API_KEYS TTS_DRAIN_TIMEOUT_S; do
        if [[ -z "${!name:-}" ]]; then
            value=$(sed -n "s/^${name}=//p" .env | tail -n1)
            [[ -n "$value" ]] && export "$name=$value"
        fi
    done
fi

PORT="${TTS_PORT:-8080}"
DEADLINE="${1:-${TTS_DRAIN_TIMEOUT_S:-60}}"
KEY="${TTS_API_KEYS%%,*}"
AUTH=()
[[ -n "${KEY// /}" ]] && AUTH=(-H "Authorization: Bearer ${KEY// /}")
BASE="http://127.0.0.1:$PORT"

if ! curl -fsS -X POST "${AUTH[@]}" "$BASE/admin/drain" >/dev/null; then
    echo "drain: server not reachable on $BASE; nothing to drain" >&2
    exit 0
fi

for ((i = 0; i < DEADLINE; i++)); do
    active=$(curl -fsS "${AUTH[@]}" "$BASE/status" | sed -n 's/.*"active_sessions":\([0-9]*\).*/\1/p')
    if [[ "${active:-0}" == "0" ]]; then
        echo "drain: idle after ${i}s"
        exit 0
    fi
    sleep 1
done
echo "drain: ${active:-?} session(s) still open after ${DEADLINE}s; stopping anyway" >&2
exit 0
