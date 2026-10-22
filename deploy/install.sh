#!/usr/bin/env bash
# Install the TTS server as a systemd service that starts on every boot and
# restarts if it exits, with Caddy in front of it on ports 80/443.
#
#   sudo deploy/install.sh
#
# Idempotent: rerun after changing TTS_PUBLIC_DOMAIN in .env or editing a
# template. Run ./run.sh once by hand first so the venv, vendor checkout and
# checkpoint exist; the service only launches.

set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "install.sh: run with sudo" >&2; exit 1; }

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${RUN_USER:-${SUDO_USER:-$(stat -c %U "$APP_DIR")}}"

[[ -x "$APP_DIR/.venv/bin/uvicorn" ]] || { echo "install.sh: no venv at $APP_DIR/.venv; run ./run.sh once first" >&2; exit 1; }

env_value() { sed -n "s/^$1=//p" "$APP_DIR/.env" 2>/dev/null | tail -n1; }
DOMAIN="$(env_value TTS_PUBLIC_DOMAIN)"
PORT="$(env_value TTS_PORT)"
PORT="${PORT:-8080}"

if ! command -v caddy >/dev/null 2>&1; then
    echo "install.sh: installing caddy"
    apt-get update -qq
    apt-get install -y -qq caddy
fi

install -d -o "$RUN_USER" -g "$(id -gn "$RUN_USER")" "$APP_DIR/.cache"
sed -e "s|@APP_DIR@|$APP_DIR|g" -e "s|@USER@|$RUN_USER|g" \
    "$APP_DIR/deploy/aica-tts.service" > /etc/systemd/system/aica-tts.service
sed -e "s|@SITE@|${DOMAIN:-:80}|g" -e "s|@PORT@|$PORT|g" \
    "$APP_DIR/deploy/Caddyfile" > /etc/caddy/Caddyfile
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null

systemctl daemon-reload
systemctl enable aica-tts.service caddy.service
systemctl reload-or-restart caddy.service

echo "install.sh: installed for user $RUN_USER in $APP_DIR, serving ${DOMAIN:-plain http on :80}"
echo "  start now:   sudo systemctl restart aica-tts"
echo "  logs:        journalctl -u aica-tts -u caddy -f"
