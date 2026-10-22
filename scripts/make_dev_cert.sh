#!/usr/bin/env bash
# A self-signed certificate for testing https:// and wss:// locally.
#
#   scripts/make_dev_cert.sh [extra-hostname-or-ip ...]
#
# Writes certs/dev.crt and certs/dev.key, valid for localhost, 127.0.0.1, this
# machine's hostname and primary IP, plus any names given. Then, in .env:
#
#   TTS_SSL_CERTFILE=certs/dev.crt
#   TTS_SSL_KEYFILE=certs/dev.key
#
# Browsers will not trust it: open https://<host>:<port>/ once and accept the
# warning, and the page's wss:// connection is then allowed too. Production
# wants a real certificate, or TLS terminated at a proxy in front.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v openssl >/dev/null || { echo "openssl is required" >&2; exit 1; }

mkdir -p certs
names="DNS:localhost,IP:127.0.0.1,DNS:$(hostname)"
ip=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ -n "$ip" ]] && names="$names,IP:$ip"
for extra in "$@"; do
    if [[ "$extra" =~ ^[0-9.]+$ ]]; then names="$names,IP:$extra"; else names="$names,DNS:$extra"; fi
done

openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
    -keyout certs/dev.key -out certs/dev.crt \
    -subj "/CN=$(hostname)" -addext "subjectAltName=$names" 2>/dev/null
chmod 600 certs/dev.key
echo "wrote certs/dev.crt and certs/dev.key for: $names"
