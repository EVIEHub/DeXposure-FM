#!/usr/bin/env bash
# Connection only; training is handed off later by cloud/train.sh over SSH.
set -euo pipefail
umask 077
run_tag="${1:?run tag}"
source_sha="${2:?source SHA-256}"
exec 9>/tmp/dexposure-https-ssh.lock
flock -n 9 || exit 0
exec > >(tee -a /tmp/dexposure-https-ssh.log) 2>&1
# Host keys and sshd can appear after the image's apt/ssh setup. Publish the
# public key as soon as it exists so log-tail discovery is not racing sshd -T.
for _ in $(seq 1 120); do
  [ -s /etc/ssh/ssh_host_ed25519_key.pub ] && break
  sleep 1
done
[ -s /etc/ssh/ssh_host_ed25519_key.pub ]
printf 'DEXPOSURE_HTTPS_IDENTITY %s %s %s\n' "$run_tag" "$source_sha" \
  "$(awk '{print $1 " " $2}' /etc/ssh/ssh_host_ed25519_key.pub)"
for _ in $(seq 1 60); do
  /usr/sbin/sshd -T >/dev/null 2>&1 && break
  sleep 1
done
cf_bin=/tmp/dexposure-cloudflared-2026.9.1
cf_sha=03f1f25d1cc93b9ad6c60569d44060bc4f17ed97075760ed8cfca4b12dcd68cc
if [ ! -f "$cf_bin" ]; then
  curl -fL --connect-timeout 15 --max-time 180 \
    https://github.com/cloudflare/cloudflared/releases/download/2026.9.1/cloudflared-linux-amd64 \
    -o "$cf_bin.download"
  printf '%s  %s\n' "$cf_sha" "$cf_bin.download" | sha256sum -c -
  mv "$cf_bin.download" "$cf_bin"
fi
printf '%s  %s\n' "$cf_sha" "$cf_bin" | sha256sum -c -
chmod 700 "$cf_bin"
# Reprint near the tunnel lines so a short provider log tail still contains it.
printf 'DEXPOSURE_HTTPS_IDENTITY %s %s %s\n' "$run_tag" "$source_sha" \
  "$(awk '{print $1 " " $2}' /etc/ssh/ssh_host_ed25519_key.pub)"
# A locally provisioned anonymous Quick Tunnel avoids the shared GPU host's
# provisioning rate limit. Its secret stays outside the training/upload tree.
if [ -s /root/.dexposure_cf_quick_response ]; then
  exec python3 - "$cf_bin" <<'PY'
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import subprocess
import sys
import threading

payload = Path('/root/.dexposure_cf_quick_response').read_bytes()

class ProvisionResponse(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != '/tunnel':
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass

server = HTTPServer(('127.0.0.1', 0), ProvisionResponse)
threading.Thread(target=server.serve_forever, daemon=True).start()
raise SystemExit(subprocess.call([
    sys.argv[1], 'tunnel', '--no-autoupdate', '--edge-ip-version', '4',
    '--quick-service', f'http://127.0.0.1:{server.server_port}',
    '--url', 'ssh://localhost:22',
]))
PY
fi
exec "$cf_bin" tunnel --no-autoupdate --edge-ip-version 4 --url ssh://localhost:22
