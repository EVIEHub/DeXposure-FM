"""Narrow Vast API helpers: checked start and pinned SSH-over-HTTPS discovery.

Use Bearer headers, never credential-bearing query URLs or SDK tracebacks.
No instance creation, model training, or global SSH/network configuration here.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def require_success(payload: dict) -> dict:
    if payload.get("success") is not True:
        error = str(payload.get("error", "unknown"))
        if not re.fullmatch(r"[A-Za-z0-9_]{1,100}", error):
            error = "unknown"
        raise RuntimeError(f"Vast API rejected request: {error}")
    return payload


def api_json(method: str, path: str, payload: dict | None = None) -> dict:
    token = os.environ.get("VAST_API_KEY", "").strip()
    if not token:
        token = (Path.home() / ".config/vastai/vast_api_key").read_text().strip()
    if not token:
        raise RuntimeError("Vast API credential is missing")
    request = Request(
        "https://console.vast.ai/api/v0" + path,
        data=None if payload is None else json.dumps(payload).encode(), method=method,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def exact_worker(instance: int, run_tag: str) -> dict:
    worker = api_json("GET", f"/instances/{instance}/").get("instances")
    if (not isinstance(worker, dict) or worker.get("id") != instance or
            worker.get("label") != "dexposure-v2-" + run_tag):
        raise RuntimeError("Exact instance id/label did not match")
    return worker


def request_start(instance: int, run_tag: str) -> None:
    exact_worker(instance, run_tag)
    require_success(api_json("PUT", f"/instances/{instance}/", {"state": "running"}))
    print(f"Start request accepted for instance={instance}; running state not yet confirmed.")


def parse_transport(log: str, run_tag: str, source_sha: str) -> tuple[str, str]:
    marker = f"DEXPOSURE_HTTPS_IDENTITY {run_tag} {source_sha} "
    if marker not in log:
        raise RuntimeError("HTTPS bootstrap identity not present yet")
    # The last matching boot must supply both its host key and its tunnel.
    tail = log.rsplit(marker, 1)[1]
    match = re.match(r"ssh-ed25519 ([A-Za-z0-9+/=]+)\r?\n", tail)
    if not match:
        raise RuntimeError("HTTPS host public key is malformed")
    raw = base64.b64decode(match[1], validate=True)
    if len(raw) != 51 or not raw.startswith(b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20"):
        raise RuntimeError("HTTPS host public key is not an Ed25519 key")
    hosts = re.findall(r"https://([a-z0-9-]+\.trycloudflare\.com)(?=[\s/|])", tail)
    hosts = [host for host in hosts if host != "api.trycloudflare.com"]
    if not hosts or "Registered tunnel connection" not in tail:
        raise RuntimeError("HTTPS tunnel has not registered yet")
    return hosts[-1], "ssh-ed25519 " + match[1]


def discover(instance: int, run_tag: str, source_sha: str, output: Path, client: Path) -> None:
    worker = exact_worker(instance, run_tag)
    if worker.get("actual_status") != "running":
        raise RuntimeError("Worker is not running")
    result = require_success(api_json("PUT", f"/instances/request_logs/{instance}/", {"tail": "2000"}))
    url = result.get("result_url", "")
    parsed = urlsplit(url)
    # This is the provider's observed log store. Do not send API auth to it.
    if parsed.scheme != "https" or parsed.hostname != "s3.amazonaws.com" or parsed.username:
        raise RuntimeError("Unexpected Vast log download host")
    deadline = time.monotonic() + 35
    while True:
        try:
            with urlopen(url, timeout=10) as response:
                log = response.read().decode("utf-8", errors="replace")
            break
        except HTTPError as exc:
            if exc.code not in (403, 404) or time.monotonic() >= deadline:
                raise RuntimeError("Vast log download is not ready") from None
            time.sleep(2)
    host, public_key = parse_transport(log, run_tag, source_sha)
    output.mkdir(parents=True, exist_ok=True)
    alias = f"dexposure-vast-{instance}"
    known_hosts = output / "https_known_hosts"
    line = f"{alias} {public_key}\n"
    if known_hosts.exists() and known_hosts.read_text() != line:
        raise RuntimeError("Host key changed; refusing to replace the existing pin")
    known_hosts.write_text(line)
    known_hosts.chmod(0o600)
    config = ("Host *\n  BatchMode yes\n  StrictHostKeyChecking yes\n"
              "  HostKeyAlgorithms ssh-ed25519\n"
              f"  HostKeyAlias {alias}\n  UserKnownHostsFile {json.dumps(str(known_hosts))}\n"
              f"  ProxyCommand {shlex.quote(str(client))} access ssh --hostname {host}\n")
    target = output / "https_ssh.config"
    target.write_text(config)
    target.chmod(0o600)
    (output / "https_transport.json").write_text(json.dumps({
        "instance_id": instance, "run_tag": run_tag, "source_sha256": source_sha,
        "hostname": host, "host_public_key": public_key, "client": str(client),
        "scope": "this controller only; no global network or SSH changes",
    }, indent=2) + "\n")
    print(f"HTTPS SSH route discovered and host key pinned for instance={instance}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "discover"))
    parser.add_argument("instance", type=int)
    parser.add_argument("run_tag")
    parser.add_argument("--source-sha")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--client", type=Path)
    args = parser.parse_args()
    if args.instance <= 0 or not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_tag):
        raise RuntimeError("Invalid instance/run identity")
    if args.action == "start":
        request_start(args.instance, args.run_tag)
    else:
        if not re.fullmatch(r"[a-f0-9]{64}", args.source_sha or "") or not args.output or not args.client:
            raise RuntimeError("HTTPS discovery requires source, output and client")
        if not args.client.is_file() or not os.access(args.client, os.X_OK):
            raise RuntimeError("Local cloudflared executable is missing")
        discover(args.instance, args.run_tag, args.source_sha, args.output.resolve(), args.client.resolve())


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except Exception as exc:
        # No traceback, request headers, response bodies, or signed URLs.
        print(f"ERROR: Vast control request failed ({type(exc).__name__}, HTTP {getattr(exc, 'code', 'unknown')})", file=sys.stderr)
        raise SystemExit(1) from None
