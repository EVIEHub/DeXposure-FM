import base64
import io
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from cloud import vast_control_api as api


def test_success_false_is_an_error_even_after_http_200():
    with pytest.raises(RuntimeError, match="resources_unavailable"):
        api.require_success({"success": False, "error": "resources_unavailable"})
    with pytest.raises(RuntimeError):
        api.require_success({"msg": "not an acceptance"})
    assert api.require_success({"success": True}) == {"success": True}


def test_start_checks_exact_identity_and_api_acceptance():
    worker = {"instances": {"id": 42, "label": "dexposure-v2-test"}}
    with patch.object(api, "api_json", side_effect=[worker, {"success": False, "error": "resources_unavailable"}]) as call:
        with pytest.raises(RuntimeError, match="resources_unavailable"):
            api.request_start(42, "test")
        assert call.call_args.args == ("PUT", "/instances/42/", {"state": "running"})
    with patch.object(api, "api_json", return_value=worker) as call:
        with pytest.raises(RuntimeError, match="id/label"):
            api.request_start(42, "different")
        assert call.call_count == 1


def test_https_discovery_rejects_stale_boot_identity_and_bad_host_keys():
    source = "a" * 64
    key = base64.b64encode(b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20" + bytes(32)).decode()
    marker = f"DEXPOSURE_HTTPS_IDENTITY test {source} ssh-ed25519 {key}\n"
    connected = "https://four-small-test-words.trycloudflare.com |\nRegistered tunnel connection\n"
    log = marker + connected
    assert api.parse_transport(log, "test", source) == ("four-small-test-words.trycloudflare.com", "ssh-ed25519 " + key)
    for bad in (log + marker, log.replace(source, "b" * 64), log.replace(key, "AAAA"), marker):
        with pytest.raises(RuntimeError):
            api.parse_transport(bad, "test", source)


def test_controller_uses_one_pinned_transport_for_both_ssh_and_rsync():
    root = Path(__file__).resolve().parents[2]
    source = (root / "cloud/vast_api_train.sh").read_text()
    assert 'SSH=(ssh "${SSH_OPTIONS[@]}" "root@$SSH_HOST")' in source
    assert "printf -v SSH_TRANSPORT '%q ' ssh \"${SSH_OPTIONS[@]}\"" in source
    assert 'SSH_OPTIONS=(-F "$CONTROLLER_DIR/https_ssh.config" -o StrictHostKeyChecking=yes' in source
    assert source.index('cloud/vast_control_api.py discover') < source.index('"${SSH[@]}" true >/dev/null')


def test_log_download_retries_transient_s3_forbidden(tmp_path):
    source = "a" * 64
    key = base64.b64encode(b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20" + bytes(32)).decode()
    log = (f"DEXPOSURE_HTTPS_IDENTITY test {source} ssh-ed25519 {key}\n"
           "https://four-small-test-words.trycloudflare.com |\nRegistered tunnel connection\n")
    url = "https://s3.amazonaws.com/provider-log"
    worker = {"instances": {"id": 42, "label": "dexposure-v2-test", "actual_status": "running"}}
    with patch.object(api, "api_json", side_effect=[worker, {"success": True, "result_url": url}]), \
         patch.object(api, "urlopen", side_effect=[HTTPError(url, 403, "not ready", {}, None), io.BytesIO(log.encode())]), \
         patch.object(api.time, "sleep") as sleep:
        api.discover(42, "test", source, tmp_path, Path("/tmp/cloudflared"))
    sleep.assert_called_once_with(2)
    assert (tmp_path / "https_known_hosts").read_text() == f"dexposure-vast-42 ssh-ed25519 {key}\n"


def test_bootstrap_publishes_identity_before_sshd_config_probe():
    source = (Path(__file__).resolve().parents[2] / "cloud" / "vast_https_bootstrap.sh").read_text()
    identity = source.index("DEXPOSURE_HTTPS_IDENTITY")
    assert identity < source.index("/usr/sbin/sshd -T")
    assert "grep -qx 'passwordauthentication no'" not in source
    assert source.count("DEXPOSURE_HTTPS_IDENTITY") == 2


def test_offer_select_can_exclude_geolocation():
    source = (Path(__file__).resolve().parents[2] / "cloud/vast_api_train.sh").read_text()
    assert 'EXCLUDE_GEO="${VAST_EXCLUDE_GEO:-}"' in source
    assert '($exclude_geo == "" or ((.geolocation // "") | test($exclude_geo) | not))' in source


def test_create_bakes_fresh_quick_tunnel_into_onstart():
    source = (Path(__file__).resolve().parents[2] / "cloud/vast_api_train.sh").read_text()
    assert 'VAST_CF_QUICK_PROVISION' in source
    assert "https://api.trycloudflare.com/tunnel" in source
    assert "DEXPOSURE_PROVISION_RESPONSE" in source
    assert "Path('/root/.dexposure_cf_quick_response')" in source
    create_at = source.index('vastai create instance "$OFFER_ID"')
    assert source.index("Provisioned Quick Tunnel hostname=") < create_at
    assert source.index("nohup bash -c $HTTPS_BOOTSTRAP_Q") < create_at
    assert source.index("nohup bash -c $HTTPS_BOOTSTRAP_Q") > source.index("DEXPOSURE_PROVISION_RESPONSE")


def test_cached_provisioning_is_loopback_only_and_outside_artifacts():
    source = (Path(__file__).resolve().parents[2] / "cloud/vast_https_bootstrap.sh").read_text()
    assert "HTTPServer(('127.0.0.1', 0), ProvisionResponse)" in source
    assert "Path('/root/.dexposure_cf_quick_response').read_bytes()" in source
    assert "'--quick-service', f'http://127.0.0.1:{server.server_port}'" in source
    assert "def log_message(self, *_args):\n        pass" in source
