"""Download the pinned release and inputs without replacing historical weights."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files():
    manifest = json.loads((ROOT / 'reproduction/release.json').read_text())
    for item in manifest['files']:
        path = ROOT / item['local_path']
        if not path.is_file() or sha256(path) != item['sha256']:
            raise ValueError(f'Missing or mismatched file: {path}; run python -m reproduction.prepare')
    return manifest


def main():
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '300')
    from huggingface_hub import hf_hub_download
    import shutil
    manifest = json.loads((ROOT / 'reproduction/release.json').read_text())
    for item in manifest['files']:
        destination = ROOT / item['local_path']
        if destination.is_file():
            if sha256(destination) == item['sha256']:
                print('Verified:', item['local_path'])
                continue
            raise ValueError(f'Refusing to replace mismatched file: {destination}. Move it aside first.')
        cached = hf_hub_download(item['repo'], item['remote_path'], revision=item['revision'])
        if sha256(cached) != item['sha256']:
            raise ValueError(f'Download hash mismatch: {item["remote_path"]}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, destination)
        print('Downloaded and verified:', item['local_path'])
    verify_files()


if __name__ == '__main__':
    main()
