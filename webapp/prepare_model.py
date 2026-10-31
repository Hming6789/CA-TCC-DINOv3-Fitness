"""Optionally fetch a checksum-pinned model bundle during a Render build."""
import os
from pathlib import Path
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile

from .inference import file_hash

FILES = {"encoder.onnx", "head.onnx", "model.json"}
MAX_BYTES = 400 * 1024 * 1024


def install_bundle(archive_path, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        if len(entries) != 3 or {item.filename for item in entries} != FILES:
            raise ValueError("Model ZIP must contain exactly encoder.onnx, head.onnx and model.json at its root.")
        if sum(item.file_size for item in entries) > MAX_BYTES:
            raise ValueError("Uncompressed model exceeds 400 MB.")
        # Copy exact names rather than extracting arbitrary archive paths.
        with tempfile.TemporaryDirectory(dir=destination.parent) as folder:
            for name in FILES:
                with archive.open(name) as source, (Path(folder) / name).open("wb") as target:
                    shutil.copyfileobj(source, target)
            destination.mkdir(exist_ok=True)
            for name in FILES:
                os.replace(Path(folder) / name, destination / name)


def main():
    url = os.getenv("MODEL_BUNDLE_URL", "").strip()
    if not url:
        print("No MODEL_BUNDLE_URL configured. Starting in video-preview mode.")
        return
    checksum = os.getenv("MODEL_BUNDLE_SHA256", "").strip().lower()
    if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("Set MODEL_BUNDLE_SHA256 to the exported bundle SHA-256.")
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError("MODEL_BUNDLE_URL must use HTTPS.")
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "model.zip"
        with urllib.request.urlopen(url, timeout=90) as response, target.open("wb") as handle:
            if urllib.parse.urlsplit(response.url).scheme != "https":
                raise ValueError("Model download must remain on HTTPS.")
            size = 0
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > MAX_BYTES:
                    raise ValueError("Model bundle exceeds 400 MB.")
                handle.write(block)
        if file_hash(target) != checksum:
            raise ValueError("Model bundle checksum mismatch.")
        install_bundle(target, os.getenv("MODEL_DIR", "model_artifacts"))
    print("Verified model bundle installed.")


if __name__ == "__main__":
    main()
