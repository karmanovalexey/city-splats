"""Download and checksum COLMAP's learned feature models for offline CUDA runs."""

import hashlib
import os
import urllib.request
from pathlib import Path

from backend.app.config import REPO_ROOT

MODELS = {
    "aliked-n16rot.onnx": "39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547",
    "aliked-lightglue.onnx": "b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d",
    "bruteforce-matcher.onnx": "3c1282f96d83f5ffc861a873298d08bbe5219f59af59223f5ceab5c41a182a47",
    "vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin": "8b2f9bdc44ca7204d8543bb3adab4c03ba9336c84ef41220b5007991036f075e",
}


def main():
    root = Path(os.environ.get("SCENES_MODEL_DIR", REPO_ROOT / "tools/models"))
    root.mkdir(parents=True, exist_ok=True)
    for name, checksum in MODELS.items():
        path = root / name
        if path.exists():
            with path.open("rb") as stream:
                verified = hashlib.file_digest(stream, "sha256").hexdigest() == checksum
            if verified:
                print(f"verified {name}", flush=True)
                continue
        temporary = path.with_suffix(path.suffix + ".download")
        print(f"downloading {name}", flush=True)
        urllib.request.urlretrieve(f"https://github.com/colmap/colmap/releases/download/3.13.0/{name}", temporary)
        with temporary.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != checksum:
            temporary.unlink()
            raise RuntimeError(f"checksum mismatch for {name}")
        temporary.replace(path)


if __name__ == "__main__":
    main()
