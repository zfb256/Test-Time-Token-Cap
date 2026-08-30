"""Small, shared primitives for binding inference artifacts to one run."""

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path


REQUIRED_RUN_SIGNATURE = {
    "model",
    "model_snapshot_sha256",
    "runtime_versions",
    "prompt_template",
    "prompt_sha256",
    "store_token_ids",
    "dtype",
    "max_model_len",
    "trust_remote_code",
}


def deterministic_chain_seed(seed, problem_id, chain_index):
    """Derive a stable unsigned 64-bit seed from one chain identity."""
    if seed is None:
        raise ValueError("Deterministic sampled decoding requires a base seed")
    payload = f"{int(seed)}\0{problem_id}\0{int(chain_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def runtime_versions():
    versions = {"python": platform.python_version()}
    for package in ("torch", "transformers", "vllm"):
        versions[package] = importlib.metadata.version(package)
    return versions


def _file_state(root, files):
    state = []
    for path in files:
        stat = path.stat()
        with path.open("rb") as handle:
            head = handle.read(64 * 1024)
            if stat.st_size > len(head):
                handle.seek(max(0, stat.st_size - 64 * 1024))
                tail = handle.read(64 * 1024)
            else:
                tail = b""
        state.append([
            path.name if root.is_file() else path.relative_to(root).as_posix(),
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            hashlib.sha256(head + tail).hexdigest(),
        ])
    return state


def model_snapshot_sha256(model):
    """Hash every file in a local model snapshot, including its relative path."""
    root = Path(model).resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"Reproducible inference requires a local model snapshot: {root}"
        )
    files = [root] if root.is_file() else sorted(
        path for path in root.rglob("*") if path.is_file()
    )
    if not files:
        raise ValueError(f"Model snapshot contains no files: {root}")

    state = _file_state(root, files)
    cache_dir = Path(__file__).resolve().parent.parent / ".model_fingerprint_cache"
    cache_path = cache_dir / (
        hashlib.sha256(str(root).encode("utf-8")).hexdigest() + ".json"
    )
    if os.environ.get("FORCE_MODEL_REHASH") != "1":
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached["files"] == state:
                return cached["sha256"]
        except (OSError, ValueError, KeyError):
            pass

    digest = hashlib.sha256()
    for path in files:
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    if _file_state(root, files) != state:
        raise RuntimeError(f"Model snapshot changed while hashing: {root}")
    value = digest.hexdigest()
    cache_dir.mkdir(exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps({"files": state, "sha256": value}, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(cache_path)
    return value
