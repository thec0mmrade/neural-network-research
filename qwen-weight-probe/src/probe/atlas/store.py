"""Crash-safe storage: atomic writes, rotating checkpoints, fingerprints, run lock."""
from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class FingerprintMismatch(RuntimeError):
    pass


class AlreadyRunning(RuntimeError):
    pass


def _fsync_dir(d: Path) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_json(path, obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=1, sort_keys=True).encode())


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError):  # ValueError covers JSON and UTF-8 decode errors
        return default


def _has_valid_sidecar(npz: Path) -> bool:
    side = read_json(npz.with_suffix(".json"))
    return isinstance(side, dict) and "sha256" in side


def save_checkpoint(dir: Path, step: int, arrays: dict, meta: dict, keep: int = 2) -> Path:
    dir = Path(dir)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    data = buf.getvalue()
    npz = dir / f"ckpt_{step:08d}.npz"
    atomic_write_bytes(npz, data)
    # sidecar last: a checkpoint without a valid sidecar is ignored
    atomic_write_json(npz.with_suffix(".json"),
                      {"step": step, "sha256": hashlib.sha256(data).hexdigest(), "meta": meta})
    # a kill can leave an .npz with no usable sidecar: it can never be loaded, so it must not take the place of one that can
    found = sorted(dir.glob("ckpt_*.npz"))
    intact = [p for p in found if _has_valid_sidecar(p)]
    for old in [p for p in found if p not in intact] + intact[:-keep]:
        old.with_suffix(".json").unlink(missing_ok=True)
        old.unlink(missing_ok=True)
    return npz


def load_latest_checkpoint(dir: Path):
    """Return (step, arrays, meta) of the newest intact checkpoint, or None."""
    for npz in sorted(Path(dir).glob("ckpt_*.npz"), reverse=True):
        side = read_json(npz.with_suffix(".json"))
        if not side:
            continue
        try:
            data = npz.read_bytes()
            if hashlib.sha256(data).hexdigest() != side["sha256"]:
                continue
            with np.load(io.BytesIO(data)) as z:
                arrays = {k: z[k] for k in z.files}
        except Exception:
            continue
        return side["step"], arrays, side["meta"]
    return None


def check_fingerprint(path: Path, parts: dict) -> None:
    path = Path(path)
    if not path.exists():
        atomic_write_json(path, parts)
        return
    stored = read_json(path)
    if not isinstance(stored, dict):
        raise FingerprintMismatch(f"fingerprint file is corrupt: {path}")
    diff = sorted(k for k in set(stored) | set(parts) if stored.get(k) != parts.get(k))
    if diff:
        raise FingerprintMismatch(f"fingerprint differs in: {', '.join(diff)}")


@contextmanager
def run_lock(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+")
    try:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AlreadyRunning(str(path)) from None
        yield
    finally:
        f.close()  # closing releases the flock
