import errno
import subprocess
import sys

import numpy as np
import pytest

from probe.atlas import store
from probe.atlas.store import (AlreadyRunning, FingerprintMismatch, check_fingerprint,
                               load_latest_checkpoint, run_lock, save_checkpoint)

LOCK_SNIPPET = """
import sys
from probe.atlas.store import run_lock, AlreadyRunning
try:
    with run_lock(sys.argv[1]):
        sys.exit(0)
except AlreadyRunning:
    sys.exit(3)
"""


def raise_enospc(*a, **k):
    raise OSError(errno.ENOSPC, "No space left on device")


def test_checkpoint_roundtrip_keeps_last_two(tmp_path):
    for s in (1, 2, 3):
        save_checkpoint(tmp_path, s, {"a": np.arange(4) * s}, {"step": s})
    step, arrays, meta = load_latest_checkpoint(tmp_path)
    assert step == 3 and meta["step"] == 3 and arrays["a"][1] == 3
    assert len(list(tmp_path.glob("ckpt_*.npz"))) == 2


def test_orphaned_checkpoints_use_no_slot_and_are_deleted(tmp_path):
    a = {"x": np.arange(3)}
    save_checkpoint(tmp_path, 1, a, {})
    save_checkpoint(tmp_path, 2, a, {})
    (tmp_path / "ckpt_00000003.npz").write_bytes(b"killed before its sidecar was written")
    (tmp_path / "ckpt_00000004.npz").write_bytes(b"sidecar is garbage")
    (tmp_path / "ckpt_00000004.json").write_bytes(b"{not json")
    save_checkpoint(tmp_path, 5, a, {})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt_00000002.json", "ckpt_00000002.npz",
                                                          "ckpt_00000005.json", "ckpt_00000005.npz"]
    (tmp_path / "ckpt_00000005.npz").write_bytes(b"damaged")
    assert load_latest_checkpoint(tmp_path)[0] == 2     # two real checkpoints survived, so the fallback still exists


def test_corrupt_newest_falls_back(tmp_path):
    save_checkpoint(tmp_path, 1, {"a": np.ones(4)}, {})
    newest = save_checkpoint(tmp_path, 2, {"a": np.ones(4) * 2}, {})
    newest.write_bytes(newest.read_bytes()[:20])
    assert load_latest_checkpoint(tmp_path)[0] == 1


def test_disk_full_keeps_previous(tmp_path, monkeypatch):
    save_checkpoint(tmp_path, 1, {"a": np.ones(4)}, {})
    monkeypatch.setattr(store.os, "fsync", raise_enospc)
    with pytest.raises(OSError):
        save_checkpoint(tmp_path, 2, {"a": np.ones(4)}, {})
    monkeypatch.undo()
    assert load_latest_checkpoint(tmp_path)[0] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_fingerprint_mismatch_names_key(tmp_path):
    check_fingerprint(tmp_path / "fp.json", {"corpus": "a", "model": "m"})
    with pytest.raises(FingerprintMismatch, match="corpus"):
        check_fingerprint(tmp_path / "fp.json", {"corpus": "b", "model": "m"})


def test_second_live_lock_refused_and_dead_lock_released(tmp_path):
    p = tmp_path / "run.lock"
    with run_lock(p):
        code = subprocess.run([sys.executable, "-c", LOCK_SNIPPET, str(p)]).returncode
        assert code == 3            # the child reports AlreadyRunning
    with run_lock(p):
        pass                        # the file still exists, but no process holds it


def test_garbage_sidecar_falls_back(tmp_path):
    save_checkpoint(tmp_path, 1, {"a": np.ones(4)}, {})
    newest = save_checkpoint(tmp_path, 2, {"a": np.ones(4) * 2}, {})
    newest.with_suffix(".json").write_bytes(b"\xff\xfe\x00garbage")
    assert load_latest_checkpoint(tmp_path)[0] == 1


@pytest.mark.parametrize("junk", [b"{not json", b"\xff\xfe\x80"])
def test_corrupt_fingerprint_raises_not_overwritten(tmp_path, junk):
    fp = tmp_path / "fp.json"
    fp.write_bytes(junk)
    with pytest.raises(FingerprintMismatch, match="fp.json"):
        check_fingerprint(fp, {"corpus": "a"})
    assert fp.read_bytes() == junk
