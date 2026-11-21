### Task 1: Environment, configuration and crash-safe storage

**Files:**
- Modify: `shell.nix` (add `pyarrow pytest fsspec rich` to `pyEnv`), `nix/cpu-shell.nix` (add `ps.rich`)
- Create: `pytest.ini`, `src/probe/atlas/__init__.py` (empty), `src/probe/atlas/config.py`, `src/probe/atlas/store.py`
- Test: `tests/atlas/conftest.py`, `tests/atlas/test_store.py`

**Interfaces:**
- Produces, `config.py`: frozen dataclass `Config` with fields and defaults: `root: Path = RESULTS / "07_atlas"`, `layers=(14,15,16,17,18,19,20)`, `layer_order=(15,19,18,16,17,14,20)`, `n_neurons=3584`, `total_tokens=50_000_000`, `multilingual_tokens=5_000_000`, `thinking_tokens=10_000_000`, `doc_cap=4096`, `min_doc_tokens=32`, `shard_tokens=1_000_000`, `top_k=50`, `dn_k=20`, `random_k=50`, `per_doc_cap=3`, `window=64`, `llm_window=24`, `batch_docs=32`, `batch_tokens=65_536`, `checkpoint_s=300`, `explain_batch=20`, `pilot_n=200`, `p50c_tokens=5_000_000`, `seed=7`, `claude_bin="claude"`, `effort="medium"`, `prompt_version="v1"`. Classmethod `Config.tiny(root: Path) -> Config` sets `total_tokens=200_000`, `multilingual_tokens=20_000`, `thinking_tokens=20_000`, `shard_tokens=20_000`, `pilot_n=14`, `p50c_tokens=20_000`, `checkpoint_s=1`. Function `neuron_id(cfg, layer, idx) -> int` = `cfg.layers.index(layer) * cfg.n_neurons + idx`, and `neuron_name(layer, idx) -> str` = `f"L{layer}.mlp.{idx}"`.
- Produces, `store.py`: `atomic_write_bytes(path, data: bytes)`, `atomic_write_json(path, obj)`, `read_json(path, default=None)`, `save_checkpoint(dir: Path, step: int, arrays: dict[str, np.ndarray], meta: dict, keep: int = 2) -> Path`, `load_latest_checkpoint(dir: Path) -> tuple[int, dict, dict] | None`, `check_fingerprint(path: Path, parts: dict[str, str]) -> None` (raises `FingerprintMismatch` naming the differing keys), `run_lock(path: Path)` (context manager over `fcntl.flock`, raises `AlreadyRunning`).

- [ ] **Step 1: Update the nix shells and confirm nothing compiles**

Run: `nix-shell --dry-run 2>&1 | grep -E "will be built" -A6`
Expected: only the `python3-*-env` derivation is built, with no torch, NCCL or arrow compile. If anything else appears, stop and report it.

Run: `nix-shell --run 'python -c "import pyarrow, pytest, fsspec, rich, torch; print(torch.cuda.is_available())"'`
Expected: `True`

- [ ] **Step 2: Write the failing tests** in `tests/atlas/test_store.py`

```python
def test_checkpoint_roundtrip_keeps_last_two(tmp_path):
    for s in (1, 2, 3):
        save_checkpoint(tmp_path, s, {"a": np.arange(4) * s}, {"step": s})
    step, arrays, meta = load_latest_checkpoint(tmp_path)
    assert step == 3 and meta["step"] == 3 and arrays["a"][1] == 3
    assert len(list(tmp_path.glob("ckpt_*.npz"))) == 2

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
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_store.py -v'`
Expected: FAIL with `ModuleNotFoundError: probe.atlas.store`

- [ ] **Step 4: Implement `config.py` and `store.py`**

Atomic write order: write `<name>.tmp`, flush, `os.fsync(file)`, `os.replace`, then `os.fsync(directory)`. Remove the temp file on any exception. A checkpoint is an uncompressed `.npz` plus a `.json` sidecar holding its sha256 and `meta`; the sidecar is written last, so a checkpoint without a valid sidecar is ignored.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_store.py -v'`
Expected: 5 passed

---

