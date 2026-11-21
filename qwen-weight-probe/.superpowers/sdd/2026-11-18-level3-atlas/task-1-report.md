# Task 1 report: environment, configuration, crash-safe storage

Status: DONE_WITH_CONCERNS (minor, see below)

## Built
- shell.nix: added pyarrow, pytest, fsspec, rich to pyEnv. nix/cpu-shell.nix: added ps.rich.
- pytest.ini (testpaths=tests, no cacheprovider, `gpu` marker registered).
- src/probe/atlas/{__init__,config,store}.py as specified in the brief (Config, Config.tiny, neuron_id, neuron_name; atomic writes, rotating checkpoints with sha256 sidecar written last, check_fingerprint, run_lock).
- tests/atlas/conftest.py (empty; no shared fixtures needed yet), tests/atlas/test_store.py (the 5 brief tests, plus imports/LOCK_SNIPPET/raise_enospc).

## Evidence
- R2: `nix-build --dry-run shell.nix`: only python3-3.13.17-env.drv and nix-shell.drv built; pyarrow/arrow-cpp etc. fetched from cache.nixos.org (48.9 MiB). No torch/NCCL compile.
- Import check: `import pyarrow, pytest, fsspec, rich, torch; torch.cuda.is_available()` printed True. cpu-shell import rich OK.
- RED (store.py temporarily removed): `ImportError: cannot import name 'store' from 'probe.atlas'`, 1 error during collection.
- GREEN: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_store.py -v'` -> 5 passed in 0.64s.
- Smoke: neuron_id(tiny,16,5)=7173, neuron_name=L16.mlp.5, Config().root = .../results/07_atlas.

## Files
Modified: shell.nix, nix/cpu-shell.nix. Created: pytest.ini, src/probe/atlas/__init__.py, config.py, store.py, tests/atlas/conftest.py, tests/atlas/test_store.py. No git used.

## Self-review / concerns
- Process slip: I wrote the implementation before the first test run, then reproduced RED by moving store.py away. Evidence is real but not strict TDD order.
- Tests were the brief's verbatim; the disk-full test patches the global os.fsync, so it covers the file-fsync failure path only.
- check_fingerprint writes the file on first call (brief implied this via its test). Checkpoint ordering relies on zero-padded step (8 digits).
- pytest.ini sets testpaths=tests and disables the cache provider (avoids .pytest_cache in the tree); easy to revert.

## Fix round
Changes: read_json now catches ValueError (JSON and UTF-8 decode), so a garbage sidecar falls back to the previous checkpoint; check_fingerprint writes only when the file is missing and raises FingerprintMismatch naming the file when it is corrupt (file left untouched); pytest.ini registers `claude` marker. Tests added: test_garbage_sidecar_falls_back, test_corrupt_fingerprint_raises_not_overwritten (2 params).
Command: nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_store.py -v'
Output:
configfile: pytest.ini
plugins: anyio-4.15.1, hypothesis-6.151.10
collecting ... collected 8 items

tests/atlas/test_store.py::test_checkpoint_roundtrip_keeps_last_two PASSED [ 12%]
tests/atlas/test_store.py::test_corrupt_newest_falls_back PASSED         [ 25%]
tests/atlas/test_store.py::test_disk_full_keeps_previous PASSED          [ 37%]
tests/atlas/test_store.py::test_fingerprint_mismatch_names_key PASSED    [ 50%]
tests/atlas/test_store.py::test_second_live_lock_refused_and_dead_lock_released PASSED [ 62%]
tests/atlas/test_store.py::test_garbage_sidecar_falls_back PASSED        [ 75%]
tests/atlas/test_store.py::test_corrupt_fingerprint_raises_not_overwritten[{not json] PASSED [ 87%]
tests/atlas/test_store.py::test_corrupt_fingerprint_raises_not_overwritten[\xff\xfe\x80] PASSED [100%]

============================== 8 passed in 0.67s ===============================
