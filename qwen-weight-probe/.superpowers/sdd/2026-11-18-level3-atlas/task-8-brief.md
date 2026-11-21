### Task 8: The recording pass

**Files:**
- Create: `src/probe/atlas/record.py`
- Test: `tests/atlas/test_record.py`

**Interfaces:**
- Consumes: `iter_docs`, `activations.*`, `LayerRecorder`, `finalize`, `store`, `Status`
- Produces:
  - `plan_batches(cfg, docs: Iterable[tuple[int, np.ndarray]], limit_tokens: int | None) -> list[list[int]]`: sorted by length within windows of 256 documents, at most `batch_docs` documents and `batch_tokens` padded tokens per batch; deterministic.
  - `record_pass(cfg, model, tok, status, out: Path, layers: Sequence[int], limit_tokens: int | None = None) -> None`: checkpoints all recorder states to `out/ckpt` at batch boundaries every `checkpoint_s` seconds, and resumes from `load_latest_checkpoint`. On completion writes `out/L{layer}.npz` and `out/summary.npz`.
  - `edited_neurons(run: str = "p50c") -> dict[int, np.ndarray]`: columns of `down_proj` that differ from the base model in L15 and L19.
  - `record_p50c(cfg, status) -> None`: runs `record_pass` on `load_model("p50c")` for layers 15 and 19 over the first `p50c_tokens`, into `root/record_p50c`, and keeps only the edited columns.

- [ ] **Step 1: Write the failing tests**

```python
def test_batches_respect_limits_and_cover_all(tiny_cfg):
    docs = [(i, np.ones(n, int)) for i, n in enumerate(rng_lengths(500))]
    b = plan_batches(tiny_cfg, docs, None)
    assert sorted(x for g in b for x in g) == list(range(500))
    assert all(len(g) <= 32 and max_len(g, docs) * len(g) <= 65_536 for g in b)

def test_edited_neurons_are_64_per_layer():
    e = edited_neurons("p50c")
    assert sorted(e) == [15, 19] and all(len(v) == 64 for v in e.values())

@pytest.mark.gpu
def test_killed_recording_resumes_to_same_result(tiny_corpus):
    clean = run_record_subprocess(tiny_corpus, "clean")
    killed = run_record_subprocess(tiny_corpus, "killed", sigkill_after_checkpoints=2)
    resumed = run_record_subprocess(tiny_corpus, "killed")
    assert killed.returncode == -9 and resumed.returncode == 0
    assert_recordings_equal(clean.out, resumed.out, rtol=1e-3)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_record.py -v'`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `record.py`**

Show measured tokens per second and the projected total through `Status` after the first 100 batches.

- [ ] **Step 4: Run the tests and confirm they pass**

Expected: 3 passed

---

