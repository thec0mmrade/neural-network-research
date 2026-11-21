# Task 8 report: the recording pass

## Status: BLOCKED (needs a ruling on `batch_tokens`)
All of Task 8 is built and working except one required check. The R12 worst-case batch test runs out of GPU memory at the
configured `batch_tokens = 65,536`. Fixing that means changing `config.py`, which is outside this task's files, and the
brief says to use its values verbatim. I changed neither the config nor the test.

**Ruling requested:** set `batch_tokens` to 49,152 (12 x 4,096). The worst-case test passes at that value, peaking at 6.80 GiB.

## The memory finding
Full `record_pass`, 7 layers, 3 batches per size, random token ids, with the desktop holding about 3.5 GB of the card
(about 8.1 GiB left for this job):

| Batch | Padded tokens | Result | Peak allocated |
|---|---|---|---|
| 10 x 4096 | 40,960 | OK | 6.03 GiB |
| 12 x 4096 | 49,152 | OK | 6.80 GiB |
| 13 x 4096 | 53,248 | OK | 7.19 GiB |
| 14 x 4096 | 57,344 | OOM | 7.58 GiB when it failed |
| 15 x 4096 | 61,440 | OOM | 7.85 GiB when it failed |
| 16 x 4096 | 65,536 | OOM | 7.72 GiB when it failed |
| 32 x 2048 | 65,536 | OOM | 7.74 GiB when it failed |

- Cause: the forward pass, not the recorder inputs. With no recorder at all, 65,536 tokens peak at 7.6 GiB (1.4 GiB model plus
  6.2 GiB of temporaries; the OOM traceback is in the torch fallback for linear attention). A no-op capture peaks the same.
- The seven recorders add about 0.8 GiB (calculated from array sizes). Extrapolating the passing sizes gives about 8.4 GiB at
  65,536 tokens. That figure is an estimate; no run at that size completed.
- 53,248 fits but leaves under 1 GiB of margin, and desktop GPU use changes. 49,152 leaves about 1.3 GiB.

## What I built
`src/probe/atlas/record.py`
- `plan_batches`: deterministic, sorted within windows of 256 documents, enforces `batch_docs` and `batch_tokens`, stops reading
  at `limit_tokens`, and raises `ValueError` for a document that cannot fit in a batch.
- `record_pass`: resumes from the newest intact checkpoint and checkpoints at batch boundaries. Calls `update` inside the capture
  callback and keeps no tensor references (R12). Counts non-finite eligible positions. Writes `L{layer}.npz`, `record_meta.json`,
  then `summary.npz` last, and only then deletes the checkpoints. Returns at once if `summary.npz` exists.
- `edited_neurons`, `record_p50c`, `_keep_columns` (safe to repeat after a crash).

Additions the brief did not ask for:
- Checkpoints store a fingerprint of manifest hash, layers and plan. Resuming against a different one raises `FingerprintMismatch`.
- `record_meta.json` has `resumed_from`, so the tests can prove a run really resumed.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by default on import, as the project CLAUDE.md advises.

## Test evidence
RED, before `record.py` existed:
```
$ nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_record.py -v'
E   ModuleNotFoundError: No module named 'probe.atlas.record'
=============================== 1 error in 1.14s ===============================
```
GREEN, CPU:
```
$ nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas -q -m "not gpu and not claude"'
145 passed, 4 deselected in 6.62s
```
GPU, at the configured limit:
```
$ nix-shell --run 'OMP_NUM_THREADS=4 timeout 1500 pytest tests/atlas/test_record.py -v -s -m gpu'
test_killed_recording_resumes_to_same_result    PASSED
test_record_p50c_keeps_only_the_edited_neurons  PASSED
test_worst_case_batch_fits_on_the_gpu           FAILED  torch.OutOfMemoryError
============ 1 failed, 2 passed, 14 deselected in 145.76s (0:02:25) ============
```
GPU, worst-case test only, with `batch_tokens` overridden to 49,152 by a scratch pytest plugin (`config.py` untouched):
```
worst case: 12 x 4096 tokens, 7 layers, peak 6.80 GiB, 6592 tokens/s over 7.5 s including file writes
====================== 1 passed, 16 deselected in 31.82s =======================
```
`test_record.py` totals: 15 CPU tests pass; 2 of 3 GPU tests pass. The last CPU test was added after the GPU runs and touches
no GPU code.

Mutation checks on a scratch copy; each deliberate bug must fail a test:
| Bug introduced | Caught by |
|---|---|
| Replay finished batches on resume | `test_crash_then_resume_matches_a_clean_run` |
| Non-deterministic document keys | `test_crash_then_resume_matches_a_clean_run` |
| Ignore the padded-token limit | `test_batches_respect_limits_and_cover_all`, `test_plan_enforces_a_small_padded_token_limit` |
| Lose the non-finite count on resume | Survived at first; caught by the added `test_nonfinite_count_survives_a_resume` |
| Skip the fingerprint check | `test_resume_refuses_a_different_plan` |
| Leave checkpoints behind | `test_record_pass_writes_every_output_and_removes_checkpoints` |

## Measured speed and memory
7 layers, random token ids, batch loop only:
| Batch | Tokens/s | Peak |
|---|---|---|
| 32 x 256 | 6,906 | 3.06 GiB |
| 32 x 512 | 8,455 | 3.76 GiB |
| 32 x 1024 | 8,747 | 5.30 GiB |
| 12 x 4096 | 9,064 | 6.80 GiB |

At these speeds 50M tokens would take about 1.5 to 2.0 hours. Treat that as optimistic: no real corpus exists yet, and each batch
here was full of equal-length documents, so real text with padding will be slower. `record_pass` reports the measured rate
through `status.note` after 100 batches.

## Files
Created:
- `src/probe/atlas/record.py`
- `tests/atlas/test_record.py`
- `.superpowers/sdd/2026-11-18-level3-atlas/task-8-report.md`

No other project files changed. Diagnostic scripts stayed in the session scratchpad.

## Self-review
- Completeness: every brief item is implemented. The worst-case test fails at the configured limit, as described above.
- Clarity: the batch loop is inline in `record_pass`, which keeps the checkpoint and resume state in one place.
- Overbuilding: the three additions above are small and each guards a real failure. No OOM retry, because a partly applied batch
  would double count.
- Test quality: the kill test uses a real subprocess and SIGKILL, and asserts `resumed_from > 0`. The mutation run found one weak
  spot, now covered.

## Concerns
1. **`batch_tokens` needs a ruling before the real run.** At 65,536 a long-document batch will crash the run, and it would crash
   again at the same batch on every resume.
2. **The safe limit depends on desktop GPU use.** These numbers assume about 3.5 GB is taken by the desktop; a browser or game
   using much more could cause an OOM even at 49,152.
3. **`record_p50c` reuses the `record` status stage.** The bar will drop from 50M/50M to 0/5M when it starts (T12 runs it at the
   end of that stage). This is cosmetic.
4. **A checkpoint at 7 layers is about 0.8 GB** (calculated) and is built in RAM before writing. Its write time was not measured.

---

# Fix report: rulings R14, R15, R16

## Status: DONE_WITH_CONCERNS (replaces the BLOCKED status above)
All 23 tests in `tests/atlas/test_record.py` pass, including the 3 GPU tests at the configured limit.

## Changes
**R14** (`src/probe/atlas/config.py`): `batch_tokens = 40_960`. No other value changed.

**R15** (`src/probe/atlas/record.py`):
- `record_pass` catches `torch.cuda.OutOfMemoryError` around each batch, calls `torch.cuda.empty_cache()`, waits 60 s and retries the
  same batch, up to 3 times. The wait is injectable through the new `wait=` keyword (default `time.sleep`).
- After the third retry it raises `RuntimeError` naming the batch and saying the last checkpoint is intact. No checkpoint is written
  on this path.
- Retrying alone would double count, so two guards were needed:
  - Layers already recorded for this batch are skipped on a retry (the `finished` set).
  - Each layer's recorder state is cloned before `update` and restored if `update` runs out of memory partway through.

**R16** (`src/probe/atlas/record.py`):
- The `record` stage total is now corpus tokens + `p50c_tokens` in both passes.
- During `record_p50c`, done = corpus tokens + p50c tokens done, capped at `p50c_tokens`. It ends at done = total even if the corpus
  is shorter than `p50c_tokens`.
- `record_pass` has a new `progress=` keyword, which `record_p50c` uses for this.

**Tests** (`tests/atlas/test_record.py`), 5 added:
- `test_oom_in_the_forward_is_retried_to_the_same_result`
- `test_oom_halfway_through_an_update_is_rolled_back`
- `test_persistent_oom_stops_clearly_and_keeps_the_checkpoint`
- `test_record_stage_covers_both_passes_and_never_moves_back`
- `test_record_stage_completes_when_the_corpus_is_shorter_than_p50c_tokens`

Two existing assertions were updated: the batch-limit test now reads `tiny_cfg.batch_tokens` instead of the brief's literal 65,536,
and the stage total now includes `p50c_tokens`.

## Commands and output
No other GPU job was running (checked with `pgrep` and `nvidia-smi` before the run).
```
$ nix-shell --run 'OMP_NUM_THREADS=4 timeout 550 pytest tests/atlas/test_record.py -v -s'
worst case: 10 x 4096 tokens, 7 layers, peak 6.03 GiB, 7098 tokens/s over 5.8 s including file writes
======================== 23 passed in 155.82s (0:02:35) ========================

$ nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas -q -m "not gpu and not claude"'
150 passed, 4 deselected in 9.17s
```
**Peak GPU memory at the configured limit: 6.03 GiB**, against about 8.1 GiB available beside the desktop.

## Mutation checks
Each deliberate bug below now fails at least one test:
| Bug introduced | Caught by |
|---|---|
| No rollback after a failed `update` | `test_oom_halfway_through_an_update_is_rolled_back` |
| Finished layers recorded again on retry | `test_oom_in_the_forward_is_retried_to_the_same_result` |
| No retry | all 3 OOM tests |
| 2 retries instead of 3 | forward-retry and persistent-failure tests |
| No wait between retries | all 3 OOM tests |
| OOM propagates unhandled | all 3 OOM tests |
| Stage total omits `p50c_tokens` | `test_record_pass_writes_every_output_and_removes_checkpoints`, `test_record_stage_covers_both_passes_and_never_moves_back` |
| p50c progress restarts the bar | `test_record_stage_covers_both_passes_and_never_moves_back` |
| No final stage update after p50c | `test_record_stage_completes_when_the_corpus_is_shorter_than_p50c_tokens` (added after this bug survived) |

## Concerns
1. **TDD order:** the R15 and R16 tests were written together with the code, so there was no failing run first. The mutation checks
   above are the evidence that these tests detect the bugs.
2. **The OOM retry has only been tested with injected failures on CPU.** No real GPU OOM has gone through the retry path.
3. **Speed after the rollback clones is not directly comparable to the earlier table.** The 7,098 tokens/s figure includes file
   writes over a single batch. The cost of the clones was not measured separately; the first real run's `status.note` will show it.
4. **The real-corpus speed is still unknown**, since all measurements use random token ids.
5. **The safe limit still depends on desktop GPU use**, though the margin is now about 2 GiB.

---

# Fix report: review round 1

## Status: DONE_WITH_CONCERNS
All 35 tests in `tests/atlas/test_record.py` and `tests/atlas/test_store.py` pass, including the 3 GPU tests.

## Changes
1. **Second model guard** (`src/probe/atlas/record.py`, `record_p50c`):
   - Raises `RuntimeError` before `load_model("p50c")` if more than 0.5 GiB is allocated. The contract is in the docstring.
   - The p50c model is freed in a `finally` block with `gc.collect()` and `torch.cuda.empty_cache()`.
   - The guard only applies when a model is about to load, so a finished recording never trips it.
   - `test_record_p50c_keeps_only_the_edited_neurons` now runs `record_p50c` in a subprocess, and the child asserts that under 0.5 GiB
     is still allocated afterwards.
   - Added `test_record_p50c_refuses_to_load_beside_another_model` and `test_record_p50c_frees_its_model_before_returning`.
2. **Worst-case test** (`tests/atlas/test_record.py`): passes a `wait` that fails the test, so a retry cannot hide an OOM.
3. **Leftover checkpoints** (`record_p50c`): `out/ckpt` is removed unconditionally before `_keep_columns`.
   Added `test_record_p50c_removes_checkpoints_left_by_a_kill_during_cleanup`.
4. **OOM note** (`record_pass`): `status.note(...)` before each wait names the batch, the retry number and the 60 s wait.
   One addition beyond the request: a "recovered" note after a successful retry, so the viewer does not keep showing a stall that has ended.
5. **Checkpoint rotation** (`src/probe/atlas/store.py`): only checkpoints with a valid sidecar count toward `keep`, and orphans are
   deleted during rotation. A valid sidecar is one that parses and holds a `sha256`; the `.npz` is not re-hashed during rotation.
   Added `test_orphaned_checkpoints_use_no_slot_and_are_deleted` in `tests/atlas/test_store.py`.

## Commands and output
RED, tests written first:
```
$ nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_record.py tests/atlas/test_store.py -q -m "not gpu"'
FAILED test_record.py::test_oom_in_the_forward_is_retried_to_the_same_result
FAILED test_record.py::test_record_p50c_refuses_to_load_beside_another_model
FAILED test_record.py::test_record_p50c_removes_checkpoints_left_by_a_kill_during_cleanup
FAILED test_store.py::test_orphaned_checkpoints_use_no_slot_and_are_deleted
4 failed, 28 passed, 3 deselected in 7.11s
```
`test_record_p50c_frees_its_model_before_returning` passed before the fix, because the earlier code already dropped the model on
success. The mutation check below shows that it detects a model that is kept.

GREEN. Only desktop processes were on the GPU before the run.
```
$ nix-shell --run 'OMP_NUM_THREADS=4 timeout 900 pytest tests/atlas/test_record.py tests/atlas/test_store.py -v -s'
worst case: 10 x 4096 tokens, 7 layers, peak 6.03 GiB, 6955 tokens/s over 5.9 s including file writes
======================== 35 passed in 158.82s (0:02:38) ========================

$ nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas -q -m "not gpu and not claude"'
154 passed, 4 deselected in 11.03s
```

## Mutation checks
| Bug introduced | Caught by |
|---|---|
| Guard removed | `test_record_p50c_refuses_to_load_beside_another_model` |
| p50c model kept alive | `test_record_p50c_frees_its_model_before_returning` |
| Leftover checkpoints not removed | `test_record_p50c_removes_checkpoints_left_by_a_kill_during_cleanup` |
| No note before the OOM wait | `test_oom_in_the_forward_is_retried_to_the_same_result` |
| No "recovered" note | `test_oom_in_the_forward_is_retried_to_the_same_result` |
| Orphans not deleted | `test_orphaned_checkpoints_use_no_slot_and_are_deleted` |
| Orphans count toward `keep` | `test_orphaned_checkpoints_use_no_slot_and_are_deleted` |

## Concerns
1. **T12 must free the base model before calling `record_p50c`,** or the guard stops the run with the new error. The plan runs
   `record_p50c` at the end of the `record` stage, so the T12 brief should say this.
2. **The OOM retry has still only been tested with injected failures.** No real GPU OOM has gone through it.
3. **Real-corpus speed is still unknown.** All measurements use random token ids.
