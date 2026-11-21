# Task 12 report: pilot, layer run and command line

Status: DONE_WITH_CONCERNS

## What was built

**`src/probe/atlas/pilot.py`**
- `choose`: seeded, live-only, split evenly across the 7 layers with `np.array_split`.
- `derangement`: seeded pairing for the control.
- `Progress`: a status view that turns each neuron's three phases (explain, test, score) into stage progress.
- `explain_test_score`: the work shared by the pilot and the layer run. It tests the dominant pole, passes that pole's `job` (R23) and saves scoring per batch of 20 neurons through `batches.process` (R26).
- `run_pilot`: writes `report.md`, then `report.json` last, so `report.json` existing means the report is complete. `summarize` gets a live mask limited to the pilot neurons (R23).
- `approve` and `is_approved`: `approve` refuses unless a finished report exists for the current `prompt_version` and `PROMPT_HASH`.

**`src/probe/atlas/runner.py`**
- `run`: holds the lock, then runs corpus, thinking, assemble, record, pilot, layers, research, report.
- `run_layer`: explains, tests and scores all live neurons of a layer, writes `layers/<prompt_version>/L{layer}/scores.json`, and skips a finished layer on restart.
- Base model: `_BaseModel` is the only owner. It is freed before `record_p50c` and reloaded by the next stage that needs it (R17).
- Status: flushed at each stage end and at exit (R9). It is created inside the lock, so a refused second run cannot overwrite `status.json`.
- Fingerprints: `root/fingerprint.json` holds `corpus`, `model` and `llm_window`. `pilot/<version>/fingerprint.json` holds `prompt` (R21).
- Done markers: LLM stages record the prompt version and hash in their `.done` file, so a new version re-runs them.
- `research` and `report` are imported only when those stages run (R7).

**`src/probe/atlas/__main__.py`**
- Commands: `run`, `approve-pilot`, `status`. Heavy modules load only for the command that needs them, and `status` does not import torch.
- Exit codes: 0 for `complete` or `waiting_for_pilot_approval`; 1 with a one-line reason on any error, including `AlreadyRunning`.

## Test evidence

RED, before the implementation existed:

```
$ nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_runner.py -v'
tests/atlas/test_runner.py:13: in <module>
    from probe.atlas import __main__ as cli
E   ImportError: cannot import name '__main__' from 'probe.atlas'
ERROR tests/atlas/test_runner.py
=============================== 1 error in 0.48s ===============================
```

GREEN:

```
$ nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_runner.py -q'
53 passed in 11.94s

$ nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas -q -m "not gpu"'
378 passed, 5 deselected in 39.29s
```

The 5 deselected tests are the GPU tests. I did not run them, because this task changes no module they cover.

Mutation checks, each applied to a scratch copy of `src/` and run against `test_runner.py`. Every mutation was caught:

| mutation | result |
|---|---|
| gate always open | 11 failed |
| `approve` accepts anything | 3 failed |
| base model not freed before `record_p50c` | 1 failed |
| control may pair a neuron with itself | 1 failed |
| tests always use the positive pole | 1 failed |
| pilot scores use the full live mask | 1 failed |
| prompt fingerprint not checked | 1 failed |
| status created outside the lock | 1 failed |

## Files

Created:
- `src/probe/atlas/pilot.py`
- `src/probe/atlas/runner.py`
- `src/probe/atlas/__main__.py`
- `tests/atlas/test_runner.py`

No existing file was changed.

## Self-review findings, all fixed

- The "waiting for approval" note stayed in the viewer after approval. It is now cleared when the gate opens, with a test.
- A test stub replaced `explain._stored_count` without need, which hid the real function. I removed the stub.

## Concerns

1. **Not run end to end on the real model or real Claude.** All stages are stubs here, as the task requires. I checked the calls to the real stage functions against their signatures by reading the code. Task 14 is the first real exercise.
2. **Two brief tests needed fixture decisions.**
   - `change_prompt_hash` also bumps `prompt_version`. The brief's test expects `waiting_for_pilot_approval`, while the controller's requirement refuses the same version with a changed hash. Both hold this way, and a separate test covers the refusal.
   - The brief's test reads `summary[l]["live"]`, but the real summary uses flat keys such as `L14.live`. `choose` takes the real format, and the `summary_some_dead` fixture supports both, so the brief's test is unchanged.
3. **Control coverage.** Neurons whose tests failed are left out of the control, on both sides of the pairing. `control.n` in the report shows how many were scored.
4. **Coarse progress during test writing.** The overall bar moves during explaining and scoring, but jumps once at the end of a layer's test writing, because `write_tests` reports no counts.
5. **Task 13.**
   - `research.json` is planned at the root, not under `prompt_version`, so a new prompt version would overwrite it.
   - `run_research(cfg, model, tok, status)` and `write_report(cfg)` must keep those signatures.
6. **Open item before launch.** Nothing in the code checks the Claude login type. `progress.md` requires `claude auth status` to report `max` before the pilot starts, and that remains a manual step.

---

# Fix report: review round 1

All six review items are fixed.

## Changes

1. **Machine-safety settings (`__main__.py`).** `OMP_NUM_THREADS=4`, `RAYON_NUM_THREADS=8` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` are set with `os.environ.setdefault` before any other import except `os` itself (`from __future__` must come first). The approval message now prints `nix-shell --run 'python -m probe.atlas run'`. A subprocess test with those variables removed checks they are set at the first `numpy` or `torch` import.
2. **Effort and model in the fingerprint (`pilot.py`, `runner.py`).** `prompt_parts` now holds `effort` and `llm.MODEL`, so `approved.json`, `report.json` and the LLM `.done` markers all carry them. The prompt fingerprint compares them too. A change under the same `prompt_version` is refused with a message naming what differs. There is one test each for effort and model, plus one that `approve` refuses a report made at another effort.
3. **Model on the GPU only while scoring (`pilot.py`, `runner.py`).** The model loads when a batch really needs scoring and is freed, with `torch.cuda.empty_cache()`, when that layer's scoring ends. The pilot's control scoring does the same. Finished layers, and layers whose scores are all saved, load nothing.
4. **Leftover checkpoint (`runner.py`).** `_record` removes `record/ckpt` when `summary.npz` exists.
5. **Error log (`__main__.py`).** The one-line error stays, and the full traceback is appended to `cfg.root / "errors.log"`. `AlreadyRunning` is not logged, because it is an expected refusal rather than a fault.
6. **Kill test (`test_runner.py`).** `test_a_kill_at_any_paid_call_repeats_nothing_on_resume` kills a layer run at every Claude and scoring call, using the real `explain_neurons`, `write_tests` and `batches.process` with a fake Claude and fake scoring. The fake refuses one explanation and one test, so the split paths are covered. After each kill it asserts that answered calls plus resumed calls equal a clean run's calls exactly, that the result is identical, and that a further run makes no calls.

## Interface change

Item 3 changes two signatures from the brief, because a loaded model can no longer be passed in:
- `pilot.run_pilot(cfg, base, status)`, was `(cfg, model, tok, status)`
- `runner.run_layer(cfg, base, status, layer)`, was `(cfg, model, tok, status, layer)`

`base` has `get()` and `free()`. `run_research(cfg, model, tok, status)` is unchanged, so `research` still holds the model for its whole stage, including any Claude work. Task 13 may want the same treatment.

## Test evidence

```
$ nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_runner.py -v'
============================= 65 passed in 18.57s ==============================

$ nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas -q -m "not gpu"'
390 passed, 5 deselected in 46.79s
```

The 5 deselected tests are the GPU tests, which I did not run.

Mutation checks on the fixes, each applied to a scratch copy of `src/`. Every mutation was caught:

| mutation | result |
|---|---|
| `numpy` imported before the settings | 1 failed |
| effort not tracked | 2 failed |
| model not tracked | 2 failed |
| model kept after scoring | 2 failed |
| model loaded before the Claude work | 3 failed |
| leftover checkpoint kept | 1 failed |
| traceback not logged | 1 failed |
| scores not saved per batch | 2 failed |

## Files changed in this round

- `src/probe/atlas/__main__.py`
- `src/probe/atlas/pilot.py`
- `src/probe/atlas/runner.py`
- `tests/atlas/test_runner.py`
