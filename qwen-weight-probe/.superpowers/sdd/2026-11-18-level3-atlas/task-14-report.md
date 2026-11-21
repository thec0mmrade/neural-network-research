# Task 14 report: end-to-end rehearsal and docs (steps 1–3 only)

Status: DONE_WITH_CONCERNS. Steps 4–5 were not done, and the real run was not started (R8). `results/07_atlas` and `results/07_atlas.md` do not exist; a test checks that the rehearsal leaves them untouched.

## What I built

### Rehearsal launcher: `tests/atlas/fixtures/rehearsal_cli.py` (test-only)
- Usage: `rehearsal_cli.py ROOT {run,approve-pilot,status} [FAKE_MODE]`. It imports `probe.atlas.__main__` and calls the real `cli.main([command])`, after swapping in:
  - the config: `Config.tiny(root)` made smaller still (90k total tokens, 10k multilingual, 30k thinking, 10k shards, 8k p50c) and pointed at a wrapper that runs `fake_claude.py`;
  - local web text through `build_sources(reader=..., revision_of=...)`, with no network; the reader adds 10 ms per document so a kill can land inside `corpus`;
  - the first 60 math practice documents (`Config.tiny` alone would read all of them, about 3M tokens, because the math source ignores `total_tokens`);
  - thinking capped at 256 new tokens per document;
  - a smaller live set: 8 random live neurons per non-research layer. In L15 and L19 it keeps every studied neuron plus enough controls; the study set is the first 6 p50c-edited neurons plus the top 4 u-writers. It wraps `load_summary` in pilot, research and report. The files on disk stay unchanged.
- It refuses a root inside `results/`.
- Nothing in `src/probe/atlas` refers to it. No environment variable or flag reaches it, so a normal `python -m probe.atlas` can't enter rehearsal mode. Tests check this (below).
- Real components: base and p50c models, tokenizer, GPU generation, recording, scoring, the stage runner, the gate, status, report and the wiring join (through a read-only symlink `<tmp>/06_wiring -> results/06_wiring`).

### Fake Claude: modes `atlas` and `atlas_refuse` in `tests/atlas/fixtures/fake_claude.py`
- It derives the ids from the prompt's `=== id ===` headers and exits with an error if they differ from the schema's ids.
- Replies come only from the prompt text:
  - explanations from a hash of the rendered examples, with activation numbers removed;
  - test passages from a hash of the description;
  - ratings from a hash of the explanation.
- Identical evidence therefore always gives an identical reply. A resumed run that showed Claude different examples would get different explanations, and the comparison would fail.
- `atlas_refuse` returns a refusal for every call that contains about 1 in 6 neurons (by hash). The batch splitting then isolates exactly those neurons.

### Tests: `tests/atlas/test_end_to_end.py`
GPU tests. Each runs the CLI in subprocesses under a pytest temp root:
- `test_clean_run_stops_at_the_gate_then_finishes_every_stage`:
  - `run` stops at `waiting_for_pilot_approval` with no `layers/` folder;
  - then `approve-pilot`, then `run` reaches `complete`;
  - all 8 stage markers exist;
  - `atlas_neurons.jsonl` has 25,088 rows, every live neuron is scored and explained, and research rows are present;
  - wiring is joined for live neurons;
  - `07_atlas.md` is under the temp root and has the research section.
- `test_a_second_run_before_approval_stays_at_the_gate`: a second `run` before approval stops again, and `status` reports `Stage: pilot`.
- `test_viewer_renders_a_real_status_file`: `atlas_watch.render` on the rehearsal's real `status.json`.
- `test_sigkill_in_each_stage_matches_clean_run[corpus|thinking|record|pilot|layers]` (the brief's test):
  - SIGKILL lands once the stage is `current_stage` in status.json with progress > 0 (record: ≥ 40%, so a checkpoint exists);
  - the test asserts the process died by SIGKILL and that the stage's `.done` marker is absent;
  - then resume through the gate to completion;
  - `assert_same_atlas` compares:
    - the corpus manifest hash;
    - every array in `record/` and `record_p50c/` (floats `rtol=1e-3`, everything else exact);
    - the pilot report (except usage and time);
    - every layer's `scores.json`, `research.json` and `atlas_neurons.jsonl`;
    - `07_atlas.md` byte for byte.
- `test_neurons_claude_refuses_are_reported_as_failed_and_the_run_still_finishes`: an `atlas_refuse` run completes with failed neurons marked in the jsonl and counted in the report's coverage table. The pilot and research outputs count the failures.
- `test_a_rehearsal_leaves_the_real_outputs_alone`: the existence and mtime of `results/07_atlas` and `results/07_atlas.md` are unchanged.

CPU tests:
- the default CLI config is the real one, even with a plausible env var set;
- no module in `src/probe/atlas` mentions a rehearsal or reads the environment (other than the existing `setdefault` caps);
- `python -m probe.atlas status` runs and creates nothing;
- the web reader's libraries (`pyarrow`, `huggingface_hub`) import;
- the launcher refuses `results/`;
- local text is reproducible;
- the fake's explain, test and rate replies pass the real schemas and validators through `llm.call_claude`;
- the GPU guard refuses when the test process holds GPU memory.

### Docs (step 3)
- `CLAUDE.md`: a new `src/probe/atlas/` entry covering:
  - `nix-shell --run 'python -m probe.atlas run'`, the stage order, resume, and the gate (`waiting_for_pilot_approval`, `results/07_atlas/pilot/v1/report.md`, `approve-pilot`, only the user approves);
  - `status`, and the viewer `nix-shell --run 'python -m probe.atlas_watch'` (optional status path);
  - never `--bare`;
  - the output in `results/07_atlas/` (with `atlas_neurons.jsonl`) and `results/07_atlas.md`;
  - the prompt_version rule and the rehearsal test.
- `TODO.md`: Level 3 now has a "Design decisions: decided 2026-11-18" block listing scope, LLM, corpus, order, pilot gate, recovery and viewer, linking the spec. The checklist items stay open until the real run finishes.

## Integration bugs found

No bug was found in `src/probe/atlas/`, and I changed no file there. Two problems came up, both in the test setup:

1. **Test suite: the session-scoped `base_model` fixture kept a model on the GPU while the rehearsal ran.**
   - Seen in the first full-suite run. `test_activations` loads the session fixture, and pytest kept it until the session ended. The rehearsal's own `record` stage then ran out of GPU memory: "batch 8 of 12 still ran out of GPU memory after 3 retries", and every rehearsal test errored.
   - It also broke the rule of one GPU process at a time.
   - Fix (`tests/atlas/conftest.py`): `base_model` is now module-scoped. On teardown it drops the model, then runs `gc.collect()` and `torch.cuda.empty_cache()`.
   - Regression guard: the `tiny_workspace` fixture calls `require_free_gpu(torch.cuda.memory_allocated())` and stops with a clear message if the test process holds more than 100 MiB. `test_the_rehearsal_refuses_to_share_the_gpu_with_the_test_process` tests that guard.
   - Side result: the pipeline's OOM retry (R15) behaved as designed under real memory pressure. It waited, retried 3 times, stopped with a clear message and left the checkpoint intact.
2. **My harness: resuming after a kill in `layers` expected the gate again.** The pilot was already approved, so `run` went straight to `complete`. Fix: `run_to_completion` skips the gate when `pilot/approved.json` exists. The `[layers]` kill test covers it.

## Further checks on the real data (one-off scripts in the scratchpad, not tests)
- **Activation recording:**
  - Replaying the exact recorded batches reproduced every stored top-list value with 0 error, for base L15 and L19 and p50c L15 and L19.
  - Re-running single documents alone differs from the batched values by a median of 0.6% and a maximum of 3%. One near-zero p50c value (0.02) differed by 0.007. This is bf16 noise from the batch composition; it comes from the batch the document ran in, not from a recording error.
  - Window alignment matched, every stored pole example is gate-positive, and the summary extremes equal the list maxima.
- **Scoring:** batched `score_tests` gave hit counts identical to running each passage alone, for 3 neurons × 30 passages.
- **Inputs:** the reasoning prefix decodes to the chat template with `<think>`, and a literal `</think>` in a passage stays plain text.
- **Rendering and docs:** a rendered neuron fits the budget (2,406 tokens) and reads correctly. Thinking documents decode to chat-template prompts plus the model's thinking.
- **Wiring join:** it reads the real `edges_raw.npz` (86,352 components, which matches `component_id(23, 3583) = 86351`). The `L15.mlp.7` edges name valid neurons in later layers.
- **p50c:** the edited neurons are 64 per layer (128 in total, as the spec says).
- **Real config smoke:** the real `python -m probe.atlas status` prints "No run has started yet." and creates nothing.

## Viewer run against a live rehearsal
I ran `python -m probe.atlas_watch <rehearsal>/status.json` in a pseudo-terminal (130×42) for 12 s while the rehearsal was in `layers`. It drew 48 frames with no "viewer error" and no traceback. What it showed:
- **Pipeline table:**
  - corpus 60,058/60,000 (100%) at 1.41e+03 tokens/s, done;
  - thinking 30,000/30,000 done;
  - assemble 93,081/93,081 done;
  - record 101,081/101,081 done at 1.87e+03 tokens/s;
  - pilot 42/42 done at 817 steps/h;
  - layers 0/249, current (▶);
  - research ~3 h and report ~0.2 h, both default estimates.
- **Layers panel:** "no layers yet".
- **Usage panel:** 14 calls, 15,246 input and 728 output tokens. Scores: primary n=14 at 7% / 0% / 13%, AC-matched n=6, AC 76% / 11% / ~1%.
- **Latest neuron:** "L20.mlp.2318 [blind_maximum 0/10, near_miss 0/10, distant 0/10]: fires on pattern 06454cbc4a (positive)".
- **Bottom bar:** "OVERALL 0.1% layers elapsed 5m paused 0s ~6d 10h left". The time left uses the default stage hours, because layers was below 2%.

## Test evidence
- Final command, from the project root:
  `nix-shell --run 'OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pytest tests/atlas -v -m "not claude"'`
  Result: `469 passed in 4079.58s (1:07:59)`, with 0 skipped, 0 failed and 0 errors. All 20 tests in `test_end_to_end.py` passed.
- The first full-suite attempt, before fix 1: `1 failed, 458 passed, 9 errors`. All of them were the OOM caused by the held `base_model`.
- Before starting each GPU job I checked that no other atlas job or pytest was running. The only other GPU users were the desktop processes (about 160 MiB).

## Measured time (RTX 4070, no flash-linear-attention)
- **One tiny run, outside pytest:** 5 min 28 s to the gate, then 3 min 30 s after approval, about 9 min in all. Per stage (active seconds):

  | stage | seconds |
  |---|---|
  | corpus | 43 |
  | thinking | 20 |
  | assemble | 22 |
  | record (both passes) | 54 |
  | pilot | 183 |
  | layers | 174 |
  | research | 28 |
  | report | under 1 |

  The pilot and layers stages are mostly model reloads (7 per stage).
- **Inside pytest:** the clean run took 522 s and 525 s, each kill test 536–567 s, and the refusal run 531 s. The rehearsal GPU tests take about 57 minutes in all (7 full rehearsals), and the whole suite 68 minutes.

## Files created or changed
- Created: `tests/atlas/test_end_to_end.py`, `tests/atlas/fixtures/rehearsal_cli.py`.
- Changed: `tests/atlas/fixtures/fake_claude.py` (new `atlas` and `atlas_refuse` modes; existing modes unchanged), `tests/atlas/conftest.py` (`base_model` is module-scoped and frees the GPU on teardown), `CLAUDE.md`, `TODO.md`.
- No file in `src/probe/atlas/` changed.

## Concerns
1. **The real web reader is untested end to end.** The rehearsal uses no network, so these were never exercised:
   - `read_web_docs`;
   - the parquet paths `data/CC-MAIN-2024-10/000_00000.parquet` and `data/<lang>/train/000_00000.parquet`;
   - `_hf_revision`;
   - whether each file holds enough tokens.

   Only the imports are checked. A wrong path will fail in the first minutes of the real `corpus` stage, before any GPU or Claude cost. That is cheap, but worth watching.
2. **The rehearsal shrinks volumes through test-only seams.** These are the math document cap, the 256-token thinking cap, the smaller live set and the smaller study set. The real-scale code paths (4000-token thinking, 200-neuron pilot, about 25k live neurons, 128 studied neurons) run the same code at larger sizes, which earlier tasks tested in isolation.
3. **GPU headroom is thin.** About 2 GiB of extra GPU use, beside about 3.4 GiB for the desktop, was enough to exhaust the card at `batch_tokens = 40,960`. During the real run, nothing else should use the GPU, including this test suite. The OOM handling recovers cleanly, but each failed batch costs 3 minutes.
4. **Activations depend slightly on the batch.** Batched and single-document activations differ by about 0.6% (bf16), up to 3%. Resumed runs match exactly only because the batch plan is fixed, and the kill tests confirm that. Scores near the 30% threshold carry about 1% noise.
5. **The rehearsal can't judge explanation quality.** Its scores and the "projected full run" line in the tiny pilot report mean nothing (fake Claude, tiny corpus). Only the real pilot can show whether the explanations are worth the cost.
6. **The suite is slow.** It now takes about 68 minutes, almost all of it the 7 rehearsals. Run `-m "not gpu"` for quick checks.
