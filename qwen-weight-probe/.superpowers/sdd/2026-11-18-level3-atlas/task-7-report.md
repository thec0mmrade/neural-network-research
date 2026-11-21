# Task 7 report: streaming recorder state and final statistics

Status: DONE_WITH_CONCERNS (four interpretation decisions need the controller's check; see Concerns)

## What was built
`src/probe/atlas/recorder.py`:
- `HIST_EDGES = torch.logspace(-4, 3, 255)`.
- `LayerRecorder(cfg, device)` with `update`, `state`, `load`. State stays on `device`.
  - Each document is processed alone: `topk` picks up to `per_doc_cap` candidates per list, and windows are gathered only for those.
  - Candidates merge once per update with a stable sort. The per-document cap is enforced again at merge, so it also holds if a document id repeats across updates.
  - Histogram and running totals run in chunks of at most 8,192 positions.
  - Random examples: one position per document, keeping the `random_k` smallest keys. Empty slots hold key +inf, doc -1, tok -1, NaN window.
- `finalize(states, cfg)`: per-layer `pos_extreme`, `neg_extreme`, `dominant`, `live`, `fire_share`, `dn_flag`.

## Files
Created:
- `src/probe/atlas/recorder.py`
- `tests/atlas/test_recorder.py`
- `.superpowers/sdd/2026-11-18-level3-atlas/task-7-report.md` (this file)

Nothing else was changed.

## RED
Command: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_recorder.py -v'`
```
tests/atlas/test_recorder.py:9: in <module>
    from probe.atlas.recorder import HIST_EDGES, LayerRecorder, finalize
E   ModuleNotFoundError: No module named 'probe.atlas.recorder'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 1.09s ===============================
```

## GREEN
Command: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_recorder.py -v'`
```
============================== 43 passed in 1.73s ==============================
```
Command: `nix-shell --run "OMP_NUM_THREADS=4 pytest tests/atlas -q -m 'not gpu and not claude'"`
```
117 passed, 1 deselected in 4.84s
```
The 43 tests are the brief's 6 plus added coverage. The brute-force recorder in the test file is a separate plain Python/numpy implementation. It is compared on values, docs, tokens, windows, random examples, histograms and running totals.

## Extra verification (ad hoc scripts in the scratchpad, not project files)
Mutation check: 35 deliberate bugs were injected into a scratch copy of the recorder.
- The first round left 5 survivors. Four were real test gaps (gate-negative in the negative pole, the dn value condition, bin-edge handling, a non-positive dn flag), and tests were added for them.
- Final round: 34 of 35 caught. The survivor raises the per-document `topk` to cap + 1, which changes nothing because the merge enforces the cap again.

GPU check (card idle apart from the desktop):
- CPU and CUDA states were identical on the same 5 batches (64 neurons; `sum` compared to rtol 1e-12, all others exactly).
- At real scale (3,584 neurons, 16 docs x 4,096 tokens, random activations): 0.25 s per update per layer, 519 MiB peak temporary memory, 111 MiB state.
- That is about 1.8 s per 65,536-token batch for 7 layers, or roughly 25 minutes of recorder time over 50M tokens. This is an extrapolation from random data, excluding the model's forward pass.

## Self-review findings (all fixed)
- A batch with no eligible position crashed on an empty chunk. Now returns early, with a test.
- `offsets` was wrong for an odd window size (the configured 64 was unaffected).
- Histogram index built from int64 temporaries; now int32 in place.
- Each random candidate copied the whole random window array even when it could not enter the list; now skipped early.
- `doc_keys` on a different device from the recorder failed; now moved to the recorder's device.

## Concerns (decisions for the controller to check)
1. **Document end is inferred.** `update` has no attention mask, so a document is taken to end at its last eligible token. A real trailing ineligible token, such as a final `<|im_end|>`, shows as NaN in windows instead of its activation. Interior ineligible tokens keep real values. Examples, histograms and totals are unaffected. An exact fix would be passing the attention mask from Task 8.
2. **`neg_val` holds the magnitude (-a), positive.** This keeps "descending, -inf when empty" true for every list. `neg_win` stays signed, so `neg_win[..., 32] == -neg_val`. Tasks 10 and 11 must read it that way.
3. **`dn_flag` with a part-empty positive list.** Empty slots count as -inf, so any positive double-negative value sets the flag: it would have reached the top 50, as the spec says. A literal "smallest kept value" reading would give False there.
4. **Non-finite activations are skipped silently.** A position where any neuron is NaN or inf is left out of examples, histograms and `count`, and no counter records it, because the brief's state arrays have no field for one. Task 8 may want to log such positions.

Notes, following the brief as written:
- `fire_share` is a slight underestimate, because the bin containing the threshold is not counted.
- `E` is a median over all neurons including silent ones, so if over half a layer is silent, the whole layer is dead.
- `finalize` does not use `cfg`; it stays in the signature the brief specifies.

---

# Fix round 1 (review findings)

Status: DONE_WITH_CONCERNS

## Changes
Files changed: `src/probe/atlas/recorder.py`, `tests/atlas/test_recorder.py`, and this report.

1. **`rnd_regime` added.** Int8, shape `[n_neurons, random_k]`, value `(gate >= 0) * 2 + (value >= 0)` at the example's position, -1 for empty slots. It is reordered together with the other random arrays when the reservoir replaces an entry, and is part of `state()` / `load()`. The brute-force oracle now computes it independently, and `test_random_examples_match_brute_force` compares it over 84 documents, so replacement is exercised.
2. **`fire_share` follows the pole's own extreme.** Only the dominant pole's regime counts: regime 3 (gate-positive, value-positive) for positive dominant, regime 2 (gate-positive, value-negative) for negative dominant, against 0.3 x that extreme. The denominator is still all eligible positions. The test covers both dominant signs, the other pole, gate-negative positions, and counts between the layer's threshold and the neuron's own.
3. **Merge-time window test added.** `test_cap_at_merge_keeps_windows_aligned_with_values` repeats a document id across updates so the cap drops entries, then checks value, doc, token and the full window of each kept entry, including entries ranked below the dropped ones.
4. **Zero-boundary test added.** `test_exact_zero_enters_no_list` has 12 cases with `a`, `gate` or `value` exactly 0.

## Behavior change needed for item 4
The brief's list rules did not exclude two zero cases: double-negative with `a == 0`, and `value == 0` with a nonzero `a` at a gate-positive position. Both are now excluded through one `active` mask. Real activations are unaffected in the second case, because `value == 0` forces `a == 0`. Such positions still count in histograms, totals and random examples.

## Test evidence
Command: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_recorder.py -v'`
```
============================== 56 passed in 1.81s ==============================
```
Command: `nix-shell --run "OMP_NUM_THREADS=4 pytest tests/atlas -q -m 'not gpu and not claude'"`
```
130 passed, 1 deselected in 5.06s
```

## Extra verification (ad hoc scripts in the scratchpad)
- Mutation check: 46 of 47 injected bugs caught. The survivor is the same equivalent mutant as before (per-document `topk` at cap + 1, which the merge trims). This round first exposed two weak new tests, for merge-window alignment and for `fire_share` using the median threshold; both were strengthened.
- GPU check, run before the final test-only edits (recorder code unchanged since): CPU and CUDA states identical; 0.26 s per update per layer at 3,584 neurons and 65,536 tokens; 519 MiB peak temporary memory.

## Concerns still open
Concerns 1, 2 and 4 from the first report stand: document end inferred from the last eligible token, `neg_val` stored as a magnitude, and non-finite positions skipped with no counter. Concern 3 (`dn_flag` with a part-empty positive list) is also unchanged.

Checkpoints written before this fix have no `rnd_regime`, so `load` raises `KeyError` on them. No real recording has run yet.
