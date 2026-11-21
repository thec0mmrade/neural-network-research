# Task 11 report: blind test writing and scoring

Status: DONE_WITH_CONCERNS (the code is complete and tested; the concerns are things Tasks 12 and 13 must do when calling it).

## What I built

`src/probe/atlas/evaluate.py`:

- `write_tests(cfg, status, items, out)`: sends explanation and context only, under anonymous ids `t01`, `t02`, ... Every reply goes through `prompts.validate_tests(data, ids)`, and a failure raises `LLMFailed`, which `batches.process` handles by splitting. Records hold `nid`, `pole`, the three passage lists, and `job` when the item has one. `job` is never sent to Claude. Writes the R10 viewer payload with `score: None` and `tokens: []`.
- `example_ids(tok, text, context, cfg)`: `plain` is the bare text cut to 128 tokens. `reasoning` puts the brief's chat-template prefix first and cuts only the passage to 128 tokens.
- `passage_hit(a, gate, eligible, start, pole, extreme) -> bool`: the pure scoring helper. It uses `a * pole` at positions that are eligible, gate-positive, inside the passage and finite (R13), against `FIRE_FRACTION` (0.3, imported from `recorder.py`) times the extreme. A pole with extreme <= 0 never hits.
- `score_tests(cfg, model, tok, layer, tests, summary)`: one layer at a time through `activations.capture`. Batches come from `record._group_window`, so they hold at most `cfg.batch_docs` (32) passages and `cfg.batch_tokens` padded tokens. Returns copies in which each category's list is replaced by its hit count. Failed records pass through. A neuron from another layer raises `ValueError`.
- `summarize(records, summary)`: `primary` and `ac_matched`, each with `n` and the three shares (hits / 10, averaged over neurons).

## Test evidence

RED, before `evaluate.py` existed:

```
$ nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_evaluate.py -v'
E   ImportError: cannot import name 'evaluate' from 'probe.atlas' (.../src/probe/atlas/__init__.py)
ERROR tests/atlas/test_evaluate.py
=============================== 1 error in 1.34s ===============================
```

The brief expected `ModuleNotFoundError`. The import form `from probe.atlas import evaluate` reports the same missing module as `ImportError`.

GREEN:

```
$ nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_evaluate.py -q'
41 passed in 28.25s

$ nix-shell --run 'OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pytest tests/atlas -q -m "not claude"'
312 passed in 188.68s (0:03:08)
```

The 41 include the one GPU test. No real `claude` calls were made: the tests replace `call_with_wait`, and the suite ran with `-m "not claude"`.

R6: `test_score_counts_only_passage_positions` runs the real model on a reasoning passage and picks the neuron and pole that fire much harder in the prefix than in the passage (prefix peak more than twice the passage peak, passage peak above 0). It then checks three things:

1. With the threshold between the two peaks, `score_tests` gives 0 hits out of 10.
2. `passage_hit` with `start=0` on the same activations returns True, so the prefix activation is what the `start` limit excludes.
3. With the threshold below the passage's own peak, the same passages score 10 of 10.

Mutation check: I applied 25 deliberate bugs to a scratch copy of `evaluate.py`, one at a time, and ran the CPU tests against each. All 25 were caught. They covered the prefix, gate, eligibility, threshold, pole sign, NaN handling, truncation, blinding, validation, and the `primary` and `ac_matched` rules. The run took two passes because one mutation's search pattern was wrong in my script; the second pass covered the remaining 9.

## Files

- Created: `src/probe/atlas/evaluate.py`
- Created: `tests/atlas/test_evaluate.py`
- Created: `.superpowers/sdd/2026-11-18-level3-atlas/task-11-report.md`

No other project files changed. `tests/atlas/conftest.py` was not needed.

## Self-review findings

- Removed an unused `import torch` and an unused `LLMFailed` test import.
- The brief's `test_ac_matched_excludes_negative_dominant_and_dead` is kept verbatim, but its summary marks both neurons positive-dominant, so the negative record is excluded as a pole mismatch. It also has no dead neuron. I added `test_negative_dominant_neuron_counts_in_primary_but_not_ac_matched` and `test_dead_neurons_count_in_neither_score` to cover those two cases directly.
- `example_ids` takes `cfg` because the brief's signature requires it, but does not use it.

## Choices the brief did not settle

- `TEST_BATCH = 5` neurons per call. `Config` has no setting for this and `config.py` is outside my files. 30 passages per neuron is about 3,000 reply tokens, so the explanation batch of 20 would ask for about 60,000 tokens in one reply. Batch size affects cost and failure rate, not how scores are computed.
- A live neuron with no record is left out of `ac_matched`, because its job label is unknown. It counts as zero in `primary`, as specified.
- A record whose `pole` differs from the summary's dominant pole counts as zero in `primary` and is left out of `ac_matched`.
- `n = 0` gives shares of 0.0.
- Duplicate records for one neuron raise `ValueError`, so a headline score can't silently use the wrong one.

## Concerns for later tasks

1. **Pilot scores (T12):** `summarize` counts every live neuron without a record as zero, as specified. Passing the full `summary.npz` for a 200-neuron pilot would push the primary score close to zero. T12 must pass a summary whose `live` mask covers only the pilot neurons.
2. **`job` (T12, T13):** items passed to `write_tests` must include the tested pole's `job` from the explanation record. Otherwise `summarize` raises `KeyError` on a positive-dominant record.
3. **Both poles (T13):** `research` tests both poles of a neuron. Batch files are named by `nid`, so each pole needs its own `out` directory, or a rerun would reuse the other pole's passages.
4. **Control (T12):** `score_tests` scores one layer per call, so a control pairing across layers must be scored under the layer of the neuron being measured.
5. **`PROMPT_HASH`:** the test prompt's layout is built in `evaluate.py`, so `PROMPT_HASH` doesn't cover it. Changing `_test_prompt` needs a new `prompt_version`. I left this alone because `prompts.py` is outside this task's files.

---

# Fix round 1 (R24, R25, R26 and literal special tokens)

## Changes

1. **R24:** `test_prompt`, `PASSAGE_TOKENS` and `REASONING_USER` moved to `prompts.py`. `PROMPT_HASH` now covers all three, using a golden render of `test_prompt`. `evaluate.py` reads the values from `prompts`, and the cached reasoning prefix is keyed on the user text. The hash value changes with this fix; nothing has been recorded under the old one.
2. **R25:** in `summarize`, AC-matched now counts a failed neuron as zero when its job is known and isn't "unclear". Live positive-dominant neurons with an unknown job are left out and counted in `ac_matched["excluded_unknown_job"]`. `write_tests` carries `job` onto failed records. `test_hand_computed_scores` holds the worked case in its docstring: primary n = 6 with shares 0.4 / 0.1 / 1/15; AC-matched n = 2 with shares 0.4 / 0.1 / 0.05 and 2 excluded.
3. **R26:** `summarize(records, summary, layers=None)` restricts both scores to the given layers. `layers=[]` means no neurons.
4. **Literal special tokens:** passages now go through `_passage_ids`, which tokenizes each literal added token in two pieces.

## Differs from the suggested fix

`split_special_tokens=True` does not work for this tokenizer. On the real local tokenizer, `a</think>b` still gave `[64, 248069, 65]`, where 248069 is `</think>`. It only splits tokens marked special (such as `<|im_start|>`). `<think>`, `</think>`, the tool tags and the `fim` tokens are added tokens without that mark, 12 of the 33 added tokens in all. `_passage_ids` handles all 33. The test checks every added token in five surroundings, in both contexts: no added id appears and the passage decodes back to the original text. Text with no added token is tokenized in one piece, exactly as before.

## Added beyond the request

Failed records also carry `pole`. Without it, a job from a test of the other pole could be taken for the dominant pole's job. A failed record with no `pole` is treated as belonging to the dominant pole.

A scored record with no `job` on a positive-dominant neuron still raises `KeyError`, because R23 requires callers to supply it.

## Test evidence

RED, with the new tests in place and before the fixes:

```
$ nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_evaluate.py tests/atlas/test_explain.py -q -m "not gpu and not claude"'
17 failed, 127 passed, 1 deselected in 7.81s
```

GREEN:

```
$ nix-shell --run 'OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pytest tests/atlas/test_evaluate.py tests/atlas/test_explain.py -q -m "not claude"'
145 passed in 29.89s

$ nix-shell --run 'OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pytest tests/atlas -q -m "not claude"'
330 passed in 172.62s (0:02:52)
```

Mutation check on the new code: 15 deliberate bugs applied to a scratch copy, 15 caught by the CPU tests.

## Files changed in this round

- `src/probe/atlas/prompts.py`
- `src/probe/atlas/evaluate.py`
- `tests/atlas/test_evaluate.py`
- `tests/atlas/test_explain.py`
- `.superpowers/sdd/2026-11-18-level3-atlas/task-11-report.md`

## Concerns

Concern 5 of the first report (`PROMPT_HASH`) is resolved by R24. Concerns 1 to 3 are covered by R23. Concern 4 (control pairs are scored under the measured neuron's layer) still applies to T12.
