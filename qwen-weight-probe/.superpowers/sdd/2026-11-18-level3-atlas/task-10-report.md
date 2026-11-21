# Task 10 report: batches, prompts and explanations

Status: DONE_WITH_CONCERNS

## What was built
- `src/probe/atlas/batches.py`: `process(items, size, out, work, key="nid")`. Each validated chunk is written atomically to `batch_<first-key>_<n>.json`. On `LLMFailed`, `Refused` or a key mismatch the chunk is halved; a single item that still fails is stored as `{key, "failed": reason}`. `LimitReached` is never caught. A small `split_<first-key>_<n>.json` marker records that a chunk was split, so a restart goes straight to the halves and does not pay for the whole chunk again. A damaged or foreign batch file is redone.
- `src/probe/atlas/prompts.py`: `EXPLAIN_SYSTEM`, `TEST_SYSTEM`, `RATE_SYSTEM`, `explain_schema`, `test_schema`, `rate_schema`, `render_neuron`, `cut_example`, `explain_prompt`, `PROMPT_HASH`. No torch import.
- `src/probe/atlas/explain.py`: `neuron_record`, `explain_neurons`, an LRU over corpus shards (checksum verified on load) and a cache of the layer recording. Batch ids `n01`, `n02`, ... are mapped back locally; the response must hold exactly the ids sent, with valid fields, or the batch is rejected and split. Status uses the R10 payload with `score` None.

Rulings: R10 (latest payload), R11 (`neg_val` magnitude, `neg_win` signed), R13 (NaN shows the token unmarked; the document's end comes from the corpus, not from NaN).

## Decisions the brief left open
1. **Which extreme a mark compares with.** Each token is compared with the extreme of the pole its own sign belongs to, in every section. This is the only reading that also works for random examples.
2. **Budget enforcement.** A neuron firing on every token would need an estimated 5,800 tokens, more than the 2,500 the brief's test allows. `render_neuron` measures the real token count and drops detail in order: weaker marks per line (8, 6, 4, 3, 2, 1), then narrower windows (18, 12, 8 tokens), and as a last resort the final examples. Typical neurons are not affected.
3. **Random examples** are cut around their strongest token in the window, or around the recorded position when nothing is active.
4. **Example text is escaped.** Literal `[`, `]`, `\` and line breaks are escaped, so corpus text cannot imitate a mark or start a forged line. `EXPLAIN_SYSTEM` explains the escapes.
5. **Unclear explanations** begin with "unclear" and may add a labeled guess, so the test writer still has something to work from. `TEST_SYSTEM` says how to handle them.
6. **`PROMPT_HASH`** also covers the render rules (counts, thresholds, ladder, line templates, escapes), because changing them changes what Claude sees.

## Test evidence
RED (before any implementation), `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_explain.py -v'`:
```
tests/atlas/test_explain.py:7: in <module>
    from probe.atlas import explain, prompts
E   ImportError: cannot import name 'explain' from 'probe.atlas'
ERROR tests/atlas/test_explain.py
=============================== 1 error in 0.45s ===============================
```
The brief expected `ModuleNotFoundError`; the `from package import module` form reports the same missing module as `ImportError`.

GREEN, same command:
```
55 passed
```
Whole atlas suite, `pytest tests/atlas -q -m 'not gpu and not claude'`:
```
234 passed, 4 deselected
```
The 4 deselected tests need the GPU or the real `claude` CLI and were not run. No real `claude` calls were made.

Mutation check: 23 deliberate faults were introduced into a scratch copy of the three modules (for example removing the split marker, swapping histogram regimes, reversing the id mapping). In the first round of 22 faults, 21 were caught. The survivor (ignoring the per-line mark cap) led to a new test, which catches it. The 23rd fault (percent formatting) was caught by the test added for that fix. The full set was not rerun against the final code, which changed afterwards only in formatting and in moving the escape table into the hash.

## Prompt tokens per neuron
Counted with the real Qwen tokenizer on synthetic recordings of 40 examples:

| case | tokens | marks kept |
|---|---|---|
| Sparse English (typical) | 1,460 | 50 |
| Dense English, every token marked | 2,444 | 240 (capped at 6 per line) |
| Random vocabulary, every token marked (worst case) | 2,115 | 160 (capped at 4 per line) |

All 40 examples and the full 24-token windows were kept in the three cases. The hard limit is 2,500. `EXPLAIN_SYSTEM` adds about 700 tokens once per call, about 35 per neuron in a batch of 20.

## Files
Created:
- `src/probe/atlas/batches.py`
- `src/probe/atlas/prompts.py`
- `src/probe/atlas/explain.py`
- `tests/atlas/test_explain.py`

Changed:
- `tests/atlas/fixtures/fake_claude.py` (new `explain` mode)

## Self-review findings, all fixed
- A share of 100% printed as `1e+02%`.
- The id-mapping test could not detect swapped ids, because all fake replies were identical. Replies now differ per id, and a new test checks that each block shows its own neuron's text.
- No test covered the per-line mark cap (found by the mutation check).
- A random example with no activation was cut around window position 0 instead of its recorded position.
- A damaged batch file holding valid JSON of the wrong shape crashed the progress count.
- `PROMPT_HASH` did not cover the render rules or escapes.
- A literal U+2028 character in the source was replaced by its escape.

## Concerns
1. **Schemas are untested against the real CLI.** `test_schema` and `rate_schema` use `minItems`, `maxItems`, `minimum` and `maximum`. The captured fixture suggests `--json-schema` works through a tool call, but no real call was allowed here. Tasks 11 and 13 should validate counts and ranges locally as well, as `explain.py` does, and the pilot should confirm the schemas are accepted.
2. **Budget tokens are Qwen tokens.** Claude's tokenizer counts differently, so real usage per neuron will differ somewhat. The pilot's measured usage is the figure to trust.
3. **Windows hold no gate information.** A marked neighbouring token can be a gate-negative activation, which is not part of a pole by the spec's definition. The peak of a pole's example is always gate-positive. The recording cannot separate the neighbours.
4. **Split characters.** `tok.decode([id])`, as the brief requires, shows a multi-byte character split across tokens as replacement characters. This may reduce explanation quality on some non-English text.
5. **Memory.** The shard cache can hold the whole corpus, about 200 MB, and the layer cache about 100 MB.

---

# Fix round 1 (review findings, rulings R19-R21)

Status: DONE_WITH_CONCERNS. One test outside this round's permitted files now fails (see "Open items").

## Changes
1. **R19, windows (critical).** `recorder.py`: `pos`, `neg` and random windows now hold only eligible gate-positive positions, and `dn` windows only eligible gate-negative positions. Everything else is NaN. Values, histograms, totals and `rnd_regime` are unchanged. The docstring and the brute-force oracle in `test_recorder.py` follow the new rule. Two tests were added, and the interior-ineligible assertion now expects NaN.
2. **Peak marks.** `prompts._marked`: P and N lines always keep their own peak when it qualifies. Other marks rank by `abs(a) / own pole's extreme`.
3. **Schema check and local validation.** Added `prompts.validate_tests(data, ids=None)` and `prompts.validate_ratings(data, ids=None)`, which raise `LLMFailed`. The optional `ids` argument also checks the exact id set.
4. **R20, outages.** `batches.py`: 5 consecutive single-item failures other than `Refused` raise `LLMFailed`, and none of those items is saved as failed. Held-back failures are stored once a later call succeeds or is refused, or when the run ends.
5. **R21, golden render.** `PROMPT_HASH` now covers two fixed synthetic neurons rendered through a stub tokenizer, one sparse and one dense enough to use the detail ladder.
6. **Multi-byte text.** `prompts.text_groups` joins tokens until the text no longer ends in U+FFFD, up to 4 tokens, and shows the group's strongest activation. If a group never completes, it falls back to the single token. Used in both the prompt and the viewer payload.

## Real schema calls
Exactly 3 real calls were made, one per schema, through `llm.call_claude` with a single id and `--effort low`, with no retries:

| schema | CLI result | local validation | input tokens | output tokens |
|---|---|---|---|---|
| explain | accepted | not applicable (checked in `explain.py` in production) | 2,487 | 261 |
| test (`minItems`, `maxItems` = 10) | accepted | `validate_tests` passed | 1,888 | 1,285 |
| rate (`minimum`, `maximum`, `minItems`, `maxItems`) | accepted | `validate_ratings` passed | 1,455 | 79 |

## Test evidence
Command: `nix-shell --run "OMP_NUM_THREADS=4 pytest tests/atlas/test_explain.py tests/atlas/test_recorder.py -q --no-header"`
```
144 passed in 7.68s
```
Command: `nix-shell --run "OMP_NUM_THREADS=4 pytest tests/atlas -q --no-header -m 'not gpu and not claude'"`
```
FAILED tests/atlas/test_record.py::test_keep_columns_is_idempotent - assert F...
1 failed, 266 passed, 4 deselected in 20.79s
```
Mutation check of this round: 18 deliberate faults in a scratch copy. 15 were caught at first; the 3 survivors (N-line peak, repeated rating id, split character in the viewer payload) each got a new test and are now caught.

Prompt tokens per neuron after the fixes (Qwen tokenizer, synthetic): typical sparse 1,460; dense English with every token marked 2,444; random vocabulary with every token marked 2,117.

## Files changed in this round
- `src/probe/atlas/recorder.py`
- `src/probe/atlas/batches.py`
- `src/probe/atlas/prompts.py`
- `src/probe/atlas/explain.py`
- `tests/atlas/test_recorder.py`
- `tests/atlas/test_explain.py`

## Open items
1. **`tests/atlas/test_record.py:421` fails because of R19, and I did not edit it.** That file is outside the files permitted for this round. The test compares `rnd_win` with `np.array_equal`, which treats NaN as unequal, and windows now contain NaN. The fix is to add `equal_nan=True` on that line. With that one change, applied to a scratch copy only, all 23 selected tests in the file pass.
2. **GPU memory was not re-measured.** R19 adds two temporary activation tensors per document, about 117 MB extra at 4,096 tokens and 3,584 neurons. The Task 8 worst-case figure of 6.03 GiB predates this change, and the GPU tests were not run in this round.
3. **The schema check covers acceptance, not enforcement.** The CLI accepted the constraints, but the calls cannot show that it would reject a wrong count, so Tasks 11 and 13 must call the new validators.

---

# Follow-up: test_record.py fix and GPU re-measurement

Status: DONE_WITH_CONCERNS. Open items 1 and 2 of fix round 1 are closed; one new observation about recording speed is below.

## test_record.py
- Line 421 now compares `rnd_win` with `equal_nan=True`.
- The other array comparisons in the file were checked. `assert_recordings_equal` already uses `equal_nan=True` for floating-point arrays, and its `array_equal` branch only sees non-float arrays. Lines 419, 420, 422, 424 and 468 compare `pos_val`, `hist`, `rnd_doc`, `fire_share` and `neurons`, none of which hold NaN windows. No other change was needed.

## GPU memory after R19
`pgrep -af "probe\."` found no other pipeline job before the runs. The desktop was using about 4.9 GiB of the 12 GiB card, including a running game (How to Fish, about 1.2 GiB).

Command: `nix-shell --run "OMP_NUM_THREADS=4 pytest tests/atlas/test_record.py -v -s --no-header -k test_worst_case_batch_fits_on_the_gpu"`
```
worst case: 10 x 4096 tokens, 7 layers, peak 6.03 GiB, 4434 tokens/s over 9.2 s including file writes
PASSED
```
- Peak from `torch.cuda.max_memory_allocated()`: **6.03 GiB** at `batch_tokens` = 40,960, the same as before R19.
- No OOM retry: the test's `wait` callback fails the test if a retry is attempted, and it passed.

## Speed with and without R19
Three alternating runs each, on the same machine state. The "without" runs used a scratch copy with R19's two masks removed.

| run | with R19 (tokens/s) | without R19 (tokens/s) | peak, both |
|---|---|---|---|
| 1 | 4,586 | 4,430 | 6.03 GiB |
| 2 | 4,356 | 4,450 | 6.03 GiB |
| 3 | 4,579 | 4,437 | 6.03 GiB |
| mean | 4,507 | 4,439 | |

R19 has no measurable speed cost.

## Whole suite
Command: `nix-shell --run "OMP_NUM_THREADS=4 pytest tests/atlas -q --no-header -m 'not claude'"` (includes the GPU tests)
```
271 passed in 188.15s (0:03:08)
```

## Concern
Task 8 recorded 6,955 tokens/s for this test, and today's runs reach about 4,500 both with and without R19, so R19 is not the cause. The game sharing the GPU is a likely explanation, but I did not test that. If the lower speed persists on an idle GPU, recording 50M tokens takes about 3.1 h instead of about 2.0 h at these worst-case rates.
