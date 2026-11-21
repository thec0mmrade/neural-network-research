# Task 13 report: research question and final report

Status: DONE_WITH_CONCERNS (one required test-stub change is outside this task's allowed files; see Concerns 1).

## What I built

`src/probe/atlas/research.py`
- `study_neurons(cfg)`: union of `edited_neurons("p50c")[l]` and the top 64 columns by `np.argsort(-np.abs(u @ W))`, for L15 and L19.
- `control_neurons(cfg, summary, study)`: seeded random live neurons, disjoint from the study set, as many as `len(study[l])`. Raises ValueError if too few remain.
- `rate_blind(cfg, status, study, control, out)`: pools and shuffles items with a seeded RNG, assigns ids `r001`, ..., and sends only explanation texts. Calls `prompts.validate_ratings`. Batches are resumable; `key.json` refuses stored ratings from a different item list.
- `permutation_test(study, control, seed)`: group means, difference, two-sided p from 10,000 seeded permutations, p = (1 + extreme) / (1 + permutations).
- `run_research(cfg, base, status)` (R27): reuses the layer run's explanations and dominant-pole scores, writes and scores only the non-dominant pole's tests in `research/<version>/L<layer>/<pole>/{tests,scores}` (R23), frees the model before all Claude work, rates each pole's explanation, and writes `research/<version>/research.json` last.

`src/probe/atlas/report.py`
- `component_id`, `component_name`, `wiring_path`, `wiring_join(cfg, layer, idx, k=8)`.
- `write_report(cfg)`: writes `cfg.root / "atlas_neurons.jsonl"` (one row per in-scope neuron, status `live`, `dead` or `failed`) and `cfg.root.parent / "07_atlas.md"` (`results/07_atlas.md` in the real run). Sections: Method, Coverage, Primary score, AC-matched score, Job labels, Research question, Wiring, Limits. Absent research data, wiring data or unscored layers are stated in the text.

`src/probe/atlas/runner.py`: `_research` now calls `run_research(cfg, base, status)`.

## Decisions the brief left open

- probe.wiring is not imported. Measured: importing it sets OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and NUMBA_NUM_THREADS to 4 in the process environment. The index is computed directly, and a test checks ids and names against `probe.wiring.components()`.
- The neuron is the unit of every permutation test (its value is the mean over its poles), because a neuron's two poles are not independent.
- In group comparisons a failed pole scores zero, matching the primary score's convention. Poles without a rating are left out of rating means. Failures are counted by kind.
- Controls match `len(study[l])`, the brief's literal test, so dead studied neurons make the analysed control group slightly larger than the analysed study group.
- Paths derive from `cfg.root.parent`, so tests never touch the real `results/`.

## Test evidence

RED (before implementation):

    $ nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_report.py -v'
    E   ImportError: cannot import name 'report' from 'probe.atlas'
    ERROR tests/atlas/test_report.py
    =============================== 1 error in 1.15s ===============================

The brief expected ModuleNotFoundError; the `from probe.atlas import report` form raises ImportError for the same missing module.

GREEN:

    $ nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_report.py -v'
    ============================== 41 passed in 14.46s ==============================

Full CPU suite, as delivered:

    $ nix-shell --run "OMP_NUM_THREADS=4 pytest tests/atlas -q -m 'not gpu'"
    FAILED tests/atlas/test_runner.py::test_waiting_note_is_cleared_once_approved
    FAILED tests/atlas/test_runner.py::test_layers_run_in_the_required_order
    FAILED tests/atlas/test_runner.py::test_stages_run_in_order
    FAILED tests/atlas/test_runner.py::test_restart_skips_finished_stages
    FAILED tests/atlas/test_runner.py::test_at_most_one_model_is_loaded_at_a_time
    5 failed, 426 passed, 5 deselected in 60.82s

All five fail with `TypeError: StubbedRunner._research() missing 1 required positional argument: 'status'`, because that stub still has the pre-R27 signature.

Same suite on a scratch copy of tests/atlas with only that stub's signature changed:

    431 passed, 5 deselected in 60.83s

GPU-marked tests were not run. No real `claude` call was made and no real run was started.

Mutation check (scratch copy of src, tests/atlas/test_report.py): the unmutated baseline passes, and all 16 mutants are caught: group leaked into the prompt, no shuffle, failed poles dropped, model kept during rating, dominant pole retested, dead neurons kept, one folder for both poles, controls overlapping the study set, broken p value, key check removed, component id ignoring heads, failed status lost, wiring dropped, research poles dropped, overall counting unscored layers, old runner signature. The overall-counting mutant survived at first; I added an assertion that now catches it.

## Files created or changed

- Created: `src/probe/atlas/research.py`
- Created: `src/probe/atlas/report.py`
- Created: `tests/atlas/test_report.py`
- Changed: `src/probe/atlas/runner.py` (one line in `_research`)

## Self-review findings

Fixed during review: an unused import; a failed record with an empty reason being treated as live; a leftover variable in a test; awkward wording for absent data; the surviving mutant above. A synthetic report was rendered and read in full. `wiring_join` was also run once against the real `edges_raw.npz`.

## Concerns

1. Action needed: `tests/atlas/test_runner.py` line 135 must become `def _research(self, cfg, base, status):`. rules.md does not allow me to change that file, so I left it, and the five tests above fail until it is changed. The scratch run shows this one line is the whole fix.
2. The rating prompt's rendering (`research._rate_prompt`) is not covered by `PROMPT_HASH`; `RATE_SYSTEM` and `rate_schema` are. R21 and R24 cover rendering for the other prompts. Moving it needs a change to `prompts.py`, which this task may not touch, and it would change the hash, so it should be decided before the pilot is approved.
3. Non-dominant poles that never fired still get tests written. They cannot score a hit (`passage_hit` returns False when the extreme is 0), so they cost a few Claude tokens and count as zero. I kept this because the spec says both poles are tested.
4. The three p values are not corrected for multiple comparisons. The report says so.

---

# Fix report: coordinator follow-ups 1 to 3

This section supersedes Concerns 1, 2 and 3 above, and the "as delivered" suite result.

## Changes

1. Stale stub: `tests/atlas/test_runner.py:135` is now `def _research(self, cfg, base, status):`.
2. Rating prompt: `research._rate_prompt` moved to `prompts.rate_prompt(blocks)`, taking (id, item) pairs like `test_prompt`. `_hash_payload` now includes a golden render of it under `"ratings"`, so `PROMPT_HASH` has changed. New tests in `tests/atlas/test_explain.py`: `test_prompt_hash_covers_the_rating_prompt` and `test_rating_prompt_shows_only_the_id_and_explanation`.
3. Untestable poles: a non-dominant pole whose extreme is 0 is marked `"untestable": true` in its row and gets no Claude call of any kind.
   - `research.json` has `untestable_poles: {"study": n, "control": n}`.
   - The report's research table has an "untestable poles" column, and its statistics paragraph explains the treatment.
   - New tests in `tests/atlas/test_report.py`: `test_a_pole_that_never_fired_costs_no_claude_call`, `test_untestable_poles_are_counted_per_group`, `test_untestable_poles_are_left_out_of_the_comparison`, `test_report_states_the_untestable_poles`.

## Decisions to check in review

The follow-up said to skip these poles and count them. It did not say how they enter the analysis, so I chose:
- They are not rated either, because rating also spends Max allowance.
- They are left out of the test-score means, so such a neuron is judged on its dominant pole alone. They do not score zero, because zero is the convention for failures and this is not a failure.
- They are left out of the job-label counts and the failure counts.

## Files changed in this round

- `src/probe/atlas/prompts.py`
- `src/probe/atlas/research.py`
- `src/probe/atlas/report.py`
- `tests/atlas/test_runner.py`
- `tests/atlas/test_explain.py`
- `tests/atlas/test_report.py`

## Test evidence

RED, new tests before the implementation:

    $ nix-shell --run "OMP_NUM_THREADS=4 pytest tests/atlas/test_report.py tests/atlas/test_explain.py -q --no-header -m 'not gpu and not claude'"
    FAILED tests/atlas/test_report.py::test_a_pole_that_never_fired_costs_no_claude_call
    FAILED tests/atlas/test_report.py::test_untestable_poles_are_counted_per_group
    FAILED tests/atlas/test_report.py::test_untestable_poles_are_left_out_of_the_comparison
    FAILED tests/atlas/test_report.py::test_report_states_the_untestable_poles
    FAILED tests/atlas/test_explain.py::test_prompt_hash_covers_the_rating_prompt
    FAILED tests/atlas/test_explain.py::test_rating_prompt_shows_only_the_id_and_explanation
    6 failed, 129 passed in 20.76s

GREEN, full suite:

    $ nix-shell --run "OMP_NUM_THREADS=4 pytest tests/atlas -q --no-header -m 'not claude and not gpu'"
    437 passed, 5 deselected in 62.74s (0:01:02)

Mutation check on a scratch copy: the unmutated baseline passes and 6 of 7 new mutants are caught. The survivor removes the `pole != main` condition. It cannot change results on real data: the dominant pole has the larger extreme, so a live neuron's dominant extreme is above 0. I did not add a test for that unreachable state.

## Remaining concerns

- The three p values are not corrected for multiple comparisons. The report says so.
- Controls match `len(study[l])`, so dead studied neurons leave the analysed control group slightly larger. The real study sets are 97 neurons (L15) and 110 (L19).
- GPU-marked and claude-marked tests were not run.

---

# Fix report: review findings 1 to 5

## Changes

1. Primary and secondary tests (`research.compare`). `comparison.primary` (scope "dominant pole") holds `ratings`, `blind_maximum`, `scores` and `specificity`. `comparison.secondary` (scope "both poles") holds the earlier both-pole `ratings`, `blind_maximum` and `scores`. `comparison.job_labels` is unchanged. The report labels the sections "Primary test: dominant pole" and "Secondary test: mean over both poles", and says why the secondary rows can be biased.
2. Specificity: `primary.specificity` is `blind_maximum` minus `near_miss` per neuron on the dominant pole, with its permutation p value. A failed pole has specificity 0, because both of its shares count as zero.
3. Stratification: `permutation_test` takes `study_layers` and `control_layers` and shuffles group labels within each layer. `compare` passes them for every test.
4. Job labels: each cell in the per-group table now reads `count (share%)`, with the share taken over that group's counted poles.
5. Wiring: `wiring_join` edges now carry `sign`, from `out_sign` and `in_sign`.

## Files changed in this round

- `src/probe/atlas/research.py`
- `src/probe/atlas/report.py`
- `tests/atlas/test_report.py`

## Test evidence

RED, before the implementation:

    $ nix-shell --run "OMP_NUM_THREADS=4 pytest tests/atlas/test_report.py -q --no-header"
    12 failed, 39 passed in 18.27s

Of the 12, 6 were new tests, 2 were existing tests extended with new assertions (wiring signs, report tables), and 4 were existing tests moved to the new `comparison.secondary` keys.

GREEN: the command output is in the section "Final results" below.

False-positive rates on synthetic data with no true group difference (10% vs 0% silent poles, 100 neurons per group, 200 seeded runs, 199 permutations per test, p < 0.05):

| test | rate |
|---|---|
| primary `blind_maximum` | 5.0% |
| primary `specificity` | 4.5% |
| primary `move_on` rating | 4.0% |
| secondary `blind_maximum` (both poles) | 30.0% |

The regression test asserts the primary rate is between 2% and 9%, and that the secondary rate is above 20%, so the test still detects the bias it guards against.

Mutation check on a scratch copy: all 8 mutants are caught. One, `compare` not passing layers, survived at first because only `permutation_test` was tested for stratification. I added `test_compare_shuffles_group_labels_within_each_layer`, which now catches it.

## Remaining concerns

- p values are not corrected for multiple comparisons. The report says so.
- The false-positive rates above come from one synthetic design, not from real atlas data.
- GPU-marked and claude-marked tests were not run.

## Final results

    $ nix-shell --run "OMP_NUM_THREADS=4 pytest tests/atlas/test_report.py -q --no-header"
    52 passed in 35.66s

    $ nix-shell --run "OMP_NUM_THREADS=4 pytest tests/atlas -q --no-header -m 'not claude and not gpu'"
    444 passed, 5 deselected in 81.90s (0:01:21)
