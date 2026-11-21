# Task 3 report: terminal viewer

## Built
`src/probe/atlas_watch.py`: `render`, `layer_bar`, `bottom_bar`, `load_status`, `main` (Live, 2 s refresh, optional path arg, default `Config().root / "status.json"`).
Imports only rich, probe.atlas.config (-> probe.weights, numpy) and probe.atlas.status/store. No torch (checked by a subprocess test).
- Layout: Pipeline panel (stage, done/total, speed, time left; unstarted stages show default hours), Layers panel, Max usage/pause/scores panel (with AC's 76% / 11% / ~1% row), Latest neuron (tokens with background shading by activation), bottom bar.
- Bottom bar is a fixed size=1 row; the rest share height by ratio, so it stays visible down to height 4 (tested at 4, 8, 12).
- Stale = no update for >=60 s and not paused; shown in the Pipeline panel title ("no update for 5m").
- Defensive: `_dict`/`_num` readers, `load_status` returns None for missing, truncated, non-UTF8 or non-object files; `main` wraps render in try/except and shows an error line; Ctrl-C exits cleanly.

## Files
- created `src/probe/atlas_watch.py`
- created `tests/atlas/test_watch.py` (27 tests: the brief's 6 plus fill, all-panels, short terminal, narrow width, 15 odd-input cases, load_status files, no-torch)

## RED
`nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_watch.py -v'`
-> `ModuleNotFoundError: No module named 'probe.atlas_watch'`, 1 error during collection.
A second RED during development: short-terminal test failed (fixed-size panels pushed the bottom bar off screen); fixed by using ratios.

## GREEN
- `pytest tests/atlas/test_watch.py -q` -> 27 passed
- `pytest tests/atlas -q` -> 41 passed
- `nix-shell nix/cpu-shell.nix --run 'timeout 5 python -m probe.atlas_watch ...; test $? -eq 124 && echo VIEWER_OK'` -> VIEWER_OK
- Rendered a real `Status`-written file at 100x24 and eyeballed it.

## Self-review
- Removed an awkward placeholder expression in the layout code and a loop-internal import.
- Shortened score-table headers (blind/near/far) after seeing truncation at width 100.
- Tests assert rendered text and bar geometry, not just "no exception".

## Concerns
- The brief does not define the `latest` neuron schema. I accept `name`, `explanation`, `tokens` as a list of str, `[text, act]` or `{"text","act"}`; Task 12 must write that shape (or tell me the real one).
- At heights under ~24 the lower stages (research, report) are clipped; the bottom bar is never lost.
- Time-left uses `active_s * (1-f)/f` from overall_fraction, so it is rough early in the run.

## Fix round (review items)
Changes in `src/probe/atlas_watch.py`:
1. Bottom bar now reads `elapsed <active_s>  paused <paused_s>  ~left`; elapsed excludes pauses. Pause panel adds a line "paused for X" (uses `pause["since"]` if the writer records it, else "paused X in total" from `paused_s`; Status.paused does not write `since` today).
2. `layer_bar(..., state=None)`: queued is dim, running is bold bright cyan, else score colour. Call site passes `L.get("state")`.
3. Layers panel is a fixed-width column of 91 (60 cells + 31 overhead), shrinking only when width-32 is smaller; the title then shows "(1 char = N neurons)". 60 cells at width >= ~123; fewer below.
4. Latest header shows R10 `score` as `[score]`.
Tests added (6 new, 33 total): paused/elapsed, pause length, state colours, 60 cells at width 160 with no scale note, shrink with correct scale at width 90, score in header.
Command: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_watch.py -q'` -> 33 passed; `pytest tests/atlas` -> all pass; CPU shell viewer -> VIEWER_OK.
Concern: a 120-column terminal gets ~55 cells (scale shown in the title); Task 2's Status could add `since` to the pause dict for an exact pause length.

## Fix round 2: layer row wrap
Cause: ROW_OVERHEAD was 31 but the longest row needs 33 (border+padding 4, "L15  " 5, " 3584/3584 finished 0.70" 24), so the score wrapped. Set `ROW_OVERHEAD = 33` in `src/probe/atlas_watch.py` (full-scale bars now need width >= 125).
Regression test `test_longest_layer_row_stays_on_one_line` (widths 100/120/140/200, states running and finished): each layer is exactly one line with its score. With the old value 31 all 8 cases fail; with 33 they pass.
`nix-shell --run 'pytest tests/atlas/test_watch.py -q'` -> 41 passed; `pytest tests/atlas` -> 55 passed.
