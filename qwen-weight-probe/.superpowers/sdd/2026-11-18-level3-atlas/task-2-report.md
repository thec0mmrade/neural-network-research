# Task 2 report: status file and overall progress

Built `src/probe/atlas/status.py` (no torch; imports only `probe.atlas.store`): STAGES, DEFAULT_HOURS, `overall_fraction`, class `Status` with public `path` (R3) and all brief methods, plus an extra `flush()` (forced write, so a run can save final state despite throttling).

Files created: src/probe/atlas/status.py, tests/atlas/test_status.py. Nothing else changed.

Behavior: `_tick()` runs at the start of every method and adds elapsed time to paused_s (if paused) else to active_s and the current stage's active_s. Writes throttled to 1/s; forced on stage change (name differs), paused, resumed, flush. `usage()` adds to totals and increments calls. Constructor reloads an existing file (started_at, counters, stages, layers, usage...); pause resets to None on restart. overall_fraction: per-stage hours = default, or active_s/3600/fraction once fraction >= 2% and active_s > 0; result = sum(hours*frac)/sum(hours).

RED: with status.py absent, `pytest tests/atlas/test_status.py` -> ModuleNotFoundError: No module named 'probe.atlas.status' (collection error).
GREEN: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_status.py -v'` -> 6 passed (3 brief tests + below-2% default, throttle/forced writes, fields + restart reload). Whole tests/atlas: 14 passed.

Self-review: tests assert file contents, not internals. Note: I wrote the impl before capturing RED, then temporarily moved the file away to show RED. FakeClock in the test exposes time() (Status calls clock.time()).
Concerns: a restart's downtime is not counted (first tick starts from construction time). Stage with done > total is clamped to 1.0. Throttled non-forced writes may be lost on crash; call flush() at end.
