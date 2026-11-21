# Final fix report: I1, I2, M3, M4, M5, M6

Status: DONE_WITH_CONCERNS (the concerns are in the last section; none of them blocks the run).
M7 is not in this wave (R30).

PROMPT_HASH and the prompts did not change, and neither did `config.py`. The pilot gate is unaffected.
`record.py` changed only in the text of one error message (M6), so the GPU rehearsal was not rerun.
`llm.py` changed in ways the GPU rehearsal also exercises (the cwd, the heartbeat thread and the failure classes). The CPU suite covers all of those through `fake_claude`.

All line numbers refer to the files after the fix.

## I1: claude's working directory

**Changes**
- `src/probe/atlas/llm.py:47`: `LLM_CWD = ~/.cache/qwen-atlas/llm_cwd`, outside every folder that holds a CLAUDE.md.
  - Tests move it into `tmp_path` (conftest `fake_cfg`), so they never touch the real folder.
- `src/probe/atlas/llm.py:132` `clean_cwd()` creates the folder if it is missing. It raises `RunStopped` in two cases:
  - a `CLAUDE.md`, `CLAUDE.local.md` or `.claude/CLAUDE.md` exists in the folder or any folder above it;
  - the folder is not empty.
- `src/probe/atlas/llm.py:158`: `call_claude` uses `clean_cwd()` instead of `cfg.root / "llm_cwd"`.

**Tests**
- `tests/atlas/test_llm.py:264` `test_default_cwd_is_outside_the_project_tree`
- `tests/atlas/test_llm.py:271` `test_cwd_below_a_claude_md_or_not_empty_stops_the_run`
- `tests/atlas/test_llm.py` `test_prompt_via_stdin_and_empty_cwd_and_clean_env`, adjusted to the new cwd.

**Verification: two real calls, no others**

Both calls went through `llm.build_argv` and `llm._run`, which builds the child environment with `child_env`.
- Arguments: `--effort low`, a one-key schema, the prompt "Reply with ok set to true.", and no `--bare`.
- ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN were absent; the script asserted this.
- The script is at `/tmp/claude-1000/-home-c0mmrade-Documents-claude-Neural-Network-Research/0adaa020-5d51-4513-8553-91aacc8efadb/scratchpad/i1/verify.py`.

| call | exact cwd | input tokens (total) | breakdown (input / cache_creation / cache_read) | output |
|---|---|---|---|---|
| 1, new cwd | `/home/c0mmrade/.cache/qwen-atlas/llm_cwd` | **1,034** | 2 / 1,032 / 0 | 52 |
| 2, in-tree (the old location) | `/home/c0mmrade/Documents/claude/Neural_Network_Research/qwen-weight-probe/results/07_atlas/llm_cwd` | **1,057** | 2 / 336 / 719 | 52 |

Both calls returned `structured_output {"ok": true}`, `is_error: false`, model `claude-opus-5-5`.
- After call 2, the temporary `results/07_atlas/llm_cwd`, and `results/07_atlas` along with it, were removed. Both were empty.
- `~/.cache/qwen-atlas/llm_cwd` is still empty.

**Finding**
- The in-tree call costs only 23 more tokens. That fits a longer cwd path in the environment block, not ~3.5k tokens of CLAUDE.md.
- So with `--setting-sources ""`, the three ancestor CLAUDE.md files were already *not* loaded, and no priming leak happened before this fix.
- The move still matters. It meets spec section 5 literally, and the `clean_cwd` guard keeps the run safe if a later Claude Code version changes how `--setting-sources` treats memory files.

**User-level `~/.claude/CLAUDE.md`**
- The file does not exist, and neither does `~/.claude/rules/`.
- No ancestor of the new cwd (`~/.cache/qwen-atlas`, `~/.cache`, `~`, `/home`, `/`) holds a CLAUDE.md or CLAUDE.local.md.
- It therefore cannot load. Whether `--setting-sources ""` would exclude it if it existed was not tested directly; the in-tree result suggests memory files are excluded.

## I2: outage handling

There are now three kinds of failure. The module docstring in `llm.py:13-19` describes them.

**Changes in `src/probe/atlas/llm.py`**
- `:74` `class TransientFailure(LLMFailed)` covers transport and service failures:
  - timeout (`:124`);
  - non-JSON or empty output (`:177`);
  - JSON `is_error` that is not a limit and not a content error, which includes 5xx, overloaded, auth and network errors (`:189`);
  - an OSError other than not-found or permission (`:164`).
- `:84` `class RunStopped(Exception)` stops the run. It is deliberately **not** an LLMFailed, so `batches` never catches it.
  - A missing or non-executable `claude` binary raises it at once (`:162`).
- Content failures stay plain `LLMFailed`:
  - schema or validation problems: no `structured_output`, missing keys (`:191-195`), and the work functions' own validation;
  - prompt or context too long;
  - CLI subtype `error_max*`, e.g. `error_max_structured_output_retries`;
  - API status 400, 413 or 422 (`:49`, `:187`).
- `:237` `call_with_wait`:
  - **Transient** (`:276`): waits 30, 120, 300, 600 and then 900 s between retries of the *same call* (`OUTAGE_BACKOFF_S`, `:42`).
    - Status shows a pause whose reason starts `network: Claude unreachable, retry N: …`, with `since` and `resumes_at`.
    - After 12 h of continuous outage (`OUTAGE_MAX_S`, `:43`) it raises `RunStopped`: "Claude has been unreachable for 12 h (since …); last error: …. Fix the network or the claude login, then rerun; the run resumes where it stopped and no neuron was marked failed".
  - **Limit** (`:262`): unchanged up to the 8-day cap. At the cap it raises `RunStopped` (`:267`) instead of `LLMFailed`, so nothing is split and no second 8-day wait follows.
  - **Content**: unchanged. Three retries at 30, 120 and 300 s, then `LLMFailed`, which `batches` splits on.
  - `_stop` (`:228`) clears the pause and writes "Run stopped: …" to the status note before raising.
- `src/probe/atlas/batches.py:54`: `except TransientFailure: raise` comes before the `Refused`/`LLMFailed` handlers. A transient failure never splits, never writes a split mark and never marks an item failed.
  - `RunStopped` and `LimitReached` pass through untouched.
  - The docstring is updated.
- `src/probe/atlas/batches.py:11-14` revises R20. Outages no longer reach `batches`, so `MAX_FAILURES_IN_A_ROW = 5` stays only as a backstop for systematic content failures. It still stores nothing, so a rerun retries.
- `src/probe/atlas/__main__.py:46` `_log_error` now writes a plain `RunStopped: <message>` line to errors.log before the traceback. The one-line stderr message is unchanged.

**Tests** (all with `fake_claude` and fake clocks, so no real waits)
- `tests/atlas/test_llm.py:283` `test_transport_failures_are_transient` covers fail (crash), garbage, empty and api_500.
- `:289` `test_timeout_is_transient`
- `:294` `test_content_failures_are_not_transient` covers missing_key, schema_retries and context.
- `:302` `test_missing_binary_stops_the_run`
- `:309` `test_short_outage_waits_and_retries_the_same_call`: slept [30, 120], each pause reason starts "network", the pause clears afterwards and one usage is recorded.
- `:323` `test_outage_over_12_hours_stops_the_run_with_a_clear_message`: 50–60 calls over 12 h, then `RunStopped`, which is not an `LLMFailed`. The pause clears and the note says "unreachable".
- `:344` `test_outage_never_splits_or_marks_neurons_failed_and_rerun_pays_once` runs the real `process` + `call_with_wait` + `fake_claude`:
  - the outage leaves no `split_*.json` and no `batch_*.json`;
  - the rerun answers all 8 items with no failed records, in exactly 2 calls for 2 chunks.
- `:358` `test_long_limit_never_splits`
- `:366` `test_transient_failure_from_work_is_never_split`
- `:201` `test_limit_longer_than_8_days_stops_the_run`, changed from `LLMFailed` to `RunStopped`.
- `:103` `test_content_failures_retry_three_times_with_backoff` replaces the old "fail" test, because "fail" is now a transport failure.
- `tests/atlas/test_runner.py:802` `test_run_stopped_writes_a_plain_line_to_errors_log`
- New fake modes in `tests/atlas/fixtures/fake_claude.py`: `empty`, `api_500`, `schema_retries`, `fail_twice_then_ok`, `slow`.

## M3: heartbeat during calls and waits

**Changes**
- `src/probe/atlas/llm.py:203` `_heartbeat(status)`: a daemon thread calls `status.beat()` every `HEARTBEAT_S = 15` s (`:44`).
  - It wraps every Claude call and every wait, whether limit, outage or content backoff.
  - The thread is joined before the main thread touches the status again, so writes never overlap.
  - An OSError in a beat is ignored and the next beat tries again.
- `llm.py:256-261`: `status.calling(since)` before each call and `status.calling(None)` after it, in a `finally`.
- `src/probe/atlas/status.py` gains:
  - `beat()` (`:106`) and `calling()` (`:110`), both forced atomic writes through the existing `atomic_write_json`;
  - a `call` field, written in `:77`;
  - an optional `since` argument for `paused()` (`:116`).
- `src/probe/atlas_watch.py:213-217`: when no pause is active, the viewer shows "running  calling Claude since HH:MM:SS (Xm)".

**Tests**
- `tests/atlas/test_llm.py:374` `test_status_heartbeat_during_a_call`: at least 3 beats during a 0.8 s call with a 0.1 s heartbeat, each with `call.since` set and a distinct `updated_at`; `call` is None afterwards.
- `:386` `test_status_heartbeat_during_a_pause`
- `:395` `test_heartbeat_default_is_under_20_seconds`
- `tests/atlas/test_status.py:107` `test_beat_and_calling_write_at_once`
- `tests/atlas/test_watch.py:85` `test_call_in_flight_shows_calling_claude_and_is_not_stale`

## M4: a dead run's pause no longer shows as PAUSED

**Changes**
- `src/probe/atlas/status.py`:
  - `pid` field (`:54`, `:77`);
  - `started()` (`:94`) records `os.getpid()` and clears the pause and call;
  - `closed()` (`:100`) sets pid, pause and call to None, then flushes, which keeps R9's exit flush.
- `src/probe/atlas/runner.py:230`: `status.started()` right after the Status is created, inside the lock. This clears the pause when the run starts.
- `src/probe/atlas/runner.py:236`: `status.closed()` in `finally` replaces `status.flush()`.
- `src/probe/atlas_watch.py`:
  - `process_alive()` (`:261`);
  - `run_state()` (`:274`) returns "stopped" when `pid` is present and is either None or a dead process. It returns "stale" when there has been no write for `STALE_S` (60 s). Pauses are no longer exempt, because they now send heartbeats;
  - `_banner()` (`:288`) shows "STOPPED: the run is not running (last update … ago)" or "STALE: no update for …";
  - the right panel (`:191-219`) shows "stopped (was paused: …)" in place of PAUSED for a dead run, and "no heartbeat: run may have stopped" under a stale pause.
- Status files without a `pid` key fall back to the time check alone.

**Tests**
- `tests/atlas/test_watch.py`:
  - `:50` (adjusted: a pause with fresh updates is not stale);
  - `:63` `test_pause_without_heartbeat_is_stale`;
  - `:69` `test_pause_of_a_dead_process_shows_stopped_not_paused`;
  - `:75` `test_exited_run_shows_stopped`;
  - `:79` `test_live_pid_with_fresh_updates_is_running`;
  - `:92` `test_odd_pid_and_call_never_crash`.
- `tests/atlas/test_status.py`:
  - `:89` `test_started_records_pid_and_clears_a_pause_left_by_a_dead_run`;
  - `:99` `test_closed_clears_pid_pause_and_call`.
- `tests/atlas/test_runner.py:782` `test_run_clears_a_stale_pause_and_marks_itself_stopped`

## M5: SIGTERM and SIGHUP

**Changes**
- `src/probe/atlas/__main__.py:57` `_exit_on_signal` raises `SystemExit(128 + signum)`, and `:65` `_install_signal_handlers()` installs it for SIGTERM and SIGHUP.
- The handlers are installed under `if __name__ == "__main__"` (`:89`), not in `main()`, so tests that call `main()` don't change pytest's own handlers.
- The existing `except BaseException` in `llm._run` (`llm.py:125`) then kills claude's process group and reaps it. The runner's `finally` frees the GPU, writes `status.closed()` and releases the lock.

**Tests**
- `tests/atlas/test_llm.py:414` `test_sigterm_and_sighup_kill_the_claude_child`, run once for each signal. A child Python installs the handlers and runs `call_claude` on fake `hang`. The test sends the signal and checks two things:
  - the child exits with 128 + sig;
  - the fake claude grandchild is gone.
- `tests/atlas/test_runner.py:815` `test_python_dash_m_installs_sigterm_and_sighup_handlers`

## M6: OOM stop message

**Change**
- `src/probe/atlas/record.py:177-179`. The message now says the last checkpoint is intact, then: "Free GPU memory (close games and other GPU apps) and rerun to resume from it. Do NOT change batch_tokens: that invalidates the recording checkpoint".
- Only the message text changed; the logic is untouched. `batch_tokens` feeds `_plan_hash` (`record.py:120`), so the warning is accurate.

**Test**
- `tests/atlas/test_record.py:297` `test_persistent_oom_stops_clearly_and_keeps_the_checkpoint` now matches the new wording. It is a CPU test and passes.

## CPU test summary

`nix-shell --run 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 pytest tests/atlas -m "not gpu and not claude" -q'`
→ **491 passed, 15 deselected in 112.40 s**.

Before the fixes, the same suite collected fewer tests. Every new test passes, and every adjusted old test passes.

## Files changed

- `src/probe/atlas/llm.py`
- `src/probe/atlas/batches.py`
- `src/probe/atlas/status.py`
- `src/probe/atlas/runner.py`: two lines, `started()` and `closed()`, required by M4's "clear the pause when the run starts"
- `src/probe/atlas/__main__.py`
- `src/probe/atlas/record.py`: message only
- `src/probe/atlas_watch.py`
- `tests/atlas/conftest.py`: `fake_cfg` moves `LLM_CWD` into `tmp_path`
- `tests/atlas/fixtures/fake_claude.py`: new modes
- `tests/atlas/test_llm.py`
- `tests/atlas/test_status.py`
- `tests/atlas/test_watch.py`
- `tests/atlas/test_runner.py`
- `tests/atlas/test_record.py`: the M6 regex only
- `.superpowers/sdd/2026-11-18-level3-atlas/final-fix-report.md`: this report

Outside the project, created by the run code: the empty folder `~/.cache/qwen-atlas/llm_cwd`.

Not changed: `config.py`, `prompts.py`, PROMPT_HASH and the project CLAUDE.md.

## Concerns

1. **Reading of "after a limit wait, if the call still fails, stop the run".** I implemented this as follows:
   - a limit that is still in force at the 8-day cap stops the run with `RunStopped`;
   - a transient failure after a limit wait follows the outage path, which waits up to 12 h and then stops;
   - a *content* failure right after a limit wait still goes the normal content path (retry, then split). It is an item problem, and stopping a days-long run for it seemed wrong.

   If the stricter reading is wanted (any failure after a limit stops the run), it is a three-line change in `call_with_wait`.
2. **`child_env` leaves some Claude Code session variables when the run is launched from inside a Claude Code session.** Examples: `CLAUDE_CODE_CHILD_SESSION`, `CLAUDE_CODE_SESSION_ID`, `CLAUDE_CODE_MESSAGING_SOCKET`/`_TOKEN`, `CLAUDE_EFFORT`, `CLAUDE_PID`.
   - R18 drops only the listed markers. My two verification calls ran with these variables present and both worked on the Max login (model `claude-opus-5-5`, no error).
   - `--effort low` was honored as far as can be seen.
   - If the real run will be started from inside a Claude Code session, consider widening R18 so `child_env` also drops every `CLAUDE_CODE_*` and `CLAUDE_EFFORT`/`CLAUDE_PID`. I did not change it, because that is outside this wave's findings.
3. **I1's leak was not happening in practice.** The ~23-token difference shows `--setting-sources ""` already kept the ancestor CLAUDE.md files out. The fix is still in place, for spec compliance and future-proofing.
4. **The GPU rehearsal was not rerun.** `llm.py` is on the rehearsal's path: the cwd moves to `~/.cache/qwen-atlas/llm_cwd`, and the heartbeat thread runs during fake calls. Both are covered on CPU with `fake_claude`. If you want end-to-end certainty before the multi-day run, the 68-minute rehearsal would confirm it.

## Addendum: child_env widened (coordinator ruling on concern 2)

Concern 2 is resolved. The coordinator accepted concerns 1 and 4 as they stand.

**Change.** `src/probe/atlas/llm.py:50-57` and `child_env` at `:93`.

The child environment now drops:
- every variable whose name starts with `ANTHROPIC_` or `CLAUDE`. That covers `CLAUDECODE`, every `CLAUDE_CODE_*`, `CLAUDE_EFFORT`, `CLAUDE_PID`, and the earlier R18 provider switches and nesting markers.
- `AI_AGENT`, Claude Code's agent marker. Its value here is `claude-code_2-1-316_agent`.

It keeps `HOME`, `PATH`, `XDG_*` and everything else. It also keeps `CLAUDE_CONFIG_DIR`, the one exception, because that variable moves `~/.claude` and with it the login. It is not a session marker, and it isn't set on this machine.

**Variable names found in this session** (`env | grep -i claude`, names only):
- Removed: `CLAUDECODE`, `CLAUDE_CODE_CERT_STORE`, `CLAUDE_CODE_CHILD_SESSION`, `CLAUDE_CODE_ENTRYPOINT`, `CLAUDE_CODE_EXECPATH`, `CLAUDE_CODE_MESSAGING_SOCKET`, `CLAUDE_CODE_MESSAGING_TOKEN`, `CLAUDE_CODE_SESSION_ATTENDED`, `CLAUDE_CODE_SESSION_ID`, `CLAUDE_EFFORT`, `CLAUDE_PID` and `AI_AGENT`.
- Kept: `PWD`, `OLDPWD` and `TERMINULL_VOLUMES_DIR`. They matched only because their values contain the word "claude"; they are not markers.

No `CLAUDE_CODE_*` variable is needed for the OAuth login. The login is read from `~/.claude`, which the child finds through `HOME`, and the real call below succeeded with every one of them removed.

**Test.** `tests/atlas/test_llm.py:26` `test_child_env_strips_every_claude_session_marker_and_keeps_the_login_path` builds a fake environment holding all the markers above plus `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_BASE_URL`. The output contains exactly `HOME`, `PATH`, `XDG_CONFIG_HOME`, `XDG_RUNTIME_DIR`, `CLAUDE_CONFIG_DIR` and `PWD`.

**Third real call.** This call went through the real `build_argv`, `_run` and `child_env` path, from `/home/c0mmrade/.cache/qwen-atlas/llm_cwd`, with effort low and no `--bare`.
- Before the call, the child environment held no names containing CLAUDE or ANTHROPIC, and no `AI_AGENT`.
- Result: rc 0, `is_error: false`, `structured_output {"ok": true}`, model `claude-opus-5-5`.
- **Input tokens: 1,034.** That is 2 input, 0 cache creation and 1,032 cache read, with 52 output tokens. It matches the first clean call exactly.
- No API key was present, so the call ran on the subscription login.
- The cwd is still empty afterwards.

**CPU suite after the change:** **492 passed, 15 deselected in 111.38 s.**

**Files changed in this addendum:** `src/probe/atlas/llm.py`, `tests/atlas/test_llm.py` and this report.
