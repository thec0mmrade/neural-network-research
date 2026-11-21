# Task 9 report: Claude Code wrapper

Status: DONE_WITH_CONCERNS

## Real call (Step 1, the only real call)
Run from an empty temp dir with CLAUDECODE and CLAUDE_CODE_ENTRYPOINT unset (the only variables from child_env's list that were set; no ANTHROPIC_* vars existed). rc=0, saved as tests/atlas/fixtures/claude_ok.json.
- Structured output: `structured_output` ({"ok":true}); `result` holds the same as a string. Failure: `is_error`, `subtype`; refusal: `stop_reason`=="refusal" (the success case shows "tool_use").
- Usage: `usage.input_tokens` (2), `usage.cache_creation_input_tokens` (1087), `usage.cache_read_input_tokens`, `usage.output_tokens` (52). Also `total_cost_usd`, `modelUsage`.
- Authentication: there is NO `apiKeySource` field. `modelUsage.<model>.provider` = "firstParty". `claude auth status` (not a model call) shows authMethod "claude.ai", loggedIn true, apiProvider firstParty: the claude.ai OAuth login, not an API key.
- Concern: `claude auth status` reports `subscriptionType: "pro"`, not "max".

## Built
- src/probe/atlas/llm.py: LLMResult, LimitReached, LLMFailed, Refused, child_env (R18), build_argv (never --bare), call_claude (stdin prompt, cwd cfg.root/"llm_cwd", process-group kill on timeout, required-key validation, refusal detection, limit regex + epoch reset parse), call_with_wait.
- tests/atlas/fixtures/fake_claude.py (modes ok, limit, limit_then_ok, garbage, refusal, missing_key, fail, hang, echo), tests/atlas/fixtures/claude_ok.json, tests/atlas/test_llm.py (14 tests).
- Changed: tests/atlas/conftest.py (added `fake_cfg` fixture: wrapper script per mode, claude_bin points to it).

## Evidence
RED: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_llm.py -v'` -> ModuleNotFoundError: No module named 'probe.atlas.llm' (collection error).
GREEN: same command -> 14 passed; with test_status.py: `20 passed in 2.32s`. (First GREEN run had one test failing because Status throttles usage writes; test fixed to read status.usage_, not the file.)

## Design notes
- Limits retried forever (no cap); other failures (incl. garbage output) retried 3 times at 30/120/300 s inside call_with_wait; Refused is not retried. call_claude itself makes a single attempt.
- Limit with no parseable reset time waits 30 min; with an epoch after "|" it waits until then +5 s. Limit text is also detected in non-JSON output.
- Input tokens recorded = input + cache_creation + cache_read.
- status.resumed() is called only if a pause occurred; usage() after each success.

## Self-review / concerns
- Reset-time format of real Max limit messages is unverified (no real limit seen); the epoch-after-"|" form is the known Claude Code format. Unparseable falls back to 30 min, safe.
- Fixture's `stop_reason` is "tool_use" on success; only "refusal" is treated specially.
- Subscription shows "pro": confirm the intended Max login is the one logged in.
- Did not run the full tests/atlas suite (GPU tests slow); ran test_llm and test_status only.

## Fix report (review round 1)
Changes in src/probe/atlas/llm.py:
1. Reset time: `call_with_wait` believes `resets_at` only if now < r <= now + 8 days, else waits 30 min; minimum wait 60 s.
2. `status.resumed()` is now called as soon as a call after a limit returns anything but LimitReached (success, LLMFailed, Refused).
3. Limit detection: `_limit_message` rejects text matching `context|prompt is too long|too many tokens`; non-JSON output is scanned only when rc != 0; `LimitReached.message` carries the matched text into the pause reason; `call_with_wait` takes an injectable `clock` and raises LLMFailed("usage limit persisted for more than 8 days") when one call stays limited longer than 8 days (waits otherwise uncapped).
4. `_kill_group` ignores ProcessLookupError; `_run` also kills the process group if `communicate` raises anything else.
Tests added (10, in test_llm.py; fake_claude.py gained modes `context`, `garbage_limit_rc0`): past epoch, far-future epoch, 60 s minimum, pause cleared on LLMFailed and on Refused, context error, non-JSON limit text with exit 0, pause reason text, 8-day cap, killpg ProcessLookupError, group kill when communicate raises.
Command: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_llm.py -v'` -> `24 passed in 6.38s`.
Note: I wrote fix and tests together and did not capture a separate RED run for these (the first combined run hung on a faulty killpg-stub test, which I fixed). No real claude calls were made.

## Fix report (review round 2)
- `NOT_LIMIT_RE` narrowed to `(?i)context (window|length)|prompt is too long|too many tokens`, so "...session limit, resets 6pm (context: weekly)" is a LimitReached.
- `_run` calls `proc.wait()` after killing the group when `communicate` raises (no zombie). Not separately tested.
- Test renamed to `test_past_reset_epoch_falls_back_to_30_minutes`; new parametrized test `test_context_errors_fail_but_limit_text_with_word_context_is_a_limit` (modes `context` -> LLMFailed, `limit_with_context_word` -> LimitReached; new fake mode added).
- Command: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_llm.py -v'` -> `25 passed in 6.54s`. No real claude calls.
