# Final review findings (to fix in one wave)

I1. llm.py:112: claude's cwd is results/07_atlas/llm_cwd, inside the project tree. Three ancestor CLAUDE.md files (qwen-weight-probe, Neural_Network_Research, claude/) may load into every call, about +3.5k tokens. They would also prime explanations with the project's hypotheses. Spec section 5 requires an empty working directory. Fix: use an empty dir OUTSIDE any folder holding CLAUDE.md, e.g. ~/.cache/qwen-atlas/llm_cwd (create it if missing). Also check whether ~/.claude/CLAUDE.md (user level) exists and whether --setting-sources "" excludes it. VERIFY with exactly ONE real minimal claude call from the new cwd via the real build_argv/child_env path, using a trivial prompt, effort low if supported. Report its input token count. About 1.1k tokens means clean. Then run one call from a dir inside the tree, for comparison. Never --bare. These two calls are the only real Claude calls allowed.

I2. Outage handling (batches.py:11, 55-61; llm.py:171-179). About 1 h of network outage drives the split tree to 5 single-item failures, and the unattended run exits. Split marks made during the outage remain, so the chunks are later paid as ~5 smaller calls. A disbelieved long limit raises LLMFailed after 8 days, then splits and waits again (up to ~11 × 8 days). Required behavior:
 - Transport or transient failures (process crash, network, non-JSON or empty output from an infrastructure error, 5xx, timeouts) must NOT split or mark neurons failed. They wait with backoff and retry the same chunk, capped at 12 h of continuous outage, then exit with a clear errors.log message. The run is resumable.
 - Splitting only happens for content failures (the reply parsed but failed the schema or validation).
 - After a limit wait, if the call still fails, stop the run with a clear message (no splitting, no repeat 8-day wait).
 - Status must show the wait (reuse the pause fields with a reason like "network").
 Keep R20's intent: a temporary outage must never permanently mark neurons failed.

M3. atlas_watch.py:251-255: STALE shows during normal multi-minute Claude calls, because nothing writes status during a blocking call (llm.py:94). Add a heartbeat: status is rewritten at least every 20 s while a call is in flight, e.g. a thread or poll loop in call_claude, plus a "calling Claude since" note in the viewer. Keep atomic writes.

M4. atlas_watch.py:253: PAUSED never clears if the run dies during a pause. The viewer should show STALE/stopped when a pause's process is gone or updated_at is far older than expected under heartbeats. Also clear the pause when the run starts.

M5. llm.py:92: a claude child started with start_new_session=True outlives Python on SIGTERM/SIGHUP. Install SIGTERM/SIGHUP handlers in __main__ that raise, e.g. SystemExit/KeyboardInterrupt, so the existing group-kill cleanup runs.

M6. record.py:177: the OOM stop message should say: free GPU memory (close games and other GPU apps) and rerun, and do NOT change batch_tokens, because that invalidates the recording checkpoint.

M7 (spec 7.3 per-example repeated-text flag) is NOT in this wave. Ruling R30 defers it, since the flag can be rebuilt from the stored doc/token.
