# Sonnet test-writer fix: report (2026-11-19)

**Status: BLOCKED.** The code changes are done and tested (501 CPU tests pass). But Sonnet 5.5 refuses the real test prompt with the same `[reasoning_extraction]` safeguard as Opus. It refuses it even for a single item with context "plain". Writing the tests with Sonnet therefore does not unblock the pilot. Per the brief, I tried no other wording and no other model.

## Real-call results (2 of the 3 allowed calls)

Both calls went through the real `llm.call_claude` path: `build_argv` with `model=cfg.test_model`, then `child_env`, then `clean_cwd` (~/.cache/qwen-atlas/llm_cwd). They used the real `prompts.TEST_SYSTEM`, `prompts.test_prompt` and `prompts.test_schema`, and no `--bare`. The items were v1 pilot explanations from `results/07_atlas/pilot/v1/L14/explain/`. The script and raw results are in the session scratchpad (`sonnet_check/`).

| call | items | succeeded | passes validate_tests | tokens in / out | passage excerpt |
|---|---|---|---|---|---|
| 1 | L14.mlp.180 positive (plain), L14.mlp.762 positive (reasoning) | no: `Refused`, "Sonnet 5.5's safeguards flagged this message … Details: `[reasoning_extraction]`" (req_011ChCNvspDNLbtmVCr5guye), 5 s | n/a | n/a (no usage returned) | none |
| 2 | L14.mlp.180 positive (plain) only | no: `Refused`, same tag `[reasoning_extraction]` (req_011ChCNx74xi8KUcVt6qpwMc), 3 s | n/a | n/a | none |

Call 2 was a diagnostic. It used the same real prompt and model, with only the plain item. It shows that the refusal does not depend on whether a "reasoning" item is present. The constant parts of the request trigger it: the system prompt, which contains the "reasoning passages run inside a model's private thinking … write first-person step-by-step reasoning" rule at prompts.py:63, and/or the schema. That means a prompt change is needed, which in turn needs a new prompt_version and a new pilot. That decision belongs to the user. I did not use the third allowed call.

## Changes (file:line)

**1. A separate test-writing model (Ruling R36)**
- `src/probe/atlas/config.py:37`: `test_model: str = "claude-sonnet-5-5"`.
- `src/probe/atlas/llm.py:34`: `MODEL` stays `claude-opus-5-5`. It is used for explanations and ratings.
- `src/probe/atlas/llm.py:99-101`: `build_argv(cfg, system, schema, model=None)` passes `--model model or MODEL`. `MODEL` is read at call time, so a patched `llm.MODEL` still applies.
- `src/probe/atlas/llm.py:163,167`: `call_claude(..., model=None)`.
- `src/probe/atlas/llm.py:245,267`: `call_with_wait(..., model=None)`.
- `src/probe/atlas/evaluate.py:29-30`: `write_tests` calls with `model=cfg.test_model`. This is the only caller that passes a model. That covers both the pilot/layers path and the research path (research.py:173).
- `src/probe/atlas/pilot.py:30-31`: `prompt_parts` now includes `test_model`. That one dict feeds all of these:
  - `approved.json` (`approve` / `is_approved`)
  - `report.json` (`{**prompt_parts}`)
  - the LLM stage `.done` markers (runner `_marker`)
  - `research.json`
- `src/probe/atlas/runner.py:168-170`: the prompt fingerprint (`pilot/<version>/fingerprint.json`) now includes `test_model`. A different `test_model` under the same version raises FingerprintMismatch, which says to set a new prompt_version. An approval for one test_model does not cover another.
- `src/probe/atlas/pilot.py:159-161`: `report.md` has a line: "Models: explanations by `claude-opus-5-5`, test passages by `claude-sonnet-5-5` (effort medium)."
- The root `fingerprint.json` (corpus, base-model revision, llm_window) is deliberately unchanged. Adding `test_model` there would make the runner refuse to resume over the existing corpus and recording. The per-version prompt fingerprint is the one that guards LLM settings (R36).

**2. A fresh pilot**
- `src/probe/atlas/config.py:38`: `prompt_version = "v2"`.
- `results/07_atlas/pilot/v1/` is untouched. Nothing under `results/07_atlas` was deleted or written. `pilot/v2/` does not exist yet.
- `CLAUDE.md:58`: the review path now reads `pilot/<prompt_version>/report.md`.

**3. Stop on mass refusal (Ruling R37)**
- `src/probe/atlas/batches.py:16-21`: `MASS_REFUSALS = 20`, `MASS_REFUSAL_SHARE = 0.5`, and a regex for the `Details: \`[tag]\`` part.
- `src/probe/atlas/batches.py:38-60`: `_Run` counts the items each `process` call settles through `work`. It also counts every single-item refusal before that refusal is stored. When refusals reach at least 20 and at least half of the settled items, `_Run`:
  1. deletes the refusal records that this call stored;
  2. writes "Run stopped: …" into status and flushes it;
  3. raises `RunStopped`. The message quotes the first refusal's tag, for example "first refusal Details: `[reasoning_extraction]`".
- `__main__` already writes any RunStopped to errors.log (`_log_error`). RunStopped is not an LLMFailed, so nothing above `process` splits or marks anything. Nothing is asked for or stored after the stop fires, and a rerun asks for the removed items again.
- `src/probe/atlas/batches.py:86,92,100`: settled-count bookkeeping. A refusal still breaks the R20 run of held-back LLMFailed items, as before.
- `src/probe/atlas/batches.py:111-112`: `process(..., status=None)`. Status is passed in at explain.py:161, evaluate.py:41, pilot.py:124 and research.py:102,182.
- "Current stage step" means one `process` call. In the pilot that is one layer's explanations, tests or scores, about 28 items. With v1's failure (every item refused), the stop fires at the 20th refusal of L14's test writing.

**Tests added or changed**
- `tests/atlas/fixtures/fake_claude.py:52`: a new `safeguard` mode. It returns the real CLI's refusal shape with ``Details: `[reasoning_extraction]` `` and logs each call's `--model`.
- `tests/atlas/test_llm.py:466`: the fake_claude test. `write_tests` on 30 items in `safeguard` mode must:
  - raise RunStopped quoting the tag, after exactly 20 refusals;
  - store no batch records;
  - leave a status note that starts "Run stopped:";
  - send every call with `--model claude-sonnet-5-5`.

  A rerun in `atlas` mode then resumes, and all 30 items come back answered.
- `tests/atlas/test_llm.py:494,499,504`: the thresholds. 19 refusals are stored as failed. 20 refusals out of 41 items do not stop the run. 20 out of 40 do stop it, removing the refusal records and asking for nothing after the stop.
- `tests/atlas/test_llm.py:46,53`: `build_argv` and `call_with_wait` thread the model. The default is still `llm.MODEL`.
- `tests/atlas/test_evaluate.py:59`: `write_tests` uses `cfg.test_model`.
- `tests/atlas/test_runner.py:340,352`: a changed `test_model` under the same version is refused and is not approved. The prompt fingerprint records `test_model`.
- `tests/atlas/test_runner.py:492,497`: report.md names both models, and report.json carries `test_model`.
- Test fakes for `call_with_wait`/`call_claude` now accept `model=None` (test_evaluate, test_report, test_runner, test_llm).
- `v1` paths in test_runner and test_end_to_end (GPU, not run) now follow `prompt_version`.

## Stage-reuse check

This check is read-only and used the real `results/07_atlas` with the new `Config()`:

| stage | marker on disk | v2 marker | result |
|---|---|---|---|
| corpus, thinking, assemble, record | `{}` | `{}` (only LLM stages carry prompt_parts) | done: reused |
| pilot | v1 prompt_parts (no test_model) | v2 + test_model | reruns, into `pilot/v2/` |
| layers, research, report | none | | not started |

- The root run fingerprint (`corpus` hash, base-model revision, `llm_window`) matches the new config. `prompt_version` and `test_model` are not part of it.
- The record checkpoint plan hash (record.py:85-87) covers only the corpus manifest hash, the layers and the batch plan. `record/summary.npz` and `record_p50c/summary.npz` both exist, and no `ckpt/` folders remain, so `_record` skips the recording.
- `PROMPT_HASH` is unchanged (`b8d2b5d4…`), because no prompt text changed.
- `approved.json` does not exist.

Result: changing `prompt_version` or `test_model` invalidates none of the corpus, thinking, assemble or record markers or checkpoints.

## Test summary

`nix-shell --run 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 pytest tests/atlas -m "not gpu and not claude" -q'` gave 501 passed and 15 deselected in 115 s. No GPU tests were run, and the GPU path is unchanged.

## Concerns

- **BLOCKED:** Sonnet 5.5 refuses too, and the refusal does not depend on the item context. A v2 pilot run as configured would now stop at the 20th refused test call in L14 with RunStopped (R37), rather than marking 200 neurons failed. Before that, it would spend about 10 Opus explanation calls (L14's 29 neurons) on a fresh, non-reused explanation pass. Don't start the run until the test prompt is changed. A prompt change needs a new prompt_version (v2 is still unused, so it can carry the change) and a new pilot.
- The v2 pilot re-explains all 200 pilot neurons with Opus. v1's 200 successful explanations are not reused, because results are keyed by prompt_version by design.
- After a mass-refusal stop, split marks (`split_*.json`) stay on disk. A rerun therefore asks each item singly. This is correct, but it uses more calls than a whole batch would.
