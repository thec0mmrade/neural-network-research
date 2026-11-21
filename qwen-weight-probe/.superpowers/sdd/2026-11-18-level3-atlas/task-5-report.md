# Task 5 report: base-model thinking generation

## Built
- `src/probe/atlas/thinking.py`: `thinking_problems(seed)` (gsm8k+math `problem` fields, split on "\n" via corpus `_jsonl_rows`, shuffled with `random.Random(seed)`) and `generate_thinking(cfg, model, tok, status, gen=None, prompts=None, problems=None)`.
  Batch b = problems[32b:32b+32], seed 32b, max_new_tokens 4000, batch_size 32. Saves `raw/thinking/batch_NNNNN.json` ({"docs", "seed"}) before the next batch; existing batch files are reused. Stops when total >= thinking_tokens, then `save_source(cfg, "thinking", ...)`. Status stage "thinking", unit "tokens". `probe.model` imported lazily only when gen/prompts are not injected.
- Doc = `tok(prompt, add_special_tokens=False).input_ids + generated ids`, capped at `doc_cap`.

## Files
- Created: `src/probe/atlas/thinking.py`, `tests/atlas/test_thinking.py`
- Changed: `tests/atlas/conftest.py` (FakeTok gained `__call__` returning `.input_ids`)

## RED
`nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_thinking.py -v'` -> ModuleNotFoundError: No module named 'probe.atlas.thinking' (collection error).
## GREEN
`nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas -q'` -> 67 passed (2 new). (An intermediate failure was FakeTok returning `.ids` instead of `.input_ids`; fixed in the fake.)

## Tests
1. Brief's resume test (reused batch 0, one gen call with seed 32, stops at 2 batches, >= 64000 tokens, no batch_00002).
2. Doc composition: prompt ids + generated, capped at doc_cap, status unit "tokens".

## Self-review / concerns
- The brief's test is adapted: it injects `prompts=` and `problems=` and sets `doc_cap=2000` (tiny cfg already has 4096, so cap is not strictly needed). Without injection it would need the real chat template and practice files.
- A resumed batch file that is corrupt reads as missing and is regenerated (read_json default). If problems run out before the target, the source is saved short with no error.
- Status is updated once per batch only (generation is the slow part; no intra-batch progress).
- Real `generate` / real tokenizer path untested (no GPU run, per rules).
