### Task 5: Base-model thinking generation

**Files:**
- Create: `src/probe/atlas/thinking.py`
- Test: `tests/atlas/test_thinking.py`

**Interfaces:**
- Consumes: `probe.model.generate`, `probe.model.build_prompts`, `corpus.save_source`, `store`
- Produces: `thinking_problems(seed: int) -> list[str]` (GSM8K and MATH problems, shuffled deterministically); `generate_thinking(cfg, model, tok, status, gen=probe.model.generate) -> None`. Each batch of 32 problems is saved to `root/raw/thinking/batch_00000.json` with prompt plus generated ids; generation stops once `thinking_tokens` is reached, then `save_source(cfg, "thinking", ...)` is called.

- [ ] **Step 1: Write the failing test**

```python
def test_resume_skips_finished_batches_and_stops_at_target(tiny_cfg, fake_tok, status):
    calls = []
    def gen(model, tok, problems, max_new_tokens, seed=0, batch_size=32):
        calls.append(seed); assert max_new_tokens == 4000 and batch_size == 32
        return [{"ids": [5] * 1000, "finished": True} for _ in problems], 400.0
    cfg = replace(tiny_cfg, thinking_tokens=64_000)
    (cfg.root / "raw/thinking").mkdir(parents=True)
    atomic_write_json(cfg.root / "raw/thinking/batch_00000.json", {"docs": [[5] * 1000] * 32, "seed": 0})
    generate_thinking(cfg, None, fake_tok, status, gen=gen)
    assert calls == [32]                  # batch 0 was reused, and 2 batches reach the target
    assert sum(map(len, load_source(cfg, "thinking"))) >= 64_000
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_thinking.py -v'`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `thinking.py`**

The seed for batch `b` is `b * 32`. Documents are `tok(build_prompts(...)).input_ids + generated ids`, capped at `doc_cap`.

- [ ] **Step 4: Run the test and confirm it passes**

Expected: 1 passed

---

