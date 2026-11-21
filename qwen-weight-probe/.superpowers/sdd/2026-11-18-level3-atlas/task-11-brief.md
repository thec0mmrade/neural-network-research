### Task 11: Blind test writing and scoring

**Files:**
- Create: `src/probe/atlas/evaluate.py`
- Test: `tests/atlas/test_evaluate.py`

**Interfaces:**
- Consumes: `batches.process`, `prompts`, `activations.capture`, `call_with_wait`
- Produces:
  - `write_tests(cfg, status, items: list[dict], out: Path) -> list[dict]`: items are `{"nid","pole","explanation","context"}`; the prompt uses anonymous ids (`t01`…) and contains only the explanation and context. Output per neuron: `blind_maximum`, `near_miss`, `distant`, each exactly 10 `{"text","context"}`.
  - `example_ids(tok, text: str, context: str, cfg) -> tuple[list[int], int]`: `plain` tokenizes the bare text truncated to 128 tokens. `reasoning` prefixes `tok.apply_chat_template([{"role":"user","content":"Think about the following."}], tokenize=False, add_generation_prompt=True, enable_thinking=True)`. Returns ids and the first passage index.
  - `score_tests(cfg, model, tok, layer: int, tests: list[dict], summary: dict) -> list[dict]`: adds `{"blind_maximum": hits, "near_miss": hits, "distant": hits}`. An example hits if its strongest pole-signed activation, at eligible gate-positive passage positions, is ≥ 0.3 × that pole's extreme.
  - `summarize(records: list[dict], summary: dict) -> dict`: returns `{"primary": {...}, "ac_matched": {...}}`, each with `n` and three shares.

- [ ] **Step 1: Write the failing tests**

```python
def test_test_writer_prompt_is_blind(tiny_cfg, capture_prompt):
    items = [{"nid": 12345, "pole": 1, "explanation": "fires on prices", "context": "plain"}]
    write_tests(tiny_cfg, capture_prompt.status, items, capture_prompt.out)
    p = capture_prompt.text
    assert "fires on prices" in p and "12345" not in p and "L15" not in p and "[" + "0." not in p

def test_summary_counts_failed_as_zero_in_primary_only():
    recs = [rec(0, hits=10, dom=1, job="one job"), {"nid": 1, "failed": "declined"}, rec(2, hits=10, dom=1, job="unclear")]
    s = summarize(recs, live_all(3))
    assert s["primary"]["n"] == 3 and s["primary"]["blind_maximum"] == pytest.approx(2 / 3)
    assert s["ac_matched"]["n"] == 1 and s["ac_matched"]["blind_maximum"] == 1.0

def test_ac_matched_excludes_negative_dominant_and_dead():
    recs = [rec(0, hits=5, dom=-1, job="one job"), rec(1, hits=5, dom=1, job="one job")]
    assert summarize(recs, live_mask([True, True]))["ac_matched"]["n"] == 1

@pytest.mark.gpu
def test_score_counts_only_passage_positions(base_model, tiny_summary):
    model, tok = base_model
    ids, start = example_ids(tok, "The answer is 42.", "reasoning", Config())
    assert start > 0 and len(ids) > start
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_evaluate.py -v'`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `evaluate.py`**

- [ ] **Step 4: Run the tests and confirm they pass**

Expected: 4 passed

---

