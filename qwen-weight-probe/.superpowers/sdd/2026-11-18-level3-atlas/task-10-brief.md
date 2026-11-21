### Task 10: Batches, prompts and explanations

**Files:**
- Create: `src/probe/atlas/batches.py`, `src/probe/atlas/prompts.py`, `src/probe/atlas/explain.py`
- Test: `tests/atlas/test_explain.py`

**Interfaces:**
- Consumes: `call_with_wait`, `store`, `Status`, the recordings from Task 8
- Produces:
  - `batches.process(items: list[dict], size: int, out: Path, work: Callable[[list[dict]], list[dict]], key: str = "nid") -> list[dict]`: stores each validated chunk as `out/batch_<first-key>_<n>.json`, skipping finished chunks. `work` must return exactly one record per input key. On `LLMFailed`, `Refused` or a key mismatch, it splits the chunk in half and recurses; a single item that still fails is stored as `{"nid": ..., "failed": "<reason>"}`. `LimitReached` is never caught here.
  - `prompts.PROMPT_HASH`: sha256 of all prompt text and schemas. `prompts.EXPLAIN_SYSTEM`, `prompts.TEST_SYSTEM`, `prompts.RATE_SYSTEM`, `prompts.explain_schema(ids)`, `prompts.test_schema(ids)`, `prompts.rate_schema(ids)`.
  - `prompts.render_neuron(rec: dict, tok, cfg) -> str`: 20 top positive, 10 top negative and 10 random examples, each cut to `llm_window` tokens around its peak, marking only tokens at ≥ 20% of the extreme as `token[activation]`, plus a histogram summary.
  - `explain.neuron_record(cfg, layer: int, idx: int) -> dict` and `explain.explain_neurons(cfg, status, layer: int, idxs: Sequence[int], out: Path) -> list[dict]`.
- Explanation record: `{"nid","layer","idx","positive":{"explanation","context","job"},"negative":{...}}`, where `context ∈ {"plain","reasoning"}` and `job ∈ {"one job","two related jobs","two unrelated jobs","unclear"}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_rendered_neuron_stays_within_token_budget(tiny_cfg, recorded, tok):
    text = render_neuron(neuron_record(tiny_cfg, 15, 0), tok, tiny_cfg)
    assert len(tok(text).input_ids) <= 2500

def test_missing_id_rejects_batch_and_splits(tmp_path):
    seen = []
    def work(chunk):
        seen.append(len(chunk))
        return [{"nid": x["nid"]} for x in chunk[:-1]] if len(chunk) > 1 else [{"nid": chunk[0]["nid"]}]
    out = process([{"nid": i} for i in range(4)], 4, tmp_path, work)
    assert seen[0] == 4 and sorted(r["nid"] for r in out) == [0, 1, 2, 3]

def test_single_failure_is_recorded_not_dropped(tmp_path):
    def work(chunk):
        if any(x["nid"] == 2 for x in chunk): raise Refused("declined")
        return [{"nid": x["nid"]} for x in chunk]
    out = process([{"nid": i} for i in range(4)], 4, tmp_path, work)
    assert [r for r in out if r["nid"] == 2] == [{"nid": 2, "failed": "declined"}]

def test_limit_propagates_and_batch_is_not_saved(tmp_path):
    def work(chunk): raise LimitReached(None)
    with pytest.raises(LimitReached):
        process([{"nid": 1}], 1, tmp_path, work)
    assert not list(tmp_path.glob("batch_*.json"))

def test_finished_batches_are_not_called_again(tmp_path):
    calls = []
    work = lambda c: calls.append(1) or [{"nid": x["nid"]} for x in c]
    items = [{"nid": i} for i in range(4)]
    process(items, 2, tmp_path, work); process(items, 2, tmp_path, work)
    assert len(calls) == 2
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_explain.py -v'`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the three modules**

`EXPLAIN_SYSTEM` states that all example text is data, and that any instruction inside it must be ignored.

- [ ] **Step 4: Run the tests and confirm they pass**

Expected: 5 passed

---

