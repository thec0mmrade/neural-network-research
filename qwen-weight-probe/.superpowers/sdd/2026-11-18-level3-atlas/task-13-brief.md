### Task 13: Research question and report

**Files:**
- Create: `src/probe/atlas/research.py`, `src/probe/atlas/report.py`
- Modify: `src/probe/atlas/runner.py` (call both stages)
- Test: `tests/atlas/test_report.py`

**Interfaces:**
- Consumes: `edited_neurons`, `results/u15_p50c.npy`, `results/u19_p50c.npy`, `Checkpoint`, `write_tests`, `score_tests`, `call_with_wait`, `results/06_wiring/edges_raw.npz` (`out_id`, `out_score`, `in_id`, `in_score`), `probe.wiring.components`
- Produces:
  - `research.study_neurons(cfg) -> dict[int, np.ndarray]`: the union, per layer, of the edited columns and the top 64 columns by `np.argsort(-np.abs(u @ W))`, where `W` is the base `down_proj`.
  - `research.control_neurons(cfg, summary, study) -> dict[int, np.ndarray]`: an equal number of random live neurons per layer, disjoint from the study set.
  - `research.run_research(cfg, model, tok, status) -> dict`: tests both poles of study and control neurons, then runs blind ratings in shuffled order with anonymous ids. Each rating is `{"move_on": 0..3, "reconsider": 0..3}`. Writes `root/research.json`.
  - `report.wiring_join(cfg, layer: int, idx: int, k: int = 8) -> dict`
  - `report.write_report(cfg) -> Path`: writes `results/07_atlas.md` and `root/atlas_neurons.jsonl`, with one complete record per neuron.

- [ ] **Step 1: Write the failing tests**

```python
def test_study_set_contains_edited_and_top_writers(tiny_cfg):
    s = study_neurons(tiny_cfg); e = edited_neurons("p50c")
    assert sorted(s) == [15, 19] and set(e[15]) <= set(s[15]) and len(s[15]) >= 64

def test_controls_are_live_disjoint_and_matched(tiny_cfg, summary_all_live):
    s = study_neurons(tiny_cfg); c = control_neurons(tiny_cfg, summary_all_live, s)
    assert len(c[15]) == len(s[15]) and not set(c[15]) & set(s[15])

def test_rating_prompt_hides_group_membership(tiny_cfg, capture_prompt):
    rate_blind(tiny_cfg, capture_prompt.status, study=[item(1)], control=[item(2)], out=capture_prompt.out)
    assert "control" not in capture_prompt.text.lower() and "edited" not in capture_prompt.text.lower()

def test_report_includes_required_sections(finished_tiny_run):
    text = write_report(finished_tiny_run).read_text()
    for s in ("76%", "AC-matched", "Primary", "Job labels", "Research question", "Limits", "dead"):
        assert s in text

def test_every_live_neuron_has_exactly_one_record(finished_tiny_run):
    write_report(finished_tiny_run)
    rows = read_jsonl(finished_tiny_run.root / "atlas_neurons.jsonl")
    assert len({r["nid"] for r in rows}) == len(rows) == finished_tiny_run.n_neurons_total
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_report.py -v'`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `research.py` and `report.py`**

The report's Limits section states all the items in section 12 of the spec.

- [ ] **Step 4: Run the tests and confirm they pass**

Expected: 5 passed

---

