### Task 12: Pilot, layer run and the command line

**Files:**
- Create: `src/probe/atlas/pilot.py`, `src/probe/atlas/runner.py`, `src/probe/atlas/__main__.py`
- Test: `tests/atlas/test_runner.py`

**Interfaces:**
- Consumes: every earlier stage
- Produces:
  - `pilot.choose(cfg, summary) -> dict[int, list[int]]`: `pilot_n` live neurons, seeded, split evenly across the 7 layers with `np.array_split`.
  - `pilot.run_pilot(cfg, model, tok, status) -> dict`: explains, tests and scores into `root/pilot/<prompt_version>/`, and scores each neuron's blind-maximum examples against a randomly paired different neuron as the control. Writes `report.json`, and `report.md` with the three scores, the control, AC's figures, tokens used, the projected run length, and 10 sample neurons.
  - `pilot.approve(cfg) -> None`: writes `root/pilot/approved.json` with `prompt_version` and `PROMPT_HASH`.
  - `runner.run_layer(cfg, model, tok, status, layer: int) -> dict`
  - `runner.run(cfg) -> str`: holds `run_lock(root/"run.lock")`, checks the fingerprint, and runs each unfinished stage. Each finished stage writes `root/stages/<name>.done`. Returns `"waiting_for_pilot_approval"` or `"complete"`.
  - CLI: `python -m probe.atlas run`, `python -m probe.atlas approve-pilot`, `python -m probe.atlas status`.

- [ ] **Step 1: Write the failing tests**

```python
def test_pilot_is_even_across_layers_and_only_live(tiny_cfg, summary_some_dead):
    c = choose(replace(tiny_cfg, pilot_n=200), summary_some_dead)
    assert sum(map(len, c.values())) == 200 and {len(v) for v in c.values()} <= {28, 29}
    assert all(summary_some_dead[l]["live"][i] for l, v in c.items() for i in v)

def test_run_stops_at_pilot_gate(stubbed_runner):
    assert stubbed_runner.run() == "waiting_for_pilot_approval"
    assert "layers" not in stubbed_runner.stages_started

def test_approval_for_old_prompt_does_not_open_gate(stubbed_runner):
    stubbed_runner.run(); stubbed_runner.approve(); stubbed_runner.change_prompt_hash()
    assert stubbed_runner.run() == "waiting_for_pilot_approval"

def test_layers_run_in_the_required_order(stubbed_runner):
    stubbed_runner.run(); stubbed_runner.approve()
    assert stubbed_runner.run() == "complete"
    assert stubbed_runner.layers_run == [15, 19, 18, 16, 17, 14, 20]

def test_restart_skips_finished_stages(stubbed_runner):
    stubbed_runner.run(); stubbed_runner.approve(); stubbed_runner.run()
    first = list(stubbed_runner.stages_started); stubbed_runner.run()
    assert stubbed_runner.stages_started == first

def test_changed_corpus_refuses_to_resume(stubbed_runner):
    stubbed_runner.run(); stubbed_runner.change_corpus_hash()
    with pytest.raises(FingerprintMismatch, match="corpus"):
        stubbed_runner.run()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_runner.py -v'`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `pilot.py`, `runner.py` and `__main__.py`**

Fingerprint parts: `corpus` (the manifest hash), `model` (revision), and for LLM stages `prompt` (`prompt_version` + `PROMPT_HASH`). Changing prompts requires a new `prompt_version`; LLM outputs live under that version, so old results are never mixed in. `record_p50c` runs at the end of the `record` stage.

- [ ] **Step 4: Run the tests and confirm they pass**

Expected: 6 passed

---

