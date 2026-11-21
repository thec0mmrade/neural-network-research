### Task 14: End-to-end rehearsal, docs and launch up to the pilot gate

**Files:**
- Create: `tests/atlas/test_end_to_end.py`
- Modify: `CLAUDE.md`, `TODO.md`

**Interfaces:**
- Consumes: the CLI from Task 12, `fake_claude.py` from Task 9, `Config.tiny`

- [ ] **Step 1: Write the rehearsal test**

```python
@pytest.mark.gpu
@pytest.mark.parametrize("stage", ["corpus", "thinking", "record", "pilot", "layers"])
def test_sigkill_in_each_stage_matches_clean_run(stage, tiny_workspace):
    clean = tiny_workspace.run_to_completion("clean")
    tiny_workspace.run_and_sigkill("resumed", during=stage)
    resumed = tiny_workspace.run_to_completion("resumed")
    assert resumed.returncode == 0
    assert_same_atlas(clean.root, resumed.root, rtol=1e-3)
```

- [ ] **Step 2: Run the rehearsal and the full test suite**

Run: `nix-shell --run 'OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pytest tests/atlas -v -m "not claude"'`
Expected: all pass, with no skipped tests.

- [ ] **Step 3: Document the atlas in `CLAUDE.md` and `TODO.md`**

Add to `CLAUDE.md`: the `run`, `approve-pilot` and `status` commands, the viewer command, the never-use-`--bare` rule, and the output folder. In `TODO.md`, mark the level 3 design decisions as decided.

- [ ] **Step 4: Tell the user the real run is about to start, then start it**

Run in the background: `nix-shell --run 'OMP_NUM_THREADS=4 RAYON_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m probe.atlas run'`
Expected: the run proceeds through `corpus`, `thinking`, `assemble`, `record` and `pilot`, then exits with `waiting_for_pilot_approval`.

- [ ] **Step 5: Stop at the gate and report the pilot to the user**

Give the user `results/07_atlas/pilot/v1/report.md`: scores against AC, the control, tokens used, projected duration and the 10 sample neurons. **Do not run `approve-pilot` without the user's explicit go-ahead.**
