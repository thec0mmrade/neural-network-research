### Task 3: Terminal viewer

**Files:**
- Create: `src/probe/atlas_watch.py`
- Test: `tests/atlas/test_watch.py`

**Interfaces:**
- Consumes: the `status.json` schema and `overall_fraction`, `STAGES` from Task 2. It must not import torch.
- Produces: `render(status: dict | None, now: float, width: int) -> rich.layout.Layout`; `layer_bar(done: int, total: int, mean_score: float | None, cells: int) -> rich.text.Text`; `bottom_bar(status: dict, now: float, width: int) -> rich.text.Text`; `load_status(path: Path) -> dict | None` (returns `None` on a missing or unparsable file); `main()`, refreshing every 2 seconds with `rich.live.Live`.

- [ ] **Step 1: Write the failing tests**

```python
def show(layout, width=120):
    c = Console(width=width, height=40, record=True, file=io.StringIO()); c.print(layout)
    return c.export_text()

def test_missing_status_says_waiting():
    assert "Waiting for the pipeline" in show(render(None, now=0, width=120))

def test_partial_status_does_not_crash():
    assert "corpus" in show(render({"current_stage": "corpus"}, now=0, width=120))

def test_stale_status_is_flagged():
    assert "no update for 5m" in show(render(sample(updated_at=0), now=300, width=120))

def test_pause_shows_reason_and_is_not_stale():
    out = show(render(sample(updated_at=0, pause={"reason": "Max limit", "resumes_at": 900}), now=300, width=120))
    assert "Max limit" in out and "no update" not in out

def test_bottom_bar_shows_percent_stage_and_times():
    t = bottom_bar(sample(), now=100, width=100).plain
    assert "OVERALL" in t and "%" in t and "elapsed" in t and "left" in t

def test_layer_bar_has_one_cell_per_60_neurons():
    assert len(layer_bar(1792, 3584, 0.7, cells=60).plain) == 60
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_watch.py -v'`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `atlas_watch.py`**

Panels from top to bottom: pipeline stages with speed and time left; a left column of layer bars; a right column of Max usage, pause state and scores next to AC's 76% / 11% / ~1%; the latest neuron, with activating tokens highlighted by background color; and the bottom bar. Every field read uses `.get` with a default. A status is stale when it has no update for 60 seconds and isn't paused.

- [ ] **Step 4: Run the tests and confirm they pass, then check it in the CPU shell**

Run: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_watch.py -v'`
Expected: 6 passed

Run: `nix-shell nix/cpu-shell.nix --run 'timeout 5 python -m probe.atlas_watch; test $? -eq 124 && echo VIEWER_OK'`
Expected: the last line is `VIEWER_OK`

---

