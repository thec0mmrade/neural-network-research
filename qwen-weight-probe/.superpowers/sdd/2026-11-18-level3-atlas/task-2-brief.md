### Task 2: Status file and overall progress

**Files:**
- Create: `src/probe/atlas/status.py`
- Test: `tests/atlas/test_status.py`

**Interfaces:**
- Consumes: `store.atomic_write_json`
- Produces: `STAGES = ("corpus","thinking","assemble","record","pilot","layers","research","report")`; `DEFAULT_HOURS = {"corpus":1,"thinking":7,"assemble":0.2,"record":10,"pilot":3,"layers":150,"research":3,"report":0.2}`; `overall_fraction(stages: dict[str, dict]) -> float`; class `Status(path: Path)` with methods `stage(name, done: int, total: int, unit: str)`, `paused(reason: str, resumes_at: float | None)`, `resumed()`, `layer(layer: int, state: str, done: int, total: int, mean_score: float | None)`, `usage(input_tokens: int, output_tokens: int)`, `scores(primary: dict, ac_matched: dict)`, `latest(neuron: dict)`, `note(text: str)`.
- `status.json` schema: `{"updated_at", "started_at", "active_s", "paused_s", "pause": {"reason","resumes_at"}|null, "current_stage", "stages": {name: {"done","total","unit","active_s"}}, "layers": {"15": {"state","done","total","mean_score"}}, "usage": {"input_tokens","output_tokens","calls"}, "scores": {"primary","ac_matched"}, "latest": {...}, "note"}`. Writes are throttled to one per second, except pause and stage changes.

- [ ] **Step 1: Write the failing tests**

```python
def test_overall_weights_by_expected_hours():
    s = {n: {"done": 0, "total": 1, "active_s": 0} for n in STAGES}
    s["corpus"] = {"done": 1, "total": 1, "active_s": 3600}
    assert overall_fraction(s) == pytest.approx(1 / sum(DEFAULT_HOURS.values()))

def test_measured_speed_replaces_default_after_2_percent():
    s = {n: {"done": 0, "total": 1, "active_s": 0} for n in STAGES}
    s["layers"] = {"done": 50, "total": 100, "active_s": 3600 * 10}   # measured: 20 h in all
    total = sum(DEFAULT_HOURS.values()) - 150 + 20
    assert overall_fraction(s) == pytest.approx(10 / total)

def test_pause_time_is_not_active_time(tmp_path):
    clock = FakeClock(); st = Status(tmp_path / "s.json", clock=clock)
    st.stage("layers", 1, 10, "neurons"); clock.advance(10)
    st.paused("Max limit", clock.now + 100); clock.advance(100); st.resumed(); clock.advance(5)
    st.stage("layers", 2, 10, "neurons")
    d = read_json(tmp_path / "s.json")
    assert d["paused_s"] == pytest.approx(100) and d["active_s"] == pytest.approx(15)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_status.py -v'`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `status.py`**

`Status.__init__` takes `clock=time` for tests, and reloads an existing file so counters survive a restart.

- [ ] **Step 4: Run the tests and confirm they pass**

Expected: 3 passed

---

