### Task 7: Streaming recorder state and final statistics

**Files:**
- Create: `src/probe/atlas/recorder.py`
- Test: `tests/atlas/test_recorder.py`

**Interfaces:**
- Produces: class `LayerRecorder(cfg, device)` with:
  - `update(a, gate, value, eligible, doc_ids: Tensor, doc_keys: Tensor) -> None`
  - `state() -> dict[str, np.ndarray]` and `load(state) -> None`
  - Arrays: `{pos,neg,dn}_{val,doc,tok,win}` of shape `[3584, k]` (windows `[3584, k, 64]`, float16), `rnd_{key,doc,tok,win}` (`[random_k]`, windows `[3584, random_k, 64]`), `hist` (`[3584, 4, 256]`, int64), `count`, `sum`, `max`, `min`.
- Produces: `HIST_EDGES = torch.logspace(-4, 3, 255)`; `finalize(states: dict[int, dict], cfg) -> dict`, returning per-layer arrays `pos_extreme`, `neg_extreme`, `dominant` (1 positive, −1 negative), `live` (bool), `fire_share`, `dn_flag`.

**Rules:**
- Positive pole candidates: `gate > 0 and a > 0`, ranked by `a`. Negative pole: `gate > 0 and a < 0`, ranked by `-a`. Double negative: `gate < 0 and value < 0`, ranked by `a`. All need `eligible`.
- Each document contributes at most `per_doc_cap` candidates per list.
- A window covers `[t-32, t+31]`, padded with NaN at document edges.
- Random examples: one eligible position per document, picked by `doc_keys`; keep the `random_k` smallest keys.
- `hist` regime index: `(gate >= 0) * 2 + (value >= 0)`, binned by `torch.bucketize(|a|, HIST_EDGES)`, counting eligible positions only.
- Dead rule, computed from gate-positive histogram bins: let `E` be the median over the layer's neurons of `max(pos_extreme, neg_extreme)`. A neuron is live if at least 100 positions have `|a| ≥ 0.3 * E`, counting a bin when its lower edge is ≥ the threshold. If `E == 0`, every neuron is dead.
- `dn_flag`: true when the largest `dn_val` exceeds the smallest kept `pos_val`.

- [ ] **Step 1: Write the failing tests**

```python
def test_streaming_matches_brute_force(tiny_cfg):
    a, g, v, elig, docs = random_batches(seed=1, n_batches=6)
    rec = LayerRecorder(replace(tiny_cfg, n_neurons=8), "cpu")
    for b in zip(a, g, v, elig, docs): rec.update(*b, doc_keys=keys_for(b[-1]))
    s = rec.state(); want = brute_force_top(a, g, v, elig, docs, k=50, per_doc=3)
    np.testing.assert_allclose(np.sort(s["pos_val"], 1), np.sort(want["pos_val"], 1), rtol=1e-5)

def test_at_most_three_examples_per_document(tiny_cfg):
    s = recorded_state(tiny_cfg, one_hot_doc=True)
    assert max(np.bincount(s["pos_doc"][0][s["pos_val"][0] > 0])) <= 3

def test_double_negative_never_enters_positive_pole(tiny_cfg):
    a = torch.tensor([[[9.0]]]); g = torch.tensor([[[-1.0]]]); v = torch.tensor([[[-30.0]]])
    s = update_once(tiny_cfg, a, g, v)
    assert s["pos_val"].max() <= 0 and s["dn_val"].max() == pytest.approx(9.0)

def test_window_is_centered_and_padded(tiny_cfg):
    s = recorded_state(tiny_cfg, peak_at=2, length=10)
    w = s["pos_win"][0, 0]
    assert np.isnan(w[0]) and w[32] == pytest.approx(s["pos_val"][0, 0])

def test_state_roundtrip_continues_identically(tiny_cfg):
    full, resumed = run_with_and_without_reload(tiny_cfg)
    for k in full: np.testing.assert_array_equal(full[k], resumed[k])

def test_silent_neuron_is_dead_without_error(tiny_cfg):
    out = finalize({15: silent_state(tiny_cfg)}, tiny_cfg)
    assert not out[15]["live"].any() and np.isfinite(out[15]["fire_share"]).all()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_recorder.py -v'`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `recorder.py`**

Keep state on the GPU. Compute histograms in chunks of at most 8,192 token positions to bound memory.

- [ ] **Step 4: Run the tests and confirm they pass**

Expected: 6 passed

---

