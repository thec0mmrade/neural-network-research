### Task 6: Neuron activations

**Files:**
- Create: `src/probe/atlas/activations.py`
- Test: `tests/atlas/test_activations.py`

**Interfaces:**
- Produces:
  - `eligible_mask(ids: torch.Tensor, attention: torch.Tensor, excluded: torch.Tensor, im_start_id: int) -> torch.Tensor` (bool): False at index 0 of each document, at padding, at any `excluded` id, and at the token right after `<|im_start|>`.
  - `excluded_ids(tok) -> torch.Tensor`: `tok.all_special_ids` plus `<think>`, `</think>`, `<|im_start|>`, `<|im_end|>`.
  - `capture(model, layers, on_layer: Callable[[int, Tensor, Tensor, Tensor], None])`: a context manager that registers a forward pre-hook on each `model.model.layers[l].mlp`, calling `on_layer(layer, a, gate, value)` with float32 tensors of shape `[B, T, 3584]`, where `gate = mlp.gate_proj(x)`, `value = mlp.up_proj(x)`, and `a = silu(gate) * value`. It raises `StopForward` after the last requested layer, so later layers never run.
  - `run_batch(model, ids, attention) -> None`: calls `model.model(input_ids=ids, attention_mask=attention, use_cache=False)` under `torch.no_grad()`, catching `StopForward`.
  - `pad_right(docs: list[np.ndarray], pad_id: int, device) -> tuple[Tensor, Tensor]`

- [ ] **Step 1: Write the failing tests**

```python
def test_eligible_excludes_first_special_role_and_padding():
    ids = torch.tensor([[9, 50, 7, 51, 8, 0]]); att = torch.tensor([[1, 1, 1, 1, 1, 0]])
    m = eligible_mask(ids, att, excluded=torch.tensor([50]), im_start_id=51)
    assert m.tolist() == [[False, False, True, False, False, False]]

@pytest.mark.gpu
def test_capture_matches_down_proj_input(base_model):
    model, tok = base_model; got, want = {}, {}
    h = model.model.layers[15].mlp.down_proj.register_forward_pre_hook(lambda m, i: want.update(a=i[0].float().clone()))
    ids, att = pad_right([np.array(tok("The sum of two numbers is 12.").input_ids)], tok.pad_token_id, model.device)
    model.model(input_ids=ids, attention_mask=att, use_cache=False); h.remove()
    with capture(model, [15], lambda l, a, g, v: got.update(a=a.clone(), g=g.clone(), v=v.clone())):
        run_batch(model, ids, att)
    assert torch.allclose(got["a"], want["a"], rtol=1e-2, atol=1e-2)
    assert torch.allclose(got["a"], F.silu(got["g"]) * got["v"], rtol=1e-3, atol=1e-3)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_activations.py -v'`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `activations.py`**

- [ ] **Step 4: Run the tests and confirm they pass**

Expected: 2 passed

---

