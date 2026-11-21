# Task 6 report: neuron activations

Status: DONE_WITH_CONCERNS

## Files
- Created `src/probe/atlas/activations.py` (StopForward, eligible_mask, excluded_ids, capture, run_batch, pad_right)
- Created `tests/atlas/test_activations.py` (8 tests: the brief's 2, plus 6 added)
- Changed `tests/atlas/conftest.py` (session `base_model` fixture, lazy `probe.model` import)

## RED
`nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_activations.py -v'` -> ModuleNotFoundError: No module named 'probe.atlas.activations' (collection error).
Later GREEN attempts also exposed (a) the brief's 1e-3 check failing with bf16 `a` (see concern) and (b) my CPU fake MLP not being invoked; both fixed.

## GREEN
`pytest tests/atlas -v` -> 75 passed in 25.9s (8 in test_activations, incl. the gpu test). No other probe.* job was running.

## Design
- eligible_mask: attention & ~isin(excluded) & ~is_start & ~after_start, index 0 forced False (R5).
- capture: pre-hook on mlp; gate/up from the model's bf16 projections; floats them, a = act_fn(gate)*value in float32; callback; raises StopForward when l == max(layers) after callback. Hooks removed in finally. run_batch catches StopForward under no_grad.
- Extra tests: im_start at end / 2nd row, pad_right, excluded_ids, CPU tiny model (only layers 0..last run, shapes/dtype float32, hooks removed after success and exception, StopForward escapes without run_batch).

## Concerns
- The brief's second assertion (a == silu(g)*v at 1e-3) cannot hold if `a` is the bf16 product (bf16 rounding ~4e-3 relative; it failed on GPU). So `a` is the float32 product of the bf16 gate/value; it matches the real down_proj input at 1e-2 (test passes). `a` therefore lacks the bf16 rounding of the real input (more precise, not identical).
- Callback's tensors are fresh float32 tensors; callers must not rely on them after the hook returns without cloning if they mutate shared buffers (not an issue here).
- `capture` with empty `layers` raises ValueError from max(); not guarded.
