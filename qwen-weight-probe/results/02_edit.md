# Phase 2: the Sparks Exp 8 edits, decomposed

All 33 edited tensors (p50c 2, p25c 1, p10c 30) were rebuilt by the logs' `apply_edit.py` and matched their sha256 hashes.

## p50c (the 48.1% MATH-500 run)

| layer | changed cols | rank-1 share of ΔW | fitted α in ΔW = −α·u(uᵀW) | fit residual | edited cols in top-64 u-writers | share of the layer's u-writing in edited cols |
|---|---|---|---|---|---|---|
| 15 | 64 | 1.0000 | 6.000 | 0.0023 | 31/64 | 0.252 |
| 19 | 64 | 1.0000 | 6.000 | 0.0029 | 18/64 | 0.160 |

α = 6 means each edited neuron's component along u goes from c to −5c: a sign flip with 5× gain, not a removal (α = 1).

## Agreement between runs

- u15 vs u19 (p50c): |cos| 0.849, signed -0.849
- p10c mean direction vs u15: 0.691; vs u19: 0.676
- p10c's 30 rank-one directions among themselves: mean |cos| 0.706, min 0.408
- random 1024-d unit vectors: E|cos| ≈ 0.025

## p25c (zeroed head 7 of layer 15): share of each head's output weight along u

Chance level for a 256-column head along one direction ≈ 0.0010.

| head | share along u15 | share along u19 |
|---|---|---|
| 0 | 0.0008 | 0.0008 |
| 1 | 0.0006 | 0.0009 |
| 2 | 0.0007 | 0.0008 |
| 3 | 0.0008 | 0.0008 |
| 4 | 0.0011 | 0.0013 |
| 5 | 0.0013 | 0.0015 |
| 6 | 0.0017 | 0.0021 |
| 7 | 0.0020 | 0.0025 |
