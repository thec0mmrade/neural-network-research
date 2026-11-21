# Level 2: wiring map (weights only)

A static map of how every component of Qwen3.5-0.8B connects to the others through the residual stream, computed from the weights alone. It covers 86,352 components: 86,016 MLP neurons, 48 full-attention heads and 288 linear-attention heads. It is **not** a trace of any input. Built by `src/probe/wiring.py` (stages `build → hubs → edges → uhood → umap → lens → export`); the page is `06_wiring/page/wiring.html`.

## Method
- **Directions:**
  - A neuron writes through its `down_proj` column and reads through its `gate` and `up` rows.
  - A head writes through its block of output columns. It reads through its q / k / v rows, plus its gate rows, and for linear attention its `a`/`b` rows.
  - Reads are scaled by that layer's `(1 + w)` RMSNorm.
- **Edge score:** 1024 · ‖R·W‖² with R and W Frobenius-normalized. A random direction scores 1. For two neurons this is 1024·cos².
- **Allowed edges** follow the residual order: attention(l) → MLP(l) → attention(l+1) → …
- **Kept:** the top 16 outgoing and top 16 incoming edges per component, about 1.3M in all, plus per-layer-pair counts above the noise floor.
- **Noise floor:** the 99.9th percentile of scores against randomly rotated reads.

  | edge type | floor |
  |---|---|
  | neuron → neuron | 12.1 |
  | neuron → head | 11.6 |
  | head → neuron | 2.7 |
  | head → head | 1.9 |

  The neuron → neuron floor matches theory: for single-direction pairs the 99.9th percentile is about 11–12.
- **Compute:** CPU at 4 threads. Edges took 228 s and UMAP 65 s; the logit lens took 4 s on the GPU.

## Checks (all passed)
- **Head slices** rebuild `o_proj` and `out_proj` exactly.
- **Edge order:** 0 edges point to an earlier layer, and every same-layer edge runs attention → MLP.
- **Empty rows** are exactly as expected: the 3,584 layer-23 neurons have no outputs, and the 16 layer-0 heads have no inputs.
- **Known answer:** u15's top MLP readers in layers 16–23 come out in the same order as in Phase 3 (`03_lens.md`): L17.mlp.20 (gate), L22.mlp.89 (gate), L22.mlp.103 (up), L22.mlp.1100 (up), L23.mlp.78, L16.mlp.438. L17.mlp.20's score of 106 equals 1024 × 0.322².
- **Hub diagnostic** (`06_wiring/hubs.json`): no shared direction dominates.
  - The largest eigen-share is 0.72% for writes and 1.5% for gate reads, against 0.098% uniform, and none passes 20× uniform.
  - The leading write and read directions are nearly unrelated (|cos| ≤ 0.1).
  - So the edges are raw; no deflation was needed.

## Findings
1. **The wiring is far above chance for every edge type.** Shares of pairs above the floor, against 0.1% by chance:

   | edge type | share above floor | × chance |
   |---|---|---|
   | neuron → neuron | 0.33% | 3× |
   | neuron → head | 0.50% | 5× |
   | head → neuron | 1.03% | 10× |
   | head → head | 17.3% | 170× |

2. **Mostly local.** Density falls steadily with layer distance: neuron → neuron goes from 0.70% between adjacent layers to 0.12% across 22 layers. Same-layer attention → MLP is the densest of all, at 4.0%.
3. **A middle-layer core.** Every edge type is densest around layers 7–14, and 9 → 10 is the densest pair for neuron → neuron, head → head and (with 9 → 9) head → neuron. The top head → head pair has 71% of its edges above the floor.
4. **u in the wiring.** 10 of u15's 40 strongest writers are layer-15 neurons, the layer Sparks edited. u19's top writers are spread across layers 15, 16, 23 and 2, so layer 19 doesn't stand out the same way.
5. **The embedding map groups components by what they write, not by layer.** The median cluster draws only 7% of its members from its most common layer, against about 4% if layers were mixed at random. Attention heads concentrate in a few clusters.
6. **Vocabulary readouts mean most in late layers.** Late neurons read cleanly, e.g. L23.mlp.1969 → "fluid / liquid / 液体" and L23.mlp.1465 → "eighth / seventh / 8". Early and middle ones are mostly multilingual fragments. That's a known limit of the logit lens, not a property of the neurons. So the cluster labels in the page, which come from each cluster's average direction, are weak.

## Limits
- Static weights ignore nonlinearities, actual activation sizes and input-dependent attention. Edges show what *could* connect, not what does on real text. Level 3 (the activation atlas) tests that.
- Heads are approximated by their top write singular direction in the map and lens.
- Only the top-16 edges per component are kept, so weaker but numerous connections are missing from the lists. They are still counted in the heatmap.
- UMAP distances are approximate.

## Files
- `06_wiring/dirs.npz` (530 MB, all directions), `hubs.json`, `edges_raw.npz`, `heatmap_raw.json`, `uhood.json`, `umap.npz`, `lens.json`
- `06_wiring/page/wiring.html` + `page/data/` (meta.json + one file per layer, 28 MB). Viewing it locally needs a web server, because the page fetches its data:
  `cd results/06_wiring/page && nix-shell -p python3 --run 'python -m http.server 8000'`, then open http://localhost:8000/wiring.html
