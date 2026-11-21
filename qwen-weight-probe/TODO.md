# TODO: mapping Qwen3.5-0.8B at four levels, then editing from the maps (level 5)

Each level builds on the one before. For how maps like these are usually visualized, see `VISUALIZATION_METHODS.md`. Shared conventions:
- **Environment:** nix-shell only. Use `nix/cpu-shell.nix` for weight-only work and `shell.nix` for GPU work. Dry-run before any new nix dependency, and cap builds with `--cores 8 --max-jobs 1` if anything would compile.
- **GPU:** RTX 4070 (12 GB). Generation at batch ≤ 32, with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Outputs:** one module per level in `src/probe/`, writing `results/NN_<name>.{json,md}`. Large arrays go under `results/<name>/`.
- **Reuse:**
  - `probe.weights` (`Checkpoint`, `layer_tensor`, `edited`)
  - `probe.lens` (`vocab`, `show`, `lens`)
  - `probe.model` (`load_model`, `generate`, `sample_problems`, `repeated_mask`)
  - `probe.causal` (`steering_hooks`)
- **Model facts:**
  - 24 layers; full attention at 3, 7, 11, 15, 19, 23, linear attention elsewhere.
  - MLP width 3,584, so 86,016 MLP neurons.
  - 8 heads × 256 dims in the full-attention layers; 16 heads × 128 in the linear-attention layers.
  - All norms scale by `(1 + w)`; embeddings are tied.

---

## Level 1: Structural map (weights only, complete)
Mostly done in Phase 1 (`results/01_survey.*`, covering output projections only).

**Status: done 2026-11-16** (`results/01_structure.md`, page `results/01_structure/page.html`; code `src/probe/structure.py`, CPU, 8 threads, 35 s). Key finds: residual dim 0 is written ~2.6× harder by 43/48 output matrices and almost erased by the final norm (0.05×); layer 23 is the most unusual layer (v_proj top direction 11× typical); no single neuron dominates any matrix. Published as a private artifact: https://claude.ai/artifact/64Ppepj8rYbwVxYkYGVQhw

- [x] Extend `probe.survey` to every matrix, not just the output projections:
  - `q/k/v_proj`, `in_proj_{qkv,z,a,b}`, `gate/up_proj`;
  - per-layer linear-attention parameters (`A_log`, `dt_bias`, `conv1d`);
  - the norm weights.
- [x] Per-matrix stats: Frobenius norm, spectrum (top singular values, effective rank), row and column norm outliers, and the bf16 value distribution.
- [x] Flag outliers: unusually large single columns or neurons, and any norm weights far from typical (e.g. the final norm mean +3.3).
- [x] Deliverable: `results/01_structure.{json,md}`, plus an explorable artifact page with a per-layer heatmap of the stats.
- **Done when** every one of the 320 language-model tensors is covered, and the outlier list is reviewed by hand.

## Level 2: Wiring map (weights only, approximate)
Static "virtual connections" through the residual stream, extending the Phase 3 method from u to every component.

**Status: done 2026-11-15.** Results are in `results/06_wiring.md`, and the page is `results/06_wiring/page/wiring.html`. Changes from the plan:
- Edge scores use one energy-based measure for every type, 1024·‖R·W‖², where a random direction scores 1.
- The hub check found no dominant direction, so no deflation was needed.
- The lens labels for early and middle layers are weak (a logit-lens limit).
- Published as a private artifact: https://claude.ai/artifact/1ZVbko75gvYQxnSZfEYAVU

- [x] Write directions for each component:
  - MLP neuron: `down_proj` column;
  - full-attention head: the slice of `o_proj` columns;
  - linear-attention head: the slice of `out_proj` columns.
- [x] Read directions for each component:
  - MLP neuron: `gate`/`up` rows × (1 + w) of that layer's pre-MLP norm;
  - attention head: the query/key/value rows × (1 + w) of the pre-attention norm.
- [x] Virtual weight from component A (earlier layer) to B (later layer) = cos(write_A, read_B). Store the top-k incoming and outgoing edges per component as sparse data, not the full 86k × 86k matrix. Batch it on the GPU or by blocks on the CPU.
- [x] Logit lens per component: the top and bottom 20 tokens of `E @ ((1+g_final) * write)`.
- [x] Controls:
  - **random baseline:** the edge strengths a random direction would get, as the noise floor;
  - **known-answer check:** the strongest readers of u should match `results/03_lens.md`.
- [x] Data deliverable: `results/06_wiring/` (edge lists, per-component lens, view data as JSON) and `results/06_wiring.md` (summary).
- [x] **Visual deliverable:** one explorable artifact page with three linked views and a lookup panel, following `VISUALIZATION_METHODS.md`. Label it clearly as **static wiring from weights**, not a traced computation for a specific prompt.
  - [x] **View A, overview heatmap** (entry point):
    - A 24 × 24 grid of layer → layer wiring strength, with a toggle to split by component type (MLP / full attention / linear attention).
    - Each cell aggregates the edges between two layers: the sum and count of edges above the noise floor.
    - Clicking a cell lists that pair's strongest edges.
    - Check: layers 15 and 19 should stand out for u-related wiring; if not, note it.
  - [x] **View B, neighborhood graph:**
    - Pick a component (or a direction such as u) and show its strongest inputs and outputs, 1–2 hops out.
    - Nodes are laid out left to right by layer; edge thickness shows strength and color shows sign. Hover shows a node's lens.
    - Cap the node count (e.g. top 8 edges per hop) so it stays readable.
    - It borrows the attribution-graph visual language, which makes the static-wiring label above essential.
  - [x] **View C, embedding map:**
    - Every component placed in 2D by its write direction (cosine similarity, via UMAP), colored by layer, with clusters labeled from their lens tokens.
    - Make it zoomable and hierarchical, not one flat scatter: a coarse cluster level with ~50–200 landmarks, then individual components on zoom. This avoids the UMAP clutter and distortion problem.
    - Note in the page that 2D distances are approximate.
    - Needs `umap-learn` (0.5.12 in nixpkgs, along with `scikit-learn` 1.8.0): add it to `nix/cpu-shell.nix` and dry-run first.
  - [x] **Lookup panel** (shared by all views): search by component id (e.g. `L15.mlp.2048`, `L19.attn.h3`). Shows:
    - the positive and negative lens tokens;
    - the strongest incoming and outgoing edges;
    - the cosine with u;
    - whether p50c edited it.
    - Selecting anything in A, B or C opens it here, and clicking an edge in the panel recenters View B.
  - [x] **Size budget:** keep the page under the 16 MB artifact limit. Ship precomputed JSON, top-k edges only, rounded floats, and lazy-load per-layer detail if needed.
- [x] Caveat to record: this ignores nonlinearities, norm scaling by the actual activation size, and input-dependent attention. Level 3 tests how far it holds.
- **Done when** edges above the noise floor exist for all components, and the u check passes.

## Level 3: Functional atlas (needs the model running)
What each neuron actually responds to, with explanations that are tested, not just written.

**Design decisions: decided 2026-11-18** (`docs/superpowers/specs/2026-11-18-level3-atlas-design.md`). The pipeline is built in `src/probe/atlas/`; the checklist below stays open until the real run finishes.
- **Scope:** all 25,088 MLP neurons of layers 14–20.
- **LLM:** Claude Opus (`claude-opus-5-5`) through the `claude` CLI on the Max plan, with no API billing. Never use `--bare`.
- **Corpus:** 50M tokens, mixed: web text, 5M multilingual, the math practice sets and 10M of the base model's own thinking.
- **Order:** layer by layer, each finished end to end, starting with L15 and L19.
- **Cost check:** a 200-neuron pilot that the user must approve before the full run starts.
- **Recovery and viewing:** resumes after a crash or power loss; progress shows in a terminal viewer (`python -m probe.atlas_watch`).

**Status, 2026-11-20: full run started.** The user approved pilot v2 and the run began with layer 15. (It was on hold at the pilot gate on 2026-11-19.)
- Corpus, thinking, assembly and recording are done: 55M tokens, reused by any later run.
- Pilot v2 (`results/07_atlas/pilot/v2/report.md`, 200 neurons, 0 failed):
  - blind maximum 66.5%, near miss 3.4%, distant 0.1% (AC: 76 / 11 / ~1);
  - AC-matched scores 68.1 / 3.2 / 0.3;
  - control 0.9%;
  - the 18 reasoning-context poles score 50.6%, because they are tested on plain text (spec amendment R38).
- Pilot v1 is the record of the refused run: Opus's safety filter blocked every test-writing call that asked for reasoning-style passages.
- **Cost:** the pilot used 1.25M input and 0.44M output tokens, about 6–9% of a weekly Max allowance (estimated from `/usage`, with no before figure). The full run (22,634 live neurons, about 113× the pilot) would take roughly 7–10 weeks of the whole allowance.
- **2026-11-20, R39:** explanation batches were almost all refused by Opus's safety filter (`[reasoning_extraction]`) and split down. That probably wasted about 10M input tokens over the first ~480 neurons, and the run's own token tally misses refused calls. Explanations now go one neuron per call (`explain_batch=1`), with no prompt change.
- **User decision:** approve as is and run layer by layer (L15 and L19 first), stopping after any layer if wanted. Not started: wait for the user to say they are ready before running `approve-pilot`.

- [ ] **Scope first.** Start with layers 14–20 (around u): 7 × 3,584 = 25,088 neurons. Expand to all 24 layers only if the pipeline holds up.
- [x] **Corpus:**
  - general text (an open web-text sample), the math practice sets, and the base model's own thinking-mode generations from `results/04/`;
  - include long contexts, since short ones made long-context neurons look polysemantic in AC;
  - target 10–50M tokens; measure throughput first.
- [x] **Recording** (GPU, batch ≤ 32):
  - For each neuron, keep the top-k positive and top-k *negative* activations with context windows. AC stresses the negative side.
  - Keep activation histograms per neuron, and the SwiGLU sign regimes (gate × value signs), to catch AC's "double negative" effect.
  - Store as sharded `.npz` under `results/07_atlas/`.
- [ ] **Explaining:**
  - An LLM writes a short explanation per neuron from its top examples.
  - Cost check first: price one layer before committing to more. That decision, and which LLM to use, is yours.
- [ ] **Testing each explanation** (AC's standard):
  - **Blind maximum:** an agent that sees only the explanation writes text meant to fire the neuron maximally.
  - **Near misses:** examples that are close but shouldn't fire it.
  - **Distant concepts:** examples of something unrelated, as the packing test.
  - Score as AC did: share of strong responses (≥30% of the neuron's max) for the written examples, the near misses and the distant concepts.
- [ ] Join with level 2: per neuron, show its explanation next to its strongest wiring edges.
- [ ] Deliverable: `results/07_atlas.md` (method, test scores, the distribution of "one job / two related jobs / unclear"), plus an explorable atlas artifact.
- [ ] **Research question:** do the edited p50c neurons and the top u-writers have coherent explanations, and do they match the u lens ("move on" vs "reconsider")?
- [ ] **Release the atlas data** (after the report stage; added 2026-11-21). Publish the explanations, tests and scores as an archive attached to a GitHub release of `thec0mmrade/neural-network-research`, not as files in git: about 200 MB of JSON would slow every clone.
  - **Include:** `atlas_neurons.jsonl`, and per layer `layers/<prompt_version>/L<n>/{explain,tests,scores}/` plus `scores.json`, the pilot reports, and `results/07_atlas.md`.
  - **Leave out:** `corpus/` and `raw/` (third-party web text and the model's own generations, which the pipeline can rebuild), `record/` (777 MB in files over GitHub's 100 MB file limit; release separately later if wanted), `refused_backup/`, `status.json` and `run.lock`.
  - **Data README** (`DATA.md` at the archive root), short:
    - the record format of each file type: the explain record's two poles (`explanation`, `job`, `context`); the test record's `blind_maximum` / `near_miss` / `distant` passages; the score record's hit counts out of 10;
    - the flags: `tested_out_of_context` (R38), `test_model` (Haiku fallback, R40), `failed`;
    - how neuron ids map to names (`nid` → `L<layer>.mlp.<idx>`, idx = nid mod 3,584);
    - what a hit means (≥ 30% of the neuron's corpus extreme, on the tested pole), and that blind maximum measures what triggers a neuron, not its effect;
    - the run's prompt version and models, the rulings that changed results (R38–R41), and a link to the report and to `ANALYSIS.md`.
  - **Check first:** look over a sample of explanations for long verbatim quotes of corpus text. Run gitleaks over the unpacked archive. Confirm GitHub's current limits on release files, since they can change.
  - Archive as `atlas-<prompt_version>-<date>.tar.zst`, with its sha256 in the release notes and linked from the repo README. Create the release only when the user asks.
- **Done when** the in-scope layers are explained and tested, and the scores can be compared directly with AC's (76% / 11% / ~1%).

## Level 4: Circuit maps (needs the model running, one per behavior)
Deep maps of specific behaviors. Phase 4 is the template: direction, causal tests, random control.

**Looping circuit: done 2026-11-15** (`results/08_circuits/looping.md`). Changes from the plan:
- Stage 2 uses direct-path attribution, because full gradient tracing ran out of GPU memory.
- Stage 3's metric changed from copy log-prob to log P(`</think>`), after the push turned out to lower the copy probability.
- Not every claimed edge is backed by a free-generation test: the diagram marks the teacher-forced-only edges as dashed.

- [x] Finish the looping circuit:
  - fold in the strength sweep (`results/05_sweep.md`);
  - trace upstream what drives the u-write: which earlier components and attention patterns feed the edited neurons during loop onset;
  - trace downstream which readers of u change the next-token distribution.
- [x] Ablation map: ablate each top level-2 component around u one at a time (zero, and mean-ablate), measuring the change in loop rate or u-projection on a fixed set of base generations. Teacher-forced, which is far cheaper than generating.
- [x] Attention around loops: which heads in layers 15/19 and their neighbours attend back to earlier copies of the repeated text (copying or induction-like heads, cf. SP Exp 3). Test with head ablation.
- [x] One more behavior, to check the method generalizes: **closing `</think>`, done 2026-11-16** (`results/08_circuits/closing.md`).
  - Subject is p50c, not the base model (the base model closes 2/64 at 8K).
  - Found a closing direction d⊥, ~98% separate from u, that switches closing on/off (192 / 0 of 192), and one head, L22.lin.h13, whose ablation cuts closing 88 → 26/192.
- [ ] Open from closing:
  - [x] does d⊥ act through L22.lin.h13? No: their effects add in log-odds at 1× and 0.25× (stages 5b, 5c).
  - [x] do the late blocks have to read d⊥ for the push to work? No, though they may still be one route (stage 7): blinding L20–23 to the d⊥ span cuts the model's own closing 88 → 31/192, but the 0.25× push still adds +1.19 log-odds (vs +0.95). Whole-block mean-ablation of L22–23 broke the model, so it can't be used;
  - [x] does the push need blocks to read it? Yes (stage 7b): blinding L16–23 gives 3/192 closed, and the 0.25× push adds nothing (2/192);
  - [x] are L18 and L20–23 redundant for closing? Yes (stage 7b): either alone can be blinded, but with both blind the push can't close (2 vs 5/192), though it still halves looping (58 → 29) and raises wrap-up talk (12 → 40/192);
  - which of L16, L17, L19 read the push into the wrap-up state? Blind L16–17 + L19 with L18 + L20–23 left intact, and measure wrap-up talk and looping as well as closing;
  - free-generation tests of the other screen hits (L23.mlp.932, L23.mlp.385, L19.mlp.31);
  - [x] is d⊥ an upstream decision or just a `</think>` logit push? A decision (stage 6): without its direct `</think>` component it still closes 192/192 (later: 2,209 vs 1,372 tokens), and with `</think>` banned it still writes final answers (177 vs 29/192) and loops on wrap-up text;
  - [x] does d⊥ act through the direct logits of other wrap-up words? No (stage 7): minus its top-20 word span 185/192, L15 only 182/192. The direct word effects mostly set the speed;
  - a candidate third behavior: `\boxed{}` answer formatting.
- [x] Repeat stage 5 with several seed sets (done: 3 sets, `stage5_pooled.json`). The base model varies 7–13/64 between sets.
- [x] Compare zero- and mean-ablation of L15.attn.h7 in free generation (done: zeroing is stronger, 5 vs 16/192, p = 0.02).
- [x] Open from looping:
  - [x] explain why copy-head ablation raises the copy log-prob: their usual output works against the copied token, mainly where they don't attend to the copy (looping stage 6). Not classic copy suppression;
  - [x] why mean-ablating L11.attn.h1 raises looping: it feeds the decision to stop, in parallel to d⊥ (closing stage 7: closing 88 → 57/192, looping 11 → 30). How it does so is indirect (it isn't a d_close writer).
- [x] Deliverable: `results/08_circuits/<behavior>.md` per behavior, each with a node-and-edge diagram tied to level-2/3 components, plus the causal evidence (ablation and steering with random controls). Done for looping.
- **Done when** each mapped circuit has every claimed edge backed by an intervention, not just correlation.

## Level 5: Editing from the atlas (experiment, needs levels 3–4)
Can an edit planned from neuron explanations alone, without copying Sparks' result, change behavior as predicted? This tests the "reading" half of weight-space programming directly. Added 2026-11-21.

**Why it is open:** the atlas measures what makes a neuron fire, not what its firing does. Sparks' edit acts through one direction, u, which plain steering reproduces without any neuron labels (`results/FINDINGS.md`, results 6–7). So neuron-level explanations may or may not be the right unit for planning an edit. This experiment settles that for looping.

- **Prerequisites:** L15 and L19 scored, and the atlas research stage run (edited-neuron explanations and the blind u-lens ratings).
- [ ] **Pick neurons from the explanations only.** In L15 and L19, take neurons whose dominant pole is rated "reconsider / keep deliberating" or "move on / wrap up" (reuse the research-stage blind ratings), and whose explanation scored at least 8/10, so the explanation can be trusted. Record the list and the selection rule before running anything.
- [ ] **Write down predictions first.** For each edit: which way looping, closing `</think>` and accuracy should move, and roughly how far. Commit them to `results/09_atlas_edit/predictions.md` before the first generation.
- [ ] **Edits.** Scale each chosen neuron's `down_proj` column: zero it, and flip it with Sparks' gain (−5×) for comparability. Arms:
  - atlas-chosen neurons;
  - the same, minus Sparks' 128 edited neurons, to show it isn't just rediscovering them;
  - **weights-chosen:** the same number of top u-writers from the weights alone (the level-2 lens method);
  - **random control:** the same number of random live neurons from the same layers, fixed seed.
- [ ] **Measure.** The phase-4 test: 64 problems at a 4K cap with identical seeds, plus the three seed sets from closing stage 5. Report looping, closed thinking and correct answers, with Fisher tests against base and against random.
- [ ] **Side effects.** Compare the edited and base models on held-out corpus web text (per-token loss) and on a small MATH sample. Use the atlas to say in advance which unrelated jobs the chosen neurons also carry, and check whether those behaviors change.
- [ ] **Deliverable:** `results/09_atlas_edit.md`, plus a section in `results/FINDINGS.md` and in `../ANALYSIS.md` (it bears on C1 and C2).
- **Done when** each prediction is confirmed or refuted against the random control, and the atlas-chosen arm is compared with the weights-chosen arm. Either answer counts as a result: edits planned from explanations working better than ones planned from weights is evidence for readable weight programming, and failing shows where neuron-level reading stops.

---

## Open decisions (for the user)
- [x] Which LLM writes and tests the level-3 explanations, and the budget for it. Decided 2026-11-18: Opus on Max, with the whole allowance (see Level 3).
- [x] Level-3 corpus source and size. Decided 2026-11-18: 50M tokens, mixed.
- [ ] Whether level-3 and level-4 maps should be published as artifact pages, or kept as local files only.
