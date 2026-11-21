# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Weight-level analysis of Qwen3.5-0.8B (`Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`) and of the weight edits released for Experiment 8 of "Sparks of Weight-Space Programming". It tests the conclusions in `../ANALYSIS.md`. The plan behind it is in phases 0–4. Each phase writes `results/NN_*.{json,md}`.

## Environment: nix only
Use nix-shell for everything, never pip, uv or venv.
- `nix-shell nix/cpu-shell.nix --run 'python -m probe.<module>'`: numpy-only shell for the weight-only phases (`survey`, `edit`, `lens`). It builds in seconds.
- `nix-shell --run '…'` (`shell.nix`): the full GPU environment, for Phase 4 model runs.
  - It imports nixpkgs with `cudaSupport = true`. That gives a CUDA torch, and it makes the package hashes match what `cache.nixos-cuda.org` builds; that cache is configured system-wide on this machine. The result is about 6 GiB of downloads and no local CUDA compiles.
  - Don't switch back to a `torch-bin` override, or drop `cudaSupport`. Either one makes the hashes miss the cache, and then nix compiles NCCL and libnvshmem on every core. That compile hard-crashed this machine on 2026-11-13.
  - If anything does have to compile, cap the load with `--cores 8 --max-jobs 1`.
  - `math-verify` and `latex2sympy2-extended` are packaged from pinned PyPI wheels in `nix/math-verify.nix`.
  - The driver comes from `/run/opengl-driver/lib`.
- Both shells put `src/` on `PYTHONPATH`.

## Layout and data flow
- `src/probe/weights.py`: memory-mapped, numpy-only safetensors reader.
  - bf16 values are kept as uint16 bits and widened to f32.
  - Provides tensor naming and `edited(run, name)`.
  - Full-attention layers are 3, 7, 11, 15, 19 and 23; the others are linear attention (`linear_attn.out_proj`).
  - All RMSNorms scale by `(1 + w)`.
- `external/lm_understand-logs/`: sparse clone of the Sparks logs (`exp8_math` only). Treat it as read-only third-party data.
- `results/edits/<run>/`: the edited tensors as `.npy`. Rebuild them with the logs' own script, which checks sha256:
  `nix-shell nix/cpu-shell.nix --run 'python -I external/lm_understand-logs/exp8_math/code/apply_edit.py <base.safetensors> external/lm_understand-logs/exp8_math/runs/<run>/edit results/edits/<run>'`
- `src/probe/structure.py` (CPU shell, BLAS capped at 8 threads, ~35 s): level 1, stats for all 320 tensors, outlier flags and residual-dim analysis. Writes `results/01_structure.{json,md}` and rebuilds `results/01_structure/page.html` from `page.template.html` (data embedded, opens as a local file). The hand review in the `.md` lives in `REVIEW` inside the module.
- Run order: `survey` → `edit` (writes `results/u15_p50c.npy`, `u19_p50c.npy`, `u_p10c.npy`) → `lens` (reads those files).
- `src/probe/model.py` (GPU shell): loads the text-only `Qwen3_5ForCausalLM` from the logs' text config.
  - It maps the multimodal checkpoint's `model.language_model.*` weights onto `model.*`, and can apply an edit run.
  - Sampling copies vLLM's order (presence penalty, then temperature, then top-k, then top-p) through a custom `LogitsProcessor`, because HF `generate` can't do that natively.
  - Also contains grading (last `\boxed{}` checked with math-verify) and loop detection (16-gram repeats).
- `src/probe/causal.py`: Phase 4, run as `gen <cond>` → `analyze` → `gen base_anti|p50c_pro|p50c_rand` → `report`. Run `report` in the GPU shell; it imports torch.
  - The steering conditions need `results/04/analysis.json` from `analyze`.
  - Use generation batch 32 at 4K tokens. Batch 64 runs out of memory on the 12 GB card, because the eager full-attention path materializes repeated KV.
  - Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  - Without the `flash-linear-attention` kernels it runs at about 550 tokens/s and takes about 8 minutes per condition.
- Dose-response sweep: run `gen <cond>_x<s>` (e.g. `p50c_pro_x0.25`) to steer at s × β. Plain names are 1×, and the random directions stay fixed (seed 123) at every strength. Then `nix-shell nix/cpu-shell.nix --run 'python -m probe.sweep'` writes `results/05_sweep.{json,md}`.
- `src/probe/wiring.py`: the level-2 wiring map (see `TODO.md`), written up in `results/06_wiring.md`.
  - Stages run in order: `build → hubs → edges → uhood → umap → lens → export`.
  - All stages except `lens` run in the CPU shell. The module caps BLAS and numba at 4 threads, so it can run alongside GPU jobs without the all-core load that crashed the machine.
  - `lens` needs the GPU shell. It takes seconds.
  - `export` writes `results/06_wiring/page/data/` (meta.json + one file per layer). The page `results/06_wiring/page/wiring.html` fetches those files, so view it through a web server (`python -m http.server` in `page/`), not as a local file.
- `src/probe/circuit.py`: the level-4 looping circuit, written up in `results/08_circuits/looping.md`. Stages run in order: `attn → upstream → downstream → ablate → confirm <cond>`, all in the GPU shell.
  - Full backprop through the model runs out of memory on 12 GB, because the torch fallback for linear attention keeps too many intermediates. Stage 2 therefore uses direct-path attribution: one no-grad forward, plus a gradient through a single MLP with the per-token RMS scale frozen.
  - Capture hooks index full-sequence positions. Never call a hooked module again on a subset of tokens, or the hook fires again and indexes out of range.
  - Attention weights come only from the eager path. Stage 1 forces it and scores each layer inside its hook.
  - `copysupp` (stage 6) checks whether the copy heads' direct output works for or against the copied token.
- `src/probe/closing.py`: the level-4 closing circuit, written up in `results/08_circuits/closing.md`. Stages run in order: `pilot → corpus → direction → attribution → screen → confirm <cond> <seed> → pool → mechanism → followup`, all in the GPU shell. Extra conditions: `ban` (blocks `</think>` at `lm_head`), `dperpnd_plus` (d⊥ minus its direct `</think>`-logit component), `dperpnw_plus` (minus its top-20 direct-logit tokens), `dperp15_plus` (L15 only), `blind:<a>-<b>` (layers a..b can't read the d⊥ span).
  - Strength suffixes need a decimal point: `dperp_plus_x0.25`. The filename tag drops the dot (`x025`); passing `_x025` would mean 25× and is now rejected.
  - The subject is the p50c-edited model (`SUBJECT`), because the base model rarely closes its thinking.
  - Generation uses an 8K cap at batch 16 (the same KV memory as 32 × 4K); one condition takes about 21 minutes.
  - Problem set A (128) is for fitting, set B (64) only for `confirm`; keep them separate.
  - `confirm` conditions combine with `+` (e.g. `dperp_plus+abl:L22.lin.h13:mean`); ablations are set up before steering so their mean input is unsteered.
  - `pool` writes `stage5_pooled.json` with its own Fisher exact test (no scipy in the shell).
- `src/probe/atlas/`: the level-3 neuron atlas of layers 14–20, all in the GPU shell. Design: `docs/superpowers/specs/2026-11-18-level3-atlas-design.md`. Output: `results/07_atlas/` (with `atlas_neurons.jsonl`), and the report `results/07_atlas.md`.
  - Start or resume: `nix-shell --run 'python -m probe.atlas run'`. Stages run in order: `corpus → thinking → assemble → record → pilot → layers → research → report`. A killed run resumes where it stopped.
  - The run stops at the pilot gate with `waiting_for_pilot_approval`. Review `results/07_atlas/pilot/<prompt_version>/report.md` (now v2; v1 is kept as the record of the refused run), approve with `nix-shell --run 'python -m probe.atlas approve-pilot'`, then run again. Only the user decides to approve.
  - Pilot v2 was approved on 2026-11-20 and the full run is under way. To resume after a crash, relaunch detached with the single-line command in `README.md`, and don't approve again. Launch with `setsid` and with `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` unset, so a closed terminal can't kill the run and it can't bill the API.
  - Progress: `nix-shell --run 'python -m probe.atlas status'` prints the stage and overall share. The live viewer is `nix-shell --run 'python -m probe.atlas_watch'`; pass a path to watch another `status.json`.
  - Claude calls go through the `claude` CLI on the Max login. Never pass `--bare`: it reads only `ANTHROPIC_API_KEY` and ignores the Max login.
  - Changing a prompt requires a new `prompt_version` in `config.py` and a new approved pilot.
  - Test-writing refusals: when Opus refuses one neuron's tests, the run asks `test_fallback_model` (Haiku 5.5) for that neuron (R40). Those records carry `test_model`, and the report lists them separately. To retry failed neurons in a layer, move their `tests/batch_<nid>_1.json` aside before that layer's test phase ends, then restart.
  - `tests/atlas/test_end_to_end.py` rehearses the whole pipeline on the GPU through `tests/atlas/fixtures/rehearsal_cli.py`: a tiny corpus in a temporary folder, a fake Claude and local web text. The launcher lives only in `tests/`, so a normal run can't enter rehearsal mode.
- `results/FINDINGS.md`: the summary, tied back to `../ANALYSIS.md`.
- `TODO.md`: map levels 1–4. `VISUALIZATION_METHODS.md`: survey of how LLM internals are usually visualized.
