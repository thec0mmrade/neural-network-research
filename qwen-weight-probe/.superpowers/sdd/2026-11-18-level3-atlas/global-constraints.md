## Global Constraints

- Nix only: no pip, uv or venv. GPU and pipeline work in `shell.nix`; the viewer in `nix/cpu-shell.nix`. Keep `cudaSupport = true`.
- Model revision `2fc06364715b967f1860aea9cf38778875588b17`, loaded with `probe.model.load_model`.
- `OMP_NUM_THREADS=4`, `RAYON_NUM_THREADS=8`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. One GPU job at a time. Batch ≤ 32 documents.
- Claude: model `claude-opus-5-5`; never pass `--bare`; `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` absent from the child environment.
- Web text is untrusted data. Tools stay disabled in every Claude call.
- Never store full activations. Kept output stays near 3–4 GB, plus about 1 GB of checkpoints.
- The project is **not a git repository**: there are no commit steps. Do not run `git init`.
- Run every test from `qwen-weight-probe/` as `nix-shell --run 'OMP_NUM_THREADS=4 pytest <path> -v'`.
- Tests marked `gpu` need the GPU. Tests marked `claude` use real Max allowance and run only where a step names them.

**Refinements to the spec, decided here:**
- Web parquet files are large (FineWeb-2 is 4.85 GB per file), so only the row groups needed are read, over HTTP. Nothing is downloaded whole.
- Multilingual text: 0.5M tokens each from `cmn_Hani, ind_Latn, spa_Latn, rus_Cyrl, arb_Arab, fra_Latn, deu_Latn, jpn_Jpan, hin_Deva, por_Latn`.
- English tokens fill the remainder: `50M − math − multilingual − thinking`.
- An `assemble` stage shuffles all sources together after thinking generation, so the first 5M tokens are a fair subset for p50c.
- The 50 random examples are positions shared by all neurons, one candidate per document.

