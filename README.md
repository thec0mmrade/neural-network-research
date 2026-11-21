# Neural network research: reading and editing Qwen3.5-0.8B

This repository holds an analysis of three articles on neural-network interpretability and weight-space editing, and an experiment that tests the analysis on a real model.

- **[`ANALYSIS.md`](ANALYSIS.md)** compares the three articles and draws conclusions C1–C8 from them. Its last sections summarize what the experiment found.
- **[`qwen-weight-probe/`](qwen-weight-probe/)** is the experiment. It analyzes Qwen3.5-0.8B and the weight edit released for Experiment 8 of "Sparks of Weight-Space Programming", then runs causal tests on the GPU.
  - Start with [`results/FINDINGS.md`](qwen-weight-probe/results/FINDINGS.md).
  - The [README](qwen-weight-probe/README.md) and [CLAUDE.md](qwen-weight-probe/CLAUDE.md) explain how to run it (Nix only).
  - [`TODO.md`](qwen-weight-probe/TODO.md) lays out the plan: structure, wiring, a neuron atlas, circuits, and editing from the atlas.

## The articles

The analysis is based on these three articles. They aren't included here; read them at their sources.

- Nick Cammarata (Goodfire), "Neural Networks Are Already Clean". Interactive version at already-clean.goodfire.com.
- Takehi, Teoh, Zhang and Khattab, "Sparks of Weight-Space Programming".
- honnibal.dev, "The 'Astra can read and write weights to improve Qwen' post is misleading", a rebuttal of the Sparks post.

## What isn't in the repository

The large and rebuildable experiment data is left out (see `.gitignore`): the edited weight tensors, model generations, wiring-page data, and the level-3 atlas data. The atlas data is about 1.3 GB, is still being written, and includes corpus web text. Its reports are kept. The Sparks logs are a separate clone (`qwen-weight-probe/external/`); `qwen-weight-probe/CLAUDE.md` explains how to get them and how to rebuild the edits.
