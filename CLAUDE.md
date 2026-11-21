# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Nature of this directory

This folder is mainly a **reading collection**, with one experiment subproject. It is not a git repository, and the top level has no build, lint or tests.

- `ANALYSIS.md`: a cross-document analysis of the three PDFs below, with conclusions C1–C8. Its last section summarizes what the experiment found.
- `qwen-weight-probe/`: a Python subproject that tests those conclusions on the real model. It analyzes Qwen3.5-0.8B and the Sparks Exp 8 weight edit, then runs causal checks on the GPU.
  - It has its own `CLAUDE.md`, the source of truth for how to run it. Everything runs in nix-shell.
  - Results are in `qwen-weight-probe/results/`; start with `FINDINGS.md`.
  - The level-3 neuron atlas (layers 14–20) has been running since 2026-11-20 and will take weeks. Its status and how to resume it are in `qwen-weight-probe/README.md`.
  - Level-4 circuit write-ups are in `qwen-weight-probe/results/08_circuits/`: `looping.md` (the u axis and copy heads) and `closing.md` (the `</think>` decision).

The PDFs are saved articles (browser "print to PDF" exports) about neural network interpretability and weight-space editing:

- `already-clean.pdf`: "Neural Networks Are Already Clean" by Nick Cammarata (Goodfire), 44 pages. Argues that polysemanticity in early models (e.g. InceptionV1) came from fixable architectural and tooling problems rather than superposition, so networks can be studied directly without SAE-style decomposition. The interactive version is at already-clean.goodfire.com.
- `Sparks of Weight-Space Programming_ ...pdf`: by Takehi, Teoh, Zhang and Khattab, 31 pages. Claims that frontier LLMs ("GPT-6 Astra") can read raw network weights and edit them like code. Examples include identifying which game a policy network was trained on, writing game-playing networks from scratch, and editing 0.016% of Qwen3.5-0.8B's weights to raise MATH-500 from 7.9% to 48.1%.
- `The 'Astra can read and write weights to improve Qwen' post is misleading · honnibal.dev.pdf`: a 4-page rebuttal of the Sparks post from honnibal.dev. It argues that the abstract mixes up experiments: the Qwen and game-network results relied on writing and running Python, not only on reading raw weights. It includes a per-experiment table covering what each one reads, writes and runs.

The Sparks post and the Honnibal rebuttal are a claim/counter-claim pair, so read them together.

## Reading the PDFs

The system is NixOS and poppler is not installed by default, so the `Read` tool cannot render PDF pages here. To extract text, use a temporary shell:

```sh
nix shell nixpkgs#poppler-utils -c pdftotext -layout "<file>.pdf" -
nix shell nixpkgs#poppler-utils -c pdfinfo "<file>.pdf"
```
