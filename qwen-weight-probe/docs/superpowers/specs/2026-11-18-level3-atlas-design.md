# Level 3 functional atlas: design

Date: 2026-11-18. Status: awaiting approval.
Project: `qwen-weight-probe` (read its `CLAUDE.md` and `TODO.md` first). Not a git repository.

## 1. Intent

Build a **complete reference for all 25,088 MLP neurons in layers 14–20** of Qwen3.5-0.8B (7 layers × 3,584). Every neuron gets recorded behavior, a written explanation, and a test of that explanation. The test standard comes from "Neural Networks Are Already Clean" (AC): text written from the explanation alone should fire the neuron, near misses should not, and distant concepts should not. AC's scores were 76% / 11% / about 1%.

The atlas also answers one research question from `TODO.md`: do p50c's 128 edited neurons and the top u-writers have coherent explanations, and do those match the u lens ("move on" vs "reconsider")?

## 2. Goals, non-goals, anti-goals

**Goals**
- Every in-scope neuron has: top examples, histograms, an explanation, a job label, and test scores.
- Scores are reported per layer and overall, next to AC's.
- A power cut or crash at any point loses minutes of work at most, and the final results match an uninterrupted run.
- Progress can be watched live in a terminal.

**Non-goals**
- Layers outside 14–20.
- Attention heads.
- The browsable end-user atlas. Its form, and whether it is published, is decided after the pilot (section 11). This design must only leave data it can be built from.

**Anti-goals (a failure even if everything runs)**
- Explanations that read well but were never tested.
- Scores that are not comparable across neurons because partial or cherry-picked subsets were reported as the whole.
- A test writer that has seen the neuron's examples. That breaks the blind test.
- Spending the full Max allowance before the pilot shows the explanations are worth it.

## 3. Decisions made by the user

| decision | choice |
|---|---|
| Scope | All 25,088 neurons, complete reference |
| LLM | Claude Opus (`claude-opus-5-5`), run through Claude Code on the user's Claude Max plan. No API key, no API billing |
| Corpus | Option C: 50M tokens, mixed |
| Processing order | Layer by layer, each finished end to end before the next, starting with L15 and L19 |
| Live view | A Python program in a terminal window, not a browser page |
| Progress | An overall completion bar at the bottom of that view |
| Recovery | Resume after power loss or crash |
| Max allowance | The run may use all available allowance, with no reserved share. The user accepts hitting limits in their own sessions while it runs |

## 4. Approaches considered (processing order)

1. **Layer by layer (chosen).** Finished, scored layers arrive early, and if Max usage runs slower than expected the result is still a set of complete layers.
2. All explanations, then all tests. Simplest, but no scores until about halfway, so a weak prompt could waste much of the allowance.
3. Important neurons first. Answers the research question soonest, but partial results are a biased sample and can't be compared with AC.

## 5. Constraints

- **Nix only.** No pip, uv or venv. GPU work in `shell.nix`, CPU work in `nix/cpu-shell.nix`.
- **New dependency:** `pyarrow` (to read the corpus parquet files) is not in the GPU shell. Dry-run first; if anything would compile, cap with `--cores 8 --max-jobs 1`. Keep `cudaSupport = true`.
- **Machine safety:** this machine hard-crashed under an all-core load. GPU jobs use `OMP_NUM_THREADS=4`, CPU work is capped at 8 threads, and only one GPU job runs at a time.
- **GPU:** RTX 4070, 12 GB. Batch ≤ 32, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Disk:** 537 GB free. Expected use about 3–4 GB kept, plus about 1 GB of checkpoints. Do not store full activations (about 2.5 TB).
- **Model:** pinned revision `2fc06364715b967f1860aea9cf38778875588b17`. Load with `probe.model.load_model`.

### Claude Code requirements (verified against `claude --help`, version 2.1.316)
- Run non-interactively with `claude -p`, `--model claude-opus-5-5`, `--tools ""`, `--json-schema`, `--output-format json`, `--no-session-persistence`, `--strict-mcp-config`, `--disable-slash-commands`, `--setting-sources ""`, and `--system-prompt` for the stage's fixed instructions.
- **Do not use `--bare`.** It reads only `ANTHROPIC_API_KEY` and never the Max login.
- **Make sure `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` are unset** in the driver's child environment, so calls can never bill an API account.
- Run from an empty working directory, so no project `CLAUDE.md` is loaded.
- The corpus is untrusted web text. It is passed as data, tools are disabled, and any instruction inside it is ignored.

## 6. Definitions

- **Neuron activation:** element *i* of `silu(gate_proj(x)) * up_proj(x)`, the input to `mlp.down_proj`, at each token position. `x` is the MLP's input.
- **Sign regime:** the pair (sign of gate, sign of value). Recorded because AC warns about the "double negative" effect.
- **Pole:** the positive or the negative side of a neuron's activation, counted **only where the gate is positive**. Gate-negative positions are the "double negative" regime: they are tracked separately (section 7.3) and never enter a pole's examples, extreme or test score.
- **Eligible position:** any token except the first token of a document and special or chat-template tokens.
- **Extreme:** the largest absolute activation of a pole over eligible positions.
- **Dominant pole:** the pole with the larger extreme.
- **Fires:** absolute activation ≥ 30% of the pole's extreme.
- **Dead neuron:** fewer than 100 eligible positions with absolute activation ≥ 30% of the median extreme of its layer. Dead neurons are counted and reported, and are not sent to Claude.
- **Subject model:** the unedited base model.

## 7. Pipeline

Stages run in order: `corpus → thinking → record → pilot → layers → report`.

### 7.1 Corpus (about 50M tokens)

| source | tokens | notes |
|---|---|---|
| English web text, FineWeb (`HuggingFaceFW/fineweb`, ODC-By) | about 32M | whole documents; only the parquet row groups needed are read |
| Multilingual web text, FineWeb-2 (`HuggingFaceFW/fineweb-2`, ODC-By) | about 5M | a spread of languages, including Chinese and Indonesian, whose words appeared in the closing direction's lens |
| Math problems | about 3M | every problem and solution in `gsm8k_train`, `math_train`, `aime` from the logs' `practice/` folder |
| Base-model thinking | about 10M | generated fresh in stage `thinking`, about 7 h |

- Keep long documents whole, up to a cap of 4,096 tokens, because short contexts made long-context neurons look polysemantic in AC.
- Fixed seed and fixed document order, so every run sees the same token sequence.
- Store as sharded token-id files with a manifest and checksums.

### 7.2 Thinking generation
Base model, Sparks think35 sampling via `probe.model.generate`, batch 32, 4K token cap, on GSM8K and MATH problems. About 2,500 answers. Unedited and unsteered only. Saved generations in `results/` are not reused, because they are nearly all edited or steered.

### 7.3 Recording
One pass over the corpus at layers 14–20. For each neuron keep:
- top 50 positive and top 50 negative activations, at eligible positions, with **at most 3 examples per document** so loops and repeated text can't fill the list;
- 50 random examples;
- top 20 double-negative (gate-negative) activations, and a flag when they would have reached a pole's top 50;
- for each example: document id, token position, sign regime, whether it is inside repeated text (`probe.model.repeated_mask`), and a 64-token window of that neuron's activations;
- activation histograms: 256 bins, one per sign regime;
- count, mean, max, min, and the share of positions where it fires.

**p50c comparison:** for the 128 edited neurons (64 in L15, 64 in L19), also record the same statistics on the p50c model over a fixed 5M-token subset.

Measure recording throughput in the first minutes and show the estimated time before continuing.

### 7.4 Explanation
- Batches of about 20 neurons per `claude -p` call.
- Input per neuron: 20 top positive, 10 top negative, and 10 random examples, each cut to a 24-token window around its peak, with activations marked only on tokens at ≥ 20% of the extreme, plus a histogram summary. The target is about 1,500 prompt tokens per neuron. The full 64-token windows stay on disk for the final atlas.
- Output, validated against a JSON schema: for **each pole**, an explanation, the context it needs (plain text or reasoning), and a job label of **one job / two related jobs / two unrelated jobs / unclear**.

### 7.5 Testing
A **separate** session, which sees only the explanation text and never the examples or the neuron's identity. For each neuron's dominant pole it writes:
- 10 **blind maximum** examples, meant to fire the neuron strongly;
- 10 **near misses**, close to the explanation but which should not fire it;
- 10 **distant concepts**, unrelated to it.

Each example is a passage of up to 128 tokens. The test writer marks it **plain** (run as bare text) or **reasoning** (run inside an open `<think>` block after the chat template), following the context named in the explanation.

> **Amended 2026-11-19 (ruling R38, prompt v2).** In the v1 pilot, Opus's safety filter (`[reasoning_extraction]`) refused all 200 test-writing calls. They asked for passages written as a model's reasoning, and Sonnet refused them too. Since v2 the test writer sees only the explanation and writes every passage as plain text, and every passage runs as bare text. Poles whose explanation names a reasoning context keep counting in the primary score. They are also flagged `tested_out_of_context` and reported in a separate `reasoning_context` row, because they are probably under-scored. The v1 pilot is kept in `results/07_atlas/pilot/v1/` as the record.

The base model then runs each example on the GPU. **Score:** the share of examples whose strongest activation on the tested pole, at eligible gate-positive positions, reaches ≥ 30% of that pole's extreme (AC's threshold), per category.

**Scores are fixed before the pilot:**
- **Primary (complete atlas):** all live neurons, dominant pole. `unclear` neurons count normally, and `failed` neurons count as scoring zero.
- **AC-matched (the comparison with 76% / 11% / about 1%):** live neurons whose dominant pole is positive and whose positive pole is not labeled `unclear`. AC's figures come from neurons whose explanations describe their positive side.

**Control:** for every pilot neuron, also score the blind-maximum examples written for a different, randomly paired neuron. Good explanations should score far above this.

### 7.6 Pilot (a required gate)
200 neurons, sampled at random and evenly across the 7 layers, run through explanation and testing. Report to the user and **stop**:
- the three scores and the control, next to AC's;
- tokens used (from each call's JSON output), the change in the `/usage` figure the user reads before and after the pilot, and the projected duration of the full run;
- 10 sample neurons to read.

The full run starts only on the user's explicit go-ahead. Prompts and effort level may be tuned here. Pilot neurons are re-run with the final prompts so the atlas uses one prompt version.

### 7.7 Research question
**Neurons studied:** p50c's 128 edited neurons, and the top 64 u-writers in each of L15 and L19, the same definition as `results/02_edit.md`. These neurons are tested on **both poles**.

**Controls:** an equal number of random live neurons from L15 and L19.

**Coherence:** compare job labels and test scores between studied and control neurons.

**Match with the u lens:** a separate blind session sees explanations from studied and control neurons mixed together, with no group labels, and rates how well each fits "move on / wrap up" or "reconsider / keep deliberating". Compare the two groups' ratings.

### 7.8 Layer run
Order: **L15, L19**, then L18, L16, L17, L14, L20. Each layer is fully explained, tested and scored before the next begins.

### 7.9 Report
`results/07_atlas.md`: method, scores per layer and overall against AC, the job-label distribution, the research question, and limits. Also join each neuron to its strongest level-2 wiring edges (`results/06_wiring/edges_raw.npz`). Update `FINDINGS.md`, `TODO.md`, `ANALYSIS.md` and the project `CLAUDE.md`.

## 8. Resume

`python -m probe.atlas run` is the only start command. It always continues from the last valid save.

| stage | saved unit | maximum loss |
|---|---|---|
| Corpus | each finished shard | one shard |
| Thinking | each finished batch | one batch |
| Recording | full checkpoint about every 5 minutes | about 5 minutes |
| Explanation and testing | each validated batch | one batch of about 20 neurons |

- **Atomic writes:** write to a temporary file, flush and fsync, rename, then fsync the directory.
- **Checksums** on every checkpoint. Keep the last two; if the newest is damaged, fall back to the previous one.
- **No double counting:** a recording checkpoint holds the state and the exact corpus position together.
- **Run fingerprint:** corpus manifest hash, model revision and prompt version. On a mismatch, refuse to resume and say why.
- **Usage limits:** there is no weekly cap. The run uses whatever allowance is available; on a Max limit it pauses, shows the status, and continues automatically once the limit resets.
- **Failed calls:** retry with backoff. A batch that keeps failing or is refused is split, and any neuron that still fails is marked `failed` with the reason, counted in the report, and never silently dropped.
- After a power cut the user restarts the command. It does not start itself.

## 9. Terminal viewer

`nix-shell nix/cpu-shell.nix --run 'python -m probe.atlas_watch'`, built with `rich` (already in the CPU shell).

- Read-only. It reads status files written atomically by the pipeline, refreshes about every 2 seconds, and can be opened or closed at any time without affecting a run.
- **Panels:** pipeline stages with speed and time left; per-layer bars (each character is about 60 neurons, colored by state, then by score); Max usage and pause status; running scores next to AC's; the latest neuron with highlighted activations.
- **Bottom line:** an overall completion bar, weighted by each stage's estimated duration, with percent, current stage, elapsed time and time left. Weights update from measured speed. Time paused at a Max limit is shown separately.

## 10. Verification

- **Activation check:** recorded values match a direct computation of `silu(gate)·up` on a small sample.
- **Top-k check:** on a small corpus, streaming results match a brute-force computation.
- **Resume test:** force-kill (`SIGKILL`) during each stage, resume, and confirm results are identical to an uninterrupted run on a small corpus.
- **Damage test:** corrupt the newest checkpoint and confirm the fallback.
- **Blindness test:** inspect the test-writer's actual prompt to confirm no examples or neuron identity are present.
- **Max-only test:** confirm the child process has no API credentials in its environment.
- **Viewer test:** the viewer runs correctly with missing, partial and stale status files.

## 11. Left open

**For the user, after the pilot:** the go-ahead for the full run, and the form of the final browsable atlas, including whether it is published.

**For the builder:**
- internal file formats and module split (`src/probe/atlas.py`, `src/probe/atlas_watch.py`, results under `results/07_atlas/`);
- batching by document length, and how many Claude calls run at once;
- the Opus effort level, chosen in the pilot;
- exact prompt wording;
- how Max limit messages are detected and when the limit resets.

## 12. Limits to state in the report

- AC studied a different kind of model, so the score comparison is a benchmark, not a like-for-like result.
- Only the dominant pole of each neuron is tested, except for the research-question neurons.
- Since R38, every test passage runs as plain text. Neurons described as working only inside reasoning are tested out of context and are probably under-scored, so the report shows them in their own row.
- Test passages are at most 128 tokens, so neurons that build up over long contexts, such as loop-related ones, may be under-scored.
- Coverage depends on the corpus: a neuron that needs text outside it can look dead or be misdescribed.
- The test writer is the same model family as the explainer.

## 13. Choices made by the designer (for the user to check)

These were not decided by the user. Each can be changed before the build.

1. **Subject:** the unedited base model, with p50c's 128 edited neurons recorded separately on a 5M-token subset.
2. **Test passages:** up to 128 tokens, run as plain text or inside an open `<think>` block.
3. **Eligible positions:** first tokens and special tokens are excluded, and each document supplies at most 3 top examples per pole.
4. **Double negatives:** gate-negative activations are tracked separately and never count toward a pole.
5. **Dead neurons:** fewer than 100 strong positions; counted in the report and not sent to Claude.
6. **Two scores, fixed before the pilot:** a primary score over all live neurons, and an AC-matched score for the comparison with AC.
7. **Job labels:** per pole, with "two unrelated jobs" added to AC's categories.
8. **Research question:** the edited neurons plus the top 64 u-writers in L15 and L19, tested on both poles against random controls, with a blind rating against the u lens.
9. **Explanation input:** 24-token windows, to limit Max usage.
10. **Corpus mix:** 32M English web, 5M multilingual web, 3M math, 10M base-model thinking.
11. **Layer order after L15 and L19:** L18, L16, L17, L14, L20.
