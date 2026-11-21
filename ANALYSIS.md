# Cross-document analysis: reading neural networks (Nov 2026)

## Sources

| Abbr. | Document | Length | Notes |
|---|---|---|---|
| **AC** | Nick Cammarata (Goodfire), *Neural Networks Are Already Clean* (`already-clean.pdf`) | 44 pp | Calls itself a "working hypothesis"; revised Oct 30 2026 |
| **SP** | Takehi, Teoh, Zhang, Khattab, *Sparks of Weight-Space Programming* (`Sparks of Weight-Space Programming_ ...pdf`) | 31 pp | Blog post; FAQ added Nov 12 2026 |
| **HR** | *The 'Astra can read and write weights to improve Qwen' post is misleading*, honnibal.dev (`The 'Astra ...' ... honnibal.dev.pdf`) | 4 pp | Rebuttal of SP |

Every number below was checked against text extracted from the PDFs with `pdftotext`.

---

## 1. Breakdown

### AC: "Neural Networks Are Already Clean"
**Thesis:** Polysemanticity was mostly an artifact, not superposition. Neurons are a network's own legible building blocks, so you can study the network directly instead of first decomposing it with SAEs.

**Four-step argument.** AC itself calls step 4 "highly speculative".
1. There is one learning principle: generalization, not packing.
2. With modern tools, ordinary science on a network works.
3. Understanding compounds as the map fills in.
4. That may be most of what interpretability needs.

**Key evidence**
- **Cat/car neuron 4e:55:** it responds to a shared V or pointed shape. Its circuit links to flames and fox snouts, and its negative pole is an "O" (round shapes).
- **The 2019 "superposition" car→dog circuit:** this was really generalized pose neurons, combined with feature-visualization reward hacking into a "dog basin".
- **InceptionV1's last layer:**
  - ~5% interpretable with the tools of that era.
  - ~50% with fixed tools.
  - ~70% when retrained with modern regularization.
  - ~99% for a parameter-matched ConvNeXt with residuals and LayerNorm.
- **Squeeze tests:**
  - Layer 3b was cut from 480 to 10 neurons. The neurons became *coarser* ("curves & loops"), not superposed, and an SAE on the squeezed layer recovers no clean curve detectors.
  - Same pattern for the ConvNeXt last layer at 13 neurons, and for a residual stream narrowed to 46 dimensions.
- **SAE comparison:** 88% of the SAE's last-layer activity is near-copies of single neurons. Removing the SAE directions that match no neuron leaves accuracy unchanged.
- **Weak activations:**
  - Removing the weakest 75% changes accuracy from 72% to 71%.
  - Removing the strongest 7% drops it from 72% to 4%.
- **Memorization:** teaching the model "umbrella → bee" reuses existing bee circuits. No hidden feature appears.
- **Language models:**
  - Qwen3-32B (preliminary): ~90% of neuron poles have one main job, ~7% have two related jobs, and ~2–4% are not yet understood.
  - GPT-2 XL: under 2% of neurons were clean in 2023; now ~70% look like "roughly one thing", but only ~5% are pristine.
  - Explanation test on 183 neurons. Examples written from the explanation alone fired the neuron strongly 76% of the time, near misses 11%, and maximally distant ("packable") concepts about 1%. 179 of 183 neurons never fired on the distant concepts.
- **New motifs:** recursive generalization ("iy" → "diy" → "do it yourself"), the summary motif on punctuation, a SwiGLU double-negative regime, and an unread saliency coordinate in the residual stream.

**Limits AC states itself**
- The language-model numbers depend on thresholds, and the details are in a "forthcoming" companion paper.
- Attention is "mostly outside the scope". The ~5% "explanatory residual" for naming heads is a low-confidence guess.
- Packing in the residual stream "may not be possible" to rule out entirely.
- The canonical cat/car claim was revised as "too strong" (fn 3).
- AC names the "explanation of the gaps" risk explicitly.

### SP: "Sparks of Weight-Space Programming"
**Thesis:** Frontier LLMs ("GPT-6 Astra") can read and write network weights "natively", like code, without gradients or training loops.

Results are as SP reports them. "Setting" is what Astra had access to.

| # | Task | Size | Setting | Result |
|---|---|---|---|---|
| 1 | Name the game a PPO policy was trained on | 17K | raw weights, no tools | 4/4 |
| 2 | Name the shape a DDPM generates | 24K | raw weights, no tools | 3/3 (drew outlines; real outputs are filled) |
| 3 | Language and copying heads of a char-LM (middle layers only) | 7.2M | Python, no forward passes | 3/3; every copying head found, none falsely named |
| 4 | Fix a stuck Kinetix car policy | 172K | raw weights + 1 recorded episode, no tools | 0% → 100%, editing 3.5% of weights |
| 5 | Write MinAtar policies from scratch | 20K | rules + source code, no tools | Beats PPO-100M. Against PPO-500M: Breakout 517 vs 25, Seaquest 53 vs 89, Asterix 53 vs 55 |
| 6 | Write a 9×9 Go network | 1.5M | Python, 0–50 forward passes | Beats equal-size PPO at 50 passes |
| 7a | Identify the sycophantic Gemma 3 270M | 270M | weights only | **Failed** (all 4 attempts) |
| 7b | Same task | 270M | + attention patterns on ≤16 texts it wrote | 3/3 |
| 8 | Make Qwen3.5-0.8B better at math | 800M | Python + 10/25/50 forward passes | MATH-500 7.9 → 48.1, GSM8K 13.8 → 67.1, IFEval 21.0 → 62.3 (at 50 passes) |
| 8b | Same task on Gemma 3 270M | 270M | same | **Failed**: no gain beyond ~1 point |

**Other points**
- SP releases complete logs (github.com/RikiyaT/lm_understand-logs) and puts the edited Qwen on Hugging Face.
- The FAQ says:
  - Weight-Space Programming is "not a method", only "an emerging concept".
  - "Natively" means "no training loop, no gradient, and no model trained specially to read or write weights".
  - The baselines test only the narrow claim of each experiment.

### HR: the rebuttal
**Thesis:** SP's abstract works like a magician doing one trick several ways. Each experiment uses a different technique, so the reader assumes one capability that doesn't exist.

**Evidence**
- HR's table shows that the "just raw weights" setting (raw floats in context, no tools, network never run) appears only in Exp 1, 2 and 4, on networks of 17K–172K weights.
- Every experiment involving a real language model uses Python.
- The headline Qwen result also runs the model on up to 50 batches.

**Concession:** "AI models of a year or two ago would have failed at all these tasks."

**Scope:** HR criticizes framing and presentation. It does not allege that the data is wrong.

---

## 2. Conclusions across all three documents

### C1. The best-supported shared conclusion: AI agents applying *ordinary* interpretability methods at scale are now productive. "Native weight reading" is a much weaker claim.
- **AC:**
  - Agents reviewed 100,000 attribution graphs and wrote 1,000 "neuron papers".
  - Explanations predicted new examples (76% fired) against near misses (11%).
  - The vision is "a laboratory run by automated scientists" where "each step is simple".
- **SP:** the experiments that reach real language models use known techniques.
  - Exp 3 rediscovered the method of Elhage et al., "which we never mentioned to Astra".
  - Exp 7b used attention patterns.
  - Exp 8 removed one direction, "much like removing a behavior by deleting a single direction from a model's weights".
  - The FAQ: Astra "can be very powerful while only composing known steps".
- **HR:** Astra "can be prompted to devise and perform tasks that a human ML engineer could do", and earlier models would have failed.
- **So:** all three authors, critic included, agree on this narrower claim.

### C2. Reading raw weights works only when the structure sits somewhere readable. It fails when the behavior is spread out.
- **Successes:**
  - Exp 1: the input-layer weights are a spatial map (the paddle on the bottom row).
  - Exp 2: each output-layer unit draws a piece of the shape's edge.
  - Exp 4: Astra had one recorded episode, and its fix adds a small override path.
- **Failures:**
  - Exp 7a: sycophancy could not be found in the weights, even though the two fine-tunes differ only in that behavior. In every attempt Astra found weight differences but couldn't map them to behavior.
  - Exp 8b: Gemma's errors were "ordinary reasoning errors… with no single component behind them to switch off".
- **This fits AC's picture:** legibility lives in localized mechanisms (neurons, heads, and the data flowing between them), not in a raw float dump.
- **AC's framing predicts the 7a → 7b flip.** Attention "where it looks… already tells you a lot", and neurons and attention are best understood "through each other".

### C3. SP's headline Qwen gain unblocks a skill the model already had. It doesn't add capability, and SP's own data shows this.
- **The gain comes from removing loops:**
  - Answers hitting the 32,768-token limit fell from 93% to 33%.
  - At 10 and 25 forward passes the loops "barely went away" (87.2% and 86.4%), and MATH-500 stayed at 12–14%.
  - The large jump appears only at 50 passes, so the feedback-driven search mattered a lot.
- **The non-looping control (Gemma) did not improve.** SP: "Editing weights could remove an obstacle that hid an ability, but it could not add an ability the model lacked."
- **The baseline was chosen to expose the flaw.** The FAQ concedes that turning thinking off raises the base score.
- **SP's own results contradict its forward-looking claims:**
  - "Zero-data capability steering… 'Can you add Capability X to the model?'" conflicts with Exp 8b.
  - "Auditing without evaluation… directly from the weights (e.g., backdoors)" conflicts with Exp 7a.

### C4. HR's framing critique holds. It applies to SP's abstract and conclusions, not to the experiments.
- **For HR:**
  - The abstract puts "just by looking at the raw weights" in the same breath as the Qwen result.
  - The related-work section claims a model that "reads and writes the weights themselves, from 171,664 weights given as text to a language model of 0.8 billion weights with tools", which blends the settings.
- **Against HR, partly:**
  - The TL;DR does say "sometimes with tools such as a Python kernel and sometimes from the raw weights alone".
  - Each experiment lists its setup, and the logs are public.
  - The FAQ entry "What does 'natively' mean, if Astra sometimes uses Python?" addresses exactly this point. These documents don't show whether that was a response to HR.
- **Verdict:** HR's core point survives. No single technique covers both "write weights from scratch" and "understand a large existing network". So SP doesn't show the unified, deep capability its framing suggests.

### C5. SP's "from scratch" results are an LLM writing a heuristic program and compiling it into weights. They don't show it reading weight space.
- **Exp 5:** Astra was given the game's source code. Its reasoning describes a landing-position predictor plus a paddle controller, which is a hand-written policy expressed in network form.
  - SP's related-work section notes that compilers from formal programs already write weights.
- **Exp 6 (Go):** a 336-line Python program encodes liberties, atari and eyes, which is human Go knowledge.
- **The baselines are weak:**
  - PPO-100M scores 15.8 on Breakout.
  - Astra loses to PPO-500M on Seaquest and Asterix.
  - SP's FAQ says the baselines only "put a price on" experience.

### C6. Both pro-legibility documents risk reading a coherent story into the model before confirming it. How much to trust each claim depends on the causal testing behind it.
- **AC knows the risk:**
  - It names "an explanation of the gaps".
  - It admits needing "a little faith" that something strange is a good pointer.
  - It revised its own canonical example.
  - It distrusts simulation scores.
- **AC's mitigations:**
  - Steering, ablations, and checking negative poles.
  - Near-miss and distant-concept controls.
  - Watching neurons grow across training checkpoints.
- **SP's testing is uneven:**
  - *Good:* Exp 3 checks the copying heads against measurement.
  - *Weak:* Exp 2 presents a *mistake* (drawing outlines instead of filled shapes) as proof that Astra really reads the weights. That is a post-hoc story.
  - Exp 1, 2, 3 and 7b use n = 3–4 with no chance baseline.
  - Astra's "reasoning summaries" are what the model wrote about itself, not evidence of mechanism.
- **Recommendation:** AC's standard (steer, ablate, check the negative pole, test near misses) is the right yardstick for SP's claimed readings.

### C7. None of the three has been independently confirmed yet, and key details are still pending.
- **AC:**
  - The language-model percentages come from "preliminary audits".
  - The Kimi K3 results and full methods are deferred to a forthcoming paper.
  - The headline claim was revised on Oct 30 2026.
- **SP:**
  - It is a blog post.
  - It says over a hundred replications and extensions exist, but names only a few, with no results.
- **HR:** a 3-minute blog post.
- **All three:** very recent (Oct–Nov 2026), and none is peer-reviewed.
- **Checkability:** SP's released logs make it the easiest to check. AC provides browsable interactive views but defers the data behind its quantitative claims.

### C8. What these documents do *not* support
- **That LLMs can read raw weights of large models without tools.** Tool-free raw-weight reading is shown only up to 172K parameters.
- **That weight edits can add capabilities.** Exp 8b failed, and SP says so itself.
- **That superposition is refuted in general:**
  - AC's step 4 is "highly speculative".
  - Attention is largely unexamined.
  - Residual-stream packing can't be fully ruled out.
  - The cleanest evidence comes from vision models.
- **That AC and SP confirm each other.** SP never tests whether neurons are clean, and AC never tests whether an LLM can read weights. They agree in outlook but don't supply evidence for each other.

---

## 3. Synthesis

Taken together, the documents support a modest, well-grounded shift: **neural networks are more legible to systematic, AI-assisted investigation than the field assumed in 2020–2023.**

- **AC supplies a reason.** Networks are built from localized, generalizing mechanisms, and much of the earlier mess came from tools and architectures.
- **SP supplies demonstrations** (Exp 3, 7b and 8) that a general-purpose LLM using standard interpretability techniques can find and act on such mechanisms.

The more dramatic versions don't survive scrutiny:
- **"LLMs natively read weights like code."** HR's analysis and SP's own failures in Exp 7a and 8b undercut it.
- **"Interpretability may be nearly solved."** AC's own hedges undercut it.

**The evidence that would settle the open questions:**
- AC's companion paper.
- The benchmark SP says it is building. To be informative, it should:
  - keep the raw-weight, Python, and forward-pass settings strictly separate;
  - report chance baselines;
  - use much larger samples than 3–4 networks per task.

---

## 4. Follow-up: testing SP's Exp 8 edit directly (Nov 2026)

The `qwen-weight-probe/` subproject took Qwen3.5-0.8B, rebuilt SP's released p50c edit, and ran causal checks on it. Full results are in `qwen-weight-probe/results/FINDINGS.md`. In brief:

- **The release reproduces exactly.** All 33 edited tensors rebuilt bit-for-bit from SP's logs.
- **The edit flips one direction ("u").** It is rank one: W′ = W − 6·u(uᵀW) on 64 neurons in each of layers 15 and 19. Each neuron's contribution along u is reversed and amplified 5×, not "subtracted" as SP describes it.
- **u is causally tied to looping.** On 64 practice problems with a 4K-token cap:
  - looping went from 13/64 (base) to 0/64 (edited);
  - pushing the edited model back along u brought it to 55/64;
  - a random push of the same size gave 1/64.
  - A fixed push along u, with no weight change at all, reproduced the edit's effect on looping.

How this bears on the conclusions above:
- **C1 supported.** A plain difference of means, run without knowing u, recovers it at cosine 0.74–0.76, against 0.025 for random directions. Ordinary methods find most of what Astra found.
- **C2 supported.** The weights alone hinted at the axis. Which neurons matter, which way the edit pushes, and what u does all needed the model to run.
- **C3 consistent but not tested.** Over 90% of answers hit the 4K cap, so accuracy couldn't tell unblocking a skill apart from adding one.
- **C6 met.** SP's mechanistic claim for Exp 8 survives the causal test (steering both ways, with a random control) that C6 calls for.

### How the experiment bears on each source document
These results cover only SP's Exp 8. They say nothing about SP's Exp 1–7, HR's points about the game-network experiments, or AC's claims that individual neurons are clean.

**SP: Sparks of Weight-Space Programming**
- **What holds up:**
  - The release is genuine: every edited tensor rebuilds bit-for-bit from the logs.
  - The mechanism is real: the edit acts on one direction, and that direction causally controls looping.
  - Independent runs (p50c, p10c) landed on the same axis.
- **What needs correcting or qualifying:**
  - **"Subtracted that direction" undersells the edit.** It reverses each neuron's contribution along u and amplifies it 5×.
  - **The 0.016% edit didn't come from reading weights.** The weights alone don't pick out the edited neurons (31/64 and 18/64 are among the top writers of u). Astra's own report says it compared activations on looping vs productive text.
  - **"Weight-Space Programming" adds little over known techniques here.** A fixed activation push along u, with no weight change, reproduces the edit's effect on looping. In this case it is a steering-vector intervention written into the weights.
  - **"Better at math" is still untested** (C3), because nearly every answer hit the 4K cap.

**HR: the honnibal.dev rebuttal**
- **Its core point gets stronger.** The weights alone suggest a plausible axis through the vocabulary, but which neurons matter and which way the edit pushes both needed the model running.
- **Ordinary methods suffice.** A plain difference of means recovers u at cosine 0.74–0.76, the routine ML engineering HR said this really was.
- **Its concession also holds.** The result is real, reproducible and non-trivial; the problem was SP's framing, not the experiment.

**AC: Neural Networks Are Already Clean**
- **Its testing standard proved decisive.** Steering both ways with a random control is what turned a plausible story into evidence.
- **Consistent with its broader optimism.** The mechanism is legible: one stable axis, recoverable by ordinary methods, with a readable vocabulary signature (" thus", " thereby" vs "...*", 而是因为 "but rather because").
- **But this is not the neuron-level evidence AC argues for.** The useful unit was a residual-stream direction written by many neurons together. The edited neurons and all the neurons in the layer show about the same effect size (d ≈ 1.4 vs 1.35). AC allows that "the directions neurons write to" are a good starting point, so there is no contradiction, but neuron cleanliness was not tested.

**Overall:** the experiment matches the analysis's best-supported conclusion, C1. SP found a real, causal, readable mechanism using ordinary interpretability methods, and the "LLM reads weights like code" framing isn't needed to explain it.

### Follow-up: the circuit around u (`qwen-weight-probe/results/08_circuits/looping.md`)
- **u is a "keep deliberating vs wrap up" axis.**
  - Pushing it cuts the probability of ending the reasoning about 4–5×, raises reconsidering tokens and lowers concluding ones.
  - It doesn't raise the probability of copying earlier text.
  - So SP's "loop direction" is better described as a *don't-conclude* direction, and loops follow from not concluding. That supports SP's mechanism while sharpening it.
- **The head SP's separate p25c run zeroed (L15.attn.h7) is a copy head.** Pooled over three seed sets, zeroing it cuts loops 28 → 5/192 (p = 3 × 10⁻⁵). But another copy head (L11.attn.h1) has the opposite effect when ablated (78/192), and three together change nothing reliably. So SP's two edits hit related parts of the same machinery, but only u is a clean lever in both directions.
- **For C2 and C6:** fixed-text (teacher-forced) intervention effects often failed to predict free-generation behavior. Causal claims about generation need generation-level tests, which SP's Exp 8 provides through its benchmarks but most interpretability shortcuts don't.

### Follow-up: the closing circuit (`qwen-weight-probe/results/08_circuits/closing.md`)
- **The decision to end reasoning has its own direction, d⊥, about 98% separate from u.** Studied in the p50c-edited model, since the unedited model almost never concludes.
  - Pushing d⊥ makes every answer close (192/192, against 88/192 unsteered), at about a third of the usual length. Pushing it the other way stops all closing (0/192), and 85% of answers loop instead. A random push of the same size does neither.
  - Pushing u toward wrap-up also raises closing (142/192), but less. So u, SP's "loop direction", is one input to a broader stop-reasoning decision, not the decision itself.
- **Accuracy follows closing under a token cap** (76 → 115 correct of 192), while the share of closed answers that are right falls (86% → 60%). This is more support for C3: much of the measured gain comes from letting the model finish, not from new ability.
- **One linear-attention head (L22.lin.h13) carries closing in free generation:** ablating it cuts closing 88 → 26/192. It was found with ordinary attribution and a teacher-forced screen, which again fits C1. Unlike the looping work, its fixed-text effect carried over to generation, so C2/C6's caution applies case by case rather than across the board. Combined tests at two strengths show the head and d⊥ add up rather than chain: d⊥ helps just as much with the head removed.
- **d⊥ is a decision to stop, not just a push on one token.** Removing its direct `</think>` logit component still closes every answer (only later), and with `</think>` banned it still makes the model commit to a final answer and loop on wrap-up text. Here a teacher-forced measure (most of the push on P(`</think>`) is direct) gave the wrong overall picture of the behavior, another case for C2/C6's caution about fixed-text tests.
- **The push works through the model's own processing.** It still closes almost every answer with its 20 most directly promoted words removed, or applied at layer 15 alone. Blinding layers 20–23 to the closing direction cuts the model's own closing by almost two-thirds (88 → 31/192) but leaves the push's added effect intact. Blinding layer 18 as well stops the push from closing, yet the model still talks about wrapping up and loops half as much, so deciding to stop and producing the closing token look like separate steps. The decision to stop is read in several places (two single heads, L22.lin.h13 and L11.attn.h1, each add to it in parallel), which fits C2's picture of behavior spread across components rather than sitting in one readable place.
- **Labels from one statistic can mislead.** The "copy heads" chosen by attention ratio (136–167× uniform) put only about 5% of their attention on the copy, and their output mostly works *against* the copied token. That's a small instance of C6: a coherent story ("copy heads drive loops") read in before the intervention checked it.

Limits: one model; 64 training-set problems per test; a 4K cap for the looping work and 8K for closing, instead of SP's 32K; our own loop metric. Phase 4 and the strength sweep used one seed set per condition; the circuit tests used three. A dose-response sweep over steering strengths is in `qwen-weight-probe/results/05_sweep.md`.
