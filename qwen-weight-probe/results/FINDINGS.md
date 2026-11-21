# Findings: Qwen3.5-0.8B and the Sparks Exp 8 weight edit

This project tests conclusions C1–C8 from `../../ANALYSIS.md` on the actual model and the actual released edit.
- **Setup:** the model is `Qwen/Qwen3.5-0.8B` at revision `2fc0636`, with the logs of `RikiyaT/lm_understand-logs`. It ran on an RTX 4070 inside nix-shell.
- **Sources:** each result names its file under `results/` (`01_survey` … `08_circuits`, `.md` and `.json`).

## Key terms
- **Residual stream:** the running list of 1,024 numbers that represents each token as it passes through the model. Every layer reads it and adds its own contribution.
- **Direction:** a particular pattern across those 1,024 numbers, scaled to length 1. "How much a state points along a direction" is a dot product, much like asking how far north something has moved.
- **u (the loop direction):** the single direction that Sparks' p50c edit acts on.
  - It was recovered by subtracting the original weights from the edited ones and taking the SVD of the difference; the whole change lies along one direction.
  - There is one per edited layer, **u15** and **u19**. They are nearly the same axis with opposite signs (cosine −0.85).
  - Saved as `results/u15_p50c.npy` and `results/u19_p50c.npy`.
  - Results 5–7 below show that it tracks and controls the model's looping.
- **u-write:** how much the 64 edited neurons in a layer push the residual stream along u at a given token (each neuron's activation × that neuron's weight along u, summed).
- **Repeated token / looping generation:**
  - A token is *repeated* if it completes a 16-token sequence that already appeared earlier in the same answer.
  - A generation is *looping* if over half of its last 1,024 tokens are repeated.
  - These are our own measures; Sparks counted answers that hit the token limit.
- **Steering:** adding a fixed vector to the residual stream after layers 15 and 19 at every position, while the model generates. No weights change.
- **Difference of means (DoM):** the average residual state on repeated tokens minus the average on non-repeated ones. A standard way to find a behavior's direction without knowing it in advance.
- **Cosine:** the similarity of two directions, from −1 to 1. Two random directions in 1,024 dimensions typically land near ±0.025.

## Results

1. **The release reproduces exactly.** `apply_edit.py` rebuilt all 33 edited tensors from the three runs (p50c, p25c, p10c) from the base model, and every sha256 hash matched. *(02_edit)*

2. **The headline edit is a rank-one sign flip with 5× gain, not a removal.**
   - In the p50c edit, ΔW on the 64 columns of each of `layers.{15,19}.mlp.down_proj` is exactly rank one: 100.00% of the change lies in one direction.
   - It fits W′ = W − 6·u(uᵀW) with α = 6.000 and a residual of 0.2–0.3%.
   - So each edited neuron's contribution c along u becomes −5c. The Sparks post calls this "subtracted that direction"; it is a reversal. *(02_edit)*

3. **Independent edits converge on one axis.**
   - u15 and u19 have cosine −0.85, so it's the same axis, written with opposite signs by the two groups of neurons.
   - The p10c run's direction has cosine 0.69 with u15, against a random baseline of 0.025.
   - p25c's zeroed head (15.7) is the layer-15 head most aligned with u, but only about 2× chance: weak evidence. *(02_edit)*

4. **The weights alone read the axis only partly.**
   - Through the vocabulary, one pole of u promotes words that wrap up or move forward (" thus", " thereby", " first", " yields", " accordingly", 即可). The other promotes markers of continuing or reconsidering ("...*", 而是因为, 他/她还是). The random controls show no such pattern.
   - But the weights don't single out the edited neurons: only 31/64 (L15) and 18/64 (L19) are among the top-64 writers of u.
   - They also can't tell which way the edit pushes, because these MLP activations can be negative. *(03_lens)*

5. **Running the model confirms that u tracks loops.** This is a forward pass over 254,662 tokens of the base model's own generations, labeling a token "repeated" if it closes a 16-gram already written earlier in the same generation.
   - **Effect size:** the 64 edited neurons write along u far more during repeated tokens. Cohen's d is −1.40 (L15) and +1.44 (L19), and ±2.5 within looping generations.
   - **Not a position artifact:** restricting to positions ≥1024 gives the same result (d −1.38 / +1.40).
   - **Timing:** the signal is flat before a loop begins and roughly triples over the next ~512 tokens. In this data it *tracks* loops; it doesn't predict them.
   - **Sign:** during loops, L15 neurons write −u15 and L19 neurons write +u19. Since u15 ≈ −u19, both push the residual stream the same way. The edit's −5c flip reverses that push and amplifies it. *(04_causal)*

6. **Ordinary methods recover the direction without Astra.**
   - The plain difference of means of the residual stream, repeated minus non-repeated tokens, has |cos| 0.74 (L15) and 0.76 (L19) with u. Position-matched it is 0.72 and 0.74.
   - It is stable: split-half cosine 0.97, against a random baseline of 0.025.
   - That fits Astra's own report that it found the direction from "matched productive/repetitive contexts". *(04_causal)*

7. **The direction controls looping in both directions, and a random push of the same size does nothing.** These are 64 problems at a 4K-token cap with identical seeds, compared with Fisher exact tests.

   | condition | looping | p vs reference |
   |---|---|---|
   | base | 13/64 (20%) | (reference) |
   | p50c (Sparks edit) | 0/64 | 0.00012 vs base |
   | base pushed off the loop side of u (no weight edit) | 0/64 | 0.00012 vs base |
   | p50c pushed back toward the loop side | **55/64 (86%)** | 8e-27 vs p50c |
   | p50c pushed along a random direction, same norm | 1/64 | 1.0 vs p50c |

   A constant push in the residual stream after layers 15 and 19 reproduces the edit's effect on looping, with no weight change. The reverse push brings back far more loops than the unedited model ever had. *(04_causal)*

8. **Dose-response sweep: the effect is graded and specific up to about 1× strength, then breaks down.** The same 64 problems were run at 0, 0.25, 0.5, 1, 2 and 4 × β. Looping generations out of 64:

   | strength | 0× | 0.25× | 0.5× | 1× | 2× | 4× |
   |---|---|---|---|---|---|---|
   | p50c pushed toward u's loop side | 0 | 1 | 9 | 55 | 62 | 62 |
   | base pushed away from it | 13 | 3 | 0 | 0 | 1 | 7* |
   | p50c pushed along a random direction | 0 | 1 | 1 | 1 | 15 | 64 |

   - **Graded:** looping rises steadily as the push toward u grows. The threshold is about 0.5×, where loops first appear (9/64, p = 0.003 vs 0×) and where pushing away removes them entirely (0/64, p = 0.0001).
   - **Specific:** random stays at 0–1/64 through 1×. Toward-u vs random: p = 0.017 at 0.5×, 4e-25 at 1×, 4e-19 at 2× (62 vs 15).
   - **Breaks down at large strengths:** at 4× any direction destroys the output. Random loops 64/64, so u (62/64) is no longer distinguishable (p = 0.5).
   - **Pushing too far away from u fails differently:** at 4× the model stops reasoning. Median length drops to 480 tokens, it writes garbled text, and it declares "Results verified." with 0/64 correct. *These 7 "loops" are short garbled repeats. This is the wrap-up pole that the vocabulary readout of u predicted.
   - **The useful range away from loops is about 0.5–1×.** There loops are gone, and closed thinking and correct answers peak (1×: 14/64 closed, 12/64 correct). *(05_sweep)*

9. **Wiring map (level 2, weights only).** Every component's static connections, published as an explorable page.
   - Edges beat chance for every type: 3× for neuron → neuron, up to 170× for head → head.
   - Wiring is mostly local and densest in layers 7–14.
   - Layer 15 holds 10 of u15's top 40 writers. *(06_wiring.md)*

10. **Looping circuit (level 4): u decides whether to conclude, and loops follow.** *(08_circuits/looping.md)*
    - **What u does:** pushing toward its loop side cuts P(`</think>`) about 4–5×. It raises reconsidering tokens (" Wait", 不是, 有没有) and lowers concluding ones (便可, " yielded"). It slightly *lowers* the copied token's probability, so u isn't a copy signal.
    - **How it acts:** entirely through later components (freezing everything downstream removes all of it), spread over the last layers. No single neuron carries more than 2%.
    - **Copy heads:** L11.attn.h1 and L15.attn.h7 attend to the earlier copy at 167× and 136× uniform. L15.attn.h7 is the head Sparks' p25c run zeroed. Some copy heads feed the loop signal.
    - **In free generation, individual copy heads have strong, reproducible but opposite effects.** Three seed sets, 192 answers each:
      - zeroing L15.attn.h7 (= Sparks p25c) cuts loops 28 → 5/192 (p = 3 × 10⁻⁵);
      - mean-ablating that head instead cuts them less (16/192);
      - mean-ablating L11.attn.h1 *raises* them to 78/192 (p = 1 × 10⁻⁸);
      - three copy heads together: no reliable change (36/192);
      - with u pushed, removing them lowers loops modestly (168 → 137/192, p = 1 × 10⁻⁴), so they contribute but aren't necessary.
    - **The "copy heads" mostly work against the copy.** Only about 5% of their attention goes to the earlier copy, and their direct output lowers the copied token's log-prob (L15.attn.h7 −0.11), mainly where they *don't* attend to it. That explains why ablating them raises the copy log-prob (looping stage 6).
    - **One seed set wasn't enough:** the base model ranged from 7 to 13 of 64 between seed sets.
    - **The edited neurons:** removing them does nothing. The edit works by reversing and amplifying them.
    - **Teacher-forced vs generation:** teacher-forced ablation effects on P(`</think>`) didn't carry over to free generation.
    - **Bottom line:** the robust causal lever is u itself. The pieces around it are distributed, and only partly predictable from fixed-text measurements.

11. **Closing circuit (level 4): the decision to end reasoning has its own direction, and one head carries it.** *(08_circuits/closing.md)*
    - **Subject:** the p50c-edited model, because the base model almost never concludes (2/64 at 8K). 64 held-out GSM8K problems, 8K cap, three seed sets (192 answers per condition).
    - **A separate closing direction:** d_close (the decision point minus mid-reasoning) is stable (split-half 0.92–0.95) and leans only slightly toward u's wrap-up side. With the u part removed, **d⊥** keeps 96–99% of its length. Its vocabulary readout promotes `</think>`, " finalize", " confident", "boxed".
    - **It jumps rather than builds up:** flat over the 480 tokens before closing, then a sharp rise in the last 32.
    - **d⊥ is an on/off switch in free generation:**

      | condition (same push strength) | closed | correct | looping |
      |---|---|---|---|
      | p50c (reference) | 88/192 | 76 | 11 |
      | push away from u's loop side | 142/192 | 104 | 2 |
      | push d⊥ toward closing | **192/192** | 115 | 1 |
      | push d⊥ away from closing | **0/192** | 0 | **163** |
      | random direction | 68/192 | 54 | 29 |
      | mean-ablate L22.lin.h13 | **26/192** | 23 | 14 |

      d⊥ vs random: p = 5 × 10⁻⁵¹ and 5 × 10⁻²⁴. Every condition moved the same way in all three seed sets.
    - **Closing sooner gets more answers in under the cap, but some close too early:** 60% of closed answers are correct with d⊥ pushed, vs 86% for the reference.
    - **Not concluding becomes looping:** with closing switched off, 85% of answers loop.
    - **L22.lin.h13:** a linear-attention head in layer 22, found by attribution and a teacher-forced ablation screen. Mean-ablating it cuts closing to 26/192 (p = 4 × 10⁻¹²) without touching looping or next-token loss. **It is the first single component here whose fixed-text effect carried over to free generation.**
    - **d⊥ mostly bypasses that head:** with L22.lin.h13 ablated, pushing d⊥ still closes 176/192 (vs 192). But 13 of those answers ended their reply inside the thinking block without ever writing `</think>`, which never happened otherwise. At a weaker push (0.25×), the head and d⊥ simply add up in log-odds: the push helps just as much with the head removed (+1.29 vs +0.96). So they're parallel inputs to closing, not a chain.
    - **d⊥ is a decision, not just a logit push on `</think>`** (stage 6). With its component along `</think>`'s direct-logit direction removed, the push still closes 192/192; it only closes later (median 2,209 vs 1,372 tokens, p = 4 × 10⁻⁷). With `</think>` banned outright, the push still makes the model commit to a boxed answer (177 vs 29/192) and then loop on wrap-up phrases ("I will output the result… Final Answer"). Teacher-forced tests had suggested most of the push on that one token is direct (79% survives freezing L16–23), but they can't see the push changing what the model writes next.
    - **It works through the model's processing, not its direct word effects** (stage 7). With the 20 words it pushes most directly removed (`</think>`, " finalize", " ready", "Final", "boxed"…), it still closes 185/192; pushed at layer 15 alone, 182/192. The late blocks (L20–23) need the closing direction for the model's *own* closing (blinding them to it: 88 → 31/192), but the push's added effect survives that blinding (+1.19 log-odds vs +0.95), so it most likely passes through L16–19 in another form.
    - **L11.attn.h1 is another parallel input:** ablating it lowers closing 88 → 57/192 and raises looping 11 → 30, and the push adds just as much without it. This is the head whose ablation raised looping in the looping work.
    - **Caveats:**
      - Blinding all of L16–23 stops closing (3/192) and removes the push (2/192), so the direct path alone isn't enough. The route looks like two steps (stage 7b). L18 and L20–23 back each other up for producing `</think>`: with either blind the push still works (+0.76 and +1.20 log-odds, +0.96 intact), and with both blind it can't close (2 vs 5/192). Even then it halves looping (58 → 29/192) and triples answers full of wrap-up talk (12 → 40/192, an exploratory measure), so earlier blocks in L16–19 still read it.
      - The random push isn't neutral at this strength (it lowered closing slightly, p = 0.048).
      - The other screened components were tested on fixed text only.

12. **Structural map (level 1, weights only).** All 320 tensors. *(01_structure.md)*
    - **One special residual dimension:** dim 0 is written about 2.6× harder than typical by 43 of 48 output matrices, then shrunk by every norm, down to 0.05× in the final norm. It looks like the "massive activation" dimension seen in other LLMs (unconfirmed without activations).
    - **Layer 23 stands out:** its attention value matrix concentrates 13% of its energy in one direction (11× the other full-attention layers), and its MLP input is similarly concentrated. That direction is unrelated to u or the closing direction.
    - **No single neuron dominates any matrix** (largest column or row ratio 7.2×), consistent with the earlier finding that weights alone don't single out the edited neurons.

## What this means for ANALYSIS.md
- **C1 (agents running ordinary methods), supported.** Astra's direction is real and causal, but a 10-line difference-of-means finds 0.74–0.76 of it. The success lies in choosing the right experiment, not in some special ability to read weights.
- **C2 (raw weights only go so far), supported.** The weights reveal a plausible axis. Which neurons matter, which sign the push has, and what the axis does all came from running the model.
- **C3 (unblocking, not adding a skill), consistent, but not tested.**
  - The edit and the steering both remove repetition.
  - At a 4K cap over 90% of answers are cut off in every condition, so accuracy (6–19%) can't separate "unblocks" from "adds ability".
  - The base_anti 19% vs base 6% hint comes from n = 64 at one push strength. The push also makes the model close its thinking more often (22% vs 6%), which by itself raises scores under a cap.
- **C6 (causal tests are the standard), met here.** Steering both ways with a random control is exactly the test that SP's narrative mostly lacks. In this case, the claim survives it.

## Limits
- Results 1–8: one model, 64 training-set practice problems, one seed set and a 4K cap. Results 10–11 use three seed sets; result 11 uses an 8K cap and the p50c model as subject. Sparks used 32K tokens and test benchmarks. Our base loop rate (20%) is far below Sparks' 85/96 cut at 32K, because loops have less time to form.
- "Looping" is our own 16-gram metric, not Sparks' token-limit measure.
- The dose-response sweep (result 8) used six strengths, but one seed set, and it scaled both layers together by the same factor.
- The steering is a constant vector added at every position, which is cruder than the edit's activation-dependent change.
