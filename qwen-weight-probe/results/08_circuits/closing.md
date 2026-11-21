# Level 4: the `</think>` (concluding) circuit in Qwen3.5-0.8B

**Question:** how does the model decide to end its reasoning?
- Is that decision carried by u alone (the looping axis), or by a separate direction?
- Which components write the closing token?

**Setup:**
- Code: `src/probe/closing.py`, stages `pilot → corpus → direction → attribution → screen → confirm <cond> <seed> → pool → mechanism → followup`. Data: `08_circuits/closing/`.
- **Subject: the p50c-edited model, not the base model.**
  - The base model almost never concludes in thinking mode. On the stage-0 pilot at 8K it closed 2/64 and looped 49/64 (`stage0_pilot_base.json`).
  - p50c differs from the base model only in Sparks' 128 edited neurons, and closes about 45% of the time. So the user chose to map closing where it actually happens.
- **Problems:** GSM8K train, disjoint from the 32 used in Phases 4–5 and in looping.
  - Set A (128) is for fitting: direction, attribution, screening.
  - Set B (64) is only for free-generation tests, so no direction is tested on the data it was fitted to.
- **Generation:** Sparks think35 sampling, 8K-token cap, batch 16.
- **Free-generation tests:** three seed sets per condition (192 answers), pooled Fisher exact tests.

## Summary
1. **Closing has its own direction, mostly separate from u.**
   - d_close is the difference of means between the decision point (the 8 tokens before `</think>`) and mid-reasoning positions. It is stable: split-half cosine 0.92–0.95 from layer 2 on.
   - It points partly toward u's wrap-up side (cosine +0.13 to +0.30), but removing that part leaves 99% (L15) and 96% (L19) of its length. The remainder is **d⊥**.
   - Through the vocabulary, d_close at L19 promotes `</think>`, " response", " confident", " finalize", "boxed", 的答案 (the answer).
   - **It jumps at the decision rather than building up.** The projection on d_close stays flat at 0.15–0.36 over the 480 tokens before closing, then reaches 1.06 in the last 32.
2. **d⊥ switches closing fully on or off in free generation.** The push is the same size as the u-steering in Phase 4:
   - toward closing: 192/192 answers close (reference 88/192), at a median 1,372 tokens instead of 3,940;
   - away from closing: 0/192 close, and 163/192 loop instead (reference 11/192);
   - a random direction of the same size: 68/192.
   - Every seed set agrees.
3. **The u axis also controls closing, but less.** Pushing away from u's loop side (the positive control) raises closing to 142/192 (p = 3 × 10⁻⁸). d⊥ is stronger: 192 vs 142/192 (p = 5 × 10⁻¹⁷).
4. **More closing means more correct answers, with a cost per answer.** Correct answers rise from 76/192 to 115/192 with d⊥ pushed (p = 1 × 10⁻⁴), because more answers finish before the cap. But the share of closed answers that are correct falls from 86% to 60%, so some closings are premature.
5. **The model's own `</think>` comes from the last three layers.**
   - Direct-path attribution puts the largest contributions in L23.mlp, L22.attn and L23.attn, then the L18–19 MLPs and L21.attn.
   - It is spread out: the top 20 components carry 8% of the decision-minus-mid difference.
6. **One head carries closing in free generation: L22.lin.h13** (linear attention, layer 22).
   - Teacher-forced, mean-ablating it lowers log P(`</think>`) at the decision point by 2.4, against a random sd of 0.06, with no cost to next-token loss on ordinary text.
   - In free generation it cuts closing from 88 to 26/192 (p = 4 × 10⁻¹²), in every seed set, without changing looping (14 vs 11/192). Correct answers fall from 76 to 23.
   - **This is the first component in this project whose fixed-text effect carried over to free generation.** In the looping circuit, none did.
7. **Not concluding turns into looping.** With closing suppressed by −d⊥, 85% of answers loop. That matches the looping circuit's reading that loops follow from not concluding.
8. **d⊥ mostly doesn't need L22.lin.h13** (stage 5b, full strength).
   - Pushing d⊥ with the head ablated still closes 176/192, against 26/192 for the ablation alone. So d⊥ works mainly through other routes.
   - The head still matters a little: 176 vs 192/192 without the ablation (p = 2 × 10⁻⁵), in every seed set.
   - The missing 16 are telling: 13 answers ended their whole reply *inside* the thinking block, never writing `</think>`, which never happened with d⊥ alone (0/192). Their last lines read "I will output the response… \boxed{900}".
   - So without the head, the model still decides to stop, but sometimes fails to emit the specific token that closes the reasoning.
9. **At a weaker push, the head and d⊥ add up rather than chain** (stage 5c, 0.25× strength, where d⊥ alone doesn't saturate).
   - d⊥ at 0.25× raises closing from 88 to 132/192. With the head ablated it raises closing from 26 to 70/192.
   - In log-odds the push adds +0.96 with the head and +1.29 without it. **Removing the head doesn't weaken d⊥'s effect**, so d⊥ doesn't act through L22.lin.h13.
   - The head is a parallel contributor: removing it costs about the same (−1.35 to −1.69 log-odds) whether or not d⊥ is pushed.
   - Unclosed endings are rare at this strength (3/192), so the 13 seen at 1× look like a strong-push effect.
10. **d⊥ moves a "decide to stop" state, not only the `</think>` token** (stage 6).
   - **The direct component isn't needed.** Removing from d⊥ its component along `</think>`'s direct-logit direction still closes 192/192 (p = 3 × 10⁻⁴⁰ vs reference; correct 119 vs 115 with the full push). It only closes later: median 2,209 vs 1,372 tokens (Mann-Whitney p = 4 × 10⁻⁷), still well before the reference's 3,940.
   - **With `</think>` banned, the push still drives wrap-up.** It writes a real boxed answer in 177/192 answers vs 29/192 unpushed (first one at a median 2,563 vs 4,061 tokens; 107 of them correct), then loops on end-of-thought phrases ("I will output the result… Final Answer…"): 179/192 loop vs 19/192, with 7,637 vs 311 wrap-up phrases in the last 3k characters.
   - So the direct `</think>` logit sets how fast the token comes once the model is done; whether it decides it's done is carried by the rest of d⊥.
11. **The push works through the model's processing, not through direct word logits, and it doesn't require the late blocks to read it** (stage 7).
   - **Not through wrap-up words.** Removing the 20 tokens d⊥ pushes most directly (`</think>`, " finalize", " ready", "Final", "boxed", "完毕"…; 8% of its length) still closes 185/192 (correct 117). It closes more slowly (121 vs 178 closed within 4K tokens).
   - **It works from layer 15 alone,** four layers before its direct path matters: 182/192 closed, 122 correct.
   - **The late blocks need the closing direction for the model's *own* closing, but not for the push.** With layers 20–23 unable to read the d⊥ span, natural closing falls from 88 to 31/192 (p = 4 × 10⁻¹⁰) and looping rises from 11 to 43. Yet the 0.25× push still adds as much (+1.19 log-odds, vs +0.95 intact): 31 → 75/192.
   - So d⊥ has to be read by some block from L16 on (blinding all of 16–23 removes the push, stage 7b). L18 and L20–23 back each other up for producing `</think>`: either alone can be blinded, but with both blind the push can't close, though it still drives wrap-up talk and halves looping.
12. **L11.attn.h1 is a third parallel input to closing** (stage 7). Mean-ablating it lowers closing 88 → 57/192 (p = 0.002) and raises looping 11 → 30 (p = 0.003), matching the looping work. The 0.25× push adds +1.37 log-odds with it removed (vs +0.95 intact), so, like L22.lin.h13, it isn't downstream of d⊥. It isn't a direct writer of d_close (rank 23 of 48 full-attention heads); its ablation lowers the d_close signal at the decision by only ~5%, and log P(`</think>`) there by 0.33.

## Circuit as established
Solid lines are backed by a free-generation test with a control. Dashed lines are backed by teacher-forced measurements only.

```mermaid
flowchart LR
  U(("u<br/>keep deliberating ↔ wrap up"))
  D(("d⊥ (closing)<br/>98% separate from u"))
  W["writers of d_close<br/>L16.lin.h2, L15.attn.h1, L18.lin.h1, L19 MLPs<br/>(direct-path, small and spread out)"]
  H["L22.lin.h13"]
  LATE["late blocks L20–23<br/>(L23.mlp, L22.attn, L23.attn,<br/>L23.mlp.932, L23.mlp.385)"]
  MID["blocks L16, L17, L19<br/>(not split by layer)"]
  WRAP["wrap-up state<br/>(final-answer talk, less looping)"]
  CLOSER["L18 or L20–23<br/>(either is enough)"]
  H2["L11.attn.h1"]
  OUT["`&lt;/think&gt;`<br/>(end reasoning)"]
  LOOP["looping"]
  W -. "direct-path attribution" .-> D
  D -- "push ±: closing 192/192 or 0/192 (ref 88)" --> OUT
  U -- "push to wrap-up side: 142/192" --> OUT
  H -- "mean-ablate: 88 → 26/192; adds to d⊥, not downstream of it" --> OUT
  LATE -- "blinded to d⊥: own closing 88 → 31/192" --> OUT
  D -- "push from L15 alone: 182/192; push's effect survives late blinding" --> MID
  MID -- "L18 + L20–23 blind: push still halves looping 58 → 29; L16–23 blind: no effect" --> WRAP
  WRAP -- "both blind: push can't close (2 vs 5/192)" --> CLOSER
  CLOSER --> OUT
  H2 -- "mean-ablate: 88 → 57/192; adds to d⊥" --> OUT
  D -- "push away from closing: looping 11 → 163/192" --> LOOP
```

d⊥ and L22.lin.h13 are parallel inputs to closing, not a chain: their effects add in log-odds (stages 5b and 5c). d⊥ is not just a lever on the `</think>` logit: without its direct component it still closes every answer, and with the token banned it still produces final answers and wrap-up text (stage 6). L11.attn.h1 is a third parallel input (stage 7). The late blocks (L20–23) need to read the closing direction for the model's own closing, but the push's extra effect doesn't depend on them reading it: it works from L15 alone and survives blinding L20–23, so it most likely passes through L16–19 in another form. Which components in L16–19 is still open.

## Stage results

### Stage 0: pilot (set A[:64], 8K)
| subject | closed | median close | correct | looping | hit cap |
|---|---|---|---|---|---|
| base model | 2/64 | | | 49/64 | |
| **p50c** | **29/64** | 4,465 | 24/64 (all closed) | 5/64 | 39/64 |

The gate was ≥ 20% closing. The base model failed it, so the subject became p50c (a user decision).

### Stage 1: corpus
Set A[64:] at seed 0 closed 26/64, giving 55 closed answers across set A with their closing positions marked.

### Stage 2: direction (teacher-forced, set A)
- Positions: 440 at the decision point (the 8 tokens before each of 55 closings), 5,500 mid-reasoning (more than 256 tokens before closing), and 7,300 from answers that never close.
- **Stability:** split-half cosine 0.85 at layer 0 and 0.90 at layer 1, then 0.92–0.95 from layer 2 on. The plan's threshold was > 0.9.
- Using only mid-reasoning positions from closed answers as the comparison gives the same direction (cosine 0.97–0.99).
- **Relation to u:** cos(d_close, −u) is +0.15 at L15 and +0.26–0.30 at L19. Removing the u part keeps 0.989 (L15) and 0.955 (L19) of the norm.
- **Lens:**
  - L19 promotes `</think>`, `<|endoftext|>`, " response", " confident", " step", 作答 (answer), " finalize", "boxed", 的答案, " ready", " output".
  - L15 promotes `<|endoftext|>`, 谢谢大家 (thanks everyone), 最终的 (final), 完毕 (done), " finalize", " final", " confident", "qed", 可以放心 (can rest assured).
  - Both suppress quotation marks and hedges (" somehow", " magically").
- **Onset (L19, 16 bins of 32 tokens):** 0.21, 0.16, 0.17, 0.29, 0.28, 0.15, 0.28, 0.31, 0.21, 0.33, 0.36, 0.27, 0.24, 0.31, 0.25, **1.06**. No ramp: the state changes at the decision itself.

### Stage 3: attribution of log P(`</think>`) (teacher-forced)
- Direct path, final-norm RMS scale frozen. **Completeness:** maximum relative error 1.2 × 10⁻³, just over the 1 × 10⁻³ target. It is small next to the effects reported, but it misses the target.
- **By block** (decision minus mid, in log-prob units): L23.mlp 3.65, L22.attn 3.34, L23.attn 3.18, L19.mlp 1.75, L18.mlp 1.69, L21.attn 1.63, L20.mlp 1.06, L17.attn 1.03. L18.attn is the only clear negative (−0.64).
- **Top single components:** L22.mlp.796 (−1.83), L18.lin.h0 (−1.82), **L22.lin.h13 (+1.56)**, L23.mlp.932 (+1.39), L23.attn.h6 (+1.19), L23.mlp.385 (+1.13).
- The top 20 components carry 8% of the total.
- The strongest direct writers of d_close are small (≤ 0.13): L16.lin.h2, L11.attn.h6, L15.attn.h1, L18.lin.h1, L19.attn.h7, L19.mlp.1, L19.mlp.31.

### Stage 4: ablation screen (teacher-forced, 30 closed answers)
Baseline log P(`</think>`) at the decision point is −1.61. Random single heads and neurons give d log P with sd 0.06 (zero) / 0.065 (mean), and d NLL sd 0.0035.

| component | why screened | d log P, zero | d log P, mean | d NLL, mean |
|---|---|---|---|---|
| **L22.lin.h13** | top + writer | −1.94 | **−2.43** | −0.0008 |
| L23.mlp.932 | top + writer | −1.86 | −1.78 | +0.0038 |
| L23.mlp.385 | top + writer | −1.23 | −1.12 | +0.0027 |
| L19.mlp.31 | top + writer | −1.00 | −0.99 | +0.0004 |
| L18.lin.h1 | d_close writer | −0.62 | −0.61 | +0.0040 |
| L15.attn.h3 | d_close writer | −0.51 | −0.53 | +0.045 |
| L18.lin.h0 | top − writer | +0.46 | +0.40 | +0.0051 |
| L11.attn.h1 | lowered closing in looping | −0.41 | −0.29 | +0.012 |

L22.lin.h13 (mean) was chosen for stage 5: the largest effect, and no collateral damage to next-token loss.

### Stage 5: free generation (set B, 3 seed sets, 192 answers per condition; `stage5_pooled.json`)
All steering is applied after layers 15 and 19 at Phase 4's β, so every pushed condition has the same strength.

| condition | closed per seed set | closed (pooled) | p | median close (tokens) | correct | looping |
|---|---|---|---|---|---|---|
| p50c (reference) | 28 / 29 / 31 | 88/192 | | 3,940 | 76 | 11 |
| push away from u's loop side (positive control) | 53 / 46 / 43 | **142/192** | 3 × 10⁻⁸ | 2,996 | 104 | 2 |
| push d⊥ toward closing | 64 / 64 / 64 | **192/192** | 3 × 10⁻⁴⁰ | 1,372 | 115 | 1 |
| push d⊥ away from closing | 0 / 0 / 0 | **0/192** | 1 × 10⁻³² | | 0 | **163** |
| random direction, same strength | 25 / 23 / 20 | 68/192 | 0.048 | 5,436 | 54 | 29 |
| mean-ablate L22.lin.h13 | 11 / 8 / 7 | **26/192** | 4 × 10⁻¹² | 4,672 | 23 | 14 |

- Every condition moved closing the same way in all three seed sets.
- **Contrasts:**
  - d⊥ vs random: p = 5 × 10⁻⁵¹ (toward closing) and 5 × 10⁻²⁴ (away).
  - d⊥ vs the u push: p = 5 × 10⁻¹⁷.
  - L22.lin.h13 vs random: p = 8 × 10⁻⁷.
- **The random control isn't neutral.** It lowered closing slightly (68 vs 88, p = 0.048) and raised looping (29 vs 11, p = 0.004). A push of this size along any direction seems to disturb the model a little, which makes the d⊥ and u effects, measured against it, larger rather than smaller.
- Positive-control check passed: the push away from u's loop side raised closing, as in the strength sweep.

### Stage 5b: push d⊥ with L22.lin.h13 ablated (set B, 3 seed sets; `stage5b.log`)
Tests whether d⊥ works through the head.

| condition | closed per seed set | closed (pooled) | ended without `</think>` | median close (IQR) | correct | looping |
|---|---|---|---|---|---|---|
| push d⊥ toward closing | 64 / 64 / 64 | 192/192 | 0 | 1,372 (875–2,307) | 115 | 1 |
| push d⊥ + mean-ablate L22.lin.h13 | 59 / 58 / 59 | **176/192** | **13** | 1,952 (510–3,500) | 109 | 7 |
| mean-ablate L22.lin.h13 alone | 11 / 8 / 7 | 26/192 | 2 | 4,672 | 23 | 14 |

- vs d⊥ alone: p = 2 × 10⁻⁵ for closing. Closing position doesn't differ reliably (Mann–Whitney p = 0.31); its spread widens both ways.
- vs the ablation alone: p = 2 × 10⁻⁵⁹. The push overrides most of the head's loss.
- The 13 answers that ended without `</think>` wrote a final answer (most with `\boxed{}`) and an end-of-text token while still in the thinking block, after lines like "I will output the response." The decision to stop was made; the closing token wasn't produced.
- At 1× the push alone saturates (192/192), so a smaller dependence on the head could be hidden. Stage 5c repeats the test at 0.25×.

### Stage 5c: the same test at 0.25× strength (set B, 3 seed sets; `stage5c.log`)

| condition | closed per seed set | closed (pooled) | log-odds | ended without `</think>` | median close | correct | looping |
|---|---|---|---|---|---|---|---|
| p50c (reference) | 28 / 29 / 31 | 88/192 | −0.17 | 0 | 3,940 | 76 | 11 |
| mean-ablate L22.lin.h13 | 11 / 8 / 7 | 26/192 | −1.85 | 2 | 4,672 | 23 | 14 |
| push d⊥ at 0.25× | 44 / 43 / 45 | **132/192** | +0.79 | 0 | 3,320 | 111 | 4 |
| push d⊥ at 0.25× + mean-ablate L22.lin.h13 | 28 / 26 / 16 | **70/192** | −0.56 | 3 | 4,428 | 64 | 11 |

- Push effect: +0.96 log-odds with the head, +1.29 without it. Head effect: −1.69 without the push, −1.35 with it.
- If d⊥ worked through the head, ablating the head would shrink the push's effect. It doesn't, so the two act in parallel.
- Tests: the 0.25× push vs reference p = 8 × 10⁻⁶; ablation with vs without the push p = 3 × 10⁻⁷ (26 vs 70) and 3 × 10⁻¹⁰ (132 vs 70). The combination vs the reference: p = 0.08.
- The combined condition varies more between seed sets (16–28) than the others.

### Stage 6: is d⊥ a decision or just a logit push? (`stage6_mechanism.json`, `stage6_ban_text.json`, `stage6.log`)
`</think>`'s direct-logit direction is w = (1 + g_final) ⊙ E[`</think>`], the residual direction that raises its logit through the final norm and unembedding alone.

**Teacher-forced (20 closed set-A answers, 40 mid positions each, plus the decision position):**
- d⊥19 lines up with w more than with any other token's direction: cosine 0.19, rank 1 of 248k (99th percentile |cos| 0.08). d⊥15: 0.12, rank 143.
- The push raises log P(`</think>`) by 3.46 at mid positions and 1.21 at the decision. With w's component removed, by 0.95 and 0.64: about 27% and 53% left.
- Freezing every block output in L16–23 at its unpushed value still leaves 79% (mid) / 72% (decision) of the gain; single-block freezes leave 90–121%. By this measure most of the push on that one token is direct.
- But at mid positions the pushed probability is still about e⁻²¹, far too small to close anything by itself. The fixed-text test only scores the next token on unpushed text; it can't see the push changing what the model writes next.

**Free generation (set B, 3 seed sets, 192 answers per condition):**

| condition | closed | median close | correct | looping | real `\boxed{}` answer in thinking | wrap-up phrases (last 3k chars) |
|---|---|---|---|---|---|---|
| p50c (reference) | 88/192 | 3,940 | 76 | 11 | – | – |
| push d⊥ | 192/192 | 1,372 | 115 | 1 | – | – |
| push d⊥ minus its w component | **192/192** (64/64/64) | **2,209** | 119 | 0 | – | – |
| ban `</think>` | 0/192 | – | 0* | 19 | 29 (first at 4,061) | 311 |
| push d⊥ + ban `</think>` | 0/192 | – | 0* | **179** | **177** (first at 2,563; 107 correct) | **7,637** |

\* The grader reads only the text after `</think>`, so banned runs score 0 by construction; the boxed counts are graded separately. Only non-empty `\boxed{…}` count: many mentions are empty format reminders.

- No-direct vs full push: closing p = 1.0, correct p = 0.75, close position p = 4 × 10⁻⁷ (later). No-direct vs reference close position: p = 9 × 10⁻¹⁰ (earlier).
- Under the ban, 10/192 pushed answers ended the whole reply inside the thinking block (0 unpushed); the rest hit the 8K cap, mostly looping on "(Final check) The response. I will output the result. Final Answer. (End of thoughts)".
- Reading: the push makes the model finish (commit to an answer, write wrap-up text), and the direct component mostly speeds up the closing token once it does. The wrap-up wording could partly be a direct effect on words like "Final" and " response", which d⊥'s lens also promotes; the no-direct result doesn't depend on that.

### Stage 7: through what? (`stage7_followup.json`, `stage7.log`, set B, 3 seed sets)
**Teacher-forced (20 closed set-A answers):**
- Tokens whose direct-logit direction d⊥ lines up with most. L19: `</think>` 0.19, " siap" (Indonesian "ready"), " finalize", "期待您的", " confident", " ready", "Ready", " response", "boxed"… L15: the same family without `</think>` at the top ("谢谢大家", " finalize", "最终的", "完毕", "qed", "Final", " concise"). Removing the top-20 span keeps 94% (L15) / 92% (L19) of d⊥'s length.
- Share of the full push's gain in log P(`</think>`) kept: without the top-20 span 37% (mid) / 55% (decision); L15 only 45% / 69%; L19 only 50% / 52%.
- Heads writing d_close more at the decision than mid: L15.attn.h1 (+0.11), L7.attn.h2, L11.attn.h4, L7.attn.h1, L11.attn.h7. L11.attn.h1 ranks 23 of 48 (+0.02).
- Mean-ablating L11.attn.h1: log P(`</think>`) −0.33 at the decision, +0.23 at mid positions; the d_close projection at the decision drops 0.07 (L15) and 0.09 (L19), about 5% of the decision-vs-mid gap.

**Free generation (192 answers per condition):**

| condition | closed per seed set | closed | p vs ref | log-odds | correct | looping | closed within 1K / 2K / 4K / 8K |
|---|---|---|---|---|---|---|---|
| p50c (reference) | 28 / 29 / 31 | 88 | | −0.17 | 76 | 11 | 5 / 19 / 47 / 88 |
| push d⊥ (1×) | 64 / 64 / 64 | 192 | 3e-40 | | 115 | 1 | 64 / 130 / 178 / 192 |
| push d⊥ minus top-20 word span | 62 / 62 / 61 | **185** | 8e-31 | | 117 | 4 | 19 / 55 / 121 / 185 |
| push d⊥ at L15 only | 60 / 59 / 63 | **182** | 5e-28 | | 122 | 0 | 35 / 67 / 120 / 182 |
| push d⊥ 0.25× | 44 / 43 / 45 | 132 | 8e-6 | +0.78 | 111 | 4 | |
| layers 20–23 blind to d⊥ span | 13 / 9 / 9 | **31** | 4e-10 | −1.63 | 25 | 43 | |
| push 0.25× + layers 20–23 blind | 27 / 26 / 22 | 75 | 0.22 | −0.44 | 66 | 19 | |
| mean-ablate L11.attn.h1 | 22 / 19 / 16 | **57** | 0.002 | −0.86 | 41 | 30 | 1 / 4 / 14 / 57 |
| push 0.25× + mean-ablate L11.attn.h1 | 47 / 36 / 37 | 120 | 0.002 | +0.51 | 87 | 9 | |

- **Blinding** projects the span of d⊥15 and d⊥19 out of the input of every attention and MLP block in layers 20–23 (at their pre-norms). The residual stream itself is untouched, so the direct path to the output still sees the push. Push effect: +0.95 intact, +1.19 blinded, +1.37 with L11.attn.h1 ablated.
- Tests: no-wrap vs full push p = 0.015; L15-only vs full p = 0.002; push + blind vs blind p = 7 × 10⁻⁷; push + L11 ablation vs ablation p = 2 × 10⁻¹⁰, vs push alone p = 0.24.
- **Whole-block ablation was tried first and failed.** Mean-ablating L22.attn, L23.attn and L23.mlp together breaks the model: 0/192 closed, 185/192 looping, text degrading into "** ** **". That's why the blinding test replaced it.
- **A run error, corrected.** The first push + ablation runs used `_x025`, which parsed as 25×, not 0.25×. At 25× the model emits d⊥'s word list as salad from the first token ("finalize confident Final boxed Ready `</think>`…"). Those runs are kept in `closing/invalid_25x_push/` and were redone at 0.25×. The parser now requires a decimal point.

### Stage 7b: where the push is read (`stage7b.log`, set B)
The push is added at the outputs of L15 and L19, so blinding only L16–19 would still let L20–23 read the L19 push. Blinding L16–23 leaves the push only the direct path to the output.

| condition | closed per seed set | closed | looping |
|---|---|---|---|
| layers 16–23 blind | 3 / 0 / 0 | **3** / 192 (p = 4 × 10⁻²⁸ vs ref) | 69 |
| push 0.25× + layers 16–23 blind | 0 / 1 / 1 | **2** / 192 | 57 |

- **The direct path alone does nothing.** Had the push kept its +0.95 log-odds, about 8 of 192 would close (P(≤ 2) = 0.02). The text stays coherent: the model reasons normally and never stops.
- **Teacher-forced numbers can't localize it.** Freezing any single block in 16–23 keeps 86–110% of the push's one-step gain in log P(`</think>`), and freezing all of 16–23 keeps 72–79% (stage 6). The one-step gain is mostly direct; closing in free generation is not.

**Single-layer screen** (push 0.25× + layers 20–23 blind + one more layer blind, seed set 0 only, 64 answers each; reference: push 0.25× + 20–23 blind, 27/64 at seed 0):

| extra blind layer | closed | p vs 27/64 | p vs 13/64 (20–23 blind, no push) | looping |
|---|---|---|---|---|
| L16 | 5 | 1 × 10⁻⁵ | 0.07 | 11 |
| L17 | 15 | 0.04 | 0.83 | 9 |
| **L18** | **1** | 8 × 10⁻⁹ | 0.001 | 9 |
| L19 | 22 | 0.47 | 0.11 | 6 |

- **Read alone, this screen overstates L18.** It has no matching no-push run, and the 3-seed test below shows L18 isn't required for the push on its own.

**Layer 18 alone (3 seed sets, 192 answers per condition):**

| condition | closed per seed set | closed | correct | looping | log-odds |
|---|---|---|---|---|---|
| p50c (reference) | 28 / 29 / 31 | 88 | 76 | 11 | −0.17 |
| push 0.25× | 44 / 43 / 45 | 132 | 111 | 4 | +0.79 |
| L18 blind | 18 / 13 / 17 | **48** (p = 3 × 10⁻⁵) | 31 | 20 | −1.10 |
| push 0.25× + L18 blind | 21 / 30 / 29 | **80** (p = 8 × 10⁻⁴ vs L18 blind) | 58 | 9 | −0.34 |

- **L18 helps the model's own closing.** Blinding it costs −0.93 log-odds.
- **L18 is not the push's only route.** The push still adds +0.76 log-odds with L18 blind, against +0.96 intact; the difference is within noise (z = −0.6).
- **L18 and L20–23 back each other up for closing.** Each can be blinded alone and the push still works; blinded together, it can't close (next table).

**L18 + layers 20–23 blind together (3 seed sets, 192 answers per condition):**

| condition | closed per seed set | closed | looping | unclosed answers with ≥ 5 wrap-up phrases |
|---|---|---|---|---|
| L18 + 20–23 blind | 0 / 3 / 2 | **5** | 58 | 12 |
| push 0.25× + L18 + 20–23 blind | 1 / 0 / 1 | **2** | 29 (p = 6 × 10⁻⁴) | **40** (p = 4 × 10⁻⁵) |
| L16–23 blind | 3 / 0 / 0 | 3 | 69 | 1 |
| push 0.25× + L16–23 blind | 0 / 1 / 1 | 2 | 57 (p = 0.23) | 2 (p = 1) |

- **The push no longer closes.** Its effect on closing is −0.80 log-odds (95% interval −2.31 to +0.70). An exact test on the 7 closings rejects the effects seen with either part blinded alone: +0.76 (p = 0.04), +0.96 (p = 0.02), +1.20 (p = 0.009). These margins are modest because so few answers close.
- **The model's own closing also falls more than the separate effects predict:** 5/192 against 13.5 expected if they simply added (p = 0.006).
- **But the push is still being read.** With L18 + L20–23 blind it halves looping and more than triples the answers full of wrap-up talk ("Ready. Final answer is 3. I will write the response now"), yet those answers don't close. With L16–23 blind, neither change appears.
- **Reading: two steps.** Blocks in L16, L17 or L19 read the push and move the model into a wrap-up state. L18 or L20–23 is then needed to turn that state into `</think>`. The first step isn't split by layer yet.
- **The wrap-up measure is exploratory.** The phrase list (stage 6's, plus "Ready") and the ≥ 5 threshold were set after reading these answers, and only unclosed answers are counted. Looping is the pre-existing metric and points the same way.

## Relation to the looping circuit
- **Same late machinery.** Both u's effect on `</think>` (looping stage 3) and the model's own closing (stage 3 here) run mostly through L23.mlp and the L21–23 attention blocks.
- **Two levers, one decision.** u and d⊥ are nearly orthogonal (d⊥ keeps 96–99% of d_close's length after the u part is removed), yet both move closing. Pushing u toward wrap-up closes 142/192; pushing d⊥ closes all of them. u looks like one input to the decision, and d⊥ like the decision itself.
- **Free generation agrees with fixed text this time.** For looping, teacher-forced ablation effects didn't carry over. Here the top screened head (L22.lin.h13) did. A plausible reason: closing is a single-token event the screen measures directly, while looping builds up over thousands of tokens.

## Limits
- **One subject model, p50c, not the base model.** The base model rarely concludes, so this maps closing in Sparks' edited model. Everything except its 128 edited neurons is the base model's.
- 64 test problems (GSM8K train) × 3 seed sets, an 8K cap. 111/192 reference answers hit the cap, so "closing rate" partly means "closing before 8K".
- d⊥ was fitted at the decision point, and its L19 lens promotes `</think>` directly. Stages 6–7 show the direct word logits aren't what does the work (no `</think>` component: 192/192; no top-20 wrap-up words: 185/192; L15 only: 182/192), but they do speed up closing.
- Blinding removes only the d⊥ span from the blocks' inputs. Blinding also hides the model's own closing signal, so near-zero closing with L18 + L20–23 blind may partly reflect a general loss of the ability to close, not only the push's route. L16, L17 and L19 were never blinded alone or apart from each other, and nothing is resolved below the layer level.
- Two single components (L22.lin.h13, L11.attn.h1) were tested in free generation, and the d⊥ + head tests used two strengths (1× and 0.25×). Log-odds additivity rests on four pooled rates. The other strong screen hits (L23.mlp.932, L23.mlp.385, L19.mlp.31) are teacher-forced only.
- Stage 3 measures direct paths only, and its completeness (1.2 × 10⁻³) is just over target.
- The random control isn't neutral at this strength (see stage 5).
