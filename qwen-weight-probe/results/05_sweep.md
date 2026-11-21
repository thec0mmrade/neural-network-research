# Dose-response sweep: steering strength vs looping

Same 64 problems, seeds, 4,096-token cap and sampling as Phase 4. Strength is a multiple of the per-layer β from `04/analysis.json` (L15 0.727, L19 1.075). 0× = the unsteered model (base or p50c). p = Fisher exact, two-sided, looping vs the row's 0× reference.

## Edited model (p50c) pushed TOWARD the loop side of u

| strength | looping | p vs 0× | mean repeated share | closed thinking | correct | hit 4K cap |
|---|---|---|---|---|---|---|
| 0× | 0/64 | 1 | 0.035 | 6/64 | 6/64 | 58/64 |
| 0.25× | 1/64 | 1 | 0.064 | 7/64 | 6/64 | 58/64 |
| 0.5× | 9/64 | 0.0029 | 0.119 | 2/64 | 2/64 | 62/64 |
| 1× | 55/64 | 8.1e-27 | 0.418 | 3/64 | 1/64 | 63/64 |
| 2× | 62/64 | 1.8e-34 | 0.809 | 0/64 | 0/64 | 64/64 |
| 4× | 62/64 | 1.8e-34 | 0.945 | 0/64 | 0/64 | 64/64 |

## Base model pushed AWAY from the loop side of u

| strength | looping | p vs 0× | mean repeated share | closed thinking | correct | hit 4K cap |
|---|---|---|---|---|---|---|
| 0× | 13/64 | 1 | 0.142 | 4/64 | 4/64 | 60/64 |
| 0.25× | 3/64 | 0.014 | 0.061 | 7/64 | 7/64 | 57/64 |
| 0.5× | 0/64 | 0.00012 | 0.033 | 10/64 | 7/64 | 57/64 |
| 1× | 0/64 | 0.00012 | 0.022 | 14/64 | 12/64 | 50/64 |
| 2× | 1/64 | 0.001 | 0.016 | 21/64 | 9/64 | 49/64 |
| 4× | 7/64 | 0.22 | 0.072 | 19/64 | 0/64 | 4/64 |

## Edited model pushed along a fixed random direction, same norm (control)

| strength | looping | p vs 0× | mean repeated share | closed thinking | correct | hit 4K cap |
|---|---|---|---|---|---|---|
| 0× | 0/64 | 1 | 0.035 | 6/64 | 6/64 | 58/64 |
| 0.25× | 1/64 | 1 | 0.041 | 5/64 | 5/64 | 59/64 |
| 0.5× | 1/64 | 1 | 0.047 | 7/64 | 7/64 | 57/64 |
| 1× | 1/64 | 1 | 0.059 | 4/64 | 4/64 | 60/64 |
| 2× | 15/64 | 2.4e-05 | 0.141 | 3/64 | 3/64 | 61/64 |
| 4× | 64/64 | 8.4e-38 | 0.810 | 0/64 | 0/64 | 64/64 |

## Toward-u vs random at matched strength

| strength | toward u | random | p |
|---|---|---|---|
| 0.25× | 1/64 | 1/64 | 1 |
| 0.5× | 9/64 | 1/64 | 0.017 |
| 1× | 55/64 | 1/64 | 4e-25 |
| 2× | 62/64 | 15/64 | 3.8e-19 |
| 4× | 62/64 | 64/64 | 0.5 |

## Where outputs degrade (decoded samples)

- **Toward u, 4×** (`p50c_pro_x4`): the loops are degenerate from the first line, e.g. "The question is 2. The question is 2. …" or one sentence repeated to the cap. At 2–4× the repeated share is 0.81–0.95.
- **Random, 4×** (`p50c_rand_x4`): the math is garbled (`\math{M`, "or, or, or"), then the output loops: 64/64 looping. **A push this large breaks the model in any direction.** At 2×, random already gives 15/64 loops, so above about 1× not all looping is specific to u.
- **Away from u, 4×** (`base_anti_x4`): the opposite failure. Median length is 480 tokens, the text is garbled with stray Unicode, and the model declares success early ("Results verified.", "All criteria satisfied.") without solving anything: 0/64 correct, and only 4/64 hit the cap. That fits the wrap-up pole that u's vocabulary readout showed (" thus", " accordingly"). Its 7/64 "looping" are short garbled repeats, not reasoning loops.
- **Away from u, 2×** (`base_anti_x2`): still mostly readable reasoning, with some LaTeX slips. Loops stay suppressed (1/64) and it closes thinking most often (21/64). Correct answers (9/64) are below 1× (12/64).

## Reading

- **u's effect is graded and specific up to about 1×.**
  - Toward u, loops go 0 → 1 → 9 → 55/64 from 0× to 1×. Random stays at 0–1/64 over the same range (pro vs random p = 0.017 at 0.5×, 4e-25 at 1×).
  - Away from u, 0.25× already cuts loops from 13 to 3/64, and 0.5× removes them.
- **The threshold is around 0.5×.** Loops first appear significantly at 0.5× toward u (9/64, p = 0.003), and are gone by 0.5× away from u.
- **Above 1×, the specificity fades.** At 2×, u still beats random clearly (62 vs 15/64, p = 4e-19). At 4×, both loop completely (62 vs 64/64, p = 0.5), because the push destroys the output either way.
- **The useful range away from loops is about 0.5–1×.** Within it, loops are gone and closed thinking and correct answers are highest (1×: 14/64 closed, 12/64 correct). By 4× the model stops reasoning and claims to be done.
