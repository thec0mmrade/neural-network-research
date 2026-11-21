# Phase 4: causal checks (64 practice problems: 32 MATH train + 32 GSM8K train, 4,096-token cap)

Sampling as Sparks' think35 protocol (thinking on, temp 0.6, top-p 0.95, top-k 20, presence penalty 1.5), same seeds in every condition. A generation is **looping** if over half of its last 1,024 tokens close a 16-gram that already occurred earlier in the same generation.

| condition | looping | hit 4K cap | closed thinking | correct | mean repeated-token share |
|---|---|---|---|---|---|
| base | 20% | 94% | 6% | 6% | 0.142 |
| p50c | 0% | 91% | 9% | 9% | 0.035 |
| base_anti | 0% | 78% | 22% | 19% | 0.022 |
| p50c_pro | 86% | 98% | 5% | 2% | 0.418 |
| p50c_rand | 2% | 94% | 6% | 6% | 0.059 |

## Teacher-forced analysis of the base generations (254,662 tokens, 13 looping generations, 14.6% of tokens repeated)

`u-write` = what the 64 edited neurons write along u per token (activation × uᵀW column). Cohen's d compares repeated vs non-repeated tokens. `DoM` = difference of means of the residual stream after the layer (repeated − non-repeated), found without using u.

| layer | u-write d (all) | d (positions ≥1024) | d (within looping gens) | all-neuron u-write d | cos(DoM, u) | cos(DoM late, u) | DoM split-half cos | random baseline |
|---|---|---|---|---|---|---|---|---|
| 15 | -1.40 | -1.38 | -2.50 | -1.35 | -0.740 | -0.724 | +0.967 | 0.025 |
| 19 | +1.44 | +1.40 | +2.49 | +1.36 | +0.756 | +0.741 | +0.974 | 0.025 |

Steering magnitude β per layer = 6 × mean |u-write| of the edited neurons (the edit's typical per-token change along u); sign = the side of u the residual sits on during repeated tokens.

| layer | loop sign | β |
|---|---|---|
| 15 | -1 | 0.727 |
| 19 | +1 | 1.075 |

## Significance (Fisher exact, two-sided, looping generations out of 64)

| comparison | looping | p |
|---|---|---|
| base vs p50c | 13 vs 0 | 0.00012 |
| base vs base_anti | 13 vs 0 | 0.00012 |
| p50c vs p50c_pro | 0 vs 55 | 8.1e-27 |
| p50c vs p50c_rand | 0 vs 1 | 1.0 |
| p50c_pro vs p50c_rand | 55 vs 1 | 4.0e-25 |

## Edited neurons' u-write around loop onset (13 looping base generations, 64-token bins from 512 before to 512 after onset)

Onset = start of the first 256-token window that is over half repeated tokens.

- L15: -0.089 -0.093 -0.107 -0.119 -0.118 -0.113 -0.111 -0.106 | **onset** | -0.130 -0.140 -0.186 -0.264 -0.307 -0.320 -0.352 -0.357
- L19: +0.102 +0.123 +0.144 +0.136 +0.142 +0.143 +0.143 +0.133 | **onset** | +0.160 +0.172 +0.213 +0.314 +0.376 +0.420 +0.481 +0.507

The signal rises with the loop and roughly triples over the next ~512 tokens. It is flat beforehand, so this data doesn't show it predicting loop onset.
