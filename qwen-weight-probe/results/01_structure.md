# Level 1: structural map (all language-model tensors, weights only)

320 tensors, 752,393,024 parameters. Code: `src/probe/structure.py`; full data: `01_structure.json`.

Columns: `fro` = Frobenius norm; `sv1` = share of squared norm in the top singular direction; `eff. rank` = exp(entropy of the normalized squared singular values); `row`/`col max/med` = largest row/column norm over the median; `kurt` = excess kurtosis of the values (0 for a Gaussian).

## Review (by hand, 2026-11-16)
The automatic flags are below. What they add up to:

1. **Residual dimension 0 is a "massive" dimension that the output can't see.**
   - 43 of the 48 write matrices (nearly every MLP, attention and linear-attention output) write it 2.6× harder than a typical dimension. It is the only dimension written over 2× anywhere.
   - The norms then shrink it when reading: 0.84× (input norms) and 0.67× (post-attention norms) of the typical scale, and **0.05×** in the final norm (w = −0.78, the only negative entry).
   - This matches the "massive activation" pattern reported in other LLMs, where one dimension carries a large constant-like value that works as a bias or attention sink. It is predicted here from weights alone; checking it needs activations.
2. **A band of dimensions is quiet in both directions.** About 400–800 dims (e.g. 443, 521, 523, 545, 807) are written at about 0.4× the median and read at about 0.7×.
3. **Some dimensions are internal-only.** Dims 42–164 (e.g. 46, 66, 74, 143) are read 1.1–1.2× more strongly than typical by the input norms, but the final norm scales them by only 0.25–0.37×. Middle layers use them, but they barely reach the vocabulary.
4. **Layer 23 is the most unusual layer.**
   - Its attention value matrix puts 13% of its energy in one direction, 11× the other full-attention layers (z = +52, the largest flag).
   - Its MLP up-projection has 8× the usual top-direction share, the largest norm, and one dominant column.
   - Its o_proj, k_proj and q_proj are also heavier-tailed.
   - That direction doesn't line up with u, d⊥ or d_close (|cos| < 0.05), or with dim 0 or the mean embedding. So it isn't the closing machinery seen at the weight level, though layer 23 does carry most of `</think>` (`08_circuits/closing.md`).
5. **Layer 0's linear attention is low-rank.** Its in_proj_qkv has effective rank 334 against a typical 678, with 16–20% of its energy in one direction that reads dims 38, 0, 119 and 83. The first layer reads only a few input features strongly.
6. **Smaller flags:**
   - The decay parameters (A_log) of layers 0 and 10 have unusual spreads.
   - Layer 3 (the first full-attention layer) has a larger q_proj.
   - The up_proj norms grow over layers 21–23.
   - The layer-1 conv1d has heavy-tailed taps (kurtosis 48).
7. **No single neuron or column dominates a matrix.** The largest column- or row-norm ratio anywhere is 7.2× (layer 4 in_proj_b), and no write-matrix column exceeds 8×. The weights alone don't point to individually special neurons. That fits Phase 1, where the weights didn't single out the edited neurons either (`03_lens.md`).
8. **The embedding has one strong common direction.** The top singular direction holds 21% of its energy, close to the mean token vector (norm 0.28 against a typical row norm of 0.63). Its effective rank is 370 of 1,024.
9. **The final norm scales by about 4.3** (mean w = +3.3), against 1.1–1.7 for the layer norms. The linear-attention gate norms (`linear_attn.norm`, F32) average 0.94, which fits a norm that scales by w itself rather than (1 + w).


### `linear_attn.in_proj_a.weight` (16, 1024)

| layer | fro | sv1 | eff. rank | rank 90% | row max/med | col max/med | kurt | max/std |
|---|---|---|---|---|---|---|---|---|
| 0 | 4.8 | 0.468 | 7 | 10 | 1.2 | 3.1 | 0.6 | 5 |
| 1 | 3.9 | 0.340 | 8 | 8 | 2.0 | 3.2 | 1.9 | 7 |
| 2 | 3.5 | 0.303 | 9 | 10 | 1.4 | 4.4 | 1.7 | 7 |
| 4 | 3.5 | 0.337 | 9 | 10 | 1.6 | 4.3 | 1.4 | 6 |
| 5 | 3.1 | 0.404 | 8 | 10 | 1.3 | 4.0 | 1.1 | 6 |
| 6 | 2.9 | 0.379 | 8 | 10 | 1.5 | 4.0 | 1.4 | 7 |
| 8 | 2.9 | 0.344 | 9 | 10 | 1.9 | 3.7 | 1.4 | 7 |
| 9 | 3.3 | 0.385 | 8 | 10 | 1.3 | 4.1 | 1.6 | 10 |
| 10 | 3.1 | 0.393 | 9 | 10 | 1.5 | 3.9 | 1.2 | 7 |
| 12 | 3.4 | 0.354 | 9 | 10 | 1.7 | 3.9 | 1.5 | 6 |
| 13 | 2.9 | 0.389 | 8 | 9 | 1.4 | 3.8 | 1.3 | 7 |
| 14 | 2.9 | 0.338 | 9 | 10 | 1.6 | 3.8 | 1.8 | 9 |
| 16 | 3.2 | 0.373 | 9 | 10 | 1.6 | 4.3 | 1.6 | 6 |
| 17 | 3.2 | 0.379 | 8 | 10 | 1.6 | 5.2 | 3.2 | 9 |
| 18 | 3.0 | 0.374 | 8 | 10 | 1.3 | 5.0 | 2.7 | 9 |
| 20 | 3.6 | 0.385 | 8 | 9 | 1.4 | 5.0 | 2.8 | 7 |
| 21 | 3.5 | 0.412 | 8 | 10 | 1.1 | 5.3 | 3.3 | 8 |
| 22 | 3.9 | 0.425 | 8 | 10 | 1.2 | 5.6 | 2.8 | 7 |

### `linear_attn.in_proj_b.weight` (16, 1024)

| layer | fro | sv1 | eff. rank | rank 90% | row max/med | col max/med | kurt | max/std |
|---|---|---|---|---|---|---|---|---|
| 0 | 2.2 | 0.347 | 7 | 8 | 2.0 | 4.4 | 3.2 | 8 |
| 1 | 2.4 | 0.394 | 7 | 8 | 1.4 | 5.7 | 4.3 | 11 |
| 2 | 2.4 | 0.307 | 7 | 7 | 1.6 | 5.9 | 4.1 | 10 |
| 4 | 2.0 | 0.373 | 7 | 8 | 1.4 | 7.2 | 4.7 | 9 |
| 5 | 2.0 | 0.413 | 7 | 8 | 1.4 | 6.4 | 3.3 | 10 |
| 6 | 2.2 | 0.479 | 6 | 8 | 1.3 | 5.3 | 2.8 | 8 |
| 8 | 1.8 | 0.455 | 7 | 8 | 1.3 | 5.1 | 2.1 | 8 |
| 9 | 2.3 | 0.355 | 8 | 8 | 1.4 | 5.2 | 2.9 | 8 |
| 10 | 2.3 | 0.397 | 7 | 8 | 1.6 | 4.3 | 3.1 | 9 |
| 12 | 2.0 | 0.424 | 7 | 8 | 1.4 | 5.2 | 2.0 | 7 |
| 13 | 2.0 | 0.510 | 6 | 7 | 1.2 | 4.2 | 1.7 | 8 |
| 14 | 2.0 | 0.332 | 9 | 9 | 1.5 | 3.5 | 2.8 | 10 |
| 16 | 2.2 | 0.350 | 8 | 8 | 1.8 | 3.9 | 3.1 | 9 |
| 17 | 2.0 | 0.406 | 7 | 8 | 1.4 | 4.9 | 2.8 | 8 |
| 18 | 2.0 | 0.435 | 7 | 9 | 1.4 | 4.6 | 2.5 | 8 |
| 20 | 2.4 | 0.424 | 7 | 9 | 1.5 | 4.1 | 2.2 | 7 |
| 21 | 2.2 | 0.452 | 7 | 8 | 1.3 | 4.0 | 1.7 | 7 |
| 22 | 2.5 | 0.401 | 7 | 9 | 1.2 | 5.0 | 3.1 | 11 |

### `linear_attn.in_proj_qkv.weight` (6144, 1024)

| layer | fro | sv1 | eff. rank | rank 90% | row max/med | col max/med | kurt | max/std |
|---|---|---|---|---|---|---|---|---|
| 0 | 41.5 | 0.162 | 334 | 658 | 7.1 | 2.0 | 5.4 | 24 |
| 1 | 41.7 | 0.016 | 760 | 742 | 4.1 | 1.2 | 2.1 | 19 |
| 2 | 43.3 | 0.018 | 726 | 729 | 3.5 | 1.3 | 2.2 | 14 |
| 4 | 47.4 | 0.028 | 712 | 722 | 5.1 | 1.3 | 2.8 | 19 |
| 5 | 44.5 | 0.026 | 683 | 703 | 3.6 | 1.2 | 2.4 | 19 |
| 6 | 42.7 | 0.017 | 700 | 696 | 4.1 | 1.1 | 2.3 | 13 |
| 8 | 42.8 | 0.030 | 648 | 659 | 4.6 | 1.1 | 2.8 | 14 |
| 9 | 40.7 | 0.017 | 671 | 660 | 3.9 | 1.1 | 2.8 | 17 |
| 10 | 41.4 | 0.016 | 617 | 626 | 4.0 | 1.2 | 3.0 | 15 |
| 12 | 46.2 | 0.027 | 650 | 656 | 4.5 | 1.1 | 3.0 | 15 |
| 13 | 43.7 | 0.023 | 648 | 658 | 3.9 | 1.1 | 3.1 | 13 |
| 14 | 44.1 | 0.021 | 665 | 657 | 4.3 | 1.1 | 3.5 | 25 |
| 16 | 53.2 | 0.038 | 681 | 695 | 4.2 | 1.4 | 3.0 | 16 |
| 17 | 56.8 | 0.042 | 714 | 722 | 4.5 | 1.4 | 2.7 | 17 |
| 18 | 58.9 | 0.055 | 675 | 728 | 3.4 | 1.5 | 3.1 | 16 |
| 20 | 59.2 | 0.054 | 662 | 720 | 3.6 | 1.8 | 3.9 | 25 |
| 21 | 56.9 | 0.051 | 685 | 729 | 4.5 | 1.7 | 3.2 | 22 |
| 22 | 50.2 | 0.054 | 690 | 738 | 3.7 | 1.5 | 3.1 | 20 |

### `linear_attn.in_proj_z.weight` (2048, 1024)

| layer | fro | sv1 | eff. rank | rank 90% | row max/med | col max/med | kurt | max/std |
|---|---|---|---|---|---|---|---|---|
| 0 | 25.1 | 0.046 | 607 | 630 | 2.6 | 1.6 | 2.2 | 27 |
| 1 | 27.8 | 0.034 | 647 | 613 | 2.0 | 1.5 | 1.0 | 10 |
| 2 | 28.3 | 0.053 | 574 | 585 | 2.3 | 1.8 | 1.5 | 10 |
| 4 | 26.0 | 0.048 | 580 | 577 | 2.6 | 1.7 | 2.0 | 10 |
| 5 | 23.9 | 0.035 | 599 | 581 | 2.6 | 1.4 | 1.9 | 13 |
| 6 | 23.1 | 0.050 | 521 | 545 | 3.0 | 1.5 | 2.0 | 13 |
| 8 | 21.4 | 0.063 | 466 | 519 | 3.4 | 1.6 | 2.9 | 24 |
| 9 | 21.1 | 0.060 | 417 | 477 | 3.5 | 1.3 | 2.2 | 26 |
| 10 | 20.0 | 0.104 | 348 | 485 | 3.3 | 1.7 | 1.5 | 12 |
| 12 | 21.7 | 0.069 | 424 | 491 | 3.3 | 1.6 | 3.0 | 13 |
| 13 | 20.9 | 0.057 | 431 | 496 | 3.3 | 1.6 | 2.5 | 13 |
| 14 | 22.4 | 0.081 | 376 | 475 | 2.9 | 2.3 | 3.1 | 13 |
| 16 | 23.0 | 0.053 | 457 | 504 | 3.1 | 1.7 | 2.8 | 13 |
| 17 | 24.1 | 0.046 | 537 | 541 | 2.8 | 1.8 | 2.1 | 10 |
| 18 | 24.9 | 0.052 | 541 | 552 | 2.6 | 1.8 | 2.4 | 13 |
| 20 | 25.4 | 0.051 | 502 | 526 | 2.9 | 1.7 | 2.7 | 18 |
| 21 | 25.8 | 0.046 | 551 | 562 | 2.5 | 1.5 | 1.7 | 16 |
| 22 | 25.9 | 0.052 | 450 | 541 | 3.0 | 2.2 | 2.8 | 21 |

### `linear_attn.out_proj.weight` (1024, 2048)

| layer | fro | sv1 | eff. rank | rank 90% | row max/med | col max/med | kurt | max/std |
|---|---|---|---|---|---|---|---|---|
| 0 | 17.9 | 0.019 | 788 | 740 | 3.6 | 1.4 | 2.5 | 23 |
| 1 | 18.3 | 0.023 | 634 | 601 | 3.7 | 1.2 | 2.7 | 26 |
| 2 | 18.1 | 0.021 | 650 | 597 | 3.6 | 1.5 | 2.6 | 24 |
| 4 | 17.0 | 0.016 | 637 | 582 | 3.2 | 1.9 | 1.8 | 21 |
| 5 | 17.1 | 0.011 | 660 | 603 | 2.8 | 1.4 | 1.6 | 23 |
| 6 | 17.5 | 0.014 | 627 | 571 | 2.6 | 1.7 | 1.4 | 18 |
| 8 | 16.6 | 0.014 | 578 | 532 | 2.8 | 1.7 | 1.5 | 20 |
| 9 | 16.3 | 0.022 | 462 | 460 | 3.1 | 1.7 | 1.8 | 21 |
| 10 | 16.9 | 0.021 | 527 | 512 | 2.4 | 1.9 | 1.6 | 23 |
| 12 | 16.2 | 0.019 | 513 | 509 | 2.4 | 1.6 | 1.6 | 18 |
| 13 | 16.8 | 0.015 | 551 | 520 | 2.3 | 1.5 | 1.6 | 27 |
| 14 | 16.9 | 0.011 | 605 | 554 | 2.2 | 2.7 | 1.8 | 15 |
| 16 | 16.0 | 0.014 | 671 | 600 | 2.7 | 2.4 | 2.2 | 24 |
| 17 | 15.5 | 0.009 | 643 | 559 | 2.6 | 2.7 | 1.9 | 16 |
| 18 | 14.6 | 0.011 | 662 | 582 | 3.0 | 1.9 | 2.6 | 24 |
| 20 | 15.2 | 0.009 | 682 | 601 | 2.8 | 2.1 | 2.3 | 25 |
| 21 | 16.8 | 0.007 | 697 | 614 | 2.4 | 1.6 | 1.5 | 22 |
| 22 | 17.7 | 0.008 | 722 | 629 | 2.6 | 1.7 | 2.4 | 27 |

### `mlp.down_proj.weight` (1024, 3584)

| layer | fro | sv1 | eff. rank | rank 90% | row max/med | col max/med | kurt | max/std |
|---|---|---|---|---|---|---|---|---|
| 0 | 16.3 | 0.012 | 854 | 743 | 3.2 | 1.1 | 1.3 | 31 |
| 1 | 16.5 | 0.011 | 815 | 706 | 3.1 | 1.2 | 1.2 | 22 |
| 2 | 15.8 | 0.013 | 810 | 697 | 3.3 | 1.3 | 1.7 | 27 |
| 3 | 15.5 | 0.009 | 790 | 677 | 2.6 | 1.2 | 1.2 | 28 |
| 4 | 15.6 | 0.008 | 804 | 690 | 2.6 | 1.4 | 1.2 | 26 |
| 5 | 15.8 | 0.010 | 774 | 668 | 2.7 | 1.3 | 1.4 | 27 |
| 6 | 15.7 | 0.007 | 729 | 637 | 2.2 | 1.7 | 1.0 | 17 |
| 7 | 15.8 | 0.011 | 739 | 647 | 2.8 | 1.4 | 1.3 | 23 |
| 8 | 15.4 | 0.010 | 727 | 637 | 2.6 | 1.4 | 1.1 | 19 |
| 9 | 15.5 | 0.012 | 694 | 628 | 2.8 | 1.6 | 1.4 | 24 |
| 10 | 15.2 | 0.012 | 703 | 622 | 2.8 | 1.9 | 1.6 | 20 |
| 11 | 15.5 | 0.012 | 690 | 624 | 2.9 | 1.8 | 1.4 | 20 |
| 12 | 15.2 | 0.009 | 699 | 620 | 2.4 | 1.5 | 1.0 | 17 |
| 13 | 15.3 | 0.008 | 692 | 613 | 2.3 | 1.8 | 1.0 | 18 |
| 14 | 15.1 | 0.008 | 750 | 659 | 2.6 | 1.8 | 1.7 | 25 |
| 15 | 16.4 | 0.008 | 766 | 673 | 2.4 | 1.8 | 1.7 | 21 |
| 16 | 16.4 | 0.006 | 766 | 654 | 2.2 | 1.7 | 1.0 | 28 |
| 17 | 16.4 | 0.005 | 752 | 640 | 2.0 | 1.3 | 0.7 | 23 |
| 18 | 16.4 | 0.006 | 742 | 633 | 2.1 | 1.3 | 0.9 | 32 |
| 19 | 17.3 | 0.006 | 790 | 686 | 2.2 | 1.5 | 0.8 | 27 |
| 20 | 17.5 | 0.005 | 790 | 680 | 2.0 | 1.4 | 0.6 | 27 |
| 21 | 17.7 | 0.005 | 788 | 681 | 1.9 | 1.6 | 0.5 | 17 |
| 22 | 17.5 | 0.005 | 786 | 683 | 2.1 | 1.5 | 0.9 | 22 |
| 23 | 16.4 | 0.007 | 769 | 680 | 2.5 | 2.0 | 3.9 | 39 |

### `mlp.gate_proj.weight` (3584, 1024)

| layer | fro | sv1 | eff. rank | rank 90% | row max/med | col max/med | kurt | max/std |
|---|---|---|---|---|---|---|---|---|
| 0 | 23.2 | 0.021 | 793 | 730 | 2.7 | 1.3 | 0.5 | 15 |
| 1 | 25.5 | 0.019 | 798 | 728 | 2.4 | 1.4 | 0.4 | 12 |
| 2 | 31.1 | 0.037 | 740 | 704 | 2.0 | 1.6 | 0.5 | 9 |
| 3 | 31.3 | 0.035 | 719 | 684 | 2.0 | 1.5 | 0.9 | 9 |
| 4 | 29.8 | 0.030 | 736 | 683 | 2.2 | 1.3 | 0.9 | 15 |
| 5 | 28.0 | 0.026 | 722 | 662 | 2.6 | 1.3 | 1.1 | 13 |
| 6 | 26.5 | 0.027 | 686 | 638 | 2.7 | 1.4 | 1.8 | 17 |
| 7 | 23.9 | 0.025 | 684 | 643 | 3.3 | 1.4 | 1.4 | 12 |
| 8 | 22.2 | 0.022 | 686 | 639 | 3.4 | 1.3 | 1.5 | 14 |
| 9 | 20.4 | 0.018 | 684 | 636 | 3.0 | 1.2 | 1.6 | 14 |
| 10 | 21.2 | 0.025 | 634 | 611 | 4.4 | 1.4 | 3.2 | 15 |
| 11 | 21.0 | 0.024 | 641 | 614 | 3.9 | 1.4 | 3.3 | 15 |
| 12 | 23.0 | 0.027 | 650 | 624 | 3.9 | 1.4 | 4.3 | 19 |
| 13 | 23.8 | 0.030 | 648 | 627 | 4.0 | 1.4 | 4.0 | 15 |
| 14 | 26.2 | 0.035 | 625 | 611 | 4.4 | 1.4 | 7.7 | 25 |
| 15 | 27.3 | 0.042 | 633 | 623 | 3.6 | 1.5 | 4.7 | 22 |
| 16 | 29.5 | 0.036 | 677 | 647 | 2.8 | 1.4 | 3.2 | 18 |
| 17 | 32.2 | 0.037 | 692 | 655 | 2.6 | 1.5 | 2.2 | 16 |
| 18 | 34.0 | 0.044 | 684 | 661 | 2.3 | 1.5 | 1.9 | 21 |
| 19 | 32.7 | 0.039 | 675 | 651 | 2.9 | 1.5 | 2.2 | 28 |
| 20 | 32.6 | 0.033 | 698 | 662 | 2.8 | 1.4 | 1.5 | 16 |
| 21 | 32.9 | 0.037 | 693 | 677 | 2.4 | 1.5 | 2.0 | 22 |
| 22 | 33.5 | 0.043 | 663 | 680 | 2.9 | 1.6 | 3.4 | 23 |
| 23 | 36.2 | 0.065 | 605 | 690 | 2.6 | 1.5 | 4.0 | 27 |

### `mlp.up_proj.weight` (3584, 1024)

| layer | fro | sv1 | eff. rank | rank 90% | row max/med | col max/med | kurt | max/std |
|---|---|---|---|---|---|---|---|---|
| 0 | 16.4 | 0.003 | 875 | 754 | 2.2 | 1.1 | 0.2 | 10 |
| 1 | 16.4 | 0.008 | 855 | 742 | 1.4 | 1.1 | 0.2 | 12 |
| 2 | 15.6 | 0.005 | 869 | 746 | 1.8 | 1.1 | 0.3 | 13 |
| 3 | 15.3 | 0.003 | 855 | 729 | 1.3 | 1.1 | 0.2 | 12 |
| 4 | 15.6 | 0.004 | 851 | 726 | 2.0 | 1.1 | 0.3 | 11 |
| 5 | 15.9 | 0.004 | 834 | 709 | 1.8 | 1.1 | 0.4 | 8 |
| 6 | 16.0 | 0.008 | 794 | 685 | 2.2 | 1.1 | 0.7 | 13 |
| 7 | 16.2 | 0.004 | 808 | 693 | 1.5 | 1.1 | 0.5 | 9 |
| 8 | 16.0 | 0.005 | 787 | 678 | 1.7 | 1.1 | 0.6 | 9 |
| 9 | 16.1 | 0.006 | 761 | 666 | 2.1 | 1.1 | 0.6 | 9 |
| 10 | 15.9 | 0.005 | 768 | 674 | 2.1 | 1.1 | 0.7 | 10 |
| 11 | 16.1 | 0.008 | 762 | 674 | 2.1 | 1.2 | 0.7 | 13 |
| 12 | 15.8 | 0.006 | 765 | 673 | 1.6 | 1.2 | 0.6 | 11 |
| 13 | 15.9 | 0.006 | 767 | 680 | 2.4 | 1.2 | 0.7 | 12 |
| 14 | 16.1 | 0.013 | 753 | 688 | 2.0 | 1.2 | 1.1 | 13 |
| 15 | 17.1 | 0.005 | 796 | 696 | 2.1 | 1.2 | 1.3 | 16 |
| 16 | 17.0 | 0.005 | 788 | 683 | 1.6 | 1.2 | 0.4 | 9 |
| 17 | 17.1 | 0.006 | 793 | 685 | 1.5 | 1.1 | 0.4 | 10 |
| 18 | 17.4 | 0.008 | 792 | 690 | 1.3 | 1.1 | 0.3 | 10 |
| 19 | 17.9 | 0.008 | 776 | 678 | 1.5 | 1.2 | 0.3 | 9 |
| 20 | 18.2 | 0.009 | 779 | 683 | 1.4 | 1.2 | 0.2 | 10 |
| 21 | 18.7 | 0.009 | 791 | 698 | 1.6 | 1.2 | 0.3 | 14 |
| 22 | 19.2 | 0.020 | 763 | 699 | 1.7 | 1.5 | 0.5 | 17 |
| 23 | 21.2 | 0.048 | 651 | 696 | 2.6 | 2.3 | 1.2 | 20 |

### `self_attn.k_proj.weight` (512, 1024)

| layer | fro | sv1 | eff. rank | rank 90% | row max/med | col max/med | kurt | max/std |
|---|---|---|---|---|---|---|---|---|
| 3 | 9.0 | 0.021 | 314 | 294 | 2.1 | 1.9 | 1.8 | 9 |
| 7 | 7.7 | 0.021 | 284 | 248 | 1.5 | 1.6 | 1.3 | 10 |
| 11 | 8.4 | 0.020 | 225 | 225 | 1.6 | 1.8 | 1.9 | 10 |
| 15 | 8.3 | 0.017 | 243 | 217 | 1.9 | 1.5 | 2.3 | 11 |
| 19 | 8.1 | 0.037 | 173 | 185 | 1.7 | 2.7 | 3.0 | 16 |
| 23 | 7.3 | 0.023 | 207 | 199 | 1.6 | 3.7 | 5.6 | 15 |

### `self_attn.o_proj.weight` (1024, 2048)

| layer | fro | sv1 | eff. rank | rank 90% | row max/med | col max/med | kurt | max/std |
|---|---|---|---|---|---|---|---|---|
| 3 | 17.6 | 0.013 | 662 | 588 | 3.1 | 2.0 | 1.8 | 27 |
| 7 | 16.2 | 0.016 | 483 | 463 | 2.0 | 2.0 | 1.3 | 14 |
| 11 | 16.9 | 0.017 | 470 | 439 | 2.1 | 1.8 | 1.5 | 11 |
| 15 | 20.0 | 0.011 | 550 | 494 | 1.7 | 2.9 | 1.4 | 12 |
| 19 | 24.2 | 0.007 | 607 | 535 | 1.9 | 1.8 | 1.2 | 13 |
| 23 | 20.1 | 0.018 | 578 | 536 | 2.1 | 4.5 | 3.8 | 19 |

### `self_attn.q_proj.weight` (4096, 1024)

| layer | fro | sv1 | eff. rank | rank 90% | row max/med | col max/med | kurt | max/std |
|---|---|---|---|---|---|---|---|---|
| 3 | 40.5 | 0.073 | 600 | 670 | 3.2 | 1.6 | 2.4 | 13 |
| 7 | 31.5 | 0.073 | 507 | 592 | 3.6 | 1.5 | 2.7 | 17 |
| 11 | 28.9 | 0.100 | 371 | 524 | 3.8 | 1.8 | 2.2 | 14 |
| 15 | 32.1 | 0.099 | 339 | 499 | 4.7 | 2.0 | 2.4 | 17 |
| 19 | 31.9 | 0.124 | 304 | 497 | 3.0 | 2.7 | 2.5 | 13 |
| 23 | 33.6 | 0.071 | 461 | 603 | 3.7 | 2.5 | 3.9 | 24 |

### `self_attn.v_proj.weight` (512, 1024)

| layer | fro | sv1 | eff. rank | rank 90% | row max/med | col max/med | kurt | max/std |
|---|---|---|---|---|---|---|---|---|
| 3 | 9.2 | 0.011 | 349 | 301 | 1.6 | 1.3 | 1.8 | 11 |
| 7 | 8.6 | 0.012 | 336 | 284 | 1.8 | 1.3 | 2.3 | 11 |
| 11 | 10.5 | 0.010 | 319 | 277 | 1.7 | 1.3 | 1.2 | 10 |
| 15 | 12.0 | 0.014 | 341 | 288 | 2.2 | 1.8 | 6.5 | 18 |
| 19 | 14.4 | 0.012 | 260 | 208 | 1.5 | 2.6 | 8.8 | 19 |
| 23 | 13.4 | 0.131 | 195 | 238 | 2.2 | 2.2 | 6.1 | 29 |

### Vectors (norms, linear-attention parameters)

| tensor kind | layers | mean (range over layers) | std (range) | max |w| | |z|>6 elements (range) |
|---|---|---|---|---|---|
| `input_layernorm.weight` | 24 | +0.410 (+0.131 to +0.732) | 0.135 (0.064 to 0.251) | 1.469 | 0–132 |
| `linear_attn.A_log` | 18 | -1.244 (-1.979 to -0.637) | 0.752 (0.277 to 1.448) | 6.594 |  |
| `linear_attn.conv1d.weight` | 18 | -0.000 (-0.001 to +0.001) | 0.106 (0.068 to 0.177) | 1.820 |  |
| `linear_attn.dt_bias` | 18 | -1.754 (-3.229 to -0.232) | 2.705 (1.648 to 5.762) | 13.062 |  |
| `linear_attn.norm.weight` | 18 | +0.941 (+0.790 to +1.042) | 0.085 (0.056 to 0.130) | 1.194 | 0–4 |
| `norm.weight` | 1 | +3.309 (+3.309 to +3.309) | 0.669 (0.669 to 0.669) | 5.312 | 38–38 |
| `post_attention_layernorm.weight` | 24 | +0.224 (+0.067 to +0.415) | 0.101 (0.039 to 0.178) | 0.820 | 4–100 |
| `self_attn.k_norm.weight` | 6 | +0.505 (+0.393 to +0.630) | 0.235 (0.154 to 0.279) | 1.633 | 0–7 |
| `self_attn.q_norm.weight` | 6 | +0.506 (+0.388 to +0.629) | 0.227 (0.129 to 0.302) | 1.297 | 0–8 |

### Outlier flags

51 flags. Thresholds: robust z > 3.5 against the same tensor kind in other layers; a single row or column > 8x its matrix's median; excess kurtosis > 30.

| type | tensor | metric | value | detail |
|---|---|---|---|---|
| heavy-tailed values | `layers.1.linear_attn.conv1d.weight` | excess kurtosis | 48.5 | max/std 27 |
| layer differs from same kind | `layers.0.linear_attn.A_log` | kurtosis | 3.95 | z +7.4 (kind median -0.165) |
| layer differs from same kind | `layers.0.linear_attn.dt_bias` | std | 5.76 | z +4.1 (kind median 2.41) |
| layer differs from same kind | `layers.0.linear_attn.in_proj_a.weight` | fro | 4.82 | z +3.8 (kind median 3.26) |
| layer differs from same kind | `layers.0.linear_attn.in_proj_a.weight` | std | 0.0377 | z +3.8 (kind median 0.0255) |
| layer differs from same kind | `layers.0.linear_attn.in_proj_b.weight` | row_max/med | 2.02 | z +5.7 (kind median 1.39) |
| layer differs from same kind | `layers.0.linear_attn.in_proj_qkv.weight` | sv1_energy_share | 0.162 | z +8.6 (kind median 0.0273) |
| layer differs from same kind | `layers.0.linear_attn.in_proj_qkv.weight` | effective_rank | 334 | z -9.4 (kind median 678) |
| layer differs from same kind | `layers.0.linear_attn.in_proj_qkv.weight` | row_max/med | 7.08 | z +4.9 (kind median 4.09) |
| layer differs from same kind | `layers.0.linear_attn.in_proj_qkv.weight` | col_max/med | 2.05 | z +3.9 (kind median 1.27) |
| layer differs from same kind | `layers.0.linear_attn.in_proj_qkv.weight` | kurtosis | 5.39 | z +6.8 (kind median 3) |
| layer differs from same kind | `layers.10.linear_attn.A_log` | kurtosis | 5.28 | z +9.8 (kind median -0.165) |
| layer differs from same kind | `layers.10.linear_attn.dt_bias` | kurtosis | 1.4 | z +3.8 (kind median -1.15) |
| layer differs from same kind | `layers.10.linear_attn.in_proj_z.weight` | sv1_energy_share | 0.104 | z +6.0 (kind median 0.0521) |
| layer differs from same kind | `layers.13.linear_attn.in_proj_b.weight` | effective_rank | 5.55 | z -3.5 (kind median 7.1) |
| layer differs from same kind | `layers.14.linear_attn.in_proj_z.weight` | col_max/med | 2.3 | z +3.8 (kind median 1.62) |
| layer differs from same kind | `layers.14.linear_attn.in_proj_z.weight` | mean | 0.000117 | z +4.3 (kind median -3.84e-05) |
| layer differs from same kind | `layers.14.linear_attn.out_proj.weight` | col_max/med | 2.69 | z +3.9 (kind median 1.69) |
| layer differs from same kind | `layers.16.linear_attn.in_proj_b.weight` | row_max/med | 1.82 | z +3.8 (kind median 1.39) |
| layer differs from same kind | `layers.17.linear_attn.out_proj.weight` | col_max/med | 2.72 | z +4.0 (kind median 1.69) |
| layer differs from same kind | `layers.19.self_attn.k_proj.weight` | sv1_energy_share | 0.0373 | z +8.5 (kind median 0.021) |
| layer differs from same kind | `layers.19.self_attn.o_proj.weight` | mean | -2.04e-05 | z -3.6 (kind median 2.53e-06) |
| layer differs from same kind | `layers.20.linear_attn.conv1d.weight` | std | 0.177 | z +3.7 (kind median 0.0855) |
| layer differs from same kind | `layers.20.linear_attn.in_proj_qkv.weight` | fro | 59.2 | z +3.5 (kind median 44.3) |
| layer differs from same kind | `layers.20.linear_attn.in_proj_qkv.weight` | std | 0.0236 | z +3.5 (kind median 0.0177) |
| layer differs from same kind | `layers.21.input_layernorm.weight` | kurtosis | 19.2 | z +4.7 (kind median 4.8) |
| layer differs from same kind | `layers.21.linear_attn.in_proj_a.weight` | kurtosis | 3.3 | z +3.6 (kind median 1.58) |
| layer differs from same kind | `layers.21.mlp.up_proj.weight` | fro | 18.7 | z +3.8 (kind median 16.2) |
| layer differs from same kind | `layers.21.mlp.up_proj.weight` | std | 0.00977 | z +3.8 (kind median 0.00843) |
| layer differs from same kind | `layers.21.post_attention_layernorm.weight` | kurtosis | 17.3 | z +4.5 (kind median 4.7) |
| layer differs from same kind | `layers.22.mlp.up_proj.weight` | fro | 19.2 | z +4.4 (kind median 16.2) |
| layer differs from same kind | `layers.22.mlp.up_proj.weight` | sv1_energy_share | 0.0195 | z +5.0 (kind median 0.00569) |
| layer differs from same kind | `layers.22.mlp.up_proj.weight` | col_max/med | 1.5 | z +5.8 (kind median 1.14) |
| layer differs from same kind | `layers.22.mlp.up_proj.weight` | std | 0.01 | z +4.4 (kind median 0.00843) |
| layer differs from same kind | `layers.23.mlp.down_proj.weight` | kurtosis | 3.87 | z +8.3 (kind median 1.15) |
| layer differs from same kind | `layers.23.mlp.gate_proj.weight` | sv1_energy_share | 0.0648 | z +3.7 (kind median 0.0316) |
| layer differs from same kind | `layers.23.mlp.up_proj.weight` | fro | 21.2 | z +7.5 (kind median 16.2) |
| layer differs from same kind | `layers.23.mlp.up_proj.weight` | sv1_energy_share | 0.0475 | z +15.2 (kind median 0.00569) |
| layer differs from same kind | `layers.23.mlp.up_proj.weight` | effective_rank | 651 | z -4.0 (kind median 789) |
| layer differs from same kind | `layers.23.mlp.up_proj.weight` | col_max/med | 2.28 | z +18.6 (kind median 1.14) |
| layer differs from same kind | `layers.23.mlp.up_proj.weight` | std | 0.0111 | z +7.5 (kind median 0.00843) |
| layer differs from same kind | `layers.23.self_attn.k_proj.weight` | col_max/med | 3.71 | z +4.6 (kind median 1.85) |
| layer differs from same kind | `layers.23.self_attn.k_proj.weight` | kurtosis | 5.64 | z +4.7 (kind median 2.07) |
| layer differs from same kind | `layers.23.self_attn.o_proj.weight` | col_max/med | 4.49 | z +9.3 (kind median 2) |
| layer differs from same kind | `layers.23.self_attn.o_proj.weight` | kurtosis | 3.82 | z +7.5 (kind median 1.45) |
| layer differs from same kind | `layers.23.self_attn.q_proj.weight` | kurtosis | 3.87 | z +7.4 (kind median 2.45) |
| layer differs from same kind | `layers.23.self_attn.v_proj.weight` | sv1_energy_share | 0.131 | z +51.6 (kind median 0.012) |
| layer differs from same kind | `layers.23.self_attn.v_proj.weight` | effective_rank | 195 | z -5.2 (kind median 328) |
| layer differs from same kind | `layers.3.self_attn.o_proj.weight` | row_max/med | 3.07 | z +8.0 (kind median 2.04) |
| layer differs from same kind | `layers.3.self_attn.q_proj.weight` | fro | 40.5 | z +5.4 (kind median 32) |
| layer differs from same kind | `layers.3.self_attn.q_proj.weight` | std | 0.0198 | z +5.4 (kind median 0.0156) |

### Residual-stream dimensions that stand out

`write` = in how many of the 48 write matrices the dim's row norm is > 2x the median row (mean ratio in brackets). `read gain` = the pre-norm scale (1 + w) for that dim over the norm's median, averaged over 24 layers; < 1 means the dim is read weaker than typical. `final` = the same for the final norm.

**Most strongly written**

| dim | write | read gain (input) | read gain (post-attn) | final |
|---|---|---|---|---|
| 0 | 43 (2.58x) | 0.84 | 0.67 | 0.05 |
| 1004 | 0 (1.17x) | 1.07 | 1.08 | 1.06 |
| 693 | 0 (1.15x) | 1.01 | 1.07 | 1.12 |
| 816 | 0 (1.14x) | 1.05 | 1.07 | 0.91 |
| 813 | 0 (1.13x) | 1.03 | 1.03 | 0.91 |
| 818 | 0 (1.13x) | 1.04 | 1.06 | 1.03 |

**Weakest read (input norms)**

| dim | write | read gain (input) | read gain (post-attn) | final |
|---|---|---|---|---|
| 523 | 0 (0.38x) | 0.71 | 0.70 | 1.01 |
| 443 | 0 (0.41x) | 0.71 | 0.69 | 0.64 |
| 521 | 0 (0.38x) | 0.72 | 0.71 | 0.93 |
| 545 | 0 (0.42x) | 0.72 | 0.70 | 0.63 |
| 684 | 0 (0.41x) | 0.72 | 0.71 | 1.02 |
| 807 | 0 (0.42x) | 0.72 | 0.69 | 0.70 |

**Strongest read (input norms)**

| dim | write | read gain (input) | read gain (post-attn) | final |
|---|---|---|---|---|
| 74 | 0 (0.73x) | 1.22 | 0.85 | 0.32 |
| 46 | 0 (0.64x) | 1.17 | 0.81 | 0.32 |
| 66 | 0 (0.75x) | 1.17 | 0.83 | 0.33 |
| 119 | 0 (0.89x) | 1.14 | 0.93 | 0.36 |
| 143 | 0 (0.86x) | 1.13 | 0.87 | 0.27 |
| 123 | 0 (0.93x) | 1.13 | 0.95 | 0.37 |

**Weakest in the final norm**

| dim | write | read gain (input) | read gain (post-attn) | final |
|---|---|---|---|---|
| 0 | 43 (2.58x) | 0.84 | 0.67 | 0.05 |
| 42 | 0 (0.76x) | 1.11 | 0.86 | 0.25 |
| 143 | 0 (0.86x) | 1.13 | 0.87 | 0.27 |
| 140 | 0 (0.87x) | 1.11 | 0.92 | 0.28 |
| 164 | 0 (0.87x) | 1.10 | 0.91 | 0.32 |
| 46 | 0 (0.64x) | 1.17 | 0.81 | 0.32 |

