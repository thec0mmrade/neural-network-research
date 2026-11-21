# Phase 1: weight survey

320 language-model tensors, 752,393,024 parameters (text model only; tied embeddings).
Final norm weight mean +3.309 (std 0.669); a mean near 0 means the RMSNorm scales by (1 + w).

Output matrices (what each block writes into the residual stream). `max/med` = largest column norm over the median column norm; `sv1 share` = fraction of squared Frobenius norm in the top singular direction.

| layer | type | down fro | down sv1 | down sv1 share | down eff. rank | down max/med col | out fro | out sv1 share | out max/med col |
|---|---|---|---|---|---|---|---|---|---|
| 0 | linear | 16.3 | 1.77 | 0.012 | 854 | 1.1 | 17.9 | 0.019 | 1.4 |
| 1 | linear | 16.5 | 1.73 | 0.011 | 815 | 1.2 | 18.3 | 0.023 | 1.2 |
| 2 | linear | 15.8 | 1.79 | 0.013 | 810 | 1.3 | 18.1 | 0.021 | 1.5 |
| 3 | full | 15.5 | 1.46 | 0.009 | 790 | 1.2 | 17.6 | 0.013 | 2.0 |
| 4 | linear | 15.6 | 1.43 | 0.008 | 804 | 1.4 | 17.0 | 0.016 | 1.9 |
| 5 | linear | 15.8 | 1.57 | 0.010 | 774 | 1.3 | 17.1 | 0.011 | 1.4 |
| 6 | linear | 15.7 | 1.33 | 0.007 | 729 | 1.7 | 17.5 | 0.014 | 1.7 |
| 7 | full | 15.8 | 1.64 | 0.011 | 739 | 1.4 | 16.2 | 0.016 | 2.0 |
| 8 | linear | 15.4 | 1.53 | 0.010 | 727 | 1.4 | 16.6 | 0.014 | 1.7 |
| 9 | linear | 15.5 | 1.70 | 0.012 | 694 | 1.6 | 16.3 | 0.022 | 1.7 |
| 10 | linear | 15.2 | 1.65 | 0.012 | 703 | 1.9 | 16.9 | 0.021 | 1.9 |
| 11 | full | 15.5 | 1.70 | 0.012 | 690 | 1.8 | 16.9 | 0.017 | 1.8 |
| 12 | linear | 15.2 | 1.43 | 0.009 | 699 | 1.5 | 16.2 | 0.019 | 1.6 |
| 13 | linear | 15.3 | 1.33 | 0.008 | 692 | 1.8 | 16.8 | 0.015 | 1.5 |
| 14 | linear | 15.1 | 1.39 | 0.008 | 750 | 1.8 | 16.9 | 0.011 | 2.7 |
| 15 | full | 16.4 | 1.47 | 0.008 | 766 | 1.8 | 20.0 | 0.011 | 2.9 |
| 16 | linear | 16.4 | 1.23 | 0.006 | 766 | 1.7 | 16.0 | 0.014 | 2.4 |
| 17 | linear | 16.4 | 1.17 | 0.005 | 752 | 1.3 | 15.5 | 0.009 | 2.7 |
| 18 | linear | 16.4 | 1.24 | 0.006 | 742 | 1.3 | 14.6 | 0.011 | 1.9 |
| 19 | full | 17.3 | 1.32 | 0.006 | 790 | 1.5 | 24.2 | 0.007 | 1.8 |
| 20 | linear | 17.5 | 1.23 | 0.005 | 790 | 1.4 | 15.2 | 0.009 | 2.1 |
| 21 | linear | 17.7 | 1.22 | 0.005 | 788 | 1.6 | 16.8 | 0.007 | 1.6 |
| 22 | linear | 17.5 | 1.28 | 0.005 | 786 | 1.5 | 17.7 | 0.008 | 1.7 |
| 23 | full | 16.4 | 1.33 | 0.007 | 769 | 2.0 | 20.1 | 0.018 | 4.5 |
