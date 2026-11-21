# Atlas pilot (v1)

200 neurons, 200 failed.

| score | primary (n=200) | AC-matched (n=97) | AC |
|---|---|---|---|
| blind_maximum | 0.0% | 0.0% | 76% |
| near_miss | 0.0% | 0.0% | 11% |
| distant | 0.0% | 0.0% | ~1% |

Control (blind-maximum passages written for a different neuron, n=0): 0.0%. Good explanations should score far above this.

## Cost

- Tokens used: 685,547 input, 97,712 output, in 46 calls.
- Elapsed: 0.78 h, not counting usage-limit pauses.
- Projected full run (22,634 live neurons): 88,641,421 tokens and 88 h of explaining, testing and scoring, not counting usage-limit pauses.
- Compare the `/usage` figure from before and after the pilot to see what share of the Max allowance it took.

## Sample neurons

## Next step

If this looks right, approve the full run with `nix-shell --run 'python -m probe.atlas approve-pilot'`, then start `run` again.
