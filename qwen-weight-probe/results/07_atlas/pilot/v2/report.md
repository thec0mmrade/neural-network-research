# Atlas pilot (v2)

200 neurons, 0 failed.

Models: explanations by `claude-opus-5-5`, test passages by `claude-opus-5-5` (effort medium).

| score | primary (n=200) | AC-matched (n=96) | AC |
|---|---|---|---|
| blind_maximum | 66.5% | 68.1% | 76% |
| near_miss | 3.4% | 3.2% | 11% |
| distant | 0.1% | 0.3% | ~1% |

Reasoning-context poles (n=18, included in primary above) were tested on plain text, outside the setting they were described in (R38): blind_maximum 50.6%, near_miss 3.9%, distant 0.0%.

Control (blind-maximum passages written for a different neuron, n=200): 0.9%. Good explanations should score far above this.

## Cost

- Tokens used: 1,248,954 input, 439,932 output, in 130 calls.
- Elapsed: 1.50 h, not counting usage-limit pauses.
- Projected full run (22,634 live neurons): 191,131,229 tokens and 169 h of explaining, testing and scoring, not counting usage-limit pauses.
- Compare the `/usage` figure from before and after the pilot to see what share of the Max allowance it took.

## Sample neurons

### L14.mlp.1019 (negative pole, one job)

Fires on website empty-search or no-results boilerplate, peaking on 'matching'/'match' and 'for your'/'your'/'the' just before 'search', 'criteria', 'selection' or 'filters'. Examples are 'No products were found matching your selection' and 'did not find any results for your search'. It needs the empty-result message context; other uses of 'matching' or 'your search' should not fire it strongly.

blind_maximum 10/10, near_miss 1/10, distant 0/10

### L16.mlp.706 (positive pole, one job)

Fires on the digits of technical-standard designation numbers that come right after a standards-body prefix such as EN, ISO, AS, BS, API, SAE, IEC, ACI, ASCE, IPC or ASME/ANSI B. The first digit after the prefix (often after a space) fires strongest, and later digits, hyphens and part numbers fire more weakly; the prefix tokens themselves activate a little. The context that decides it is "in accordance with / meets / complies with [standard] [number]", so plain numbers without a standards prefix do not fire.

blind_maximum 10/10, near_miss 2/10, distant 0/10

### L16.mlp.1487 (positive pole, one job)

Percent-to-decimal conversion in math word-problem solutions. It fires on the leading "0" and "." of a decimal rate written as 0.XX (e.g. "= 0.20 ×", "- 0.04", "(1 - 0.05)"). It also fires on the phrase "as a decimal" / "rate (decimal form)" / "converted to a decimal". The context must be a percentage being turned into a multiplier, usually in step-by-step solutions; plain decimals with no percentage nearby should not fire it much.

blind_maximum 10/10, near_miss 0/10, distant 0/10

### L17.mlp.433 (positive pole, one job)

Fires on the word piece "is"/"Is" when it begins a longer word rather than serving as the verb, i.e. the prefixes iso-, isch-, isl-, isop-, as in is|chemic, is|oelectronic, is|oxazoline, is|let, is|ochronous, is|ocyanate, is|olated, is|opropyl. What decides it is context: "is" lands where the verb cannot fit grammatically, such as after an adjective, article, hyphen or noun modifier ("acute is", "a beautiful is", "Mobile Is", "broad-spectrum is"), often in technical, medical or chemical text. A real verb "is", even sentence-initial "Is it...", fires only weakly (about 40%), and the following pieces (chem, oe, op) carry only small activations.

blind_maximum 9/10, near_miss 0/10, distant 0/10

### L17.mlp.2706 (positive pole, one job)

Fires on phrases that open the first step of a process: 'The journey/process begins', 'It all begins', 'Start by...', 'a great place to start', 'We start off by'. The peak comes on the subject noun ('journey', 'process', 'It') or on 'Start' or 'place'. It favors step-by-step or how-to lists, including the bullet that opens a drafting step in reasoning outlines, so the 'beginning of a procedure' meaning decides.

blind_maximum 3/10, near_miss 0/10, distant 0/10

### L18.mlp.1439 (negative pole, one job)

Fires in step-by-step math solutions on the words that set up what must be found or identified: 'find', 'Identify', 'for some', 'specify', 'the specific', and the bullet or start token that opens a planning step. The context that decides is a model's structured reasoning over a math problem; ordinary math text without that framing barely fires.

blind_maximum 0/10, near_miss 0/10, distant 0/10

### L18.mlp.3479 (negative pole, one job)

Fires strongly on the copula ("are", "was", "is") or an adverb like "sadly" when the text predicts that something has ended or no longer exists, as in "Those days are long gone", "she was gone", "no longer with us", "is no more" and "are definitely over". Earlier text has to set up an old state of affairs ("Time was...", "Those days", "when you needed...") that is now being dismissed. The peak falls on the copula itself, with weaker spill onto the subject ("days", "you", "she") and onto "long", "gone", "no longer" and "far behind". The same copula in ordinary statements does not fire it.

blind_maximum 9/10, near_miss 0/10, distant 0/10

### L19.mlp.2 (negative pole, one job)

Fires on operators and closing brackets in the middle of inline LaTeX expressions inside math reasoning. Typical tokens are the '=' in derivations like '$S - S/2 = S/2$', the '}' in '\mathbf{a}', and the minus sign in '$-A/B$'. A few weaker hits fall on prose tokens ('requirements', a comma, Chinese 及其), so the math context is what decides.

blind_maximum 8/10, near_miss 5/10, distant 0/10

### L20.mlp.2028 (positive pole, one job)

Japanese comparison constructions. It fires on the particle に or と that marks what something is compared with, just before 比べ(て/る), 似た, 相当する, 同じく, 取って代わり, 換算 or ない ("Xに比べ", "他のビューアにない"). The Japanese comparison frame decides.

blind_maximum 2/10, near_miss 0/10, distant 0/10

### L20.mlp.2273 (negative pole, one job)

Fires on commas, "and"/"e"/"dan" and introducers like "misalnya", "including" or "de" inside enumerations of simple molecules or elements, such as gas and atmosphere composition lists (CO2, water, nitrogen, oxygen, H2O, OH). It works in English, Portuguese, Indonesian and Hindi. The list separators carry the strongest activation, and the species names are weaker. The list-of-chemical-species context decides, so commas in ordinary lists do not fire it.

blind_maximum 10/10, near_miss 1/10, distant 0/10

## Next step

If this looks right, approve the full run with `nix-shell --run 'python -m probe.atlas approve-pilot'`, then start `run` again.
