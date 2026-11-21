# qwen-weight-probe

This project analyzes the weights of Qwen3.5-0.8B and the "Sparks of Weight-Space Programming" Exp 8 weight edit. It tests the conclusions in `../ANALYSIS.md`.

- **Results:** start with `results/FINDINGS.md`.
- **Open work:** the map levels in `TODO.md`.
- **Running anything:** see `CLAUDE.md`. Everything runs in nix-shell; never use pip, uv or venv.

## Level 3 atlas: resuming the run

**Status (2026-11-21):** the full run is going. It started on 2026-11-20 after pilot v2 was approved.
- The corpus and recording stages are done.
- Pilot v2 passed: blind maximum 66.5%, against AC's 76%. The report is in `results/07_atlas/pilot/v2/report.md`.
- **Layer 15 is done:** 3,187 neurons, blind maximum 66.8%, near miss 5.2%, distant 0.2% (AC: 76 / 11 / ~1). Layer 19 is running.
- **Cost:** about 10,000 neurons per week of Max allowance, so the whole run (22,634 neurons) takes about 2–2.5 weeks of it. Scoring runs on the GPU and is free. The earlier 7–10 week estimate predates the refusal fix R39.
- The progress counter counts 3 steps per neuron (explain, test, score), and it doesn't move during a layer's test step.

### Easiest: ask Claude Code

In a Claude Code session in this folder, say "resume the level 3 atlas". The pilot is already approved, so after a crash or a pause Claude only restarts the run in the background (step 2 below).

### Doing it yourself

Run these from `qwen-weight-probe/`:

```sh
# 1. Approve the v2 pilot (only needed once)
nix-shell --run 'python -m probe.atlas approve-pilot'

# 2. Start the run in the background, so closing the terminal won't stop it.
#    It is one long line: copy all of it. Splitting it across lines breaks it.
env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN setsid -f nohup nix-shell --run 'OMP_NUM_THREADS=4 RAYON_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m probe.atlas run' > atlas_run.log 2>&1 < /dev/null

# 3. Watch progress (optional; safe to close at any time)
nix-shell --run 'python -m probe.atlas_watch'
```

Before step 2, check that `claude auth status` shows the Max login. Step 2 removes any API keys from the environment, so the run can't fall back to paid API billing.

### What happens next

- The run skips everything already done (corpus, recording, pilot) and starts on layer 15, then layer 19, then layers 18, 16, 17, 14 and 20.
- When a Max usage limit is hit, it pauses and continues on its own after the reset.
- After a crash or power loss, run step 2 again and it resumes where it stopped. Don't repeat step 1.
- Other Claude use on the account hits limits sooner while the run is going, because the run uses each new allowance as soon as it opens.

### Stopping

- **Pause:** run `pkill -TERM -f "[p]ython -m probe.atlas run"` on its own, not in the same command as the relaunch line (pkill would match that shell too). Nothing is lost, and step 2 resumes it.
- **Stop after a layer:** each finished layer is saved and usable on its own, so stopping after L15 and L19 keeps their results.

### Changing the prompts

Any change to the prompts needs a new `prompt_version` in `src/probe/atlas/config.py` and a new pilot. The new pilot also has to be approved before the full run.
