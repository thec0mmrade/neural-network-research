# Rules for every task in this plan
- Project root: /home/c0mmrade/Documents/claude/Neural_Network_Research/qwen-weight-probe. Work there. Read its CLAUDE.md first.
- This project is NOT a git repository. Never run `git init`, and skip every commit step in the templates. In your report, list every file you created or changed.
- Nix only: no pip, uv or venv. Run tests as: nix-shell --run 'OMP_NUM_THREADS=4 pytest <path> -v'
- One GPU job at a time. Never run an all-core CPU load (this machine crashed from one). Cap threads at 4 for GPU jobs and 8 for CPU work.
- Do not start the real atlas run. Do not change anything outside the files your brief names, plus tests/atlas/conftest.py for shared fixtures.
- Rulings that override the brief are in progress.md in this folder, section "Rulings". Follow those that mention your task.
- Do not spawn subagents.
