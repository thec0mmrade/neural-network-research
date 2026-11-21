"""Reasoning text from the unedited model, saved batch by batch so a crash loses at most one batch."""
from __future__ import annotations

import random

from probe.atlas.config import Config
from probe.atlas.corpus import _jsonl_rows, save_source
from probe.atlas.store import atomic_write_json, read_json
from probe.weights import LOGS

BATCH = 32
MAX_NEW_TOKENS = 4000
PROBLEM_FILES = ("gsm8k_train.jsonl", "math_train.jsonl")


def thinking_problems(seed: int) -> list[str]:
    """GSM8K and MATH training problems, shuffled deterministically."""
    problems = [row["problem"] for name in PROBLEM_FILES
                for row in _jsonl_rows(LOGS / "practice" / name)]
    random.Random(seed).shuffle(problems)
    return problems


def generate_thinking(cfg: Config, model, tok, status, gen=None, prompts=None, problems=None) -> None:
    """Generate until `thinking_tokens` is reached, then save the `thinking` source.

    Batch b covers problems [32b, 32b+32) with seed 32b; finished batch files are reused on resume.
    `gen`, `prompts` and `problems` default to the real ones; tests inject fakes (the real
    `probe.model` import loads torch, so it stays out of module scope).
    """
    if gen is None or prompts is None:
        from probe import model as pm
        gen = gen or pm.generate
        prompts = prompts or pm.build_prompts
    if problems is None:
        problems = thinking_problems(cfg.seed)

    folder = cfg.root / "raw" / "thinking"
    docs: list[list[int]] = []
    total = 0
    for b in range(0, len(problems), BATCH):
        path = folder / f"batch_{b // BATCH:05d}.json"
        saved = read_json(path)
        if saved is None:
            batch = problems[b:b + BATCH]
            outs, _tps = gen(model, tok, batch, MAX_NEW_TOKENS, seed=b, batch_size=BATCH)
            batch_docs = [(tok(p, add_special_tokens=False).input_ids + o["ids"])[:cfg.doc_cap]
                          for p, o in zip(prompts(tok, batch), outs)]
            atomic_write_json(path, {"docs": batch_docs, "seed": b})
        else:
            batch_docs = saved["docs"]
        docs += batch_docs
        total += sum(map(len, batch_docs))
        status.stage("thinking", min(total, cfg.thinking_tokens), cfg.thinking_tokens, "tokens")
        if total >= cfg.thinking_tokens:
            break
    save_source(cfg, "thinking", docs, {"problems": "gsm8k_train+math_train", "seed": cfg.seed,
                                        "max_new_tokens": MAX_NEW_TOKENS})
