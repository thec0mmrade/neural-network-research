"""Test-only launcher: runs the real atlas command line on a tiny corpus under a temporary root, with a fake Claude and local web text.

Usage: rehearsal_cli.py ROOT {run,approve-pilot,status} [FAKE_MODE]

Nothing in `src/probe/atlas` knows this file exists, and no environment variable or flag reaches it, so a normal
`python -m probe.atlas` can never enter rehearsal mode. The model, tokenizer, GPU, recording and scoring are the real ones.
"""
from __future__ import annotations

import dataclasses
import functools
import random
import stat
import sys
import time
from pathlib import Path

import numpy as np

from probe.atlas import __main__ as cli     # first: it sets the thread caps before the heavy libraries load
from probe.atlas import corpus, pilot, report, research, thinking
from probe.atlas.config import Config
from probe.weights import RESULTS

FAKE = Path(__file__).parent / "fake_claude.py"
MATH_DOCS = 60              # the practice sets hold about 3M tokens; the tiny corpus keeps this many documents of them
NEW_TOKENS = 256            # per generated thinking document
READ_DELAY_S = 0.01         # per local web document, so that a kill can land inside the corpus stage
LIVE_OTHER_LAYERS = 8       # neurons kept live outside the research layers
EDITED_STUDIED = 6          # p50c-edited neurons studied per research layer
TOP_WRITERS = 4
WORDS = {"cmn_Hani": "我们 今天 学习 数学 问题 因为 所以 答案 世界 城市", "ind_Latn": "kami belajar hari ini tentang kota dunia dan masalah",
         "spa_Latn": "hoy estudiamos la ciudad y el mundo porque el problema", "rus_Cyrl": "сегодня мы изучаем город и мир потому что задача",
         "arb_Arab": "اليوم ندرس المدينة والعالم لأن المسألة صعبة جدا", "fra_Latn": "nous étudions la ville et le monde parce que le problème",
         "deu_Latn": "heute lernen wir die Stadt und die Welt weil das Problem", "jpn_Jpan": "今日 私たち は 都市 と 世界 を 学ぶ 問題 答え",
         "hin_Deva": "आज हम शहर और दुनिया के बारे में पढ़ते हैं", "por_Latn": "hoje estudamos a cidade e o mundo porque o problema"}
ENGLISH = ("the survey of coastal towns found that river traffic grew after the new bridge opened while older markets moved "
           "inland and local councils argued about taxes schools roads and the price of grain during a long dry summer").split()


def rehearsal_config(root: Path, fake_mode: str = "atlas") -> Config:
    """`Config.tiny` made smaller still, with the fake Claude as `claude_bin`."""
    root = Path(root).resolve()
    if root == RESULTS or RESULTS in root.parents:
        sys.exit(f"refusing to rehearse inside {RESULTS}")
    wrapper = root.parent / "fake_claude.sh"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} {FAKE} {fake_mode} {root.parent / "fake_state"} "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return dataclasses.replace(Config.tiny(root), total_tokens=90_000, multilingual_tokens=10_000, thinking_tokens=30_000,
                               shard_tokens=10_000, p50c_tokens=8_000, claude_bin=str(wrapper))


def local_docs(repo: str, path: str, revision: str):
    """Endless, reproducible local stand-in for a remote parquet file."""
    lang = next((name for name in WORDS if name in path), None)
    words = WORDS[lang].split() if lang else ENGLISH
    rng = random.Random(path)
    while True:
        time.sleep(READ_DELAY_S)
        yield " ".join(rng.choices(words, k=rng.randint(60, 400)))


def _kept_live(cfg: Config, summary: dict) -> dict:
    """The summary with few live neurons, so the layer run stays small. Every studied neuron and its controls stay live."""
    study = research.study_neurons(cfg)
    out = dict(summary)
    for layer in cfg.layers:
        live = np.flatnonzero(summary[f"L{layer}.live"])
        rng = np.random.default_rng([cfg.seed, 99, layer])
        if layer in study:
            chosen = np.intersect1d(study[layer], live)
            pool = np.setdiff1d(live, study[layer])
            kept = np.union1d(chosen, rng.choice(pool, len(study[layer]) + 4, replace=False))
        else:
            kept = rng.choice(live, LIVE_OTHER_LAYERS, replace=False)
        out[f"L{layer}.live"] = np.isin(np.arange(cfg.n_neurons), kept)
    return out


def install(cfg: Config) -> None:
    """Swap in the local data sources and the small limits. Only this process is affected."""
    real_math, real_summary, real_edited = corpus.math_docs, pilot.load_summary, research.edited_neurons
    corpus.math_docs = lambda: real_math()[:MATH_DOCS]
    corpus.build_sources = functools.partial(corpus.build_sources, reader=local_docs, revision_of=lambda repo: "local-rehearsal")
    thinking.MAX_NEW_TOKENS = NEW_TOKENS
    research.TOP_WRITERS = TOP_WRITERS
    research.edited_neurons = lambda run="p50c": {l: v[:EDITED_STUDIED] for l, v in real_edited(run).items()}

    @functools.lru_cache(maxsize=1)
    def kept(path_mtime):
        return _kept_live(cfg, real_summary(cfg))

    def load_summary(_cfg: Config) -> dict:
        return kept((cfg.root / "record" / "summary.npz").stat().st_mtime_ns)

    pilot.load_summary = research.load_summary = report.load_summary = load_summary
    cli.Config = lambda: cfg


def main(argv: list[str]) -> int:
    root, command = Path(argv[0]), argv[1]
    cfg = rehearsal_config(root, *argv[2:3])
    install(cfg)
    return cli.main([command])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
