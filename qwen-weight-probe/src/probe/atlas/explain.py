"""Explanation stage: show Claude each neuron's recorded examples and keep what it says each pole responds to."""
from __future__ import annotations

import bisect
import math
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from probe.atlas import prompts
from probe.atlas.batches import process
from probe.atlas.config import Config, neuron_id, neuron_name
from probe.atlas.corpus import _load_arrays, _shard_sha
from probe.atlas.llm import LLMFailed, call_with_wait
from probe.atlas.recorder import _positions_at_or_above
from probe.atlas.store import read_json

SHARD_CACHE = 64     # more than the corpus has, so a neuron's examples never push each other's shards out


@lru_cache(maxsize=2)    # one layer's recording and the summary
def _npz(path: Path, _mtime_ns: int) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files if not k.startswith("dn_")}


def _load(path: Path) -> dict[str, np.ndarray]:
    return _npz(path, path.stat().st_mtime_ns)


@lru_cache(maxsize=SHARD_CACHE)
def _shard(path: Path, sha256: str) -> tuple[np.ndarray, np.ndarray]:
    ids, offsets = _load_arrays(path)
    if _shard_sha(ids, offsets) != sha256:
        raise ValueError(f"checksum mismatch for {path.name}")
    return ids, offsets


def _doc_ids(cfg: Config, shards: list[dict], doc: int) -> np.ndarray:
    sh = shards[bisect.bisect_right([s["first_doc"] for s in shards], doc) - 1]
    ids, offsets = _shard(cfg.root / "corpus" / sh["file"], sh["sha256"])
    k = doc - sh["first_doc"]
    return ids[offsets[k]:offsets[k + 1]]


def _example(cfg: Config, shards: list[dict], doc: int, at: int, win: np.ndarray, centered: bool) -> dict:
    """An example's token ids and activations, aligned by window position; `ids` is None outside the document.

    `peak` is the window position to cut around: the recorded token for a pole's example (`centered`), and the strongest
    activation for a random one, whose recorded position is arbitrary. NaN means no activation is shown there (R13).
    """
    text = _doc_ids(cfg, shards, doc)
    first = at - cfg.window // 2
    ids = [int(text[p]) if 0 <= p < len(text) else None for p in range(first, first + cfg.window)]
    win = win.astype(np.float64)
    peak = cfg.window // 2
    strength = np.where(np.isfinite(win), np.abs(win), 0.0)
    if not centered and strength.max() > 0:
        peak = int(strength.argmax())
    return {"doc": doc, "tok": at, "ids": ids, "win": win.tolist(), "peak": peak}


def _reach(hist: np.ndarray, extreme: float, count: int) -> list[float]:
    """Percent of all recorded tokens at or above each of the prompt's fractions of the pole's extreme."""
    if extreme <= 0 or count <= 0:
        return [0.0] * len(prompts.REACH_FRACTIONS)
    return [100 * float(_positions_at_or_above(hist[None], np.array([f * extreme]))[0]) / count for f in prompts.REACH_FRACTIONS]


def neuron_record(cfg: Config, layer: int, idx: int) -> dict:
    """Everything the prompt shows about one neuron: its extremes, histogram summary and examples with their token ids."""
    state = _load(cfg.root / "record" / f"L{layer}.npz")
    summary = _load(cfg.root / "record" / "summary.npz")
    shards = read_json(cfg.root / "corpus" / "manifest.json")["shards"]

    def top(pole: str, n: int) -> list[dict]:
        docs, toks, wins = state[f"{pole}_doc"][idx], state[f"{pole}_tok"][idx], state[f"{pole}_win"][idx]
        return [_example(cfg, shards, int(docs[s]), int(toks[s]), wins[s], True) for s in np.flatnonzero(docs >= 0)[:n]]

    filled = np.flatnonzero(state["rnd_doc"] >= 0)[:prompts.N_RANDOM]
    pos_extreme, neg_extreme = float(summary[f"L{layer}.pos_extreme"][idx]), float(summary[f"L{layer}.neg_extreme"][idx])
    count = int(state["count"][idx])
    return {
        "nid": neuron_id(cfg, layer, idx), "layer": layer, "idx": idx,
        "pos_extreme": pos_extreme, "neg_extreme": neg_extreme, "dominant": int(summary[f"L{layer}.dominant"][idx]),
        "reach": {"positive": _reach(state["hist"][idx, 3], pos_extreme, count),      # regime 3: gate > 0, value > 0
                  "negative": _reach(state["hist"][idx, 2], neg_extreme, count)},     # regime 2: gate > 0, value < 0
        "positive": top("pos", prompts.N_POSITIVE), "negative": top("neg", prompts.N_NEGATIVE),
        "random": [_example(cfg, shards, int(state["rnd_doc"][s]), int(state["rnd_tok"][s]), state["rnd_win"][idx, s], False)
                   for s in filled],
    }


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer   # lazy: slow to import
    from probe.weights import snapshot_dir

    return AutoTokenizer.from_pretrained(snapshot_dir())


def _checked_pole(pole) -> dict:
    if not isinstance(pole, dict):
        raise LLMFailed("a pole is not an object")
    text, context, job = pole.get("explanation"), pole.get("context"), pole.get("job")
    if not isinstance(text, str) or not text.strip():
        raise LLMFailed("empty explanation")
    if context not in prompts.CONTEXTS or job not in prompts.JOBS:
        raise LLMFailed(f"invalid context or job: {context!r}, {job!r}")
    return {"explanation": text.strip(), "context": context, "job": job}


def _latest(rec: dict, record: dict, tok, cfg: Config) -> dict:
    """The viewer's payload (R10) for a neuron: its dominant pole's explanation and strongest example."""
    pole = "positive" if rec["dominant"] == 1 else "negative"
    shown = prompts.cut_example(rec[pole][0], cfg.llm_window) if rec[pole] else []
    groups = prompts.text_groups(shown, prompts.memo_decode(tok))
    return {"name": neuron_name(rec["layer"], rec["idx"]), "explanation": record[pole]["explanation"], "score": None,
            "tokens": [[text, a if math.isfinite(a) else 0.0] for text, a, _ in groups]}


def _stored_count(out: Path, nids: set[int]) -> int:
    found = set()
    for path in out.glob("batch_*.json"):
        rows = read_json(path)
        if isinstance(rows, list):
            found.update(r["nid"] for r in rows if isinstance(r, dict) and r.get("nid") in nids)
    return len(found)


def explain_neurons(cfg: Config, status, layer: int, idxs: Sequence[int], out: Path) -> list[dict]:
    """Explanation records for `idxs` of `layer`, in order. Finished batches in `out` are reused, so a restart pays for nothing twice."""
    out = Path(out)
    tok = _tokenizer()
    items = [{"nid": neuron_id(cfg, layer, int(i)), "idx": int(i)} for i in idxs]
    done = _stored_count(out, {x["nid"] for x in items})
    status.layer(layer, "running", done, len(items), None)

    def work(chunk: list[dict]) -> list[dict]:
        nonlocal done
        recs = [neuron_record(cfg, layer, x["idx"]) for x in chunk]
        ids = [f"n{n:02d}" for n in range(1, len(chunk) + 1)]
        prompt = prompts.explain_prompt([(i, prompts.render_neuron(r, tok, cfg)) for i, r in zip(ids, recs)])
        data = call_with_wait(cfg, status, prompts.EXPLAIN_SYSTEM, prompt, prompts.explain_schema(ids)).data
        if set(data) != set(ids):
            raise LLMFailed("response ids differ from the ids sent")
        records = []
        for i, rec in zip(ids, recs):
            if not isinstance(data[i], dict):
                raise LLMFailed("a neuron's entry is not an object")
            records.append({"nid": rec["nid"], "layer": layer, "idx": rec["idx"],
                            "positive": _checked_pole(data[i].get("positive")),
                            "negative": _checked_pole(data[i].get("negative"))})
        done += len(chunk)
        status.layer(layer, "running", done, len(items), None)
        status.latest(_latest(recs[-1], records[-1], tok, cfg))
        return records

    results = process(items, cfg.explain_batch, out, work, status=status)
    status.layer(layer, "running", len(items), len(items), None)
    return results
