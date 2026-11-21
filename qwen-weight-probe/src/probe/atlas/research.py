"""The research question: do p50c's edited neurons and the top u-writers have coherent explanations that match u's meaning?

Studied neurons are compared with random live controls from the same layers. The layer run already explained every live neuron
and tested its dominant pole, so only the other pole's tests are new. A separate blind session then rates every explanation.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

from probe.atlas import prompts
from probe.atlas.batches import process
from probe.atlas.config import Config, neuron_id
from probe.atlas.evaluate import score_tests, write_tests
from probe.atlas.llm import call_with_wait
from probe.atlas.pilot import SCORE_BATCH, _pole_name, load_summary, prompt_parts
from probe.atlas.record import P50C_LAYERS, edited_neurons
from probe.atlas.runner import _layer_dir
from probe.atlas.store import atomic_write_json, read_json
from probe.weights import RESULTS, Checkpoint, layer_tensor

RESEARCH_LAYERS = P50C_LAYERS
TOP_WRITERS = 64
RATE_BATCH = 40
PERMUTATIONS = 10_000
THEMES = ("move_on", "reconsider")
GROUPS = ("study", "control")
JOB_LABELS = (*prompts.JOBS, "unknown")     # "unknown": the explanation itself failed
STEPS = 3     # writing tests, scoring them, rating


def research_dir(cfg: Config) -> Path:
    return cfg.root / "research" / cfg.prompt_version


def study_neurons(cfg: Config) -> dict[int, np.ndarray]:
    """Per layer, the sorted union of p50c's edited columns and the top writers to u through the base `down_proj`."""
    edited = edited_neurons("p50c")
    base = Checkpoint()
    found = {}
    for layer in RESEARCH_LAYERS:
        u = np.load(RESULTS / f"u{layer}_p50c.npy")
        W = base.get(layer_tensor(layer, "down"))
        found[layer] = np.union1d(edited[layer], np.argsort(-np.abs(u @ W))[:TOP_WRITERS])
    return found


def control_neurons(cfg: Config, summary, study: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Per layer, as many random live neurons as the study set holds, none of them studied."""
    found = {}
    for layer, chosen in study.items():
        pool = np.setdiff1d(np.flatnonzero(summary[f"L{layer}.live"]), chosen)
        if len(pool) < len(chosen):
            raise ValueError(f"layer {layer} has {len(pool)} live neurons outside the study set, but needs {len(chosen)} controls")
        found[layer] = np.sort(np.random.default_rng([cfg.seed, 13, layer]).choice(pool, len(chosen), replace=False))
    return found


def layer_explanations(cfg: Config, layer: int) -> dict[int, dict]:
    """The layer run's explanation record per neuron id. A stored success wins over a stored failure."""
    found: dict[int, dict] = {}
    for path in sorted((_layer_dir(cfg, layer) / "explain").glob("batch_*.json")):
        for r in read_json(path, []):
            if "failed" not in r or r["nid"] not in found:
                found[r["nid"]] = r
    return found


def layer_scores(cfg: Config, layer: int) -> dict[int, dict] | None:
    """The layer run's scored record per neuron id, or None if the layer has not finished."""
    stored = read_json(_layer_dir(cfg, layer) / "scores.json")
    return None if stored is None else {r["nid"]: r for r in stored["records"]}


def rate_blind(cfg: Config, status, study: list[dict], control: list[dict], out: Path) -> list[dict]:
    """Rate every item's explanation against the two themes. Returns the items with their group, id and ratings or `failed`.

    Items are shuffled together and sent under anonymous ids, so the rater sees only explanation texts. The assignment is stored
    in `key.json`: stored ratings belong to those ids, so a different item list is refused rather than matched to them.
    """
    out = Path(out)
    pooled = [{**x, "group": "study"} for x in study] + [{**x, "group": "control"} for x in control]
    order = np.random.default_rng([cfg.seed, 13]).permutation(len(pooled))
    items = [{**pooled[int(at)], "id": f"r{n:03d}"} for n, at in enumerate(order, 1)]

    key = [{k: x[k] for k in ("id", "group", "nid", "pole", "explanation")} for x in items]
    stored = read_json(out / "key.json")
    if stored is not None and stored != key:
        raise ValueError(f"{out} holds ratings for a different list of explanations")
    atomic_write_json(out / "key.json", key)

    def work(chunk: list[dict]) -> list[dict]:
        ids = [x["id"] for x in chunk]
        prompt = prompts.rate_prompt([(x["id"], x) for x in chunk])
        data = call_with_wait(cfg, status, prompts.RATE_SYSTEM, prompt, prompts.rate_schema(ids)).data
        prompts.validate_ratings(data, ids)
        return data["ratings"]

    return [{**x, **r} for x, r in zip(items, process(items, RATE_BATCH, out, work, key="id", status=status))]


def permutation_test(study: list[float], control: list[float], seed, study_layers=None, control_layers=None) -> dict:
    """Means of the two groups and a two-sided permutation test of their difference.

    With layers given, group labels are shuffled only within each layer, so a difference between layers cannot pass for a group effect.
    """
    a, b = np.asarray(study, float), np.asarray(control, float)
    sizes = {"n_study": len(a), "n_control": len(b)}
    if not len(a) or not len(b):
        return {"study": None, "control": None, "difference": None, "p": None, **sizes}
    observed = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    labels = np.concatenate([np.ones(len(a), bool), np.zeros(len(b), bool)])
    layers = np.zeros(len(pooled), int) if study_layers is None else np.concatenate([study_layers, control_layers])
    strata = [np.flatnonzero(layers == layer) for layer in np.unique(layers)]
    rng = np.random.default_rng(seed)
    mixed = labels.copy()
    extreme = 0
    for _ in range(PERMUTATIONS):
        for stratum in strata:
            mixed[stratum] = rng.permutation(labels[stratum])
        extreme += abs(pooled[mixed].mean() - pooled[~mixed].mean()) >= abs(observed) - 1e-12     # rounding must not break ties
    return {"study": float(a.mean()), "control": float(b.mean()), "difference": observed,
            "p": (1 + extreme) / (1 + PERMUTATIONS), **sizes}


def _pole_rows(cfg: Config, summary, groups: dict[str, dict[int, np.ndarray]]) -> tuple[list[dict], dict]:
    """One row per pole of every live studied or control neuron, with the layer run's results, and the dead neurons left out.

    A non-dominant pole that never fired is marked `untestable`: no passage can reach an extreme of zero, so it gets no Claude call.
    """
    rows, dead = [], {g: {str(l): 0 for l in RESEARCH_LAYERS} for g in groups}
    for layer in RESEARCH_LAYERS:
        scored = layer_scores(cfg, layer)
        if scored is None:
            raise RuntimeError(f"L{layer} has no scores.json: the research stage needs the finished layer run")
        explained = layer_explanations(cfg, layer)
        for group, chosen in groups.items():
            for idx in map(int, chosen[layer]):
                if not summary[f"L{layer}.live"][idx]:
                    dead[group][str(layer)] += 1
                    continue
                nid = neuron_id(cfg, layer, idx)
                rec = explained.get(nid, {"failed": "no explanation stored"})
                main = int(summary[f"L{layer}.dominant"][idx])
                for pole in (main, -main):
                    row = {"group": group, "nid": nid, "layer": layer, "idx": idx, "pole": pole, "dominant": pole == main,
                           "explanation": None, "context": None, "job": None}
                    if "failed" not in rec:
                        row.update(rec[_pole_name(pole)])
                    if pole != main and summary[f"L{layer}.{'pos' if pole == 1 else 'neg'}_extreme"][idx] <= 0:
                        rows.append({**row, "untestable": True})
                        continue
                    if "failed" in rec:
                        rows.append({**row, "failed": rec["failed"]})
                        continue
                    if pole == main:
                        found = scored.get(nid, {"failed": "no test record"})
                        row.update({"failed": found["failed"]} if "failed" in found else {c: found[c] for c in prompts.TEST_CATEGORIES})
                    rows.append(row)
    return rows, dead


def _test_other_poles(cfg: Config, base, status, rows: list[dict], summary, out: Path) -> None:
    """Write and score the tests of each non-dominant pole, in place. Each layer and pole has its own folders (R23)."""
    pending = {(l, p): [r for r in rows if not r["dominant"] and "failed" not in r and "untestable" not in r
                        and r["layer"] == l and r["pole"] == p]
               for l in RESEARCH_LAYERS for p in (1, -1)}
    folder = lambda layer, pole: out / f"L{layer}" / _pole_name(pole)
    written = {(l, p): write_tests(cfg, status, [{k: r[k] for k in ("nid", "pole", "job", "explanation", "context")} for r in todo],
                                   folder(l, p) / "tests")
               for (l, p), todo in pending.items()}
    status.stage("research", 1, STEPS, "steps")

    for (layer, pole), tests in written.items():
        def work(chunk: list[dict]) -> list[dict]:
            return chunk if all("failed" in t for t in chunk) else score_tests(cfg, *base.get(), layer, chunk, summary)

        for row, result in zip(pending[layer, pole], process(tests, SCORE_BATCH, folder(layer, pole) / "scores", work, status=status)):
            row.update({"failed": result["failed"]} if "failed" in result else {c: result[c] for c in prompts.TEST_CATEGORIES})
    base.free()
    status.stage("research", 2, STEPS, "steps")


def _rate_rows(cfg: Config, status, rows: list[dict], out: Path) -> None:
    """Rate every testable pole that has an explanation, in place."""
    items = {g: [{k: r[k] for k in ("nid", "pole", "explanation")} for r in rows
                 if r["group"] == g and r["explanation"] is not None and "untestable" not in r]
             for g in GROUPS}
    rated = {(r["nid"], r["pole"]): r for r in rate_blind(cfg, status, items["study"], items["control"], out / "ratings")}
    for row in rows:
        rating = rated.get((row["nid"], row["pole"]))
        if rating is not None:
            row.update({"rating_failed": rating["failed"]} if "failed" in rating else {t: rating[t] for t in THEMES})


def _neuron_values(rows: list[dict], value: Callable[[dict], float | None]) -> dict[str, dict[str, list]]:
    """Per group, each neuron's mean `value` over its poles and its layer. Poles without a value are skipped, as are neurons with none."""
    poles: dict[tuple[str, int, int], list[float]] = {}
    for r in rows:
        v = value(r)
        if v is not None:
            poles.setdefault((r["group"], r["nid"], r["layer"]), []).append(v)
    return {g: {"values": [float(np.mean(v)) for (group, _, _), v in poles.items() if group == g],
                "layers": [layer for (group, _, layer) in poles if group == g]} for g in GROUPS}


def _share(category: str) -> Callable[[dict], float | None]:
    """A pole's share of hits in `category`. A failed pole scores zero, as in the primary score; an untestable one has no score."""
    return lambda r: None if "untestable" in r else 0.0 if "failed" in r else r[category] / prompts.TESTS_PER_CATEGORY


def _specificity(r: dict) -> float | None:
    """How much more often a pole's blind-maximum passages fire it than its near misses do."""
    blind, near = _share("blind_maximum")(r), _share("near_miss")(r)
    return None if blind is None else blind - near


def _test(cfg: Config, rows: list[dict], value: Callable[[dict], float | None], n: int) -> dict:
    values = _neuron_values(rows, value)
    study, control = values["study"], values["control"]
    return permutation_test(study["values"], control["values"], [cfg.seed, 13, n], study["layers"], control["layers"])


def _scope(cfg: Config, rows: list[dict], scope: str, first_seed: int) -> dict:
    """The rating and test-score comparisons over `rows`."""
    categories = {c: _neuron_values(rows, _share(c)) for c in prompts.TEST_CATEGORIES}
    return {"scope": scope,
            "ratings": {t: _test(cfg, rows, lambda r, t=t: r.get(t), first_seed + n) for n, t in enumerate(THEMES)},
            "blind_maximum": _test(cfg, rows, _share("blind_maximum"), first_seed + len(THEMES)),
            "scores": {g: {c: float(np.mean(v[g]["values"])) if v[g]["values"] else None for c, v in categories.items()} for g in GROUPS}}


def compare(cfg: Config, rows: list[dict]) -> dict:
    """The comparison of studied and control neurons, with the neuron as the unit of every test.

    The dominant pole is the primary test. The both-pole average is secondary: a neuron with a silent pole averages only its stronger
    pole, so groups that differ in how many poles are silent would differ in that average even if their neurons were alike.
    """
    dominant = [r for r in rows if r["dominant"]]
    jobs = {g: Counter(r["job"] or "unknown" for r in rows if r["group"] == g and "untestable" not in r) for g in GROUPS}
    return {"primary": {**_scope(cfg, dominant, "dominant pole", 0), "specificity": _test(cfg, dominant, _specificity, 10)},
            "secondary": _scope(cfg, rows, "both poles", 20),
            "job_labels": {g: {j: jobs[g][j] for j in JOB_LABELS} for g in GROUPS}}


def _failures(rows: list[dict], group: str) -> dict:
    mine = [r for r in rows if r["group"] == group and "untestable" not in r]
    return {"explanation": sum(r["explanation"] is None for r in mine),
            "tests": sum("failed" in r and r["explanation"] is not None for r in mine),
            "ratings": sum("rating_failed" in r for r in mine)}


def run_research(cfg: Config, base, status) -> dict:
    """Answer the research question and write `research.json`, last. The model is held only while scoring."""
    out = research_dir(cfg)
    stored = read_json(out / "research.json")
    if stored is not None:
        return stored

    base.free()     # usage-limit waits during the Claude calls can last days
    summary = load_summary(cfg)
    study = study_neurons(cfg)
    groups = {"study": study, "control": control_neurons(cfg, summary, study)}
    rows, dead = _pole_rows(cfg, summary, groups)
    _test_other_poles(cfg, base, status, rows, summary, out)
    _rate_rows(cfg, status, rows, out)

    result = {**prompt_parts(cfg), "seed": cfg.seed, "permutations": PERMUTATIONS, "dead": dead,
              "neurons": {g: {str(l): len({r["nid"] for r in rows if r["group"] == g and r["layer"] == l}) for l in RESEARCH_LAYERS}
                          for g in GROUPS},
              "failed": {g: _failures(rows, g) for g in GROUPS},
              "untestable_poles": {g: sum(r["group"] == g and "untestable" in r for r in rows) for g in GROUPS},
              "comparison": compare(cfg, rows), "rows": rows}
    atomic_write_json(out / "research.json", result)
    status.stage("research", STEPS, STEPS, "steps")
    return result
