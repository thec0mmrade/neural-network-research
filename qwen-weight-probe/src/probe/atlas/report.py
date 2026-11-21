"""The atlas's final report: `07_atlas.md` beside the atlas folder, and one complete record per neuron in `atlas_neurons.jsonl`."""
from __future__ import annotations

import bisect
import json
from collections import Counter
from functools import lru_cache
from itertools import accumulate
from pathlib import Path

import numpy as np

from probe.atlas import prompts
from probe.atlas.config import Config, neuron_id, neuron_name
from probe.atlas.evaluate import summarize
from probe.atlas.pilot import AC_FIGURES, _percent, _pole_name, load_summary
from probe.atlas.recorder import FIRE_FRACTION
from probe.atlas.research import JOB_LABELS, THEMES, layer_explanations, layer_scores, research_dir
from probe.atlas.store import atomic_write_bytes, read_json
from probe.weights import FULL_ATTN, N_LAYERS

EDGES_SHOWN = 8
WIRING_NEURONS = 3584     # the level-2 component table, computed here because importing probe.wiring changes the process's thread settings

LIMITS = (
    "AC studied a different kind of model, so the score comparison is a benchmark, not a like-for-like result.",
    "Only the dominant pole of each neuron is tested, except for the research-question neurons, which are tested on both poles.",
    "Test passages always run as plain text, so poles specific to step-by-step reasoning are tested outside their setting (R38).",
    "Test passages are at most 128 tokens, so neurons that build up over long contexts, such as loop-related ones, may be under-scored.",
    "Coverage depends on the corpus: a neuron that needs text outside it can look dead or be misdescribed.",
    "The test writer is the same model family as the explainer.",
)


def _heads(layer: int) -> int:
    return 8 if layer in FULL_ATTN else 16


_STARTS = [0, *accumulate(_heads(l) + WIRING_NEURONS for l in range(N_LAYERS))]


def component_id(layer: int, idx: int) -> int:
    """The level-2 component id of MLP neuron `idx`: within a layer, the attention heads come first."""
    return _STARTS[layer] + _heads(layer) + idx


def component_name(cid: int) -> str:
    layer = bisect.bisect_right(_STARTS, cid) - 1
    offset = cid - _STARTS[layer]
    if offset >= _heads(layer):
        return neuron_name(layer, offset - _heads(layer))
    return f"L{layer}.{'attn' if layer in FULL_ATTN else 'lin'}.h{offset}"


def wiring_path(cfg: Config) -> Path:
    return cfg.root.parent / "06_wiring" / "edges_raw.npz"


@lru_cache(maxsize=1)
def _edges(path: Path, _mtime_ns: int) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in ("out_id", "out_score", "out_sign", "in_id", "in_score", "in_sign")}


def wiring_join(cfg: Config, layer: int, idx: int, k: int = EDGES_SHOWN) -> dict:
    """The neuron's `k` strongest outgoing and incoming level-2 edges. Stored edges are sorted by score, and unused slots have id -1."""
    path = wiring_path(cfg)
    edges = _edges(path, path.stat().st_mtime_ns)
    row = component_id(layer, idx)
    return {side: [{"id": int(j), "name": component_name(int(j)), "score": round(float(s), 3), "sign": int(g)}
                   for j, s, g in zip(edges[f"{side}_id"][row][:k], edges[f"{side}_score"][row][:k], edges[f"{side}_sign"][row][:k]) if j >= 0]
            for side in ("out", "in")}


def _neuron_rows(cfg: Config, summary, scores: dict[int, dict], explained: dict[int, dict], research: dict | None) -> list[dict]:
    """One record per in-scope neuron, in neuron id order."""
    studied: dict[int, list[dict]] = {}
    for r in (research or {}).get("rows", []):
        studied.setdefault(r["nid"], []).append(r)
    has_wiring = wiring_path(cfg).exists()

    rows = []
    for layer in cfg.layers:
        live, main = summary[f"L{layer}.live"], summary[f"L{layer}.dominant"]
        for idx in range(cfg.n_neurons):
            nid = neuron_id(cfg, layer, idx)
            rec = explained[layer].get(nid, {})
            scored = None if layer not in scores else scores[layer].get(nid, {"failed": "no test record"})
            failed = None
            if live[idx] and (scored is None or "failed" in scored):
                failed = "layer not scored" if scored is None else scored["failed"] or "no reason recorded"
            state = "dead" if not live[idx] else "failed" if failed else "live"
            poles = studied.get(nid)
            rows.append({
                "nid": nid, "layer": layer, "idx": idx, "name": neuron_name(layer, idx), "status": state, "failed": failed,
                "dominant": int(main[idx]), "pos_extreme": float(summary[f"L{layer}.pos_extreme"][idx]),
                "neg_extreme": float(summary[f"L{layer}.neg_extreme"][idx]),
                "positive": rec.get("positive"), "negative": rec.get("negative"),
                "scores": None if state != "live" else {"pole": scored["pole"], "job": scored["job"],
                                                        **{c: scored[c] for c in prompts.TEST_CATEGORIES}},
                "research": poles and {"group": poles[0]["group"],
                                       "poles": [{k: v for k, v in p.items() if k not in ("group", "nid", "layer", "idx")} for p in poles]},
                "wiring": wiring_join(cfg, layer, idx) if has_wiring else None,
            })
    return rows


def _table(head: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head), *("| " + " | ".join(map(str, r)) + " |" for r in rows)]


def _shares(score: dict) -> list[str]:
    return [_percent(score[c]) for c in prompts.TEST_CATEGORIES]


def _ac_row(blanks: int) -> list[str]:
    return ["AC", *["-"] * blanks, *(AC_FIGURES[c] for c in prompts.TEST_CATEGORIES)]


def _number(x, form: str) -> str:
    return "n/a" if x is None else format(x, form)


def _test_row(name: str, t: dict, form: str) -> list[str]:
    return [name, _number(t["study"], form), _number(t["control"], form), _number(t["difference"], "+" + form),
            _number(t["p"], ".4f"), t["n_study"], t["n_control"]]


def _score_table(scope: dict) -> list[str]:
    return _table(["group", *prompts.TEST_CATEGORIES],
                  [[g, *("n/a" if scope["scores"][g][c] is None else _percent(scope["scores"][g][c]) for c in prompts.TEST_CATEGORIES)]
                   for g in ("study", "control")])


def _count_share(count: int, total: int) -> str:
    return f"{count} ({_percent(count / total) if total else 'n/a'})"


def _research_section(cfg: Config, research: dict | None) -> list[str]:
    lines = ["## Research question", "",
             "Do the neurons edited by p50c and the strongest writers to the \"keep deliberating\" direction u have coherent "
             "explanations that match u's meaning, compared with random live control neurons from the same layers?", ""]
    if research is None:
        return lines + [f"No result: research.json is absent (expected at {research_dir(cfg) / 'research.json'}).", ""]

    groups = ("study", "control")
    head = ["test", "study", "control", "difference", "p", "n study", "n control"]
    comparison = research["comparison"]
    lines += ["Neurons compared, with dead neurons left out of the analysis and failures counted per pole:", ""]
    lines += _table(["group", "L15 neurons", "L19 neurons", "dead (L15 / L19)", "explanation failed", "tests failed", "ratings failed", "untestable poles"],
                    [[g, research["neurons"][g]["15"], research["neurons"][g]["19"],
                      f"{research['dead'][g]['15']} / {research['dead'][g]['19']}", *(research["failed"][g][k] for k in ("explanation", "tests", "ratings")),
                      research["untestable_poles"][g]]
                     for g in groups])
    primary, secondary = comparison["primary"], comparison["secondary"]
    lines += ["", f"Statistics: groups are compared by the difference of their neuron means, with a two-sided permutation test that shuffles "
                  f"group labels within each layer ({research['permutations']:,} permutations, seed {research['seed']}). The p values are not "
                  "corrected for multiple comparisons. Poles without a rating are left out of the rating means, and failed poles score zero "
                  "in the test scores. Untestable poles are non-dominant poles that never fired: they were not tested or rated, and are left "
                  "out of every mean and of the job labels.", "",
              "### Primary test: dominant pole", "",
              "Ratings are on a scale of 0 to 3. `blind_maximum` is the share of blind-maximum passages that fired the neuron, and "
              "`specificity` is that share minus the near-miss share.", ""]
    lines += _table(head, [*(_test_row(t, primary["ratings"][t], ".2f") for t in THEMES),
                           _test_row("blind_maximum", primary["blind_maximum"], ".3f"), _test_row("specificity", primary["specificity"], ".3f")])
    lines += ["", "Mean test scores per group, dominant pole:", "", *_score_table(primary)]
    lines += ["", "### Secondary test: mean over both poles", "",
              "A neuron with an untestable pole is averaged over its dominant pole alone, so these rows are biased when the groups differ in "
              "how many poles are untestable. Read them only alongside the primary test.", ""]
    lines += _table(head, [*(_test_row(t, secondary["ratings"][t], ".2f") for t in THEMES),
                           _test_row("blind_maximum", secondary["blind_maximum"], ".3f")])
    lines += ["", "Mean test scores per group, both poles:", "", *_score_table(secondary)]
    lines += ["", "### Job labels per group", "", "Counted over both poles, with each label's share of the group in brackets.", ""]
    labels = comparison["job_labels"]
    lines += _table(["group", *JOB_LABELS], [[g, *(_count_share(labels[g][j], sum(labels[g].values())) for j in JOB_LABELS)] for g in groups])
    return lines + [""]


def _markdown(cfg: Config, summary, scores: dict[int, dict], explained: dict[int, dict], rows: list[dict], research: dict | None) -> str:
    finished = [l for l in cfg.layers if l in scores]
    per_layer = {l: summarize(list(scores[l].values()), summary, layers={l}) for l in finished}
    overall = summarize([r for l in finished for r in scores[l].values()], summary, layers=set(finished))
    missing = [l for l in cfg.layers if l not in scores]
    not_scored = [f"Not scored: {', '.join(f'L{l}' for l in missing)}. The overall rows cover only the scored layers.", ""] if missing else []

    def count(layer, state):
        return sum(r["status"] == state and (layer is None or r["layer"] == layer) for r in rows)

    def coverage(name, layer):
        total = sum(layer is None or r["layer"] == layer for r in rows)
        return [name, total, total - count(layer, "dead"), count(layer, "dead"), count(layer, "failed")]

    jobs = {l: Counter() for l in finished}
    for l in finished:
        for idx in np.flatnonzero(summary[f"L{l}.live"]):
            rec = explained[l].get(neuron_id(cfg, l, int(idx)), {})
            pole = rec.get(_pole_name(int(summary[f"L{l}.dominant"][idx])))
            jobs[l][pole["job"] if pole else "unknown"] += 1

    lines = [f"# Neuron atlas of Qwen3.5-0.8B, layers {cfg.layers[0]} to {cfg.layers[-1]} (prompt version {cfg.prompt_version})", "",
             "## Method", "",
             "Each live neuron's recorded examples were explained by Claude, pole by pole. A separate session, which saw only the "
             f"explanation, wrote {prompts.TESTS_PER_CATEGORY} blind-maximum, {prompts.TESTS_PER_CATEGORY} near-miss and "
             f"{prompts.TESTS_PER_CATEGORY} distant passages for the dominant pole. A passage counts as a hit when the base model's "
             f"activation on that pole reaches {FIRE_FRACTION:.0%} of the pole's recorded extreme.", "",
             "## Coverage", "", "Dead neurons were not sent to Claude. Failed neurons are live neurons without a scored test.", ""]
    lines += _table(["layer", "neurons", "live", "dead", "failed"],
                    [*(coverage(f"L{l}", l) for l in cfg.layers), coverage("Overall", None)])
    lines += ["", "## Primary score", "", "All live neurons on their dominant pole. Failed neurons count as zero.", "", *not_scored]
    lines += _table(["layer", "n", *prompts.TEST_CATEGORIES],
                    [*([f"L{l}", per_layer[l]["primary"]["n"], *_shares(per_layer[l]["primary"])] for l in finished),
                     ["Overall", overall["primary"]["n"], *_shares(overall["primary"])], _ac_row(1)])
    lines += ["", "## AC-matched score", "",
              "Live neurons with a positive dominant pole whose job is known and not \"unclear\". Failed neurons count as zero. Neurons "
              "whose job is unknown because the explanation failed are left out and counted in `excluded_unknown_job`.", ""]
    lines += _table(["layer", "n", "excluded_unknown_job", *prompts.TEST_CATEGORIES],
                    [*([f"L{l}", per_layer[l]["ac_matched"]["n"], per_layer[l]["ac_matched"]["excluded_unknown_job"],
                        *_shares(per_layer[l]["ac_matched"])] for l in finished),
                     ["Overall", overall["ac_matched"]["n"], overall["ac_matched"]["excluded_unknown_job"], *_shares(overall["ac_matched"])],
                     _ac_row(2)])
    lines += ["", "## Reasoning-context neurons", "",
              "Live neurons whose dominant pole was described as specific to step-by-step reasoning. Their test passages ran as "
              "plain text, outside that setting (R38), so these scores likely understate them. They are included in the primary score.", ""]
    lines += _table(["layer", "n", *prompts.TEST_CATEGORIES],
                    [*([f"L{l}", per_layer[l]["reasoning_context"]["n"], *_shares(per_layer[l]["reasoning_context"])] for l in finished),
                     ["Overall", overall["reasoning_context"]["n"], *_shares(overall["reasoning_context"])]])
    lines += ["", "## Fallback-tested neurons", "",
              "Live neurons whose test passages the test model refused, so the fallback model wrote them (R40). Most explain a "
              "model's own reasoning moves (backtracking, re-reading), which Opus's reasoning_extraction safeguard declines. "
              "They are included in the primary score.", ""]
    lines += _table(["layer", "n", *prompts.TEST_CATEGORIES],
                    [*([f"L{l}", per_layer[l]["fallback_tests"]["n"], *_shares(per_layer[l]["fallback_tests"])] for l in finished),
                     ["Overall", overall["fallback_tests"]["n"], *_shares(overall["fallback_tests"])]])
    lines += ["", "## Job labels", "", "The dominant pole of each live neuron. \"unknown\" means the explanation failed.", ""]
    lines += _table(["layer", *JOB_LABELS], [[f"L{l}", *(jobs[l][j] for j in JOB_LABELS)] for l in finished])
    lines += ["", *_research_section(cfg, research), "## Wiring", ""]
    lines += [f"Each record in `atlas_neurons.jsonl` lists the neuron's {EDGES_SHOWN} strongest outgoing and incoming level-2 edges."
              if wiring_path(cfg).exists() else f"No wiring join: edges_raw.npz is absent (expected at {wiring_path(cfg)})."]
    lines += ["", "## Limits", "", *(f"- {limit}" for limit in LIMITS)]
    return "\n".join(lines) + "\n"


def write_report(cfg: Config) -> Path:
    """Write `atlas_neurons.jsonl` and then the report, from whatever the run has finished. Returns the report's path."""
    summary = load_summary(cfg)
    stored = {l: layer_scores(cfg, l) for l in cfg.layers}
    scores = {l: found for l, found in stored.items() if found is not None}
    explained = {l: layer_explanations(cfg, l) for l in cfg.layers}
    research = read_json(research_dir(cfg) / "research.json")

    rows = _neuron_rows(cfg, summary, scores, explained, research)
    atomic_write_bytes(cfg.root / "atlas_neurons.jsonl", "".join(json.dumps(r) + "\n" for r in rows).encode())
    path = cfg.root.parent / "07_atlas.md"
    atomic_write_bytes(path, _markdown(cfg, summary, scores, explained, rows, research).encode())
    return path
