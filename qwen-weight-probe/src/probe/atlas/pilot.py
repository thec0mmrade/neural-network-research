"""The pilot: a small sample of neurons run end to end, which the user must approve before the full run starts."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from probe.atlas import llm, prompts
from probe.atlas.batches import process
from probe.atlas.config import Config, neuron_name
from probe.atlas.store import atomic_write_bytes, atomic_write_json, read_json

AC_FIGURES = {"blind_maximum": "76%", "near_miss": "11%", "distant": "~1%"}
APPROVE_COMMAND = "nix-shell --run 'python -m probe.atlas approve-pilot'"
PHASES = 3            # each neuron is explained, tested and scored
SCORE_BATCH = 20      # neurons scored between saves
N_SAMPLES = 10


class PilotNotReady(RuntimeError):
    pass


def pilot_dir(cfg: Config) -> Path:
    return cfg.root / "pilot" / cfg.prompt_version


def prompt_parts(cfg: Config) -> dict:
    """Everything that decides what Claude writes: results and approvals hold only for exactly these values."""
    return {"prompt_version": cfg.prompt_version, "prompt_hash": prompts.PROMPT_HASH, "effort": cfg.effort, "model": llm.MODEL,
            "test_model": cfg.test_model}


def load_summary(cfg: Config) -> dict[str, np.ndarray]:
    with np.load(cfg.root / "record" / "summary.npz") as z:
        return {k: z[k] for k in z.files}


def choose(cfg: Config, summary) -> dict[int, list[int]]:
    """`pilot_n` random live neurons, split evenly across the layers."""
    sizes = [len(part) for part in np.array_split(np.arange(cfg.pilot_n), len(cfg.layers))]
    chosen = {}
    for layer, size in zip(cfg.layers, sizes):
        live = np.flatnonzero(summary[f"L{layer}.live"])
        if len(live) < size:
            raise ValueError(f"layer {layer} has {len(live)} live neurons, but the pilot needs {size}")
        picked = np.random.default_rng([cfg.seed, layer]).choice(live, size, replace=False)
        chosen[layer] = sorted(int(i) for i in picked)
    return chosen


def derangement(n: int, seed) -> np.ndarray:
    """A random permutation of range(n) that moves every position. Needs n >= 2."""
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if (perm != np.arange(n)).all():
            return perm


class Progress:
    """Status as seen by one layer's work: counts each neuron's three phases as progress of the stage."""

    def __init__(self, status, stage: str, layer: int, start: int, total: int, n: int, show_layer: bool):
        self._status, self._stage, self._layer = status, stage, layer
        self._start, self._total, self._n, self._show_layer = start, total, n, show_layer

    def __getattr__(self, name):
        return getattr(self._status, name)

    def advance(self, units: int) -> None:
        self._status.stage(self._stage, self._start + units, self._total, "steps")
        if self._show_layer:
            self._status.layer(self._layer, "running", units, PHASES * self._n, None)

    def layer(self, _layer: int, _state: str, done: int, _total: int, _mean_score) -> None:
        self.advance(done)     # only the explanation phase reports through this


def _pole_name(pole: int) -> str:
    return "positive" if pole == 1 else "negative"


def _hits_text(r: dict) -> str:
    return ", ".join(f"{c} {r[c]}/{prompts.TESTS_PER_CATEGORY}" for c in prompts.TEST_CATEGORIES)


def explain_test_score(cfg: Config, base, progress: Progress, layer: int, idxs: list[int], summary,
                       out: Path) -> tuple[list[dict], list[dict]]:
    """Explain `idxs`, test each neuron's dominant pole and score the tests. Returns (scored records, the tests with their passages).

    `base` loads the model only when a batch really needs scoring and frees it afterwards: usage-limit waits during the Claude
    phases can last days, and the GPU must not stay occupied through them.
    """
    from probe.atlas.evaluate import score_tests, write_tests
    from probe.atlas.explain import _stored_count, explain_neurons

    n = len(idxs)
    poles = [int(summary[f"L{layer}.dominant"][i]) for i in idxs]
    explained = explain_neurons(cfg, progress, layer, idxs, out / "explain")
    progress.advance(n)

    items = [{"nid": r["nid"], "layer": layer, "idx": i, "pole": p, **r[_pole_name(p)]}
             for r, i, p in zip(explained, idxs, poles) if "failed" not in r]
    written = {t["nid"]: t for t in write_tests(cfg, progress, items, out / "tests")}
    tests = [written.get(r["nid"], {"nid": r["nid"], "pole": p, "failed": r.get("failed")}) for r, p in zip(explained, poles)]
    progress.advance(2 * n)

    texts = {x["nid"]: x["explanation"] for x in items}
    done = _stored_count(out / "scores", {t["nid"] for t in tests})

    def work(chunk: list[dict]) -> list[dict]:
        nonlocal done
        scored = chunk if all("failed" in t for t in chunk) else score_tests(cfg, *base.get(), layer, chunk, summary)
        done += len(chunk)
        progress.advance(2 * n + done)
        finished = [r for r in scored if "failed" not in r]
        if finished:
            last = finished[-1]
            progress.latest({"name": neuron_name(layer, last["nid"] % cfg.n_neurons), "explanation": texts[last["nid"]],
                             "score": _hits_text(last), "tokens": []})
        return scored

    scored = process(tests, SCORE_BATCH, out / "scores", work, status=progress)
    base.free()
    progress.advance(PHASES * n)
    return [{**r, "layer": layer, "idx": i, "explanation": texts.get(r["nid"])} for r, i in zip(scored, idxs)], tests


def _control(cfg: Config, base, tests: list[dict], summary, out: Path) -> dict:
    """Each neuron's hits on the blind-maximum passages written for a different, randomly paired neuron."""
    from probe.atlas.evaluate import score_tests

    usable = [t for t in tests if "failed" not in t]
    if len(usable) < 2:
        return {"n": 0, "blind_maximum": 0.0}
    by_layer: dict[int, list[dict]] = {}
    for t, donor in zip(usable, derangement(len(usable), [cfg.seed, 2])):
        by_layer.setdefault(cfg.layers[t["nid"] // cfg.n_neurons], []).append(
            {"nid": t["nid"], "pole": t["pole"], "paired_with": usable[donor]["nid"],
             "blind_maximum": usable[donor]["blind_maximum"], "near_miss": [], "distant": []})
    hits = 0
    for layer, items in by_layer.items():
        scored = process(items, SCORE_BATCH, out / f"L{layer}",
                         lambda chunk: score_tests(cfg, *base.get(), layer, chunk, summary))
        hits += sum(r["blind_maximum"] for r in scored)
    base.free()
    return {"n": len(usable), "blind_maximum": hits / (prompts.TESTS_PER_CATEGORY * len(usable))}


def _percent(x: float) -> str:
    return f"{100 * x:.1f}%"


def _markdown(report: dict) -> str:
    primary, matched = report["scores"]["primary"], report["scores"]["ac_matched"]
    usage, projection = report["usage"], report["projection"]
    lines = [f"# Atlas pilot ({report['prompt_version']})", "",
             f"{report['neurons']} neurons, {report['failed']} failed.", "",
             f"Models: explanations by `{report['model']}`, test passages by `{report['test_model']}` "
             f"(effort {report['effort']}).", "",
             f"| score | primary (n={primary['n']}) | AC-matched (n={matched['n']}) | AC |", "|---|---|---|---|"]
    lines += [f"| {c} | {_percent(primary[c])} | {_percent(matched[c])} | {AC_FIGURES[c]} |" for c in prompts.TEST_CATEGORIES]
    rc = report["scores"].get("reasoning_context")
    if rc:
        lines += ["", f"Reasoning-context poles (n={rc['n']}, included in primary above) were tested on plain text, outside the "
                      "setting they were described in (R38): " + ", ".join(f"{c} {_percent(rc[c])}" for c in prompts.TEST_CATEGORIES) + "."]
    lines += ["", f"Control (blind-maximum passages written for a different neuron, n={report['control']['n']}): "
                  f"{_percent(report['control']['blind_maximum'])}. Good explanations should score far above this.", "",
              "## Cost", "",
              f"- Tokens used: {usage['input_tokens']:,} input, {usage['output_tokens']:,} output, in {usage['calls']} calls.",
              f"- Elapsed: {report['elapsed_s'] / 3600:.2f} h, not counting usage-limit pauses.",
              f"- Projected full run ({projection['live_neurons']:,} live neurons): {projection['tokens']:,.0f} tokens and "
              f"{projection['hours']:.0f} h of explaining, testing and scoring, not counting usage-limit pauses.",
              "- Compare the `/usage` figure from before and after the pilot to see what share of the Max allowance it took.", "",
              "## Sample neurons", ""]
    for s in report["samples"]:
        lines += [f"### {s['name']} ({_pole_name(s['pole'])} pole, {s['job']})", "", s["explanation"], "", _hits_text(s), ""]
    lines += ["## Next step", "", f"If this looks right, approve the full run with `{APPROVE_COMMAND}`, then start `run` again."]
    return "\n".join(lines) + "\n"


def run_pilot(cfg: Config, base, status) -> dict:
    """Run the pilot neurons through every step and write `report.md` and `report.json`, the latter last."""
    from probe.atlas.evaluate import summarize

    out = pilot_dir(cfg)
    summary = load_summary(cfg)
    chosen = choose(cfg, summary)
    status.flush()
    if read_json(out / "started.json") is None:     # kept across restarts, so the cost covers the whole pilot
        atomic_write_json(out / "started.json", {"usage": status.usage_, "active_s": status.active_s})
    started = read_json(out / "started.json")

    total, start, records, tests = PHASES * cfg.pilot_n, 0, [], []
    for layer in cfg.layers:
        progress = Progress(status, "pilot", layer, start, total, len(chosen[layer]), show_layer=False)
        scored, written = explain_test_score(cfg, base, progress, layer, chosen[layer], summary, out / f"L{layer}")
        records += scored
        tests += written
        start += PHASES * len(chosen[layer])
    control = _control(cfg, base, tests, summary, out / "control")

    only_pilot = dict(summary)
    for layer, idxs in chosen.items():
        only_pilot[f"L{layer}.live"] = np.isin(np.arange(cfg.n_neurons), idxs)
    scores = summarize(records, only_pilot)

    status.flush()
    usage = {k: status.usage_[k] - started["usage"][k] for k in ("input_tokens", "output_tokens", "calls")}
    elapsed_s = status.active_s - started["active_s"]
    live = sum(int(np.count_nonzero(summary[f"L{layer}.live"])) for layer in cfg.layers)
    scale = live / cfg.pilot_n
    finished = [r for r in records if "failed" not in r]
    picks = np.random.default_rng([cfg.seed, 3]).permutation(len(finished))[:N_SAMPLES]
    report = {**prompt_parts(cfg), "neurons": cfg.pilot_n, "failed": len(records) - len(finished), "scores": scores,
              "control": control, "ac": AC_FIGURES, "usage": usage, "elapsed_s": elapsed_s,
              "projection": {"live_neurons": live, "tokens": (usage["input_tokens"] + usage["output_tokens"]) * scale,
                             "hours": elapsed_s * scale / 3600},
              "samples": [{"name": neuron_name(r["layer"], r["idx"]), "pole": r["pole"], "job": r["job"],
                           "explanation": r["explanation"], **{c: r[c] for c in prompts.TEST_CATEGORIES}}
                          for r in (finished[int(i)] for i in sorted(picks))]}
    atomic_write_bytes(out / "report.md", _markdown(report).encode())
    atomic_write_json(out / "report.json", report)
    status.scores(scores["primary"], scores["ac_matched"])
    status.stage("pilot", total, total, "steps")
    return report


def is_approved(cfg: Config) -> bool:
    return read_json(cfg.root / "pilot" / "approved.json") == prompt_parts(cfg)


def approve(cfg: Config) -> None:
    """Record the user's go-ahead for the current prompts. Refuses unless a finished pilot report exists for exactly these prompts."""
    report = read_json(pilot_dir(cfg) / "report.json")
    parts = prompt_parts(cfg)
    if not isinstance(report, dict) or any(report.get(k) != v for k, v in parts.items()):
        raise PilotNotReady(f"no finished pilot report for prompt version {cfg.prompt_version} with the current prompts; "
                            "run the pilot and review its report first")
    atomic_write_json(cfg.root / "pilot" / "approved.json", parts)
