"""The research question and the final report, over a small synthetic finished run. No real model, no Claude."""
import json
import re
import sys
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from probe.atlas import evaluate, prompts, report, research, runner
from probe.atlas.config import Config, neuron_id
from probe.atlas.llm import Refused
from probe.atlas.record import edited_neurons
from probe.atlas.report import wiring_join, write_report
from probe.atlas.research import compare, control_neurons, permutation_test, rate_blind, run_research, study_neurons
from probe.atlas.status import Status
from probe.atlas.store import atomic_write_json, read_json
from probe.weights import RESULTS, Checkpoint, layer_tensor

N = 3584
LIVE = 40                 # the first LIVE neurons of each layer are live, the rest dead
TEST_FAILED, EXPLAIN_FAILED, UNCLEAR = 6, 8, 10     # all positive-dominant, in every layer
REFUSED = (15, 2)         # Claude refuses to write tests for this neuron's non-dominant pole
STUDY = {15: np.array([0, 1, 2, TEST_FAILED, EXPLAIN_FAILED, 100]), 19: np.array([0, 1, 3])}     # L15.100 is dead
DOMINANT_HITS, STUDY_HITS, CONTROL_HITS = 6, 9, 3


@dataclass(frozen=True)
class FinishedRun(Config):
    @property
    def n_neurons_total(self):
        return len(self.layers) * self.n_neurons


def dominant(idx):
    return 1 if idx % 2 == 0 else -1


def pole_text(nid, pole):
    return f"neuron-{nid}-{'positive' if pole == 1 else 'negative'}"


def build_summary(cfg):
    out = {}
    for layer in cfg.layers:
        live = np.zeros(N, bool)
        live[:LIVE] = True
        out[f"L{layer}.live"] = live
        out[f"L{layer}.dominant"] = np.array([dominant(i) for i in range(N)], np.int8)
        out[f"L{layer}.pos_extreme"] = np.ones(N)
        out[f"L{layer}.neg_extreme"] = np.ones(N)
    return out


def explanation(cfg, layer, idx):
    nid = neuron_id(cfg, layer, idx)
    if idx == EXPLAIN_FAILED:
        return {"nid": nid, "failed": "bad output"}

    def pole_record(pole):
        job = "unclear" if idx == UNCLEAR and pole == dominant(idx) else "one job" if pole == 1 else "two related jobs"
        refuse = " REFUSE" if (layer, idx) == REFUSED and pole != dominant(idx) else ""
        return {"explanation": pole_text(nid, pole) + refuse, "context": "plain", "job": job}

    return {"nid": nid, "layer": layer, "idx": idx, "positive": pole_record(1), "negative": pole_record(-1)}


def score_record(cfg, layer, idx):
    nid = neuron_id(cfg, layer, idx)
    pole = dominant(idx)
    where = {"layer": layer, "idx": idx}
    if idx == EXPLAIN_FAILED:
        return {"nid": nid, "pole": pole, "failed": "bad output", "explanation": None, **where}
    known = explanation(cfg, layer, idx)["positive" if pole == 1 else "negative"]
    if idx == TEST_FAILED:
        return {"nid": nid, "pole": pole, "job": known["job"], "failed": "no tests", "explanation": known["explanation"], **where}
    return {"nid": nid, "pole": pole, "job": known["job"], "explanation": known["explanation"],
            "blind_maximum": DOMINANT_HITS, "near_miss": 1, "distant": 0, **where}


def build_layers(cfg, layers=None):
    (cfg.root / "record").mkdir(parents=True, exist_ok=True)
    np.savez(cfg.root / "record" / "summary.npz", **build_summary(cfg))
    for layer in layers or cfg.layers:
        out = runner._layer_dir(cfg, layer)
        rows = [explanation(cfg, layer, i) for i in range(LIVE)]
        for start in range(0, LIVE, 20):
            chunk = rows[start:start + 20]
            atomic_write_json(out / "explain" / f"batch_{chunk[0]['nid']}_{len(chunk)}.json", chunk)
        atomic_write_json(out / "scores.json", {"layer": layer, "records": [score_record(cfg, layer, i) for i in range(LIVE)]})


def study_nids(cfg):
    return {neuron_id(cfg, l, int(i)) for l, idxs in STUDY.items() for i in idxs}


class FakeBase:
    def __init__(self):
        self.loaded, self.loads, self.frees = False, 0, 0

    def get(self):
        self.loaded = True
        self.loads += 1
        return "model", "tok"

    def free(self):
        self.loaded = False
        self.frees += 1


class FakeWorld:
    """Stands in for Claude and the GPU, and checks that the model is never held during a Claude call."""

    def __init__(self, cfg, base):
        self.cfg, self.base = cfg, base
        self.test_prompts, self.rate_prompts, self.scored = [], [], []

    def write(self, _cfg, _status, system, prompt, schema, model=None):
        assert system == prompts.TEST_SYSTEM and not self.base.loaded
        self.test_prompts.append(prompt)
        if "REFUSE" in prompt:
            raise Refused("refused")
        passages = [{"text": "some text"}] * prompts.TESTS_PER_CATEGORY
        return SimpleNamespace(data={i: dict.fromkeys(prompts.TEST_CATEGORIES, passages) for i in schema["required"]})

    def rate(self, _cfg, _status, system, prompt, schema):
        assert system == prompts.RATE_SYSTEM and not self.base.loaded
        self.rate_prompts.append(prompt)
        ids = schema["properties"]["ratings"]["items"]["properties"]["id"]["enum"]
        texts = dict(re.findall(r"=== (r\d+) ===\n(.*)", prompt))
        nid = {i: int(re.match(r"neuron-(\d+)-", texts[i])[1]) for i in ids}
        return SimpleNamespace(data={"ratings": [{"id": i, "move_on": 3 if nid[i] in study_nids(self.cfg) else 1, "reconsider": 0}
                                                 for i in ids]})

    def score(self, _cfg, model, _tok, layer, tests, _summary):
        assert model == "model" and self.base.loaded
        self.scored += [(layer, t["nid"], t["pole"]) for t in tests]
        hits = lambda t: STUDY_HITS if t["nid"] in study_nids(self.cfg) else CONTROL_HITS
        return [t if "failed" in t else {**t, "blind_maximum": hits(t), "near_miss": 2, "distant": 1} for t in tests]


def install(mp, cfg):
    base = FakeBase()
    world = FakeWorld(cfg, base)
    mp.setattr(research, "study_neurons", lambda _cfg: {l: v.copy() for l, v in STUDY.items()})
    mp.setattr(evaluate, "call_with_wait", world.write)
    mp.setattr(research, "call_with_wait", world.rate)
    mp.setattr(research, "score_tests", world.score)
    return base, world


def write_edges(cfg):
    from probe.wiring import components

    n = len(components()[0])
    arrays = {f"{side}_{kind}": np.full((n, 16), -1, np.int32 if kind == "id" else np.float32)
              for side in ("out", "in") for kind in ("id", "score")}
    row = report.component_id(15, 3)
    arrays.update({f"{side}_sign": np.zeros((n, 16), np.int8) for side in ("out", "in")})
    arrays["out_id"][row, :3], arrays["out_score"][row, :3], arrays["out_sign"][row, :3] = [row + 4000, 7, 9], [9.5, 4.25, 1.5], [1, -1, 1]
    arrays["in_id"][row, 0], arrays["in_score"][row, 0], arrays["in_sign"][row, 0] = 12, 2.5, -1
    path = report.wiring_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


@pytest.fixture(scope="module")
def finished_tiny_run(tmp_path_factory):
    cfg = FinishedRun.tiny(tmp_path_factory.mktemp("finished") / "07_atlas")
    build_layers(cfg)
    write_edges(cfg)
    with pytest.MonkeyPatch.context() as mp:
        base, world = install(mp, cfg)
        run_research(cfg, base, Status(cfg.root / "status.json"))
    return cfg


@pytest.fixture
def small_run(tiny_cfg, monkeypatch):
    build_layers(tiny_cfg, layers=research.RESEARCH_LAYERS)
    base, world = install(monkeypatch, tiny_cfg)
    return SimpleNamespace(cfg=tiny_cfg, base=base, world=world, status=Status(tiny_cfg.root / "status.json"))


@pytest.fixture
def summary_all_live(tiny_cfg):
    return {f"L{l}.live": np.ones(N, bool) for l in tiny_cfg.layers}


@pytest.fixture
def capture_prompt(tiny_cfg, tmp_path, monkeypatch):
    seen = SimpleNamespace(status=Status(tmp_path / "s.json"), out=tmp_path / "ratings", text="", calls=0)

    def fake(_cfg, _status, system, prompt, schema):
        seen.text += prompt
        seen.calls += 1
        ids = schema["properties"]["ratings"]["items"]["properties"]["id"]["enum"]
        return SimpleNamespace(data={"ratings": [{"id": i, "move_on": 1, "reconsider": 2} for i in ids]})

    monkeypatch.setattr(research, "call_with_wait", fake)
    return seen


def item(n):
    return {"nid": n, "pole": 1, "explanation": f"fires on the word w{n}"}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_study_set_contains_edited_and_top_writers(tiny_cfg):
    s = study_neurons(tiny_cfg)
    e = edited_neurons("p50c")
    assert sorted(s) == [15, 19]
    for l in (15, 19):
        u = np.load(RESULTS / f"u{l}_p50c.npy")
        top = np.argsort(-np.abs(u @ Checkpoint().get(layer_tensor(l, "down"))))[:64]
        assert set(s[l]) == set(e[l]) | set(top.tolist()) and len(s[l]) >= 64


def test_controls_are_live_disjoint_and_matched(tiny_cfg, summary_all_live):
    s = study_neurons(tiny_cfg)
    c = control_neurons(tiny_cfg, summary_all_live, s)
    for l in (15, 19):
        assert len(c[l]) == len(s[l]) and not set(c[l].tolist()) & set(s[l].tolist())
    assert all(np.array_equal(c[l], control_neurons(tiny_cfg, summary_all_live, s)[l]) for l in c)


def test_controls_come_only_from_live_neurons(tiny_cfg):
    summary = {f"L{l}.live": np.arange(N) < 30 for l in (15, 19)}
    c = control_neurons(tiny_cfg, summary, {15: np.arange(10), 19: np.arange(10)})
    assert all(set(c[l].tolist()) <= set(range(10, 30)) for l in c)


def test_controls_refuse_when_too_few_live_neurons_remain(tiny_cfg):
    summary = {f"L{l}.live": np.arange(N) < 12 for l in (15, 19)}
    with pytest.raises(ValueError, match="live"):
        control_neurons(tiny_cfg, summary, {15: np.arange(10), 19: np.arange(10)})


def test_rating_prompt_hides_group_membership(tiny_cfg, capture_prompt):
    rate_blind(tiny_cfg, capture_prompt.status, study=[item(1)], control=[item(2)], out=capture_prompt.out)
    assert "control" not in capture_prompt.text.lower() and "edited" not in capture_prompt.text.lower()
    assert "study" not in capture_prompt.text.lower()


def test_rating_prompt_holds_only_anonymous_ids_and_texts(tiny_cfg, capture_prompt):
    rate_blind(tiny_cfg, capture_prompt.status, study=[item(1)], control=[item(2)], out=capture_prompt.out)
    lines = [l for l in capture_prompt.text.splitlines() if l]
    assert sorted(lines) == sorted(["=== r001 ===", "=== r002 ===", "fires on the word w1", "fires on the word w2"])


def test_rating_order_mixes_groups_and_is_seeded(tiny_cfg, capture_prompt, tmp_path):
    study, control = [item(n) for n in range(100, 130)], [item(n) for n in range(200, 230)]
    rated = rate_blind(tiny_cfg, capture_prompt.status, study, control, capture_prompt.out)
    groups = [r["group"] for r in sorted(rated, key=lambda r: r["id"])]
    assert groups != sorted(groups) and groups != sorted(groups, reverse=True)
    again = rate_blind(tiny_cfg, capture_prompt.status, study, control, tmp_path / "second")
    assert [(r["id"], r["nid"]) for r in again] == [(r["id"], r["nid"]) for r in rated]


def test_rating_returns_each_score_with_its_item(tiny_cfg, capture_prompt):
    rated = rate_blind(tiny_cfg, capture_prompt.status, [item(1)], [item(2)], capture_prompt.out)
    assert {(r["nid"], r["group"], r["move_on"], r["reconsider"]) for r in rated} == {(1, "study", 1, 2), (2, "control", 1, 2)}


def test_rating_refuses_stored_batches_from_a_different_item_list(tiny_cfg, capture_prompt):
    rate_blind(tiny_cfg, capture_prompt.status, [item(1)], [item(2)], capture_prompt.out)
    with pytest.raises(ValueError, match="different"):
        rate_blind(tiny_cfg, capture_prompt.status, [item(3)], [item(2)], capture_prompt.out)


def test_invalid_rating_is_recorded_as_failed(tiny_cfg, capture_prompt, monkeypatch):
    def bad(_cfg, _status, _system, _prompt, schema):
        ids = schema["properties"]["ratings"]["items"]["properties"]["id"]["enum"]
        return SimpleNamespace(data={"ratings": [{"id": i, "move_on": 9, "reconsider": 0} for i in ids]})

    monkeypatch.setattr(research, "call_with_wait", bad)
    rated = rate_blind(tiny_cfg, capture_prompt.status, [item(1)], [item(2)], capture_prompt.out)
    assert all("failed" in r and "move_on" not in r for r in rated)


def test_permutation_test_on_separated_and_identical_groups():
    far = permutation_test([3.0] * 20, [0.0] * 20, seed=1)
    assert far["difference"] == 3.0 and far["p"] < 0.001 and far["n_study"] == far["n_control"] == 20
    same = permutation_test([1.0] * 20, [1.0] * 20, seed=1)
    assert same["difference"] == 0.0 and same["p"] == 1.0
    assert permutation_test([1.0, 2.0, 0.5], [0.0, 1.5, 1.0], seed=4) == permutation_test([1.0, 2.0, 0.5], [0.0, 1.5, 1.0], seed=4)


def test_permutation_test_with_an_empty_group_gives_no_p_value():
    assert permutation_test([], [1.0], seed=1)["p"] is None


def test_stratified_permutation_keeps_a_layer_effect_out_of_the_group_difference():
    rng = np.random.default_rng(5)
    high, low = rng.normal(5, 0.5, 40), rng.normal(0, 0.5, 40)     # values depend only on the layer
    study, control = np.r_[high[:30], low[:10]], np.r_[high[30:], low[10:]]
    study_layers, control_layers = [15] * 30 + [19] * 10, [15] * 10 + [19] * 30
    assert permutation_test(study, control, seed=1)["p"] < 0.01
    assert permutation_test(study, control, seed=1, study_layers=study_layers, control_layers=control_layers)["p"] > 0.05


def null_rows(rng, silent_study, silent_control, n=100):
    """Neurons whose two groups differ only in how many non-dominant poles never fired."""
    rows = []
    for group, silent in (("study", silent_study), ("control", silent_control)):
        for k in range(n):
            nid = k if group == "study" else k + n
            base = {"group": group, "nid": nid, "layer": 15 if k % 2 else 19, "job": "one job", "explanation": "text"}
            hits = lambda p: {c: int(rng.binomial(10, q)) for c, q in zip(prompts.TEST_CATEGORIES, (p, 0.1, 0.01))}
            rows.append({**base, "pole": 1, "dominant": True, **hits(0.7), "move_on": int(rng.integers(1, 4)), "reconsider": 1})
            if rng.random() < silent:
                rows.append({**base, "pole": -1, "dominant": False, "untestable": True})
            else:
                rows.append({**base, "pole": -1, "dominant": False, **hits(0.2), "move_on": int(rng.integers(0, 2)), "reconsider": 1})
    return rows


def test_unequal_silent_poles_do_not_make_the_primary_test_significant(tiny_cfg, monkeypatch):
    monkeypatch.setattr(research, "PERMUTATIONS", 199)
    runs = 200
    primary = secondary = 0
    for run in range(runs):
        result = compare(replace(tiny_cfg, seed=run), null_rows(np.random.default_rng([21, run]), 0.10, 0.0))
        primary += result["primary"]["blind_maximum"]["p"] < 0.05
        secondary += result["secondary"]["blind_maximum"]["p"] < 0.05
    assert 0.02 <= primary / runs <= 0.09
    assert secondary / runs > 0.2     # the both-pole average is the biased one, which is why it is not the primary test


def test_compare_shuffles_group_labels_within_each_layer(tiny_cfg, monkeypatch):
    monkeypatch.setattr(research, "PERMUTATIONS", 999)
    rng = np.random.default_rng(9)
    rows = []
    for nid, (group, layer) in enumerate([("study", 15)] * 30 + [("study", 19)] * 10 + [("control", 15)] * 10 + [("control", 19)] * 30):
        hits = {"blind_maximum": int(rng.binomial(10, 0.9 if layer == 15 else 0.1)), "near_miss": 0, "distant": 0}     # only the layer matters
        rows.append({"group": group, "nid": nid, "layer": layer, "pole": 1, "dominant": True, "job": "one job", "move_on": 1, "reconsider": 1, **hits})
    test = compare(tiny_cfg, rows)["primary"]["blind_maximum"]
    assert test["difference"] > 0.3 and test["p"] > 0.05


def test_primary_comparison_uses_only_the_dominant_pole(small_run):
    primary = run_research(small_run.cfg, small_run.base, small_run.status)["comparison"]["primary"]
    assert primary["scope"] == "dominant pole"
    # six neurons hit 6 of 10 on the dominant pole, and two failed there
    assert primary["blind_maximum"]["study"] == pytest.approx(6 * 0.6 / 8) and primary["blind_maximum"]["n_study"] == 8
    assert primary["scores"]["study"]["blind_maximum"] == pytest.approx(primary["blind_maximum"]["study"])
    assert primary["scores"]["study"]["near_miss"] == pytest.approx(6 * 0.1 / 8)
    assert primary["ratings"]["move_on"]["study"] == 3 and primary["ratings"]["move_on"]["n_study"] == 7


def test_secondary_comparison_is_labelled_as_both_poles(small_run):
    assert run_research(small_run.cfg, small_run.base, small_run.status)["comparison"]["secondary"]["scope"] == "both poles"


def test_specificity_is_blind_maximum_minus_near_miss_on_the_dominant_pole(small_run):
    primary = run_research(small_run.cfg, small_run.base, small_run.status)["comparison"]["primary"]
    assert primary["specificity"]["study"] == pytest.approx(6 * (0.6 - 0.1) / 8)
    assert 0 < primary["specificity"]["p"] <= 1 and primary["specificity"]["n_study"] == 8


def test_research_writes_to_the_prompt_version_folder(small_run):
    run_research(small_run.cfg, small_run.base, small_run.status)
    assert (small_run.cfg.root / "research" / small_run.cfg.prompt_version / "research.json").exists()
    assert not (small_run.cfg.root / "research.json").exists()


def test_only_the_non_dominant_pole_is_tested_again(small_run):
    result = run_research(small_run.cfg, small_run.base, small_run.status)
    new = {(nid, pole) for _, nid, pole in small_run.world.scored}
    assert new and all(pole == -dominant(nid % N) for nid, pole in new)
    kept = [r for r in result["rows"] if r["dominant"] and "blind_maximum" in r]
    assert kept and all(r["blind_maximum"] == DOMINANT_HITS for r in kept)


def test_each_pole_gets_its_own_output_folders(small_run):
    run_research(small_run.cfg, small_run.base, small_run.status)
    out = research.research_dir(small_run.cfg)
    for pole in ("positive", "negative"):
        assert list((out / "L15" / pole / "tests").glob("batch_*.json"))
        assert list((out / "L15" / pole / "scores").glob("batch_*.json"))


def test_dead_study_neurons_are_excluded_and_counted(small_run):
    result = run_research(small_run.cfg, small_run.base, small_run.status)
    assert result["dead"]["study"] == {"15": 1, "19": 0} and result["dead"]["control"] == {"15": 0, "19": 0}
    assert result["neurons"]["study"] == {"15": 5, "19": 3}
    assert neuron_id(small_run.cfg, 15, 100) not in {r["nid"] for r in result["rows"]}


def test_each_live_neuron_has_both_poles_in_the_rows(small_run):
    result = run_research(small_run.cfg, small_run.base, small_run.status)
    nids = {r["nid"] for r in result["rows"]}
    assert len(result["rows"]) == 2 * len(nids) and {r["group"] for r in result["rows"]} == {"study", "control"}


def test_blind_maximum_comparison_counts_failed_poles_as_zero(small_run):
    result = run_research(small_run.cfg, small_run.base, small_run.status)
    # per neuron, the mean of its two poles: three full neurons in L19, and L15 with one refusal and two kinds of failure
    expected = np.mean([0.75, 0.75, (0.6 + 0.0) / 2, (0.0 + 0.9) / 2, 0.0, 0.75, 0.75, 0.75])
    assert result["comparison"]["secondary"]["blind_maximum"]["study"] == pytest.approx(expected)
    assert result["comparison"]["secondary"]["blind_maximum"]["n_study"] == 8


def test_rating_comparison_uses_only_rated_poles(small_run):
    result = run_research(small_run.cfg, small_run.base, small_run.status)
    move_on = result["comparison"]["secondary"]["ratings"]["move_on"]
    assert move_on["study"] == 3 and move_on["control"] == 1 and move_on["difference"] == 2
    assert move_on["n_study"] == 7     # the neuron whose explanation failed has no text to rate
    assert 0 < move_on["p"] < 0.05
    assert result["comparison"]["secondary"]["ratings"]["reconsider"]["difference"] == 0


def test_failures_are_counted_by_kind(small_run):
    result = run_research(small_run.cfg, small_run.base, small_run.status)
    failed = result["failed"]["study"]
    assert failed["explanation"] == 2     # both poles of the neuron without an explanation
    assert failed["tests"] == 2           # the refused pole and the dominant pole whose tests failed
    assert failed["ratings"] == 0


def test_job_labels_are_counted_per_group(small_run):
    result = run_research(small_run.cfg, small_run.base, small_run.status)
    jobs = result["comparison"]["job_labels"]["study"]
    assert jobs["unknown"] == 2 and sum(jobs.values()) == 16
    assert jobs["one job"] + jobs["two related jobs"] == 14


def test_category_means_are_reported_per_group(small_run):
    result = run_research(small_run.cfg, small_run.base, small_run.status)
    scores = result["comparison"]["secondary"]["scores"]
    assert set(scores) == {"study", "control"} and set(scores["study"]) == set(prompts.TEST_CATEGORIES)
    assert scores["study"]["blind_maximum"] == pytest.approx(result["comparison"]["secondary"]["blind_maximum"]["study"])


def test_model_is_freed_before_rating_and_loaded_only_to_score(small_run):
    run_research(small_run.cfg, small_run.base, small_run.status)     # FakeWorld asserts this on every call
    assert small_run.base.loads >= 1 and not small_run.base.loaded


def test_a_restart_pays_for_nothing_twice(small_run):
    first = run_research(small_run.cfg, small_run.base, small_run.status)
    calls = len(small_run.world.test_prompts) + len(small_run.world.rate_prompts), len(small_run.world.scored)
    (research.research_dir(small_run.cfg) / "research.json").unlink()
    second = run_research(small_run.cfg, small_run.base, small_run.status)
    assert (len(small_run.world.test_prompts) + len(small_run.world.rate_prompts), len(small_run.world.scored)) == calls
    assert second["comparison"] == first["comparison"]


def test_finished_research_is_only_read_back(small_run):
    first = run_research(small_run.cfg, small_run.base, small_run.status)
    loads = small_run.base.loads
    assert run_research(small_run.cfg, small_run.base, small_run.status) == first and small_run.base.loads == loads


@pytest.fixture
def silent_pole_run(small_run):
    """L19.3 is negative-dominant, and its positive pole never fired."""
    path = small_run.cfg.root / "record" / "summary.npz"
    with np.load(path) as z:
        summary = {k: z[k] for k in z.files}
    summary["L19.pos_extreme"][3] = 0.0
    np.savez(path, **summary)
    small_run.nid = neuron_id(small_run.cfg, 19, 3)
    return small_run


def test_a_pole_that_never_fired_costs_no_claude_call(silent_pole_run):
    run = silent_pole_run
    run_research(run.cfg, run.base, run.status)
    silent = pole_text(run.nid, 1)
    assert not any(silent in p for p in run.world.test_prompts + run.world.rate_prompts)
    assert (run.nid, 1) not in {(nid, pole) for _, nid, pole in run.world.scored}
    assert any(pole_text(run.nid, -1) in p for p in run.world.rate_prompts)     # its dominant pole is still rated


def test_untestable_poles_are_counted_per_group(silent_pole_run):
    run = silent_pole_run
    result = run_research(run.cfg, run.base, run.status)
    assert result["untestable_poles"] == {"study": 1, "control": 0}
    assert [r["pole"] for r in result["rows"] if r.get("untestable")] == [1]
    assert result["failed"]["study"] == {"explanation": 2, "tests": 2, "ratings": 0}


def test_untestable_poles_are_left_out_of_the_comparison(silent_pole_run):
    run = silent_pole_run
    result = run_research(run.cfg, run.base, run.status)
    # L19.3 is now judged by its dominant pole alone (0.6), not averaged with a zero
    expected = np.mean([0.75, 0.75, (0.6 + 0.0) / 2, (0.0 + 0.9) / 2, 0.0, 0.75, 0.75, 0.6])
    assert result["comparison"]["secondary"]["blind_maximum"]["study"] == pytest.approx(expected)
    assert sum(result["comparison"]["job_labels"]["study"].values()) == 15


def test_report_states_the_untestable_poles(silent_pole_run):
    run = silent_pole_run
    run_research(run.cfg, run.base, run.status)
    text = write_report(run.cfg).read_text()
    section = text[text.index("## Research question"):text.index("## Limits")]
    assert "untestable poles" in section and "| study | 5 | 3 | 1 / 0 | 2 | 2 | 0 | 1 |" in section
    rows = {r["nid"]: r for r in read_jsonl(run.cfg.root / "atlas_neurons.jsonl")}
    assert [p["pole"] for p in rows[run.nid]["research"]["poles"] if p.get("untestable")] == [1]


def test_research_needs_the_finished_layers(tiny_cfg, monkeypatch):
    build_layers(tiny_cfg, layers=[15])
    base, _ = install(monkeypatch, tiny_cfg)
    with pytest.raises(RuntimeError, match="L19"):
        run_research(tiny_cfg, base, Status(tiny_cfg.root / "status.json"))


def test_runner_passes_the_base_handle_to_research(tiny_cfg, monkeypatch):
    seen = []
    monkeypatch.setitem(sys.modules, "probe.atlas.research", SimpleNamespace(run_research=lambda *a: seen.append(a)))
    runner._research(tiny_cfg, "status", "base")
    assert seen == [(tiny_cfg, "base", "status")]


def test_component_ids_and_names_match_the_wiring_table():
    from probe.wiring import comp_name, components

    layer, kind, idx = components()
    for cid in np.flatnonzero(kind == 0)[::997]:
        assert report.component_id(int(layer[cid]), int(idx[cid])) == cid
    for cid in range(0, len(layer), 211):
        assert report.component_name(cid) == comp_name(layer[cid], kind[cid], idx[cid])


def test_wiring_join_returns_the_strongest_edges(finished_tiny_run):
    joined = wiring_join(finished_tiny_run, 15, 3, k=2)
    row = report.component_id(15, 3)
    assert [e["id"] for e in joined["out"]] == [row + 4000, 7] and [e["score"] for e in joined["out"]] == [9.5, 4.25]
    assert joined["out"][1]["name"] == report.component_name(7)
    assert [e["sign"] for e in joined["out"]] == [1, -1] and joined["in"][0]["sign"] == -1
    assert [e["id"] for e in joined["in"]] == [12]


def test_report_includes_required_sections(finished_tiny_run):
    text = write_report(finished_tiny_run).read_text()
    for s in ("76%", "11%", "~1%", "AC-matched", "Primary", "Job labels", "Research question", "Limits", "dead", "excluded_unknown_job"):
        assert s in text


def test_report_is_written_beside_the_atlas_folder(finished_tiny_run):
    assert write_report(finished_tiny_run) == finished_tiny_run.root.parent / "07_atlas.md"


def test_report_states_every_limit_of_the_design(finished_tiny_run):
    text = write_report(finished_tiny_run).read_text()
    limits = text[text.index("## Limits"):]
    for phrase in ("different kind of model", "dominant pole", "128 tokens", "corpus", "same model family"):
        assert phrase in limits


def test_report_scores_are_computed_from_the_records(finished_tiny_run):
    text = write_report(finished_tiny_run).read_text()
    primary = text[text.index("## Primary"):text.index("## AC-matched")]
    matched = text[text.index("## AC-matched"):text.index("## Job labels")]
    # 40 live neurons: two score zero, the others hit 6 of 10
    assert "| L15 | 40 | 57.0% | 9.5% | 0.0% |" in primary and "| Overall | 280 | 57.0% | 9.5% | 0.0% |" in primary
    # 20 positive-dominant, less one unknown and one unclear; the neuron whose tests failed counts as zero
    assert "| L15 | 18 | 1 | 56.7% | 9.4% | 0.0% |" in matched and "| Overall | 126 | 7 | 56.7% | 9.4% | 0.0% |" in matched
    assert "| AC |" in primary and "| AC |" in matched


def test_report_counts_dead_and_failed_neurons(finished_tiny_run):
    text = write_report(finished_tiny_run).read_text()
    assert f"| L15 | {N} | {LIVE} | {N - LIVE} | 2 |" in text
    assert f"| Overall | {7 * N} | {7 * LIVE} | {7 * (N - LIVE)} | 14 |" in text


def test_report_shows_job_labels_of_live_neurons(finished_tiny_run):
    text = write_report(finished_tiny_run).read_text()
    section = text[text.index("## Job labels"):text.index("## Research question")]
    # dominant pole: 18 positive "one job", 20 negative "two related jobs", 1 unclear, 1 unknown
    assert "| L15 | 18 | 20 | 0 | 1 | 1 |" in section


def test_report_shows_the_research_comparison(finished_tiny_run):
    text = write_report(finished_tiny_run).read_text()
    stored = read_json(research.research_dir(finished_tiny_run) / "research.json")
    section = text[text.index("## Research question"):text.index("## Limits")]
    move_on = stored["comparison"]["primary"]["ratings"]["move_on"]
    assert f"| move_on | {move_on['study']:.2f} | {move_on['control']:.2f} | {move_on['difference']:+.2f} | {move_on['p']:.4f} |" in section
    assert "10,000" in section and "permutation" in section and "within each layer" in section
    first, second = section.index("Primary test"), section.index("Secondary")
    primary = stored["comparison"]["primary"]
    for name, test in (("blind_maximum", primary["blind_maximum"]), ("specificity", primary["specificity"])):
        assert f"| {name} | {test['study']:.3f} | {test['control']:.3f} | {test['difference']:+.3f} | {test['p']:.4f} |" in section[first:second]
    both = stored["comparison"]["secondary"]["blind_maximum"]
    assert f"| blind_maximum | {both['study']:.3f} | {both['control']:.3f} | {both['difference']:+.3f} | {both['p']:.4f} |" in section[second:]


def test_report_shows_each_job_label_share_next_to_its_count(finished_tiny_run):
    section = write_report(finished_tiny_run).read_text().split("## Research question")[1]
    assert "| study | 7 (43.8%) | 7 (43.8%) | 0 (0.0%) | 0 (0.0%) | 2 (12.5%) |" in section


def test_every_live_neuron_has_exactly_one_record(finished_tiny_run):
    write_report(finished_tiny_run)
    rows = read_jsonl(finished_tiny_run.root / "atlas_neurons.jsonl")
    assert len({r["nid"] for r in rows}) == len(rows) == finished_tiny_run.n_neurons_total


def test_neuron_records_carry_status_explanation_scores_and_wiring(finished_tiny_run):
    write_report(finished_tiny_run)
    rows = {r["nid"]: r for r in read_jsonl(finished_tiny_run.root / "atlas_neurons.jsonl")}
    cfg = finished_tiny_run
    live = rows[neuron_id(cfg, 15, 3)]
    assert live["status"] == "live" and live["name"] == "L15.mlp.3" and live["dominant"] == -1
    assert live["negative"]["explanation"] == pole_text(neuron_id(cfg, 15, 3), -1)
    assert live["scores"] == {"pole": -1, "job": "two related jobs", "blind_maximum": DOMINANT_HITS, "near_miss": 1, "distant": 0}
    assert live["wiring"]["out"][0]["score"] == 9.5
    assert rows[neuron_id(cfg, 15, TEST_FAILED)]["status"] == "failed" and rows[neuron_id(cfg, 15, TEST_FAILED)]["scores"] is None
    assert rows[neuron_id(cfg, 15, TEST_FAILED)]["failed"] == "no tests"
    assert rows[neuron_id(cfg, 15, EXPLAIN_FAILED)]["status"] == "failed"
    dead = rows[neuron_id(cfg, 15, 500)]
    assert dead["status"] == "dead" and dead["positive"] is None and dead["scores"] is None and dead["wiring"] == {"out": [], "in": []}


def test_status_counts_cover_every_neuron(finished_tiny_run):
    write_report(finished_tiny_run)
    rows = read_jsonl(finished_tiny_run.root / "atlas_neurons.jsonl")
    counts = {s: sum(r["status"] == s for r in rows) for s in ("live", "dead", "failed")}
    assert counts == {"live": 7 * (LIVE - 2), "dead": 7 * (N - LIVE), "failed": 14}


def test_research_neurons_keep_both_tested_poles_in_their_record(finished_tiny_run):
    write_report(finished_tiny_run)
    rows = {r["nid"]: r for r in read_jsonl(finished_tiny_run.root / "atlas_neurons.jsonl")}
    studied = rows[neuron_id(finished_tiny_run, 19, 0)]["research"]
    assert studied["group"] == "study" and {p["pole"] for p in studied["poles"]} == {1, -1}
    assert rows[neuron_id(finished_tiny_run, 14, 0)]["research"] is None


def test_report_says_when_data_is_absent(tiny_cfg):
    build_layers(tiny_cfg, layers=[15])
    text = write_report(tiny_cfg).read_text()
    assert "research.json is absent" in text and "edges_raw.npz is absent" in text
    assert "Not scored: L14, L16, L17, L18, L19, L20" in text
    assert "| L15 | 40 | 57.0% |" in text and "| L19 | 40 |" not in text
    assert "| Overall | 40 | 57.0% |" in text     # unscored layers must not count as zero (R26)
    rows = read_jsonl(tiny_cfg.root / "atlas_neurons.jsonl")
    assert len(rows) == 7 * N and all(r["wiring"] is None for r in rows)
    unscored = next(r for r in rows if r["nid"] == neuron_id(tiny_cfg, 19, 0))
    assert unscored["status"] == "failed" and unscored["failed"] == "layer not scored"
