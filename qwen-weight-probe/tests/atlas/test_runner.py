"""The pilot gate, the stage order and the command line. Every stage is a stub: no real model, no Claude, no network."""
import gc
import json
import os
import re
import subprocess
import sys
import weakref
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from probe.atlas import __main__ as cli
from probe.atlas import corpus, evaluate, explain, llm, pilot, prompts, record, runner, thinking
from probe.atlas.config import Config, neuron_id
from probe.atlas.llm import LLMResult, Refused
from probe.atlas.pilot import choose
from probe.atlas.status import STAGES, Status
from probe.atlas.store import AlreadyRunning, FingerprintMismatch, atomic_write_json, read_json, run_lock

N = 3584


class FlatSummary(dict):
    """summary.npz's flat keys ("L14.live"), also readable as summary[14]["live"]."""

    def __missing__(self, layer):
        if not isinstance(layer, int):
            raise KeyError(layer)
        return {"live": self[f"L{layer}.live"]}


def make_summary(cfg, live, dominant=1):
    out = FlatSummary()
    for layer in cfg.layers:
        out[f"L{layer}.live"] = live(layer)
        out[f"L{layer}.dominant"] = np.full(N, dominant, np.int8)
        out[f"L{layer}.pos_extreme"] = np.ones(N)
        out[f"L{layer}.neg_extreme"] = np.ones(N)
    return out


def save_summary(cfg, summary):
    (cfg.root / "record").mkdir(parents=True, exist_ok=True)
    np.savez(cfg.root / "record" / "summary.npz", **summary)


def first_live(count):
    def live(_layer):
        mask = np.zeros(N, bool)
        mask[:count] = True
        return mask
    return live


@pytest.fixture
def summary_some_dead(tiny_cfg):
    def live(layer):
        mask = np.ones(N, bool)
        mask[::3] = False
        mask[layer] = False
        return mask
    return make_summary(tiny_cfg, live)


class FakeModel:
    pass


class StubbedRunner:
    """Runs the real `runner.run` over stubs that note what started and how many models were alive."""

    def __init__(self, cfg, monkeypatch):
        self.cfg, self.monkeypatch = cfg, monkeypatch
        self.stages_started, self.layers_run, self.events = [], [], []
        self.models = []
        self.most_alive = 0
        self.alive_at_p50c = None
        self.corpus_hash = "corpus-1"
        for target, name, stub in (
                (corpus, "build_sources", self._build_sources), (corpus, "assemble", self._assemble),
                (thinking, "generate_thinking", self._generate_thinking),
                (record, "record_pass", self._record_pass), (record, "record_p50c", self._record_p50c),
                (pilot, "run_pilot", self._run_pilot), (runner, "run_layer", self._run_layer),
                (runner, "_load_base", self._load_base), (runner, "_empty_gpu", lambda: None),
                (runner, "_tokenizer", lambda: "tokenizer")):
            monkeypatch.setattr(target, name, stub)
        monkeypatch.setitem(sys.modules, "probe.atlas.research", SimpleNamespace(run_research=self._research))
        monkeypatch.setitem(sys.modules, "probe.atlas.report", SimpleNamespace(write_report=self._report))

    def alive(self):
        gc.collect()
        return sum(ref() is not None for ref in self.models)

    def _load_base(self):
        self.events.append("load")
        model = FakeModel()
        self.models.append(weakref.ref(model))
        self.most_alive = max(self.most_alive, self.alive())
        return model, "tok"

    def _build_sources(self, cfg, tok, status):
        self.stages_started.append("corpus")

    def _generate_thinking(self, cfg, model, tok, status):
        self.stages_started.append("thinking")

    def _assemble(self, cfg, status):
        self.stages_started.append("assemble")
        atomic_write_json(cfg.root / "corpus" / "manifest.json", {"hash": self.corpus_hash, "shards": []})

    def _record_pass(self, cfg, model, tok, status, out, layers):
        self.stages_started.append("record")
        self.events.append("record_pass")

    def _record_p50c(self, cfg, status):
        self.events.append("record_p50c")
        self.alive_at_p50c = self.alive()

    def _run_pilot(self, cfg, base, status):
        self.stages_started.append("pilot")
        base.get()
        base.free()
        atomic_write_json(pilot.pilot_dir(cfg) / "report.json", pilot.prompt_parts(cfg))

    def _run_layer(self, cfg, base, status, layer):
        self.stages_started.append("layers")
        base.get()
        base.free()
        self.layers_run.append(layer)

    def _research(self, cfg, base, status):
        self.stages_started.append("research")

    def _report(self, cfg):
        self.stages_started.append("report")

    def run(self):
        return runner.run(self.cfg)

    def approve(self):
        pilot.approve(self.cfg)

    def change_prompt_hash(self):
        """New prompts always come with a new prompt_version; the same version with new prompts is refused."""
        self.monkeypatch.setattr(prompts, "PROMPT_HASH", "changed")
        self.cfg = replace(self.cfg, prompt_version=self.cfg.prompt_version + "-next")

    def change_corpus_hash(self):
        atomic_write_json(self.cfg.root / "corpus" / "manifest.json", {"hash": "corpus-2", "shards": []})


@pytest.fixture
def stubbed_runner(tiny_cfg, monkeypatch):
    return StubbedRunner(tiny_cfg, monkeypatch)


# --- choosing the pilot neurons ---

def test_pilot_is_even_across_layers_and_only_live(tiny_cfg, summary_some_dead):
    c = choose(replace(tiny_cfg, pilot_n=200), summary_some_dead)
    assert sum(map(len, c.values())) == 200 and {len(v) for v in c.values()} <= {28, 29}
    assert all(summary_some_dead[l]["live"][i] for l, v in c.items() for i in v)


def test_pilot_choice_is_seeded(tiny_cfg, summary_some_dead):
    assert choose(tiny_cfg, summary_some_dead) == choose(tiny_cfg, summary_some_dead)
    assert choose(tiny_cfg, summary_some_dead) != choose(replace(tiny_cfg, seed=8), summary_some_dead)


# --- the gate ---

def test_run_stops_at_pilot_gate(stubbed_runner):
    assert stubbed_runner.run() == "waiting_for_pilot_approval"
    assert "layers" not in stubbed_runner.stages_started


def test_gate_message_says_what_to_review_and_how_to_approve(stubbed_runner, capsys):
    stubbed_runner.run()
    out = capsys.readouterr().out
    assert str(pilot.pilot_dir(stubbed_runner.cfg) / "report.md") in out and "python -m probe.atlas approve-pilot" in out


def test_run_stays_at_the_gate_until_approved(stubbed_runner):
    stubbed_runner.run()
    assert stubbed_runner.run() == "waiting_for_pilot_approval"
    assert stubbed_runner.stages_started.count("pilot") == 1 and "layers" not in stubbed_runner.stages_started


def test_approval_for_old_prompt_does_not_open_gate(stubbed_runner):
    stubbed_runner.run(); stubbed_runner.approve(); stubbed_runner.change_prompt_hash()
    assert stubbed_runner.run() == "waiting_for_pilot_approval"


def test_new_prompt_version_runs_its_own_pilot(stubbed_runner):
    stubbed_runner.run(); stubbed_runner.approve(); stubbed_runner.change_prompt_hash()
    stubbed_runner.run()
    assert stubbed_runner.stages_started.count("pilot") == 2 and "layers" not in stubbed_runner.stages_started


@pytest.mark.parametrize("approval", [{"prompt_version": "v1", "prompt_hash": "other"},
                                      {"prompt_version": "v0", "prompt_hash": None}, {}, "yes"])
def test_a_mismatched_approval_file_does_not_open_gate(stubbed_runner, approval):
    stubbed_runner.run()
    if isinstance(approval, dict) and approval.get("prompt_hash", "") is None:
        approval = {**approval, "prompt_hash": prompts.PROMPT_HASH}
    atomic_write_json(stubbed_runner.cfg.root / "pilot" / "approved.json", approval)
    assert stubbed_runner.run() == "waiting_for_pilot_approval"
    assert "layers" not in stubbed_runner.stages_started


def test_waiting_note_is_cleared_once_approved(stubbed_runner):
    stubbed_runner.run()
    assert "approved" in read_json(stubbed_runner.cfg.root / "status.json")["note"]
    stubbed_runner.approve(); stubbed_runner.run()
    assert read_json(stubbed_runner.cfg.root / "status.json")["note"] is None


def test_approve_refuses_without_a_report(tiny_cfg):
    with pytest.raises(pilot.PilotNotReady, match=tiny_cfg.prompt_version):
        pilot.approve(tiny_cfg)
    assert not (tiny_cfg.root / "pilot" / "approved.json").exists()


def test_approve_refuses_a_report_from_other_prompts(stubbed_runner, monkeypatch):
    stubbed_runner.run()
    monkeypatch.setattr(prompts, "PROMPT_HASH", "changed")
    with pytest.raises(pilot.PilotNotReady):
        stubbed_runner.approve()


def test_approve_records_version_and_hash(stubbed_runner):
    stubbed_runner.run(); stubbed_runner.approve()
    assert read_json(stubbed_runner.cfg.root / "pilot" / "approved.json") == {
        "prompt_version": "v2", "prompt_hash": prompts.PROMPT_HASH, "effort": stubbed_runner.cfg.effort, "model": llm.MODEL,
        "test_model": "claude-opus-5-5"}


# --- stages ---

def test_layers_run_in_the_required_order(stubbed_runner):
    stubbed_runner.run(); stubbed_runner.approve()
    assert stubbed_runner.run() == "complete"
    assert stubbed_runner.layers_run == [15, 19, 18, 16, 17, 14, 20]


def test_stages_run_in_order(stubbed_runner):
    stubbed_runner.run(); stubbed_runner.approve(); stubbed_runner.run()
    assert list(dict.fromkeys(stubbed_runner.stages_started)) == list(STAGES)


def test_restart_skips_finished_stages(stubbed_runner):
    stubbed_runner.run(); stubbed_runner.approve(); stubbed_runner.run()
    first = list(stubbed_runner.stages_started); stubbed_runner.run()
    assert stubbed_runner.stages_started == first


def test_finished_stages_leave_done_files(stubbed_runner):
    stubbed_runner.run()
    done = {p.name for p in (stubbed_runner.cfg.root / "stages").iterdir()}
    assert done == {"corpus.done", "thinking.done", "assemble.done", "record.done", "pilot.done"}


def test_a_crashed_stage_runs_again_and_earlier_ones_do_not(stubbed_runner, monkeypatch):
    def crash(*_):
        raise RuntimeError("power cut")

    monkeypatch.setattr(thinking, "generate_thinking", crash)
    with pytest.raises(RuntimeError, match="power cut"):
        stubbed_runner.run()
    monkeypatch.setattr(thinking, "generate_thinking", stubbed_runner._generate_thinking)
    stubbed_runner.run()
    assert stubbed_runner.stages_started.count("corpus") == 1 and stubbed_runner.stages_started.count("thinking") == 1


def test_status_is_flushed_at_each_stage_end_and_exit(stubbed_runner, monkeypatch):
    flushed = []
    real = Status.flush
    monkeypatch.setattr(Status, "flush", lambda self: (flushed.append(1), real(self))[1])
    stubbed_runner.run()
    assert len(flushed) >= 6     # five stages, then the exit
    assert (stubbed_runner.cfg.root / "status.json").exists()


def test_a_second_run_is_refused_and_leaves_the_status_alone(stubbed_runner):
    path = stubbed_runner.cfg.root / "status.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"current_stage": "record"}')
    with run_lock(stubbed_runner.cfg.root / "run.lock"):
        with pytest.raises(AlreadyRunning):
            stubbed_runner.run()
    assert json.loads(path.read_text()) == {"current_stage": "record"}


# --- fingerprint ---

def test_changed_corpus_refuses_to_resume(stubbed_runner):
    stubbed_runner.run(); stubbed_runner.change_corpus_hash()
    with pytest.raises(FingerprintMismatch, match="corpus"):
        stubbed_runner.run()


@pytest.mark.parametrize("field, value", [("model", "other-revision"), ("llm_window", 99)])
def test_changed_run_settings_refuse_to_resume(stubbed_runner, field, value):
    stubbed_runner.run()
    path = stubbed_runner.cfg.root / "fingerprint.json"
    atomic_write_json(path, {**read_json(path), field: value})
    with pytest.raises(FingerprintMismatch, match=field):
        stubbed_runner.run()


def test_changed_prompts_under_the_same_version_are_refused(stubbed_runner, monkeypatch):
    stubbed_runner.run()
    monkeypatch.setattr(prompts, "PROMPT_HASH", "changed")
    with pytest.raises(FingerprintMismatch, match="prompt_version"):
        stubbed_runner.run()


def test_changed_effort_under_the_same_version_is_refused(stubbed_runner):
    stubbed_runner.run(); stubbed_runner.approve()
    stubbed_runner.cfg = replace(stubbed_runner.cfg, effort="high")
    assert not pilot.is_approved(stubbed_runner.cfg)
    with pytest.raises(FingerprintMismatch, match="effort.*prompt_version"):
        stubbed_runner.run()
    assert "layers" not in stubbed_runner.stages_started


def test_changed_model_under_the_same_version_is_refused(stubbed_runner, monkeypatch):
    stubbed_runner.run(); stubbed_runner.approve()
    monkeypatch.setattr(llm, "MODEL", "claude-other")
    assert not pilot.is_approved(stubbed_runner.cfg)
    with pytest.raises(FingerprintMismatch, match="model.*prompt_version"):
        stubbed_runner.run()
    assert "layers" not in stubbed_runner.stages_started


def test_changed_test_model_under_the_same_version_is_refused(stubbed_runner):
    """R36: an approval covers one test-writing model; a different one needs a new prompt_version and pilot."""
    stubbed_runner.run(); stubbed_runner.approve()
    stubbed_runner.cfg = replace(stubbed_runner.cfg, test_model="claude-other")
    assert not pilot.is_approved(stubbed_runner.cfg)
    with pytest.raises(FingerprintMismatch, match="test_model.*prompt_version"):
        stubbed_runner.run()
    assert "layers" not in stubbed_runner.stages_started
    with pytest.raises(pilot.PilotNotReady):
        stubbed_runner.approve()


def test_prompt_fingerprint_records_the_test_model(stubbed_runner):
    stubbed_runner.run()
    assert read_json(pilot.pilot_dir(stubbed_runner.cfg) / "fingerprint.json")["test_model"] == stubbed_runner.cfg.test_model == "claude-opus-5-5"


def test_approve_refuses_a_report_from_another_effort(stubbed_runner):
    stubbed_runner.run()
    stubbed_runner.cfg = replace(stubbed_runner.cfg, effort="high")
    with pytest.raises(pilot.PilotNotReady):
        stubbed_runner.approve()


# --- one model at a time ---

def test_at_most_one_model_is_loaded_at_a_time(stubbed_runner):
    stubbed_runner.run(); stubbed_runner.approve(); stubbed_runner.run()
    assert stubbed_runner.most_alive == 1 and stubbed_runner.alive() == 0


def test_base_model_is_freed_before_p50c_and_loaded_again_after(stubbed_runner):
    stubbed_runner.run()
    assert stubbed_runner.alive_at_p50c == 0
    events = stubbed_runner.events
    assert events.index("load") < events.index("record_pass") < events.index("record_p50c")
    assert "load" in events[events.index("record_p50c"):]


def test_stages_without_a_model_do_not_load_one(stubbed_runner):
    stubbed_runner.run(); stubbed_runner.events.clear(); stubbed_runner.run()     # only the gate is left
    assert stubbed_runner.events == []


def test_resumed_record_stage_removes_a_leftover_checkpoint(stubbed_runner):
    ckpt = stubbed_runner.cfg.root / "record" / "ckpt"
    ckpt.mkdir(parents=True)
    (ckpt / "ckpt_00000001.npz").write_bytes(b"old")
    (stubbed_runner.cfg.root / "record" / "summary.npz").write_bytes(b"finished")
    stubbed_runner.run()
    assert not ckpt.exists() and "record_pass" not in stubbed_runner.events


# --- the real pilot and layer run, over stubbed Claude and scoring ---

class Stages:
    """Stubs for explain_neurons, write_tests and score_tests."""

    def __init__(self, monkeypatch):
        self.items, self.score_calls, self.control = [], [], []
        self.crash_on_score_call = None
        self.explained = 0
        self.loads, self.gpu_empties = 0, 0
        self.model_during_claude = []
        monkeypatch.setattr(runner, "_load_base", self.load_base)
        monkeypatch.setattr(runner, "_empty_gpu", self.empty_gpu)
        self.base = runner._BaseModel()
        monkeypatch.setattr(explain, "explain_neurons", self.explain_neurons)
        monkeypatch.setattr(evaluate, "write_tests", self.write_tests)
        monkeypatch.setattr(evaluate, "score_tests", self.score_tests)

    def load_base(self):
        self.loads += 1
        return FakeModel(), "tok"

    def empty_gpu(self):
        self.gpu_empties += 1

    def explain_neurons(self, cfg, status, layer, idxs, out):
        self.model_during_claude.append(self.base._loaded is not None)
        self.explained += len(idxs)
        status.usage(100, 10)
        status.layer(layer, "running", len(idxs), len(idxs), None)
        return [{"nid": neuron_id(cfg, layer, i), "layer": layer, "idx": i,
                 "positive": {"explanation": f"positive {i}", "context": "plain", "job": "one job"},
                 "negative": {"explanation": f"negative {i}", "context": "reasoning", "job": "unclear"}} for i in idxs]

    def write_tests(self, cfg, status, items, out):
        self.model_during_claude.append(self.base._loaded is not None)
        self.items += items
        return [{"nid": x["nid"], "pole": x["pole"], "job": x["job"],
                 **{c: [{"text": f"{c} of {x['nid']}"}] * 10 for c in prompts.TEST_CATEGORIES}}
                for x in items]

    def score_tests(self, cfg, model, tok, layer, tests, summary):
        self.score_calls.append([t["nid"] for t in tests])
        if self.crash_on_score_call == len(self.score_calls):
            raise RuntimeError("crash while scoring")
        out = []
        for t in tests:
            if "failed" in t:
                out.append(t)
            elif t["near_miss"]:
                out.append({**t, "blind_maximum": 8, "near_miss": 1, "distant": 0})
            else:
                self.control.append((layer, t))
                out.append({**t, "blind_maximum": 2, "near_miss": 0, "distant": 0})
        return out


@pytest.fixture
def stages(monkeypatch):
    return Stages(monkeypatch)


def live_only(count):
    return first_live(count)


def test_pilot_scores_only_its_own_neurons(tiny_cfg, stages, status):
    save_summary(tiny_cfg, make_summary(tiny_cfg, live_only(40)))
    report = pilot.run_pilot(tiny_cfg, stages.base, status)
    assert report["scores"]["primary"]["n"] == tiny_cfg.pilot_n
    assert report["scores"]["primary"]["blind_maximum"] == pytest.approx(0.8)
    assert report["scores"]["primary"]["near_miss"] == pytest.approx(0.1)


def test_control_pairs_each_neuron_with_a_different_one(tiny_cfg, stages, status):
    save_summary(tiny_cfg, make_summary(tiny_cfg, live_only(40)))
    report = pilot.run_pilot(tiny_cfg, stages.base, status)
    measured = [t["nid"] for _, t in stages.control]
    donors = [int(t["blind_maximum"][0]["text"].rsplit(" ", 1)[1]) for _, t in stages.control]
    assert len(measured) == tiny_cfg.pilot_n and sorted(measured) == sorted(donors)
    assert all(m != d for m, d in zip(measured, donors))
    assert report["control"] == {"n": tiny_cfg.pilot_n, "blind_maximum": pytest.approx(0.2)}


def test_control_is_scored_under_the_measured_neurons_own_layer_and_pole(tiny_cfg, stages, status):
    save_summary(tiny_cfg, make_summary(tiny_cfg, live_only(40), dominant=-1))
    pilot.run_pilot(tiny_cfg, stages.base, status)
    assert all(t["nid"] // N == tiny_cfg.layers.index(layer) and t["pole"] == -1 for layer, t in stages.control)


def test_control_pairing_is_seeded(tiny_cfg):
    assert list(pilot.derangement(50, 7)) == list(pilot.derangement(50, 7))
    assert all((pilot.derangement(n, 7) != np.arange(n)).all() for n in (2, 3, 10, 50))


def test_pilot_report_holds_what_the_user_must_review(tiny_cfg, stages, status):
    save_summary(tiny_cfg, make_summary(tiny_cfg, live_only(40)))
    report = pilot.run_pilot(tiny_cfg, stages.base, status)
    text = (pilot.pilot_dir(tiny_cfg) / "report.md").read_text()
    for needle in ("76%", "11%", "~1%", "Control", "/usage", "approve-pilot", "Projected", llm.MODEL, tiny_cfg.test_model):
        assert needle in text
    assert len(report["samples"]) == 10 and all(s["explanation"] and "blind_maximum" in s for s in report["samples"])
    assert report["samples"][0]["explanation"] in text
    assert read_json(pilot.pilot_dir(tiny_cfg) / "report.json")["prompt_hash"] == prompts.PROMPT_HASH
    assert read_json(pilot.pilot_dir(tiny_cfg) / "report.json")["test_model"] == tiny_cfg.test_model


def test_pilot_report_counts_tokens_used_by_the_pilot_alone(tiny_cfg, stages, status):
    save_summary(tiny_cfg, make_summary(tiny_cfg, live_only(40)))
    status.usage(5000, 5000)     # earlier calls are not the pilot's
    report = pilot.run_pilot(tiny_cfg, stages.base, status)
    assert report["usage"] == {"input_tokens": 700, "output_tokens": 70, "calls": 7}


def test_projection_scales_the_pilot_to_all_live_neurons(tiny_cfg, stages, status):
    save_summary(tiny_cfg, make_summary(tiny_cfg, live_only(40)))
    report = pilot.run_pilot(tiny_cfg, stages.base, status)
    scale = 40 * 7 / tiny_cfg.pilot_n
    assert report["projection"]["live_neurons"] == 280
    assert report["projection"]["tokens"] == pytest.approx(770 * scale)
    assert report["projection"]["hours"] == pytest.approx(report["elapsed_s"] * scale / 3600)


def test_pilot_writes_nothing_into_the_layer_table(tiny_cfg, stages, status):
    save_summary(tiny_cfg, make_summary(tiny_cfg, live_only(40)))
    pilot.run_pilot(tiny_cfg, stages.base, status)
    assert status.layers == {} and status.stages["pilot"]["done"] == status.stages["pilot"]["total"]


def test_layer_tests_the_dominant_pole_with_that_poles_job(tiny_cfg, stages, status):
    save_summary(tiny_cfg, make_summary(tiny_cfg, live_only(3), dominant=-1))
    runner.run_layer(tiny_cfg, stages.base, status, 15)
    assert {x["pole"] for x in stages.items} == {-1}
    assert stages.items[0]["explanation"] == "negative 0" and stages.items[0]["job"] == "unclear"
    assert stages.items[0]["context"] == "reasoning"


def test_layer_writes_scores_and_updates_status(tiny_cfg, stages, status):
    save_summary(tiny_cfg, make_summary(tiny_cfg, live_only(3)))
    out = runner.run_layer(tiny_cfg, stages.base, status, 15)
    stored = read_json(tiny_cfg.root / "layers" / tiny_cfg.prompt_version / "L15" / "scores.json")
    assert stored == out and len(stored["records"]) == 3
    assert status.layers["15"] == {"state": "done", "done": 3, "total": 3, "mean_score": pytest.approx(0.8)}
    assert status.scores_["primary"]["n"] == 3     # only the finished layer counts
    assert status.latest_["score"] and status.latest_["name"] == "L15.mlp.2"


def test_scores_cover_every_finished_layer(tiny_cfg, stages, status):
    save_summary(tiny_cfg, make_summary(tiny_cfg, live_only(3)))
    runner.run_layer(tiny_cfg, stages.base, status, 15)
    runner.run_layer(tiny_cfg, stages.base, status, 19)
    assert status.scores_["primary"]["n"] == 6


def test_a_failed_explanation_counts_as_zero(tiny_cfg, stages, status, monkeypatch):
    save_summary(tiny_cfg, make_summary(tiny_cfg, live_only(2)))
    real = stages.explain_neurons

    def one_fails(cfg, st, layer, idxs, out):
        return [{"nid": r["nid"], "failed": "refused"} if r["idx"] == 0 else r for r in real(cfg, st, layer, idxs, out)]

    monkeypatch.setattr(explain, "explain_neurons", one_fails)
    out = runner.run_layer(tiny_cfg, stages.base, status, 15)
    assert len(stages.items) == 1 and out["records"][0]["failed"] == "refused"
    assert out["summary"]["primary"]["n"] == 2 and out["summary"]["primary"]["blind_maximum"] == pytest.approx(0.4)
    assert out["summary"]["ac_matched"]["excluded_unknown_job"] == 1


def test_finished_layer_is_skipped_on_restart(tiny_cfg, stages, status):
    save_summary(tiny_cfg, make_summary(tiny_cfg, live_only(3)))
    first = runner.run_layer(tiny_cfg, stages.base, status, 15)
    stages.explained = 0
    assert runner.run_layer(tiny_cfg, stages.base, status, 15) == first and stages.explained == 0


def test_scoring_resumes_from_the_last_saved_batch(tiny_cfg, stages, status):
    save_summary(tiny_cfg, make_summary(tiny_cfg, live_only(45)))
    stages.crash_on_score_call = 2
    with pytest.raises(RuntimeError, match="crash while scoring"):
        runner.run_layer(tiny_cfg, stages.base, status, 15)
    saved = stages.score_calls[0]
    stages.crash_on_score_call = None
    stages.score_calls.clear()
    out = runner.run_layer(tiny_cfg, stages.base, status, 15)
    assert saved not in stages.score_calls and len(stages.score_calls) == 2 and len(out["records"]) == 45


def test_model_is_on_the_gpu_only_while_scoring(tiny_cfg, stages, status):
    save_summary(tiny_cfg, make_summary(tiny_cfg, live_only(3)))
    runner.run_layer(tiny_cfg, stages.base, status, 15)
    runner.run_layer(tiny_cfg, stages.base, status, 19)
    assert stages.model_during_claude == [False] * 4
    assert stages.loads == 2 and stages.gpu_empties == 2 and stages.base._loaded is None


def test_pilot_frees_the_model_before_each_layers_claude_work(tiny_cfg, stages, status):
    save_summary(tiny_cfg, make_summary(tiny_cfg, live_only(40)))
    pilot.run_pilot(tiny_cfg, stages.base, status)
    assert not any(stages.model_during_claude) and stages.base._loaded is None


def test_finished_layer_does_not_load_the_model(tiny_cfg, stages, status):
    save_summary(tiny_cfg, make_summary(tiny_cfg, live_only(3)))
    runner.run_layer(tiny_cfg, stages.base, status, 15)
    stages.loads = 0
    runner.run_layer(tiny_cfg, stages.base, status, 15)
    assert stages.loads == 0


def test_layer_whose_scores_are_all_saved_does_not_load_the_model(tiny_cfg, stages, status):
    save_summary(tiny_cfg, make_summary(tiny_cfg, live_only(3)))
    runner.run_layer(tiny_cfg, stages.base, status, 15)
    (tiny_cfg.root / "layers" / tiny_cfg.prompt_version / "L15" / "scores.json").unlink()
    stages.loads = 0
    runner.run_layer(tiny_cfg, stages.base, status, 15)
    assert stages.loads == 0


def test_layer_progress_never_falls_between_phases(tiny_cfg, stages, status, monkeypatch):
    save_summary(tiny_cfg, make_summary(tiny_cfg, live_only(45)))
    seen = []
    real = status.stage
    monkeypatch.setattr(status, "stage", lambda name, done, total, unit: (seen.append(done), real(name, done, total, unit))[1])
    runner.run_layer(tiny_cfg, stages.base, status, 19)     # second in the order: starts after L15's share
    assert seen == sorted(seen) and seen[0] >= 3 * 45 and seen[-1] == 2 * 3 * 45


# --- killed at every paid call: real explain_neurons, write_tests and batches, with a fake Claude and fake scoring ---

class Killed(BaseException):
    """Not an Exception, so nothing on the way up can catch it, like a real kill."""


REFUSED_EXPLANATION, REFUSED_TEST = 3, 5


class FakeClaude:
    def __init__(self, monkeypatch, kill_at=None):
        self.sent, self.kill_at = [], kill_at
        monkeypatch.setattr(explain, "call_with_wait", self.call)
        monkeypatch.setattr(evaluate, "call_with_wait", self.call)
        monkeypatch.setattr(evaluate, "score_tests", self.score)
        monkeypatch.setattr(explain, "_tokenizer", lambda: "tok")
        monkeypatch.setattr(explain, "neuron_record", lambda cfg, layer, i: {"nid": neuron_id(cfg, layer, i), "layer": layer, "idx": i})
        monkeypatch.setattr(explain, "_latest", lambda *_: {})
        monkeypatch.setattr(prompts, "render_neuron", lambda rec, tok, cfg: f"IDX{rec['idx']}")
        monkeypatch.setattr(runner, "_load_base", lambda: (FakeModel(), "tok"))
        monkeypatch.setattr(runner, "_empty_gpu", lambda: None)

    def tick(self, call):
        self.sent.append(call)
        if len(self.sent) - 1 == self.kill_at:
            raise Killed

    def call(self, cfg, status, system, prompt, schema, model=None):
        self.tick(prompt)
        ids = schema["required"]
        if system == prompts.EXPLAIN_SYSTEM:
            idx = {k: int(v) for k, v in re.findall(r"=== (n\d+) ===\nIDX(\d+)", prompt)}
            if REFUSED_EXPLANATION in idx.values():
                raise Refused("declined")
            pole = lambda name, i: {"explanation": f"{name} {i}", "context": "plain", "job": "one job"}
            return LLMResult({k: {"positive": pole("positive", idx[k]), "negative": pole("negative", idx[k])} for k in ids}, 10, 1)
        idx = {k: int(v) for k, v in re.findall(r"=== (t\d+) ===\nDescription: \w+ (\d+)", prompt)}
        if REFUSED_TEST in idx.values():
            raise Refused("declined")
        return LLMResult({k: {c: [{"text": f"{c} {idx[k]}"}] * 10 for c in prompts.TEST_CATEGORIES} for k in ids}, 10, 1)

    def score(self, cfg, model, tok, layer, tests, summary):
        self.tick("score " + ",".join(str(t["nid"]) for t in tests))
        return [t if "failed" in t else {**t, "blind_maximum": 7, "near_miss": 1, "distant": 0} for t in tests]


def kill_case(root):
    cfg = replace(Config.tiny(root), explain_batch=4)
    save_summary(cfg, make_summary(cfg, live_only(12)))
    return cfg


def test_a_kill_at_any_paid_call_repeats_nothing_on_resume(tmp_path, monkeypatch):
    def layer_run(cfg, name):
        return runner.run_layer(cfg, runner._BaseModel(), Status(cfg.root / f"{name}.json"), 15)

    clean = FakeClaude(monkeypatch)
    want = layer_run(kill_case(tmp_path / "clean"), "status")
    assert len(clean.sent) > 10 and any(s.startswith("score") for s in clean.sent)
    assert {r["idx"] for r in want["records"] if "failed" in r} == {REFUSED_EXPLANATION, REFUSED_TEST}

    for kill_at in range(len(clean.sent)):
        cfg = kill_case(tmp_path / f"kill_{kill_at}")
        killed = FakeClaude(monkeypatch, kill_at)
        with pytest.raises(Killed):
            layer_run(cfg, "status")
        resumed = FakeClaude(monkeypatch)
        got = layer_run(cfg, "status")
        answered = Counter(killed.sent[:-1])     # the killed call was never answered, so sending it again is not a repeat
        assert answered + Counter(resumed.sent) == Counter(clean.sent), f"killed at call {kill_at}"
        assert got == want, f"killed at call {kill_at}"

        finished = FakeClaude(monkeypatch)
        layer_run(cfg, "status")
        assert finished.sent == [], f"killed at call {kill_at}"


# --- command line ---

@pytest.fixture
def cli_cfg(tiny_cfg, monkeypatch):
    monkeypatch.setattr(cli, "Config", lambda: tiny_cfg)
    return tiny_cfg


@pytest.mark.parametrize("result", ["complete", "waiting_for_pilot_approval"])
def test_run_exits_zero_when_finished_or_waiting(cli_cfg, monkeypatch, result):
    monkeypatch.setattr(runner, "run", lambda cfg: result)
    assert cli.main(["run"]) == 0


def test_run_error_exits_nonzero_with_one_line(cli_cfg, monkeypatch, capsys):
    def fail(cfg):
        raise FingerprintMismatch("fingerprint differs in: corpus")

    monkeypatch.setattr(runner, "run", fail)
    assert cli.main(["run"]) != 0
    err = capsys.readouterr().err
    assert "corpus" in err and len(err.strip().splitlines()) == 1


def test_already_running_exits_nonzero_with_a_clear_message(cli_cfg, monkeypatch, capsys):
    def busy(cfg):
        raise AlreadyRunning(str(cfg.root / "run.lock"))

    monkeypatch.setattr(runner, "run", busy)
    assert cli.main(["run"]) != 0
    assert "already running" in capsys.readouterr().err


def test_error_traceback_is_appended_to_the_log(cli_cfg, monkeypatch, capsys):
    def fail(cfg):
        raise RuntimeError("first\nsecond line")

    monkeypatch.setattr(runner, "run", fail)
    assert cli.main(["run"]) != 0 and cli.main(["run"]) != 0
    assert capsys.readouterr().err.count("\n") == 2     # one line per failed run
    log = (cli_cfg.root / "errors.log").read_text()
    assert log.count("Traceback (most recent call last)") == 2 and "RuntimeError: first" in log and "in fail" in log


def test_approval_message_gives_the_full_restart_command(cli_cfg, capsys):
    atomic_write_json(pilot.pilot_dir(cli_cfg) / "report.json", pilot.prompt_parts(cli_cfg))
    cli.main(["approve-pilot"])
    assert "nix-shell --run 'python -m probe.atlas run'" in capsys.readouterr().out


def test_machine_safety_settings_are_set_before_heavy_imports():
    code = """
import importlib.abc, json, os, sys
NAMES = ("OMP_NUM_THREADS", "RAYON_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF")
seen = {}
class Watch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name in ("numpy", "torch") and not seen:
            seen.update({k: os.environ.get(k) for k in NAMES})
        return None
sys.meta_path.insert(0, Watch())
import probe.atlas.__main__
print(json.dumps(seen))
"""
    names = ("OMP_NUM_THREADS", "RAYON_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF")
    env = {k: v for k, v in os.environ.items() if k not in names}
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, env=env).stdout
    assert json.loads(out) == {"OMP_NUM_THREADS": "4", "RAYON_NUM_THREADS": "8",
                               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def test_approve_pilot_without_a_report_exits_nonzero(cli_cfg, capsys):
    assert cli.main(["approve-pilot"]) != 0
    assert "pilot" in capsys.readouterr().err


def test_approve_pilot_writes_the_approval(cli_cfg):
    atomic_write_json(pilot.pilot_dir(cli_cfg) / "report.json", pilot.prompt_parts(cli_cfg))
    assert cli.main(["approve-pilot"]) == 0
    assert pilot.is_approved(cli_cfg)


def test_status_prints_stage_and_overall_percent(cli_cfg, capsys):
    st = Status(cli_cfg.root / "status.json")
    st.stage("record", 5, 10, "tokens")
    st.flush()
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "record" in out and "%" in out


def test_status_before_any_run(cli_cfg, capsys):
    assert cli.main(["status"]) == 0
    assert "No run" in capsys.readouterr().out


def test_status_command_does_not_import_torch(tiny_cfg):
    code = ("import sys; from pathlib import Path; from probe.atlas import __main__ as m; from probe.atlas.config import Config;"
            f"m.Config = lambda: Config.tiny(Path({str(tiny_cfg.root)!r})); m.main(['status']);"
            "sys.exit('torch' in sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


# ---- final review fix M4: the run clears a dead run's pause when it starts, and marks itself stopped at exit ----
def test_run_clears_a_stale_pause_and_marks_itself_stopped(stubbed_runner):
    import os
    path = stubbed_runner.cfg.root / "status.json"
    atomic_write_json(path, {"pause": {"reason": "network: down", "resumes_at": 1}, "pid": 999999,
                             "call": {"since": 1}})
    seen = []
    real = runner._run_stages

    def spy(cfg, status, base):
        seen.append(read_json(path))
        return real(cfg, status, base)

    stubbed_runner.monkeypatch.setattr(runner, "_run_stages", spy)
    stubbed_runner.run()
    assert seen[0]["pause"] is None and seen[0]["call"] is None and seen[0]["pid"] == os.getpid()
    d = read_json(path)
    assert d["pid"] is None and d["pause"] is None


# ---- final review fixes I2 and M5: a stopped run says why in errors.log; python -m installs the signal handlers ----
def test_run_stopped_writes_a_plain_line_to_errors_log(cli_cfg, monkeypatch, capsys):
    from probe.atlas.llm import RunStopped

    def down(cfg):
        raise RunStopped("Claude has been unreachable for 12 h (since 2026-11-19 03:00); last error: x. Fix the network, then rerun")

    monkeypatch.setattr(runner, "run", down)
    assert cli.main(["run"]) != 0
    assert "unreachable for 12 h" in capsys.readouterr().err
    log = (cli_cfg.root / "errors.log").read_text().splitlines()
    assert log[1].startswith("RunStopped: Claude has been unreachable for 12 h")


def test_python_dash_m_installs_sigterm_and_sighup_handlers(tmp_path):
    import signal as sg
    code = ("import runpy, signal, sys; from probe.atlas import __main__ as m;"
            "m.main = lambda argv=None: 0;"
            "sys.argv = ['x', 'status'];"
            "import probe.atlas.__main__ as mm;"
            "src = open(mm.__file__).read().replace('sys.exit(main())', 'pass');"
            "g = {'__name__': '__main__', '__file__': mm.__file__}; exec(compile(src, mm.__file__, 'exec'), g);"
            "print(signal.getsignal(signal.SIGTERM) is g['_exit_on_signal'], signal.getsignal(signal.SIGHUP) is g['_exit_on_signal'])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.stdout.split() == ["True", "True"], out.stderr
