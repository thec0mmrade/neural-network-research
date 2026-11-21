"""Runs the atlas stages in order, resuming after any stop. The full run never starts before the user approves the pilot."""
from __future__ import annotations

import gc
import shutil

import numpy as np

from probe.atlas import pilot
from probe.atlas.config import Config
from probe.atlas.pilot import PHASES
from probe.atlas.status import STAGES, Status
from probe.atlas.store import FingerprintMismatch, atomic_write_json, check_fingerprint, read_json, run_lock
from probe.weights import REVISION

WAITING = "waiting_for_pilot_approval"
COMPLETE = "complete"
LLM_STAGES = ("pilot", "layers", "research", "report")     # their results belong to one prompt version
MODEL_STAGES = ("thinking", "record", "research")     # the pilot and the layer run load the model only while scoring
WAITING_NOTE = "Waiting for the pilot to be reviewed and approved"


def _load_base():
    from probe.model import load_model

    return load_model()


def _empty_gpu() -> None:
    import torch

    gc.collect()     # the model's modules can hold reference cycles
    torch.cuda.empty_cache()


def _tokenizer():
    from probe.atlas.explain import _tokenizer as load

    return load()


class _BaseModel:
    """The only owner of the base model. Callers must not keep their own reference across `free`."""

    def __init__(self):
        self._loaded = None

    def get(self):
        if self._loaded is None:
            self._loaded = _load_base()
        return self._loaded

    def free(self) -> None:
        if self._loaded is not None:
            self._loaded = None
            _empty_gpu()


def _layer_dir(cfg: Config, layer: int):
    return cfg.root / "layers" / cfg.prompt_version / f"L{layer}"


def _update_scores(cfg: Config, status, summary) -> None:
    """Report the scores over every finished layer; unfinished layers do not count as zero (R26)."""
    from probe.atlas.evaluate import summarize

    finished = {l: read_json(_layer_dir(cfg, l) / "scores.json") for l in cfg.layer_order}
    finished = {l: stored for l, stored in finished.items() if stored is not None}
    records = [r for stored in finished.values() for r in stored["records"]]
    scores = summarize(records, summary, layers=set(finished))
    status.scores(scores["primary"], scores["ac_matched"])


def run_layer(cfg: Config, base, status, layer: int) -> dict:
    """Explain, test and score every live neuron of `layer`, then write its `scores.json`. A finished layer is only read back."""
    from probe.atlas.evaluate import summarize

    summary = pilot.load_summary(cfg)
    counts = {l: int(np.count_nonzero(summary[f"L{l}.live"])) for l in cfg.layer_order}
    n = counts[layer]
    total = PHASES * sum(counts.values())
    start = PHASES * sum(counts[l] for l in cfg.layer_order[:cfg.layer_order.index(layer)])

    out = _layer_dir(cfg, layer)
    result = read_json(out / "scores.json")
    if result is None:
        progress = pilot.Progress(status, "layers", layer, start, total, n, show_layer=True)
        idxs = [int(i) for i in np.flatnonzero(summary[f"L{layer}.live"])]
        records, _ = pilot.explain_test_score(cfg, base, progress, layer, idxs, summary, out)
        result = {"layer": layer, "records": records, "summary": summarize(records, summary, layers={layer})}
        atomic_write_json(out / "scores.json", result)
    status.layer(layer, "done", n, n, result["summary"]["primary"]["blind_maximum"])
    status.stage("layers", start + PHASES * n, total, "steps")
    _update_scores(cfg, status, summary)
    return result


def _corpus(cfg: Config, status, base: _BaseModel) -> None:
    from probe.atlas.corpus import build_sources

    build_sources(cfg, _tokenizer(), status)


def _thinking(cfg: Config, status, base: _BaseModel) -> None:
    from probe.atlas.thinking import generate_thinking

    generate_thinking(cfg, *base.get(), status)


def _assemble(cfg: Config, status, base: _BaseModel) -> None:
    from probe.atlas.corpus import assemble

    assemble(cfg, status)


def _record(cfg: Config, status, base: _BaseModel) -> None:
    from probe.atlas.record import record_p50c, record_pass

    out = cfg.root / "record"
    if (out / "summary.npz").exists():
        shutil.rmtree(out / "ckpt", ignore_errors=True)     # a kill can land between the summary and record_pass's own cleanup
    else:
        record_pass(cfg, *base.get(), status, out, cfg.layers)
    base.free()     # record_p50c needs the GPU to itself (R17); the next stage that needs the base model loads it again
    record_p50c(cfg, status)


def _pilot(cfg: Config, status, base: _BaseModel) -> None:
    pilot.run_pilot(cfg, base, status)


def _layers(cfg: Config, status, base: _BaseModel) -> None:
    for layer in cfg.layer_order:
        run_layer(cfg, base, status, layer)


def _research(cfg: Config, status, base: _BaseModel) -> None:
    from probe.atlas.research import run_research

    run_research(cfg, base, status)


def _report(cfg: Config, status, base: _BaseModel) -> None:
    from probe.atlas.report import write_report

    write_report(cfg)


_STAGE_WORK = {"corpus": _corpus, "thinking": _thinking, "assemble": _assemble, "record": _record,
               "pilot": _pilot, "layers": _layers, "research": _research, "report": _report}


def _check_run_fingerprint(cfg: Config) -> None:
    """Refuse to resume over a different corpus, model or example window. Before the corpus exists there is nothing to compare."""
    manifest = read_json(cfg.root / "corpus" / "manifest.json")
    if manifest is None:
        return
    parts = {"corpus": manifest["hash"], "model": REVISION, "llm_window": cfg.llm_window}
    try:
        check_fingerprint(cfg.root / "fingerprint.json", parts)
    except FingerprintMismatch as e:
        raise FingerprintMismatch(f"refusing to resume from {cfg.root}: {e}") from None


def _check_prompt_fingerprint(cfg: Config) -> None:
    parts = pilot.prompt_parts(cfg)
    try:
        check_fingerprint(pilot.pilot_dir(cfg) / "fingerprint.json", {"prompt": f"{parts['prompt_version']}:{parts['prompt_hash']}",
                                                                      "effort": parts["effort"], "model": parts["model"],
                                                                      "test_model": parts["test_model"]})
    except FingerprintMismatch as e:
        raise FingerprintMismatch(f"{e}, but prompt_version is still {cfg.prompt_version!r}; set a new prompt_version in config.py "
                                  "so old and new results are never mixed, then review and approve a new pilot") from None


def _done_path(cfg: Config, name: str):
    return cfg.root / "stages" / f"{name}.done"


def _marker(cfg: Config, name: str) -> dict:
    return pilot.prompt_parts(cfg) if name in LLM_STAGES else {}


def _is_done(cfg: Config, name: str) -> bool:
    return read_json(_done_path(cfg, name)) == _marker(cfg, name)


def _finish_stage(cfg: Config, status, name: str) -> None:
    st = status.stages.get(name)
    if not st or st["done"] < st["total"]:     # a stage can end short of its estimate, or report no progress at all
        status.stage(name, 1, 1, "stage")
    atomic_write_json(_done_path(cfg, name), _marker(cfg, name))
    status.flush()


def _gate_message(cfg: Config) -> str:
    return (f"The pilot is finished. Review {pilot.pilot_dir(cfg) / 'report.md'}\n"
            "and compare the /usage figure from before and after the pilot.\n"
            f"To start the full run, approve it with: {pilot.APPROVE_COMMAND}\n"
            "Then start the run again.")


def _run_stages(cfg: Config, status, base: _BaseModel) -> str:
    _check_run_fingerprint(cfg)
    for name in STAGES:
        if name == "pilot":
            _check_prompt_fingerprint(cfg)
        if name == "layers":
            if not pilot.is_approved(cfg):
                status.note(WAITING_NOTE)
                print(_gate_message(cfg))
                return WAITING
            if status.note_ == WAITING_NOTE:
                status.note(None)
        if _is_done(cfg, name):
            continue
        if name not in MODEL_STAGES:
            base.free()
        _STAGE_WORK[name](cfg, status, base)
        if name == "assemble":
            _check_run_fingerprint(cfg)
        _finish_stage(cfg, status, name)
    return COMPLETE


def run(cfg: Config) -> str:
    """Run every unfinished stage. Returns "waiting_for_pilot_approval" at the gate and "complete" at the end."""
    cfg.root.mkdir(parents=True, exist_ok=True)
    with run_lock(cfg.root / "run.lock"):
        status = Status(cfg.root / "status.json")     # inside the lock: a refused second run must not overwrite the status
        status.started()     # records this pid and clears a pause or call left by a run that died (M4)
        base = _BaseModel()
        try:
            return _run_stages(cfg, status, base)
        finally:
            base.free()
            status.closed()     # the viewer shows the run as stopped, not paused or calling
