"""The whole atlas pipeline through its real command line: real model, tokenizer and GPU, fake Claude, local web text, temporary root."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from probe.atlas import __main__ as cli
from probe.atlas import llm, prompts
from probe.atlas.config import Config
from probe.atlas.status import STAGES
from probe.atlas.store import read_json
from probe.weights import RESULTS

PV = Config().prompt_version     # results live under a folder named after it

HERE = Path(__file__).parent
DRIVER = HERE / "fixtures" / "rehearsal_cli.py"
PROJECT = HERE.parents[1]
WAITING, COMPLETE = "waiting_for_pilot_approval", "complete"
STAGE_TIMEOUT_S = 1200
MAX_PARENT_GPU_BYTES = 100 * 2**20
MIN_FRACTION = {"record": 0.4}     # far enough along that a checkpoint exists to resume from


@dataclass
class Finished:
    root: Path
    returncode: int
    gate_output: str | None = None
    layers_before_approval: bool | None = None
    second_gate_output: str | None = None
    status_at_gate: str | None = None


class TinyWorkspace:
    """One folder per named run under a temporary directory, each driven through the rehearsal command line."""

    def __init__(self, base: Path):
        self.base = base
        self.fake_modes: dict[str, str] = {}

    def root(self, name: str) -> Path:
        folder = self.base / name
        folder.mkdir(parents=True, exist_ok=True)
        wiring = folder / "06_wiring"     # read-only: lets the report's level-2 join run on the real edge table
        if not wiring.exists() and (RESULTS / "06_wiring").exists():
            wiring.symlink_to(RESULTS / "06_wiring")
        return folder / "atlas"

    def _argv(self, name: str, command: str) -> list[str]:
        return [sys.executable, str(DRIVER), str(self.root(name)), command, self.fake_modes.get(name, "atlas")]

    def cli(self, name: str, command: str) -> subprocess.CompletedProcess:
        done = subprocess.run(self._argv(name, command), cwd=PROJECT, capture_output=True, text=True, timeout=STAGE_TIMEOUT_S)
        with open(self.base / name / "commands.log", "a") as log:
            log.write(f"$ {command} -> {done.returncode}\n{done.stdout}\n{done.stderr}\n")
        return done

    def _expect(self, name: str, command: str, wanted: str) -> subprocess.CompletedProcess:
        done = self.cli(name, command)
        assert done.returncode == 0 and wanted in done.stdout, f"{command}: rc {done.returncode}\n{done.stdout}\n{done.stderr[-3000:]}"
        return done

    def run_to_gate(self, name: str) -> subprocess.CompletedProcess:
        return self._expect(name, "run", WAITING)

    def run_to_completion(self, name: str, recheck_gate: bool = False) -> Finished:
        """Run, stop at the pilot gate, approve, and run again until the atlas is complete. An approved run goes straight through."""
        root = self.root(name)
        if (root / "pilot" / "approved.json").exists():
            self._expect(name, "run", COMPLETE)
            return Finished(root, 0)
        gate = self.run_to_gate(name)
        second = self.run_to_gate(name).stdout if recheck_gate else None     # without approval, another run must stop again
        status = self.cli(name, "status").stdout if recheck_gate else None
        layers_before = (root / "layers").exists()
        self._expect(name, "approve-pilot", "Pilot approved")
        self._expect(name, "run", COMPLETE)
        return Finished(root, 0, gate.stdout, layers_before, second, status)

    def run_and_sigkill(self, name: str, during: str) -> dict:
        """SIGKILL a run once `during` has reported progress in status.json, and prove that the stage had not finished."""
        if during == "layers":
            self.run_to_gate(name)
            self._expect(name, "approve-pilot", "Pilot approved")
        root = self.root(name)
        with open(self.base / name / "killed_run.log", "w") as log:
            proc = subprocess.Popen(self._argv(name, "run"), cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            try:
                seen = self._wait_for_stage(proc, root, during)
                proc.send_signal(signal.SIGKILL)
                assert proc.wait(timeout=60) == -signal.SIGKILL
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        assert not (root / "stages" / f"{during}.done").exists(), f"the kill landed after {during} had finished"
        return seen

    def _wait_for_stage(self, proc: subprocess.Popen, root: Path, stage: str) -> dict:
        deadline = time.monotonic() + STAGE_TIMEOUT_S
        while time.monotonic() < deadline:
            status = read_json(root / "status.json")
            st = (status or {}).get("stages", {}).get(stage)
            if status and status.get("current_stage") == stage and st and st["done"] > 0 \
                    and st["done"] >= MIN_FRACTION.get(stage, 0) * st["total"]:
                return st
            assert proc.poll() is None, f"run exited with {proc.returncode} before reaching {stage}"
            time.sleep(0.05)
        raise TimeoutError(f"{stage} did not report progress")


def _same(a, b, rtol: float, where: str) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), f"{where}: keys differ: {sorted(set(a) ^ set(b))[:10]}"
        for k in a:
            _same(a[k], b[k], rtol, f"{where}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"{where}: lengths differ: {len(a)} and {len(b)}"
        for n, (x, y) in enumerate(zip(a, b)):
            _same(x, y, rtol, f"{where}[{n}]")
    elif isinstance(a, float) and isinstance(b, (int, float)) or isinstance(b, float) and isinstance(a, (int, float)):
        same = (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=rtol, abs_tol=1e-6)
        assert same, f"{where}: {a} and {b}"
    else:
        assert a == b, f"{where}: {a!r} and {b!r}"


def _same_npz(a: Path, b: Path, rtol: float) -> None:
    with np.load(a) as x, np.load(b) as y:
        assert set(x.files) == set(y.files), a.name
        for k in x.files:
            if np.issubdtype(x[k].dtype, np.floating):
                assert np.allclose(x[k], y[k], rtol=rtol, atol=1e-6, equal_nan=True), f"{a.name}:{k}"
            else:
                assert np.array_equal(x[k], y[k]), f"{a.name}:{k}"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def assert_same_atlas(clean: Path, resumed: Path, rtol: float = 1e-3) -> None:
    """Same corpus and activations within `rtol`; same explanations, tests and scores."""
    assert read_json(clean / "corpus" / "manifest.json")["hash"] == read_json(resumed / "corpus" / "manifest.json")["hash"]
    for folder in ("record", "record_p50c"):
        names = sorted(p.name for p in (clean / folder).glob("*.npz"))
        assert names == sorted(p.name for p in (resumed / folder).glob("*.npz")) and names
        for name in names:
            _same_npz(clean / folder / name, resumed / folder / name, rtol)

    def report(root: Path) -> dict:
        found = read_json(root / "pilot" / PV / "report.json")
        return {k: found[k] for k in ("neurons", "failed", "scores", "control", "samples")}     # usage and time differ after a kill

    _same(report(clean), report(resumed), rtol, "pilot")
    layers = sorted(p.name for p in (clean / "layers" / PV).iterdir())
    assert layers == sorted(p.name for p in (resumed / "layers" / PV).iterdir()) and layers
    for layer in layers:
        _same(read_json(clean / "layers" / PV / layer / "scores.json"), read_json(resumed / "layers" / PV / layer / "scores.json"), rtol, layer)
    _same(read_json(clean / "research" / PV / "research.json"), read_json(resumed / "research" / PV / "research.json"), rtol, "research")
    _same(_jsonl(clean / "atlas_neurons.jsonl"), _jsonl(resumed / "atlas_neurons.jsonl"), rtol, "atlas")
    assert (clean.parent / "07_atlas.md").read_text() == (resumed.parent / "07_atlas.md").read_text()


@pytest.fixture(scope="session")
def real_outputs():
    """What the real output paths look like before any rehearsal runs."""
    def snapshot():
        return {p: p.stat().st_mtime_ns if p.exists() else None for p in (RESULTS / "07_atlas", RESULTS / "07_atlas.md")}

    return snapshot, snapshot()


def require_free_gpu(allocated_bytes: int) -> None:
    """A model left loaded by this test process would make the rehearsal's recording run out of GPU memory."""
    if allocated_bytes > MAX_PARENT_GPU_BYTES:
        raise RuntimeError(f"the test process still holds {allocated_bytes / 2**30:.2f} GiB of GPU memory; the rehearsal needs the GPU to itself")


@pytest.fixture(scope="session")
def tiny_workspace(tmp_path_factory, real_outputs):
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    require_free_gpu(torch.cuda.memory_allocated())
    return TinyWorkspace(tmp_path_factory.mktemp("rehearsal"))


@pytest.fixture(scope="session")
def clean(tiny_workspace):
    started = time.monotonic()
    finished = tiny_workspace.run_to_completion("clean", recheck_gate=True)
    print(f"\nclean tiny run: {time.monotonic() - started:.0f} s")
    return finished


@pytest.mark.gpu
def test_clean_run_stops_at_the_gate_then_finishes_every_stage(tiny_workspace, clean):
    assert WAITING in clean.gate_output and not clean.layers_before_approval
    for stage in STAGES:
        assert (clean.root / "stages" / f"{stage}.done").exists(), stage
    rows = _jsonl(clean.root / "atlas_neurons.jsonl")
    assert len(rows) == 7 * 3584
    live = [r for r in rows if r["status"] == "live"]
    assert live and all(r["scores"] and r["positive"]["explanation"] for r in live)
    assert any(r["research"] for r in rows)
    assert all(r["wiring"] for r in live) == (RESULTS / "06_wiring" / "edges_raw.npz").exists()
    report = (clean.root.parent / "07_atlas.md").read_text()
    assert "## Research question" in report and "No result" not in report


@pytest.mark.gpu
def test_a_second_run_before_approval_stays_at_the_gate(clean):
    assert WAITING in clean.second_gate_output and not clean.layers_before_approval
    assert "Stage: pilot" in clean.status_at_gate


@pytest.mark.gpu
def test_viewer_renders_a_real_status_file(clean):
    from rich.console import Console

    from probe.atlas_watch import load_status, render

    console = Console(record=True, width=140, height=50, file=open(os.devnull, "w"))
    console.print(render(load_status(clean.root / "status.json"), time.time(), 140))
    text = console.export_text()
    assert "viewer error" not in text and all(stage in text for stage in ("corpus", "record", "layers", "research"))


@pytest.mark.gpu
@pytest.mark.parametrize("stage", ["corpus", "thinking", "record", "pilot", "layers"])
def test_sigkill_in_each_stage_matches_clean_run(stage, tiny_workspace, clean):
    name = f"killed_{stage}"
    tiny_workspace.run_and_sigkill(name, during=stage)
    resumed = tiny_workspace.run_to_completion(name)
    assert resumed.returncode == 0
    assert_same_atlas(clean.root, resumed.root, rtol=1e-3)


@pytest.mark.gpu
def test_neurons_claude_refuses_are_reported_as_failed_and_the_run_still_finishes(tiny_workspace):
    tiny_workspace.fake_modes["refusals"] = "atlas_refuse"
    finished = tiny_workspace.run_to_completion("refusals")
    rows = _jsonl(finished.root / "atlas_neurons.jsonl")
    failed = [r for r in rows if r["status"] == "failed"]
    live = [r for r in rows if r["status"] == "live"]
    assert failed and live, "the rehearsal needs both refused and answered neurons"
    assert all(r["failed"] and r["scores"] is None for r in failed)
    assert all(r["scores"] is not None for r in live)
    pilot_report = read_json(finished.root / "pilot" / PV / "report.json")
    assert pilot_report["scores"]["primary"]["n"] == pilot_report["neurons"]
    research = read_json(finished.root / "research" / PV / "research.json")
    assert sum(sum(research["failed"][g].values()) for g in ("study", "control")) > 0
    text = (finished.root.parent / "07_atlas.md").read_text()
    overall = next(line for line in text.splitlines() if line.startswith("| Overall |"))
    assert int(overall.split("|")[5]) == len(failed)


@pytest.mark.gpu
def test_a_rehearsal_leaves_the_real_outputs_alone(clean, real_outputs):
    snapshot, before = real_outputs
    assert snapshot() == before


def _load_driver():
    spec = importlib.util.spec_from_file_location("rehearsal_cli", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_normal_command_line_uses_the_real_configuration(monkeypatch):
    seen = []
    monkeypatch.setitem(cli.COMMANDS, "status", lambda cfg: seen.append(cfg) or 0)
    monkeypatch.setenv("PROBE_ATLAS_REHEARSAL", "1")
    assert cli.main(["status"]) == 0
    assert seen == [Config()]
    assert seen[0].root == RESULTS / "07_atlas" and seen[0].claude_bin == "claude"
    assert seen[0].total_tokens == 50_000_000


def test_the_rehearsal_refuses_to_share_the_gpu_with_the_test_process():
    require_free_gpu(MAX_PARENT_GPU_BYTES)
    with pytest.raises(RuntimeError, match="GPU to itself"):
        require_free_gpu(2 * 2**30)


def test_the_real_entry_point_starts_and_status_writes_nothing():
    before = (RESULTS / "07_atlas").exists()
    done = subprocess.run([sys.executable, "-m", "probe.atlas", "status"], cwd=PROJECT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert (RESULTS / "07_atlas").exists() == before


def test_the_real_web_reader_has_its_libraries():
    import huggingface_hub
    import pyarrow.parquet

    assert huggingface_hub.HfFileSystem and pyarrow.parquet.ParquetFile


def test_no_switch_in_the_package_can_enable_a_rehearsal():
    for path in (PROJECT / "src" / "probe" / "atlas").glob("*.py"):
        text = path.read_text()
        assert "rehearsal" not in text.lower(), path.name
        assert not re.search(r"environ(\.get)?[\[(]|getenv", text), f"{path.name} reads the environment"


def test_the_rehearsal_refuses_the_real_results_folder():
    with pytest.raises(SystemExit):
        _load_driver().rehearsal_config(RESULTS / "07_atlas")


def test_local_web_text_is_reproducible_and_long_enough():
    driver = _load_driver()
    read = lambda: [d for _, d in zip(range(5), driver.local_docs("repo", "data/fra_Latn/train/000_00000.parquet", "rev"))]
    assert read() == read() and all(len(d.split()) >= 60 for d in read())


@pytest.mark.parametrize("kind", ["explain", "test", "rate"])
def test_fake_claude_answers_each_request_with_a_schema_valid_reply(kind, fake_cfg):
    cfg = fake_cfg("atlas")
    if kind == "explain":
        ids = ["n01", "n02"]
        prompt = prompts.explain_prompt([(i, f"Positive pole: strongest value 3\nP1: word{i}[1.5] next") for i in ids])
        schema, system = prompts.explain_schema(ids), prompts.EXPLAIN_SYSTEM
    elif kind == "test":
        ids = ["t01", "t02"]
        prompt = prompts.test_prompt([(i, {"explanation": f"fires on {i}", "context": "plain"}) for i in ids])
        schema, system = prompts.test_schema(ids), prompts.TEST_SYSTEM
    else:
        ids = ["r001", "r002"]
        prompt = prompts.rate_prompt([(i, {"explanation": f"fires on {i}"}) for i in ids])
        schema, system = prompts.rate_schema(ids), prompts.RATE_SYSTEM
    data = llm.call_claude(cfg, system, prompt, schema).data
    if kind == "explain":
        assert set(data) == set(ids) and all(data[i]["positive"]["job"] in prompts.JOBS for i in ids)
    elif kind == "test":
        prompts.validate_tests(data, ids)
    else:
        prompts.validate_ratings(data, ids)
