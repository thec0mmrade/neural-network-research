"""The recording pass. CPU tests drive `record_pass` with a tiny fake model; gpu tests use the real one.

Run directly, this file is the child process of two gpu tests: `<root> <out>` records, `p50c <root>` runs `record_p50c`.
"""
import json
import subprocess
import sys
import time
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from probe.atlas.config import Config
from probe.atlas.corpus import assemble, iter_docs, save_source
from probe.atlas.record import _keep_columns, edited_neurons, plan_batches, record_p50c, record_pass
from probe.atlas.recorder import LayerRecorder
from probe.atlas.status import Status
from probe.atlas.store import FingerprintMismatch, load_latest_checkpoint

NAN_TOKEN = 13
LAYERS = (15, 19)
P50C_TEST_TOKENS = 3000


def rng_lengths(n, seed=0, low=32, high=4096):
    return np.random.default_rng(seed).integers(low, high + 1, n).tolist()


def max_len(group, docs):
    return max(len(docs[i][1]) for i in group)


class FakeMLP(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.gate_proj = nn.Linear(d, n, bias=False)
        self.up_proj = nn.Linear(d, n, bias=False)
        self.down_proj = nn.Linear(n, d, bias=False)
        self.act_fn = nn.functional.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FakeLayer(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.mlp = FakeMLP(d, n)

    def forward(self, x):
        return x + self.mlp(x)


class FakeBody(nn.Module):
    def __init__(self, d, n, n_layers):
        super().__init__()
        self.embed = nn.Embedding(100, d)
        self.embed.weight.data[NAN_TOKEN] = torch.nan
        self.layers = nn.ModuleList(FakeLayer(d, n) for _ in range(n_layers))

    def forward(self, input_ids, attention_mask, use_cache):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x


def fake_model(cfg):
    torch.manual_seed(0)
    return SimpleNamespace(model=FakeBody(8, cfg.n_neurons, max(LAYERS) + 2), parameters=lambda: iter([torch.zeros(1)]))


FAKE_TOK = SimpleNamespace(all_special_ids=[0], pad_token_id=0,
                           convert_tokens_to_ids={"<think>": 1, "</think>": 2, "<|im_start|>": 3, "<|im_end|>": 4}.get)


def build_corpus(cfg, docs):
    save_source(cfg, "test", [list(map(int, d)) for d in docs], {})
    assemble(cfg, SimpleNamespace(stage=lambda *a: None))


@pytest.fixture
def small(tmp_path):
    """A six-neuron config over a small corpus with a checkpoint after every batch."""
    cfg = replace(Config.tiny(tmp_path / "atlas"), n_neurons=6, top_k=4, dn_k=2, random_k=4, window=8,
                  batch_docs=4, checkpoint_s=0)
    rng = np.random.default_rng(1)
    build_corpus(cfg, [rng.integers(5, 13, int(n)) for n in rng.integers(32, 80, 40)])
    return cfg


class Crash(Exception):
    pass


class CrashingStatus(Status):
    """Dies during the progress report of batch `at` (1-based), so that batch is finished but not checkpointed."""

    def __init__(self, path, at):
        super().__init__(path)
        self.at, self.calls = at, 0

    def stage(self, *args):
        super().stage(*args)
        self.calls += 1
        if self.calls == self.at + 1:    # the first call reports the starting position
            raise Crash


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def assert_recordings_equal(a, b, rtol):
    names = sorted(p.name for p in Path(a).glob("*.npz"))
    assert names and names == sorted(p.name for p in Path(b).glob("*.npz"))
    for name in names:
        x, y = load_npz(Path(a) / name), load_npz(Path(b) / name)
        assert sorted(x) == sorted(y)
        for k in x:
            assert x[k].shape == y[k].shape, f"{name}:{k}"
            if np.issubdtype(x[k].dtype, np.floating):
                assert np.allclose(x[k], y[k], rtol=rtol, atol=0, equal_nan=True), f"{name}:{k}"
            else:
                assert np.array_equal(x[k], y[k]), f"{name}:{k}"


def test_batches_respect_limits_and_cover_all(tiny_cfg):
    docs = [(i, np.ones(n, int)) for i, n in enumerate(rng_lengths(500))]
    b = plan_batches(tiny_cfg, docs, None)
    assert sorted(x for g in b for x in g) == list(range(500))
    assert all(len(g) <= 32 and max_len(g, docs) * len(g) <= tiny_cfg.batch_tokens for g in b)


def test_plan_is_deterministic_and_sorts_within_windows_of_256(tiny_cfg):
    docs = [(i, np.ones(n, int)) for i, n in enumerate(rng_lengths(600))]
    b = plan_batches(tiny_cfg, docs, None)
    assert b == plan_batches(tiny_cfg, list(docs), None)
    flat = [x for g in b for x in g]
    assert sorted(flat[:256]) == list(range(256))     # the first window is used up before the second starts
    lengths = [len(docs[i][1]) for i in flat[:256]]
    assert lengths == sorted(lengths)


def test_plan_enforces_a_small_padded_token_limit(tiny_cfg):
    cfg = replace(tiny_cfg, batch_tokens=1000)
    docs = [(i, np.ones(n, int)) for i, n in enumerate(rng_lengths(300, high=900))]
    assert all(max_len(g, docs) * len(g) <= 1000 for g in plan_batches(cfg, docs, None))


def test_plan_rejects_a_document_that_cannot_fit(tiny_cfg):
    with pytest.raises(ValueError, match="batch_tokens"):
        plan_batches(replace(tiny_cfg, batch_tokens=100), [(0, np.ones(101, int))], None)


def test_plan_stops_reading_at_the_token_limit(tiny_cfg):
    read = []

    def docs():
        for i in range(1000):
            read.append(i)
            yield i, np.ones(100, int)

    b = plan_batches(tiny_cfg, docs(), 1050)
    assert sorted(x for g in b for x in g) == list(range(11)) and len(read) == 11


def test_edited_neurons_are_64_per_layer():
    e = edited_neurons("p50c")
    assert sorted(e) == [15, 19] and all(len(v) == 64 for v in e.values())
    assert all(np.all(np.diff(v) > 0) and v[0] >= 0 and v[-1] < 3584 for v in e.values())


def test_record_pass_writes_every_output_and_removes_checkpoints(small, tmp_path):
    out = tmp_path / "rec"
    status = Status(tmp_path / "status.json")
    record_pass(small, fake_model(small), FAKE_TOK, status, out, LAYERS)
    for layer in LAYERS:
        state = load_npz(out / f"L{layer}.npz")
        assert state["pos_val"].shape == (6, 4) and state["hist"].shape == (6, 4, 256)
    summary = load_npz(out / "summary.npz")
    assert {f"L{l}.{k}" for l in LAYERS for k in ("live", "dominant", "fire_share")} <= set(summary)
    assert not (out / "ckpt").exists()
    total = sum(len(d) for _, d in iter_docs(small))
    assert status.stages["record"]["done"] == total
    assert status.stages["record"]["total"] == total + small.p50c_tokens     # the p50c pass follows in the same stage


def test_positions_are_counted_once_each(small, tmp_path):
    out = tmp_path / "rec"
    record_pass(small, fake_model(small), FAKE_TOK, Status(tmp_path / "s.json"), out, LAYERS)
    eligible = sum(len(d) - 1 for _, d in iter_docs(small))     # index 0 of each document never counts
    assert load_npz(out / "L15.npz")["count"].tolist() == [eligible] * 6


def test_nonfinite_positions_are_counted_and_skipped(tmp_path):
    cfg = replace(Config.tiny(tmp_path / "atlas"), n_neurons=6, top_k=4, dn_k=2, random_k=4, window=8, checkpoint_s=0)
    docs = [[5] * 40 for _ in range(6)]
    docs[2][7] = docs[2][9] = docs[4][20] = NAN_TOKEN
    build_corpus(cfg, docs)
    out = tmp_path / "rec"
    record_pass(cfg, fake_model(cfg), FAKE_TOK, Status(tmp_path / "s.json"), out, LAYERS)
    meta = json.loads((out / "record_meta.json").read_text())
    assert meta["nonfinite"] == {"15": 3, "19": 3} and meta["nonfinite_total"] == 6
    assert load_npz(out / "L15.npz")["count"].tolist() == [6 * 39 - 3] * 6


def test_crash_then_resume_matches_a_clean_run(small, tmp_path):
    clean, crashed = tmp_path / "clean", tmp_path / "crashed"
    record_pass(small, fake_model(small), FAKE_TOK, Status(tmp_path / "a.json"), clean, LAYERS)
    with pytest.raises(Crash):
        record_pass(small, fake_model(small), FAKE_TOK, CrashingStatus(tmp_path / "b.json", 5), crashed, LAYERS)
    assert load_latest_checkpoint(crashed / "ckpt")[0] == 4     # batch 5 ran, but its checkpoint was never saved
    record_pass(small, fake_model(small), FAKE_TOK, Status(tmp_path / "b.json"), crashed, LAYERS)
    assert json.loads((crashed / "record_meta.json").read_text())["resumed_from"] == 4
    assert json.loads((clean / "record_meta.json").read_text())["resumed_from"] == 0
    assert_recordings_equal(clean, crashed, rtol=0)


def test_nonfinite_count_survives_a_resume(tmp_path):
    cfg = replace(Config.tiny(tmp_path / "atlas"), n_neurons=6, top_k=4, dn_k=2, random_k=4, window=8, batch_docs=1,
                  checkpoint_s=0)
    docs = [[5] * 40 for _ in range(6)]
    for d in docs:
        d[7] = NAN_TOKEN
    build_corpus(cfg, docs)
    out = tmp_path / "rec"
    with pytest.raises(Crash):
        record_pass(cfg, fake_model(cfg), FAKE_TOK, CrashingStatus(tmp_path / "a.json", 4), out, LAYERS)
    record_pass(cfg, fake_model(cfg), FAKE_TOK, Status(tmp_path / "b.json"), out, LAYERS)
    meta = json.loads((out / "record_meta.json").read_text())
    assert meta["resumed_from"] == 3 and meta["nonfinite"] == {"15": 6, "19": 6}


def oom_in_forward(model, batches, layer=17):
    """Make the forward run out of memory at `layer`, after L15 is recorded, on the listed forward calls (1-based)."""
    calls = [0]

    def hook(module, args):
        calls[0] += 1
        if calls[0] in batches:
            raise torch.cuda.OutOfMemoryError("injected")

    model.model.layers[layer].register_forward_pre_hook(hook)
    return model


def test_oom_in_the_forward_is_retried_to_the_same_result(small, tmp_path):
    clean, retried = tmp_path / "clean", tmp_path / "retried"
    record_pass(small, fake_model(small), FAKE_TOK, Status(tmp_path / "a.json"), clean, LAYERS)
    waits, notes = [], []
    status = Status(tmp_path / "b.json")

    def wait(seconds):
        waits.append(seconds)
        notes.append(status.note_)

    model = oom_in_forward(fake_model(small), {3, 4, 5})      # three failures in a row on the third batch
    record_pass(small, model, FAKE_TOK, status, retried, LAYERS, wait=wait)
    assert waits == [60, 60, 60]
    assert_recordings_equal(clean, retried, rtol=0)
    for retry, note in enumerate(notes, 1):     # the viewer explains the stall while it lasts
        assert "Batch 3 of" in note and "GPU memory" in note and f"retry {retry} of 3" in note and "60 s" in note
    assert "Batch 3 of" in status.note_ and "recovered" in status.note_     # and does not keep reporting a stall that ended


def test_oom_halfway_through_an_update_is_rolled_back(small, tmp_path, monkeypatch):
    clean, retried = tmp_path / "clean", tmp_path / "retried"
    record_pass(small, fake_model(small), FAKE_TOK, Status(tmp_path / "a.json"), clean, LAYERS)
    merge, calls = LayerRecorder._merge, [0]

    def failing_merge(self, *args):     # by now `update` has already added this batch to the histograms and totals
        calls[0] += 1
        if calls[0] == 8:
            raise torch.cuda.OutOfMemoryError("injected")
        return merge(self, *args)

    monkeypatch.setattr(LayerRecorder, "_merge", failing_merge)
    waits = []
    record_pass(small, fake_model(small), FAKE_TOK, Status(tmp_path / "b.json"), retried, LAYERS, wait=waits.append)
    assert waits == [60]
    assert_recordings_equal(clean, retried, rtol=0)


def test_persistent_oom_stops_clearly_and_keeps_the_checkpoint(small, tmp_path):
    clean, out = tmp_path / "clean", tmp_path / "rec"
    record_pass(small, fake_model(small), FAKE_TOK, Status(tmp_path / "a.json"), clean, LAYERS)
    waits = []
    model = oom_in_forward(fake_model(small), set(range(4, 1000)))      # the fourth batch and every attempt after it
    with pytest.raises(RuntimeError, match=r"batch 4 of \d+ .*after 3 retries.*checkpoint.*intact"
                                           r".*Free GPU memory \(close games and other GPU apps\) and rerun"
                                           r".*Do NOT change batch_tokens.*invalidates the recording checkpoint") as failure:
        record_pass(small, model, FAKE_TOK, Status(tmp_path / "b.json"), out, LAYERS, wait=waits.append)
    assert not isinstance(failure.value, torch.cuda.OutOfMemoryError) and waits == [60, 60, 60]
    assert load_latest_checkpoint(out / "ckpt")[0] == 3 and not (out / "summary.npz").exists()
    record_pass(small, fake_model(small), FAKE_TOK, Status(tmp_path / "b.json"), out, LAYERS)
    assert json.loads((out / "record_meta.json").read_text())["resumed_from"] == 3
    assert_recordings_equal(clean, out, rtol=0)


def fake_p50c(monkeypatch, cfg):
    """Stand in for the p50c model on a GPU with nothing else loaded."""
    monkeypatch.setattr("probe.model.load_model", lambda run: (fake_model(cfg), FAKE_TOK))
    monkeypatch.setattr("probe.atlas.record.edited_neurons", lambda run: {15: np.array([1, 4]), 19: np.array([0, 2])})
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)


def refuse_to_load(run):
    raise AssertionError("load_model must not be called")


def test_record_p50c_refuses_to_load_beside_another_model(small, tmp_path, monkeypatch):
    monkeypatch.setattr("probe.model.load_model", refuse_to_load)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 2**30)
    with pytest.raises(RuntimeError, match=r"1\.00 GiB.*already allocated.*base model"):
        record_p50c(small, Status(tmp_path / "s.json"))
    assert not (small.root / "record_p50c").exists()


def test_record_p50c_frees_its_model_before_returning(small, tmp_path, monkeypatch):
    fake_p50c(monkeypatch, small)
    bodies, alive_at_empty_cache = [], []

    def load(run):
        model = fake_model(small)
        bodies.append(weakref.ref(model.model))
        return model, FAKE_TOK

    monkeypatch.setattr("probe.model.load_model", load)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: alive_at_empty_cache.append(bodies[0]() is not None))
    record_p50c(small, Status(tmp_path / "s.json"))
    assert alive_at_empty_cache == [False]


def test_record_p50c_removes_checkpoints_left_by_a_kill_during_cleanup(small, tmp_path, monkeypatch):
    out = small.root / "record_p50c"
    record_pass(small, fake_model(small), FAKE_TOK, Status(tmp_path / "a.json"), out, LAYERS)
    (out / "ckpt").mkdir()
    (out / "ckpt" / "ckpt_00000001.npz").write_bytes(b"left behind")
    monkeypatch.setattr("probe.model.load_model", refuse_to_load)       # the recording is finished: nothing to load
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 2**30)  # so the guard has no reason to fire
    monkeypatch.setattr("probe.atlas.record.edited_neurons", lambda run: {15: np.array([1, 4]), 19: np.array([0, 2])})
    record_p50c(small, Status(tmp_path / "b.json"))
    assert not (out / "ckpt").exists()
    assert load_npz(out / "L15.npz")["neurons"].tolist() == [1, 4]


def test_record_stage_covers_both_passes_and_never_moves_back(small, tmp_path, monkeypatch):
    cfg = replace(small, p50c_tokens=300)
    seen = []

    class Recording(Status):
        def stage(self, name, done, total, unit):
            seen.append((name, done, total))
            super().stage(name, done, total, unit)

    fake_p50c(monkeypatch, cfg)
    status = Recording(tmp_path / "s.json")
    record_pass(cfg, fake_model(cfg), FAKE_TOK, status, tmp_path / "rec", LAYERS)
    corpus = sum(len(d) for _, d in iter_docs(cfg))
    assert seen[-1] == ("record", corpus, corpus + 300)
    main_calls = len(seen)
    record_p50c(cfg, status)
    assert len(seen) > main_calls + 1 and {s[0] for s in seen} == {"record"}
    assert {s[2] for s in seen} == {corpus + 300}
    done = [s[1] for s in seen]
    assert done == sorted(done) and done[-1] == corpus + 300


def test_record_stage_completes_when_the_corpus_is_shorter_than_p50c_tokens(small, tmp_path, monkeypatch):
    corpus = sum(len(d) for _, d in iter_docs(small))
    cfg = replace(small, p50c_tokens=corpus + 1000)
    fake_p50c(monkeypatch, cfg)
    status = Status(tmp_path / "s.json")
    record_p50c(cfg, status)
    assert status.stages["record"]["done"] == status.stages["record"]["total"] == 2 * corpus + 1000


def test_resume_refuses_a_different_plan(small, tmp_path):
    out = tmp_path / "rec"
    with pytest.raises(Crash):
        record_pass(small, fake_model(small), FAKE_TOK, CrashingStatus(tmp_path / "a.json", 3), out, LAYERS)
    with pytest.raises(FingerprintMismatch):
        record_pass(small, fake_model(small), FAKE_TOK, Status(tmp_path / "b.json"), out, LAYERS, limit_tokens=100)


def test_finished_recording_returns_at_once(small, tmp_path):
    out = tmp_path / "rec"
    (out / "ckpt").mkdir(parents=True)
    (out / "summary.npz").write_bytes(b"done")
    record_pass(small, None, None, None, out, LAYERS)     # would fail on the first use of any of these
    assert (out / "summary.npz").read_bytes() == b"done" and not (out / "ckpt").exists()


def test_speed_note_appears_after_100_batches(tmp_path):
    cfg = replace(Config.tiny(tmp_path / "atlas"), n_neurons=6, top_k=4, dn_k=2, random_k=4, window=8, batch_docs=1,
                  checkpoint_s=10_000)
    build_corpus(cfg, [[5] * 32 for _ in range(101)])
    status = Status(tmp_path / "s.json")
    record_pass(cfg, fake_model(cfg), FAKE_TOK, status, tmp_path / "rec", (15,))
    assert "tokens/s" in status.note_ and " h" in status.note_


def test_keep_columns_is_idempotent(small, tmp_path):
    out = tmp_path / "rec"
    record_pass(small, fake_model(small), FAKE_TOK, Status(tmp_path / "s.json"), out, LAYERS)
    full = load_npz(out / "L15.npz")
    full_summary = load_npz(out / "summary.npz")
    keep = {15: np.array([1, 4]), 19: np.array([0, 2])}
    for _ in range(2):      # the second call stands for a restart after a crash
        _keep_columns(out, keep)
        kept = load_npz(out / "L15.npz")
        assert kept["neurons"].tolist() == [1, 4]
        assert np.array_equal(kept["pos_val"], full["pos_val"][[1, 4]])
        assert np.array_equal(kept["hist"], full["hist"][[1, 4]])
        assert np.array_equal(kept["rnd_win"], full["rnd_win"][[1, 4]], equal_nan=True)
        assert np.array_equal(kept["rnd_doc"], full["rnd_doc"])     # per-example, not per-neuron
        summary = load_npz(out / "summary.npz")
        assert np.array_equal(summary["L19.fire_share"], full_summary["L19.fire_share"][[0, 2]])
        assert summary["L19.neurons"].tolist() == [0, 2]


@pytest.fixture
def tiny_corpus(tmp_path):
    cfg = Config.tiny(tmp_path / "atlas")
    rng = np.random.default_rng(2)
    build_corpus(cfg, [rng.integers(1000, 50_000, int(n)) for n in rng.integers(64, 256, 200)])
    return cfg.root


def run_record_subprocess(root, name, sigkill_after_checkpoints=None):
    out = root.parent / name
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(root), str(out)])
    steps = set()
    while sigkill_after_checkpoints and child.poll() is None and len(steps) < sigkill_after_checkpoints:
        steps |= {p.name for p in (out / "ckpt").glob("ckpt_*.json")}
        time.sleep(0.02)
    if sigkill_after_checkpoints:
        child.kill()
    return SimpleNamespace(returncode=child.wait(timeout=900), out=out)


@pytest.mark.gpu
def test_killed_recording_resumes_to_same_result(tiny_corpus):
    clean = run_record_subprocess(tiny_corpus, "clean")
    killed = run_record_subprocess(tiny_corpus, "killed", sigkill_after_checkpoints=2)
    assert killed.returncode == -9 and not (killed.out / "summary.npz").exists()
    resumed = run_record_subprocess(tiny_corpus, "killed")
    assert clean.returncode == 0 and resumed.returncode == 0
    assert json.loads((resumed.out / "record_meta.json").read_text())["resumed_from"] > 0
    assert_recordings_equal(clean.out, resumed.out, rtol=1e-3)


@pytest.mark.gpu
def test_record_p50c_keeps_only_the_edited_neurons(tiny_corpus):
    cfg = replace(Config.tiny(tiny_corpus), p50c_tokens=P50C_TEST_TOKENS)
    # its own process, because the session's `base_model` fixture would trip the guard against two loaded models
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "p50c", str(tiny_corpus)], check=True, timeout=900)
    out = cfg.root / "record_p50c"
    edited = edited_neurons("p50c")
    for layer in LAYERS:
        state = load_npz(out / f"L{layer}.npz")
        assert np.array_equal(state["neurons"], edited[layer])
        assert state["pos_val"].shape == (64, cfg.top_k) and state["count"].max() > 0
    assert 3000 <= json.loads((out / "record_meta.json").read_text())["tokens"] < 3000 + cfg.doc_cap
    assert load_npz(out / "summary.npz")["L19.live"].shape == (64,)


@pytest.mark.gpu
def test_worst_case_batch_fits_on_the_gpu(tmp_path, base_model):
    model, tok = base_model
    cfg = Config.tiny(tmp_path / "atlas")
    rng = np.random.default_rng(3)
    n_docs = cfg.batch_tokens // cfg.doc_cap
    build_corpus(cfg, [rng.integers(1000, 50_000, cfg.doc_cap) for _ in range(n_docs)])
    assert [len(g) for g in plan_batches(cfg, iter_docs(cfg), None)] == [n_docs]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()

    def fail_on_retry(seconds):
        pytest.fail("the worst-case batch ran out of GPU memory, so a retry would have hidden it")

    record_pass(cfg, model, tok, Status(tmp_path / "s.json"), tmp_path / "rec", cfg.layers, wait=fail_on_retry)
    seconds = time.monotonic() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"\nworst case: {n_docs} x {cfg.doc_cap} tokens, {len(cfg.layers)} layers, peak {peak:.2f} GiB, "
          f"{n_docs * cfg.doc_cap / seconds:.0f} tokens/s over {seconds:.1f} s including file writes")
    assert load_npz(tmp_path / "rec" / "L20.npz")["count"].max() > 0


if __name__ == "__main__":
    if sys.argv[1] == "p50c":
        root = Path(sys.argv[2])
        record_p50c(replace(Config.tiny(root), p50c_tokens=P50C_TEST_TOKENS), Status(root.parent / "p50c_status.json"))
        assert torch.cuda.memory_allocated() < 0.5 * 2**30, "record_p50c left its model on the GPU"
    else:
        from probe.model import load_model

        root, child_out = Path(sys.argv[1]), Path(sys.argv[2])
        child_cfg = replace(Config.tiny(root), checkpoint_s=0)
        child_model, child_tok = load_model()
        record_pass(child_cfg, child_model, child_tok, Status(child_out.parent / f"{child_out.name}.status.json"),
                    child_out, LAYERS)
