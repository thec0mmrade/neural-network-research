"""The recording pass: stream the corpus through the model and keep each neuron's statistics, surviving a kill at any moment.

The batch plan is a pure function of the corpus, so a checkpoint's batch count is an exact corpus position. Checkpoints hold
that position and all recorder states together, and `summary.npz` is written last, so its presence means the pass finished.
"""
from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import torch

from probe.atlas import activations, store
from probe.atlas.config import Config
from probe.atlas.corpus import iter_docs
from probe.atlas.recorder import LayerRecorder, finalize
from probe.weights import Checkpoint, edited, layer_tensor

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # read when CUDA first allocates

SORT_WINDOW = 256
NOTE_AFTER_BATCHES = 100
OOM_RETRIES = 3
OOM_WAIT_S = 60
MAX_GIB_BEFORE_P50C = 0.5   # more than this allocated means another model is still on the GPU
P50C_LAYERS = (15, 19)
PER_EXAMPLE = {"rnd_key", "rnd_doc", "rnd_tok"}   # recorder arrays indexed by example; all others are indexed by neuron


def _group_window(cfg: Config, window: list[tuple[int, int]]) -> list[list[int]]:
    """Split (length, doc_id) pairs, shortest first, into batches within `batch_docs` and `batch_tokens`."""
    batches, group = [], []
    for length, doc_id in sorted(window):
        if length > cfg.batch_tokens:
            raise ValueError(f"document {doc_id} has {length} tokens, more than batch_tokens={cfg.batch_tokens}")
        if group and (len(group) == cfg.batch_docs or length * (len(group) + 1) > cfg.batch_tokens):
            batches.append(group)
            group = []
        group.append(doc_id)
    return batches + [group] if group else batches


def plan_batches(cfg: Config, docs: Iterable[tuple[int, np.ndarray]], limit_tokens: int | None) -> list[list[int]]:
    """Doc ids per batch, sorted by length within windows of 256 documents. Reading stops once `limit_tokens` is reached."""
    plan, window, seen = [], [], 0
    for doc_id, ids in docs:
        window.append((len(ids), doc_id))
        seen += len(ids)
        if len(window) == SORT_WINDOW:
            plan += _group_window(cfg, window)
            window = []
        if limit_tokens is not None and seen >= limit_tokens:
            break
    return plan + _group_window(cfg, window)


def _note_lengths(docs: Iterable[tuple[int, np.ndarray]], lengths: dict[int, int]) -> Iterator[tuple[int, np.ndarray]]:
    for doc_id, ids in docs:
        lengths[doc_id] = len(ids)
        yield doc_id, ids


def _planned_docs(cfg: Config, plan: list[list[int]], start: int) -> Iterator[tuple[int, list[int], list[np.ndarray]]]:
    """Yield (batch index, doc ids, token arrays) from `start` on, holding only the documents of the current window."""
    docs = iter_docs(cfg)
    held: dict[int, np.ndarray] = {}
    for i, group in enumerate(plan):
        for doc_id in group:
            while doc_id not in held:
                found, ids = next(docs)
                held[found] = ids
        arrays = [held.pop(doc_id) for doc_id in group]
        if i >= start:
            yield i, group, arrays


def _plan_hash(cfg: Config, layers: Sequence[int], plan: list[list[int]]) -> str:
    manifest = store.read_json(cfg.root / "corpus" / "manifest.json")
    return hashlib.sha256(json.dumps([manifest["hash"], list(layers), plan]).encode()).hexdigest()


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def record_pass(cfg: Config, model, tok, status, out: Path, layers: Sequence[int], limit_tokens: int | None = None,
                wait: Callable[[float], None] = time.sleep, progress: Callable[[int, int], None] | None = None) -> None:
    """Record `layers` over the corpus into `out`, resuming from the newest intact checkpoint in `out/ckpt`.

    A batch that runs out of GPU memory is retried after `wait(OOM_WAIT_S)`, up to OOM_RETRIES times. `progress(done, total)`
    reports tokens; by default the `record` stage, whose total also covers the p50c pass that follows this one.
    """
    out, ckpt = Path(out), Path(out) / "ckpt"
    if (out / "summary.npz").exists():
        shutil.rmtree(ckpt, ignore_errors=True)   # a crash may have come between the summary and this cleanup
        return

    if progress is None:
        def progress(done, total):
            status.stage("record", done, total + cfg.p50c_tokens, "tokens")

    lengths: dict[int, int] = {}
    plan = plan_batches(cfg, _note_lengths(iter_docs(cfg), lengths), limit_tokens)
    tokens_total = sum(lengths[d] for group in plan for d in group)
    fingerprint = _plan_hash(cfg, layers, plan)

    device = next(model.parameters()).device
    recorders = {l: LayerRecorder(cfg, device) for l in layers}
    excluded = activations.excluded_ids(tok)
    im_start_id = tok.convert_tokens_to_ids("<|im_start|>")

    start, tokens_done, nonfinite = 0, 0, {str(l): 0 for l in layers}
    latest = store.load_latest_checkpoint(ckpt)
    if latest is not None:
        start, arrays, meta = latest
        if meta["plan"] != fingerprint:
            raise store.FingerprintMismatch(f"checkpoint in {ckpt} belongs to a different corpus, plan or layer set")
        for l, rec in recorders.items():
            rec.load({k: arrays[f"L{l}.{k}"] for k in rec.t})
        tokens_done, nonfinite = meta["tokens_done"], meta["nonfinite"]
    progress(tokens_done, tokens_total)

    def record_batch(group: list[int], docs: list[np.ndarray], finished: set[int]) -> None:
        """Forward one batch. Layers in `finished` were recorded by an earlier attempt, so each layer counts it exactly once."""
        ids, attention = activations.pad_right(docs, tok.pad_token_id, device)
        eligible = activations.eligible_mask(ids, attention, excluded, im_start_id)
        doc_ids = torch.tensor(group, dtype=torch.int64, device=device)
        keys = np.stack([np.random.default_rng([cfg.seed, d]).random(2) for d in group])
        doc_keys = torch.tensor(keys, dtype=torch.float32, device=device)

        def on_layer(layer, a, gate, value):   # keeps no reference: one layer's inputs can take 2.6 GiB
            if layer in finished:
                return
            bad = int((eligible & ~a.sum(-1).isfinite()).sum())
            rec = recorders[layer]
            backup = {k: v.clone() for k, v in rec.t.items()}   # `update` can fail halfway through its changes
            try:
                rec.update(a, gate, value, eligible, doc_ids, doc_keys)
            except torch.cuda.OutOfMemoryError:
                rec.t = backup
                raise
            nonfinite[str(layer)] += bad
            finished.add(layer)

        with activations.capture(model, layers, on_layer):
            activations.run_batch(model, ids, attention)

    began = last_save = time.monotonic()
    tokens_at_start = tokens_done
    for i, group, docs in _planned_docs(cfg, plan, start):
        finished: set[int] = set()
        for attempt in range(OOM_RETRIES + 1):
            try:
                record_batch(group, docs, finished)
                if attempt:
                    status.note(f"Batch {i + 1} of {len(plan)} recovered after {attempt} GPU memory retries")
                break
            except torch.cuda.OutOfMemoryError:
                pass   # leaving the handler drops the traceback, and with it the batch's tensors
            torch.cuda.empty_cache()
            if attempt == OOM_RETRIES:
                raise RuntimeError(f"batch {i + 1} of {len(plan)} still ran out of GPU memory after {OOM_RETRIES} retries; "
                                   f"the last checkpoint in {ckpt} is intact. Free GPU memory (close games and other GPU apps) "
                                   "and rerun to resume from it. Do NOT change batch_tokens: that invalidates the recording checkpoint")
            status.note(f"Batch {i + 1} of {len(plan)} ran out of GPU memory; retry {attempt + 1} of {OOM_RETRIES} in {OOM_WAIT_S} s")
            wait(OOM_WAIT_S)

        tokens_done += sum(len(d) for d in docs)
        progress(tokens_done, tokens_total)
        if i + 1 - start == NOTE_AFTER_BATCHES:
            rate = (tokens_done - tokens_at_start) / max(time.monotonic() - began, 1e-9)
            status.note(f"Recording at {rate:,.0f} tokens/s, projected {tokens_total / rate / 3600:.1f} h in total")
        if i + 1 < len(plan) and time.monotonic() - last_save >= cfg.checkpoint_s:
            arrays = {f"L{l}.{k}": v for l, rec in recorders.items() for k, v in rec.state().items()}
            store.save_checkpoint(ckpt, i + 1, arrays, {"plan": fingerprint, "tokens_done": tokens_done, "nonfinite": nonfinite})
            last_save = time.monotonic()

    states = {l: rec.state() for l, rec in recorders.items()}
    for l, state in states.items():
        store.atomic_write_bytes(out / f"L{l}.npz", _npz_bytes(state))
    store.atomic_write_json(out / "record_meta.json", {
        "layers": list(layers), "batches": len(plan), "tokens": tokens_total, "resumed_from": start,
        "nonfinite": nonfinite, "nonfinite_total": sum(nonfinite.values())})
    summary = {f"L{l}.{k}": v for l, arrays in finalize(states, cfg).items() for k, v in arrays.items()}
    store.atomic_write_bytes(out / "summary.npz", _npz_bytes(summary))
    shutil.rmtree(ckpt, ignore_errors=True)


def edited_neurons(run: str = "p50c") -> dict[int, np.ndarray]:
    """Sorted `down_proj` columns (neuron indices) that differ from the base model, for L15 and L19."""
    base = Checkpoint()
    found = {}
    for l in P50C_LAYERS:
        name = layer_tensor(l, "down")
        found[l] = np.flatnonzero((base.get(name) != edited(run, name)).any(axis=0))
    return found


def _keep_columns(out: Path, keep: dict[int, np.ndarray]) -> None:
    """Cut each layer's recording and the summary down to the `keep` neurons. Files already cut are left alone."""
    for l, idx in keep.items():
        state = _load_npz(out / f"L{l}.npz")
        if "neurons" not in state:
            cut = {k: v if k in PER_EXAMPLE else v[idx] for k, v in state.items()}
            store.atomic_write_bytes(out / f"L{l}.npz", _npz_bytes({**cut, "neurons": idx}))
    summary = _load_npz(out / "summary.npz")
    if not any(k.endswith(".neurons") for k in summary):
        cut = {k: v[keep[int(k[1:].split(".")[0])]] for k, v in summary.items()}
        cut.update({f"L{l}.neurons": idx for l, idx in keep.items()})
        store.atomic_write_bytes(out / "summary.npz", _npz_bytes(cut))


def record_p50c(cfg: Config, status) -> None:
    """Record the p50c-edited model over the first `p50c_tokens`, keeping only the edited neurons of L15 and L19.

    Progress continues the `record` stage from where the corpus pass ended, so the bar never moves back.

    Contract: the caller must free the base model first. Two models do not fit beside a recording batch, so this raises
    RuntimeError if more than MAX_GIB_BEFORE_P50C GiB is allocated when the p50c model is about to load. The p50c model is
    freed again before this returns.
    """
    out = cfg.root / "record_p50c"
    manifest = store.read_json(cfg.root / "corpus" / "manifest.json")
    corpus_tokens = sum(shard["n_tokens"] for shard in manifest["shards"])
    total = corpus_tokens + cfg.p50c_tokens

    def progress(done: int, _planned: int) -> None:
        status.stage("record", corpus_tokens + min(done, cfg.p50c_tokens), total, "tokens")

    if not (out / "summary.npz").exists():
        from probe.model import load_model   # lazy: loading transformers is slow and needs the GPU shell

        allocated = torch.cuda.memory_allocated() / 2**30
        if allocated > MAX_GIB_BEFORE_P50C:
            raise RuntimeError(f"record_p50c needs the GPU to itself, but {allocated:.2f} GiB is already allocated "
                               f"(limit {MAX_GIB_BEFORE_P50C} GiB); free the base model first")
        model, tok = load_model("p50c")
        try:
            record_pass(cfg, model, tok, status, out, P50C_LAYERS, limit_tokens=cfg.p50c_tokens, progress=progress)
        finally:
            del model
            gc.collect()   # the model's modules can hold reference cycles
            torch.cuda.empty_cache()
    shutil.rmtree(out / "ckpt", ignore_errors=True)   # a kill during record_pass's own cleanup can leave them behind
    _keep_columns(out, edited_neurons("p50c"))
    status.stage("record", total, total, "tokens")
