"""Streaming per-neuron statistics for one layer: strongest examples, random examples, histograms and running totals.

Lists, per neuron, best first:
- `pos`: gate > 0 and a > 0, ranked by a. `neg`: gate > 0 and a < 0, ranked by -a, so `neg_val` holds the magnitude.
- `dn` (double negative): gate < 0 and value < 0, ranked by a.
A position where `a` or `value` is exactly 0 enters no list. Empty slots hold val -inf, doc -1, tok -1 and a NaN window.
Windows hold the signed activation only at positions valid for their list, and NaN everywhere else (R19): eligible gate-positive
positions for `pos`, `neg` and random examples, eligible gate-negative positions for `dn`. Random examples also keep each neuron's
sign regime, (gate >= 0) * 2 + (value >= 0).

`update` has no attention mask, so a document is taken to end at its last eligible token: later positions read as
padding (NaN in windows). Eligible positions with a non-finite activation are skipped entirely.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

HIST_EDGES = torch.logspace(-4, 3, 255)

HIST_CHUNK = 8192  # token positions per histogram step, which bounds its temporary memory
FIRE_FRACTION = 0.3
MIN_LIVE_POSITIONS = 100
LISTS = ("pos", "neg", "dn")


def _rank_within_doc(docs: Tensor) -> Tensor:
    """For each entry of [N, M] `docs`, how many entries to its left in the row have the same doc."""
    perm = docs.argsort(dim=1, stable=True)
    ordered = docs.gather(1, perm)
    new_run = torch.ones_like(ordered, dtype=torch.bool)
    new_run[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    index = torch.arange(docs.shape[1], device=docs.device).expand_as(docs)
    run_start = (index * new_run).cummax(dim=1).values
    return torch.zeros_like(docs).scatter(1, perm, index - run_start)


class LayerRecorder:
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        self.edges = HIST_EDGES.to(device)
        half = cfg.window // 2
        self.offsets = torch.arange(-half, cfg.window - half, device=device)
        n, w = cfg.n_neurons, cfg.window

        def full(shape, value, dtype):
            return torch.full(shape, value, dtype=dtype, device=device)

        self.t: dict[str, Tensor] = {}
        for name, k in (("pos", cfg.top_k), ("neg", cfg.top_k), ("dn", cfg.dn_k)):
            self.t[f"{name}_val"] = full((n, k), -torch.inf, torch.float32)
            self.t[f"{name}_doc"] = full((n, k), -1, torch.int64)
            self.t[f"{name}_tok"] = full((n, k), -1, torch.int64)
            self.t[f"{name}_win"] = full((n, k, w), torch.nan, torch.float16)
        self.t["rnd_key"] = full((cfg.random_k,), torch.inf, torch.float32)
        self.t["rnd_doc"] = full((cfg.random_k,), -1, torch.int64)
        self.t["rnd_tok"] = full((cfg.random_k,), -1, torch.int64)
        self.t["rnd_win"] = full((n, cfg.random_k, w), torch.nan, torch.float16)
        self.t["rnd_regime"] = full((n, cfg.random_k), -1, torch.int8)
        self.t["hist"] = full((n, 4, 256), 0, torch.int64)
        self.t["count"] = full((n,), 0, torch.int64)
        self.t["sum"] = full((n,), 0.0, torch.float64)
        self.t["max"] = full((n,), -torch.inf, torch.float32)
        self.t["min"] = full((n,), torch.inf, torch.float32)

    def state(self) -> dict[str, np.ndarray]:
        return {k: v.cpu().numpy().copy() for k, v in self.t.items()}

    def load(self, state) -> None:
        loaded = {}
        for k, mine in self.t.items():
            arr = np.array(state[k])  # always a copy, so the recorder never aliases the caller's array
            if arr.shape != tuple(mine.shape) or torch.from_numpy(arr).dtype != mine.dtype:
                raise ValueError(f"{k}: state has {arr.shape} {arr.dtype}, recorder expects {tuple(mine.shape)} {mine.dtype}")
            loaded[k] = torch.from_numpy(arr).to(self.device)
        self.t = loaded

    @torch.no_grad()
    def update(self, a: Tensor, gate: Tensor, value: Tensor, eligible: Tensor, doc_ids: Tensor, doc_keys: Tensor) -> None:
        """Add one batch. `a`, `gate`, `value`: [B, T, n_neurons]; `eligible`: bool [B, T]; `doc_ids`: [B];
        `doc_keys`: [B, 2], each document's random priority and a uniform number that picks its random position."""
        ok = eligible & a.sum(-1).isfinite()  # the sum is non-finite when any neuron is, without a [B, T, N] temporary
        if not ok.any():
            return
        doc_keys = doc_keys.to(self.device)
        self._update_stats(a, gate, value, ok)
        found = {name: ([], [], [], []) for name in LISTS}
        for b, doc in enumerate(doc_ids.tolist()):
            where = eligible[b].nonzero()
            if not len(where):
                continue
            length = int(where[-1]) + 1
            doc_a, doc_gate, doc_value, doc_ok = a[b, :length], gate[b, :length], value[b, :length], ok[b, :length, None]
            masks = {"pos": (doc_gate > 0) & (doc_a > 0), "neg": (doc_gate > 0) & (doc_a < 0),
                     "dn": (doc_gate < 0) & (doc_value < 0)}
            active = (doc_a != 0) & (doc_value != 0) & doc_ok
            gate_positive = doc_a.where((doc_gate > 0) & doc_ok, torch.nan)
            gate_negative = doc_a.where((doc_gate < 0) & doc_ok, torch.nan)
            for name, (vals, docs, toks, wins) in found.items():
                score = torch.where(masks[name] & active, -doc_a if name == "neg" else doc_a, -torch.inf)
                val, tok = score.topk(min(self.cfg.per_doc_cap, length), dim=0)
                val, tok = val.T, tok.T
                vals.append(val)
                docs.append(torch.full_like(tok, doc))
                toks.append(tok)
                wins.append(self._windows(gate_negative if name == "dn" else gate_positive, tok))
            self._add_random(gate_positive, doc_gate, doc_value, doc_ok[:, 0], doc, doc_keys[b])
        for name, parts in found.items():
            if parts[0]:
                self._merge(name, *(torch.cat(p, dim=1) for p in parts))

    def _windows(self, shown: Tensor, tok: Tensor) -> Tensor:
        """Float16 [N, k, window] of `shown` [T, N] around the positions `tok` [N, k] of each neuron."""
        at = tok.unsqueeze(-1) + self.offsets
        inside = (at >= 0) & (at < shown.shape[0])
        got = shown.T.gather(1, at.clamp(0, shown.shape[0] - 1).flatten(1)).view(at.shape)
        return got.where(inside, torch.nan).to(torch.float16)

    def _update_stats(self, a: Tensor, gate: Tensor, value: Tensor, ok: Tensor) -> None:
        neuron_base = torch.arange(self.cfg.n_neurons, device=self.device, dtype=torch.int32) * 1024
        for chunk in ok.nonzero().split(HIST_CHUNK):
            b, t = chunk.unbind(1)
            x = a[b, t]
            flat = torch.bucketize(x.abs(), self.edges, out_int32=True)
            flat += neuron_base  # flattened [neuron, regime, bin]; regime = (gate >= 0) * 2 + (value >= 0)
            flat += (gate[b, t] >= 0).int() * 512
            flat += (value[b, t] >= 0).int() * 256
            self.t["hist"].view(-1).add_(torch.bincount(flat.flatten(), minlength=self.t["hist"].numel()))
            self.t["count"] += x.shape[0]
            self.t["sum"] += x.sum(0, dtype=torch.float64)
            self.t["max"] = torch.maximum(self.t["max"], x.max(0).values)
            self.t["min"] = torch.minimum(self.t["min"], x.min(0).values)

    def _add_random(self, shown: Tensor, doc_gate: Tensor, doc_value: Tensor, doc_ok: Tensor, doc: int, keys: Tensor) -> None:
        where = doc_ok.nonzero()[:, 0]
        key = keys[0]
        if not len(where) or key >= self.t["rnd_key"][-1] or doc in self.t["rnd_doc"].tolist():
            return
        tok = where[min(int(float(keys[1]) * len(where)), len(where) - 1)]
        window = self._windows(shown, tok.expand(self.cfg.n_neurons, 1))
        regime = (doc_gate[tok] >= 0).to(torch.int8) * 2 + (doc_value[tok] >= 0).to(torch.int8)
        all_keys = torch.cat([self.t["rnd_key"], key[None]])
        order = all_keys.argsort(stable=True)[:self.cfg.random_k]
        self.t["rnd_key"] = all_keys[order]
        self.t["rnd_doc"] = torch.cat([self.t["rnd_doc"], torch.tensor([doc], device=self.device)])[order]
        self.t["rnd_tok"] = torch.cat([self.t["rnd_tok"], tok[None]])[order]
        self.t["rnd_win"] = torch.cat([self.t["rnd_win"], window], dim=1)[:, order]
        self.t["rnd_regime"] = torch.cat([self.t["rnd_regime"], regime[:, None]], dim=1)[:, order]

    def _merge(self, name: str, val: Tensor, doc: Tensor, tok: Tensor, win: Tensor) -> None:
        """Merge [N, M] candidates into a list. Kept entries come first; equal values keep their input order."""
        t = self.t
        val = torch.cat([t[f"{name}_val"], val], dim=1)
        doc = torch.cat([t[f"{name}_doc"], doc], dim=1)
        tok = torch.cat([t[f"{name}_tok"], tok], dim=1)
        win = torch.cat([t[f"{name}_win"], win], dim=1)
        by_value = val.argsort(dim=1, descending=True, stable=True)
        keep = (val.gather(1, by_value) > -torch.inf) & (_rank_within_doc(doc.gather(1, by_value)) < self.cfg.per_doc_cap)
        first = (~keep).int().argsort(dim=1, stable=True)[:, :t[f"{name}_val"].shape[1]]
        filled = keep.gather(1, first)
        chosen = by_value.gather(1, first)
        t[f"{name}_val"] = val.gather(1, chosen).where(filled, -torch.inf)
        t[f"{name}_doc"] = doc.gather(1, chosen).where(filled, -1)
        t[f"{name}_tok"] = tok.gather(1, chosen).where(filled, -1)
        wide = chosen[..., None].expand(-1, -1, win.shape[-1])
        t[f"{name}_win"] = win.gather(1, wide).where(filled[..., None], torch.nan)


def _positions_at_or_above(gate_positive: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """Per neuron, the positions in bins whose lower edge is >= its threshold."""
    lower = np.concatenate([[0.0], HIST_EDGES.numpy().astype(np.float64)])
    return (gate_positive * (lower[None] >= threshold[:, None])).sum(1)


def _finalize_layer(s) -> dict[str, np.ndarray]:
    pos_extreme = np.maximum(s["pos_val"].max(1).astype(np.float64), 0.0)
    neg_extreme = np.maximum(s["neg_val"].max(1).astype(np.float64), 0.0)
    top = np.maximum(pos_extreme, neg_extreme)
    gate_positive = s["hist"][:, 2:].sum(1)

    median = float(np.median(top))
    live = np.zeros(len(top), dtype=bool)
    if median > 0:
        live = _positions_at_or_above(gate_positive, np.full(len(top), FIRE_FRACTION * median)) >= MIN_LIVE_POSITIONS

    # Firing is relative to the dominant pole's own extreme, so only that pole's regime counts: 3 positive, 2 negative.
    dominant = np.where(pos_extreme >= neg_extreme, 1, -1).astype(np.int8)
    own_pole = s["hist"][np.arange(len(top)), np.where(dominant == 1, 3, 2)]
    fires = _positions_at_or_above(own_pole, FIRE_FRACTION * top)
    has_share = (top > 0) & (s["count"] > 0)
    fire_share = np.divide(fires, s["count"], out=np.zeros(len(top)), where=has_share)

    # An unfilled pos slot stays at -inf, so a list with room counts as reachable by any positive double negative.
    dn_top = s["dn_val"].max(1)
    return {"pos_extreme": pos_extreme, "neg_extreme": neg_extreme,
            "dominant": dominant, "live": live,
            "fire_share": fire_share, "dn_flag": (dn_top > 0) & (dn_top > s["pos_val"].min(1))}


def finalize(states: dict[int, dict], cfg) -> dict:
    """Per-layer neuron statistics from finished recorder states."""
    return {layer: _finalize_layer(s) for layer, s in states.items()}
