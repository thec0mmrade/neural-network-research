"""Recorder tests. `brute_force` is an independent, plain-Python/numpy reimplementation of the recording rules."""
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from probe.atlas.recorder import HIST_EDGES, LayerRecorder, finalize

N = 8
LOWER_EDGES = np.concatenate([[0.0], HIST_EDGES.numpy().astype(np.float64)])


def keys_for(docs):
    return torch.tensor(np.array([np.random.default_rng(int(d)).random(2) for d in docs]), dtype=torch.float32)


def random_batches(seed, n_batches, n_docs=7):
    """Right-padded batches with garbage at padding and ineligible first and interior tokens."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    out = ([], [], [], [], [])
    next_doc = 100
    for _ in range(n_batches):
        lengths = rng.integers(5, 120, n_docs)
        T = int(lengths.max())
        g = torch.randn(n_docs, T, N, generator=gen)
        v = torch.randn(n_docs, T, N, generator=gen) * 4
        a = F.silu(g) * v
        elig = torch.rand(n_docs, T, generator=gen) > 0.1
        for b, length in enumerate(lengths):
            elig[b, 0] = False
            elig[b, length - 1] = True  # the document's extent is visible to the recorder
            elig[b, length:] = False
            a[b, length:] = 1e4  # padding garbage must never be recorded
            g[b, length:] = 1.0
            v[b, length:] = 1e4
        docs = torch.arange(next_doc, next_doc + n_docs, dtype=torch.int64)
        next_doc += n_docs
        for lst, x in zip(out, (a, g, v, elig, docs)):
            lst.append(x)
    return out


def doc_rows(a, g, v, elig, docs):
    for ab, gb, vb, eb, db in zip(a, g, v, elig, docs):
        for row in range(len(db)):
            e = eb[row].numpy()
            length = int(np.flatnonzero(e)[-1]) + 1 if e.any() else 0
            yield int(db[row]), ab[row].numpy(), gb[row].numpy(), vb[row].numpy(), e, length


def brute_window(col, valid, t, length):
    """The 64 activations around `t`; NaN outside the document and wherever `valid` is False."""
    return [float(col[p]) if 0 <= p < length and valid[p] else np.nan for p in range(t - 32, t + 32)]


def brute_list(rows, neuron, select, k, per_doc):
    """Top `k` of (score, doc, tok, window), taking each document's best `per_doc` first."""
    pooled = []
    for doc, a, g, v, e, length in rows:
        here = []
        for t in range(length):
            if e[t] and select(a[t, neuron], g[t, neuron], v[t, neuron]):
                score = a[t, neuron] if select.__name__ != "neg" else -a[t, neuron]
                here.append((float(score), doc, t))
        here.sort(key=lambda c: -c[0])
        valid = e & (g[:, neuron] < 0 if select.__name__ == "dn" else g[:, neuron] > 0)
        pooled += [(s, d, t, brute_window(a[:, neuron], valid, t, length)) for s, d, t in here[:per_doc]]
    pooled.sort(key=lambda c: -c[0])
    return pooled[:k]


def pos(a, g, v): return g > 0 and a > 0 and v != 0
def neg(a, g, v): return g > 0 and a < 0 and v != 0
def dn(a, g, v): return g < 0 and v < 0 and a != 0


def brute_force_top(a, g, v, elig, docs, k=50, per_doc=3, dn_k=20):
    rows = list(doc_rows(a, g, v, elig, docs))
    out = {}
    for name, select, size in (("pos", pos, k), ("neg", neg, k), ("dn", dn, dn_k)):
        val = np.full((N, size), -np.inf, np.float32)
        doc = np.full((N, size), -1, np.int64)
        tok = np.full((N, size), -1, np.int64)
        win = np.full((N, size, 64), np.nan, np.float16)
        for n in range(N):
            for i, (s, d, t, w) in enumerate(brute_list(rows, n, select, size, per_doc)):
                val[n, i], doc[n, i], tok[n, i] = s, d, t
                win[n, i] = np.array(w, np.float32).astype(np.float16)
        out |= {f"{name}_val": val, f"{name}_doc": doc, f"{name}_tok": tok, f"{name}_win": win}
    return out


def brute_force_random(a, g, v, elig, docs, random_k=50):
    picks = []
    for ab, gb, vb, eb, db in zip(a, g, v, elig, docs):
        for key, u, row in zip(keys_for(db).numpy()[:, 0], keys_for(db).numpy()[:, 1], range(len(db))):
            where = np.flatnonzero(eb[row].numpy())
            if len(where):
                t = int(where[min(int(float(u) * len(where)), len(where) - 1)])
                regime = (gb[row, t].numpy() >= 0).astype(np.int8) * 2 + (vb[row, t].numpy() >= 0).astype(np.int8)
                valid = eb[row].numpy()[:, None] & (gb[row].numpy() > 0)
                picks.append((float(key), int(db[row]), t, ab[row].numpy(), int(where[-1]) + 1, regime, valid))
    picks.sort(key=lambda p: p[0])
    return picks[:random_k]


def brute_force_stats(a, g, v, elig):
    hist = np.zeros((N, 4, 256), np.int64)
    edges = HIST_EDGES.numpy()
    count, total, hi, lo = 0, np.zeros(N), np.full(N, -np.inf), np.full(N, np.inf)
    for ab, gb, vb, eb in zip(a, g, v, elig):
        for x, y, z in zip(ab[eb].numpy(), gb[eb].numpy(), vb[eb].numpy()):
            count += 1
            total += x.astype(np.float64)
            hi, lo = np.maximum(hi, x), np.minimum(lo, x)
            for n in range(N):
                hist[n, int(y[n] >= 0) * 2 + int(z[n] >= 0), np.searchsorted(edges, abs(x[n]), side="left")] += 1
    return hist, count, total, hi, lo


def recorded(cfg, batches, device="cpu"):
    rec = LayerRecorder(replace(cfg, n_neurons=N), device)
    for b in zip(*batches):
        rec.update(*b, doc_keys=keys_for(b[-1]))
    return rec


def update_once(cfg, a, g, v, eligible=None):
    rec = LayerRecorder(replace(cfg, n_neurons=a.shape[-1]), "cpu")
    docs = torch.arange(a.shape[0], dtype=torch.int64)
    eligible = torch.ones(a.shape[:2], dtype=torch.bool) if eligible is None else eligible
    rec.update(a, g, v, eligible, docs, doc_keys=keys_for(docs))
    return rec.state()


def recorded_state(cfg, one_hot_doc=False, peak_at=None, length=None):
    if one_hot_doc:  # one document, strongly active at every token, seen in two updates
        a = torch.zeros(1, 30, N); a[0, :, 0] = torch.linspace(1, 9, 30)
        g, v = torch.ones_like(a), torch.ones_like(a)
        rec = LayerRecorder(replace(cfg, n_neurons=N), "cpu")
        docs = torch.tensor([5], dtype=torch.int64)
        for _ in range(2):
            rec.update(a, g, v, torch.ones(1, 30, dtype=torch.bool), docs, doc_keys=keys_for(docs))
        return rec.state()
    a = torch.zeros(1, length, N); a[0, peak_at, 0] = 7.5
    return update_once(cfg, a, torch.ones_like(a), torch.ones_like(a))


def run_with_and_without_reload(cfg):
    batches = random_batches(seed=3, n_batches=6)
    full = recorded(cfg, batches).state()
    first = recorded(cfg, [x[:3] for x in batches]).state()
    resumed = LayerRecorder(replace(cfg, n_neurons=N), "cpu")
    resumed.load(first)
    for b in zip(*[x[3:] for x in batches]):
        resumed.update(*b, doc_keys=keys_for(b[-1]))
    return full, resumed.state()


def silent_state(cfg):
    a = torch.zeros(2, 40, N)
    return update_once(cfg, a, torch.ones_like(a), torch.ones_like(a))


# ---------------------------------------------------------------- the brief's six tests

def test_streaming_matches_brute_force(tiny_cfg):
    a, g, v, elig, docs = random_batches(seed=1, n_batches=6)
    rec = LayerRecorder(replace(tiny_cfg, n_neurons=8), "cpu")
    for b in zip(a, g, v, elig, docs): rec.update(*b, doc_keys=keys_for(b[-1]))
    s = rec.state(); want = brute_force_top(a, g, v, elig, docs, k=50, per_doc=3)
    np.testing.assert_allclose(np.sort(s["pos_val"], 1), np.sort(want["pos_val"], 1), rtol=1e-5)


def test_at_most_three_examples_per_document(tiny_cfg):
    s = recorded_state(tiny_cfg, one_hot_doc=True)
    assert max(np.bincount(s["pos_doc"][0][s["pos_val"][0] > 0])) <= 3
    assert (s["pos_val"][0] > 0).sum() == 3


def test_double_negative_never_enters_positive_pole(tiny_cfg):
    a = torch.tensor([[[9.0]]]); g = torch.tensor([[[-1.0]]]); v = torch.tensor([[[-30.0]]])
    s = update_once(tiny_cfg, a, g, v)
    assert s["pos_val"].max() <= 0 and s["dn_val"].max() == pytest.approx(9.0)


def test_gate_negative_position_never_enters_negative_pole(tiny_cfg):
    a = torch.tensor([[[-9.0]]]); g = torch.tensor([[[-1.0]]]); v = torch.tensor([[[30.0]]])
    s = update_once(tiny_cfg, a, g, v)
    assert np.isneginf(s["neg_val"]).all() and np.isneginf(s["pos_val"]).all()


def test_double_negative_needs_a_negative_value(tiny_cfg):
    a = torch.tensor([[[5.0]]]); g = torch.tensor([[[-1.0]]]); v = torch.tensor([[[2.0]]])
    assert np.isneginf(update_once(tiny_cfg, a, g, v)["dn_val"]).all()


def test_window_is_centered_and_padded(tiny_cfg):
    s = recorded_state(tiny_cfg, peak_at=2, length=10)
    w = s["pos_win"][0, 0]
    assert np.isnan(w[0]) and w[32] == pytest.approx(s["pos_val"][0, 0])
    assert np.isnan(w[:30]).all() and not np.isnan(w[30:40]).any() and np.isnan(w[40:]).all()


def test_state_roundtrip_continues_identically(tiny_cfg):
    full, resumed = run_with_and_without_reload(tiny_cfg)
    for k in full: np.testing.assert_array_equal(full[k], resumed[k])


@pytest.mark.parametrize("state", [silent_state, lambda c: LayerRecorder(replace(c, n_neurons=N), "cpu").state()],
                         ids=["zero_activations", "never_updated"])
def test_silent_neuron_is_dead_without_error(tiny_cfg, state):
    out = finalize({15: state(tiny_cfg)}, tiny_cfg)
    assert not out[15]["live"].any() and np.isfinite(out[15]["fire_share"]).all()
    assert (out[15]["fire_share"] == 0).all() and (out[15]["pos_extreme"] == 0).all() and (out[15]["neg_extreme"] == 0).all()


# ---------------------------------------------------------------- full comparison with the brute-force recorder

@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("name", ["pos", "neg", "dn"])
def test_every_list_matches_brute_force_exactly(tiny_cfg, seed, name):
    batches = random_batches(seed=seed, n_batches=6)
    s = recorded(tiny_cfg, batches).state()
    want = brute_force_top(*batches)
    np.testing.assert_allclose(s[f"{name}_val"], want[f"{name}_val"], rtol=1e-6)
    np.testing.assert_array_equal(s[f"{name}_doc"], want[f"{name}_doc"])
    np.testing.assert_array_equal(s[f"{name}_tok"], want[f"{name}_tok"])
    np.testing.assert_array_equal(s[f"{name}_win"], want[f"{name}_win"])


def test_random_examples_match_brute_force(tiny_cfg):
    batches = random_batches(seed=4, n_batches=12)
    s = recorded(tiny_cfg, batches).state()
    want = brute_force_random(*batches)
    assert len(want) == 50
    np.testing.assert_array_equal(s["rnd_doc"], [p[1] for p in want])
    np.testing.assert_array_equal(s["rnd_tok"], [p[2] for p in want])
    np.testing.assert_allclose(s["rnd_key"], [p[0] for p in want], rtol=1e-6)
    for i, (_, _, t, a, length, regime, valid) in enumerate(want):
        np.testing.assert_array_equal(s["rnd_regime"][:, i], regime)
        for n in range(N):
            w = np.array(brute_window(a[:, n], valid[:, n], t, length), np.float32).astype(np.float16)
            np.testing.assert_array_equal(s["rnd_win"][n, i], w)


def test_random_slots_stay_marked_until_filled(tiny_cfg):
    batches = random_batches(seed=4, n_batches=1, n_docs=3)
    s = recorded(tiny_cfg, batches).state()
    assert (s["rnd_doc"][:3] >= 0).all() and (s["rnd_doc"][3:] == -1).all() and (s["rnd_tok"][3:] == -1).all()
    assert np.isposinf(s["rnd_key"][3:]).all() and np.isnan(s["rnd_win"][:, 3:]).all()
    assert (s["rnd_regime"][:, :3] >= 0).all() and (s["rnd_regime"][:, 3:] == -1).all()


def test_document_without_eligible_position_gives_no_random_example(tiny_cfg):
    a = torch.ones(2, 6, N)
    elig = torch.ones(2, 6, dtype=torch.bool); elig[1] = False
    s = update_once(tiny_cfg, a, a, a, elig)
    assert s["rnd_doc"].tolist()[:2] == [0, -1]


def test_repeated_document_gives_one_random_example(tiny_cfg):
    s = recorded_state(tiny_cfg, one_hot_doc=True)
    assert s["rnd_doc"].tolist().count(5) == 1


def test_batch_without_eligible_positions_changes_nothing(tiny_cfg):
    a = torch.ones(2, 6, N)
    s = update_once(tiny_cfg, a, a, a, torch.zeros(2, 6, dtype=torch.bool))
    fresh = LayerRecorder(replace(tiny_cfg, n_neurons=N), "cpu").state()
    for k in fresh: np.testing.assert_array_equal(s[k], fresh[k])


def test_histogram_and_running_stats_match_brute_force(tiny_cfg):
    batches = random_batches(seed=5, n_batches=4)
    s = recorded(tiny_cfg, batches).state()
    hist, count, total, hi, lo = brute_force_stats(*batches[:4])
    np.testing.assert_array_equal(s["hist"], hist)
    assert (s["count"] == count).all() and s["hist"].sum() == count * N
    np.testing.assert_allclose(s["sum"], total, rtol=1e-12)
    np.testing.assert_array_equal(s["max"], hi.astype(np.float32))
    np.testing.assert_array_equal(s["min"], lo.astype(np.float32))


def test_value_on_a_bin_edge_goes_to_the_bin_it_closes(tiny_cfg):
    a = HIST_EDGES[100].reshape(1, 1, 1)
    s = update_once(tiny_cfg, a, torch.ones_like(a), torch.ones_like(a))
    assert s["hist"][0, 3, 100] == 1


def test_histogram_runs_in_bounded_chunks(tiny_cfg, monkeypatch):
    a = torch.randn(4, 4096, N); g = torch.randn(4, 4096, N); v = torch.randn(4, 4096, N)
    sizes, real = [], torch.bucketize
    monkeypatch.setattr(torch, "bucketize", lambda x, *p, **k: sizes.append(x.shape[0]) or real(x, *p, **k))
    s = update_once(tiny_cfg, a, g, v)
    assert max(sizes) <= 8192 and sum(sizes) == 4 * 4096 and s["hist"].sum() == 4 * 4096 * N


# ---------------------------------------------------------------- markers, ties, state handling

def test_unfilled_slots_are_marked_and_lists_sorted(tiny_cfg):
    batches = random_batches(seed=6, n_batches=1, n_docs=2)
    s = recorded(tiny_cfg, batches).state()
    for name in ("pos", "neg", "dn"):
        val, empty = s[f"{name}_val"], np.isneginf(s[f"{name}_val"])
        assert empty.any() and (val == -np.sort(-val, axis=1)).all()
        assert (s[f"{name}_doc"][empty] == -1).all() and (s[f"{name}_tok"][empty] == -1).all()
        assert (s[f"{name}_doc"][~empty] >= 0).all() and np.isnan(s[f"{name}_win"][empty]).all()


def test_negative_pole_stores_magnitude_and_signed_window(tiny_cfg):
    a = torch.zeros(1, 5, 1); a[0, 2, 0] = -2.5
    s = update_once(tiny_cfg, a, torch.ones_like(a), -torch.ones_like(a))
    assert s["neg_val"][0, 0] == pytest.approx(2.5) and s["neg_win"][0, 0, 32] == pytest.approx(-2.5)


def test_windows_hold_only_positions_valid_for_their_list(tiny_cfg):
    """R19. Tokens: 0 ineligible, 1 gate-negative, 2 gate-positive peak, 3 gate-negative double negative, 4 gate-positive."""
    a = torch.tensor([[[5.0], [-3.0], [9.0], [4.0], [-2.0]]])
    g = torch.tensor([[[1.0], [-1.0], [1.0], [-1.0], [1.0]]])
    v = torch.tensor([[[1.0], [1.0], [1.0], [-1.0], [-1.0]]])
    elig = torch.tensor([[False, True, True, True, True]])
    s = update_once(tiny_cfg, a, g, v, elig)

    def by_token(win, t):
        return {p: float(win[32 + p - t]) for p in range(5)}

    for name, t in (("pos", 2), ("neg", 4)):
        assert s[f"{name}_tok"][0, 0] == t
        got = by_token(s[f"{name}_win"][0, 0], t)
        assert got[2] == 9 and got[4] == -2 and all(np.isnan(got[p]) for p in (0, 1, 3)), name
    assert s["dn_tok"][0, 0] == 3
    got = by_token(s["dn_win"][0, 0], 3)
    assert got[1] == -3 and got[3] == 4 and all(np.isnan(got[p]) for p in (0, 2, 4))
    got = by_token(s["rnd_win"][0, 0], int(s["rnd_tok"][0]))
    assert got[2] == 9 and got[4] == -2 and all(np.isnan(got[p]) for p in (0, 1, 3))


def test_hiding_window_positions_leaves_values_and_statistics_alone(tiny_cfg):
    batches = random_batches(seed=9, n_batches=3)
    s = recorded(tiny_cfg, batches).state()
    want = brute_force_top(*batches)
    hist, count, total, hi, lo = brute_force_stats(*batches[:4])
    for name in ("pos", "neg", "dn"):
        np.testing.assert_allclose(s[f"{name}_val"], want[f"{name}_val"], rtol=1e-6)
    np.testing.assert_array_equal(s["hist"], hist)
    assert (s["count"] == count).all()
    np.testing.assert_allclose(s["max"], hi, rtol=1e-6)
    np.testing.assert_allclose(s["min"], lo, rtol=1e-6)
    regimes = [p[5] for p in brute_force_random(*batches)]
    np.testing.assert_array_equal(s["rnd_regime"][:, :len(regimes)].T, regimes)


def test_cap_at_merge_keeps_windows_aligned_with_values(tiny_cfg):
    def row(peaks, background):
        a = (torch.linspace(0.01, 0.12, 12) + background).repeat(2, 1).T
        for neuron, tokens in enumerate(peaks):
            for tok, val in tokens.items():
                a[tok, neuron] = val
        return a
    first = row([{2: 9.0, 5: 7.0, 8: 5.0}, {1: 6.0, 4: 5.0, 9: 4.0}], 0.0)
    second = row([{3: 8.0, 6: 6.0, 10: 4.0}, {2: 9.0, 7: 8.0, 11: 7.0}], 0.2)
    other = row([{4: 7.5}, {5: 8.5}], 0.4)
    rec = LayerRecorder(replace(tiny_cfg, n_neurons=2), "cpu")
    for rows, ids in (([first], [5]), ([second, other], [5, 6])):
        a = torch.stack(rows); docs = torch.tensor(ids, dtype=torch.int64)
        rec.update(a, torch.ones_like(a), torch.ones_like(a), torch.ones(a.shape[:2], dtype=torch.bool), docs, doc_keys=keys_for(docs))
    s = rec.state()
    want = {0: [(9.0, 5, 2, first), (8.0, 5, 3, second), (7.5, 6, 4, other), (7.0, 5, 5, first)],
            1: [(9.0, 5, 2, second), (8.5, 6, 5, other), (8.0, 5, 7, second), (7.0, 5, 11, second)]}
    for n, entries in want.items():  # entries ranked below the dropped ones must still carry their own windows
        entries += [(other[t, n].item(), 6, t, other) for t in (11, 10)]
        assert (s["pos_doc"][n] == 5).sum() == 3  # the weaker entries of document 5 were dropped at merge
        for i, (val, doc, tok, source) in enumerate(entries):
            assert (s["pos_val"][n, i], s["pos_doc"][n, i], s["pos_tok"][n, i]) == (val, doc, tok)
            expected = np.array(brute_window(source[:, n].numpy(), np.ones(12, bool), tok, 12), np.float32).astype(np.float16)
            np.testing.assert_array_equal(s["pos_win"][n, i], expected)


@pytest.mark.parametrize("a, g, v", [
    (0.0, 1.0, 1.0), (0.0, 1.0, -1.0), (0.0, -1.0, -1.0), (0.0, -1.0, 1.0),  # a is zero
    (1.0, 0.0, 1.0), (-1.0, 0.0, 1.0), (1.0, 0.0, -1.0), (0.0, 0.0, -1.0),   # gate is zero
    (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0),    # value is zero
])
def test_exact_zero_enters_no_list(tiny_cfg, a, g, v):
    s = update_once(tiny_cfg, torch.tensor([[[a]]]), torch.tensor([[[g]]]), torch.tensor([[[v]]]))
    for name in ("pos", "neg", "dn"):
        assert np.isneginf(s[f"{name}_val"]).all() and (s[f"{name}_doc"] == -1).all()
    assert s["count"][0] == 1  # the position still counts as eligible


def test_ties_keep_input_order(tiny_cfg):
    a = torch.ones(30, 1, 1)
    rec = LayerRecorder(replace(tiny_cfg, n_neurons=1), "cpu")
    for start in (0, 30):
        docs = torch.arange(start, start + 30, dtype=torch.int64)
        rec.update(a, a, a, torch.ones(30, 1, dtype=torch.bool), docs, doc_keys=keys_for(docs))
    assert rec.state()["pos_doc"][0].tolist() == list(range(50))


def test_identical_input_gives_identical_state(tiny_cfg):
    batches = random_batches(seed=7, n_batches=3)
    one, two = recorded(tiny_cfg, batches).state(), recorded(tiny_cfg, batches).state()
    for k in one: np.testing.assert_array_equal(one[k], two[k])


def test_state_is_a_snapshot_and_load_copies(tiny_cfg):
    batches = random_batches(seed=8, n_batches=2)
    rec = recorded(tiny_cfg, [x[:1] for x in batches])
    snap = rec.state(); frozen = {k: v.copy() for k, v in snap.items()}
    other = LayerRecorder(replace(tiny_cfg, n_neurons=N), "cpu"); other.load(snap)
    for r in (rec, other):
        r.update(batches[0][1], batches[1][1], batches[2][1], batches[3][1], batches[4][1], doc_keys=keys_for(batches[4][1]))
    for k in frozen: np.testing.assert_array_equal(snap[k], frozen[k])


def test_state_arrays_have_the_contract_shapes_and_types(tiny_cfg):
    s = LayerRecorder(replace(tiny_cfg, n_neurons=N), "cpu").state()
    for name, k in (("pos", 50), ("neg", 50), ("dn", 20)):
        assert s[f"{name}_val"].shape == (N, k) and s[f"{name}_val"].dtype == np.float32
        assert s[f"{name}_doc"].shape == s[f"{name}_tok"].shape == (N, k)
        assert s[f"{name}_win"].shape == (N, k, 64) and s[f"{name}_win"].dtype == np.float16
    assert s["rnd_key"].shape == s["rnd_doc"].shape == s["rnd_tok"].shape == (50,)
    assert s["rnd_win"].shape == (N, 50, 64) and s["rnd_win"].dtype == np.float16
    assert s["rnd_regime"].shape == (N, 50) and s["rnd_regime"].dtype == np.int8
    assert s["hist"].shape == (N, 4, 256) and s["hist"].dtype == np.int64
    assert s["count"].dtype == np.int64 and s["sum"].dtype == np.float64


def test_load_rejects_a_state_of_another_shape(tiny_cfg):
    s = LayerRecorder(replace(tiny_cfg, n_neurons=N), "cpu").state()
    with pytest.raises(ValueError, match="pos_val"):
        LayerRecorder(replace(tiny_cfg, n_neurons=N + 1), "cpu").load(s)


def test_non_finite_positions_are_skipped(tiny_cfg):
    a = torch.ones(1, 4, 2); a[0, 1, 0] = float("nan")
    s = update_once(tiny_cfg, a, torch.ones_like(a), -torch.ones_like(a))
    assert (s["count"] == 3).all() and s["hist"].sum() == 6 and np.isfinite(s["sum"]).all()
    assert 1 not in s["dn_tok"][0].tolist() + s["pos_tok"][0].tolist()


def test_positions_after_the_last_eligible_token_count_as_padding(tiny_cfg):
    a = torch.ones(1, 8, 1)
    elig = torch.ones(1, 8, dtype=torch.bool); elig[0, 3] = False; elig[0, 6:] = False
    s = update_once(tiny_cfg, a * 2, a, a, elig)
    w, t = s["pos_win"][0, 0], s["pos_tok"][0, 0]
    assert np.isnan(w[32 + 3 - t])  # interior ineligible token: not valid for any list (R19)
    assert np.isnan(w[32 + 6 - t]) and np.isnan(w[32 + 7 - t])


# ---------------------------------------------------------------- finalize

def bin_with_lower_edge_at_least(x):
    return int(np.searchsorted(LOWER_EDGES, x, side="left"))


def hand_state(cfg, n=5):
    s = LayerRecorder(replace(cfg, n_neurons=n), "cpu").state()
    s["pos_val"][:, 0] = [10, 4, -np.inf, 6, 5]
    s["neg_val"][:, 0] = [2, 1, -np.inf, 6, -np.inf]
    s["count"][:] = 200
    return s


def test_dead_rule_uses_gate_positive_bins_at_or_above_the_threshold(tiny_cfg):
    s = hand_state(tiny_cfg)  # extremes 10, 4, 0, 6, 5: median E = 5, threshold 1.5
    above, straddling = bin_with_lower_edge_at_least(1.5), bin_with_lower_edge_at_least(1.5) - 1
    s["hist"][0, 3, above] = 100          # exactly enough
    s["hist"][1, 3, above] = 99           # one short
    s["hist"][2, 2, above], s["hist"][2, 3, above] = 60, 40  # both gate-positive regimes add up
    s["hist"][3, 0, above], s["hist"][3, 1, above] = 500, 500  # gate-negative positions do not count
    s["hist"][4, 3, straddling] = 1000    # lower edge is below the threshold
    assert LOWER_EDGES[straddling] < 1.5 <= LOWER_EDGES[above]
    assert finalize({15: s}, tiny_cfg)[15]["live"].tolist() == [True, False, True, False, False]


def test_median_extreme_of_zero_makes_every_neuron_dead(tiny_cfg):
    s = LayerRecorder(replace(tiny_cfg, n_neurons=3), "cpu").state()
    s["pos_val"][0, 0] = 5.0
    s["hist"][:, 3, 200] = 1000
    s["count"][:] = 1000
    assert not finalize({15: s}, tiny_cfg)[15]["live"].any()


def test_extremes_and_dominant_pole(tiny_cfg):
    out = finalize({15: hand_state(tiny_cfg)}, tiny_cfg)[15]
    assert out["pos_extreme"].tolist() == [10, 4, 0, 6, 5]
    assert out["neg_extreme"].tolist() == [2, 1, 0, 6, 0]
    assert out["dominant"].tolist() == [1, 1, 1, 1, 1]
    s = hand_state(tiny_cfg); s["neg_val"][1, 0] = 4.5
    assert finalize({15: s}, tiny_cfg)[15]["dominant"].tolist() == [1, -1, 1, 1, 1]


def test_fire_share_uses_own_dominant_extreme_and_all_eligible_positions(tiny_cfg):
    s = hand_state(tiny_cfg)  # neuron 0: dominant extreme 10, threshold 3
    s["hist"][0, 3, bin_with_lower_edge_at_least(2.5)] = 100   # above the layer's 0.3 * median, below its own threshold
    s["hist"][0, 3, bin_with_lower_edge_at_least(3.0)] = 20    # its own pole, at the threshold
    s["hist"][0, 2, bin_with_lower_edge_at_least(9.0)] = 10    # the other pole
    s["hist"][0, 0, bin_with_lower_edge_at_least(9.0)] = 70    # gate-negative
    s["neg_val"][1, 0] = 8.0                                   # neuron 1: negative dominant, threshold 2.4
    s["hist"][1, 2, bin_with_lower_edge_at_least(2.0)] = 11    # above the layer's 0.3 * median, below its own threshold
    s["hist"][1, 2, bin_with_lower_edge_at_least(2.4)] = 30    # its own pole, at the threshold
    s["hist"][1, 3, bin_with_lower_edge_at_least(9.0)] = 50    # the other pole
    s["hist"][1, 1, bin_with_lower_edge_at_least(9.0)] = 40    # gate-negative
    out = finalize({15: s}, tiny_cfg)[15]
    assert out["dominant"][1] == -1
    assert out["fire_share"][0] == pytest.approx(20 / 200)
    assert out["fire_share"][1] == pytest.approx(30 / 200)
    assert out["fire_share"][2] == 0


@pytest.mark.parametrize("dn_top, kept, want", [
    (5.0, [9.0] * 49 + [3.0], True),      # beats the weakest kept example
    (2.0, [9.0] * 49 + [3.0], False),     # does not
    (3.0, [9.0] * 49 + [3.0], False),     # equal is not "exceeds"
    (1.0, [9.0] * 10 + [-np.inf] * 40, True),   # a list with room would have taken it
    (-np.inf, [-np.inf] * 50, False),     # nothing recorded
    (-1.0, [-np.inf] * 50, False),        # a pole only holds positive values
])
def test_dn_flag(tiny_cfg, dn_top, kept, want):
    s = LayerRecorder(replace(tiny_cfg, n_neurons=1), "cpu").state()
    s["dn_val"][0, 0] = dn_top
    s["pos_val"][0] = kept
    assert finalize({15: s}, tiny_cfg)[15]["dn_flag"].tolist() == [want]


def test_finalize_handles_each_layer_separately(tiny_cfg):
    out = finalize({15: hand_state(tiny_cfg), 19: silent_state(tiny_cfg)}, tiny_cfg)
    assert sorted(out) == [15, 19] and out[15]["pos_extreme"][0] == 10 and out[19]["pos_extreme"][0] == 0
