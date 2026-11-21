"""Level 2: static wiring map of Qwen3.5-0.8B from weights alone (CPU, 4 threads).

    nix-shell nix/cpu-shell.nix --run 'python -m probe.wiring <stage>'
stages: build -> hubs -> edges -> uhood -> umap        (lens: GPU shell, after the steering sweep)

Components: 86,016 MLP neurons + 336 attention heads (48 full-attention, 288 linear-attention).
Edge score between a writer W (1024 x m) and a reader R (r x 1024), both Frobenius-normalized:
    score = 1024 * ||R W||_F^2        (expected value 1.0 for a random direction)
For two single neurons this is 1024 * cos^2. Reads go through each layer's (1 + w) RMSNorm.
Allowed edges follow the residual order: attention(l) -> MLP(l) -> attention(l+1) -> ...
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "4")  # user constraint: 4 threads while the GPU sweep runs

import json
import sys
import time

import numpy as np

from probe.weights import FULL_ATTN, LM, N_LAYERS, RESULTS, Checkpoint, layer_tensor, save_json

OUT = RESULTS / "06_wiring"
D = 1024
NEUR = 3584
TOPK = 16


# ---------------------------------------------------------------- components and their directions
def components():
    """Global component table, ordered by layer; within a layer: attention heads, then MLP neurons."""
    layer, kind, idx = [], [], []
    for l in range(N_LAYERS):
        nh = 8 if l in FULL_ATTN else 16
        layer += [l] * nh; kind += [1 if l in FULL_ATTN else 2] * nh; idx += list(range(nh))
        layer += [l] * NEUR; kind += [0] * NEUR; idx += list(range(NEUR))
    return np.array(layer, np.int16), np.array(kind, np.int8), np.array(idx, np.int16)


def comp_name(l, k, i):
    return f"L{l}." + {0: f"mlp.{i}", 1: f"attn.h{i}", 2: f"lin.h{i}"}[int(k)]


def _rows_unit(m):
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def _fro(m):
    return m / np.linalg.norm(m)


def head_blocks(ck, l):
    """Per head: write block (1024 x k) and read blocks {type: (r x 1024)}, reads scaled by the pre-attention norm."""
    g = 1.0 + ck.get(layer_tensor(l, "in_norm"))
    p = f"{LM}layers.{l}."
    heads = []
    if l in FULL_ATTN:
        o, q, k, v = (ck.get(p + f"self_attn.{n}_proj.weight") for n in ("o", "q", "k", "v"))
        assert o.shape == (D, 2048) and q.shape == (4096, D) and k.shape == (512, D)
        for h in range(8):
            gq = h // 4
            heads.append({"write": o[:, h * 256:(h + 1) * 256], "reads": {
                "q": q[h * 512: h * 512 + 256] * g, "gate": q[h * 512 + 256:(h + 1) * 512] * g,
                "k": k[gq * 256:(gq + 1) * 256] * g, "v": v[gq * 256:(gq + 1) * 256] * g}})
        assert np.array_equal(np.concatenate([x["write"] for x in heads], 1), o)
    else:
        o = ck.get(p + "linear_attn.out_proj.weight")
        qkv, z = ck.get(p + "linear_attn.in_proj_qkv.weight"), ck.get(p + "linear_attn.in_proj_z.weight")
        a, b = ck.get(p + "linear_attn.in_proj_a.weight"), ck.get(p + "linear_attn.in_proj_b.weight")
        assert o.shape == (D, 2048) and qkv.shape == (6144, D) and z.shape == (2048, D) and a.shape == (16, D)
        for h in range(16):
            s = slice(h * 128, (h + 1) * 128)
            heads.append({"write": o[:, s], "reads": {
                "q": qkv[s] * g, "k": qkv[2048 + h * 128: 2048 + (h + 1) * 128] * g,
                "v": qkv[4096 + h * 128: 4096 + (h + 1) * 128] * g, "z": z[s] * g,
                "a": a[h:h + 1] * g, "b": b[h:h + 1] * g}})
        assert np.array_equal(np.concatenate([x["write"] for x in heads], 1), o)
    for x in heads:
        x["write"] = _fro(x["write"])
        x["reads"] = {t: _fro(r) for t, r in x["reads"].items()}
    return heads


def load_layer(ck, l):
    gm = 1.0 + ck.get(layer_tensor(l, "post_norm"))
    return {"n_write": _rows_unit(ck.get(layer_tensor(l, "down")).T),            # (3584, 1024)
            "n_gate": _rows_unit(ck.get(layer_tensor(l, "gate")) * gm),         # (3584, 1024)
            "n_up": _rows_unit(ck.get(layer_tensor(l, "up")) * gm),
            "heads": head_blocks(ck, l)}


def cmd_build():
    """Step 1: all directions, normalized, saved as float16 (neuron writes/reads; head top write direction)."""
    ck = Checkpoint()
    OUT.mkdir(parents=True, exist_ok=True)
    nw, ng, nu, hw = [], [], [], []
    for l in range(N_LAYERS):
        L = load_layer(ck, l)
        nw.append(L["n_write"]); ng.append(L["n_gate"]); nu.append(L["n_up"])
        for h in L["heads"]:
            U, s, _ = np.linalg.svd(h["write"], full_matrices=False)
            hw.append(U[:, 0] * np.sign(U[:, 0].sum()))
    layer, kind, idx = components()
    np.savez(OUT / "dirs.npz", neuron_write=np.concatenate(nw).astype(np.float16),
             neuron_gate=np.concatenate(ng).astype(np.float16), neuron_up=np.concatenate(nu).astype(np.float16),
             head_write_top=np.stack(hw).astype(np.float16), layer=layer, kind=kind, idx=idx)
    print(f"components: {len(layer):,} ({(kind == 0).sum():,} neurons, {(kind == 1).sum()} full-attn heads, "
          f"{(kind == 2).sum()} linear-attn heads); dirs.npz {os.path.getsize(OUT / 'dirs.npz') / 1e6:.0f} MB")


# ---------------------------------------------------------------- step 2: hub diagnostic
def cmd_hubs():
    z = np.load(OUT / "dirs.npz")
    res = {}
    for name in ("neuron_write", "neuron_gate", "neuron_up"):
        X = z[name].astype(np.float32)
        C = X.T @ X / len(X)                       # uncentered second moment; trace = 1
        ev, V = np.linalg.eigh(C)
        ev, V = ev[::-1], V[:, ::-1]
        coord = (X**2).mean(0)
        res[name] = {"top_eig_share": ev[:8].tolist(), "uniform_share": 1 / D,
                     "n_eig_over_20x_uniform": int((ev > 20 / D).sum()),
                     "max_coord_share": float(coord.max()), "argmax_coord": int(coord.argmax())}
        np.save(OUT / f"pc_{name}.npy", V[:, :8].astype(np.float32))
        print(name, {k: (np.round(v, 4).tolist() if isinstance(v, list) else v) for k, v in res[name].items()})
    # cosine between the leading write PC and leading read PCs: a shared hub direction would show up here
    pw, pg, pu = (np.load(OUT / f"pc_{n}.npy") for n in ("neuron_write", "neuron_gate", "neuron_up"))
    res["write_vs_read_pc_abs_cos"] = {"gate": np.abs(pw.T @ pg)[:3, :3].round(3).tolist(),
                                       "up": np.abs(pw.T @ pu)[:3, :3].round(3).tolist()}
    print("write PC vs read PC |cos| (3x3):", res["write_vs_read_pc_abs_cos"])
    save_json(res, "06_wiring/hubs.json")


# ---------------------------------------------------------------- step 3: edges
class TopK:
    """Running top-k (by score) per row, with partner ids and signs."""

    def __init__(self, n, k=TOPK):
        self.s = np.full((n, k), -1.0, np.float32)
        self.j = np.full((n, k), -1, np.int32)
        self.sg = np.zeros((n, k), np.int8)

    def update(self, rows, scores, partners, signs):
        """rows: (n,) global ids; scores/signs: (n, m); partners: (m,) global ids."""
        k = self.s.shape[1]
        m = scores.shape[1]
        if m > k:
            part = np.argpartition(-scores, k, axis=1)[:, :k]
            sc = np.take_along_axis(scores, part, 1); sg = np.take_along_axis(signs, part, 1); pj = partners[part]
        else:
            sc, sg, pj = scores, signs, np.broadcast_to(partners, scores.shape)
        S = np.concatenate([self.s[rows], sc], 1); J = np.concatenate([self.j[rows], pj], 1)
        G = np.concatenate([self.sg[rows], sg], 1)
        o = np.argsort(-S, 1)[:, :k]
        self.s[rows] = np.take_along_axis(S, o, 1); self.j[rows] = np.take_along_axis(J, o, 1)
        self.sg[rows] = np.take_along_axis(G, o, 1)


def block_scores(src, tgt):
    """Scores (n_tgt, n_src), signs, and target read type, for one source group and one target group.
    src: ("n", W (n,1024) unit rows) or ("h", [write blocks]); tgt: ("n", gate, up) or ("h", [read dicts])."""
    if src[0] == "n":
        W = src[1]
        if tgt[0] == "n":
            cg, cu = tgt[1] @ W.T, tgt[2] @ W.T                       # (n_tgt, n_src)
            sg, su = D * cg**2, D * cu**2
            use_g = sg >= su
            return np.where(use_g, sg, su), np.sign(np.where(use_g, cg, cu)).astype(np.int8), np.where(use_g, 0, 1)
        out, typ = [], []
        for rd in tgt[1]:
            per = {t: D * ((R @ W.T) ** 2).sum(0) for t, R in rd.items()}  # (n_src,)
            names = list(per)
            st = np.stack([per[t] for t in names])
            out.append(st.max(0)); typ.append(st.argmax(0))
        return np.stack(out), np.zeros((len(out), W.shape[0]), np.int8), np.stack(typ)
    blocks = src[1]
    if tgt[0] == "n":
        cols = []
        for O in blocks:
            sg_, su_ = D * ((tgt[1] @ O) ** 2).sum(1), D * ((tgt[2] @ O) ** 2).sum(1)
            cols.append((np.maximum(sg_, su_), np.where(sg_ >= su_, 0, 1)))
        return (np.stack([c[0] for c in cols], 1), np.zeros((tgt[1].shape[0], len(blocks)), np.int8),
                np.stack([c[1] for c in cols], 1))
    S = np.zeros((len(tgt[1]), len(blocks)), np.float32); T = np.zeros_like(S, dtype=np.int64)
    for a, rd in enumerate(tgt[1]):
        for b, O in enumerate(blocks):
            per = [D * float(((R @ O) ** 2).sum()) for R in rd.values()]
            S[a, b], T[a, b] = max(per), int(np.argmax(per))
    return S, np.zeros(S.shape, np.int8), T


def groups(L, rot=None):
    """Source and target groups for one layer; rot (1024x1024) rotates the READ side for the noise floor."""
    r = (lambda m: m @ rot) if rot is not None else (lambda m: m)
    return {"src_n": ("n", L["n_write"]), "src_h": ("h", [h["write"] for h in L["heads"]]),
            "tgt_n": ("n", r(L["n_gate"]), r(L["n_up"])),
            "tgt_h": ("h", [{t: r(R) for t, R in h["reads"].items()} for h in L["heads"]])}


def allowed_pairs():
    """(src layer, src group, tgt layer, tgt group) in residual order."""
    for s in range(N_LAYERS):
        yield s, "src_h", s, "tgt_n"                                   # attention(l) -> MLP(l)
        for t in range(s + 1, N_LAYERS):
            for sg in ("src_n", "src_h"):
                for tg in ("tgt_n", "tgt_h"):
                    yield s, sg, t, tg


def block_kind(sg, tg):
    return {"src_n": "n", "src_h": "h"}[sg] + "->" + {"tgt_n": "n", "tgt_h": "h"}[tg]


def noise_floor(ck, deflate=None):
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    dist = {}
    for s, t in ((2, 10), (10, 17), (15, 19)):
        A, B = prep(load_layer(ck, s), deflate), prep(load_layer(ck, t), deflate)
        gs, gt = groups(A), groups(B, rot=Q)
        for sg in ("src_n", "src_h"):
            for tg in ("tgt_n", "tgt_h"):
                S, _, _ = block_scores(gs[sg], gt[tg])
                dist.setdefault(block_kind(sg, tg), []).append(S.ravel())
    return {k: float(np.percentile(np.concatenate(v), 99.9)) for k, v in dist.items()}


def prep(L, deflate):
    """Optionally project shared hub directions (rows of `deflate`, orthonormal) out of every direction."""
    if deflate is None:
        return L
    P = np.eye(D, dtype=np.float32) - deflate.T @ deflate
    L = dict(L)
    L["n_write"] = _rows_unit(L["n_write"] @ P); L["n_gate"] = _rows_unit(L["n_gate"] @ P); L["n_up"] = _rows_unit(L["n_up"] @ P)
    L["heads"] = [{"write": _fro(P @ h["write"]), "reads": {t: _fro(R @ P) for t, R in h["reads"].items()}} for h in L["heads"]]
    return L


def cmd_edges(deflate_k=0):
    ck = Checkpoint()
    layer, kind, idx = components()
    n = len(layer)
    starts = {}
    for l in range(N_LAYERS):
        ids = np.flatnonzero(layer == l)
        starts[l] = {"h": ids[kind[ids] != 0], "n": ids[kind[ids] == 0]}
    deflate = None
    tag = "raw"
    if deflate_k:
        deflate = np.load(OUT / "pc_neuron_write.npy")[:, :deflate_k].T.astype(np.float32)
        tag = f"deflate{deflate_k}"
    t0 = time.time()
    floor = noise_floor(ck, deflate)
    print(f"[{tag}] noise floor (99.9th pct, rotated reads): {floor}  ({time.time() - t0:.0f}s)")
    out_k, in_k = TopK(n), TopK(n)
    out_t, in_t = np.full((n, TOPK), -1, np.int8), None  # read type for out edges is stored per edge below
    heat = {}
    cache = {}

    def get(l):
        if l not in cache:
            cache.clear() if len(cache) > 6 else None
            cache[l] = groups(prep(load_layer(ck, l), deflate))
        return cache[l]

    t0 = time.time()
    for s, sg, t, tg in allowed_pairs():
        S, sign, _typ = block_scores(get(s)[sg], get(t)[tg])
        src_ids = starts[s]["n" if sg == "src_n" else "h"]
        tgt_ids = starts[t]["n" if tg == "tgt_n" else "h"]
        out_k.update(src_ids, S.T, tgt_ids, sign.T)
        in_k.update(tgt_ids, S, src_ids, sign)
        bk = block_kind(sg, tg)
        above = S > floor[bk]
        heat[(s, t, bk)] = (int(above.sum()), float(S[above].sum()), int(S.size), float(S.max()))
        if t == s + 1 and sg == "src_n" and tg == "tgt_n":
            print(f"  layer {s:2d} done ({time.time() - t0:.0f}s)", flush=True)
    np.savez(OUT / f"edges_{tag}.npz", out_score=out_k.s, out_id=out_k.j, out_sign=out_k.sg,
             in_score=in_k.s, in_id=in_k.j, in_sign=in_k.sg)
    save_json({"tag": tag, "floor": floor, "cells": [
        {"src": s, "tgt": t, "kind": k, "n_above": a, "sum_above": sm, "n_pairs": np_, "max": mx}
        for (s, t, k), (a, sm, np_, mx) in heat.items()]}, f"06_wiring/heatmap_{tag}.json")
    print(f"[{tag}] edges done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- step 4: u neighborhood
def cmd_uhood():
    ck = Checkpoint()
    layer, kind, idx = components()
    res = {}
    for name in ("u15", "u19"):
        u = np.load(RESULTS / f"{name}_p50c.npy").astype(np.float32)
        u /= np.linalg.norm(u)
        writers, readers = [], []
        for l in range(N_LAYERS):
            L = load_layer(ck, l)
            ids = np.flatnonzero(layer == l)
            hid, nid = ids[kind[ids] != 0], ids[kind[ids] == 0]
            c = L["n_write"] @ u
            writers += [(float(D * x**2), int(np.sign(x)), int(i)) for x, i in zip(c, nid)]
            writers += [(float(D * ((h["write"].T @ u) ** 2).sum()), 0, int(i)) for h, i in zip(L["heads"], hid)]
            cg, cu = L["n_gate"] @ u, L["n_up"] @ u
            for x, y, i in zip(cg, cu, nid):
                readers.append((float(D * max(x**2, y**2)), int(np.sign(x if x**2 >= y**2 else y)), int(i),
                                "gate" if x**2 >= y**2 else "up"))
            for h, i in zip(L["heads"], hid):
                per = {t: float(D * ((R @ u) ** 2).sum()) for t, R in h["reads"].items()}
                tt = max(per, key=per.get)
                readers.append((per[tt], 0, int(i), tt))
        writers.sort(key=lambda r: -r[0]); readers.sort(key=lambda r: -r[0])
        nm = lambda i: comp_name(layer[i], kind[i], idx[i])
        res[name] = {"top_writers": [{"id": i, "name": nm(i), "score": s, "sign": g} for s, g, i in writers[:40]],
                     "top_readers": [{"id": i, "name": nm(i), "score": s, "sign": g, "via": t} for s, g, i, t in readers[:40]]}
        late = [r for r in res[name]["top_readers"] if layer[r["id"]] >= 16 and kind[r["id"]] == 0]
        print(name, "top MLP readers in layers 16-23:", [(r["name"], r["via"], round(r["score"], 1)) for r in late[:6]])
    save_json(res, "06_wiring/uhood.json")


# ---------------------------------------------------------------- step 5: embedding coordinates
def cmd_umap():
    import umap
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    z = np.load(OUT / "dirs.npz")
    X = np.concatenate([z["head_write_top"], z["neuron_write"]]).astype(np.float32)
    layer, kind, idx = components()
    order = np.concatenate([np.flatnonzero(kind != 0), np.flatnonzero(kind == 0)])  # matches X row order
    t0 = time.time()
    Xp = PCA(128, random_state=0).fit_transform(X)
    Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
    xy = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=0, low_memory=True).fit_transform(Xp)
    km = MiniBatchKMeans(150, random_state=0, n_init=3).fit(Xp)
    print(f"umap + kmeans in {time.time() - t0:.0f}s")
    coords = np.zeros((len(layer), 2), np.float32); cl = np.zeros(len(layer), np.int16)
    coords[order] = xy; cl[order] = km.labels_
    cent = np.array([coords[cl == c].mean(0) for c in range(150)])
    np.savez(OUT / "umap.npz", xy=coords, cluster=cl, centroid=cent)
    print("saved umap.npz;", "cluster sizes min/median/max:", np.bincount(cl).min(), int(np.median(np.bincount(cl))), np.bincount(cl).max())


# ---------------------------------------------------------------- step 6: logit lens for every component (GPU shell)
def cmd_lens(k=10, chunk=2048):
    """Top/bottom-k tokens of E @ ((1+g_final) * w) for every write direction (heads: top singular direction),
    and for each UMAP cluster's mean write direction. Needs torch + CUDA: run in the GPU shell."""
    import torch
    from probe.lens import show, vocab
    ck = Checkpoint()
    dev = "cuda"
    E = torch.tensor(ck.get(f"{LM}embed_tokens.weight"), dtype=torch.float16, device=dev)   # (V, 1024)
    g = torch.tensor(1.0 + ck.get(f"{LM}norm.weight"), dtype=torch.float32, device=dev)
    z = np.load(OUT / "dirs.npz")
    layer, kind, idx = components()
    W = np.zeros((len(layer), D), np.float32)
    W[kind != 0] = z["head_write_top"].astype(np.float32)
    W[kind == 0] = z["neuron_write"].astype(np.float32)
    cl = np.load(OUT / "umap.npz")["cluster"]
    C = np.stack([W[cl == c].mean(0) for c in range(cl.max() + 1)])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    inv = vocab()

    def run(M):
        tops, bots = [], []
        for s in range(0, len(M), chunk):
            x = torch.tensor(M[s:s + chunk], device=dev) * g
            logit = x.half() @ E.T                                              # (chunk, V)
            tops.append(torch.topk(logit, k, dim=1).indices.cpu()); bots.append(torch.topk(-logit, k, dim=1).indices.cpu())
        return torch.cat(tops).numpy(), torch.cat(bots).numpy()

    t0 = time.time()
    tp, bt = run(W)
    ctp, cbt = run(C)
    tok = lambda ids: [show(inv.get(int(i), f"<{i}>")) for i in ids]
    lens = {"note": "tokens promoted (+) / suppressed (-) by writing +direction; heads use their top singular "
                    "direction, whose sign is arbitrary", "k": k,
            "components": [[tok(a), tok(b)] for a, b in zip(tp, bt)],
            "clusters": [[tok(a), tok(b)] for a, b in zip(ctp, cbt)]}
    (OUT / "lens.json").write_text(json.dumps(lens, ensure_ascii=False, separators=(",", ":")))
    print(f"lens for {len(W):,} components + {len(C)} clusters in {time.time() - t0:.0f}s; "
          f"lens.json {os.path.getsize(OUT / 'lens.json') / 1e6:.1f} MB")
    for c in np.argsort(-np.bincount(cl))[:8]:
        print(f"  cluster {c} (n={np.sum(cl == c)}): + {' | '.join(lens['clusters'][c][0][:6])}")


# ---------------------------------------------------------------- step 7a: export page data
def cmd_export(n_edges=8, n_tok=5):
    """results/06_wiring/page/data/: meta.json (small, loaded first) + L00..L23.json (lazy, per layer)."""
    layer, kind, idx = components()
    z = np.load(OUT / "dirs.npz")
    W = np.zeros((len(layer), D), np.float32)
    W[kind != 0] = z["head_write_top"].astype(np.float32); W[kind == 0] = z["neuron_write"].astype(np.float32)
    u15 = np.load(RESULTS / "u15_p50c.npy").astype(np.float32); u15 /= np.linalg.norm(u15)
    ucos = W @ u15
    e = np.load(OUT / "edges_raw.npz")
    um = np.load(OUT / "umap.npz")
    lens = json.loads((OUT / "lens.json").read_text())
    heat = json.loads((OUT / "heatmap_raw.json").read_text())
    uh = json.loads((RESULTS / "06_wiring" / "uhood.json").read_text())
    ed = json.loads((RESULTS / "02_edit.json").read_text())["p50c"]
    edited = []
    for l in (15, 19):
        base = int(np.flatnonzero((layer == l) & (kind == 0))[0])
        edited += [base + c for c in ed[str(l)]["columns"]]
    xy = um["xy"]; lo, hi = xy.min(0), xy.max(0)
    xyn = np.round((xy - lo) / (hi - lo) * 1000).astype(int)          # 0..1000 grid
    cl = um["cluster"]
    clusters = []
    for c in range(cl.max() + 1):
        m = cl == c
        lc = np.bincount(layer[m], minlength=N_LAYERS)
        top_layers = [[int(l), round(float(lc[l] / m.sum()), 2)] for l in np.argsort(-lc)[:3] if lc[l]]
        clusters.append({"n": int(m.sum()), "xy": np.round(xyn[m].mean(0)).astype(int).tolist(),
                         "layers": top_layers, "heads": int((kind[m] != 0).sum()),
                         "plus": lens["clusters"][c][0][:n_tok], "minus": lens["clusters"][c][1][:n_tok]})
    offsets = [int(np.flatnonzero(layer == l)[0]) for l in range(N_LAYERS)]
    nheads = [int(((layer == l) & (kind != 0)).sum()) for l in range(N_LAYERS)]
    meta = {"n": int(len(layer)), "offsets": offsets, "nheads": nheads, "full_attn": sorted(FULL_ATTN),
            "xy": xyn.ravel().tolist(), "cluster": cl.tolist(), "clusters": clusters, "edited": edited,
            "floor": heat["floor"], "heat": heat["cells"],
            "uhood": {k: {"writers": v["top_writers"][:12], "readers": v["top_readers"][:12]} for k, v in uh.items()}}
    pd = OUT / "page" / "data"
    pd.mkdir(parents=True, exist_ok=True)
    (pd / "meta.json").write_text(json.dumps(meta, separators=(",", ":"), ensure_ascii=False))
    total = os.path.getsize(pd / "meta.json")
    for l in range(N_LAYERS):
        ids = np.flatnonzero(layer == l)
        rows = []
        for i in ids:
            out = [[int(j), int(round(float(s))), int(g)] for s, j, g in zip(e["out_score"][i][:n_edges], e["out_id"][i][:n_edges], e["out_sign"][i][:n_edges]) if j >= 0]
            inn = [[int(j), int(round(float(s))), int(g)] for s, j, g in zip(e["in_score"][i][:n_edges], e["in_id"][i][:n_edges], e["in_sign"][i][:n_edges]) if j >= 0]
            rows.append([lens["components"][i][0][:n_tok], lens["components"][i][1][:n_tok], out, inn, round(float(ucos[i]), 3)])
        f = pd / f"L{l:02d}.json"
        f.write_text(json.dumps(rows, separators=(",", ":"), ensure_ascii=False))
        total += os.path.getsize(f)
    print(f"exported meta.json ({os.path.getsize(pd / 'meta.json') / 1e6:.2f} MB) + 24 layer files; total {total / 1e6:.1f} MB")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"build": cmd_build, "hubs": cmd_hubs, "uhood": cmd_uhood, "umap": cmd_umap, "lens": cmd_lens, "export": cmd_export,
     "edges": lambda: cmd_edges(int(sys.argv[2]) if len(sys.argv) > 2 else 0)}[cmd]()
