"""Phase 2: decompose the three released Sparks Exp 8 edits into the directions they act on.

Requires results/edits/<run>/ from the logs' apply_edit.py (hash-checked). CPU, weights only.
    nix-shell nix/cpu-shell.nix --run 'python -m probe.edit'
"""
import numpy as np

from probe.weights import LM, LOGS, RESULTS, Checkpoint, edited, layer_tensor, save_json


def unit(v):
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(abs(unit(a) @ unit(b)))  # directions are sign-free here


def p50c(ck):
    """Each tensor: which columns changed, is the change rank one, and is it a projection W' = W - a*u(u^T W)?"""
    res, dirs = {}, {}
    for layer in (15, 19):
        name = layer_tensor(layer, "down")
        w, w2 = ck.get(name), edited("p50c", name)
        d = w2 - w
        cols = np.flatnonzero(np.abs(d).sum(0) > 0)
        dc, wc = d[:, cols], w[:, cols]
        U, s, _ = np.linalg.svd(dc, full_matrices=False)
        u = U[:, 0]
        uw = u @ wc
        if uw.sum() < 0:  # orient u along what these columns write
            u, uw = -u, -uw
        alpha = -float((u @ dc) @ uw / (uw @ uw))  # best a in d ~ -a u (u^T W)
        resid = np.linalg.norm(dc + alpha * np.outer(u, uw)) / np.linalg.norm(dc)
        # Of all 3584 columns, where do the edited ones rank by how strongly they write u?
        write = np.abs(u @ w) / np.linalg.norm(w, axis=0)  # |cos| of each column with u
        rank = np.argsort(-np.abs(u @ w))
        top64 = set(rank[:64].tolist())
        res[layer] = {
            "tensor": name, "changed_columns": int(len(cols)), "columns": cols.tolist(),
            "singular_values_top5": s[:5].tolist(), "rank1_share": float(s[0]**2 / (s**2).sum()),
            "alpha": alpha, "rank1_projection_residual": float(resid),
            "edited_cols_in_top64_by_abs_write": len(top64 & set(cols.tolist())),
            "edited_cols_mean_abs_cos": float(write[cols].mean()), "all_cols_mean_abs_cos": float(write.mean()),
            "share_of_u_written_by_edited_cols": float(np.linalg.norm(u @ wc)**2 / np.linalg.norm(u @ w)**2),
        }
        dirs[layer] = u
    return res, dirs


def p10c():
    """Rank-one factors W' = W + a b^T: is a the same direction in all 30 tensors?"""
    z = np.load(LOGS / "runs/p10c/edit/edit.npz")
    import json
    man = json.loads((LOGS / "runs/p10c/edit/manifest.json").read_text())
    vecs = {}
    for i, t in enumerate(man["tensors"]):
        a = z[f"t{i}_a"]
        assert a.shape[1] == 1, a.shape
        vecs[t["name"].removeprefix(LM)] = unit(a[:, 0].astype(np.float64))
    V = np.stack(list(vecs.values()))
    pair = np.abs(V @ V.T)
    return vecs, {"n_tensors": len(vecs), "min_pairwise_abs_cos": float(pair.min()),
                  "mean_pairwise_abs_cos": float(pair[np.triu_indices(len(V), 1)].mean())}


def p25c(ck, dirs):
    """Head 7 of layer 15 was zeroed. How much does each layer-15 head write along the p50c directions?"""
    name = layer_tensor(15, "out")
    w, w2 = ck.get(name), edited("p25c", name)
    zeroed = np.flatnonzero(np.abs(w2).sum(0) == 0)
    out = {"zeroed_columns": [int(zeroed.min()), int(zeroed.max())], "n_zeroed": int(len(zeroed)), "heads": []}
    hd = 256
    for h in range(w.shape[1] // hd):
        wh = w[:, h * hd:(h + 1) * hd]
        row = {"head": h, "fro": float(np.linalg.norm(wh))}
        for layer, u in dirs.items():
            row[f"share_along_u{layer}"] = float(np.linalg.norm(u @ wh)**2 / np.linalg.norm(wh)**2)
        out["heads"].append(row)
    return out


def main():
    ck = Checkpoint()
    r50, dirs = p50c(ck)
    v10, s10 = p10c()
    r25 = p25c(ck, dirs)
    d10 = np.mean(np.stack([np.sign(v @ dirs[15]) * v for v in v10.values()]), 0)
    d10 = unit(d10)
    cross = {
        "cos_u15_u19": cos(dirs[15], dirs[19]),
        "signed_cos_u15_u19": float(unit(dirs[15]) @ unit(dirs[19])),
        "cos_p10c_mean_dir_u15": cos(d10, dirs[15]),
        "cos_p10c_mean_dir_u19": cos(d10, dirs[19]),
        "cos_p10c_per_tensor_vs_u15": {k: cos(v, dirs[15]) for k, v in v10.items()},
        "random_baseline_abs_cos_1024d": float(np.sqrt(2 / (np.pi * 1024))),  # E|cos| of random unit vectors
    }
    save_json({"p50c": r50, "p10c": s10, "p25c": r25, "cross": cross}, "02_edit.json")
    np.save(RESULTS / "u15_p50c.npy", dirs[15]); np.save(RESULTS / "u19_p50c.npy", dirs[19]); np.save(RESULTS / "u_p10c.npy", d10)
    write_md(r50, s10, r25, cross)
    for k, v in r50.items():
        print(f"p50c L{k}: {v['changed_columns']} cols, rank1 share {v['rank1_share']:.4f}, alpha {v['alpha']:.3f}, "
              f"proj resid {v['rank1_projection_residual']:.4f}, edited∩top64 {v['edited_cols_in_top64_by_abs_write']}, "
              f"share of u-writing {v['share_of_u_written_by_edited_cols']:.3f}")
    print("p10c:", s10)
    print("cross:", {k: v for k, v in cross.items() if not isinstance(v, dict)})
    print("p25c zeroed", r25["zeroed_columns"], r25["n_zeroed"])
    for h in r25["heads"]:
        print(f"  L15 head {h['head']}: fro {h['fro']:.2f}  share along u15 {h['share_along_u15']:.4f}  u19 {h['share_along_u19']:.4f}")


def write_md(r50, s10, r25, cross):
    L = ["# Phase 2: the Sparks Exp 8 edits, decomposed", "",
         "All 33 edited tensors (p50c 2, p25c 1, p10c 30) were rebuilt by the logs' `apply_edit.py` and matched their sha256 hashes.", "",
         "## p50c (the 48.1% MATH-500 run)", "",
         "| layer | changed cols | rank-1 share of ΔW | fitted α in ΔW = −α·u(uᵀW) | fit residual | edited cols in top-64 u-writers | share of the layer's u-writing in edited cols |",
         "|---|---|---|---|---|---|---|"]
    for k, v in r50.items():
        L.append(f"| {k} | {v['changed_columns']} | {v['rank1_share']:.4f} | {v['alpha']:.3f} | {v['rank1_projection_residual']:.4f} | "
                 f"{v['edited_cols_in_top64_by_abs_write']}/64 | {v['share_of_u_written_by_edited_cols']:.3f} |")
    L += ["", "α = 6 means each edited neuron's component along u goes from c to −5c: a sign flip with 5× gain, not a removal (α = 1).", "",
          "## Agreement between runs", "",
          f"- u15 vs u19 (p50c): |cos| {cross['cos_u15_u19']:.3f}, signed {cross['signed_cos_u15_u19']:+.3f}",
          f"- p10c mean direction vs u15: {cross['cos_p10c_mean_dir_u15']:.3f}; vs u19: {cross['cos_p10c_mean_dir_u19']:.3f}",
          f"- p10c's 30 rank-one directions among themselves: mean |cos| {s10['mean_pairwise_abs_cos']:.3f}, min {s10['min_pairwise_abs_cos']:.3f}",
          f"- random 1024-d unit vectors: E|cos| ≈ {cross['random_baseline_abs_cos_1024d']:.3f}", "",
          "## p25c (zeroed head 7 of layer 15): share of each head's output weight along u", "",
          f"Chance level for a 256-column head along one direction ≈ {1/1024:.4f}.", "", "| head | share along u15 | share along u19 |", "|---|---|---|"]
    L += [f"| {h['head']} | {h['share_along_u15']:.4f} | {h['share_along_u19']:.4f} |" for h in r25["heads"]]
    (RESULTS / "02_edit.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
