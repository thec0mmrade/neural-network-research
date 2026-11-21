"""Level 1: structural map of every language-model tensor (CPU, weights only). Plan: TODO.md level 1.

    nix-shell nix/cpu-shell.nix --run 'python -m probe.structure'

Writes results/01_structure.json (every tensor's stats + outlier list) and results/01_structure.md. BLAS is capped at
8 threads (user setting for running next to GPU jobs); the 2026-11-13 crash came from an all-core load.

Conventions: Linear weights are (out, in). For a matrix, "rows" are output units and "cols" input units, so for the
write matrices (down_proj, o_proj, out_proj) a row is one residual-stream dimension and a column is one neuron / head
dim. Singular values come from the Gram matrix on the smaller side (exact up to float64 rounding).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "8"

import re  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from probe.weights import FULL_ATTN, LM, N_LAYERS, RESULTS, Checkpoint, save_json  # noqa: E402

WRITE_KINDS = {"mlp.down_proj.weight", "self_attn.o_proj.weight", "linear_attn.out_proj.weight"}
NORM_KINDS = {"input_layernorm.weight", "post_attention_layernorm.weight"}


def kind_of(name):
    """('mlp.down_proj.weight', 15) for a layer tensor, ('embed_tokens.weight', None) otherwise."""
    m = re.match(re.escape(LM) + r"layers\.(\d+)\.(.+)", name)
    return (m.group(2), int(m.group(1))) if m else (name[len(LM):], None)


def dist_stats(x):
    x = x.astype(np.float64).ravel()
    sd = x.std()
    a = np.abs(x)
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(sd), "min": float(x.min()), "max": float(x.max()),
            "max_abs_over_std": float(a.max() / sd) if sd > 0 else None,
            "excess_kurtosis": float(((x - x.mean()) ** 4).mean() / sd ** 4 - 3) if sd > 0 else None,
            "frac_zero": float((x == 0).mean()), "abs_p999": float(np.percentile(a, 99.9))}


def norm_profile(v, k=3):
    """max / median of a set of norms, how many exceed 5x the median, and the top indices."""
    med = float(np.median(v))
    top = np.argsort(v)[::-1][:k]
    return {"median": med, "max": float(v.max()), "max_over_median": float(v.max() / med) if med > 0 else None,
            "n_over_5x": int((v > 5 * med).sum()), "top": [[int(i), float(v[i])] for i in top]}


def spectrum(w, k=8):
    w = w.astype(np.float64)
    G = w.T @ w if w.shape[0] >= w.shape[1] else w @ w.T
    ev = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None)
    s = np.sqrt(ev)
    p = ev / ev.sum()
    nz = p[p > 0]
    return {"top_sv": s[:k].tolist(), "sv1_energy_share": float(p[0]),
            "effective_rank": float(np.exp(-(nz * np.log(nz)).sum())),
            "rank_90": int(np.searchsorted(np.cumsum(p), 0.9) + 1), "min_sv": float(s[-1])}


def matrix_stats(w):
    out = {"shape": list(w.shape), "fro": float(np.linalg.norm(w)), **spectrum(w), "values": dist_stats(w)}
    out["rows"] = norm_profile(np.linalg.norm(w, axis=1))
    out["cols"] = norm_profile(np.linalg.norm(w, axis=0))
    return out


def chunked_matrix_stats(w, chunk=16384):
    """Embedding (248k x 1024): same stats, Gram matrix and moments accumulated in chunks."""
    G = np.zeros((w.shape[1], w.shape[1]))
    rn = np.empty(w.shape[0]); cs = np.zeros(w.shape[1])
    m1 = m2 = m4 = 0.0; mx = -np.inf; mn = np.inf; amax = 0.0; nzero = 0
    for i in range(0, w.shape[0], chunk):
        c = w[i:i + chunk].astype(np.float64)
        G += c.T @ c
        rn[i:i + chunk] = np.linalg.norm(c, axis=1)
        cs += (c ** 2).sum(0)
        m1 += c.sum(); m2 += (c ** 2).sum(); m4 += (c ** 4).sum()
        mx = max(mx, c.max()); mn = min(mn, c.min()); amax = max(amax, np.abs(c).max()); nzero += int((c == 0).sum())
    n = w.size
    mean = m1 / n; var = m2 / n - mean ** 2; sd = np.sqrt(var)
    ev = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None); p = ev / ev.sum(); nz = p[p > 0]
    return {"shape": list(w.shape), "fro": float(np.sqrt(m2)),
            "top_sv": np.sqrt(ev[:8]).tolist(), "sv1_energy_share": float(p[0]),
            "effective_rank": float(np.exp(-(nz * np.log(nz)).sum())),
            "rank_90": int(np.searchsorted(np.cumsum(p), 0.9) + 1), "min_sv": float(np.sqrt(ev[-1])),
            "values": {"n": int(n), "mean": float(mean), "std": float(sd), "min": float(mn), "max": float(mx),
                       "max_abs_over_std": float(amax / sd), "excess_kurtosis": float((m4 / n) / var ** 2 - 3),  # mean ~ 0
                       "frac_zero": nzero / n, "abs_p999": None},
            "rows": norm_profile(rn), "cols": norm_profile(np.sqrt(cs))}


def vector_stats(name, v):
    out = {"shape": list(v.shape), "values": dist_stats(v)}
    if v.ndim == 1 and v.size >= 64:
        med = np.median(v); mad = np.median(np.abs(v - med)) * 1.4826
        z = (v - med) / mad if mad > 0 else np.zeros_like(v)
        top = np.argsort(-np.abs(z))[:5]
        out["robust_z_top"] = [[int(i), float(v[i]), float(z[i])] for i in top]
        out["n_robust_z_over_6"] = int((np.abs(z) > 6).sum())
    if name.endswith("conv1d.weight"):  # (channels, 1, taps): energy per tap and per channel
        c = v[:, 0, :].astype(np.float64)
        out["tap_energy_share"] = ((c ** 2).sum(0) / (c ** 2).sum()).tolist()
        out["channels"] = norm_profile(np.linalg.norm(c, axis=1))
        out["channels_by_section"] = {s: float(np.linalg.norm(c[a:b])) for s, (a, b) in
                                      {"q": (0, 2048), "k": (2048, 4096), "v": (4096, 6144)}.items()}
    if name.endswith("A_log"):  # gated delta rule decay: A = -exp(A_log)
        out["A"] = (-np.exp(v.astype(np.float64))).tolist()
    return out


def robust_z(xs):
    xs = np.asarray(xs, dtype=np.float64)
    med = np.median(xs); mad = np.median(np.abs(xs - med)) * 1.4826
    return (xs - med) / mad if mad > 0 else np.zeros_like(xs)


def find_outliers(T, emb_tokens):
    """Within each tensor kind (the same matrix across layers), flag layers whose metric is far from the others,
    plus absolute flags: single rows/cols far above their matrix's median, heavy-tailed values, residual dims that
    are extreme in many norm weights or write matrices."""
    flags = []
    by_kind = {}
    for n, t in T.items():
        by_kind.setdefault(t["kind"], []).append(n)
    metrics = {"fro": lambda t: t["fro"], "sv1_energy_share": lambda t: t["sv1_energy_share"],
               "effective_rank": lambda t: t["effective_rank"], "row_max/med": lambda t: t["rows"]["max_over_median"],
               "col_max/med": lambda t: t["cols"]["max_over_median"], "kurtosis": lambda t: t["values"]["excess_kurtosis"],
               "mean": lambda t: t["values"]["mean"], "std": lambda t: t["values"]["std"]}
    for kind, names in by_kind.items():
        if len(names) < 6:
            continue
        for mname, f in metrics.items():
            vals = []
            for n in names:
                try:
                    vals.append(f(T[n]))
                except (KeyError, TypeError):
                    vals = None; break
            if not vals or any(v is None for v in vals):
                continue
            z = robust_z(vals)
            for n, zi, vi in zip(names, z, vals):
                if abs(zi) > 3.5:
                    flags.append({"type": "layer differs from same kind", "tensor": n, "metric": mname, "value": vi,
                                  "robust_z": float(zi), "kind_median": float(np.median(vals))})
    for n, t in T.items():
        for side in ("rows", "cols"):
            if side in t and (t[side]["max_over_median"] or 0) > 8:
                flags.append({"type": f"single {side[:-1]} dominates", "tensor": n, "metric": f"{side} max/median",
                              "value": t[side]["max_over_median"], "top": t[side]["top"]})
        k = t["values"].get("excess_kurtosis")
        if k is not None and t["values"]["n"] > 1000 and k > 30:
            flags.append({"type": "heavy-tailed values", "tensor": n, "metric": "excess kurtosis", "value": k,
                          "max_abs_over_std": t["values"]["max_abs_over_std"]})
    return flags


def residual_dims(ck, k=12):
    """Residual-stream dimensions that stand out across layers.
    write: in how many write matrices (down_proj / o_proj / out_proj, 48 in all) the dim's row norm is > 2x the median
    row, and the mean ratio. read gain: each pre-norm's (1 + w) for the dim over that norm's median, averaged over the
    24 layers (input and post-attention norms separately); below 1 means the model reads that dim weaker than typical."""
    D = 1024
    ratios = []
    for i in range(N_LAYERS):
        for kk in WRITE_KINDS:
            n = f"{LM}layers.{i}.{kk}"
            if n in ck.names():
                rn = np.linalg.norm(ck.get(n), axis=1); ratios.append(rn / np.median(rn))
    R = np.stack(ratios)
    gain = {}
    for kk in NORM_KINDS:
        M = np.stack([1 + ck.get(f"{LM}layers.{i}.{kk}") for i in range(N_LAYERS)])
        gain[kk.split(".")[0]] = (M / np.median(M, axis=1, keepdims=True)).mean(0)
    fn = 1 + ck.get(f"{LM}norm.weight"); fg = fn / np.median(fn)
    def row(d):
        return {"dim": int(d), "n_write_over_2x": int((R[:, d] > 2).sum()), "write_ratio_mean": float(R[:, d].mean()),
                **{f"read_gain_{n}": float(g[d]) for n, g in gain.items()}, "final_norm_gain": float(fg[d])}
    nw = (R > 2).sum(0)
    out = {"n_write_matrices": len(R),
           "top_written": [row(d) for d in np.argsort(-R.mean(0))[:k]],
           "lowest_read_gain_input": [row(d) for d in np.argsort(gain["input_layernorm"])[:k]],
           "highest_read_gain_input": [row(d) for d in np.argsort(-gain["input_layernorm"])[:k]],
           "lowest_final_norm_gain": [row(d) for d in np.argsort(fg)[:k]],
           "n_dims_written_over_2x_anywhere": int((nw > 0).sum())}
    return out


def main():
    t0 = time.time()
    ck = Checkpoint()
    names = ck.names()
    T = {}
    for n in names:
        kind, layer = kind_of(n)
        w = ck.get(n)
        if n.endswith("embed_tokens.weight"):
            st = chunked_matrix_stats(w)
        elif w.ndim == 2:
            st = matrix_stats(w)
        else:
            st = vector_stats(n, w)
        st.update({"kind": kind, "layer": layer, "dtype": ck.info(n)["dtype"],
                   "layer_type": None if layer is None else ("full_attention" if layer in FULL_ATTN else "linear_attention")})
        T[n] = st
        print(f"{n[len(LM):]:52s} {str(w.shape):18s} {time.time() - t0:6.1f}s", flush=True)
    emb_rows = None
    try:
        from probe.lens import show, vocab
        V = vocab()
        e = T[f"{LM}embed_tokens.weight"]
        emb_rows = {"top_norm_tokens": [[i, show(V[i]) if i < len(V) else None, v] for i, v in e["rows"]["top"]]}
    except Exception as ex:  # tokenizer unavailable offline: keep ids only
        emb_rows = {"error": str(ex)}
    out = {"n_tensors": len(T), "n_params": int(sum(np.prod(t["shape"]) for t in T.values())),
           "tensors": {n[len(LM):]: t for n, t in T.items()},
           "outliers": find_outliers(T, emb_rows), "residual_dims": residual_dims(ck), "embedding_rows": emb_rows}
    save_json(out, "01_structure.json")
    write_md(out)
    export_page(out)
    print(f"done: {len(T)} tensors, {len(out['outliers'])} flags, {time.time() - t0:.0f}s")


REVIEW = """## Review (by hand, 2026-11-16)
The automatic flags are below. What they add up to:

1. **Residual dimension 0 is a "massive" dimension that the output can't see.**
   - 43 of the 48 write matrices (nearly every MLP, attention and linear-attention output) write it 2.6× harder than a typical dimension. It is the only dimension written over 2× anywhere.
   - The norms then shrink it when reading: 0.84× (input norms) and 0.67× (post-attention norms) of the typical scale, and **0.05×** in the final norm (w = −0.78, the only negative entry).
   - This matches the "massive activation" pattern reported in other LLMs, where one dimension carries a large constant-like value that works as a bias or attention sink. It is predicted here from weights alone; checking it needs activations.
2. **A band of dimensions is quiet in both directions.** About 400–800 dims (e.g. 443, 521, 523, 545, 807) are written at about 0.4× the median and read at about 0.7×.
3. **Some dimensions are internal-only.** Dims 42–164 (e.g. 46, 66, 74, 143) are read 1.1–1.2× more strongly than typical by the input norms, but the final norm scales them by only 0.25–0.37×. Middle layers use them, but they barely reach the vocabulary.
4. **Layer 23 is the most unusual layer.**
   - Its attention value matrix puts 13% of its energy in one direction, 11× the other full-attention layers (z = +52, the largest flag).
   - Its MLP up-projection has 8× the usual top-direction share, the largest norm, and one dominant column.
   - Its o_proj, k_proj and q_proj are also heavier-tailed.
   - That direction doesn't line up with u, d⊥ or d_close (|cos| < 0.05), or with dim 0 or the mean embedding. So it isn't the closing machinery seen at the weight level, though layer 23 does carry most of `</think>` (`08_circuits/closing.md`).
5. **Layer 0's linear attention is low-rank.** Its in_proj_qkv has effective rank 334 against a typical 678, with 16–20% of its energy in one direction that reads dims 38, 0, 119 and 83. The first layer reads only a few input features strongly.
6. **Smaller flags:**
   - The decay parameters (A_log) of layers 0 and 10 have unusual spreads.
   - Layer 3 (the first full-attention layer) has a larger q_proj.
   - The up_proj norms grow over layers 21–23.
   - The layer-1 conv1d has heavy-tailed taps (kurtosis 48).
7. **No single neuron or column dominates a matrix.** The largest column- or row-norm ratio anywhere is 7.2× (layer 4 in_proj_b), and no write-matrix column exceeds 8×. The weights alone don't point to individually special neurons. That fits Phase 1, where the weights didn't single out the edited neurons either (`03_lens.md`).
8. **The embedding has one strong common direction.** The top singular direction holds 21% of its energy, close to the mean token vector (norm 0.28 against a typical row norm of 0.63). Its effective rank is 370 of 1,024.
9. **The final norm scales by about 4.3** (mean w = +3.3), against 1.1–1.7 for the layer norms. The linear-attention gate norms (`linear_attn.norm`, F32) average 0.94, which fits a norm that scales by w itself rather than (1 + w).

"""


def write_md(o):
    T = o["tensors"]
    L = ["# Level 1: structural map (all language-model tensors, weights only)", "",
         f"{o['n_tensors']} tensors, {o['n_params']:,} parameters. Code: `src/probe/structure.py`; full data: `01_structure.json`.", "",
         "Columns: `fro` = Frobenius norm; `sv1` = share of squared norm in the top singular direction; `eff. rank` = "
         "exp(entropy of the normalized squared singular values); `row`/`col max/med` = largest row/column norm over the "
         "median; `kurt` = excess kurtosis of the values (0 for a Gaussian).", ""] + REVIEW.split("\n")
    kinds = sorted({t["kind"] for t in T.values() if t["layer"] is not None and len(t["shape"]) == 2})
    for k in kinds:
        rows = sorted([t for t in T.values() if t["kind"] == k], key=lambda t: t["layer"])
        L += [f"### `{k}` {tuple(rows[0]['shape'])}", "",
              "| layer | fro | sv1 | eff. rank | rank 90% | row max/med | col max/med | kurt | max/std |",
              "|---|---|---|---|---|---|---|---|---|"]
        for t in rows:
            v = t["values"]
            L.append(f"| {t['layer']} | {t['fro']:.1f} | {t['sv1_energy_share']:.3f} | {t['effective_rank']:.0f} | {t['rank_90']} | "
                     f"{t['rows']['max_over_median']:.1f} | {t['cols']['max_over_median']:.1f} | {v['excess_kurtosis']:.1f} | "
                     f"{v['max_abs_over_std']:.0f} |")
        L.append("")
    L += ["### Vectors (norms, linear-attention parameters)", "",
          "| tensor kind | layers | mean (range over layers) | std (range) | max |w| | |z|>6 elements (range) |", "|---|---|---|---|---|---|"]
    vk = sorted({t["kind"] for t in T.values() if len(t["shape"]) != 2})
    for k in vk:
        rows = [t for t in T.values() if t["kind"] == k]
        ms = [t["values"]["mean"] for t in rows]; ss = [t["values"]["std"] for t in rows]
        mx = max(max(abs(t["values"]["min"]), abs(t["values"]["max"])) for t in rows)
        zz = [t.get("n_robust_z_over_6") for t in rows if t.get("n_robust_z_over_6") is not None]
        L.append(f"| `{k}` | {len(rows)} | {np.mean(ms):+.3f} ({min(ms):+.3f} to {max(ms):+.3f}) | {np.mean(ss):.3f} "
                 f"({min(ss):.3f} to {max(ss):.3f}) | {mx:.3f} | {f'{min(zz)}–{max(zz)}' if zz else ''} |")
    L += ["", "### Outlier flags", "",
          f"{len(o['outliers'])} flags. Thresholds: robust z > 3.5 against the same tensor kind in other layers; a single row "
          "or column > 8x its matrix's median; excess kurtosis > 30.", "",
          "| type | tensor | metric | value | detail |", "|---|---|---|---|---|"]
    for f in sorted(o["outliers"], key=lambda f: (f["type"], f["tensor"])):
        det = f"z {f['robust_z']:+.1f} (kind median {f['kind_median']:.3g})" if "robust_z" in f else \
            (f"top {f['top'][:2]}" if "top" in f else f"max/std {f.get('max_abs_over_std', 0):.0f}")
        L.append(f"| {f['type']} | `{f['tensor'][len(LM):] if f['tensor'].startswith(LM) else f['tensor']}` | {f['metric']} | {f['value']:.3g} | {det} |")
    rd = o["residual_dims"]
    L += ["", "### Residual-stream dimensions that stand out", "",
          f"`write` = in how many of the {rd['n_write_matrices']} write matrices the dim's row norm is > 2x the median row "
          "(mean ratio in brackets). `read gain` = the pre-norm scale (1 + w) for that dim over the norm's median, averaged "
          "over 24 layers; < 1 means the dim is read weaker than typical. `final` = the same for the final norm.", ""]
    for title, key in [("Most strongly written", "top_written"), ("Weakest read (input norms)", "lowest_read_gain_input"),
                       ("Strongest read (input norms)", "highest_read_gain_input"), ("Weakest in the final norm", "lowest_final_norm_gain")]:
        L += [f"**{title}**", "", "| dim | write | read gain (input) | read gain (post-attn) | final |", "|---|---|---|---|---|"]
        for d in rd[key][:6]:
            L.append(f"| {d['dim']} | {d['n_write_over_2x']} ({d['write_ratio_mean']:.2f}x) | {d['read_gain_input_layernorm']:.2f} | "
                     f"{d['read_gain_post_attention_layernorm']:.2f} | {d['final_norm_gain']:.2f} |")
        L.append("")
    (RESULTS / "01_structure.md").write_text("\n".join(L) + "\n")


def export_page(d):
    """results/01_structure/page.html: the heatmap page (template + embedded data, no server needed)."""
    import json
    rows = {}
    for t in d["tensors"].values():
        if t["layer"] is None:
            continue
        m = {"fro": t.get("fro"), "sv1": t.get("sv1_energy_share"), "erank": t.get("effective_rank"),
             "rowmm": t.get("rows", {}).get("max_over_median"), "colmm": t.get("cols", {}).get("max_over_median"),
             "kurt": t["values"]["excess_kurtosis"], "std": t["values"]["std"], "mean": t["values"]["mean"],
             "maxstd": t["values"]["max_abs_over_std"]}
        det = {"shape": t["shape"], "dtype": t["dtype"], "top_sv": [round(x, 3) for x in t.get("top_sv", [])[:5]],
               "rank90": t.get("rank_90"), "rows_top": t.get("rows", {}).get("top"), "cols_top": t.get("cols", {}).get("top"),
               "zt": t.get("robust_z_top"), "tap": t.get("tap_energy_share")}
        rows.setdefault(t["kind"], {})[t["layer"]] = {"m": {k: round(v, 5) if isinstance(v, float) else v for k, v in m.items()}, "d": det}
    flags = [{**f, "tensor": f["tensor"].replace(LM, "")} for f in d["outliers"]]
    data = {"kinds": rows, "flags": flags, "rd": d["residual_dims"], "n_tensors": d["n_tensors"], "n_params": d["n_params"]}
    page = RESULTS / "01_structure"
    tpl = (page / "page.template.html").read_text()
    (page / "page.html").write_text(tpl.replace("__DATA__", json.dumps(data, separators=(",", ":"))))


if __name__ == "__main__":
    main()
