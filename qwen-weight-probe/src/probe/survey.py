"""Phase 1: structural survey of the language model's weights (CPU, weights only).

    nix-shell nix/cpu-shell.nix --run 'python -m probe.survey'
"""
import numpy as np

from probe.weights import FULL_ATTN, N_LAYERS, RESULTS, Checkpoint, layer_tensor, save_json


def mat_stats(w: np.ndarray, k: int = 8) -> dict:
    col = np.linalg.norm(w, axis=0)  # one column per input unit (neuron / head dim) -> what it writes
    s = np.linalg.svd(w, compute_uv=False)
    p = s**2 / (s**2).sum()
    return {
        "shape": list(w.shape),
        "fro": float(np.linalg.norm(w)),
        "col_norm_median": float(np.median(col)),
        "col_norm_max": float(col.max()),
        "col_norm_max_over_median": float(col.max() / np.median(col)),
        "top_sv": s[:k].tolist(),
        "sv1_energy_share": float(p[0]),
        "effective_rank": float(np.exp(-(p * np.log(p + 1e-30)).sum())),
    }


def main():
    ck = Checkpoint()
    names = ck.names()
    out = {"n_tensors": len(names), "n_params": int(sum(np.prod(ck.info(n)["shape"]) for n in names)), "layers": []}
    for i in range(N_LAYERS):
        row = {"layer": i, "type": "full_attention" if i in FULL_ATTN else "linear_attention"}
        for kind in ("down", "out"):
            row[kind] = mat_stats(ck.get(layer_tensor(i, kind)))
        for kind in ("in_norm", "post_norm"):
            g = ck.get(layer_tensor(i, kind))
            row[kind] = {"mean": float(g.mean()), "std": float(g.std()), "max_abs": float(np.abs(g).max())}
        out["layers"].append(row)
        print(f"layer {i:2d} {row['type'][:4]}  down fro {row['down']['fro']:.1f} sv1 {row['down']['top_sv'][0]:.2f}"
              f"  out fro {row['out']['fro']:.1f}  in_norm mean {row['in_norm']['mean']:+.3f}")
    fn = ck.get("model.language_model.norm.weight")
    out["final_norm"] = {"mean": float(fn.mean()), "std": float(fn.std())}
    emb = ck.get("model.language_model.embed_tokens.weight")
    rn = np.linalg.norm(emb, axis=1)
    out["embedding"] = {"shape": list(emb.shape), "row_norm_median": float(np.median(rn)), "row_norm_max": float(rn.max())}
    save_json(out, "01_survey.json")
    write_md(out)


def write_md(o):
    L = ["# Phase 1: weight survey", "",
         f"{o['n_tensors']} language-model tensors, {o['n_params']:,} parameters (text model only; tied embeddings).",
         f"Final norm weight mean {o['final_norm']['mean']:+.3f} (std {o['final_norm']['std']:.3f}); "
         "a mean near 0 means the RMSNorm scales by (1 + w).", "",
         "Output matrices (what each block writes into the residual stream). `max/med` = largest column norm over the "
         "median column norm; `sv1 share` = fraction of squared Frobenius norm in the top singular direction.", "",
         "| layer | type | down fro | down sv1 | down sv1 share | down eff. rank | down max/med col | out fro | out sv1 share | out max/med col |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in o["layers"]:
        d, a = r["down"], r["out"]
        L.append(f"| {r['layer']} | {r['type'].split('_')[0]} | {d['fro']:.1f} | {d['top_sv'][0]:.2f} | {d['sv1_energy_share']:.3f} | "
                 f"{d['effective_rank']:.0f} | {d['col_norm_max_over_median']:.1f} | {a['fro']:.1f} | {a['sv1_energy_share']:.3f} | "
                 f"{a['col_norm_max_over_median']:.1f} |")
    (RESULTS / "01_survey.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
