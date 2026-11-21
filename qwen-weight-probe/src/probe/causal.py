"""Phase 4: causal checks of the p50c "loop direction" on the GPU.

    nix-shell --run 'python -m probe.causal gen base'      # also: gen p50c
    nix-shell --run 'python -m probe.causal analyze'       # teacher-forced pass over base generations
    nix-shell --run 'python -m probe.causal gen base_anti' # also: p50c_pro, p50c_rand (need analyze first)
    nix-shell --run 'python -m probe.causal report'

Conditions (same 64 practice problems, same seeds, 4096-token cap, Sparks think35 sampling):
  base       unedited model
  p50c       Sparks' released edit
  base_anti  base model, residual stream after layers 15 and 19 pushed AWAY from the loop side of u (activation analog of the edit)
  p50c_pro   edited model pushed TOWARD the loop side of u (does it bring loops back?)
  p50c_rand  edited model pushed along a random unit direction with the same magnitude (control)
"""
import json
import sys
import time

import numpy as np
import torch

from probe.model import generate, grade, load_model, loop_stats, repeated_mask, sample_problems
from probe.weights import RESULTS, save_json

MAX_NEW = 4096
LAYERS = (15, 19)
OUT = RESULTS / "04"


def unit_dirs(device):
    return {l: torch.tensor(np.load(RESULTS / f"u{l}_p50c.npy"), dtype=torch.float32, device=device) for l in LAYERS}


def steering_hooks(model, vecs):
    """Add a fixed vector to the residual stream output of each decoder layer in `vecs` ({layer: tensor[1024]})."""
    hs = []
    for l, v in vecs.items():
        def hook(mod, inp, out, v=v):
            if isinstance(out, tuple):
                return (out[0] + v.to(out[0].dtype),) + tuple(out[1:])
            return out + v.to(out.dtype)
        hs.append(model.model.layers[l].register_forward_hook(hook))
    return hs


def parse_cond(cond):
    """'p50c_pro_x0.25' -> ('p50c_pro', 0.25); plain names are 1x (dose-response sweep naming)."""
    if "_x" in cond:
        base, s = cond.rsplit("_x", 1)
        return base, float(s)
    return cond, 1.0


def steer_vectors(cond, device):
    cond, scale = parse_cond(cond)
    a = json.loads((OUT / "analysis.json").read_text())
    u = unit_dirs(device)
    vecs = {}
    rng = np.random.default_rng(123)  # same random directions at every strength
    for l in LAYERS:
        s, beta = a["steer"][str(l)]["loop_sign"], scale * a["steer"][str(l)]["beta"]
        if cond == "base_anti":
            vecs[l] = -s * beta * u[l]
        elif cond == "p50c_pro":
            vecs[l] = s * beta * u[l]
        elif cond == "p50c_rand":
            r = rng.standard_normal(1024)
            vecs[l] = beta * torch.tensor(r / np.linalg.norm(r), dtype=torch.float32, device=device)
    return vecs


def cmd_gen(cond):
    edit = "p50c" if cond.startswith("p50c") else None
    model, tok = load_model(edit)
    hooks = steering_hooks(model, steer_vectors(cond, model.device)) if cond not in ("base", "p50c") else []
    probs = sample_problems(32, 32)
    t0 = time.time()
    res, tps = generate(model, tok, [r["problem"] for r in probs], MAX_NEW, seed=0, batch_size=32)
    for h in hooks:
        h.remove()
    rows = []
    for r, g in zip(probs, res):
        text = tok.decode(g["ids"])
        rows.append({"id": r["id"], "source": r["source"], "gold": r["answer"], "n_tokens": len(g["ids"]),
                     "finished": g["finished"], "closed_think": "</think>" in text, "correct": grade(text, r["answer"]),
                     **loop_stats(g["ids"]), "ids": g["ids"]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"gen_{cond}.json").write_text(json.dumps(rows))
    s = summarize(rows)
    print(f"{cond}: {s} | {tps:.0f} tok/s, {time.time() - t0:.0f}s")


def summarize(rows):
    n = len(rows)
    return {"n": n, "looping": sum(r["looping"] for r in rows) / n, "hit_cap": sum(not r["finished"] for r in rows) / n,
            "closed_think": sum(r["closed_think"] for r in rows) / n, "correct": sum(r["correct"] for r in rows) / n,
            "mean_rep_frac": float(np.mean([r["rep_frac"] for r in rows]))}


@torch.no_grad()
def cmd_analyze():
    """Teacher-forced pass over the base generations: what do the edited neurons write along u during loops?"""
    model, tok = load_model()
    from probe.model import build_prompts
    rows = json.loads((OUT / "gen_base.json").read_text())
    probs = {r["id"]: r for r in sample_problems(32, 32)}
    edit = json.loads((RESULTS / "02_edit.json").read_text())["p50c"]
    u = unit_dirs(model.device)
    W = {l: model.model.layers[l].mlp.down_proj.weight.float() for l in LAYERS}  # [1024, 3584]
    cols = {l: torch.tensor(edit[str(l)]["columns"], device=model.device) for l in LAYERS}
    wu = {l: u[l] @ W[l] for l in LAYERS}  # each neuron's write along u, per unit activation
    cap = {}
    hooks = []
    for l in LAYERS:
        def pre(mod, inp, l=l):
            cap[l] = inp[0][0].float()  # [T, 3584] (batch 1)
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(pre))

    per_tok = {l: {"edited": [], "all": []} for l in LAYERS}
    resid = {l: [] for l in LAYERS}
    labels, positions, gen_idx = [], [], []
    for gi, r in enumerate(rows):
        prompt = build_prompts(tok, [probs[r["id"]]["problem"]])[0]
        p_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        ids = torch.tensor([p_ids + r["ids"]], device=model.device)
        out = model.model(ids, output_hidden_states=True)  # body only: no 248k-vocab logits needed
        P, T = len(p_ids), len(r["ids"])
        for l in LAYERS:
            a = cap[l][P:]  # generated positions only
            per_tok[l]["edited"].append((a[:, cols[l]] @ wu[l][cols[l]]).cpu().numpy())
            per_tok[l]["all"].append((a @ wu[l]).cpu().numpy())
            resid[l].append(out.hidden_states[l + 1][0, P:].float().cpu().numpy())  # residual after layer l
        labels.append(repeated_mask(r["ids"]))
        positions.append(np.arange(T))
        gen_idx.append(np.full(T, gi))
    for h in hooks:
        h.remove()

    rep, pos, gidx = np.concatenate(labels), np.concatenate(positions), np.concatenate(gen_idx)
    looping_gen = np.array([r["looping"] for r in rows])[gidx]
    res = {"n_gens": len(rows), "n_tokens": int(len(rep)), "repeated_token_share": float(rep.mean()),
           "loop_gens": int(sum(r["looping"] for r in rows)), "layers": {}, "steer": {}}
    rng = np.random.default_rng(0)
    for l in LAYERS:
        e, al = np.concatenate(per_tok[l]["edited"]), np.concatenate(per_tok[l]["all"])
        R = np.concatenate(resid[l])
        late = pos >= 1024  # position-matched comparison (repeated tokens skew late)
        d = lambda x, m: float((x[m].mean() - x[~m].mean()) / np.sqrt(0.5 * (x[m].var() + x[~m].var())))
        # onset-aligned profile of the edited neurons' u-write
        prof = []
        for gi, r in enumerate(rows):
            o = r["loop_onset"]
            if o is not None and o >= 512:
                prof.append(per_tok[l]["edited"][gi][o - 512:o + 512].reshape(-1, 64).mean(1) if
                            len(per_tok[l]["edited"][gi]) >= o + 512 else None)
        prof = [p for p in prof if p is not None]
        # independent direction: difference of means of the residual, repeated vs not
        def dom(mask_sel):
            v = R[mask_sel & rep].mean(0) - R[mask_sel & ~rep].mean(0)
            return v / np.linalg.norm(v)
        all_m = np.ones_like(rep)
        half = np.isin(gidx, rng.permutation(len(rows))[: len(rows) // 2])
        d_all, d_late = dom(all_m), dom(late)
        d_h1, d_h2 = dom(half), dom(~half)
        un = np.load(RESULTS / f"u{l}_p50c.npy")
        res["layers"][str(l)] = {
            "edited_uwrite_mean_repeated": float(e[rep].mean()), "edited_uwrite_mean_nonrepeated": float(e[~rep].mean()),
            "edited_uwrite_cohens_d": d(e, rep), "edited_uwrite_cohens_d_pos_ge_1024": d(e[late], rep[late]),
            "edited_uwrite_cohens_d_within_looping_gens": d(e[looping_gen], rep[looping_gen]),
            "all_neurons_uwrite_cohens_d": d(al, rep),
            "edited_uwrite_mean_abs": float(np.abs(e).mean()),
            "resid_proj_u_mean_repeated": float((R[rep] @ un).mean()), "resid_proj_u_mean_nonrepeated": float((R[~rep] @ un).mean()),
            "dom_cos_u": float(d_all @ un), "dom_late_cos_u": float(d_late @ un), "dom_splithalf_cos": float(d_h1 @ d_h2),
            "dom_random_baseline": float(np.sqrt(2 / (np.pi * 1024))),
            "onset_profile_bins64": np.mean(prof, 0).tolist() if prof else None, "onset_profile_n": len(prof),
        }
        np.save(OUT / f"dom{l}.npy", d_all)
        # Steering: sign of u associated with loops in the residual; magnitude = the edit's typical per-token change
        # along u (|-6 c|, c = edited neurons' u-write), averaged over all generated tokens.
        loop_sign = float(np.sign((R[rep] @ un).mean() - (R[~rep] @ un).mean()))
        res["steer"][str(l)] = {"loop_sign": loop_sign, "beta": float(6 * np.abs(e).mean())}
    save_json(res, "04/analysis.json")
    for l in LAYERS:
        print(l, {k: v for k, v in res["layers"][str(l)].items() if k != "onset_profile_bins64"})
    print("steer", res["steer"])


def cmd_report():
    conds = [c for c in ("base", "p50c", "base_anti", "p50c_pro", "p50c_rand") if (OUT / f"gen_{c}.json").exists()]
    summ = {c: summarize(json.loads((OUT / f"gen_{c}.json").read_text())) for c in conds}
    save_json(summ, "04/summary.json")
    a = json.loads((OUT / "analysis.json").read_text()) if (OUT / "analysis.json").exists() else None
    L = ["# Phase 4: causal checks (64 practice problems: 32 MATH train + 32 GSM8K train, 4,096-token cap)", "",
         "Sampling as Sparks' think35 protocol (thinking on, temp 0.6, top-p 0.95, top-k 20, presence penalty 1.5), "
         "same seeds in every condition. A generation is **looping** if over half of its last 1,024 tokens close a "
         "16-gram that already occurred earlier in the same generation.", "",
         "| condition | looping | hit 4K cap | closed thinking | correct | mean repeated-token share |", "|---|---|---|---|---|---|"]
    for c, s in summ.items():
        L.append(f"| {c} | {s['looping']:.0%} | {s['hit_cap']:.0%} | {s['closed_think']:.0%} | {s['correct']:.0%} | {s['mean_rep_frac']:.3f} |")
    if a:
        L += ["", f"## Teacher-forced analysis of the base generations ({a['n_tokens']:,} tokens, {a['loop_gens']} looping generations, "
              f"{a['repeated_token_share']:.1%} of tokens repeated)", "",
              "`u-write` = what the 64 edited neurons write along u per token (activation × uᵀW column). Cohen's d compares "
              "repeated vs non-repeated tokens. `DoM` = difference of means of the residual stream after the layer "
              "(repeated − non-repeated), found without using u.", "",
              "| layer | u-write d (all) | d (positions ≥1024) | d (within looping gens) | all-neuron u-write d | cos(DoM, u) | cos(DoM late, u) | DoM split-half cos | random baseline |",
              "|---|---|---|---|---|---|---|---|---|"]
        for l, v in a["layers"].items():
            L.append(f"| {l} | {v['edited_uwrite_cohens_d']:+.2f} | {v['edited_uwrite_cohens_d_pos_ge_1024']:+.2f} | "
                     f"{v['edited_uwrite_cohens_d_within_looping_gens']:+.2f} | {v['all_neurons_uwrite_cohens_d']:+.2f} | "
                     f"{v['dom_cos_u']:+.3f} | {v['dom_late_cos_u']:+.3f} | {v['dom_splithalf_cos']:+.3f} | {v['dom_random_baseline']:.3f} |")
        L += ["", "Steering magnitude β per layer = 6 × mean |u-write| of the edited neurons (the edit's typical per-token change "
              "along u); sign = the side of u the residual sits on during repeated tokens.", "",
              "| layer | loop sign | β |", "|---|---|---|"]
        L += [f"| {l} | {v['loop_sign']:+.0f} | {v['beta']:.3f} |" for l, v in a["steer"].items()]
    (RESULTS / "04_causal.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"gen": lambda: cmd_gen(sys.argv[2]), "analyze": cmd_analyze, "report": cmd_report}[sys.argv[1]]()
