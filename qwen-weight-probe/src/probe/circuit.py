"""Level 4: the looping circuit (GPU shell). Plan: TODO.md level 4.

    nix-shell --run 'python -m probe.circuit attn'        # stage 1 (+ the Phase-4 reproduction gate)
    nix-shell --run 'python -m probe.circuit upstream'    # stage 2
    nix-shell --run 'python -m probe.circuit downstream'  # stage 3
    nix-shell --run 'python -m probe.circuit ablate'      # stage 4
    nix-shell --run 'python -m probe.circuit confirm <cond>'  # stage 5
    nix-shell --run 'python -m probe.circuit report'      # stage 6

Metrics are computed teacher-forced on the saved Phase 4 generations (results/04/gen_base.json):
  copy target: at a repeated token t (the 16-gram ending at t first ended at p), the loop "predicts" ids[p+1];
  copy log-prob = log p(ids[p+1] | <= t).
"""
import json
import os
import sys
import time

import numpy as np
import torch

from probe.model import build_prompts, load_model, sample_problems
from probe.weights import RESULTS, save_json

torch.set_num_threads(4)
OUT = RESULTS / "08_circuits" / "looping"
FULL_ATTN = (3, 7, 11, 15, 19, 23)
N_GRAM = 16
MAX_POS = 200  # repeated positions sampled per generation


# ---------------------------------------------------------------- data
def base_gens():
    rows = json.loads((RESULTS / "04" / "gen_base.json").read_text())
    probs = {r["id"]: r for r in sample_problems(32, 32)}
    loop = [i for i, r in enumerate(rows) if r["looping"]]
    pool = [i for i, r in enumerate(rows) if not r["looping"]]
    pair = {}
    for i in sorted(loop, key=lambda i: -rows[i]["n_tokens"]):  # length-matched non-looping partner per looping gen
        j = min(pool, key=lambda j: abs(rows[j]["n_tokens"] - rows[i]["n_tokens"]))
        pair[i] = j; pool.remove(j)
    matched = [pair[i] for i in loop]  # aligned with `loop`
    return rows, probs, loop, matched


def copy_targets(ids, n=N_GRAM):
    """[(t, p1)]: t closes a 16-gram whose first occurrence ended at p; p1 = p+1 is the loop's predicted next index."""
    first, out = {}, []
    for t in range(n - 1, len(ids) - 1):
        g = tuple(ids[t - n + 1:t + 1])
        if g in first:
            out.append((t, first[g] + 1))
        else:
            first[g] = t
    return out


def full_ids(tok, problem, gen_ids, device):
    p = tok(build_prompts(tok, [problem])[0], return_tensors="pt").input_ids[0].tolist()
    return torch.tensor([p + gen_ids], device=device), len(p)


def edited_cols(device):
    ed = json.loads((RESULTS / "02_edit.json").read_text())["p50c"]
    return {l: torch.tensor(ed[str(l)]["columns"], device=device) for l in (15, 19)}


def u_dirs(device):
    return {l: torch.tensor(np.load(RESULTS / f"u{l}_p50c.npy"), dtype=torch.float32, device=device) for l in (15, 19)}


def cohens_d(x, m):
    return float((x[m].mean() - x[~m].mean()) / np.sqrt(0.5 * (x[m].var() + x[~m].var())))


# ---------------------------------------------------------------- stage 1: attention heads that copy
@torch.no_grad()
def cmd_attn():
    model, tok = load_model()
    model.config._attn_implementation = "eager"  # only the eager path returns attention weights
    for l in FULL_ATTN:
        model.model.layers[l].self_attn.config._attn_implementation = "eager"
    dev = model.device
    rows, probs, loop, matched = base_gens()
    cols, u = edited_cols(dev), u_dirs(dev)
    wu = {l: u[l] @ model.model.layers[l].mlp.down_proj.weight.float() for l in (15, 19)}
    rng = np.random.default_rng(0)

    cap, attn_sel = {}, {}
    hooks = []
    for l in (15, 19):
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(
            lambda m, i, l=l: cap.__setitem__(l, i[0][0].float())))
    for l in FULL_ATTN:
        def ahook(mod, inp, out, l=l):
            w = out[1]
            if w is None or "q" not in attn_sel:
                return
            q, k, P = attn_sel["q"], attn_sel["k"], attn_sel["P"]
            a = w[0].float()                                  # (heads, T, T)
            tgt = a[:, q, k]                                  # attention to the copy target
            base = torch.stack([a[:, qi, P:qi].mean(-1) for qi in q.tolist()], 1)  # mean over earlier generated tokens
            attn_sel.setdefault("res", {})[l] = (tgt.cpu().numpy(), base.cpu().numpy())
        hooks.append(model.model.layers[l].self_attn.register_forward_hook(ahook))

    # synthetic positive control: 200 random tokens repeated twice
    g = torch.Generator().manual_seed(0)
    seq = torch.randint(1000, 50000, (200,), generator=g)
    ids = torch.cat([seq, seq])[None].to(dev)
    q = torch.arange(200, 399); attn_sel.update(q=q, k=q - 199, P=0)
    model.model(ids)
    synth = {l: (attn_sel["res"][l][0].mean(1) / attn_sel["res"][l][1].mean(1)).tolist() for l in FULL_ATTN}
    synth_abs = {l: attn_sel["res"][l][0].mean(1).tolist() for l in FULL_ATTN}
    attn_sel.clear()

    uw = {15: [], 19: []}; reps = []; per_head = {l: [[], []] for l in FULL_ATTN}; copy_lp = []
    t0 = time.time()
    for gi, r in enumerate(rows):
        ids, P = full_ids(tok, probs[r["id"]]["problem"], r["ids"], dev)
        tg = copy_targets(r["ids"]) if gi in loop else []
        if len(tg) > MAX_POS:
            tg = [tg[i] for i in sorted(rng.choice(len(tg), MAX_POS, replace=False))]
        if tg:
            attn_sel.update(q=torch.tensor([P + t for t, _ in tg]), k=torch.tensor([P + p for _, p in tg]), P=P)
        out = model.model(ids)
        if tg:
            for l in FULL_ATTN:
                per_head[l][0].append(attn_sel["res"][l][0]); per_head[l][1].append(attn_sel["res"][l][1])
            h = out.last_hidden_state[0, [P + t for t, _ in tg]]
            lp = torch.log_softmax(model.lm_head(h).float(), -1)
            copy_lp.append(lp[torch.arange(len(tg)), torch.tensor([r["ids"][p] for _, p in tg], device=dev)].cpu().numpy())
            attn_sel.clear()
        for l in (15, 19):
            a = cap[l][P:]
            uw[l].append((a[:, cols[l]] @ wu[l][cols[l]]).cpu().numpy())
        from probe.model import repeated_mask
        reps.append(repeated_mask(r["ids"]))
    for hk in hooks:
        hk.remove()
    rep = np.concatenate(reps)
    repro = {l: cohens_d(np.concatenate(uw[l]), rep) for l in (15, 19)}
    print(f"forward passes: {time.time() - t0:.0f}s; reproduction d: L15 {repro[15]:+.2f} (Phase 4: -1.40), L19 {repro[19]:+.2f} (+1.44)")
    heads = []
    for l in FULL_ATTN:
        tgt, base = np.concatenate(per_head[l][0], 1), np.concatenate(per_head[l][1], 1)
        for h in range(8):
            heads.append({"name": f"L{l}.attn.h{h}", "layer": l, "head": h,
                          "copy_attn": float(tgt[h].mean()), "ratio_vs_uniform": float(tgt[h].mean() / base[h].mean()),
                          "synthetic_ratio": synth[l][h], "synthetic_copy_attn": synth_abs[l][h]})
    heads.sort(key=lambda x: -x["copy_attn"])
    res = {"reproduction_d": repro, "repro_ok": abs(repro[15] + 1.40) < 0.05 and abs(repro[19] - 1.44) < 0.05,
           "n_positions": int(sum(len(x) for x in copy_lp)), "baseline_copy_logprob_mean": float(np.concatenate(copy_lp).mean()),
           "heads": heads}
    save_json(res, "08_circuits/looping/stage1_attn.json")
    np.save(OUT / "baseline_copy_logprob.npy", np.concatenate(copy_lp))
    print("repro_ok:", res["repro_ok"], "| positions:", res["n_positions"], "| mean copy log-prob:", round(res["baseline_copy_logprob_mean"], 3))
    print("top copy heads (on real loops):")
    for x in heads[:10]:
        print(f"  {x['name']}: copy attn {x['copy_attn']:.3f} ({x['ratio_vs_uniform']:.0f}x uniform); synthetic {x['synthetic_copy_attn']:.3f}")
    h7 = next(x for x in heads if x["name"] == "L15.attn.h7")
    print(f"L15.attn.h7 (Sparks p25c): rank {heads.index(h7) + 1}/48, copy attn {h7['copy_attn']:.3f}")


# ---------------------------------------------------------------- stage 2: what drives the edited neurons' u-write?
def out_proj_of(model, l):
    a = model.model.layers[l]
    return a.self_attn.o_proj if l in FULL_ATTN else a.linear_attn.out_proj


def cmd_upstream():
    """Direct-path attribution of the edited neurons' u-write (L15, L19, toward the loop side) to every earlier
    component. The residual stream x just before layer l's MLP is exactly embeddings + every earlier attention/MLP
    output, so component k's direct contribution at token t is g_t . out_k(t), with g_t = dc_t/dx_t. Only g needs
    a gradient, and it flows through one MLP; everything else is a no-grad forward. Indirect paths (k -> j -> target)
    are not counted here; stage 4's ablations measure those causally."""
    model, tok = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    dev = model.device
    rows, probs, loop, matched = base_gens()
    window = {}
    for li, mi in zip(loop, matched):
        o = rows[li]["loop_onset"]
        window[li] = (max(N_GRAM, o - 256), o + 256)
        window[mi] = (max(N_GRAM, o - 256), min(o + 256, rows[mi]["n_tokens"] - 1))
    cols, u = edited_cols(dev), u_dirs(dev)
    sign = {int(k): v["loop_sign"] for k, v in json.loads((RESULTS / "04" / "analysis.json").read_text())["steer"].items()}
    rng = np.random.default_rng(1)
    from probe.model import repeated_mask
    cap, sel_t = {}, {}

    def grab(key, take):
        def pre(mod, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1])  # (tokens, features) whether or not a batch dim is present
            cap[key] = x[sel_t["abs"]].float() if take else x.float()
        return pre
    hooks = []
    for l in range(20):
        hooks.append(out_proj_of(model, l).register_forward_pre_hook(grab(("h", l), True)))
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(grab(("n", l), True)))
    for l in (15, 19):  # residual right before the MLP = input of post_attention_layernorm
        hooks.append(model.model.layers[l].post_attention_layernorm.register_forward_pre_hook(grab(("x", l), True)))
    hooks.append(model.model.embed_tokens.register_forward_hook(
        lambda m, i, o: cap.__setitem__(("e", 0), o.reshape(-1, o.shape[-1])[sel_t["abs"]].float())))

    acc = {grp: {tgt: {} for tgt in (15, 19)} for grp in ("loop", "ctrl")}
    check = []
    t0 = time.time()
    for grp, idxs in (("loop", loop), ("ctrl", matched)):
        for gi in idxs:
            r = rows[gi]
            rep = repeated_mask(r["ids"])
            lo, hi = window[gi]
            pool = np.flatnonzero(rep if grp == "loop" else ~rep)
            pool = pool[(pool >= lo) & (pool <= hi)]
            sel = np.sort(rng.choice(pool, min(MAX_POS, len(pool)), replace=False))
            ids, P = full_ids(tok, probs[r["id"]]["problem"], r["ids"][:hi + 1], dev)
            sel_t["abs"] = torch.tensor(P + sel, device=dev)
            with torch.no_grad():
                model.model(ids)
            for tgt in (15, 19):
                layer = model.model.layers[tgt]
                x = cap[("x", tgt)].clone().requires_grad_(True)
                with torch.enable_grad():
                    # RMSNorm by hand (scale = 1 + w): calling the module would re-fire our capture hook on it
                    ln = layer.post_attention_layernorm
                    # freeze each token's RMS scale (standard in direct attribution): with it live, c is invariant to
                    # rescaling x, so g.x == 0 and the contributions would sum to zero by construction
                    rs = torch.rsqrt(x.detach().pow(2).mean(-1, keepdim=True) + ln.eps)
                    h = x * rs * (1.0 + ln.weight.float())
                    mlp = layer.mlp
                    a = mlp.act_fn(h @ mlp.gate_proj.weight.float().T) * (h @ mlp.up_proj.weight.float().T)
                    wu = u[tgt] @ mlp.down_proj.weight.float()
                    c = sign[tgt] * (a[:, cols[tgt]] @ wu[cols[tgt]]).sum() / len(sel)
                    G, = torch.autograd.grad(c, x)                   # (n_sel, 1024)
                A = acc[grp][tgt]
                parts = {"emb": float((G * cap[("e", 0)]).sum())}
                for l in range(tgt + 1):
                    hs = cap[("h", l)]; Wo = out_proj_of(model, l).weight.float()
                    hd = 256 if l in FULL_ATTN else 128
                    contrib = ((G @ Wo).view(len(sel), -1, hd) * hs.view(len(sel), -1, hd)).sum(-1).sum(0)  # per head
                    A[f"h{l}"] = A.get(f"h{l}", 0) + contrib.cpu().numpy()
                    parts[f"h{l}"] = float(contrib.sum())
                    if l < tgt:
                        an = cap[("n", l)]; Wd = model.model.layers[l].mlp.down_proj.weight.float()
                        cn = (an * (G @ Wd)).sum(0)                  # per neuron
                        A[f"n{l}"] = A.get(f"n{l}", 0) + cn.cpu().numpy()
                        parts[f"n{l}"] = float(cn.sum())
                A["emb"] = A.get("emb", 0.0) + parts["emb"]
                total = float((G * x.detach()).sum())                # exact: g . (emb + sum of outputs)
                check.append(abs(sum(parts.values()) - total) / (sum(abs(v) for v in parts.values()) + 1e-9))
            cap.clear()
            print(f"  {grp} gen {gi}: {time.time() - t0:.0f}s", flush=True)
    for hk in hooks:
        hk.remove()
    res = {"method": "direct-path attribution (g . component output), loop side positive",
           "completeness_rel_err_max": float(max(check)), "completeness_rel_err_median": float(np.median(check)), "targets": {}}
    for tgt in (15, 19):
        L, C = acc["loop"][tgt], acc["ctrl"][tgt]
        items = []
        for key in L:
            if key == "emb":
                continue
            kind, l = key[0], int(key[1:])
            diff = L[key] / len(loop) - C[key] / len(matched)
            for i, v in enumerate(diff):
                name = f"L{l}.mlp.{i}" if kind == "n" else (f"L{l}.attn.h{i}" if l in FULL_ATTN else f"L{l}.lin.h{i}")
                items.append((float(v), name, float(L[key][i] / len(loop)), float(C[key][i] / len(matched))))
        items.sort(key=lambda x: -abs(x[0]))
        absd = sum(abs(x[0]) for x in items)
        by_layer = {}
        for d, n, _, _ in items:
            k = n.split(".")[0] + (".mlp" if ".mlp." in n else ".attn")
            by_layer[k] = by_layer.get(k, 0) + d
        res["targets"][str(tgt)] = {
            "loop_total": sum(float(np.sum(v)) for k, v in L.items()) / len(loop),
            "ctrl_total": sum(float(np.sum(v)) for k, v in C.items()) / len(matched),
            "emb_loop": L["emb"] / len(loop), "emb_ctrl": C["emb"] / len(matched),
            "top": [{"name": n, "diff": d, "loop": a, "ctrl": b} for d, n, a, b in items[:60]],
            "share_top20_of_abs_diff": float(sum(abs(x[0]) for x in items[:20]) / absd),
            "by_layer_block": dict(sorted(by_layer.items(), key=lambda kv: -abs(kv[1]))[:12])}
    save_json(res, "08_circuits/looping/stage2_upstream.json")
    print(f"done in {time.time() - t0:.0f}s; completeness rel. error max {res['completeness_rel_err_max']:.1e}")
    for tgt in (15, 19):
        t = res["targets"][str(tgt)]
        print(f"L{tgt} edited u-write (loop side +): loop {t['loop_total']:+.3f} vs ctrl {t['ctrl_total']:+.3f}; "
              f"top-20 components carry {t['share_top20_of_abs_diff']:.0%} of |loop-ctrl|")
        print("   by block:", {k: round(v, 3) for k, v in list(t["by_layer_block"].items())[:8]})
        for x in t["top"][:12]:
            print(f"   {x['name']:>14}  loop-ctrl {x['diff']:+.4f}  (loop {x['loop']:+.4f}, ctrl {x['ctrl']:+.4f})")

# ---------------------------------------------------------------- stage 3: how does the push along u reach the prediction?
def attn_mod(model, l):
    a = model.model.layers[l]
    return a.self_attn if l in FULL_ATTN else a.linear_attn


@torch.no_grad()
def cmd_downstream(n_gens=8):
    """Edited model + push toward the loop side of u (1x beta, the p50c_pro condition). Measure the copy log-prob gain
    from the push, then re-run the push with one downstream component frozen at its un-pushed value: the share of the
    gain that disappears is what that component carries. 'all downstream frozen' leaves only the direct path to the
    output (push -> final norm -> unembedding)."""
    from probe.causal import steer_vectors, steering_hooks
    model, tok = load_model("p50c")
    dev = model.device
    rows = json.loads((RESULTS / "04" / "gen_p50c_pro.json").read_text())
    probs = {r["id"]: r for r in sample_problems(32, 32)}
    use = [i for i, r in enumerate(rows) if r["looping"]][:n_gens]
    push = steer_vectors("p50c_pro", dev)
    uh = json.loads((RESULTS / "06_wiring" / "uhood.json").read_text())
    readers = []
    for k in ("u15", "u19"):
        for r in uh[k]["top_readers"]:
            n = r["name"]
            if ".mlp." in n and int(n.split(".")[0][1:]) >= 16 and n not in readers:
                readers.append(n)
    readers = readers[:16]
    rng = np.random.default_rng(7)
    rand = [f"L{l}.mlp.{j}" for l, j in zip(rng.integers(16, 24, 16), rng.integers(0, 3584, 16))]
    blocks = [f"L{l}.{b}" for l in range(16, 24) for b in ("attn", "mlp")]
    conds = ["all_downstream"] + blocks + readers + rand
    rec = {}

    def record_hooks():
        hs = []
        for l in range(16, 24):
            hs.append(attn_mod(model, l).register_forward_hook(lambda m, i, o, l=l: rec.__setitem__(f"L{l}.attn", (o[0] if isinstance(o, tuple) else o).clone())))
            hs.append(model.model.layers[l].mlp.register_forward_hook(lambda m, i, o, l=l: rec.__setitem__(f"L{l}.mlp", o.clone())))
            hs.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, i, l=l: rec.__setitem__(f"L{l}.down_in", i[0].clone())))
        return hs

    def freeze_hooks(cond):
        hs = []
        names = blocks if cond == "all_downstream" else [cond]
        for name in names:
            l = int(name.split(".")[0][1:])
            if name.endswith(".attn"):
                def fa(m, i, o, key=name):
                    return (rec[key],) + tuple(o[1:]) if isinstance(o, tuple) else rec[key]
                hs.append(attn_mod(model, l).register_forward_hook(fa))
            elif name.endswith(".mlp"):
                hs.append(model.model.layers[l].mlp.register_forward_hook(lambda m, i, o, key=name: rec[key]))
            else:  # single neuron: hold its down_proj input at the un-pushed value
                j = int(name.split(".")[-1])
                def fn(m, i, l=l, j=j):
                    x = i[0].clone(); x[..., j] = rec[f"L{l}.down_in"][..., j]; return (x,)
                hs.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(fn))
        return hs

    think_end = tok.convert_tokens_to_ids("</think>")

    def copy_lp(ids, pos, tgt):
        """Primary metric (revised after the first stage-3 run): mean log P(</think>) at sampled positions.
        The push toward the loop side LOWERS the copy log-prob, so u is not a copy signal; it controls concluding."""
        h = model.model(ids).last_hidden_state[0, pos]
        return torch.log_softmax(model.lm_head(h).float(), -1)[:, think_end].mean().item()

    acc = {c: [] for c in ["clean", "pushed"] + conds}
    t0 = time.time()
    for gi in use:
        r = rows[gi]
        ids, P = full_ids(tok, probs[r["id"]]["problem"], r["ids"], dev)
        sel = np.sort(rng.choice(len(r["ids"]), min(300, len(r["ids"])), replace=False))
        pos = torch.tensor(P + sel, device=dev)
        tgt = None
        hs = record_hooks()
        acc["clean"].append(copy_lp(ids, pos, tgt))           # edited model, no push: records un-pushed values
        for h in hs:
            h.remove()
        sh = steering_hooks(model, push)
        acc["pushed"].append(copy_lp(ids, pos, tgt))
        for c in conds:
            fh = freeze_hooks(c)
            acc[c].append(copy_lp(ids, pos, tgt))
            for h in fh:
                h.remove()
        for h in sh:
            h.remove()
        rec.clear()
        print(f"  gen {gi}: clean {acc['clean'][-1]:.3f} pushed {acc['pushed'][-1]:.3f} ({time.time() - t0:.0f}s)", flush=True)
    clean, pushed = np.array(acc["clean"]), np.array(acc["pushed"])
    gain = pushed - clean
    res = {"n_gens": len(use), "clean_copy_lp": float(clean.mean()), "pushed_copy_lp": float(pushed.mean()),
           "gain": float(gain.mean()), "components": {}}
    for c in conds:
        g = np.array(acc[c]) - clean
        res["components"][c] = {"gain_left": float(g.mean()), "share_carried": float(1 - g.mean() / gain.mean()),
                                "group": "block" if c in blocks else ("reader" if c in readers else ("random" if c in rand else "all"))}
    res["metric"] = "mean log P(</think>) at 300 uniformly sampled positions per generation"
    save_json(res, "08_circuits/looping/stage3_downstream.json")
    print(f"log P(</think>): clean {res['clean_copy_lp']:.3f} -> pushed {res['pushed_copy_lp']:.3f} (gain {res['gain']:+.3f})")
    print(f"all downstream frozen: share carried by everything after the push = {res['components']['all_downstream']['share_carried']:.0%} "
          f"(direct path = the rest)")
    for grp in ("block", "reader", "random"):
        xs = sorted(((v["share_carried"], k) for k, v in res["components"].items() if v["group"] == grp), reverse=True)
        print(f"  {grp}: " + ", ".join(f"{k} {s:+.0%}" for s, k in xs[:8]))


# ---------------------------------------------------------------- stage 4: ablation map
def parse_comp(name):
    """'L15.attn.h7' / 'L3.lin.h2' -> ('head', l, slice); 'L17.mlp.20' -> ('neuron', l, [j]); 'L23.mlp' -> ('block_mlp', l);
    'L21.attn' -> ('block_attn', l); 'L15.edited' -> ('neuron', 15, edited cols)."""
    p = name.split(".")
    l = int(p[0][1:])
    if len(p) == 2:
        return ("block_" + p[1], l, None)
    if p[1] == "mlp":
        return ("neuron", l, [int(p[2])])
    hd = 256 if l in FULL_ATTN else 128
    h = int(p[2][1:])
    return ("head", l, slice(h * hd, (h + 1) * hd))


@torch.no_grad()
def cmd_ablate():
    model, tok = load_model()
    dev = model.device
    rows, probs, loop, matched = base_gens()
    cols, u = edited_cols(dev), u_dirs(dev)
    sign = {int(k): v["loop_sign"] for k, v in json.loads((RESULTS / "04" / "analysis.json").read_text())["steer"].items()}
    wu = {l: u[l] @ model.model.layers[l].mlp.down_proj.weight.float() for l in (15, 19)}
    think_end = tok.convert_tokens_to_ids("</think>")
    rng = np.random.default_rng(4)
    from probe.model import repeated_mask

    # ---- component list
    s1 = json.loads((OUT / "stage1_attn.json").read_text())["heads"]
    s2 = json.loads((OUT / "stage2_upstream.json").read_text())["targets"]
    s3 = json.loads((OUT / "stage3_downstream.json").read_text())["components"]
    uh = json.loads((RESULTS / "06_wiring" / "uhood.json").read_text())
    comps = {}
    for x in s1[:12]: comps[x["name"]] = "copy head"
    for t in ("15", "19"):
        for x in s2[t]["top"][:15]: comps.setdefault(x["name"], f"upstream L{t}")
    for k, v in s3.items():
        if v["group"] == "block" and v["share_carried"] >= 0.10: comps.setdefault(k, "downstream block")
    for k in ("u15", "u19"):
        for x in uh[k]["top_writers"][:5] + uh[k]["top_readers"][:5]:
            comps.setdefault(x["name"], "static u-neighbor")
    comps["L15.edited"] = "edited group"; comps["L19.edited"] = "edited group"
    ctrl = {}
    for _ in range(10):
        l = int(rng.integers(0, 24)); nh = 8 if l in FULL_ATTN else 16
        ctrl[f"L{l}.{'attn' if l in FULL_ATTN else 'lin'}.h{int(rng.integers(0, nh))}"] = "random head"
    for _ in range(10):
        ctrl[f"L{int(rng.integers(0, 24))}.mlp.{int(rng.integers(0, 3584))}"] = "random neuron"
    for i, l in enumerate((15, 19, 15)):
        ctrl[f"L{l}.randgroup{i}"] = "random 64-group"
    for b in ("L18.mlp", "L18.attn", "L20.mlp"):
        if b not in comps: ctrl[b] = "other block"
    groups = {**{f"L{l}.edited": (l, cols[l]) for l in (15, 19)},
              **{f"L{l}.randgroup{i}": (l, torch.tensor(np.sort(rng.choice(np.setdiff1d(np.arange(3584), cols[l].cpu().numpy()), 64, replace=False)), device=dev))
                 for i, l in enumerate((15, 19, 15))}}

    # ---- fixed positions per generation
    gens = []
    for gi in loop + matched:
        r = rows[gi]; rep = repeated_mask(r["ids"])
        ids, P = full_ids(tok, probs[r["id"]]["problem"], r["ids"], dev)
        tg = copy_targets(r["ids"]) if gi in loop else []
        if len(tg) > MAX_POS: tg = [tg[i] for i in sorted(rng.choice(len(tg), MAX_POS, replace=False))]
        uni = np.sort(rng.choice(len(r["ids"]) - 1, min(300, len(r["ids"]) - 1), replace=False))
        nonrep = np.array([t for t in uni if not rep[t + 1]]) if gi in matched else np.array([], int)
        gens.append({"loop": gi in loop, "ids": ids, "P": P, "copy_pos": [P + t for t, _ in tg],
                     "copy_tgt": [r["ids"][p] for _, p in tg], "uni": (P + uni).tolist(),
                     "nll_pos": (P + nonrep).tolist(), "nll_tgt": [r["ids"][t + 1] for t in nonrep],
                     "rep_pos": (P + np.flatnonzero(rep)).tolist() if gi in loop else []})

    # ---- mean activations (for mean-ablation), from the matched generations
    means, cnt = {}, 0
    def mcap(key):
        def pre(m, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float().mean(0); means[key] = means.get(key, 0) + x
        return pre
    def mcap_out(key):
        def fwd(m, i, o):
            x = (o[0] if isinstance(o, tuple) else o); x = x.reshape(-1, x.shape[-1]).float().mean(0); means[key] = means.get(key, 0) + x
        return fwd
    hs = []
    for l in range(24):
        hs += [out_proj_of(model, l).register_forward_pre_hook(mcap(("o", l))),
               model.model.layers[l].mlp.down_proj.register_forward_pre_hook(mcap(("d", l))),
               attn_mod(model, l).register_forward_hook(mcap_out(("A", l))),
               model.model.layers[l].mlp.register_forward_hook(mcap_out(("M", l)))]
    for g in gens:
        if not g["loop"]:
            model.model(g["ids"]); cnt += 1
    for h in hs: h.remove()
    means = {k: v / cnt for k, v in means.items()}

    # ---- metrics for one condition
    cap = {}
    capt = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, i, l=l: cap.__setitem__(l, i[0][0])) for l in (15, 19)]

    def logp_at(hid, pos, tgt):
        out = []
        for s in range(0, len(pos), 128):
            lp = torch.log_softmax(model.lm_head(hid[pos[s:s + 128]]).float(), -1)
            out.append(lp[torch.arange(len(pos[s:s + 128])), tgt[s:s + 128]])
        return torch.cat(out) if out else torch.zeros(0, device=dev)

    def run():
        m = {"think": [], "copy": [], "u15": [], "u19": [], "nll": []}
        for g in gens:
            hid = model.model(g["ids"]).last_hidden_state[0]
            m["think"].append(logp_at(hid, g["uni"], [think_end] * len(g["uni"])).mean().item())
            if g["loop"]:
                m["copy"].append(logp_at(hid, g["copy_pos"], g["copy_tgt"]).mean().item())
                for l in (15, 19):
                    a = cap[l][g["rep_pos"]].float()
                    m[f"u{l}"].append((sign[l] * (a[:, cols[l]] @ wu[l][cols[l]])).mean().item())
            elif g["nll_pos"]:
                m["nll"].append(-logp_at(hid, g["nll_pos"], g["nll_tgt"]).mean().item())
        return {k: float(np.mean(v)) for k, v in m.items()}

    def ablation_hooks(name, mode):
        hs = []
        if name in groups:
            l, cc = groups[name]
            def fn(m, i, l=l, cc=cc):
                x = i[0].clone(); x[..., cc] = 0 if mode == "zero" else means[("d", l)][cc].to(x.dtype); return (x,)
            return [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(fn)]
        kind, l, idx = parse_comp(name)
        if kind == "head":
            def fh(m, i, l=l, idx=idx):
                x = i[0].clone(); x[..., idx] = 0 if mode == "zero" else means[("o", l)][idx].to(x.dtype); return (x,)
            hs.append(out_proj_of(model, l).register_forward_pre_hook(fh))
        elif kind == "neuron":
            def fn(m, i, l=l, idx=idx):
                x = i[0].clone(); x[..., idx] = 0 if mode == "zero" else means[("d", l)][idx].to(x.dtype); return (x,)
            hs.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(fn))
        elif kind == "block_attn":
            def fa(m, i, o, l=l):
                x = o[0] if isinstance(o, tuple) else o
                y = torch.zeros_like(x) if mode == "zero" else means[("A", l)].to(x.dtype).expand_as(x)
                return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
            hs.append(attn_mod(model, l).register_forward_hook(fa))
        else:
            def fm(m, i, o, l=l):
                return torch.zeros_like(o) if mode == "zero" else means[("M", l)].to(o.dtype).expand_as(o)
            hs.append(model.model.layers[l].mlp.register_forward_hook(fm))
        return hs

    t0 = time.time()
    base = run()
    print(f"baseline: {base}  ({time.time() - t0:.0f}s per condition)", flush=True)
    res = {"baseline": base, "conditions": []}
    allc = {**comps, **ctrl}
    for i, (name, why) in enumerate(allc.items()):
        for mode in ("zero", "mean"):
            hs = ablation_hooks(name, mode)
            m = run()
            for h in hs: h.remove()
            res["conditions"].append({"name": name, "why": why, "mode": mode, **{f"d_{k}": m[k] - base[k] for k in base}})
        last = res["conditions"][-2]
        print(f"[{i + 1}/{len(allc)}] {name:>16} ({why}): zero d_think {last['d_think']:+.3f} d_copy {last['d_copy']:+.3f} "
              f"d_u15 {last['d_u15']:+.3f} d_u19 {last['d_u19']:+.3f} d_nll {last['d_nll']:+.3f}  {time.time() - t0:.0f}s", flush=True)
    for h in capt: h.remove()
    save_json(res, "08_circuits/looping/stage4_ablate.json")


# ---------------------------------------------------------------- stage 5: does it hold in real generation?
CONFIRM = {  # name: (edit, push condition or None, [(component, mode)])
    "base_zero_L15h7": ("p25c", None, []),                          # Sparks' own p25c edit (zeroes L15.attn.h7)
    "base_mean_L11h1": (None, None, [("L11.attn.h1", "mean")]),
    "base_copyheads": (None, None, [("L15.attn.h7", "mean"), ("L11.attn.h1", "mean"), ("L23.attn.h5", "mean")]),
    "base_randheads": (None, None, "random3"),                      # 3 random non-copy heads, mean-ablated
    "p50c_pro_copyheads": ("p50c", "p50c_pro", [("L15.attn.h7", "mean"), ("L11.attn.h1", "mean"), ("L23.attn.h5", "mean")]),
    "base_mean_L15h7": (None, None, [("L15.attn.h7", "mean")]),     # zero vs mean ablation of the same head
    "base_none": (None, None, []),                                  # references for seeds != 0 (seed 0: results/04)
    "p50c_pro_none": ("p50c", "p50c_pro", []),
}


@torch.no_grad()
def cmd_confirm(cond, seed=0):
    from probe.causal import steer_vectors, steering_hooks, summarize
    from probe.model import generate, grade, loop_stats
    edit, push, abl = CONFIRM[cond]
    model, tok = load_model(edit)
    dev = model.device
    if abl == "random3":
        s1 = {x["name"] for x in json.loads((OUT / "stage1_attn.json").read_text())["heads"][:12]}
        rng = np.random.default_rng(11)
        pool = [f"L{l}.attn.h{h}" for l in FULL_ATTN for h in range(8) if f"L{l}.attn.h{h}" not in s1]
        abl = [(n, "mean") for n in rng.choice(pool, 3, replace=False)]
    # mean head outputs (o_proj input) from the base model's matched non-looping generations
    rows, probs, loop, matched = base_gens()
    layers = sorted({int(n.split(".")[0][1:]) for n, _ in abl})
    means, cnt = {}, 0
    hs = [out_proj_of(model, l).register_forward_pre_hook(
        lambda m, i, l=l: means.__setitem__(l, means.get(l, 0) + i[0].reshape(-1, i[0].shape[-1]).float().mean(0))) for l in layers]
    for gi in matched[:6]:
        ids, _ = full_ids(tok, probs[rows[gi]["id"]]["problem"], rows[gi]["ids"], dev); model.model(ids); cnt += 1
    for h in hs: h.remove()
    hooks = []
    for name, mode in abl:
        _, l, sl = parse_comp(name)
        mv = means[l] / cnt
        def fh(m, i, sl=sl, mv=mv, mode=mode):
            x = i[0].clone(); x[..., sl] = 0 if mode == "zero" else mv[sl].to(x.dtype); return (x,)
        hooks.append(out_proj_of(model, l).register_forward_pre_hook(fh))
    if push:
        hooks += steering_hooks(model, steer_vectors(push, dev))
    pr = sample_problems(32, 32)
    t0 = time.time()
    res, tps = generate(model, tok, [r["problem"] for r in pr], 4096, seed=1000 * seed, batch_size=32)
    for h in hooks: h.remove()
    out = []
    for r, g in zip(pr, res):
        text = tok.decode(g["ids"])
        out.append({"id": r["id"], "n_tokens": len(g["ids"]), "finished": g["finished"], "closed_think": "</think>" in text,
                    "correct": grade(text, r["answer"]), **loop_stats(g["ids"]), "ids": g["ids"]})
    suffix = f"_s{seed}" if seed else ""  # seed 0 keeps the original file names
    (OUT / f"gen_{cond}{suffix}.json").write_text(json.dumps({"ablated": abl, "edit": edit, "push": push, "seed": seed, "rows": out}))
    print(f"{cond} seed {seed}: ablated {abl}; {summarize(out)} | {tps:.0f} tok/s, {time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def cmd_copysupp(heads=("L15.attn.h7", "L11.attn.h1", "L23.attn.h5"), controls=("L15.attn.h0", "L19.attn.h3", "L7.attn.h2")):
    """Why does ablating a copy head *raise* the copy log-prob (stage 4)? Test the copy-suppression reading: the head
    attends to the earlier copy of the next token and writes *against* that token's logit. Per head, at the copy
    positions of the base model's looping generations:
      - direct effect of the head's output on the copy target's log-prob (final RMS scale frozen):
        d.(W[tgt] - p.W), with d = head output * (1 + g_final) / rms;
      - correlation of that direct effect with the head's attention to the copy source;
      - mean-ablation: actual change in copy log-prob vs the change predicted by the direct path alone."""
    model, tok = load_model()
    model.config._attn_implementation = "eager"
    for l in FULL_ATTN:
        model.model.layers[l].self_attn.config._attn_implementation = "eager"
    dev = model.device
    rows, probs, loop, matched = base_gens()
    rng = np.random.default_rng(0)
    E = model.lm_head.weight.float(); g = 1 + model.model.norm.weight.float()
    allh = list(heads) + list(controls)
    spec = {n: parse_comp(n) for n in allh}
    layers = sorted({l for _, l, _ in spec.values()})
    zin, attw, fin = {}, {}, {}
    hs = [out_proj_of(model, l).register_forward_pre_hook(lambda m, i, l=l: zin.__setitem__(l, i[0][0])) for l in layers]
    sel = {}  # query positions to keep attention rows for (full T x T weights would not fit next to each other)
    hs += [model.model.layers[l].self_attn.register_forward_hook(
        lambda m, i, o, l=l: attw.__setitem__(l, o[1][0][:, sel["q"], :].float()) if "q" in sel else None) for l in layers]
    hs.append(model.model.norm.register_forward_pre_hook(lambda m, i: fin.__setitem__("x", i[0][0])))
    # mean o_proj input over the matched non-looping generations (as in stage 4)
    msum, cnt = {l: 0 for l in layers}, 0
    for gi in matched:
        r = rows[gi]; ids, P = full_ids(tok, probs[r["id"]]["problem"], r["ids"], dev)
        model.model(ids); cnt += 1
        for l in layers:
            msum[l] = msum[l] + zin[l].float().mean(0)
    means = {l: msum[l] / cnt for l in layers}
    acc = {n: {"dla": [], "attn": [], "pred_abl": [], "act_abl": []} for n in allh}
    base_lp = []
    t0 = time.time()
    for gi in loop:
        r = rows[gi]
        tg = copy_targets(r["ids"])
        if not tg:
            continue
        if len(tg) > MAX_POS:
            tg = [tg[i] for i in sorted(rng.choice(len(tg), MAX_POS, replace=False))]
        ids, P = full_ids(tok, probs[r["id"]]["problem"], r["ids"], dev)
        q = torch.tensor([P + t for t, _ in tg], device=dev); k = torch.tensor([P + p for _, p in tg], device=dev)
        tgt = torch.tensor([r["ids"][p] for _, p in tg], device=dev)
        sel["q"] = q
        hid = model.model(ids).last_hidden_state[0, q]
        sel.clear()
        lp_all = torch.log_softmax(model.lm_head(hid).float(), -1)
        lp0 = lp_all[torch.arange(len(q)), tgt]; base_lp.append(lp0.mean().item())
        pW = lp_all.exp() @ E                                     # (n, 1024): expected unembedding row
        eps = getattr(model.model.norm, "eps", getattr(model.model.norm, "variance_epsilon", 1e-6))
        x = fin["x"][q].float(); rms = torch.sqrt((x * x).mean(-1, keepdim=True) + eps)
        grad = g * (E[tgt] - pW) / rms                            # d logP(tgt) / d residual, final norm scale frozen
        snap = {l: (zin[l][q].float().clone(), attw[l]) for l in layers}
        for n in allh:
            _, l, sl = spec[n]; h = sl.start // 256
            Wo = out_proj_of(model, l).weight.float()[:, sl]
            z, aw = snap[l][0][:, sl], snap[l][1][h]
            acc[n]["dla"].append(((z @ Wo.T) * grad).sum(-1).cpu().numpy())
            acc[n]["attn"].append(aw[torch.arange(len(q)), k].cpu().numpy())
            acc[n]["pred_abl"].append((((means[l][sl] - z) @ Wo.T) * grad).sum(-1).mean().item())
        for n in allh:
            _, l, sl = spec[n]
            def fh(m, i, sl=sl, l=l):
                y = i[0].clone(); y[..., sl] = means[l][sl].to(y.dtype); return (y,)
            hh = out_proj_of(model, l).register_forward_pre_hook(fh)
            h2 = model.model(ids).last_hidden_state[0, q]
            lp1 = torch.log_softmax(model.lm_head(h2).float(), -1)[torch.arange(len(q)), tgt]
            hh.remove()
            acc[n]["act_abl"].append((lp1 - lp0).mean().item())
        print(f"  gen {gi}: {len(q)} positions ({time.time() - t0:.0f}s)", flush=True)
    for h in hs:
        h.remove()
    out = {"n_gens": len(base_lp), "baseline_copy_lp": float(np.mean(base_lp)), "heads": {}}
    for n in allh:
        d, at = np.concatenate(acc[n]["dla"]), np.concatenate(acc[n]["attn"])
        hi = at > np.percentile(at, 75); lo = at < np.percentile(at, 25)
        out["heads"][n] = {"copy_head": n in heads, "direct_effect_mean": float(d.mean()), "frac_negative": float((d < 0).mean()),
                           "attn_to_copy_mean": float(at.mean()), "corr_direct_vs_attn": float(np.corrcoef(d, at)[0, 1]),
                           "direct_effect_top_quartile_attn": float(d[hi].mean()), "direct_effect_bottom_quartile_attn": float(d[lo].mean()),
                           "mean_abl_actual": float(np.mean(acc[n]["act_abl"])), "mean_abl_direct_predicted": float(np.mean(acc[n]["pred_abl"]))}
    save_json(out, "08_circuits/looping/stage6_copysupp.json")
    print(f"baseline copy log-prob {out['baseline_copy_lp']:.3f} over {out['n_gens']} looping gens")
    print(f"{'head':14s} {'direct':>8s} {'%neg':>5s} {'attn':>6s} {'corr':>6s} {'dir hiA':>8s} {'dir loA':>8s} {'abl act':>8s} {'abl pred':>8s}")
    for n, v in out["heads"].items():
        print(f"{n:14s} {v['direct_effect_mean']:+8.3f} {v['frac_negative']:5.2f} {v['attn_to_copy_mean']:6.3f} {v['corr_direct_vs_attn']:+6.2f} "
              f"{v['direct_effect_top_quartile_attn']:+8.3f} {v['direct_effect_bottom_quartile_attn']:+8.3f} {v['mean_abl_actual']:+8.3f} {v['mean_abl_direct_predicted']:+8.3f}")


if __name__ == "__main__":
    {"attn": cmd_attn, "upstream": cmd_upstream, "downstream": cmd_downstream, "ablate": cmd_ablate, "copysupp": cmd_copysupp,
     "confirm": lambda: cmd_confirm(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)}[sys.argv[1]]()
