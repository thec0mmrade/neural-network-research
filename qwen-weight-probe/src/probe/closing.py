"""Level 4, second behavior: the </think> (concluding) circuit. Plan: TODO.md level 4.

    nix-shell --run 'python -m probe.closing pilot'        # stage 0: subject model (p50c), set A[:64], 8K cap
    nix-shell --run 'python -m probe.closing corpus'       # stage 1: set A[64:128]
    nix-shell --run 'python -m probe.closing direction'    # stage 2: d_close per layer, d_perp (u removed) at L15/L19
    nix-shell --run 'python -m probe.closing attribution'  # stage 3: direct-path attribution of log P(</think>)
    nix-shell --run 'python -m probe.closing screen'       # stage 4: teacher-forced ablation screen
    nix-shell --run 'python -m probe.closing confirm <cond> <seed>'  # stage 5: free generation on set B
    nix-shell --run 'python -m probe.closing pool'         # pool stage-5 runs: Fisher exact vs 'none'

confirm conditions: none | anti_u | dperp_plus | dperp_minus | rand | abl:<component>:<zero|mean>,
optionally scaled with _x<s> (e.g. dperp_plus_x0.25) and combined with '+'. Write-up: results/08_circuits/closing.md.

Problem sets are GSM8K train problems disjoint from the 32 used in Phases 4-5 and looping.
SUBJECT = the p50c-edited model: the base model almost never concludes in thinking mode (stage-0 pilot on the base model:
2/64 closed at 8K, 49/64 looping; results/08_circuits/closing/stage0_pilot_base.json), so the user chose to map closing
where it actually happens. Everything except Sparks' 128 edited neurons is identical to the base model.

  set A (128): fitting (direction, attribution, screening)    set B (64): free-generation tests only
"""
import json
import sys
import time

import numpy as np
import torch

from probe.model import generate, grade, load_model, loop_stats, sample_problems
from probe.weights import LOGS, RESULTS, save_json

torch.set_num_threads(4)
OUT = RESULTS / "08_circuits" / "closing"
SUBJECT = "p50c"
CAP = 8192
BATCH = 16


def problem_sets():
    used = {r["id"] for r in sample_problems(32, 32)}
    data = [json.loads(l) for l in (LOGS / "practice" / "gsm8k_train.jsonl").read_text().split("\n") if l.strip()]
    pool = [d for d in data if d["id"] not in used]
    rng = np.random.default_rng(2026)
    pick = [pool[i] for i in rng.choice(len(pool), 192, replace=False)]
    return {"A": pick[:128], "B": pick[128:]}


def run_gen(name, problems, seed=0, edit=None, hooks_fn=None):
    """Generate with the base (or edited) model; record the </think> position of each answer."""
    model, tok = load_model(edit)
    model._probe_tok = tok
    hooks = hooks_fn(model) if hooks_fn else []
    end_id = tok.convert_tokens_to_ids("</think>")
    t0 = time.time()
    res, tps = generate(model, tok, [p["problem"] for p in problems], CAP, seed=1000 * seed, batch_size=BATCH)
    for h in hooks:
        h.remove()
    rows = []
    for p, g in zip(problems, res):
        text = tok.decode(g["ids"])
        close = g["ids"].index(end_id) if end_id in g["ids"] else None
        rows.append({"id": p["id"], "gold": p["answer"], "n_tokens": len(g["ids"]), "finished": g["finished"],
                     "close_pos": close, "closed_think": close is not None, "correct": grade(text, p["answer"]),
                     **loop_stats(g["ids"]), "ids": g["ids"]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"gen_{name}.json").write_text(json.dumps({"seed": seed, "edit": edit, "rows": rows}))
    closed = [r for r in rows if r["closed_think"]]
    s = {"n": len(rows), "closed": len(closed), "closed_rate": len(closed) / len(rows),
         "close_pos_median": float(np.median([r["close_pos"] for r in closed])) if closed else None,
         "correct": sum(r["correct"] for r in rows), "correct_given_closed": sum(r["correct"] for r in closed),
         "looping": sum(r["looping"] for r in rows), "hit_cap": sum(not r["finished"] for r in rows)}
    print(f"{name} (seed {seed}): {s} | {tps:.0f} tok/s, {time.time() - t0:.0f}s", flush=True)
    return s


def cmd_pilot():
    s = run_gen("A0", problem_sets()["A"][:64], edit=SUBJECT)
    save_json(s, "08_circuits/closing/stage0_pilot.json")
    print("GATE:", "pass" if s["closed_rate"] >= 0.2 else "FAIL (closing rate < 20%: adjust cap before continuing)")


def cmd_corpus():
    save_json(run_gen("A1", problem_sets()["A"][64:], edit=SUBJECT), "08_circuits/closing/stage1_corpus.json")


# ---------------------------------------------------------------- stage 2: the closing direction
def corpus_rows():
    rows = []
    for name in ("A0", "A1"):
        rows += json.loads((OUT / f"gen_{name}.json").read_text())["rows"]
    return rows


def position_sets(rows, rng, n_mid=100):
    """decision = the 8 tokens ending right before </think> (the last one predicts it); mid = > 256 tokens before
    closing in closed answers; other = anywhere in answers that never close."""
    out = []
    for r in rows:
        if r["closed_think"]:
            c = r["close_pos"]
            dec = list(range(max(0, c - 8), c))
            pool = np.arange(16, max(16, c - 256))
            mid = sorted(rng.choice(pool, min(n_mid, len(pool)), replace=False).tolist()) if len(pool) else []
            out.append({"dec": dec, "mid": mid, "other": [], "end": c})
        else:
            pool = np.arange(16, r["n_tokens"])
            out.append({"dec": [], "mid": [], "other": sorted(rng.choice(pool, min(n_mid, len(pool)), replace=False).tolist()),
                        "end": r["n_tokens"] - 1})
    return out


@torch.no_grad()
def cmd_direction():
    from probe.circuit import full_ids
    from probe.lens import lens, vocab
    model, tok = load_model(SUBJECT)
    dev = model.device
    rows = corpus_rows()
    probs = {p["id"]: p for p in problem_sets()["A"]}
    rng = np.random.default_rng(5)
    sets = position_sets(rows, rng)
    L = len(model.model.layers)
    acc = {k: [[] for _ in range(L)] for k in ("dec", "mid", "other")}
    prof = []       # residual after L19 over the last 512 tokens before closing
    gen_of = {k: [] for k in ("dec", "mid", "other")}
    t0 = time.time()
    for gi, (r, s) in enumerate(zip(rows, sets)):
        ids, P = full_ids(tok, probs[r["id"]]["problem"], r["ids"][:s["end"] + 1], dev)
        hs = model.model(ids, output_hidden_states=True).hidden_states
        for k in ("dec", "mid", "other"):
            if s[k]:
                idx = torch.tensor([P + t for t in s[k]], device=dev)
                for l in range(L):
                    acc[k][l].append(hs[l + 1][0, idx].float().cpu().numpy())
                gen_of[k] += [gi] * len(s[k])
        if r["closed_think"] and s["end"] >= 512:
            prof.append(hs[20][0, P + s["end"] - 512:P + s["end"]].float().cpu().numpy())
        if gi % 16 == 0:
            print(f"  gen {gi}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    X = {k: [np.concatenate(acc[k][l]) for l in range(L)] for k in acc}
    G = {k: np.array(v) for k, v in gen_of.items()}
    u = {l: np.load(RESULTS / f"u{l}_p50c.npy") for l in (15, 19)}
    sign = {int(k): v["loop_sign"] for k, v in json.loads((RESULTS / "04" / "analysis.json").read_text())["steer"].items()}
    anti = {l: -sign[l] * u[l] / np.linalg.norm(u[l]) for l in (15, 19)}  # the 'wrap up' side of u
    half = rng.permutation(len(rows))[: len(rows) // 2]
    unit = lambda v: v / np.linalg.norm(v)
    res = {"n_closed": int(sum(r["closed_think"] for r in rows)), "n_rows": len(rows),
           "n_positions": {k: int(len(G[k])) for k in G}, "layers": []}
    dirs = {}
    for l in range(L):
        ref = np.concatenate([X["mid"][l], X["other"][l]])
        ref_g = np.concatenate([G["mid"], G["other"]])
        d = unit(X["dec"][l].mean(0) - ref.mean(0))
        h1 = np.isin(G["dec"], half); r1 = np.isin(ref_g, half)
        d1 = unit(X["dec"][l][h1].mean(0) - ref[r1].mean(0)); d2 = unit(X["dec"][l][~h1].mean(0) - ref[~r1].mean(0))
        row = {"layer": l, "splithalf_cos": float(d1 @ d2),
               "cos_anti_u15": float(d @ anti[15]), "cos_anti_u19": float(d @ anti[19]),
               "dec_vs_mid_only_cos": float(d @ unit(X["dec"][l].mean(0) - X["mid"][l].mean(0)))}
        res["layers"].append(row); dirs[l] = d
    for l in (15, 19):  # steering vectors for stage 5: closing direction with its u component removed
        uu = u[l] / np.linalg.norm(u[l])
        dp = dirs[l] - (dirs[l] @ uu) * uu
        res[f"perp_norm_L{l}"] = float(np.linalg.norm(dp))
        np.save(OUT / f"dclose{l}.npy", dirs[l]); np.save(OUT / f"dperp{l}.npy", unit(dp))
    np.save(OUT / "dclose_all_layers.npy", np.stack([dirs[l] for l in range(L)]))
    # logit lens of the closing direction at the last layers
    from probe.weights import Checkpoint
    ck = Checkpoint(); inv = vocab()
    E = ck.get("model.language_model.embed_tokens.weight"); gF = ck.get("model.language_model.norm.weight")
    res["lens"] = {f"L{l}": lens(dirs[l], E, gF, inv, k=15) for l in (15, 19, 23)}
    # onset profile: projection on d_close (L19) over the last 512 tokens before </think>, 32-token bins
    if prof:
        p = np.stack(prof) @ dirs[19]
        base = (np.concatenate([X["mid"][19], X["other"][19]]) @ dirs[19]).mean()
        res["onset_profile_L19_bins32"] = (p.reshape(len(prof), 16, 32).mean((0, 2)) - base).round(3).tolist()
        res["onset_profile_n"] = len(prof)
    save_json(res, "08_circuits/closing/stage2_direction.json")
    print(f"closed answers {res['n_closed']}/{res['n_rows']}; positions {res['n_positions']} ({time.time() - t0:.0f}s)")
    for row in res["layers"]:
        if row["layer"] in (3, 7, 11, 15, 17, 19, 21, 23):
            print(f"  L{row['layer']:2d}: split-half {row['splithalf_cos']:+.3f}  cos(wrap-up side of u15) {row['cos_anti_u15']:+.3f}  u19 {row['cos_anti_u19']:+.3f}")
    print(f"  |d_close - u part|: L15 {res['perp_norm_L15']:.3f}, L19 {res['perp_norm_L19']:.3f}")
    for k, v in res["lens"].items():
        print(f"  lens {k} +: {' | '.join(v['promote'][:10])}")
    print("  onset profile (L19, 32-token bins, last 512 before </think>):", res.get("onset_profile_L19_bins32"))


# ---------------------------------------------------------------- stage 3: who writes the </think> prediction?
def cmd_attribution(n_mid=8):
    """Direct-path attribution of log P(</think>) at the closing decision (the token right before </think>), with the
    final RMSNorm scale frozen, to every head and neuron in layers 0-23 + embeddings; contrast with mid-reasoning
    positions of the same answers. Also: each component's direct push along d_close (L19) at the same positions."""
    from probe.circuit import FULL_ATTN as FA, full_ids, out_proj_of
    model, tok = load_model(SUBJECT)
    for p in model.parameters():
        p.requires_grad_(False)
    dev = model.device
    rows = [r for r in corpus_rows() if r["closed_think"]]
    probs = {p["id"]: p for p in problem_sets()["A"]}
    end_id = tok.convert_tokens_to_ids("</think>")
    d19 = torch.tensor(np.load(OUT / "dclose19.npy"), dtype=torch.float32, device=dev)
    rng = np.random.default_rng(6)
    cap, sel = {}, {}
    hooks = []
    def grab(key):
        def pre(mod, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]); cap[key] = x[sel["abs"]].float()
        return pre
    for l in range(24):
        hooks.append(out_proj_of(model, l).register_forward_pre_hook(grab(("h", l))))
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(grab(("n", l))))
    hooks.append(model.model.norm.register_forward_pre_hook(grab(("x", 0))))
    hooks.append(model.model.embed_tokens.register_forward_hook(lambda m, i, o: cap.__setitem__(("e", 0), o.reshape(-1, o.shape[-1])[sel["abs"]].float())))
    E = model.lm_head.weight.float()
    fn = model.model.norm
    acc = {"dec": {}, "mid": {}, "dproj_dec": {}, "dproj_mid": {}}
    check = []
    t0 = time.time()
    for r in rows:
        c = r["close_pos"]
        mids = sorted(rng.choice(np.arange(16, max(17, c - 256)), min(n_mid, max(1, c - 272)), replace=False).tolist())
        pos = [c - 1] + mids                         # position c-1 predicts </think>
        ids, P = full_ids(tok, probs[r["id"]]["problem"], r["ids"][:c + 1], dev)
        sel["abs"] = torch.tensor([P + t for t in pos], device=dev)
        with torch.no_grad():
            model.model(ids)
        x = cap[("x", 0)].clone().requires_grad_(True)
        with torch.enable_grad():
            rs = torch.rsqrt(x.detach().pow(2).mean(-1, keepdim=True) + fn.eps)   # frozen final-norm scale
            lp = torch.log_softmax((x * rs * (1.0 + fn.weight.float())) @ E.T, -1)[:, end_id]
            G = torch.cat([torch.autograd.grad(lp[i], x, retain_graph=True)[0][i:i + 1] for i in range(len(pos))])
        grp = np.array(["dec"] + ["mid"] * len(mids))
        parts = torch.zeros(len(pos), device=dev)
        def add(key, contrib, dproj):                 # contrib, dproj: (n_pos, n_units)
            nonlocal parts
            parts = parts + contrib.sum(1)
            for g, gm in (("dec", grp == "dec"), ("mid", grp == "mid")):
                m = torch.tensor(gm, device=dev)
                acc[g][key] = acc[g].get(key, 0) + contrib[m].mean(0).cpu().numpy()
                if dproj is not None:
                    acc["dproj_" + g][key] = acc["dproj_" + g].get(key, 0) + dproj[m].mean(0).cpu().numpy()
        add("emb", (G * cap[("e", 0)]).sum(1, keepdim=True), None)
        for l in range(24):
            hd = 256 if l in FA else 128
            Wo = out_proj_of(model, l).weight.float(); z = cap[("h", l)]
            per_head_out = (Wo.view(1024, -1, hd).permute(1, 0, 2) @ z.view(len(pos), -1, hd).permute(1, 2, 0))  # (heads, 1024, n_pos)
            hc = torch.einsum("pd,hdp->ph", G, per_head_out)
            hp = torch.einsum("d,hdp->ph", d19, per_head_out) if l <= 19 else None
            add(f"h{l}", hc, hp)
            Wd = model.model.layers[l].mlp.down_proj.weight.float(); a = cap[("n", l)]
            add(f"n{l}", a * (G @ Wd), (a * (d19 @ Wd)) if l <= 19 else None)
        total = (G * x.detach()).sum(1)
        check.append(float(((parts - total).abs().sum() / (parts.abs().sum() + 1e-9)).item()))
        cap.clear()
    for h in hooks:
        h.remove()
    n = len(rows)
    def rank(A, B, key_ok=lambda k: True):
        items = []
        for k in A:
            if k == "emb" or not key_ok(k):
                continue
            kind, l = k[0], int(k[1:]); diff = A[k] / n - B[k] / n
            for i, v in enumerate(diff):
                name = f"L{l}.mlp.{i}" if kind == "n" else (f"L{l}.attn.h{i}" if l in FA else f"L{l}.lin.h{i}")
                items.append((float(v), name, float(A[k][i] / n), float(B[k][i] / n)))
        items.sort(key=lambda t: -abs(t[0]))
        return items
    logit = rank(acc["dec"], acc["mid"]); dproj = rank(acc["dproj_dec"], acc["dproj_mid"])
    blocks = {}
    for d, nm, _, _ in logit:
        k = nm.split(".")[0] + (".mlp" if ".mlp." in nm else ".attn"); blocks[k] = blocks.get(k, 0) + d
    res = {"n_closed": n, "completeness_rel_err_max": max(check),
           "logP_think_top": [{"name": nm, "diff": d, "dec": a, "mid": b} for d, nm, a, b in logit[:60]],
           "logP_think_share_top20": sum(abs(t[0]) for t in logit[:20]) / sum(abs(t[0]) for t in logit),
           "logP_think_by_block": dict(sorted(blocks.items(), key=lambda kv: -abs(kv[1]))[:14]),
           "emb_dec_minus_mid": float(acc["dec"]["emb"][0] / n - acc["mid"]["emb"][0] / n),
           "dclose_writers_top": [{"name": nm, "diff": d, "dec": a, "mid": b} for d, nm, a, b in dproj[:60]]}
    save_json(res, "08_circuits/closing/stage3_attribution.json")
    print(f"{n} closings; completeness max rel err {res['completeness_rel_err_max']:.1e}; top-20 share {res['logP_think_share_top20']:.0%} ({time.time() - t0:.0f}s)")
    print("  by block:", {k: round(v, 2) for k, v in list(res["logP_think_by_block"].items())[:10]})
    print("  top writers of log P(</think>) (decision - mid):")
    for t in res["logP_think_top"][:12]:
        print(f"    {t['name']:>14} {t['diff']:+.3f}  (dec {t['dec']:+.3f}, mid {t['mid']:+.3f})")
    print("  top direct pushers along d_close at L19 (decision - mid):")
    for t in res["dclose_writers_top"][:10]:
        print(f"    {t['name']:>14} {t['diff']:+.3f}")


# ---------------------------------------------------------------- stage 4: teacher-forced ablation screen
@torch.no_grad()
def cmd_screen(n_gens=30):
    from probe.circuit import FULL_ATTN as FA, full_ids, out_proj_of, parse_comp
    model, tok = load_model(SUBJECT)
    dev = model.device
    allrows = corpus_rows()
    probs = {p["id"]: p for p in problem_sets()["A"]}
    end_id = tok.convert_tokens_to_ids("</think>")
    rng = np.random.default_rng(8)
    s3 = json.loads((OUT / "stage3_attribution.json").read_text())
    comps = {}
    for t in s3["logP_think_top"][:20]: comps[t["name"]] = "logit writer +" if t["diff"] > 0 else "logit writer -"
    for t in s3["dclose_writers_top"][:10]: comps.setdefault(t["name"], "d_close pusher")
    comps.setdefault("L11.attn.h1", "looping: lowered closing")
    for _ in range(10):
        l = int(rng.integers(0, 24)); nh = 8 if l in FA else 16
        comps.setdefault(f"L{l}.{'attn' if l in FA else 'lin'}.h{int(rng.integers(0, nh))}", "random head")
    for _ in range(10):
        comps.setdefault(f"L{int(rng.integers(0, 24))}.mlp.{int(rng.integers(0, 3584))}", "random neuron")
    closed = [r for r in allrows if r["closed_think"]][:n_gens]
    other = [r for r in allrows if not r["closed_think"]][:8]
    gens = []
    for r in closed:
        c = r["close_pos"]
        ids, P = full_ids(tok, probs[r["id"]]["problem"], r["ids"][:c + 1], dev)
        mid = np.sort(rng.choice(np.arange(16, max(17, c - 256)), min(64, max(1, c - 272)), replace=False))
        gens.append({"ids": ids, "dec": P + c - 1, "mid": (P + mid).tolist(), "mid_tgt": [r["ids"][t + 1] for t in mid]})
    means, cnt = {}, 0
    hs = []
    for l in range(24):
        hs.append(out_proj_of(model, l).register_forward_pre_hook(lambda m, i, l=l: means.__setitem__(("o", l), means.get(("o", l), 0) + i[0].reshape(-1, i[0].shape[-1]).float().mean(0))))
        hs.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, i, l=l: means.__setitem__(("d", l), means.get(("d", l), 0) + i[0].reshape(-1, i[0].shape[-1]).float().mean(0))))
    for r in other:
        ids, _ = full_ids(tok, probs[r["id"]]["problem"], r["ids"], dev); model.model(ids); cnt += 1
    for h in hs: h.remove()
    means = {k: v / cnt for k, v in means.items()}

    def run():
        think, nll = [], []
        for g in gens:
            hid = model.model(g["ids"]).last_hidden_state[0]
            think.append(torch.log_softmax(model.lm_head(hid[g["dec"]]).float(), -1)[end_id].item())
            lp = torch.log_softmax(model.lm_head(hid[g["mid"]]).float(), -1)
            nll.append(-lp[torch.arange(len(g["mid"])), torch.tensor(g["mid_tgt"], device=dev)].mean().item())
        return {"think": float(np.mean(think)), "nll": float(np.mean(nll))}

    def hooks_for(name, mode):
        kind, l, idx = parse_comp(name)
        mod = out_proj_of(model, l) if kind == "head" else model.model.layers[l].mlp.down_proj
        key = ("o", l) if kind == "head" else ("d", l)
        def f(m, i, idx=idx, key=key):
            x = i[0].clone(); x[..., idx] = 0 if mode == "zero" else means[key][idx].to(x.dtype); return (x,)
        return [mod.register_forward_pre_hook(f)]
    t0 = time.time()
    base = run()
    print(f"baseline log P(</think>) at decision {base['think']:.3f}, mid NLL {base['nll']:.3f} ({time.time() - t0:.0f}s/condition)", flush=True)
    res = {"baseline": base, "n_gens": len(gens), "conditions": []}
    for name, why in comps.items():
        for mode in ("zero", "mean"):
            hs = hooks_for(name, mode); m = run()
            for h in hs: h.remove()
            res["conditions"].append({"name": name, "why": why, "mode": mode, "d_think": m["think"] - base["think"], "d_nll": m["nll"] - base["nll"]})
    ctrl = [c for c in res["conditions"] if c["why"].startswith("random")]
    res["random_sd"] = {m: {"d_think": float(np.std([c["d_think"] for c in ctrl if c["mode"] == m])),
                            "d_nll": float(np.std([c["d_nll"] for c in ctrl if c["mode"] == m]))} for m in ("zero", "mean")}
    save_json(res, "08_circuits/closing/stage4_screen.json")
    print("random sd:", res["random_sd"], f"({time.time() - t0:.0f}s)")
    for c in sorted(res["conditions"], key=lambda c: c["d_think"])[:12]:
        print(f"  {c['name']:>14} [{c['why']}/{c['mode']}] d log P(</think>) {c['d_think']:+.3f}  d NLL {c['d_nll']:+.4f}")
    print("  ... largest increases:")
    for c in sorted(res["conditions"], key=lambda c: -c["d_think"])[:5]:
        print(f"  {c['name']:>14} [{c['why']}/{c['mode']}] d log P(</think>) {c['d_think']:+.3f}  d NLL {c['d_nll']:+.4f}")


# ---------------------------------------------------------------- stage 5: free-generation tests on set B
def confirm_hooks(cond):
    """cond: 'none' | 'anti_u' | 'dperp_plus' | 'dperp_minus' | 'rand' | 'abl:<component>:<zero|mean>'.
    Steering adds a fixed vector to the residual after layers 15 and 19, with the same per-layer norm beta as the
    looping work (results/04/analysis.json). Combine conditions with '+', e.g. 'dperp_plus+abl:L22.lin.h13:mean'."""
    if "+" in cond:
        # ablations first, so their mean input is measured before any steering hook is registered
        parts = [confirm_hooks(c) for c in sorted(cond.split("+"), key=lambda c: not c.startswith("abl:"))]
        def make_all(model):
            for mk in parts:  # measure every ablation's mean on the intact model before any hook is registered
                if hasattr(mk, "prep"):
                    mk.prep(model)
            return [h for mk in parts for h in mk(model)]
        return make_all
    from probe.causal import steering_hooks
    a = json.loads((RESULTS / "04" / "analysis.json").read_text())["steer"]

    cache = {}

    def abl_parts(model):
        from probe.circuit import out_proj_of, parse_comp
        _, name, mode = cond.split(":")
        kind, l, idx = parse_comp(name)
        mod = out_proj_of(model, l) if kind in ("head", "block_attn") else model.model.layers[l].mlp.down_proj
        idx = slice(None) if idx is None else idx  # whole block: replace its entire output
        if "mean" not in cache:
            cache["mean"] = None
            if mode == "mean":  # mean input over set A answers that never close (same as the stage-4 screen)
                from probe.circuit import full_ids
                probs = {p["id"]: p for p in problem_sets()["A"]}
                tok = model._probe_tok
                vals = []
                hh = mod.register_forward_pre_hook(lambda m, i: vals.append(i[0].reshape(-1, i[0].shape[-1]).float().mean(0)))
                with torch.no_grad():
                    for r in [r for r in corpus_rows() if not r["closed_think"]][:8]:
                        ids, _ = full_ids(tok, probs[r["id"]]["problem"], r["ids"], model.device); model.model(ids)
                hh.remove()
                cache["mean"] = torch.stack(vals).mean(0)
        return mod, idx, cache["mean"]

    def make(model):
        dev = model.device
        if cond == "none":
            return []
        if cond == "ban":  # forbid </think>: the model can never leave the thinking block
            end_id = model._probe_tok.convert_tokens_to_ids("</think>")
            def fb(m, i, o):
                o = o.clone(); o[..., end_id] = torch.finfo(o.dtype).min; return o
            return [model.lm_head.register_forward_hook(fb)]
        if cond.startswith("blind:"):  # blocks in layers a..b can't read the d_perp span (stage 7)
            lo, hi = map(int, cond.split(":")[1].split("-"))
            Q, _ = np.linalg.qr(np.stack([np.load(OUT / f"dperp{l}.npy") for l in (15, 19)], 1))
            Qt = torch.tensor(Q, dtype=torch.float32, device=dev)
            def fp(m, i):
                x = i[0]; y = x.float(); y = y - (y @ Qt) @ Qt.T; return (y.to(x.dtype),) + tuple(i[1:])
            return [n.register_forward_pre_hook(fp) for l in range(lo, hi + 1)
                    for n in (model.model.layers[l].input_layernorm, model.model.layers[l].post_attention_layernorm)]
        if cond.startswith("abl:"):
            mod, idx, mean = abl_parts(model)
            def f(m, i, idx=idx, mean=mean):
                x = i[0].clone(); x[..., idx] = 0 if mean is None else mean[idx].to(x.dtype); return (x,)
            return [mod.register_forward_pre_hook(f)]
        vecs = {}
        rng = np.random.default_rng(123)
        base, scale = cond, 1.0
        if "_x" in cond:
            base, x = cond.rsplit("_x", 1)
            if "." not in x:  # 'x025' is a filename tag, not a strength: it would parse as 25x
                raise ValueError(f"write the strength with a decimal point, e.g. _x0.25, not _x{x}")
            scale = float(x)
        for l in (15, 19):
            beta = scale * a[str(l)]["beta"]
            if base == "anti_u":
                u = np.load(RESULTS / f"u{l}_p50c.npy"); v = -a[str(l)]["loop_sign"] * u / np.linalg.norm(u)
            elif base in ("dperp_plus", "dperp_minus"):
                v = np.load(OUT / f"dperp{l}.npy") * (1 if base == "dperp_plus" else -1)
            elif base == "dperpnd_plus":  # d_perp minus its direct </think>-logit component (stage 6)
                end_id = model._probe_tok.convert_tokens_to_ids("</think>")
                w = ((1 + model.model.norm.weight.float()) * model.lm_head.weight[end_id].float()).detach().cpu().numpy()
                w_hat = w / np.linalg.norm(w)
                v = np.load(OUT / f"dperp{l}.npy"); v = v - (v @ w_hat) * w_hat
            elif base == "dperpnw_plus":  # d_perp minus the span of its top-K direct-logit tokens (stage 7)
                v = np.load(OUT / f"dperp{l}.npy"); Q = wrap_span(model, l); v = v - Q @ (Q.T @ v)
            elif base == "dperp15_plus":  # push at L15 only (stage 7)
                if l == 19:
                    continue
                v = np.load(OUT / f"dperp{l}.npy")
            elif base == "rand":
                r = rng.standard_normal(1024); v = r / np.linalg.norm(r)
            else:
                raise ValueError(cond)
            vecs[l] = torch.tensor(beta * v, dtype=torch.float32, device=dev)
        return steering_hooks(model, vecs)
    if cond.startswith("abl:"):
        make.prep = abl_parts
    return make


WRAP_K = 20


def direct_logit_dirs(model):
    """Rows (1 + g_final) * E[t]: the residual direction that raises token t's logit through final norm + unembedding."""
    return ((1 + model.model.norm.weight.float()) * model.lm_head.weight.float()).detach()


def wrap_span(model, l, k=WRAP_K):
    """Orthonormal basis (1024, k) of the direct-logit directions of the k tokens most aligned with d_perp at layer l."""
    W = direct_logit_dirs(model).cpu().numpy()
    v = np.load(OUT / f"dperp{l}.npy")
    c = (W @ v) / np.linalg.norm(W, axis=1)
    top = np.argsort(-c)[:k]
    Q, _ = np.linalg.qr(W[top].T)
    return Q


def cmd_confirm(cond, seed):
    tag = cond.replace("+", "__").replace(":", "_").replace(".", "")
    run_gen(f"B_{tag}_s{seed}", problem_sets()["B"], seed=seed, edit=SUBJECT, hooks_fn=confirm_hooks(cond))


# ---------------------------------------------------------------- stage 6: decision or logit push?
@torch.no_grad()
def cmd_mechanism(n_gens=20, n_mid=40):
    """Teacher-forced. Push d_perp (1x, L15+L19) on closed set-A answers and measure the gain in log P(</think>) at
    mid-reasoning positions and at the decision point. Then repeat the push with downstream blocks frozen at their
    un-pushed outputs: what survives 'all of 16-23 frozen' is the direct path (push -> final norm -> unembedding).
    Also push d_perp with its direct-logit component removed (the part along (1+g_final) * E[</think>])."""
    from probe.causal import steering_hooks
    from probe.circuit import attn_mod, full_ids
    model, tok = load_model(SUBJECT)
    dev = model.device
    end_id = tok.convert_tokens_to_ids("</think>")
    a = json.loads((RESULTS / "04" / "analysis.json").read_text())["steer"]
    dp = {l: np.load(OUT / f"dperp{l}.npy") for l in (15, 19)}
    E = model.lm_head.weight.float()
    g = 1 + model.model.norm.weight.float()
    w = (g * E[end_id]).cpu().numpy(); w_hat = w / np.linalg.norm(w)
    # static: how aligned is d_perp with the direct </think> logit direction, vs every other token's?
    W = (g * E).cpu().numpy(); Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    static = {}
    for l in (15, 19):
        c = Wn @ dp[l]
        static[l] = {"cos_with_think_direct": float(c[end_id]), "rank_of_think": int((c > c[end_id]).sum()),
                     "abs_cos_p99_all_tokens": float(np.percentile(np.abs(c), 99))}
    def push(kind):
        out = {}
        for l in (15, 19):
            v = dp[l]
            if kind == "no_direct":
                v = v - (v @ w_hat) * w_hat
            out[l] = torch.tensor(a[str(l)]["beta"] * v, dtype=torch.float32, device=dev)
        return out
    blocks = [f"L{l}.{b}" for l in range(16, 24) for b in ("attn", "mlp")]
    freezes = {"all_16_23": blocks, "late_20_23": [b for b in blocks if int(b[1:3]) >= 20]}
    freezes.update({b: [b] for b in blocks})
    rec = {}
    def record_hooks():
        hs = []
        for l in range(16, 24):
            hs.append(attn_mod(model, l).register_forward_hook(lambda m, i, o, l=l: rec.__setitem__(f"L{l}.attn", (o[0] if isinstance(o, tuple) else o).clone())))
            hs.append(model.model.layers[l].mlp.register_forward_hook(lambda m, i, o, l=l: rec.__setitem__(f"L{l}.mlp", o.clone())))
        return hs
    def freeze_hooks(names):
        hs = []
        for name in names:
            l = int(name.split(".")[0][1:])
            if name.endswith(".attn"):
                hs.append(attn_mod(model, l).register_forward_hook(
                    lambda m, i, o, key=name: (rec[key],) + tuple(o[1:]) if isinstance(o, tuple) else rec[key]))
            else:
                hs.append(model.model.layers[l].mlp.register_forward_hook(lambda m, i, o, key=name: rec[key]))
        return hs
    probs = {p["id"]: p for p in problem_sets()["A"]}
    rows = [r for r in corpus_rows() if r["closed_think"]][:n_gens]
    rng = np.random.default_rng(11)
    acc = {}
    def lp(ids, pos):
        h = model.model(ids).last_hidden_state[0, pos]
        return torch.log_softmax(model.lm_head(h).float(), -1)[:, end_id]
    t0 = time.time()
    for r in rows:
        c = r["close_pos"]
        ids, P = full_ids(tok, probs[r["id"]]["problem"], r["ids"][:c], dev)  # up to, not including, </think>
        pool = np.arange(16, max(17, c - 256))
        mid = np.sort(rng.choice(pool, min(n_mid, len(pool)), replace=False))
        pos = torch.tensor(np.r_[P + mid - 1, P + c - 1], device=dev)  # last entry predicts the real </think>
        res = {}
        hs = record_hooks(); res["clean"] = lp(ids, pos); [h.remove() for h in hs]
        for kind in ("pushed", "no_direct"):
            sh = steering_hooks(model, push("pushed" if kind == "pushed" else "no_direct"))
            res[kind] = lp(ids, pos)
            if kind == "pushed":
                for fname, names in freezes.items():
                    fh = freeze_hooks(names); res[f"frozen:{fname}"] = lp(ids, pos); [h.remove() for h in fh]
            [h.remove() for h in sh]
        for k, v in res.items():
            acc.setdefault(k, {"mid": [], "dec": []})
            acc[k]["mid"].append(v[:-1].mean().item()); acc[k]["dec"].append(v[-1].item())
        print(f"  gen {r['id']}: mid gain {acc['pushed']['mid'][-1] - acc['clean']['mid'][-1]:+.2f} ({time.time() - t0:.0f}s)", flush=True)
    m = {k: {s: float(np.mean(v[s])) for s in ("mid", "dec")} for k, v in acc.items()}
    gain = {s: m["pushed"][s] - m["clean"][s] for s in ("mid", "dec")}
    out = {"n_gens": len(rows), "n_mid_per_gen": n_mid, "static": static, "mean_logP": m, "push_gain": gain,
           "no_direct_gain": {s: m["no_direct"][s] - m["clean"][s] for s in ("mid", "dec")},
           "share_of_gain_left_when_frozen": {k.split(":", 1)[1]: {s: (m[k][s] - m["clean"][s]) / gain[s] for s in ("mid", "dec")}
                                             for k in m if k.startswith("frozen:")}}
    save_json(out, "08_circuits/closing/stage6_mechanism.json")
    print(json.dumps({k: out[k] for k in ("static", "push_gain", "no_direct_gain")}, indent=1))
    for k, v in sorted(out["share_of_gain_left_when_frozen"].items(), key=lambda kv: kv[1]["mid"]):
        print(f"  frozen {k:12s} gain left: mid {v['mid']:+.2f}  dec {v['dec']:+.2f}")


# ---------------------------------------------------------------- stage 7 (teacher-forced): wrap-up words, L11.attn.h1
@torch.no_grad()
def cmd_followup(n_gens=20, n_mid=40):
    """(a) Which tokens' direct-logit directions d_perp lines up with, and how much of the push on log P(</think>)
    survives removing the top-K of them (dperpnw) or pushing at L15 only (dperp15). (b) L11.attn.h1, whose ablation
    raised looping in the looping work: what it writes along d_close and u at the decision vs mid positions (ranked
    against every full-attention head), and what mean-ablating it does to log P(</think>) and the d_close signal."""
    from probe.circuit import full_ids, parse_comp
    model, tok = load_model(SUBJECT)
    model._probe_tok = tok
    dev = model.device
    end_id = tok.convert_tokens_to_ids("</think>")
    a = json.loads((RESULTS / "04" / "analysis.json").read_text())["steer"]
    dp = {l: np.load(OUT / f"dperp{l}.npy") for l in (15, 19)}
    W = direct_logit_dirs(model).cpu().numpy(); Wnorm = np.linalg.norm(W, axis=1)
    out = {"wrap_k": WRAP_K, "tokens": {}, "norm_kept": {}}
    for l in (15, 19):
        c = (W @ dp[l]) / Wnorm / np.linalg.norm(dp[l])
        top = np.argsort(-c)[:30]
        out["tokens"][l] = [{"tok": tok.decode([int(t)]), "id": int(t), "cos": float(c[t])} for t in top]
        Q = wrap_span(model, l); v = dp[l] - Q @ (Q.T @ dp[l])
        out["norm_kept"][l] = float(np.linalg.norm(v) / np.linalg.norm(dp[l]))
        print(f"L{l} top tokens:", ", ".join(f"{x['tok']!r} {x['cos']:.2f}" for x in out["tokens"][l][:20]))
        print(f"  norm kept after removing top-{WRAP_K} span: {out['norm_kept'][l]:.3f}")

    # ---- (b) setup: per-head outputs at every full-attention layer
    FA = (3, 7, 11, 15, 19, 23)
    from probe.circuit import out_proj_of
    zin = {}
    hz = [out_proj_of(model, l).register_forward_pre_hook(lambda m, i, l=l: zin.__setitem__(l, i[0][0])) for l in FA]
    dcl = np.load(OUT / "dclose_all_layers.npy")
    dhat = {l: torch.tensor(dcl[l] / np.linalg.norm(dcl[l]), device=dev) for l in FA}
    uhat = {l: torch.tensor((lambda u: u / np.linalg.norm(u))(np.load(RESULTS / f"u{l}_p50c.npy")), device=dev) for l in (15, 19)}
    sign = {l: a[str(l)]["loop_sign"] for l in (15, 19)}
    res_cap = {}
    hr = [model.model.layers[l].register_forward_hook(
        lambda m, i, o, l=l: res_cap.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0])) for l in (15, 19)]

    def heads_proj(pos):
        """{name: (proj on d_close_l, proj on u15 loop side, proj on u19 loop side)} averaged over positions."""
        r = {}
        for l in FA:
            Wo = out_proj_of(model, l).weight.float(); z = zin[l][pos].float()
            for h in range(8):
                o = z[:, h * 256:(h + 1) * 256] @ Wo[:, h * 256:(h + 1) * 256].T
                r[f"L{l}.attn.h{h}"] = [(o @ dhat[l]).mean().item(), (sign[15] * o @ uhat[15]).mean().item(),
                                        (sign[19] * o @ uhat[19]).mean().item()]
        return r

    def push(kind):
        v = {}
        for l in (15, 19):
            x = dp[l]
            if kind == "dperpnw":
                Q = wrap_span(model, l); x = x - Q @ (Q.T @ x)
            if kind == "dperp15" and l == 19:
                continue
            if kind == "dperp19" and l == 15:
                continue
            v[l] = torch.tensor(a[str(l)]["beta"] * x, dtype=torch.float32, device=dev)
        return v

    from probe.causal import steering_hooks
    probs = {p["id"]: p for p in problem_sets()["A"]}
    rows = [r for r in corpus_rows() if r["closed_think"]][:n_gens]
    rng = np.random.default_rng(11)
    acc, hp = {}, {"dec": [], "mid": []}
    abl_mk = confirm_hooks("abl:L11.attn.h1:mean"); abl_mk.prep(model)
    t0 = time.time()
    for r in rows:
        c = r["close_pos"]
        ids, P = full_ids(tok, probs[r["id"]]["problem"], r["ids"][:c], dev)
        pool = np.arange(16, max(17, c - 256))
        mid = np.sort(rng.choice(pool, min(n_mid, len(pool)), replace=False))
        pos = torch.tensor(np.r_[P + mid - 1, P + c - 1], device=dev)
        decw = torch.arange(P + c - 8, P + c, device=dev)  # the 8 decision tokens, as d_close was fitted
        def measure():
            h = model.model(ids).last_hidden_state[0, pos]
            lp = torch.log_softmax(model.lm_head(h).float(), -1)[:, end_id]
            m = {"lp": lp}
            for l in (15, 19):
                x = res_cap[l].float()
                m[f"dc{l}_mid"] = (x[pos[:-1]] @ torch.tensor(dcl[l] / np.linalg.norm(dcl[l]), device=dev)).mean()
                m[f"dc{l}_dec"] = (x[decw] @ torch.tensor(dcl[l] / np.linalg.norm(dcl[l]), device=dev)).mean()
            return m
        res = {"clean": measure()}
        hp["dec"].append(heads_proj(decw)); hp["mid"].append(heads_proj(pos[:-1]))
        for kind in ("pushed", "dperpnw", "dperp15", "dperp19"):
            sh = steering_hooks(model, push(kind)); res[kind] = measure(); [x.remove() for x in sh]
        ah = abl_mk(model); res["abl_L11h1"] = measure(); [x.remove() for x in ah]
        for k, m in res.items():
            d = acc.setdefault(k, {})
            d.setdefault("mid", []).append(m["lp"][:-1].mean().item()); d.setdefault("dec", []).append(m["lp"][-1].item())
            for l in (15, 19):
                for s in ("mid", "dec"):
                    d.setdefault(f"dc{l}_{s}", []).append(m[f"dc{l}_{s}"].item())
    for x in hz + hr:
        x.remove()
    M = {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in acc.items()}
    gain = {k: {s: M[k][s] - M["clean"][s] for s in ("mid", "dec")} for k in M if k != "clean"}
    out["mean"] = M
    out["gain_logP_think"] = gain
    out["share_of_push_gain"] = {k: {s: gain[k][s] / gain["pushed"][s] for s in ("mid", "dec")}
                                 for k in ("dperpnw", "dperp15", "dperp19")}
    out["abl_L11h1_delta"] = {s: M["abl_L11h1"][s] - M["clean"][s] for s in M["clean"]}
    names = list(hp["dec"][0])
    diff = {n: [float(np.mean([g[n][j] for g in hp["dec"]]) - np.mean([g[n][j] for g in hp["mid"]])) for j in range(3)] for n in names}
    order = sorted(names, key=lambda n: -diff[n][0])
    out["heads_dec_minus_mid"] = {n: {"d_close": diff[n][0], "u15_loop": diff[n][1], "u19_loop": diff[n][2],
                                      "rank_d_close": order.index(n) + 1} for n in names}
    out["n_gens"] = len(rows)
    save_json(out, "08_circuits/closing/stage7_followup.json")
    print(f"({time.time() - t0:.0f}s)")
    print("gain in log P(</think>):", json.dumps(gain, indent=0))
    print("share of the full push's gain:", out["share_of_push_gain"])
    print("mean-ablate L11.attn.h1, change:", {k: round(v, 3) for k, v in out["abl_L11h1_delta"].items()})
    print("heads writing d_close more at the decision than mid (top 8 of 48):")
    for n in order[:8]:
        print(f"  {n}: {diff[n][0]:+.3f}  (u15 loop {diff[n][1]:+.3f}, u19 loop {diff[n][2]:+.3f})")
    x = out["heads_dec_minus_mid"]["L11.attn.h1"]
    print(f"  L11.attn.h1: {x['d_close']:+.3f}, rank {x['rank_d_close']}/48 (u15 loop {x['u15_loop']:+.3f}, u19 loop {x['u19_loop']:+.3f})")


CONDS = ["none", "anti_u", "dperp_plus", "dperp_minus", "rand", "abl_L22linh13_mean", "dperp_plus__abl_L22linh13_mean",
         "dperp_plus_x025", "dperp_plus_x025__abl_L22linh13_mean", "ban", "dperp_plus__ban", "dperpnd_plus",
         "dperpnw_plus", "dperp15_plus", "abl_L11attnh1_mean", "dperp_plus_x025__abl_L11attnh1_mean",
         "blind_20-23", "dperp_plus_x025__blind_20-23", "blind_16-23", "dperp_plus_x025__blind_16-23",
         "blind_18-18", "dperp_plus_x025__blind_18-18",
         "blind_18-18__blind_20-23", "dperp_plus_x025__blind_18-18__blind_20-23"]


def fisher_p(a, n1, b, n2):
    """Two-sided Fisher exact p for a/n1 vs b/n2 (sum of tables no more likely than the observed one)."""
    from math import comb
    k, n = a + b, n1 + n2
    pr = lambda x: comb(n1, x) * comb(n2, k - x) / comb(n, k)
    p0 = pr(a)
    return min(1.0, sum(pr(x) for x in range(max(0, k - n2), min(k, n1) + 1) if pr(x) <= p0 * (1 + 1e-9)))


def cmd_pool():
    """Stage 5 pooling: 3 seed sets per condition, Fisher exact vs the unsteered p50c reference."""
    metrics = ["closed_think", "correct", "looping", "hit_cap"]
    per = {}
    for c in CONDS:
        per[c] = []
        for s in range(3):
            rows = json.load(open(OUT / f"gen_B_{c}_s{s}.json"))["rows"]
            per[c].append({"n": len(rows), "closed_think": sum(r["closed_think"] for r in rows),
                           "correct": sum(bool(r["correct"]) for r in rows), "looping": sum(r["looping"] for r in rows),
                           "hit_cap": sum(not r["finished"] for r in rows),
                           "close_pos": [r["close_pos"] for r in rows if r["close_pos"] is not None]})
    out = {"subject": SUBJECT, "cap": CAP, "problems": "set B (64 GSM8K train)", "seeds": [0, 1000, 2000], "conditions": {}}
    ref = per["none"]
    for c in CONDS:
        d = {"per_seed": {m: [x[m] for x in per[c]] for m in metrics}, "n": sum(x["n"] for x in per[c])}
        d["pooled"] = {m: sum(d["per_seed"][m]) for m in metrics}
        cp = sum((x["close_pos"] for x in per[c]), [])
        d["close_pos_median"] = float(np.median(cp)) if cp else None
        d["close_pos_median_per_seed"] = [float(np.median(x["close_pos"])) if x["close_pos"] else None for x in per[c]]
        if c != "none":
            d["p_vs_none"] = {m: fisher_p(d["pooled"][m], d["n"], sum(x[m] for x in ref), d["n"]) for m in metrics}
            d["same_direction_all_seeds"] = {
                m: len({np.sign(x[m] - r[m]) for x, r in zip(per[c], ref)}) == 1 and per[c][0][m] != ref[0][m]
                for m in metrics}
        out["conditions"][c] = d
    C = out["conditions"]
    out["contrasts"] = {
        f"{a}_vs_{b}_{m}": fisher_p(C[a]["pooled"][m], C[a]["n"], C[b]["pooled"][m], C[b]["n"])
        for a, b, m in [("dperp_plus", "rand", "closed_think"), ("dperp_minus", "rand", "closed_think"),
                        ("dperp_plus", "anti_u", "closed_think"), ("anti_u", "rand", "closed_think"),
                        ("abl_L22linh13_mean", "rand", "closed_think"), ("dperp_minus", "none", "looping"),
                        ("dperp_plus__abl_L22linh13_mean", "dperp_plus", "closed_think"),
                        ("dperp_plus__abl_L22linh13_mean", "abl_L22linh13_mean", "closed_think"),
                        ("dperp_plus_x025", "none", "closed_think"),
                        ("dperp_plus_x025__abl_L22linh13_mean", "dperp_plus_x025", "closed_think"),
                        ("dperp_plus_x025__abl_L22linh13_mean", "abl_L22linh13_mean", "closed_think"),
                        ("dperp_plus_x025__abl_L22linh13_mean", "none", "closed_think"),
                        ("dperp_plus__ban", "ban", "looping"), ("dperpnd_plus", "dperp_plus", "closed_think"),
                        ("dperpnd_plus", "dperp_plus", "correct"),
                        ("dperpnw_plus", "dperp_plus", "closed_think"), ("dperp15_plus", "dperp_plus", "closed_think"),
                        ("dperp_plus_x025__abl_L11attnh1_mean", "abl_L11attnh1_mean", "closed_think"),
                        ("dperp_plus_x025__abl_L11attnh1_mean", "dperp_plus_x025", "closed_think"),
                        ("dperp_plus_x025__blind_20-23", "blind_20-23", "closed_think"),
                        ("dperp_plus_x025__blind_20-23", "dperp_plus_x025", "closed_think"),
                        ("blind_16-23", "blind_20-23", "closed_think"),
                        ("dperp_plus_x025__blind_16-23", "blind_16-23", "closed_think"),
                        ("dperp_plus_x025__blind_16-23", "dperp_plus_x025__blind_20-23", "closed_think"),
                        ("dperp_plus_x025__blind_18-18", "blind_18-18", "closed_think"),
                        ("dperp_plus_x025__blind_18-18", "dperp_plus_x025", "closed_think"),
                        ("dperp_plus_x025__blind_18-18__blind_20-23", "blind_18-18__blind_20-23", "closed_think"),
                        ("blind_18-18__blind_20-23", "blind_20-23", "closed_think"),
                        ("blind_18-18__blind_20-23", "blind_18-18", "closed_think"),
                        ("dperp_plus_x025__blind_18-18__blind_20-23", "dperp_plus_x025__blind_20-23", "closed_think")]}
    save_json(out, "08_circuits/closing/stage5_pooled.json")
    print(f"{'condition':22s} {'closed':>14s} {'pooled':>7s} {'p':>9s} {'correct':>8s} {'looping':>8s} {'med close':>9s}")
    for c, d in C.items():
        ps = d["per_seed"]
        print(f"{c:22s} {'/'.join(map(str, ps['closed_think'])):>14s} {d['pooled']['closed_think']:>7d} "
              f"{d.get('p_vs_none', {}).get('closed_think', float('nan')):9.2g} {d['pooled']['correct']:>8d} "
              f"{d['pooled']['looping']:>8d} {d['close_pos_median'] or 0:9.0f}  same-dir {d.get('same_direction_all_seeds', {})}")
    print(json.dumps(out["contrasts"], indent=1))


if __name__ == "__main__":
    {"pilot": cmd_pilot, "corpus": cmd_corpus, "direction": cmd_direction, "attribution": cmd_attribution, "screen": cmd_screen,
     "confirm": lambda: cmd_confirm(sys.argv[2], int(sys.argv[3])), "pool": cmd_pool, "mechanism": cmd_mechanism, "followup": cmd_followup}[sys.argv[1]]()
