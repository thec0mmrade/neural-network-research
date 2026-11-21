"""Dose-response sweep report: steering strength (multiples of beta) vs looping, with Fisher exact tests.

    nix-shell nix/cpu-shell.nix --run 'python -m probe.sweep'
Reads results/04/gen_<cond>.json (0x = base / p50c, 1x = plain names, others = <cond>_x<s>).
"""
import json
from math import comb

from probe.weights import RESULTS, save_json

OUT = RESULTS / "04"
SCALES = [0, 0.25, 0.5, 1, 2, 4]
ROWS = {"p50c_pro": "p50c", "base_anti": "base", "p50c_rand": "p50c"}  # row -> its 0x reference


def fisher(a, n1, b, n2):
    K, N = a + b, n1 + n2
    pr = lambda x: comb(n1, x) * comb(n2, K - x) / comb(N, K)
    p0 = pr(a)
    return min(1.0, sum(pr(x) for x in range(max(0, K - n2), min(K, n1) + 1) if pr(x) <= p0 * (1 + 1e-9)))


def fname(row, s):
    if s == 0:
        return ROWS[row]
    return row if s == 1 else f"{row}_x{s:g}"


def stats(cond):
    p = OUT / f"gen_{cond}.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text())
    n = len(rows)
    return {"cond": cond, "n": n, "looping": sum(r["looping"] for r in rows),
            "rep_frac": sum(r["rep_frac"] for r in rows) / n, "closed_think": sum(r["closed_think"] for r in rows),
            "correct": sum(r["correct"] for r in rows), "hit_cap": sum(not r["finished"] for r in rows)}


def main():
    res = {}
    for row, ref in ROWS.items():
        r0 = stats(ref)
        res[row] = []
        for s in SCALES:
            st = stats(fname(row, s))
            if st is None:
                continue
            st["scale"] = s
            st["p_vs_0x"] = fisher(st["looping"], st["n"], r0["looping"], r0["n"])
            res[row].append(st)
    matched = []
    for s in SCALES[1:]:
        a, b = stats(fname("p50c_pro", s)), stats(fname("p50c_rand", s))
        if a and b:
            matched.append({"scale": s, "pro": a["looping"], "rand": b["looping"],
                            "p": fisher(a["looping"], a["n"], b["looping"], b["n"])})
    res["pro_vs_rand"] = matched
    save_json(res, "05_sweep.json")
    L = ["# Dose-response sweep: steering strength vs looping", "",
         "Same 64 problems, seeds, 4,096-token cap and sampling as Phase 4. Strength is a multiple of the per-layer β "
         "from `04/analysis.json` (L15 0.727, L19 1.075). 0× = the unsteered model (base or p50c). p = Fisher exact, "
         "two-sided, looping vs the row's 0× reference.", ""]
    titles = {"p50c_pro": "Edited model (p50c) pushed TOWARD the loop side of u",
              "base_anti": "Base model pushed AWAY from the loop side of u",
              "p50c_rand": "Edited model pushed along a fixed random direction, same norm (control)"}
    for row in ROWS:
        L += [f"## {titles[row]}", "", "| strength | looping | p vs 0× | mean repeated share | closed thinking | correct | hit 4K cap |",
              "|---|---|---|---|---|---|---|"]
        for st in res[row]:
            L.append(f"| {st['scale']:g}× | {st['looping']}/{st['n']} | {st['p_vs_0x']:.2g} | {st['rep_frac']:.3f} | "
                     f"{st['closed_think']}/{st['n']} | {st['correct']}/{st['n']} | {st['hit_cap']}/{st['n']} |")
        L.append("")
    L += ["## Toward-u vs random at matched strength", "", "| strength | toward u | random | p |", "|---|---|---|---|"]
    L += [f"| {m['scale']:g}× | {m['pro']}/64 | {m['rand']}/64 | {m['p']:.2g} |" for m in matched]
    (RESULTS / "05_sweep.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
