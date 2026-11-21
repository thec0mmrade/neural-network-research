"""Phase 3: read the p50c direction u from the weights alone, with random controls.

  - logit lens: which tokens does writing +u / -u promote, through the final (1+w) RMSNorm and tied embedding?
  - writers: which output columns in all layers write along u (and where Astra's 64 edited columns rank);
  - readers: which later MLP neurons read u (gate/up rows, through each layer's (1+w) pre-MLP norm).
    nix-shell nix/cpu-shell.nix --run 'python -m probe.lens'
"""
import json

import numpy as np

from probe.weights import N_LAYERS, RESULTS, Checkpoint, layer_tensor, save_json, snapshot_dir

TOPK = 25


def vocab():
    v = json.loads((snapshot_dir() / "vocab.json").read_text())
    inv = {i: t for t, i in v.items()}
    tok = json.loads((snapshot_dir() / "tokenizer.json").read_text())
    for a in tok.get("added_tokens", []):
        inv[a["id"]] = a["content"]
    return inv


def _byte_decoder():
    """Inverse of GPT-2's bytes_to_unicode table used by byte-level BPE vocabularies."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BD = _byte_decoder()


def show(t: str) -> str:
    try:
        # partial multi-byte fragments show as \xNN escapes rather than U+FFFD
        s = bytes(_BD[ch] for ch in t).decode("utf-8", errors="backslashreplace")
    except KeyError:  # added/special tokens are stored as plain text
        s = t
    # some vocab tokens literally contain U+FFFD (it occurs in training text); show it as an escape
    return repr(s)[1:-1].replace("�", "\\ufffd")


def lens(vec, emb, g, inv, k=TOPK):
    logits = emb @ (vec * (1.0 + g))
    o = np.argsort(logits)
    return {"promote": [show(inv.get(int(i), f"<{i}>")) for i in o[::-1][:k]],
            "suppress": [show(inv.get(int(i), f"<{i}>")) for i in o[:k]]}


def main():
    ck = Checkpoint()
    inv = vocab()
    emb = ck.get("model.language_model.embed_tokens.weight")
    g = ck.get("model.language_model.norm.weight")
    u15, u19 = np.load(RESULTS / "u15_p50c.npy"), np.load(RESULTS / "u19_p50c.npy")
    edit = json.loads((RESULTS / "02_edit.json").read_text())["p50c"]
    rng = np.random.default_rng(0)
    w15 = ck.get(layer_tensor(15, "down"))
    unedited = sorted(set(range(w15.shape[1])) - set(edit["15"]["columns"]))
    rand_col = int(rng.choice(unedited))
    out = {"lens": {
        "u15 (+u: what the original edited neurons wrote)": lens(u15, emb, g, inv),
        "u19": lens(u19, emb, g, inv),
        "control: random unit vector": lens(rng.standard_normal(1024), emb, g, inv),
        f"control: unedited L15 down_proj column {rand_col}": lens(w15[:, rand_col], emb, g, inv),
    }}

    # Writers: |cos| of every output column in every layer with u15.
    writers = []
    for i in range(N_LAYERS):
        for kind in ("down", "out"):
            w = ck.get(layer_tensor(i, kind))
            c = (u15 @ w) / np.linalg.norm(w, axis=0)
            for j in np.argsort(-np.abs(c))[:40]:
                writers.append((float(c[j]), i, kind, int(j), float(u15 @ w[:, j])))
    writers.sort(key=lambda r: -abs(r[0]))
    ed = {(15, j) for j in edit["15"]["columns"]} | {(19, j) for j in edit["19"]["columns"]}
    top = writers[:50]
    out["writers_top50"] = [{"cos": c, "layer": i, "kind": k, "col": j, "edited": (k == "down" and (i, j) in ed),
                             "u_component": a} for c, i, k, j, a in top]
    out["writers_by_layer_count_in_top200"] = {}
    for c, i, k, j, a in writers[:200]:
        key = f"L{i}.{k}"
        out["writers_by_layer_count_in_top200"][key] = out["writers_by_layer_count_in_top200"].get(key, 0) + 1

    # Readers: MLP gate/up rows in layers 16-23 (after the edit), read through the pre-MLP (1+w) norm.
    readers = []
    for i in range(16, N_LAYERS):
        gn = ck.get(layer_tensor(i, "post_norm"))
        for kind in ("gate", "up"):
            w = ck.get(layer_tensor(i, kind)) * (1.0 + gn)[None, :]
            c = (w @ u15) / np.linalg.norm(w, axis=1)
            for j in np.argsort(-np.abs(c))[:10]:
                readers.append({"layer": i, "kind": kind, "neuron": int(j), "cos": float(c[j])})
    readers.sort(key=lambda r: -abs(r["cos"]))
    out["readers_top20"] = readers[:20]
    out["reader_random_baseline_max_abs_cos"] = float(np.sqrt(2 * np.log(3584 * 2 * 8) / 1024))  # ~max over random
    save_json(out, "03_lens.json")
    write_md(out)
    for k, v in out["lens"].items():
        print(f"\n[{k}]\n  +: {' | '.join(v['promote'][:15])}\n  -: {' | '.join(v['suppress'][:15])}")
    print("\nwriters top 12:")
    for r in out["writers_top50"][:12]:
        print(f"  L{r['layer']:2d} {r['kind']:4s} col {r['col']:4d} cos {r['cos']:+.3f} edited={r['edited']}")
    print("writer layer counts (top200):", out["writers_by_layer_count_in_top200"])
    print("readers top 8:", [(r['layer'], r['kind'], r['neuron'], round(r['cos'], 3)) for r in out['readers_top20'][:8]])


def write_md(o):
    L = ["# Phase 3: reading u from the weights alone", "",
         "Logit lens: `E @ ((1+g_final) * v)` with the tied embedding E. Top 25 tokens each way.", ""]
    for k, v in o["lens"].items():
        L += [f"## {k}", "", "**+**: " + " · ".join(f"`{t}`" for t in v["promote"]), "",
              "**−**: " + " · ".join(f"`{t}`" for t in v["suppress"]), ""]
    L += ["## Strongest writers of u15 (|cos| of output column with u15)", "", "| layer | kind | col | cos | edited by p50c |", "|---|---|---|---|---|"]
    L += [f"| {r['layer']} | {r['kind']} | {r['col']} | {r['cos']:+.3f} | {'yes' if r['edited'] else ''} |" for r in o["writers_top50"][:25]]
    L += ["", "Layer counts among the top 200 writers: " + ", ".join(f"{k}: {v}" for k, v in sorted(o["writers_by_layer_count_in_top200"].items(), key=lambda kv: -kv[1])), "",
          "## Strongest readers of u15 in layers 16-23 (MLP gate/up rows through the pre-MLP norm)", "",
          f"Rough max |cos| expected for random directions: {o['reader_random_baseline_max_abs_cos']:.3f}", "",
          "| layer | kind | neuron | cos |", "|---|---|---|---|"]
    L += [f"| {r['layer']} | {r['kind']} | {r['neuron']} | {r['cos']:+.3f} |" for r in o["readers_top20"]]
    (RESULTS / "03_lens.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
