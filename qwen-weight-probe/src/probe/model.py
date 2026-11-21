"""GPU side: load the Qwen3.5-0.8B text model (base or with a Sparks edit), sample like the Sparks eval, grade, detect loops.

Run inside the CUDA shell:  nix-shell --run 'python -m probe.<module>'
"""
import re
import time

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from probe.weights import LOGS, RESULTS, snapshot_dir

TEXT_CFG = LOGS / "base_model" / "Qwen3.5-0.8B-text"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
# Sparks Exp 8 "think35" protocol (code/evaluate.py SAMPLING_THINK35), with our own max_new_tokens cap.
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}


def load_model(edit: str | None = None, device="cuda"):
    """Text-only Qwen3_5ForCausalLM with the checkpoint's model.language_model.* weights; optional Sparks edit run."""
    cfg = AutoConfig.from_pretrained(TEXT_CFG)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).to(device)
    sd = {}
    path = next(snapshot_dir().glob("model*.safetensors"))
    with safe_open(path, framework="pt", device=device) as f:
        for k in f.keys():
            if k.startswith("model.language_model."):
                sd["model." + k.removeprefix("model.language_model.")] = f.get_tensor(k)
    if edit:
        for p in sorted((RESULTS / "edits" / edit).glob("*.npy")):
            k = "model." + p.stem.removeprefix("model.language_model.")
            assert k in sd, k
            sd[k] = torch.from_numpy(np.load(p)).to(device=device, dtype=torch.bfloat16)
    own = model.state_dict()
    missing = [k for k in own if k not in sd and not k.startswith("lm_head")]
    unexpected = [k for k in sd if k not in own]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    model.load_state_dict(sd, strict=False)
    model.tie_weights()
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr() or torch.equal(
        model.lm_head.weight, model.model.embed_tokens.weight)
    model.eval()
    tok = AutoTokenizer.from_pretrained(snapshot_dir())
    tok.padding_side = "left"
    return model, tok


class VllmLikeSampling(LogitsProcessor):
    """vLLM order: presence penalty on generated tokens, then temperature, top-k, top-p. HF then samples the result."""

    def __init__(self, prompt_len, temperature, top_k, top_p, presence_penalty):
        self.P, self.t, self.k, self.p, self.pp = prompt_len, temperature, top_k, top_p, presence_penalty

    def __call__(self, input_ids, scores):
        scores = scores.float()
        gen = input_ids[:, self.P:]
        if gen.shape[1]:
            seen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, gen, True)
            scores = scores - self.pp * seen
        scores = scores / self.t
        kth = torch.topk(scores, self.k, dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < kth, float("-inf"))
        sp, si = torch.sort(scores, descending=True, dim=-1)
        cum = sp.softmax(-1).cumsum(-1)
        drop = cum - sp.softmax(-1) > self.p  # keep the smallest prefix with mass >= p
        sp = sp.masked_fill(drop, float("-inf"))
        return torch.full_like(scores, float("-inf")).scatter(1, si, sp)


def build_prompts(tok, problems):
    return [tok.apply_chat_template([{"role": "user", "content": p + "\n" + INSTRUCTION}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True) for p in problems]


@torch.no_grad()
def generate(model, tok, problems, max_new_tokens, seed=0, batch_size=32):
    """Returns a list of generated token-id lists (eos and padding stripped) and tokens/s."""
    out, t0, n_tok = [], time.time(), 0
    for b in range(0, len(problems), batch_size):
        enc = tok(build_prompts(tok, problems[b:b + batch_size]), return_tensors="pt", padding=True).to(model.device)
        P = enc.input_ids.shape[1]
        torch.manual_seed(seed + b)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                           logits_processor=LogitsProcessorList([VllmLikeSampling(P, **SAMPLING)]),
                           pad_token_id=tok.pad_token_id, eos_token_id=model.generation_config.eos_token_id)
        eos = set(np.atleast_1d(model.generation_config.eos_token_id).tolist()) | {tok.pad_token_id}
        for row in g[:, P:].tolist():
            cut = next((i for i, t in enumerate(row) if t in eos), None)
            ids = row if cut is None else row[:cut]
            out.append({"ids": ids, "finished": cut is not None})
            n_tok += len(ids)
    return out, n_tok / (time.time() - t0)


# ---- grading (same rule as the logs' mathqa.py: last \boxed{...}, brace matched; math-verify 0.9.0)
def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j + 1:k]
    return None


def grade(text, gold):
    from math_verify import parse, verify
    pred = last_boxed(text.split("</think>")[-1]) if "</think>" in text else None
    if pred is None:
        return False
    try:
        return bool(verify(parse("\\boxed{" + gold + "}"), parse("\\boxed{" + pred + "}")))
    except Exception:
        return False


# ---- loops: a token is "repeated" if the n-gram ending at it already occurred earlier in the generation
def repeated_mask(ids, n=16):
    seen, rep = set(), np.zeros(len(ids), bool)
    for t in range(n - 1, len(ids)):
        g = tuple(ids[t - n + 1:t + 1])
        rep[t] = g in seen
        seen.add(g)
    return rep


def loop_stats(ids, n=16, tail=1024, thresh=0.5):
    rep = repeated_mask(ids, n)
    tail_frac = float(rep[-tail:].mean()) if len(ids) else 0.0
    onset = None
    if tail_frac > thresh:  # first position from which >thresh of a 256-token window is repeated
        w = np.convolve(rep, np.ones(256), "valid") / 256
        onset = int(np.argmax(w > thresh)) if (w > thresh).any() else None
    return {"rep_frac": float(rep.mean()) if len(ids) else 0.0, "tail_rep_frac": tail_frac,
            "looping": tail_frac > thresh, "loop_onset": onset}


def sample_problems(n_math=32, n_gsm=32, seed=0):
    import json
    rng = np.random.default_rng(seed)
    rows = []
    for fn, n in (("math_train.jsonl", n_math), ("gsm8k_train.jsonl", n_gsm)):
        # split on "\n" only: str.splitlines() also breaks on U+2028 etc. inside JSON strings
        data = [json.loads(l) for l in (LOGS / "practice" / fn).read_text().split("\n") if l.strip()]
        rows += [data[i] for i in rng.choice(len(data), n, replace=False)]
    return rows


def smoke():
    t = time.time()
    model, tok = load_model()
    print(f"loaded in {time.time() - t:.1f}s; params {sum(p.numel() for p in model.parameters()):,}; "
          f"gpu mem {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    probs = sample_problems(4, 4)
    for bs in (8,):
        res, tps = generate(model, tok, [r["problem"] for r in probs], max_new_tokens=256, batch_size=bs)
        print(f"batch {bs}: {tps:.0f} tok/s, peak mem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(repr(tok.decode(res[0]["ids"])[:600]))


if __name__ == "__main__":
    smoke()
