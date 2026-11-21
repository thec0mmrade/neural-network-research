"""Testing stage: a separate Claude session writes passages from a neuron's explanation alone, and the base model shows whether they fire it."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from torch import Tensor

from probe.atlas import activations, prompts
from probe.atlas.batches import process
from probe.atlas.config import Config, neuron_id, neuron_name
from probe.atlas.llm import Refused, call_with_wait
from probe.atlas.record import _group_window
from probe.atlas.recorder import FIRE_FRACTION
from probe.atlas.store import atomic_write_json

TEST_BATCH = 5     # each neuron's 30 passages can take about 3,000 tokens of the reply


def write_tests(cfg: Config, status, items: list[dict], out: Path) -> list[dict]:
    """Test passages per item, in order. Finished batches in `out` are reused, so a restart pays for nothing twice.

    An item's `job`, when present, is copied to its record, failed or not: `summarize` needs it, and it is never sent to Claude.
    """
    def work(chunk: list[dict]) -> list[dict]:
        ids = [f"t{n:02d}" for n in range(1, len(chunk) + 1)]
        prompt = prompts.test_prompt(list(zip(ids, chunk)))
        # R40: batches split a refused chunk down to single neurons; only those go to the fallback model. The test model's
        # refusal is stored first, so a kill before the fallback answers never pays for the refused call again on resume.
        # A refusal by the fallback too propagates, so the neuron is stored as failed and still counts toward R37.
        single = len(chunk) == 1 and bool(cfg.test_fallback_model)
        marker = Path(out) / f"refused_{chunk[0]['nid']}.json"
        fallback = cfg.test_fallback_model if single and marker.exists() else None
        if fallback is None:
            try:
                data = call_with_wait(cfg, status, prompts.TEST_SYSTEM, prompt, prompts.test_schema(ids),
                                      model=cfg.test_model).data
                prompts.validate_tests(data, ids)      # placeholder passages raise Refused here (R41)
            except Refused as e:
                if not single:
                    raise
                atomic_write_json(marker, {"model": cfg.test_model, "reason": e.reason})
                fallback = cfg.test_fallback_model
        if fallback is not None:
            data = call_with_wait(cfg, status, prompts.TEST_SYSTEM, prompt, prompts.test_schema(ids), model=fallback).data
        prompts.validate_tests(data, ids)
        last = chunk[-1]
        status.latest({"name": neuron_name(cfg.layers[last["nid"] // cfg.n_neurons], last["nid"] % cfg.n_neurons),
                       "explanation": last["explanation"], "score": None, "tokens": []})
        return [{"nid": x["nid"], **known(x), **({"test_model": fallback} if fallback else {}),
                 **{c: data[i][c] for c in prompts.TEST_CATEGORIES}} for i, x in zip(ids, chunk)]

    def known(x: dict) -> dict:
        """Fields copied from the item, failed or not. Reasoning-context poles are tested on plain text (R38)."""
        return {"pole": x["pole"], **({"job": x["job"]} if "job" in x else {}),
                **({"tested_out_of_context": True} if x.get("context") == "reasoning" else {})}

    records = process(items, TEST_BATCH, Path(out), work, status=status)
    return [{**r, **known(x)} if "failed" in r else r for x, r in zip(items, records)]


@lru_cache(maxsize=1)
def _reasoning_prefix(tok, user: str) -> tuple[int, ...]:
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)
    return tuple(tok(text, add_special_tokens=False).input_ids)


@lru_cache(maxsize=1)
def _added_tokens(tok) -> re.Pattern | None:
    added = sorted(tok.get_added_vocab(), key=len, reverse=True)
    return re.compile("(" + "|".join(map(re.escape, added)) + ")") if added else None


def _passage_ids(tok, text: str) -> list[int]:
    """Token ids of `text` as plain text: a literal `</think>` in a passage must not close the think block.

    `split_special_tokens=True` is not enough, because `<think>`, `</think>` and the tool tags are added tokens that are not
    marked special. Each literal occurrence is tokenized in two pieces, so it can never match an added token.
    """
    pattern = _added_tokens(tok)
    pieces = pattern.split(text) if pattern else [text]     # odd indices hold the literal added tokens
    parts = [part for n, piece in enumerate(pieces) for part in ((piece[:1], piece[1:]) if n % 2 else (piece,)) if part]
    return [i for part in parts for i in tok(part, add_special_tokens=False).input_ids]


def example_ids(tok, text: str, context: str, cfg: Config) -> tuple[list[int], int]:
    """A passage's token ids as the model runs it, and the index of its first token."""
    if context not in prompts.CONTEXTS:
        raise ValueError(f"unknown context: {context!r}")
    passage = _passage_ids(tok, text)[:prompts.PASSAGE_TOKENS]
    prefix = list(_reasoning_prefix(tok, prompts.REASONING_USER)) if context == "reasoning" else []
    return prefix + passage, len(prefix)


def passage_hit(a: Tensor, gate: Tensor, eligible: Tensor, start: int, pole: int, extreme: float) -> bool:
    """True if the pole-signed activation reaches FIRE_FRACTION of `extreme` at an eligible, gate-positive passage position.

    All tensors are [T] for one neuron. A non-finite activation is no reading (R13), and a pole that never fired has nothing to reach.
    """
    if extreme <= 0:
        return False
    signed = a * pole
    usable = eligible & (gate > 0) & signed.isfinite()
    usable[:start] = False
    return bool(usable.any()) and float(signed[usable].max()) >= FIRE_FRACTION * extreme


def score_tests(cfg: Config, model, tok, layer: int, tests: list[dict], summary: dict) -> list[dict]:
    """Copies of `tests` with each category's passages replaced by its number of hits. Failed records pass through unchanged."""
    passages, hits = [], [dict.fromkeys(prompts.TEST_CATEGORIES, 0) for _ in tests]
    for r, test in enumerate(tests):
        if "failed" in test:
            continue
        idx = test["nid"] - neuron_id(cfg, layer, 0)
        if not 0 <= idx < cfg.n_neurons:
            raise ValueError(f"neuron {test['nid']} is not in layer {layer}")
        extreme = float(summary[f"L{layer}.{'pos' if test['pole'] == 1 else 'neg'}_extreme"][idx])
        for category in prompts.TEST_CATEGORIES:
            for p in test[category]:
                ids, start = example_ids(tok, p["text"], "plain", cfg)     # R38: every test passage runs as plain text
                passages.append({"ids": ids, "start": start, "record": r, "category": category,
                                 "idx": idx, "pole": test["pole"], "extreme": extreme})

    device = next(model.parameters()).device
    excluded = activations.excluded_ids(tok)
    im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
    for group in _group_window(cfg, [(len(p["ids"]), n) for n, p in enumerate(passages)]):
        batch = [passages[n] for n in group]
        ids, attention = activations.pad_right([np.array(p["ids"]) for p in batch], tok.pad_token_id, device)
        eligible = activations.eligible_mask(ids, attention, excluded, im_start_id)

        def on_layer(_layer: int, a: Tensor, gate: Tensor, _value: Tensor) -> None:
            for slot, p in enumerate(batch):
                i = p["idx"]
                if passage_hit(a[slot, :, i], gate[slot, :, i], eligible[slot], p["start"], p["pole"], p["extreme"]):
                    hits[p["record"]][p["category"]] += 1

        with activations.capture(model, [layer], on_layer):
            activations.run_batch(model, ids, attention)
    return [test if "failed" in test else {**test, **hits[r]} for r, test in enumerate(tests)]


def _shares(records: list[dict | None]) -> dict:
    """`n` and each category's share of hits; None stands for a neuron that scores zero."""
    n = len(records)
    scored = [r for r in records if r is not None]
    return {"n": n, **{c: sum(r[c] for r in scored) / (prompts.TESTS_PER_CATEGORY * n) if n else 0.0 for c in prompts.TEST_CATEGORIES}}


def summarize(records: list[dict], summary: dict, layers=None) -> dict:
    """The two scores fixed in the design: `primary` over all live neurons, `ac_matched` for the comparison with AC's figures.

    A live neuron without a scored record on its dominant pole counts as zero in `primary`. `ac_matched` covers live,
    positive-dominant neurons whose job is known and not "unclear", with failed ones counting as zero (R25); those whose job is
    unknown are left out and counted in `excluded_unknown_job`. `layers` limits both scores to those layers (R26).
    """
    by_nid: dict[int, dict] = {}
    for r in records:
        if r["nid"] in by_nid:
            raise ValueError(f"more than one record for neuron {r['nid']}")
        by_nid[r["nid"]] = r

    cfg = Config()
    primary, matched, unknown_job, out_of_context, fallback = [], [], 0, [], []
    for key in sorted(summary):
        found = re.fullmatch(r"L(\d+)\.live", key)
        if not found:
            continue
        layer = int(found[1])
        if layers is not None and layer not in layers:
            continue
        for idx in np.flatnonzero(summary[key]):
            r = by_nid.get(neuron_id(cfg, layer, int(idx)))
            pole = int(summary[f"L{layer}.dominant"][idx])
            own = r if r is not None and r.get("pole", pole) == pole else None     # a record for the other pole says nothing here
            scored = own if own is not None and "failed" not in own else None
            primary.append(scored)
            if (own or {}).get("tested_out_of_context"):
                out_of_context.append(scored)
            if (own or {}).get("test_model"):
                fallback.append(scored)
            if pole != 1:
                continue
            job = scored["job"] if scored is not None else (own or {}).get("job")
            if job is None:
                unknown_job += 1
            elif job != "unclear":
                matched.append(scored)
    return {"primary": _shares(primary), "ac_matched": {**_shares(matched), "excluded_unknown_job": unknown_job},
            "reasoning_context": _shares(out_of_context), "fallback_tests": _shares(fallback)}
