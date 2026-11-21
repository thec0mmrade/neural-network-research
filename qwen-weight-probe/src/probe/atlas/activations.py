"""What a "neuron activation" is for the atlas: the input of `down_proj` in an MLP layer, a = act_fn(gate) * up.

`capture` hooks the MLP input, takes gate/value from the model's own bf16 projections, forms a = act_fn(gate) * value in float32
(the real `down_proj` input before its bf16 rounding), and hands all three to a callback. It aborts the forward after the last
requested layer, so the layers above are never computed.
"""
from contextlib import contextmanager
from typing import Callable

import numpy as np
import torch
from torch import Tensor


class StopForward(Exception):
    """Raised by `capture` once the last requested layer's callback has run."""


def eligible_mask(ids: Tensor, attention: Tensor, excluded: Tensor, im_start_id: int) -> Tensor:
    """Bool [B, T]: True where a token's activation counts. False at index 0 of each document, at padding,
    at any `excluded` id, at `<|im_start|>` itself and at the role token right after it."""
    is_start = ids == im_start_id
    after_start = torch.zeros_like(is_start)
    after_start[:, 1:] = is_start[:, :-1]
    ok = attention.bool() & ~torch.isin(ids, excluded.to(ids.device)) & ~is_start & ~after_start
    ok[:, 0] = False
    return ok


def excluded_ids(tok) -> Tensor:
    """All tokenizer specials plus `<think>`, `</think>`, `<|im_start|>`, `<|im_end|>`."""
    extra = [tok.convert_tokens_to_ids(t) for t in ("<think>", "</think>", "<|im_start|>", "<|im_end|>")]
    return torch.tensor(sorted(set(tok.all_special_ids) | set(extra)), dtype=torch.long)


@contextmanager
def capture(model, layers, on_layer: Callable[[int, Tensor, Tensor, Tensor], None]):
    """Call `on_layer(layer, a, gate, value)` (float32, [B, T, I]) for each requested layer during a forward pass."""
    last = max(layers)
    handles = []

    def make(l):
        def hook(mlp, args):
            x = args[0]
            gate, value = mlp.gate_proj(x), mlp.up_proj(x)
            gate, value = gate.float(), value.float()
            # float32 product of the model's own bf16 gate/up outputs: equals the real down_proj input up to its bf16 rounding
            on_layer(l, mlp.act_fn(gate) * value, gate, value)
            if l == last:
                raise StopForward
        return hook

    try:
        for l in layers:
            handles.append(model.model.layers[l].mlp.register_forward_pre_hook(make(l)))
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def run_batch(model, ids: Tensor, attention: Tensor) -> None:
    """Forward the transformer body only (no lm_head); stops early when `capture` raises StopForward."""
    try:
        model.model(input_ids=ids, attention_mask=attention, use_cache=False)
    except StopForward:
        pass


def pad_right(docs: list[np.ndarray], pad_id: int, device) -> tuple[Tensor, Tensor]:
    """Right-pad token arrays to one [B, T] long tensor plus its 0/1 attention mask."""
    T = max(len(d) for d in docs)
    ids = torch.full((len(docs), T), pad_id, dtype=torch.long)
    att = torch.zeros(len(docs), T, dtype=torch.long)
    for i, d in enumerate(docs):
        ids[i, :len(d)] = torch.as_tensor(np.asarray(d), dtype=torch.long)
        att[i, :len(d)] = 1
    return ids.to(device), att.to(device)
