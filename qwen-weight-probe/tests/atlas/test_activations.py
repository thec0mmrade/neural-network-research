import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from probe.atlas.activations import (StopForward, capture, eligible_mask, excluded_ids, pad_right,
                                     run_batch)


def test_eligible_excludes_first_special_role_and_padding():
    ids = torch.tensor([[9, 50, 7, 51, 8, 0]]); att = torch.tensor([[1, 1, 1, 1, 1, 0]])
    m = eligible_mask(ids, att, excluded=torch.tensor([50]), im_start_id=51)
    assert m.tolist() == [[False, False, True, False, False, False]]


def test_eligible_im_start_at_end_and_second_row():
    ids = torch.tensor([[5, 6, 51], [5, 6, 7]]); att = torch.ones(2, 3, dtype=torch.long)
    m = eligible_mask(ids, att, excluded=torch.tensor([], dtype=torch.long), im_start_id=51)
    assert m.tolist() == [[False, True, False], [False, True, True]]


def test_pad_right():
    ids, att = pad_right([np.array([1, 2, 3]), np.array([4])], pad_id=99, device="cpu")
    assert ids.tolist() == [[1, 2, 3], [4, 99, 99]]
    assert att.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert ids.dtype == torch.long


def test_excluded_ids_has_specials_and_chat_tokens():
    class Tok:
        all_special_ids = [1, 2]
        def convert_tokens_to_ids(self, t):
            return {"<think>": 10, "</think>": 11, "<|im_start|>": 12, "<|im_end|>": 2}[t]
    assert sorted(excluded_ids(Tok()).tolist()) == [1, 2, 10, 11, 12]


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _Layer(nn.Module):
    def __init__(self, calls, i):
        super().__init__()
        self.mlp = _MLP()
        self.calls, self.i = calls, i

    def forward(self, x):
        self.calls.append(self.i)
        return x + self.mlp(x)


class _Tiny(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.calls = []
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_Layer(self.calls, i) for i in range(n))

    def forward_model(self, input_ids=None, attention_mask=None, use_cache=False):
        x = torch.randn(*input_ids.shape, 4)
        for l in self.model.layers:
            x = l(x)
        return x


def _tiny():
    t = _Tiny()
    # run_batch calls model.model(...), so expose the stack as a callable
    t.model.forward = t.forward_model
    return t


def _n_hooks(m):
    return sum(len(x._forward_pre_hooks) for x in m.modules())


def test_capture_stops_after_last_layer_and_removes_hooks():
    t = _tiny(); seen = []
    ids = torch.zeros(2, 3, dtype=torch.long)
    with capture(t, [1, 2], lambda l, a, g, v: seen.append((l, a.shape, a.dtype))):
        run_batch(t, ids, torch.ones_like(ids))
    assert [s[0] for s in seen] == [1, 2]
    assert all(s[1] == (2, 3, 6) and s[2] == torch.float32 for s in seen)
    assert t.calls == [0, 1, 2]  # layer 3 never ran
    assert _n_hooks(t) == 0


def test_capture_removes_hooks_on_exception():
    t = _tiny()
    with pytest.raises(RuntimeError):
        with capture(t, [0], lambda *a: None):
            assert _n_hooks(t) == 1
            raise RuntimeError("boom")
    assert _n_hooks(t) == 0


def test_stop_forward_propagates_without_run_batch():
    t = _tiny(); ids = torch.zeros(1, 2, dtype=torch.long)
    with capture(t, [0], lambda *a: None):
        with pytest.raises(StopForward):
            t.model(input_ids=ids)


@pytest.mark.gpu
def test_capture_matches_down_proj_input(base_model):
    model, tok = base_model; got, want = {}, {}
    h = model.model.layers[15].mlp.down_proj.register_forward_pre_hook(lambda m, i: want.update(a=i[0].float().clone()))
    ids, att = pad_right([np.array(tok("The sum of two numbers is 12.").input_ids)], tok.pad_token_id, model.device)
    model.model(input_ids=ids, attention_mask=att, use_cache=False); h.remove()
    with capture(model, [15], lambda l, a, g, v: got.update(a=a.clone(), g=g.clone(), v=v.clone())):
        run_batch(model, ids, att)
    assert got["a"].shape == (1, ids.shape[1], 3584) and got["a"].dtype == torch.float32
    assert torch.allclose(got["a"], want["a"], rtol=1e-2, atol=1e-2)
    assert torch.allclose(got["a"], F.silu(got["g"]) * got["v"], rtol=1e-3, atol=1e-3)
