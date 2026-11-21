"""Lazy, numpy-only access to the pinned Qwen3.5-0.8B checkpoint and the rebuilt Sparks edits.

bfloat16 tensors are memory-mapped as uint16 bits and widened to float32 on access, so the
1.7 GB file is never loaded whole and no torch is needed.
"""
import json
import os
import struct
from functools import cached_property
from pathlib import Path

import numpy as np

REPO = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
LOGS = ROOT / "external" / "lm_understand-logs" / "exp8_math"
LM = "model.language_model."
N_LAYERS = 24
FULL_ATTN = {3, 7, 11, 15, 19, 23}  # layer_types in config: every 4th layer is full attention

_DT = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16}


def snapshot_dir() -> Path:
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return hf / "hub" / f"models--{REPO.replace('/', '--')}" / "snapshots" / REVISION


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


class Checkpoint:
    def __init__(self, path: Path | None = None):
        if path is None:
            path = next(snapshot_dir().glob("model*.safetensors"))
        self.path = Path(path)
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.offset = 8 + n
        self.header.pop("__metadata__", None)

    @cached_property
    def _mm(self):
        return np.memmap(self.path, dtype=np.uint8, mode="r")

    def names(self, prefix: str = LM):
        return [k for k in self.header if k.startswith(prefix)]

    def info(self, name):
        return self.header[name]

    def get(self, name: str) -> np.ndarray:
        """Tensor as float32 (a copy; edits never touch the file)."""
        h = self.header[name]
        a, b = h["data_offsets"]
        raw = self._mm[self.offset + a: self.offset + b].view(_DT[h["dtype"]]).reshape(h["shape"])
        return bf16_to_f32(raw) if h["dtype"] == "BF16" else raw.astype(np.float32)


def layer_tensor(i: int, kind: str) -> str:
    """kind: 'down' | 'up' | 'gate' | 'out' (attention/linear-attention output projection) | 'in_norm' | 'post_norm'."""
    p = f"{LM}layers.{i}."
    return {
        "down": p + "mlp.down_proj.weight",
        "up": p + "mlp.up_proj.weight",
        "gate": p + "mlp.gate_proj.weight",
        "out": p + ("self_attn.o_proj.weight" if i in FULL_ATTN else "linear_attn.out_proj.weight"),
        "in_norm": p + "input_layernorm.weight",
        "post_norm": p + "post_attention_layernorm.weight",
    }[kind]


def edited(run: str, name: str) -> np.ndarray | None:
    """An edited tensor from results/edits/<run>/ (written by the logs' apply_edit.py), or None if untouched."""
    p = RESULTS / "edits" / run / f"{name}.npy"
    return np.load(p) if p.exists() else None


def save_json(obj, rel: str):
    p = RESULTS / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=float))
    return p
