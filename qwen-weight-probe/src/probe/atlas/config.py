"""Settings for the neuron-atlas pipeline (Level 3)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from probe.weights import RESULTS


@dataclass(frozen=True)
class Config:
    root: Path = RESULTS / "07_atlas"
    layers: tuple = (14, 15, 16, 17, 18, 19, 20)
    layer_order: tuple = (15, 19, 18, 16, 17, 14, 20)
    n_neurons: int = 3584
    total_tokens: int = 50_000_000
    multilingual_tokens: int = 5_000_000
    thinking_tokens: int = 10_000_000
    doc_cap: int = 4096
    min_doc_tokens: int = 32
    shard_tokens: int = 1_000_000
    top_k: int = 50
    dn_k: int = 20
    random_k: int = 50
    per_doc_cap: int = 3
    window: int = 64
    llm_window: int = 24
    batch_docs: int = 32
    batch_tokens: int = 40_960
    checkpoint_s: int = 300
    explain_batch: int = 1      # R39: 20-neuron prompts were refused by Opus safeguards ([reasoning_extraction]); single neurons pass
    pilot_n: int = 200
    p50c_tokens: int = 5_000_000
    seed: int = 7
    claude_bin: str = "claude"
    effort: str = "medium"
    test_model: str = "claude-opus-5-5"     # test-writing calls only; explanations and ratings use llm.MODEL (R36)
    # R40: a single neuron whose tests test_model refuses is asked once more here. Kept out of prompt_parts on purpose:
    # it touches only refused neurons, so the approved pilot still describes the other 99.7%. "" turns the fallback off.
    test_fallback_model: str = "claude-haiku-5-5"
    prompt_version: str = "v2"     # v1's tests were all refused by Opus 5.5's safeguards (reasoning_extraction)

    @classmethod
    def tiny(cls, root: Path) -> "Config":
        return cls(root=root, total_tokens=200_000, multilingual_tokens=20_000,
                   thinking_tokens=20_000, shard_tokens=20_000, pilot_n=14,
                   p50c_tokens=20_000, checkpoint_s=1, explain_batch=20)   # tests exercise multi-neuron batches and splitting


def neuron_id(cfg: Config, layer: int, idx: int) -> int:
    return cfg.layers.index(layer) * cfg.n_neurons + idx


def neuron_name(layer: int, idx: int) -> str:
    return f"L{layer}.mlp.{idx}"
