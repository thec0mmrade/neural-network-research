import json
from types import SimpleNamespace

import pytest

from probe.atlas.config import Config
from probe.atlas.status import Status


class FakeTok:
    """Whitespace tokenizer with the tokenizers `encode_batch` API; the id is the word's length."""

    def encode_batch(self, texts, add_special_tokens=True):
        return [SimpleNamespace(ids=[len(w) for w in t.split()]) for t in texts]

    def __call__(self, text, add_special_tokens=True):
        return SimpleNamespace(input_ids=self.encode_batch([text], add_special_tokens)[0].ids)


@pytest.fixture
def tiny_cfg(tmp_path):
    return Config.tiny(tmp_path / "atlas")


@pytest.fixture
def fake_tok():
    return FakeTok()


@pytest.fixture
def status(tmp_path):
    return Status(tmp_path / "status.json")


@pytest.fixture(scope="module")
def base_model():
    """Released after each module: the end-to-end rehearsal's own processes need the GPU to themselves."""
    import gc

    import torch

    from probe.model import load_model  # lazy: keeps non-GPU tests fast

    loaded = load_model()
    yield loaded
    del loaded
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def fake_cfg(tmp_path, monkeypatch):
    """fake_cfg(mode) -> Config whose claude_bin is a wrapper running fake_claude.py in that mode.

    claude's working directory moves into tmp_path, so tests never touch the real ~/.cache/qwen-atlas/llm_cwd."""
    import dataclasses
    import stat
    import sys
    from pathlib import Path

    import probe.atlas.llm as llm

    monkeypatch.setattr(llm, "LLM_CWD", tmp_path / "llm_cwd")
    fake = Path(__file__).parent / "fixtures" / "fake_claude.py"
    count = [0]

    def make(mode):
        count[0] += 1
        state = tmp_path / f"fake_state_{count[0]}"
        wrapper = tmp_path / f"fake_claude_{count[0]}.sh"
        wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} {fake} {mode} {state} "$@"\n')
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
        cfg = dataclasses.replace(Config.tiny(tmp_path / "atlas"), claude_bin=str(wrapper))
        return cfg

    return make
