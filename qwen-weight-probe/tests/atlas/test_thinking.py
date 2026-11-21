from dataclasses import replace

from probe.atlas.corpus import load_source
from probe.atlas.store import atomic_write_json
from probe.atlas.thinking import generate_thinking


def fake_prompts(tok, problems):
    return [f"prompt {p}" for p in problems]       # two words each, so the fake tokenizer gives 2 ids


def test_resume_skips_finished_batches_and_stops_at_target(tiny_cfg, fake_tok, status):
    calls = []

    def gen(model, tok, problems, max_new_tokens, seed=0, batch_size=32):
        calls.append(seed)
        assert max_new_tokens == 4000 and batch_size == 32 and len(problems) == 32
        return [{"ids": [5] * 1000, "finished": True} for _ in problems], 400.0

    cfg = replace(tiny_cfg, thinking_tokens=64_000, doc_cap=2000)
    (cfg.root / "raw/thinking").mkdir(parents=True)
    atomic_write_json(cfg.root / "raw/thinking/batch_00000.json", {"docs": [[5] * 1000] * 32, "seed": 0})
    generate_thinking(cfg, None, fake_tok, status, gen=gen,
                      prompts=fake_prompts, problems=[str(i) for i in range(1000)])
    assert calls == [32]                  # batch 0 was reused, and 2 batches reach the target
    assert sum(map(len, load_source(cfg, "thinking"))) >= 64_000
    assert (cfg.root / "raw/thinking/batch_00001.json").exists()
    assert not (cfg.root / "raw/thinking/batch_00002.json").exists()


def test_doc_is_prompt_ids_plus_generated_capped(tiny_cfg, fake_tok, status):
    def gen(model, tok, problems, max_new_tokens, seed=0, batch_size=32):
        return [{"ids": [9] * 10, "finished": False} for _ in problems], 1.0

    cfg = replace(tiny_cfg, thinking_tokens=1, doc_cap=8)
    generate_thinking(cfg, None, fake_tok, status, gen=gen,
                      prompts=fake_prompts, problems=["a"] * 32)
    docs = load_source(cfg, "thinking")
    assert len(docs) == 32 and docs[0] == [6, 1] + [9] * 6     # "prompt" -> 6, "a" -> 1; capped at 8
    assert status.stages["thinking"]["unit"] == "tokens"
