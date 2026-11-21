import dataclasses
import json

import pytest

from probe.atlas import corpus
from probe.atlas.corpus import (LANGS, assemble, build_sources, iter_docs, load_source,
                                math_docs, save_source, tokenize_until)
from probe.atlas.store import read_json


def test_tokenize_truncates_drops_short_and_stops(tiny_cfg, fake_tok):
    docs = tokenize_until(["a " * 10, "b " * 9000, "c " * 500], fake_tok, target=4300, cfg=tiny_cfg)
    assert [len(d) for d in docs] == [4096, 500]


def test_save_load_roundtrip(tiny_cfg):
    docs = [[1, 2, 3], [4], [5, 6]]
    save_source(tiny_cfg, "x", docs, {"k": 1})
    assert load_source(tiny_cfg, "x") == docs


def test_assemble_is_deterministic_and_complete(tiny_cfg, status):
    save_source(tiny_cfg, "a", [[1] * 40] * 30, {}); save_source(tiny_cfg, "b", [[2] * 40] * 30, {})
    m1 = assemble(tiny_cfg, status); docs1 = [d.tolist() for _, d in iter_docs(tiny_cfg)]
    m2 = assemble(tiny_cfg, status)
    assert m1["hash"] == m2["hash"] and len(docs1) == 60
    assert {d[0] for d in docs1[:20]} == {1, 2}      # the sources are mixed early
    assert [i for i, _ in iter_docs(tiny_cfg)] == list(range(60))


def test_assemble_splits_into_shards(tiny_cfg, status):
    cfg = dataclasses.replace(tiny_cfg, shard_tokens=400)
    save_source(cfg, "a", [[1] * 40] * 30, {})
    m = assemble(cfg, status)
    assert [s["n_docs"] for s in m["shards"]] == [10, 10, 10]
    assert [s["first_doc"] for s in m["shards"]] == [0, 10, 20]
    assert len(list(iter_docs(cfg))) == 30


def test_iter_docs_rejects_damaged_shard(tiny_cfg, status):
    save_source(tiny_cfg, "a", [[1] * 40] * 30, {}); assemble(tiny_cfg, status)
    shard = next((tiny_cfg.root / "corpus").glob("shard_*.npz")); shard.write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        list(iter_docs(tiny_cfg))


def test_math_docs_reads_all_rows_and_survives_u2028(tmp_path, monkeypatch):
    p = tmp_path / "practice"; p.mkdir()
    rows = {"gsm8k_train.jsonl": {"problem": "p1", "answer": "a1", "solution": "s1"},
            "math_train.jsonl": {"problem": "p 2", "answer": "a2"},
            "aime.jsonl": {"problem": "p3", "answer": "a3"}}
    for name, r in rows.items():
        (p / name).write_text(json.dumps(r, ensure_ascii=False) + "\n")
    monkeypatch.setattr(corpus, "LOGS", tmp_path)
    assert math_docs() == ["p1\n\ns1", "p 2\n\na2", "p3\n\na3"]


def test_build_sources_resumes_without_rereading(tiny_cfg, fake_tok, status, monkeypatch):
    monkeypatch.setattr(corpus, "math_docs", lambda: ["m " * 50] * 10)
    calls = []

    def reader(repo, path, revision):
        calls.append((repo, path, revision))
        return iter(["w " * 60] * 5000)

    kw = dict(reader=reader, revision_of=lambda repo: "sha-" + repo)
    build_sources(tiny_cfg, fake_tok, status, **kw); n = len(calls)
    assert n == len(LANGS) + 1
    build_sources(tiny_cfg, fake_tok, status, **kw)
    assert len(calls) == n
    assert read_json(tiny_cfg.root / "raw" / "revisions.json") == {
        "HuggingFaceFW/fineweb": "sha-HuggingFaceFW/fineweb",
        "HuggingFaceFW/fineweb-2": "sha-HuggingFaceFW/fineweb-2"}
    assert all(c[2].startswith("sha-") for c in calls)
    # english fills total - math - languages - thinking, overshooting by less than one doc
    langs = sum(sum(map(len, load_source(tiny_cfg, l))) for l in LANGS)
    english = sum(map(len, load_source(tiny_cfg, "english")))
    assert 200_000 - 20_000 - 500 - langs <= english < 200_000 - 20_000 - 500 - langs + 60


def test_tokenize_matches_real_tokenizer(tiny_cfg):
    from transformers import AutoTokenizer
    from probe.weights import snapshot_dir
    tok = AutoTokenizer.from_pretrained(snapshot_dir(), local_files_only=True)
    texts = ["The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun.",
             "def add(a, b):\n    return a + b  # adds two numbers together, nothing else here"]
    cfg = dataclasses.replace(tiny_cfg, min_doc_tokens=1)
    got = tokenize_until(texts, tok, target=10**6, cfg=cfg)
    assert got == [tok(t, add_special_tokens=False).input_ids for t in texts]


def test_progress_rises_during_a_source(tiny_cfg, fake_tok):
    seen = []
    tokenize_until(["w " * 60] * 200, fake_tok, target=10000, cfg=tiny_cfg, on_progress=seen.append)
    assert len(seen) > 1 and seen == sorted(seen) and len(set(seen)) == len(seen)


def test_build_sources_status_rises_within_source(tiny_cfg, fake_tok, monkeypatch):
    monkeypatch.setattr(corpus, "math_docs", lambda: ["m " * 50] * 10)
    stages = []

    class S:
        def stage(self, name, done, total, unit):
            stages.append(done)

    build_sources(tiny_cfg, fake_tok, S(), reader=lambda *a: iter(["w " * 60] * 5000),
                  revision_of=lambda r: "x")
    assert stages == sorted(stages) and len(set(stages)) > 2 * len(LANGS)
