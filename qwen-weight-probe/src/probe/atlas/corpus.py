"""Fixed text corpus for the atlas: sources, tokenizing, shuffled shards."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from probe.atlas.config import Config
from probe.atlas.store import atomic_write_bytes, atomic_write_json, read_json
from probe.weights import LOGS

os.environ.setdefault("RAYON_NUM_THREADS", "8")   # the tokenizer must not take every core

LANGS = ("cmn_Hani", "ind_Latn", "spa_Latn", "rus_Cyrl", "arb_Arab",
         "fra_Latn", "deu_Latn", "jpn_Jpan", "hin_Deva", "por_Latn")
ENGLISH_REPO = "HuggingFaceFW/fineweb"
ENGLISH_PATH = "data/CC-MAIN-2024-10/000_00000.parquet"
MULTI_REPO = "HuggingFaceFW/fineweb-2"
PRACTICE_FILES = ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl")
ENCODE_BATCH = 64


def read_web_docs(repo: str, path: str, revision: str) -> Iterator[str]:
    """Yield the `text` column of a remote parquet file, one row group at a time."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    with fs.open(f"hf://datasets/{repo}@{revision}/{path}", "rb") as f:
        pf = pq.ParquetFile(f)
        for i in range(pf.num_row_groups):
            yield from pf.read_row_group(i, columns=["text"]).column("text").to_pylist()


def _encode_batch(tok, texts: list[str]) -> list:
    # transformers tokenizers wrap the fast one as `backend_tokenizer`; plain tokenizers.Tokenizer has none.
    # Literal control text such as "<|im_start|>" in web docs becomes a special token here; that is
    # fine, because eligible_mask later excludes tok.all_special_ids.
    return getattr(tok, "backend_tokenizer", tok).encode_batch(texts, add_special_tokens=False)


def tokenize_until(docs: Iterable[str], tok, target: int, cfg: Config,
                   on_progress: Callable[[int], None] | None = None) -> list[list[int]]:
    """Token ids per doc, truncated to doc_cap; short docs dropped; stops once `target` is reached.

    `on_progress(running_token_count)` is called once per tokenized batch.
    """
    out: list[list[int]] = []
    total = 0
    it = iter(docs)
    while total < target:
        batch = [d for _, d in zip(range(ENCODE_BATCH), it)]
        if not batch:
            break
        for enc in _encode_batch(tok, batch):
            ids = list(enc.ids[:cfg.doc_cap])
            if len(ids) < cfg.min_doc_tokens:
                continue
            out.append(ids)
            total += len(ids)
            if total >= target:
                break
        if on_progress:
            on_progress(total)
    return out


def _jsonl_rows(path: Path) -> Iterator[dict]:
    # split on "\n" only: str.splitlines() also breaks on U+2028, which occurs inside JSON strings
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            yield json.loads(line)


def math_docs() -> list[str]:
    docs = []
    for name in PRACTICE_FILES:
        for row in _jsonl_rows(LOGS / "practice" / name):
            docs.append(row["problem"] + "\n\n" + (row.get("solution") or row["answer"]))
    return docs


def save_source(cfg: Config, name: str, docs: list[list[int]], meta: dict) -> None:
    ids = np.fromiter((t for d in docs for t in d), dtype=np.uint32)
    offsets = np.zeros(len(docs) + 1, dtype=np.int64)
    np.cumsum([len(d) for d in docs], out=offsets[1:])
    buf = io.BytesIO()
    np.savez(buf, ids=ids, offsets=offsets, meta=np.array(json.dumps(meta)))
    atomic_write_bytes(cfg.root / "raw" / f"{name}.npz", buf.getvalue())


def _load_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        return z["ids"], z["offsets"]


def load_source(cfg: Config, name: str) -> list[list[int]]:
    ids, off = _load_arrays(cfg.root / "raw" / f"{name}.npz")
    return [ids[a:b].tolist() for a, b in zip(off[:-1], off[1:])]


def _hf_revision(repo: str) -> str:
    from huggingface_hub import HfApi
    return HfApi().dataset_info(repo).sha


def build_sources(cfg: Config, tok, status, reader=read_web_docs,
                  revision_of: Callable[[str], str] = _hf_revision) -> None:
    """Tokenize each source once, in order math, languages, english. Finished sources are skipped."""
    raw = cfg.root / "raw"
    rev_path = raw / "revisions.json"
    revisions = read_json(rev_path, {})

    def revision(repo: str) -> str:
        if repo not in revisions:
            revisions[repo] = revision_of(repo)   # pinned on first use, then reused on resume
            atomic_write_json(rev_path, revisions)
        return revisions[repo]

    def exists(name: str) -> bool:
        return (raw / f"{name}.npz").exists()

    per_lang = cfg.multilingual_tokens // len(LANGS)
    done = [0]
    total = cfg.total_tokens - cfg.thinking_tokens

    def report() -> None:
        status.stage("corpus", done[0], total, "tokens")

    def progress(n: int) -> None:   # tokens of the source in progress, on top of finished sources
        status.stage("corpus", done[0] + n, total, "tokens")

    report()
    if not exists("math"):
        docs = tokenize_until(math_docs(), tok, 1 << 62, cfg,
                              on_progress=progress)
        save_source(cfg, "math", docs, {"source": "practice"})
    math_tokens = int(_load_arrays(raw / "math.npz")[1][-1])
    done[0] += math_tokens
    report()

    for lang in LANGS:
        if not exists(lang):
            path = f"data/{lang}/train/000_00000.parquet"
            docs = tokenize_until(reader(MULTI_REPO, path, revision(MULTI_REPO)), tok, per_lang, cfg,
                                  on_progress=progress)
            save_source(cfg, lang, docs, {"repo": MULTI_REPO, "path": path})
        done[0] += int(_load_arrays(raw / f"{lang}.npz")[1][-1])
        report()

    if not exists("english"):
        target = max(total - done[0], 0)
        docs = tokenize_until(reader(ENGLISH_REPO, ENGLISH_PATH, revision(ENGLISH_REPO)), tok, target, cfg,
                              on_progress=progress)
        save_source(cfg, "english", docs, {"repo": ENGLISH_REPO, "path": ENGLISH_PATH})
    done[0] += int(_load_arrays(raw / "english.npz")[1][-1])
    report()


def _shard_sha(ids: np.ndarray, offsets: np.ndarray) -> str:
    # hash the content, not the zip file, so the manifest hash is reproducible
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(ids).tobytes())
    h.update(np.ascontiguousarray(offsets).tobytes())
    return h.hexdigest()


def assemble(cfg: Config, status) -> dict:
    """Shuffle every raw source (including any thinking source) into fixed shards plus a manifest."""
    raw = cfg.root / "raw"
    names = sorted(p.stem for p in raw.glob("*.npz"))
    arrays, sources = [], {}
    for name in names:
        ids, off = _load_arrays(raw / f"{name}.npz")
        sources[name] = int(len(off) - 1)
        arrays.append((ids, off))
    docs = [(ids[a:b]) for ids, off in arrays for a, b in zip(off[:-1], off[1:])]
    order = np.random.default_rng(cfg.seed).permutation(len(docs))

    out = cfg.root / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("shard_*.npz"):
        old.unlink()
    shards, cur, cur_tokens, first, done = [], [], 0, 0, 0
    total = sum(len(d) for d in docs)

    def flush() -> None:
        nonlocal cur, cur_tokens, first
        if not cur:
            return
        ids = np.concatenate(cur).astype(np.uint32)
        offsets = np.zeros(len(cur) + 1, dtype=np.int64)
        np.cumsum([len(d) for d in cur], out=offsets[1:])
        buf = io.BytesIO()
        np.savez(buf, ids=ids, offsets=offsets)
        fname = f"shard_{len(shards):05d}.npz"
        atomic_write_bytes(out / fname, buf.getvalue())
        shards.append({"file": fname, "sha256": _shard_sha(ids, offsets), "n_docs": len(cur),
                       "n_tokens": int(cur_tokens), "first_doc": first})
        first += len(cur)
        cur, cur_tokens = [], 0

    for i in order:
        cur.append(docs[i])
        cur_tokens += len(docs[i])
        done += len(docs[i])
        if cur_tokens >= cfg.shard_tokens:
            flush()
            status.stage("assemble", done, total, "tokens")
    flush()
    status.stage("assemble", done, total, "tokens")

    manifest = {"shards": shards, "sources": sources,
                "revisions": read_json(raw / "revisions.json", {}),
                "hash": hashlib.sha256("".join(s["sha256"] for s in shards).encode()).hexdigest()}
    atomic_write_json(out / "manifest.json", manifest)
    return manifest


def iter_docs(cfg: Config) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (doc_id, ids) in manifest order, verifying each shard's checksum first."""
    out = cfg.root / "corpus"
    manifest = read_json(out / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(f"no corpus manifest in {out}")
    for sh in manifest["shards"]:
        try:
            ids, off = _load_arrays(out / sh["file"])
            ok = _shard_sha(ids, off) == sh["sha256"]
        except Exception:
            ok = False
        if not ok:
            raise ValueError(f"checksum mismatch for {sh['file']}")
        for k, (a, b) in enumerate(zip(off[:-1], off[1:])):
            yield sh["first_doc"] + k, ids[a:b]
