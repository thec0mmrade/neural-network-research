### Task 4: Corpus sources and assembly

**Files:**
- Create: `src/probe/atlas/corpus.py`
- Test: `tests/atlas/test_corpus.py`

**Interfaces:**
- Consumes: `Config`, `store`, `Status`, `probe.weights.LOGS`
- Produces:
  - `read_web_docs(repo: str, path: str, revision: str) -> Iterator[str]`: opens `hf://datasets/{repo}@{revision}/{path}` with `HfFileSystem`, reads one row group at a time with `pyarrow.parquet.ParquetFile`, column `text`.
  - `tokenize_until(docs: Iterable[str], tok, target: int, cfg) -> list[list[int]]`: truncates to `doc_cap`, drops documents under `min_doc_tokens`, stops at `target` tokens.
  - `math_docs() -> list[str]`: every row of `gsm8k_train`, `math_train`, `aime` in `LOGS/"practice"`, as `problem + "\n\n" + (solution or answer)`.
  - `save_source(cfg, name: str, docs: list[list[int]], meta: dict) -> None` and `load_source(cfg, name) -> list[list[int]]`: `root/raw/<name>.npz` with `ids` (uint32) and `offsets` (int64).
  - `build_sources(cfg, tok, status, reader=read_web_docs) -> None`: resumable per source, in the order `math`, each language, `english`.
  - `assemble(cfg, status) -> dict`: shuffles all documents with `default_rng(cfg.seed)`, writes `root/corpus/shard_00000.npz` and so on, then `root/corpus/manifest.json` with `{"shards": [{"file","sha256","n_docs","n_tokens","first_doc"}], "sources", "revisions", "hash"}`.
  - `iter_docs(cfg) -> Iterator[tuple[int, np.ndarray]]`, yielding `(doc_id, ids)` in manifest order after verifying checksums.

- [ ] **Step 1: Write the failing tests**

```python
def test_tokenize_truncates_drops_short_and_stops(tiny_cfg, fake_tok):
    docs = tokenize_until(["a " * 10, "b " * 9000, "c " * 500], fake_tok, target=4300, cfg=tiny_cfg)
    assert [len(d) for d in docs] == [4096, 500]

def test_assemble_is_deterministic_and_complete(tiny_cfg, status):
    save_source(tiny_cfg, "a", [[1] * 40] * 30, {}); save_source(tiny_cfg, "b", [[2] * 40] * 30, {})
    m1 = assemble(tiny_cfg, status); docs1 = [d.tolist() for _, d in iter_docs(tiny_cfg)]
    m2 = assemble(tiny_cfg, status)
    assert m1["hash"] == m2["hash"] and len(docs1) == 60
    assert {d[0] for d in docs1[:20]} == {1, 2}      # the sources are mixed early

def test_iter_docs_rejects_damaged_shard(tiny_cfg, status):
    save_source(tiny_cfg, "a", [[1] * 40] * 30, {}); assemble(tiny_cfg, status)
    shard = next((tiny_cfg.root / "corpus").glob("shard_*.npz")); shard.write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        list(iter_docs(tiny_cfg))

def test_build_sources_resumes_without_rereading(tiny_cfg, fake_tok, status):
    calls = []
    reader = counting_reader(calls)
    build_sources(tiny_cfg, fake_tok, status, reader=reader); n = len(calls)
    build_sources(tiny_cfg, fake_tok, status, reader=reader)
    assert len(calls) == n
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_corpus.py -v'`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `corpus.py`**

File paths: English `data/CC-MAIN-2024-10/000_00000.parquet` in `HuggingFaceFW/fineweb`; multilingual `data/{lang}/train/000_00000.parquet` in `HuggingFaceFW/fineweb-2`, with the 10 languages from Global Constraints. Pin each dataset's commit sha from `HfApi().dataset_info(repo).sha` on first use, and record it in `root/raw/revisions.json`. English target: `total_tokens − math tokens − multilingual tokens − thinking_tokens`.

- [ ] **Step 4: Run the tests and confirm they pass**

Expected: 4 passed

---

