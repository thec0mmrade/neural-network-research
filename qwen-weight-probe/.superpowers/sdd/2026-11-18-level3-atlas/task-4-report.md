# Task 4 report: corpus sources and assembly

Status: DONE_WITH_CONCERNS

## Files
- Created: src/probe/atlas/corpus.py, tests/atlas/test_corpus.py
- Changed: tests/atlas/conftest.py (was empty; now has `tiny_cfg`, `fake_tok`, `status`)

## Built
read_web_docs, tokenize_until, math_docs, save_source/load_source, build_sources (with keyword `revision_of`, R4), assemble, iter_docs, as in the brief. RAYON_NUM_THREADS set via os.environ.setdefault("…","8") at import.

## Evidence
RED (corpus.py moved aside): `ImportError: cannot import name 'corpus' from 'probe.atlas'` , collection error.
GREEN: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas -v'` -> 62 passed (8 in test_corpus.py).
math_docs() on the real files returns 15458 rows, equal to the jsonl line counts (U+2028-safe split on "\n").
Tests: the brief's 4 plus roundtrip, multi-shard split, math_docs with U+2028, revision pinning and token budget in the resume test. No network used (injected reader and revision_of).

## Design notes
- Tokenizer API: `tok.encode_batch(list[str])` returning objects with `.ids` (HF tokenizers). FakeTok in conftest mimics it. The real tokenizer must be a tokenizers.Tokenizer (not a transformers one).
- Shard sha256 hashes ids+offsets bytes, not the zip file, because np.savez embeds timestamps and the manifest hash must be reproducible. iter_docs recomputes it; any load failure or mismatch raises ValueError("checksum mismatch ...").
- assemble uses every raw/*.npz (sorted), so a later thinking source is picked up automatically. Old shard files are deleted before rewriting.
- English target = total_tokens - thinking_tokens - (math + language tokens actually kept), floor 0. Languages' actual counts are used, so they can overshoot by up to one doc each.
- Math goes through tokenize_until too, so docs under 32 tokens are dropped and docs capped at 4096.
- Resume granularity is per source (file exists = done); a crash mid-source re-reads that source.
- Status: stage "corpus" (tokens) updated per source; stage "assemble" per shard.

## Self-review
Removed a leftover sloppy assertion I had written in the first draft. Not covered by tests: read_web_docs (needs network; untested against real HF), the default _hf_revision.

## Concerns
- read_web_docs untested live; `HfFileSystem.open("hf://datasets/repo@rev/path")` form is per the brief.
- The thinking source name/format is not defined here; it must be saved with save_source into raw/ before assemble.
- Progress inside one large source isn't reported (status moves per source); english takes the longest.

## Fix round 1 (review)
Changes in src/probe/atlas/corpus.py: `_encode_batch` helper (`getattr(tok, "backend_tokenizer", tok).encode_batch(texts, add_special_tokens=False)`) used by `tokenize_until`; optional `on_progress` callback called once per batch, wired in `build_sources` to `status.stage("corpus", finished + running, ...)` for math, languages and English; comment that literal control text becomes a special token and is excluded later by eligible_mask; dead `batch` assignment removed.
Tests added: real-tokenizer equality (local files), progress rises in tokenize_until, status rises within sources in build_sources. conftest FakeTok.encode_batch now accepts `add_special_tokens`.
Command: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas -q'`; result: all pass (see reply: 65 passed).
