"""Run items through a costly `work` function in chunks, saving every validated chunk so that nothing is paid for twice."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Callable

from probe.atlas.llm import LLMFailed, Refused, RunStopped, TransientFailure
from probe.atlas.store import atomic_write_json, read_json

# Outages no longer reach here (call_with_wait waits them out and raises RunStopped after 12 h, I2), so every
# LLMFailed seen here is a content failure. This many single items failing in a row is still treated as something
# systematic (R20): the run stops and stores none of them, so a rerun tries them again.
MAX_FAILURES_IN_A_ROW = 5
# Mass refusal (R37): once this many single items are refused and they are at least half of the items this `process` call
# has settled, the refusals are systematic (v1's test prompt was refused by Opus 5.5's safeguards for every neuron). The
# run stops with RunStopped and this call's stored refusal records are removed, so a rerun asks for them again.
MASS_REFUSALS = 20
MASS_REFUSAL_SHARE = 0.5
_DETAILS_RE = re.compile(r"Details:\s*`([^`]*)`")


def _matches(records, keys: list, key: str) -> bool:
    """True if `records` holds exactly one dict per key."""
    if not isinstance(records, list) or not all(isinstance(r, dict) and key in r for r in records):
        return False
    return Counter(repr(r[key]) for r in records) == Counter(repr(k) for k in keys)


def _details(reason: str) -> str:
    """The `Details: `[tag]`` part of a refusal, or its first line when there is none."""
    m = _DETAILS_RE.search(reason)
    return m.group(1) if m else (reason.strip().splitlines() or ["(no reason given)"])[0][:200]


class _Run:
    def __init__(self, out: Path, work: Callable, key: str, status=None):
        self.out, self.work, self.key, self.status = out, work, key, status
        self.unsaved: list[tuple[Path, dict]] = []   # single-item failures that may still turn out to be an outage
        self.refused: list[tuple[Path, str]] = []    # (stored file, reason) of each single-item refusal in this call
        self.settled = 0                             # items this call resolved through `work` (stored batches excluded)
        # R39: answered items from stored batches of any size, so a smaller batch size never pays for them again
        self.known: dict[str, dict] = {}
        for stored in sorted(Path(out).glob("batch_*.json")) if Path(out).is_dir() else []:
            rows = read_json(stored)
            if isinstance(rows, list):
                self.known.update((repr(r[key]), r) for r in rows if isinstance(r, dict) and key in r and "failed" not in r)

    def refuse(self, path: Path, reason: str) -> None:
        """Count a single-item refusal before it is stored; on mass refusal remove this call's refusals and stop (R37)."""
        self.refused.append((path, reason))
        n = len(self.refused)
        if n >= MASS_REFUSALS and n >= MASS_REFUSAL_SHARE * self.settled:
            for stored, _ in self.refused:
                stored.unlink(missing_ok=True)       # a rerun asks for them again: no neuron stays failed by this stop
            first = self.refused[0][1]
            message = (f"Claude refused {n} of the {self.settled} items processed so far in {self.out} "
                       f"(first refusal Details: `{_details(first)}`). Mass refusal means the prompt or the model "
                       "must change: set a new prompt_version (and test_model if needed) in config.py, then rerun; "
                       "no refused item was marked failed")
            if self.status is not None:
                self.status.note(f"Run stopped: {message}")
                self.status.flush()
            raise RunStopped(message)

    def settle(self) -> None:
        """The service answered, so the failures held back were the items' own: store them."""
        for path, record in self.unsaved:
            atomic_write_json(path, [record])
        self.unsaved = []

    def solve(self, chunk: list[dict]) -> list[dict]:
        keys = [x[self.key] for x in chunk]
        name = f"{keys[0]}_{len(chunk)}"
        path, split_mark = self.out / f"batch_{name}.json", self.out / f"split_{name}.json"

        def split() -> list[dict]:
            atomic_write_json(split_mark, keys)
            mid = len(chunk) // 2
            return self.solve(chunk[:mid]) + self.solve(chunk[mid:])

        stored = read_json(path)
        if _matches(stored, keys, self.key):
            return stored
        if all(repr(k) in self.known for k in keys):
            return [self.known[repr(k)] for k in keys]
        if read_json(split_mark) == keys:   # split by an earlier run: asking for the whole chunk again would pay for it twice
            return split()
        try:
            records = self.work(chunk)
            if not _matches(records, keys, self.key):
                raise LLMFailed("response ids differ from the ids sent")
            self.settled += len(chunk)
        except TransientFailure:
            raise           # an outage says nothing about these items: never split or mark them (I2)
        except Refused as e:
            if len(chunk) > 1:
                return split()
            self.settled += 1
            self.refuse(path, e.reason)
            records = [{self.key: keys[0], "failed": e.reason}]
        except LLMFailed as e:
            if len(chunk) > 1:
                return split()
            record = {self.key: keys[0], "failed": e.reason}
            self.unsaved.append((path, record))
            self.settled += 1
            if len(self.unsaved) >= MAX_FAILURES_IN_A_ROW:
                raise LLMFailed(f"{MAX_FAILURES_IN_A_ROW} items failed in a row, last: {e.reason}") from e
            return [record]
        self.settle()
        by_key = {repr(r[self.key]): r for r in records}
        ordered = [by_key[repr(k)] for k in keys]
        atomic_write_json(path, ordered)
        return ordered


def process(items: list[dict], size: int, out: Path, work: Callable[[list[dict]], list[dict]], key: str = "nid",
            status=None) -> list[dict]:
    """Records for `items`, in input order, calling `work` on chunks of `size` and storing each as `out/batch_<first-key>_<n>.json`.

    `work` must return one record per input key. On a content `LLMFailed`, `Refused` or a key mismatch the chunk is halved and retried;
    a single item that still fails is stored as `{key: ..., "failed": reason}`. After MAX_FAILURES_IN_A_ROW single items fail
    with anything but `Refused`, this raises `LLMFailed` and stores none of them, so a rerun tries them again. `LimitReached`,
    `TransientFailure` and `RunStopped` pass through, saving nothing and writing no split mark.

    Once MASS_REFUSALS single-item refusals make up at least MASS_REFUSAL_SHARE of the items this call settled, it raises
    RunStopped (noted in `status` when given), first removing the refusal records this call stored, so none stays failed (R37).
    """
    run = _Run(Path(out), work, key, status)
    results = [r for start in range(0, len(items), size) for r in run.solve(items[start:start + size])]
    run.settle()
    return results
