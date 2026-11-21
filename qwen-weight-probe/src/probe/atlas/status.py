"""status.json writer and overall-progress estimate. No torch: the viewer imports this on CPU."""
from __future__ import annotations

import os
import time
from pathlib import Path

from probe.atlas.store import atomic_write_json, read_json

STAGES = ("corpus", "thinking", "assemble", "record", "pilot", "layers", "research", "report")
DEFAULT_HOURS = {"corpus": 1, "thinking": 7, "assemble": 0.2, "record": 10,
                 "pilot": 3, "layers": 150, "research": 3, "report": 0.2}
MEASURE_AFTER = 0.02     # trust the measured speed once a stage is this far along
WRITE_INTERVAL_S = 1.0


def overall_fraction(stages: dict[str, dict]) -> float:
    """Progress over all stages, each weighted by its expected hours.

    Expected hours are the default until a stage is MEASURE_AFTER done, then
    active_s / fraction (the stage's measured total).
    """
    total = done = 0.0
    for name in STAGES:
        st = stages.get(name) or {}
        frac = min(st.get("done", 0) / st["total"], 1.0) if st.get("total") else 0.0
        hours = DEFAULT_HOURS[name]
        if frac >= MEASURE_AFTER and st.get("active_s", 0) > 0:
            hours = st["active_s"] / 3600 / frac
        total += hours
        done += hours * frac
    return done / total if total else 0.0


class Status:
    def __init__(self, path: Path, clock=time):
        self.path = Path(path)
        self._clock = clock
        old = read_json(self.path)
        d = old if isinstance(old, dict) else {}
        now = clock.time()
        self.started_at = d.get("started_at", now)
        self.active_s = d.get("active_s", 0.0)
        self.paused_s = d.get("paused_s", 0.0)
        self.current_stage = d.get("current_stage")
        self.stages = d.get("stages", {})
        self.layers = d.get("layers", {})
        self.usage_ = d.get("usage", {"input_tokens": 0, "output_tokens": 0, "calls": 0})
        self.scores_ = d.get("scores", {"primary": {}, "ac_matched": {}})
        self.latest_ = d.get("latest", {})
        self.note_ = d.get("note")
        self.pause = None            # a restarted run is not paused
        self.call = None             # {"since": t} while a Claude call is in flight
        self.pid = d.get("pid")      # the writing run's process id; None once it has exited
        self._last = now
        self._last_write = float("-inf")

    def _tick(self) -> float:
        now = self._clock.time()
        dt = max(now - self._last, 0.0)
        self._last = now
        if self.pause:
            self.paused_s += dt
        else:
            self.active_s += dt
            if self.current_stage in self.stages:
                self.stages[self.current_stage]["active_s"] += dt
        return now

    def _write(self, now: float, force: bool = False) -> None:
        if not force and now - self._last_write < WRITE_INTERVAL_S:
            return
        self._last_write = now
        atomic_write_json(self.path, {
            "updated_at": now, "started_at": self.started_at,
            "active_s": self.active_s, "paused_s": self.paused_s,
            "pause": self.pause, "call": self.call, "pid": self.pid, "current_stage": self.current_stage,
            "stages": self.stages, "layers": self.layers, "usage": self.usage_,
            "scores": self.scores_, "latest": self.latest_, "note": self.note_,
        })

    def flush(self) -> None:
        self._write(self._tick(), force=True)

    def stage(self, name: str, done: int, total: int, unit: str) -> None:
        now = self._tick()
        changed = name != self.current_stage
        self.current_stage = name
        prev = self.stages.get(name, {})
        self.stages[name] = {"done": done, "total": total, "unit": unit,
                             "active_s": prev.get("active_s", 0.0)}
        self._write(now, force=changed)

    def started(self) -> None:
        """A run has started in this process: record its pid and clear any pause or call a dead run left behind (M4)."""
        now = self._tick()
        self.pid, self.pause, self.call = os.getpid(), None, None
        self._write(now, force=True)

    def closed(self) -> None:
        """The run is exiting: the final flush (R9), and the viewer must not keep showing it as running, paused or calling."""
        self._tick()
        self.pid, self.pause, self.call = None, None, None
        self.flush()

    def beat(self) -> None:
        """Heartbeat during a long blocking wait: rewrite the file so the viewer knows the run is alive (M3)."""
        self._write(self._tick(), force=True)

    def calling(self, since: float | None) -> None:
        """Mark a Claude call as in flight since `since`, or finished (None)."""
        now = self._tick()
        self.call = None if since is None else {"since": since}
        self._write(now, force=True)

    def paused(self, reason: str, resumes_at: float | None, since: float | None = None) -> None:
        now = self._tick()
        self.pause = {"reason": reason, "resumes_at": resumes_at}
        if since is not None:
            self.pause["since"] = since
        self._write(now, force=True)

    def resumed(self) -> None:
        now = self._tick()
        self.pause = None
        self._write(now, force=True)

    def layer(self, layer: int, state: str, done: int, total: int,
              mean_score: float | None) -> None:
        now = self._tick()
        self.layers[str(layer)] = {"state": state, "done": done, "total": total,
                                   "mean_score": mean_score}
        self._write(now)

    def usage(self, input_tokens: int, output_tokens: int) -> None:
        now = self._tick()
        self.usage_["input_tokens"] += input_tokens
        self.usage_["output_tokens"] += output_tokens
        self.usage_["calls"] += 1
        self._write(now)

    def scores(self, primary: dict, ac_matched: dict) -> None:
        now = self._tick()
        self.scores_ = {"primary": primary, "ac_matched": ac_matched}
        self._write(now)

    def latest(self, neuron: dict) -> None:
        now = self._tick()
        self.latest_ = neuron
        self._write(now)

    def note(self, text: str) -> None:
        now = self._tick()
        self.note_ = text
        self._write(now)
