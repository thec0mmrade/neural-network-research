import pytest

from probe.atlas.status import DEFAULT_HOURS, STAGES, Status, overall_fraction
from probe.atlas.store import read_json


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def advance(self, s):
        self.now += s


def test_overall_weights_by_expected_hours():
    s = {n: {"done": 0, "total": 1, "active_s": 0} for n in STAGES}
    s["corpus"] = {"done": 1, "total": 1, "active_s": 3600}
    assert overall_fraction(s) == pytest.approx(1 / sum(DEFAULT_HOURS.values()))


def test_measured_speed_replaces_default_after_2_percent():
    s = {n: {"done": 0, "total": 1, "active_s": 0} for n in STAGES}
    s["layers"] = {"done": 50, "total": 100, "active_s": 3600 * 10}   # measured: 20 h in all
    total = sum(DEFAULT_HOURS.values()) - 150 + 20
    assert overall_fraction(s) == pytest.approx(10 / total)


def test_pause_time_is_not_active_time(tmp_path):
    clock = FakeClock(); st = Status(tmp_path / "s.json", clock=clock)
    st.stage("layers", 1, 10, "neurons"); clock.advance(10)
    st.paused("Max limit", clock.now + 100); clock.advance(100); st.resumed(); clock.advance(5)
    st.stage("layers", 2, 10, "neurons")
    d = read_json(tmp_path / "s.json")
    assert d["paused_s"] == pytest.approx(100) and d["active_s"] == pytest.approx(15)


def test_below_2_percent_uses_default_hours():
    s = {n: {"done": 0, "total": 1000, "active_s": 0} for n in STAGES}
    s["layers"] = {"done": 10, "total": 1000, "active_s": 3600}   # 1%: measured rate is ignored
    assert overall_fraction(s) == pytest.approx(0.01 * 150 / sum(DEFAULT_HOURS.values()))


def test_writes_throttled_except_pause_and_stage_change(tmp_path):
    clock = FakeClock(); st = Status(tmp_path / "s.json", clock=clock)
    st.stage("layers", 1, 10, "neurons")
    clock.advance(0.2); st.stage("layers", 2, 10, "neurons")      # throttled
    assert read_json(st.path)["stages"]["layers"]["done"] == 1
    st.stage("record", 0, 5, "docs")                               # stage change writes
    assert read_json(st.path)["current_stage"] == "record"
    st.note("hi")                                                  # throttled
    assert read_json(st.path)["note"] is None
    st.paused("limit", None)                                       # pause writes
    d = read_json(st.path)
    assert d["note"] == "hi" and d["pause"] == {"reason": "limit", "resumes_at": None}
    clock.advance(1.5); st.resumed()
    assert read_json(st.path)["pause"] is None
    clock.advance(1.5); st.note("later")
    assert read_json(st.path)["note"] == "later"                   # throttle window passed


def test_fields_and_restart_reload(tmp_path):
    clock = FakeClock(); st = Status(tmp_path / "s.json", clock=clock)
    st.stage("layers", 3, 10, "neurons")
    st.layer(15, "running", 3, 10, None)
    st.usage(100, 20); st.usage(5, 1)
    st.scores({"a": 1}, {"b": 2}); st.latest({"id": 7}); st.note("n")
    clock.advance(10); st.stage("layers", 4, 10, "neurons"); st.flush()
    d = read_json(st.path)
    assert d["layers"]["15"] == {"state": "running", "done": 3, "total": 10, "mean_score": None}
    assert d["usage"] == {"input_tokens": 105, "output_tokens": 21, "calls": 2}
    assert d["scores"] == {"primary": {"a": 1}, "ac_matched": {"b": 2}}
    assert d["latest"] == {"id": 7} and d["note"] == "n"
    assert d["stages"]["layers"]["active_s"] == pytest.approx(10)
    clock.advance(500)
    st2 = Status(tmp_path / "s.json", clock=clock)                 # restart
    st2.stage("layers", 5, 10, "neurons"); st2.flush()
    d2 = read_json(st2.path)
    assert d2["started_at"] == d["started_at"] and d2["usage"]["calls"] == 2
    assert d2["active_s"] == pytest.approx(10) and d2["stages"]["layers"]["done"] == 5


# ---- final review fixes (M3, M4) ----
import os


def test_started_records_pid_and_clears_a_pause_left_by_a_dead_run(tmp_path):
    clock = FakeClock(); st = Status(tmp_path / "s.json", clock=clock)
    st.paused("network: down", clock.now + 60, since=clock.now); st.calling(clock.now)
    assert read_json(st.path)["pause"]["since"] == clock.now
    restarted = Status(tmp_path / "s.json", clock=clock)
    restarted.started()
    d = read_json(st.path)
    assert d["pause"] is None and d["call"] is None and d["pid"] == os.getpid()


def test_closed_clears_pid_pause_and_call(tmp_path):
    clock = FakeClock(); st = Status(tmp_path / "s.json", clock=clock)
    st.started(); st.paused("limit", None); st.calling(clock.now)
    st.closed()
    d = read_json(st.path)
    assert d["pid"] is None and d["pause"] is None and d["call"] is None


def test_beat_and_calling_write_at_once(tmp_path):
    clock = FakeClock(); st = Status(tmp_path / "s.json", clock=clock)
    st.stage("layers", 1, 10, "neurons")
    clock.advance(0.1); st.calling(clock.now)                       # not throttled
    assert read_json(st.path)["call"] == {"since": clock.now}
    clock.advance(0.1); st.beat()
    assert read_json(st.path)["updated_at"] == clock.now
