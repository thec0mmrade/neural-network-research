import io
import json
import subprocess
import sys

import pytest
from rich.console import Console

from probe.atlas.status import STAGES
from probe.atlas_watch import _layers_width, bottom_bar, layer_bar, load_status, render


def show(layout, width=120, height=40):
    c = Console(width=width, height=height, record=True, file=io.StringIO())
    c.print(layout)
    return c.export_text()


def sample(**kw):
    s = {
        "updated_at": 0, "started_at": 0, "active_s": 3600.0, "paused_s": 0.0, "pause": None,
        "current_stage": "layers",
        "stages": {n: {"done": 0, "total": 10, "unit": "neurons", "active_s": 0.0} for n in STAGES},
        "layers": {"15": {"state": "running", "done": 1792, "total": 3584, "mean_score": 0.7},
                   "19": {"state": "queued", "done": 0, "total": 3584, "mean_score": None}},
        "usage": {"input_tokens": 1200, "output_tokens": 300, "calls": 7},
        "scores": {"primary": {"n": 40, "blind_maximum": 0.5, "near_miss": 0.2, "distant": 0.05},
                   "ac_matched": {"n": 30, "blind_maximum": 0.6, "near_miss": 0.1, "distant": 0.0}},
        "latest": {"name": "L15.mlp.42", "explanation": "fires on prices",
                   "tokens": [["the", 0.0], ["$", 0.9], ["5", 0.4]]},
        "note": None,
    }
    s["stages"]["layers"].update(done=5, active_s=3600.0)
    s.update(kw)
    return s


def test_missing_status_says_waiting():
    assert "Waiting for the pipeline" in show(render(None, now=0, width=120))


def test_partial_status_does_not_crash():
    assert "corpus" in show(render({"current_stage": "corpus"}, now=0, width=120))


def test_stale_status_is_flagged():
    assert "no update for 5m" in show(render(sample(updated_at=0), now=300, width=120))


def test_pause_shows_reason_and_is_not_stale_while_heartbeats_arrive():
    out = show(render(sample(updated_at=290, pause={"reason": "Max limit", "resumes_at": 900}),
                      now=300, width=120))
    assert "Max limit" in out and "no update" not in out and "STOPPED" not in out


# ---- final review fixes (M3, M4) ----
def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_pause_without_heartbeat_is_stale():
    """M4: a pause rewrites status.json every 15 s, so an old pause means the run died."""
    out = show(render(sample(updated_at=0, pause={"reason": "Max limit", "resumes_at": 900}), now=300, width=140), width=140)
    assert "STALE" in out and "no update for 5m" in out and "may have stopped" in out


def test_pause_of_a_dead_process_shows_stopped_not_paused():
    st = sample(updated_at=295, pid=_dead_pid(), pause={"reason": "network: down", "resumes_at": 900})
    out = show(render(st, now=300, width=140), width=140)
    assert "STOPPED" in out and "PAUSED" not in out


def test_exited_run_shows_stopped():
    assert "STOPPED" in show(render(sample(updated_at=295, pid=None), now=300, width=140), width=140)


def test_live_pid_with_fresh_updates_is_running():
    import os
    out = show(render(sample(updated_at=295, pid=os.getpid()), now=300, width=140), width=140)
    assert "STOPPED" not in out and "STALE" not in out and "running" in out


def test_call_in_flight_shows_calling_claude_and_is_not_stale():
    """M3: a multi-minute call keeps the status fresh through heartbeats, and the viewer says what it waits on."""
    out = show(render(sample(updated_at=295, call={"since": 120}), now=300, width=140), width=140)
    assert "calling Claude since" in out and "(3m)" in out and "STALE" not in out


@pytest.mark.parametrize("odd", [{"pid": "x"}, {"pid": -1}, {"pid": True}, {"call": "now"}, {"call": {"since": "x"}}])
def test_odd_pid_and_call_never_crash(odd):
    show(render(sample(**odd), now=10, width=100))


def test_bottom_bar_shows_percent_stage_and_times():
    t = bottom_bar(sample(), now=100, width=100).plain
    assert "OVERALL" in t and "%" in t and "elapsed" in t and "left" in t


def test_layer_bar_has_one_cell_per_60_neurons():
    assert len(layer_bar(1792, 3584, 0.7, cells=60).plain) == 60


def test_layer_bar_fill_matches_progress():
    plain = layer_bar(1792, 3584, None, cells=60).plain
    assert plain.count("█") == 30 and plain.count("░") == 30


def test_full_render_shows_every_panel():
    out = show(render(sample(), now=10, width=120))
    for want in ("corpus", "report", "L15", "L19", "76%", "11%", "fires on prices", "OVERALL"):
        assert want in out


def test_bottom_line_visible_in_short_terminal():
    for h in (4, 8, 12):
        lines = show(render(sample(), now=10, width=100), height=h).rstrip().splitlines()
        assert "OVERALL" in lines[-1], h


def test_bottom_bar_fits_narrow_width():
    assert len(bottom_bar(sample(), now=100, width=60).plain) <= 60


@pytest.mark.parametrize("odd", [
    {}, {"stages": None}, {"stages": {"layers": "x"}}, {"stages": {"layers": {"done": "a", "total": 0}}},
    {"layers": [1, 2]}, {"layers": {"x": None, "15": {"done": float("nan"), "total": "?"}}},
    {"usage": "none"}, {"scores": {"primary": None}}, {"latest": "text"},
    {"latest": {"tokens": [None, 5, ["a"], {"text": "t", "act": "bad"}]}},
    {"pause": "yes"}, {"updated_at": "never"}, {"active_s": None, "started_at": "x"},
    {"current_stage": 7}, {"stages": {"layers": {"done": 1, "total": 2, "active_s": -5}}},
])
def test_odd_status_never_crashes(odd):
    show(render(odd, now=1000, width=100))
    bottom_bar(odd, now=1000, width=100)


def test_load_status_handles_bad_files(tmp_path):
    p = tmp_path / "s.json"
    assert load_status(p) is None                       # missing
    p.write_text('{"current_stage": "co')
    assert load_status(p) is None                       # truncated
    p.write_text("[1, 2]")
    assert load_status(p) is None                       # valid JSON, wrong shape
    p.write_bytes(b"\xff\xfe")
    assert load_status(p) is None                       # not text
    p.write_text(json.dumps({"current_stage": "corpus"}))
    assert load_status(p) == {"current_stage": "corpus"}


def test_viewer_does_not_import_torch():
    code = "import sys, probe.atlas_watch; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_bottom_bar_shows_paused_and_elapsed_excludes_it():
    t = bottom_bar(sample(active_s=18000, paused_s=3700), now=10**6, width=120).plain
    assert "elapsed 5h 00m" in t and "paused 1h 01m" in t


def test_pause_panel_shows_current_pause_length():
    pause = {"reason": "Max limit", "resumes_at": 900, "since": 100}
    assert "paused for 3m" in show(render(sample(updated_at=290, pause=pause), now=300, width=140))


def _style(bar):
    return str(bar.style)


def test_layer_bar_state_colours():
    assert "dim" in _style(layer_bar(50, 100, 0.9, 10, "queued"))
    assert "cyan" in _style(layer_bar(50, 100, 0.9, 10, "running"))
    assert _style(layer_bar(100, 100, 0.9, 10, "done")) == "green"
    assert _style(layer_bar(100, 100, 0.1, 10, "done")) == "red"


def _bar_cells(out, layer="L15"):
    line = next(l for l in out.splitlines() if l.lstrip("│ ").startswith(layer))
    return sum(line.count(c) for c in "█░")


def test_wide_terminal_gets_60_cells_per_layer():
    out = show(render(sample(), now=10, width=160), width=160)
    assert _bar_cells(out) == 60 and "1 char =" not in out


def test_narrow_terminal_shrinks_and_states_scale():
    out = show(render(sample(), now=10, width=90), width=90)
    n = _bar_cells(out)
    assert n < 60 and f"1 char = {-(-3584 // n)} neurons" in out


def test_latest_header_shows_score():
    s = sample(); s["latest"]["score"] = "0.83"
    assert "[0.83]" in show(render(s, now=10, width=120))


@pytest.mark.parametrize("width", [100, 120, 140, 200])
@pytest.mark.parametrize("state", ["running", "finished"])
def test_longest_layer_row_stays_on_one_line(width, state):
    from probe.atlas_watch import _layers_panel
    st = {"layers": {k: {"state": state, "done": 3584, "total": 3584, "mean_score": 0.7}
                     for k in ("15", "19")}}
    out = show(_layers_panel(st, width), width=_layers_width(width))
    rows = [l for l in out.splitlines() if "3584/3584" in l]
    assert len(rows) == 2 and all("0.70" in l for l in rows)
    assert sum(l.count("L1") for l in out.splitlines()) == 2
