"""Live terminal viewer for the atlas run: `python -m probe.atlas_watch [status.json]`.

Reads status.json (written by probe.atlas.status) every 2 s. CPU shell only: no torch.
It is meant to stay open for days, so every field read is defensive and a render
failure shows a message instead of killing the viewer.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

from probe.atlas.config import Config
from probe.atlas.status import DEFAULT_HOURS, STAGES, overall_fraction

REFRESH_S = 2.0
STALE_S = 60      # the run rewrites status.json every 15 s during Claude calls and pauses (llm.HEARTBEAT_S)
AC_SHARES = {"blind_maximum": "76%", "near_miss": "11%", "distant": "~1%"}   # AC's figures
FULL, EMPTY = "█", "░"


# ---- tolerant readers -------------------------------------------------------
def _dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def _num(x, default: float = 0.0) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return default
    return float(x)


def _dur(s: float) -> str:
    s = int(max(s, 0))
    d, r = divmod(s, 86400)
    h, r = divmod(r, 3600)
    m = r // 60
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m" if m else f"{s}s"


def _pct(x) -> str:
    return f"{_num(x) * 100:.0f}%"


def load_status(path: Path) -> dict | None:
    """The parsed status, or None when the file is missing, mid-write, or not a JSON object."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):       # UnicodeDecodeError is a ValueError
        return None
    return data if isinstance(data, dict) else None


# ---- pieces -----------------------------------------------------------------
def layer_bar(done: int, total: int, mean_score: float | None, cells: int,
              state: str | None = None) -> Text:
    """One cell per total/cells neurons, filled in proportion to done.

    Colour by state first: queued is dim, running is bright cyan; anything else
    (finished) uses the mean-score colour.
    """
    cells = max(int(cells), 0)
    frac = min(max(_num(done) / _num(total), 0.0), 1.0) if _num(total) > 0 else 0.0
    filled = round(frac * cells)
    if state == "queued":
        color = "dim"
    elif state == "running":
        color = "bold bright_cyan"
    elif mean_score is None or not math.isfinite(_num(mean_score, math.nan)):
        color = "grey58"
    else:
        color = "green" if mean_score >= 0.6 else "yellow" if mean_score >= 0.3 else "red"
    t = Text(FULL * filled, style=color)
    t.append(EMPTY * (cells - filled), style="grey30")
    return t


def _stage_frac(st: dict) -> float:
    total = _num(st.get("total"))
    return min(max(_num(st.get("done")) / total, 0.0), 1.0) if total > 0 else 0.0


def _eta(status: dict) -> float | None:
    """Active seconds left, extrapolated from the active time so far."""
    frac = overall_fraction(_stages(status))
    active = _num(status.get("active_s"))
    if frac <= 0 or active <= 0:
        return None
    return active * (1 - frac) / frac


def _stages(status: dict) -> dict:
    return {k: _dict(v) for k, v in _dict(status.get("stages")).items()}


def bottom_bar(status: dict, now: float, width: int) -> Text:
    """The always-visible line: overall bar, percent, stage, elapsed and time left."""
    status = _dict(status)
    frac = overall_fraction(_stages(status))
    elapsed = _dur(_num(status.get("active_s")))      # active time only; pauses are shown apart
    paused = _dur(_num(status.get("paused_s")))
    eta = _eta(status)
    left = _dur(eta) if eta is not None else "?"
    stage = status.get("current_stage")
    rest = f" {frac * 100:5.1f}%  {stage if isinstance(stage, str) else '-'}  elapsed {elapsed}  paused {paused}  ~{left} left"
    label = "OVERALL "
    cells = max(min(width - len(label) - len(rest), 40), 0)
    t = Text(label, style="bold")
    t.append_text(layer_bar(round(frac * 1000), 1000, 0.7, cells))
    t.append(rest)
    t.truncate(max(width, 1), overflow="ellipsis")
    return t


def _stage_table(status: dict) -> Table:
    tbl = Table.grid(padding=(0, 2))
    stages = _stages(status)
    current = status.get("current_stage")
    for name in STAGES:
        st = stages.get(name, {})
        frac = _stage_frac(st)
        mark = "▶" if name == current else " "
        row = [Text(f"{mark} {name}", style="bold cyan" if name == current else "")]
        if not st:
            row += [Text("-", style="dim"), Text(f"~{DEFAULT_HOURS[name]:g} h", style="dim"), Text("")]
        else:
            done, total, active = _num(st.get("done")), _num(st.get("total")), _num(st.get("active_s"))
            unit = st.get("unit") if isinstance(st.get("unit"), str) else ""
            rate = done / active if active > 0 else 0.0
            speed = (f"{rate:.3g} {unit}/s" if rate >= 1 else f"{rate * 3600:.3g} {unit}/h") if rate > 0 else "-"
            left = _dur((total - done) / rate) if rate > 0 and total > done else ("done" if frac >= 1 else "?")
            row += [Text(f"{done:,.0f}/{total:,.0f} ({frac * 100:.0f}%)"), Text(speed), Text(left)]
        tbl.add_row(*row)
    return tbl


ROW_OVERHEAD = 33      # panel border+padding (4) + "L15  " (5) + " 3584/3584 finished 0.70" (24)
RIGHT_MIN = 32         # narrowest the right-hand panel may get
FULL_CELLS = 60        # 3,584 neurons / 60 = about 60 neurons per character


def _layers_width(width: int) -> int:
    """Width of the layers panel: enough for full-scale bars, less only if the terminal is narrow."""
    return max(min(FULL_CELLS + ROW_OVERHEAD, width - RIGHT_MIN), ROW_OVERHEAD + 8)


def _layer_cells(width: int) -> int:
    return min(_layers_width(width) - ROW_OVERHEAD, FULL_CELLS)


def _layers_panel(status: dict, width: int) -> Panel:
    layers = _dict(status.get("layers"))

    def key(k):
        return (0, int(k)) if str(k).isdigit() else (1, str(k))

    cells = _layer_cells(width)
    biggest = max((_num(_dict(v).get("total")) for v in layers.values()), default=0.0)
    rows = []
    for k in sorted(layers, key=key):
        L = _dict(layers[k])
        t = Text(f"L{k:<3} ")
        t.append_text(layer_bar(_num(L.get("done")), _num(L.get("total")), L.get("mean_score"), cells, L.get("state")))
        score = L.get("mean_score")
        t.append(f" {_num(L.get('done')):.0f}/{_num(L.get('total')):.0f} {str(L.get('state', ''))[:8]}")
        if score is not None and math.isfinite(_num(score, math.nan)):
            t.append(f" {score:.2f}")
        rows.append(t)
    body = Group(*rows) if rows else Text("no layers yet", style="dim")
    title = "Layers"
    if cells < FULL_CELLS and biggest > 0:
        title += f"  (1 char = {math.ceil(biggest / cells)} neurons)"
    return Panel(body, title=title, border_style="blue")


def _right_panel(status: dict, now: float, state: str = "live") -> Panel:
    usage = _dict(status.get("usage"))
    parts = [Text(f"Max usage: {_num(usage.get('calls')):.0f} calls, "
                  f"{_num(usage.get('input_tokens')):,.0f} in / {_num(usage.get('output_tokens')):,.0f} out")]
    pause = status.get("pause")
    if state == "stopped":
        parts.append(Text("stopped" + (" (was paused: " + str(_dict(pause).get("reason", "?")) + ")" if pause else ""),
                          style="bold red"))
    elif pause:
        p = _dict(pause)
        t = Text(f"PAUSED: {p.get('reason', 'unknown')}", style="bold yellow")
        resumes = _num(p.get("resumes_at"), math.nan)
        if math.isfinite(resumes):
            t.append(f"  resumes in {_dur(resumes - now)}")
        since = _num(p.get("since"), math.nan)      # start of this pause, when the writer records it
        if math.isfinite(since):
            lasted = f"paused for {_dur(now - since)}"
        else:
            lasted = f"paused {_dur(_num(status.get('paused_s')))} in total"
        parts.append(t)
        parts.append(Text(lasted))
    else:
        t = Text("running", style="green")
        since = _num(_dict(status.get("call")).get("since"), math.nan)
        if math.isfinite(since):
            t.append(f"  calling Claude since {time.strftime('%H:%M:%S', time.localtime(since))} ({_dur(now - since)})")
        parts.append(t)
    if state == "stale" and pause:
        parts.append(Text("no heartbeat: run may have stopped", style="bold red"))
    scores = _dict(status.get("scores"))
    tbl = Table(box=None, padding=(0, 1), show_header=True, header_style="bold")
    for col in ("", "n", "blind", "near", "far"):
        tbl.add_column(col)
    for label, key in (("prim", "primary"), ("AC-m", "ac_matched")):
        s = _dict(scores.get(key))
        tbl.add_row(label, f"{_num(s.get('n')):.0f}", *(_pct(s.get(k)) for k in AC_SHARES))
    tbl.add_row("AC", "", *AC_SHARES.values(), style="dim")
    parts.append(tbl)
    note = status.get("note")
    if isinstance(note, str) and note:
        parts.append(Text(note, style="italic"))
    return Panel(Group(*parts), title="Max usage and scores", border_style="magenta")


def _latest_panel(status: dict) -> Panel:
    latest = _dict(status.get("latest"))
    if not latest:
        return Panel(Text("no neuron yet", style="dim"), title="Latest neuron", border_style="green")
    head = Text(str(latest.get("name", "?")), style="bold")
    if latest.get("score") is not None:
        head.append(f"  [{latest['score']}]", style="bold yellow")
    if latest.get("explanation"):
        head.append(f": {latest['explanation']}")
    toks = []
    for tok in latest.get("tokens") if isinstance(latest.get("tokens"), list) else []:
        if isinstance(tok, str):
            toks.append((tok, 0.0))
        elif isinstance(tok, (list, tuple)) and tok and isinstance(tok[0], str):
            toks.append((tok[0], _num(tok[1] if len(tok) > 1 else 0)))
        elif isinstance(tok, dict) and isinstance(tok.get("text"), str):
            toks.append((tok["text"], _num(tok.get("act"))))
    top = max((abs(a) for _, a in toks), default=0.0)
    line = Text()
    for text, act in toks:
        w = min(abs(act) / top, 1.0) if top > 0 else 0.0
        style = Style(bgcolor=f"rgb({int(40 + 180 * w)},{int(40 + 80 * w)},40)") if w > 0 else None
        line.append(text, style=style)
    return Panel(Group(head, line), title="Latest neuron", border_style="green")


def process_alive(pid) -> bool | None:
    """True/False if `pid` names a live/dead process on this machine; None if unknown (no pid recorded)."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:          # e.g. PermissionError: it exists but belongs to someone else
        return True
    return True


def run_state(status: dict, now: float, alive=process_alive) -> str:
    """The run's state: "stopped" (it exited or its process is gone), "stale" (no write for STALE_S) or "live".

    A pause or a Claude call no longer exempts the run from the stale check: both rewrite the file every 15 s (M4)."""
    if "pid" in status:
        pid = status.get("pid")
        if pid is None or alive(pid) is False:
            return "stopped"
    updated = _num(status.get("updated_at"), math.nan)
    if math.isfinite(updated) and now - updated >= STALE_S:
        return "stale"
    return "live"


def _banner(status: dict, now: float, state: str) -> str:
    updated = _num(status.get("updated_at"), math.nan)
    ago = f" (last update {_dur(now - updated)} ago)" if math.isfinite(updated) else ""
    if state == "stopped":
        return f" STOPPED: the run is not running{ago} "
    if state == "stale":
        return f" STALE: no update for {_dur(now - updated)} "
    return ""


# ---- layout -----------------------------------------------------------------
def render(status: dict | None, now: float, width: int) -> Layout:
    """The whole screen. Everything above the bottom bar shares the height by ratio; the bar has a fixed
    one-line slot, so it survives short terminals."""
    root = Layout()
    if not isinstance(status, dict):
        root.split_column(Layout(Panel(Text("Waiting for the pipeline to write status.json ..."),
                                       border_style="yellow"), name="body"),
                          Layout(Text("OVERALL  waiting"), name="bottom", size=1))
        return root
    state = run_state(status, now)
    banner = _banner(status, now, state)
    stages = Panel(_stage_table(status), title="Pipeline" + (f"  [bold red]{banner}[/]" if banner else ""),
                   border_style="red" if banner else "cyan")
    middle = Layout(name="middle", ratio=4)
    middle.split_row(Layout(_layers_panel(status, width), size=_layers_width(width)),
                     Layout(_right_panel(status, now, state)))
    root.split_column(Layout(stages, name="stages", ratio=4, minimum_size=1),
                      middle,
                      Layout(_latest_panel(status), name="latest", ratio=2),
                      Layout(bottom_bar(status, now, width), name="bottom", size=1))
    return root


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Config().root / "status.json"
    try:
        with Live(refresh_per_second=4, screen=True) as live:
            while True:
                width = live.console.size.width
                try:
                    view = render(load_status(path), time.time(), width)
                except Exception as e:      # never let odd input kill a days-long viewer
                    view = Layout(Text(f"viewer error (will retry): {e!r}", style="red"))
                live.update(view)
                time.sleep(REFRESH_S)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
