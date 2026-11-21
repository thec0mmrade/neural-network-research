"""Command line: `python -m probe.atlas run | approve-pilot | status`. Heavy modules load only for the command that needs them."""
from __future__ import annotations

import os

# Before any other import: these are read when the libraries load, and an uncapped run can crash this machine.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("RAYON_NUM_THREADS", "8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import signal
import sys
import time
import traceback

from probe.atlas.config import Config
from probe.atlas.status import overall_fraction
from probe.atlas.store import AlreadyRunning, read_json


def _run(cfg: Config) -> int:
    from probe.atlas import runner

    print(f"Atlas run: {runner.run(cfg)}")
    return 0


def _approve(cfg: Config) -> int:
    from probe.atlas import pilot

    pilot.approve(cfg)
    print(f"Pilot approved for prompt version {cfg.prompt_version}. Start the full run with: nix-shell --run 'python -m probe.atlas run'")
    return 0


def _status(cfg: Config) -> int:
    status = read_json(cfg.root / "status.json")
    if not isinstance(status, dict):
        print("No run has started yet.")
        return 0
    print(f"Stage: {status.get('current_stage')}  Overall: {100 * overall_fraction(status.get('stages') or {}):.1f}%")
    return 0


def _log_error(cfg: Config, e: BaseException) -> None:
    """Append the error, in one plain line, then its traceback to `errors.log`. A log that cannot be written must not hide the error itself."""
    try:
        cfg.root.mkdir(parents=True, exist_ok=True)
        with open(cfg.root / "errors.log", "a") as f:
            f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
                    f"{type(e).__name__}: {' '.join(str(e).split())}\n{traceback.format_exc()}\n")
    except OSError:
        pass


def _exit_on_signal(signum, frame):
    """SIGTERM/SIGHUP end the run like Ctrl-C would: by an exception, so every cleanup runs (M5).

    llm._run then kills claude's process group (claude runs in its own session, so the signal never reaches it),
    and the runner frees the GPU, releases the lock and writes the status."""
    raise SystemExit(128 + signum)


def _install_signal_handlers() -> None:
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _exit_on_signal)


COMMANDS = {"run": _run, "approve-pilot": _approve, "status": _status}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m probe.atlas", description="Build the Qwen3.5-0.8B neuron atlas.")
    parser.add_argument("command", choices=COMMANDS)
    command = parser.parse_args(argv).command
    cfg = Config()
    try:
        return COMMANDS[command](cfg)
    except AlreadyRunning as e:
        print(f"error: another atlas run is already running (lock: {e})", file=sys.stderr)
    except Exception as e:
        print("error: " + " ".join(str(e).split()), file=sys.stderr)
        _log_error(cfg, e)
    return 1


if __name__ == "__main__":
    _install_signal_handlers()     # here, not in main(): tests call main() inside pytest's own process
    sys.exit(main())
