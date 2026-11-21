"""Claude Code wrapper: every explanation and test in the atlas goes through `claude -p`
on the Max login. It never bills an API account, and it waits out Max usage limits.

Fields of `claude -p --output-format json` (captured in tests/atlas/fixtures/claude_ok.json):
  structured_output     the object that matched --json-schema (`result` holds the same as a string)
  is_error, subtype     failure flag; `result` then holds the error text
  stop_reason           "refusal" when the model declined
  usage.input_tokens, usage.cache_creation_input_tokens, usage.cache_read_input_tokens,
  usage.output_tokens   token counts (input is the sum of the three input fields)
The output carries no apiKeySource field; authentication is "claude.ai" (see `claude auth status`).
Never pass --bare: it ignores the Max login.

Failures fall into three kinds (I2):
  content    the reply parsed but failed the schema or validation: LLMFailed, which batches.process splits on
  transient  crash, network, timeout, 5xx, non-JSON or empty output: TransientFailure. call_with_wait waits and
             retries the same call for up to 12 h; it never reaches batches.process as an item failure
  fatal      RunStopped: the run must stop (outage over 12 h, usage limit over 8 days, no claude binary,
             unclean working directory). It is not an LLMFailed, so nothing is split or marked failed
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

MODEL = "claude-opus-5-5"     # explanations and ratings; test writing uses cfg.test_model (R36)
LIMIT_RE = re.compile(r"(?i)(usage|rate|session|weekly).{0,40}limit|limit.{0,40}(reached|reset)")
RESET_RE = re.compile(r"\|\s*(\d{9,11})\b")      # "...limit reached|1763500000"
RESET_TEXT_RE = re.compile(r"(?i)resets\s+(?:([a-z]{3,9})\s+(\d{1,2}),?\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)"
                           r"(?:\s*\(([A-Za-z_]+(?:/[A-Za-z_+-]+)*)\))?")
NOT_LIMIT_RE = re.compile(r"(?i)context (window|length)|prompt is too long|too many tokens")
DEFAULT_WAIT_S = 1800
MIN_WAIT_S = 60
MAX_RESET_AHEAD_S = 8 * 86400     # a reset further away than this is not believed
RETRY_BACKOFF_S = (30, 120, 300)        # content failures: the same call is tried 3 more times, then batches splits
OUTAGE_BACKOFF_S = (30, 120, 300, 600, 900)   # transient failures: then every 900 s ...
OUTAGE_MAX_S = 12 * 3600                 # ... until 12 h of continuous outage stop the run
HEARTBEAT_S = 15                         # status.json is rewritten this often during a call or a wait (M3)
# Spec section 5: an empty working directory with no CLAUDE.md in it or above it. It lives outside the project
# tree (I1); the qwen-weight-probe, Neural_Network_Research and claude/ folders above results/ each hold one.
LLM_CWD = Path.home() / ".cache" / "qwen-atlas" / "llm_cwd"
MEMORY_FILES = ("CLAUDE.md", "CLAUDE.local.md", ".claude/CLAUDE.md")
_CONTENT_HTTP = {400, 413, 422}          # the request itself was rejected: smaller chunks may pass
# R18, widened by the coordinator's ruling on the final fix: the run is launched from inside a Claude Code session,
# so every CLAUDE* variable is a session or nesting marker (CLAUDECODE, CLAUDE_CODE_*, CLAUDE_EFFORT, CLAUDE_PID, ...)
# or a provider switch (CLAUDE_CODE_USE_BEDROCK/VERTEX/FOUNDRY). AI_AGENT is Claude Code's agent marker.
# The Max login lives in ~/.claude, found through HOME (kept), so no CLAUDE* variable is needed for it.
_DROP_PREFIXES = ("ANTHROPIC_", "CLAUDE")
_DROP_EXACT = {"AI_AGENT"}
_KEEP_EXACT = {"CLAUDE_CONFIG_DIR"}      # relocates ~/.claude, and with it the login; not a session marker


@dataclass
class LLMResult:
    data: dict
    input_tokens: int
    output_tokens: int


class LimitReached(Exception):
    def __init__(self, resets_at: float | None, message: str = ""):
        super().__init__(f"usage limit reached (resets_at={resets_at}) {message}".strip())
        self.resets_at = resets_at
        self.message = message


class LLMFailed(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class TransientFailure(LLMFailed):
    """The service or the transport failed, not the reply: retried by call_with_wait, never split or marked failed."""


class Refused(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class RunStopped(Exception):
    """Stop the whole run with this message; a rerun resumes. Deliberately not an LLMFailed (I2)."""


def child_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Environment for the child: no API credentials, provider switches, or session and nesting markers (R18, widened)."""
    return {k: v for k, v in environ.items()
            if k in _KEEP_EXACT or not (k.startswith(_DROP_PREFIXES) or k in _DROP_EXACT)}


def build_argv(cfg, system: str, schema: dict, model: str | None = None) -> list[str]:
    """`model` defaults to MODEL, read at call time so a patched MODEL applies."""
    return [cfg.claude_bin, "-p", "--model", model or MODEL, "--effort", cfg.effort, "--tools", "",
            "--json-schema", json.dumps(schema), "--output-format", "json",
            "--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands",
            "--setting-sources", "", "--system-prompt", system]


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _limit_message(text: str) -> str | None:
    """The matched usage-limit text, or None (context/prompt-length errors are not limits)."""
    if NOT_LIMIT_RE.search(text):
        return None
    m = LIMIT_RE.search(text)
    return text.strip()[:200] if m else None


def _run(argv, prompt, cwd, timeout_s):
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, cwd=cwd,
                            env=child_env(os.environ), start_new_session=True)
    try:
        out, err = proc.communicate(prompt, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_group(proc.pid)
        proc.communicate()
        raise TransientFailure(f"claude timed out after {timeout_s}s") from None
    except BaseException:    # includes the SystemExit that SIGTERM/SIGHUP raise in __main__ (M5)
        _kill_group(proc.pid)
        proc.wait()          # reap the child: no zombie
        raise
    return proc.returncode, out, err


def clean_cwd(path: Path | None = None) -> Path:
    """The empty working directory for claude, created if missing. Raises RunStopped if a CLAUDE.md could load from it."""
    path = Path(LLM_CWD if path is None else path)
    path.mkdir(parents=True, exist_ok=True)
    path = path.resolve()
    found = [str(d / m) for d in (path, *path.parents) for m in MEMORY_FILES if (d / m).is_file()]
    if found:
        raise RunStopped(f"claude's working directory {path} would load {', '.join(found)}; "
                         "set probe.atlas.llm.LLM_CWD to an empty folder outside any project")
    if any(path.iterdir()):
        raise RunStopped(f"claude's working directory {path} is not empty; empty it and rerun")
    return path


def _http_status(obj: dict) -> int | None:
    s = obj.get("api_error_status")
    return s if isinstance(s, int) and not isinstance(s, bool) else None


def _reset_time(text: str, now: float | None = None) -> float | None:
    """The reset as a Unix time: "|<epoch>", or the CLI's "resets 3:50pm (America/Chicago)" /
    "resets Nov 21, 5am (America/Chicago)" text. A clock time with no date that has passed means tomorrow."""
    m = RESET_RE.search(text)
    if m:
        return float(m.group(1))
    m = RESET_TEXT_RE.search(text)
    if not m:
        return None
    mon, day, hour, minute, ampm, tzname = m.groups()
    try:
        tz = ZoneInfo(tzname) if tzname else None
    except (ZoneInfoNotFoundError, ValueError):
        tz = None                        # fall back to the machine's local zone
    hour = int(hour) % 12 + (12 if ampm.lower() == "pm" else 0)
    ref = datetime.fromtimestamp(time.time() if now is None else now, tz)
    t = ref.replace(hour=hour, minute=int(minute or 0), second=0, microsecond=0)
    if mon:
        try:
            month = datetime.strptime(mon[:3].title(), "%b").month
        except ValueError:
            return None
        t = t.replace(month=month, day=int(day))
        if t < ref - timedelta(days=1):  # "Jan 2" read in late December
            t = t.replace(year=t.year + 1)
    elif t <= ref:
        t += timedelta(days=1)
    return t.timestamp()


def call_claude(cfg, system: str, prompt: str, schema: dict, timeout_s: int = 1800, model: str | None = None) -> LLMResult:
    """One call. Raises LimitReached, Refused, LLMFailed (content), TransientFailure or RunStopped (see the module doc)."""
    cwd = clean_cwd()
    try:
        rc, out, err = _run(build_argv(cfg, system, schema, model), prompt, cwd, timeout_s)
    except (FileNotFoundError, PermissionError) as e:
        raise RunStopped(f"cannot run {cfg.claude_bin}: {e}") from e
    except OSError as e:
        raise TransientFailure(f"cannot run {cfg.claude_bin}: {e}") from e
    try:
        obj = json.loads(out)
    except ValueError:
        obj = None
    if not isinstance(obj, dict):
        text = f"{out}\n{err}"
        msg = _limit_message(text) if rc != 0 else None
        if msg:
            raise LimitReached(_reset_time(text), msg)
        detail = f"claude exit {rc}, unparseable output: {text.strip()[:300]}"
        if NOT_LIMIT_RE.search(text):
            raise LLMFailed(detail)          # the prompt was too long: smaller chunks can pass
        raise TransientFailure(detail)       # crash, network, empty output
    if obj.get("stop_reason") == "refusal":
        raise Refused(str(obj.get("result") or "model refused"))
    if rc != 0 or obj.get("is_error"):
        text = str(obj.get("result") or err)
        msg = _limit_message(text)
        if msg:
            raise LimitReached(_reset_time(text), msg)
        detail = f"claude exit {rc}: {text.strip()[:300]}"
        subtype = str(obj.get("subtype") or "")
        if NOT_LIMIT_RE.search(text) or subtype.startswith("error_max") or _http_status(obj) in _CONTENT_HTTP:
            raise LLMFailed(detail)          # the model or the request could not satisfy the schema
        raise TransientFailure(detail)       # API errors (5xx, overloaded, auth, network)
    data = obj.get("structured_output")
    if not isinstance(data, dict):
        raise LLMFailed("no structured_output object in claude result")
    missing = [k for k in schema.get("required", []) if k not in data]
    if missing:
        raise LLMFailed(f"structured_output lacks required keys {missing}")
    u = obj.get("usage") or {}
    inp = sum(int(u.get(k) or 0) for k in
              ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    return LLMResult(data, inp, int(u.get("output_tokens") or 0))


@contextmanager
def _heartbeat(status):
    """Rewrite status.json every HEARTBEAT_S seconds while the body blocks (a call or a long wait)."""
    stop = threading.Event()
    every = HEARTBEAT_S

    def beat():
        while not stop.wait(every):
            try:
                status.beat()
            except OSError:      # a full disk must not kill the heartbeat; the next beat tries again
                pass

    thread = threading.Thread(target=beat, name="status-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()        # no status write from the thread once the main thread uses the status again


def _stamp(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


def _stop(status, message: str, cause: BaseException) -> RunStopped:
    status.resumed()
    status.note(f"Run stopped: {message}")
    status.flush()
    err = RunStopped(message)
    err.__cause__ = cause
    return err


def call_with_wait(cfg, status, system: str, prompt: str, schema: dict,
                   sleep=time.sleep, clock=time.time, model: str | None = None) -> LLMResult:
    """The same call, retried until it answers.

    Usage limits: wait for the reset; weekly limits can last days, but one call limited for more than 8 days stops
    the run (RunStopped). Transient failures: wait with OUTAGE_BACKOFF_S and retry, and stop the run after 12 h of
    continuous outage. Both waits show as a pause in status.json. Content failures: retried 3 times with
    RETRY_BACKOFF_S, then LLMFailed, which batches.process splits on. Refused is raised at once.
    `model` picks the Claude model (default MODEL); test writing passes cfg.test_model.
    """
    failures = outage_tries = 0
    limited_since = outage_since = None
    paused = False

    def wait(seconds):
        with _heartbeat(status):
            sleep(seconds)

    while True:
        try:
            status.calling(clock())
            try:
                with _heartbeat(status):
                    result = call_claude(cfg, system, prompt, schema, model=model)
            finally:
                status.calling(None)
        except LimitReached as e:
            now = clock()
            outage_since, outage_tries = None, 0      # the service answered
            limited_since = now if limited_since is None else limited_since
            if now - limited_since > MAX_RESET_AHEAD_S:
                raise _stop(status, f"the Claude Max usage limit has lasted more than 8 days (since {_stamp(limited_since)}): "
                                    f"{e.message}. Check `claude` and /usage, then rerun; the run resumes where it stopped", e)
            r = e.resets_at
            resume = r if r is not None and now < r <= now + MAX_RESET_AHEAD_S else now + DEFAULT_WAIT_S
            resume = max(resume, now + MIN_WAIT_S)
            status.paused(f"Claude Max usage limit reached: {e.message}", resume, since=limited_since)
            paused = True
            wait(resume - now)
            continue
        except TransientFailure as e:
            now = clock()
            outage_since = now if outage_since is None else outage_since
            if now - outage_since >= OUTAGE_MAX_S:
                raise _stop(status, f"Claude has been unreachable for {OUTAGE_MAX_S // 3600} h (since {_stamp(outage_since)}); "
                                    f"last error: {e.reason}. Fix the network or the claude login, then rerun; "
                                    "the run resumes where it stopped and no neuron was marked failed", e)
            pause = OUTAGE_BACKOFF_S[min(outage_tries, len(OUTAGE_BACKOFF_S) - 1)]
            outage_tries += 1
            status.paused(f"network: Claude unreachable, retry {outage_tries}: {e.reason[:160]}", now + pause, since=outage_since)
            paused = True
            wait(pause)
            continue
        except (LLMFailed, Refused) as e:
            if paused:
                status.resumed()
                paused = False
            limited_since = outage_since = None
            outage_tries = 0
            if isinstance(e, Refused) or failures >= len(RETRY_BACKOFF_S):
                raise
            wait(RETRY_BACKOFF_S[failures])
            failures += 1
            continue
        if paused:
            status.resumed()
        status.usage(result.input_tokens, result.output_tokens)
        return result
