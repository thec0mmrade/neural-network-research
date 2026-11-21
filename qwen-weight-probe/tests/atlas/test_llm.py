import os
import signal
import time

import pytest

from probe.atlas.llm import (LimitReached, LLMFailed, Refused, RunStopped, TransientFailure, build_argv,
                             call_claude, call_with_wait, child_env, clean_cwd)
from probe.atlas.store import read_json

SCHEMA = {"type": "object"}


def test_child_env_removes_api_credentials():
    env = child_env({"ANTHROPIC_API_KEY": "x", "ANTHROPIC_AUTH_TOKEN": "y", "HOME": "/h", "PATH": "/b"})
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env and env["HOME"] == "/h"


def test_child_env_removes_provider_and_nesting_markers():
    dirty = {"ANTHROPIC_BASE_URL": "u", "ANTHROPIC_FOO": "1", "CLAUDE_CODE_USE_BEDROCK": "1",
             "CLAUDE_CODE_USE_VERTEX": "1", "CLAUDE_CODE_USE_FOUNDRY": "1", "CLAUDECODE": "1",
             "CLAUDE_CODE_ENTRYPOINT": "cli", "CLAUDE_CODE_SSE_PORT": "9", "HOME": "/h"}
    assert child_env(dirty) == {"HOME": "/h"}


def test_child_env_strips_every_claude_session_marker_and_keeps_the_login_path():
    """Ruling on concern 2: the run starts inside a Claude Code session; the child must look like a fresh terminal."""
    session = {k: "x" for k in (
        "CLAUDECODE", "CLAUDE_CODE_CERT_STORE", "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_EXECPATH", "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN",
        "CLAUDE_CODE_SESSION_ATTENDED", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_SSE_PORT", "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_EFFORT", "CLAUDE_PID", "AI_AGENT", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")}
    keep = {"HOME": "/home/u", "PATH": "/bin", "XDG_CONFIG_HOME": "/home/u/.config", "XDG_RUNTIME_DIR": "/run/user/1000",
            "CLAUDE_CONFIG_DIR": "/home/u/.claude", "PWD": "/home/u/claude-stuff"}
    assert child_env({**session, **keep}) == keep


def test_argv_has_required_flags_and_never_bare(tiny_cfg):
    argv = build_argv(tiny_cfg, "sys", {"type": "object"})
    for flag in ("-p", "--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands"):
        assert flag in argv
    assert argv[argv.index("--model") + 1] == "claude-opus-5-5"
    assert argv[argv.index("--tools") + 1] == "" and "--bare" not in argv


def test_argv_uses_the_model_it_is_given(tiny_cfg):
    """R36: test writing runs on cfg.test_model; everything else keeps llm.MODEL."""
    argv = build_argv(tiny_cfg, "sys", {"type": "object"}, model=tiny_cfg.test_model)
    assert argv[argv.index("--model") + 1] == tiny_cfg.test_model == "claude-opus-5-5" and "--bare" not in argv
    assert build_argv(tiny_cfg, "sys", {"type": "object"})[argv.index("--model") + 1] == llm.MODEL == "claude-opus-5-5"


def test_call_with_wait_passes_the_model_to_claude(fake_cfg, status):
    argv = call_with_wait(fake_cfg("echo"), status, "s", "p", SCHEMA, model="claude-sonnet-5-5").data["argv"]
    assert argv[argv.index("--model") + 1] == "claude-sonnet-5-5"


def test_ok_response_parsed(fake_cfg):
    r = call_claude(fake_cfg("ok"), "s", "p", {"type": "object"})
    assert r.data == {"ok": True} and r.input_tokens > 0 and r.output_tokens == 52


def test_prompt_via_stdin_and_empty_cwd_and_clean_env(fake_cfg, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "leak")
    cfg = fake_cfg("echo")
    d = call_claude(cfg, "s", "hello prompt", SCHEMA).data
    assert d["prompt"] == "hello prompt" and "hello prompt" not in d["argv"]
    import probe.atlas.llm as llm
    assert d["cwd"] == str(llm.LLM_CWD.resolve())
    assert list(llm.LLM_CWD.iterdir()) == []
    assert "ANTHROPIC_API_KEY" not in d["env"]


def test_limit_message_raises_limit_reached(fake_cfg):
    with pytest.raises(LimitReached) as e:
        call_claude(fake_cfg("limit"), "s", "p", {"type": "object"})
    assert e.value.resets_at > time.time()


def test_call_with_wait_pauses_then_retries_same_call(fake_cfg, status):
    slept = []
    r = call_with_wait(fake_cfg("limit_then_ok"), status, "s", "p", {"type": "object"}, sleep=slept.append)
    assert r.data == {"ok": True} and slept and read_json(status.path)["pause"] is None
    assert status.usage_["calls"] == 1 and status.usage_["output_tokens"] == 52  # usage writes are throttled


def test_status_is_paused_while_waiting(fake_cfg, status):
    seen = []
    call_with_wait(fake_cfg("limit_then_ok"), status, "s", "p", SCHEMA,
                   sleep=lambda s: seen.append(read_json(status.path)["pause"]))
    assert seen[0]["resumes_at"] > time.time() and "limit" in seen[0]["reason"].lower()


def test_limit_without_reset_time_waits_30_minutes_and_keeps_retrying(tiny_cfg, status, monkeypatch):
    import probe.atlas.llm as llm
    calls = []

    def fake_call(cfg, system, prompt, schema, timeout_s=1800, model=None):
        calls.append(1)
        if len(calls) <= 5:
            raise LimitReached(None)
        return llm.LLMResult({"ok": True}, 1, 1)

    monkeypatch.setattr(llm, "call_claude", fake_call)
    slept = []
    call_with_wait(tiny_cfg, status, "s", "p", SCHEMA, sleep=slept.append)
    assert len(calls) == 6 and len(slept) == 5 and all(1700 <= s <= 1900 for s in slept)


def test_invalid_output_is_failure_not_success(fake_cfg):
    with pytest.raises(LLMFailed):
        call_claude(fake_cfg("garbage"), "s", "p", {"type": "object"})


def test_missing_required_key_is_failure(fake_cfg):
    with pytest.raises(LLMFailed):
        call_claude(fake_cfg("missing_key"), "s", "p", {"type": "object", "required": ["ok"]})


def test_refusal_raises_refused_and_is_not_retried(fake_cfg, status):
    slept = []
    with pytest.raises(Refused):
        call_with_wait(fake_cfg("refusal"), status, "s", "p", SCHEMA, sleep=slept.append)
    assert slept == []


def test_content_failures_retry_three_times_with_backoff(fake_cfg, status):
    slept = []
    with pytest.raises(LLMFailed) as e:
        call_with_wait(fake_cfg("missing_key"), status, "s", "p", {"type": "object", "required": ["ok"]},
                       sleep=slept.append)
    assert not isinstance(e.value, TransientFailure) and slept == [30, 120, 300]


def test_timeout_kills_child_and_raises(fake_cfg, tmp_path):
    with pytest.raises(LLMFailed, match="timed out"):
        call_claude(fake_cfg("hang"), "s", "p", SCHEMA, timeout_s=1)
    pid = int(next(tmp_path.glob("fake_state_*")).read_text())
    time.sleep(0.3)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


# ---- review fixes ----
import subprocess

import probe.atlas.llm as llm


def _limited_then_ok(monkeypatch, resets_at):
    calls = []

    def fake_call(cfg, system, prompt, schema, timeout_s=1800, model=None):
        calls.append(1)
        if len(calls) == 1:
            raise LimitReached(resets_at, "usage limit reached")
        return llm.LLMResult({"ok": True}, 1, 1)

    monkeypatch.setattr(llm, "call_claude", fake_call)


def test_past_reset_epoch_falls_back_to_30_minutes(tiny_cfg, status, monkeypatch):
    now = 1_000_000.0
    _limited_then_ok(monkeypatch, now - 500)
    slept = []
    call_with_wait(tiny_cfg, status, "s", "p", SCHEMA, sleep=slept.append, clock=lambda: now)
    assert slept == [1800]  # not believed: falls back to 30 minutes


def test_imminent_reset_waits_minimum_60s(tiny_cfg, status, monkeypatch):
    now = 1_000_000.0
    _limited_then_ok(monkeypatch, now + 5)
    slept = []
    call_with_wait(tiny_cfg, status, "s", "p", SCHEMA, sleep=slept.append, clock=lambda: now)
    assert slept == [60]


def test_far_future_reset_epoch_falls_back_to_30_minutes(tiny_cfg, status, monkeypatch):
    now = 1_000_000.0
    _limited_then_ok(monkeypatch, now + 9 * 86400)
    slept = []
    call_with_wait(tiny_cfg, status, "s", "p", SCHEMA, sleep=slept.append, clock=lambda: now)
    assert slept == [1800]


def test_pause_cleared_when_call_after_limit_fails_or_is_refused(tiny_cfg, status, monkeypatch):
    for exc in (Refused("no"), LLMFailed("bad")):
        calls = []

        def fake_call(cfg, system, prompt, schema, timeout_s=1800, exc=exc, model=None):
            calls.append(1)
            if len(calls) == 1:
                raise LimitReached(None, "usage limit")
            raise exc

        monkeypatch.setattr(llm, "call_claude", fake_call)
        seen = []

        def sleep(s):
            seen.append(read_json(status.path)["pause"])

        with pytest.raises(type(exc)):
            call_with_wait(tiny_cfg, status, "s", "p", SCHEMA, sleep=sleep)
        assert seen[0] is not None and read_json(status.path)["pause"] is None


@pytest.mark.parametrize("mode,exc", [("context", LLMFailed), ("limit_with_context_word", LimitReached)])
def test_context_errors_fail_but_limit_text_with_word_context_is_a_limit(fake_cfg, mode, exc):
    with pytest.raises(exc):
        call_claude(fake_cfg(mode), "s", "p", SCHEMA)


def test_non_json_limit_text_with_exit_zero_is_not_a_limit(fake_cfg):
    with pytest.raises(LLMFailed):
        call_claude(fake_cfg("garbage_limit_rc0"), "s", "p", SCHEMA)


def test_pause_reason_contains_matched_message(fake_cfg, status):
    seen = []
    call_with_wait(fake_cfg("limit_then_ok"), status, "s", "p", SCHEMA,
                   sleep=lambda s: seen.append(read_json(status.path)["pause"]["reason"]))
    assert "usage limit reached" in seen[0]


def test_limit_longer_than_8_days_stops_the_run(tiny_cfg, status, monkeypatch):
    """I2: not an LLMFailed, so batches.process neither splits the chunk nor waits another 8 days per half."""
    def always(cfg, system, prompt, schema, timeout_s=1800, model=None):
        raise LimitReached(None, "usage limit")

    monkeypatch.setattr(llm, "call_claude", always)
    t = [0.0]
    with pytest.raises(RunStopped, match="8 days") as e:
        call_with_wait(tiny_cfg, status, "s", "p", SCHEMA,
                       sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0])
    assert not isinstance(e.value, LLMFailed) and t[0] > 8 * 86400
    d = read_json(status.path)
    assert d["pause"] is None and "Run stopped" in d["note"] and "rerun" in d["note"]


def test_kill_ignores_missing_process(fake_cfg, monkeypatch):
    real = os.killpg
    first = []

    def flaky(pid, sig):
        if not first:           # the first kill finds "no such process"; later kills are real
            first.append(1)
            raise ProcessLookupError
        real(pid, sig)

    monkeypatch.setattr(llm.os, "killpg", flaky)
    # first kill raises ProcessLookupError, which must be ignored; kill the child ourselves
    orig = llm._kill_group

    def kill_twice(pid):
        orig(pid)
        real(pid, signal.SIGKILL)

    monkeypatch.setattr(llm, "_kill_group", kill_twice)
    with pytest.raises(LLMFailed, match="timed out"):
        call_claude(fake_cfg("hang"), "s", "p", SCHEMA, timeout_s=1)
    assert first


def test_group_killed_when_communicate_raises(fake_cfg, tmp_path, monkeypatch):
    killed = []
    real = os.killpg
    monkeypatch.setattr(llm.os, "killpg", lambda pid, sig: (killed.append(pid), real(pid, sig)))

    def boom(self, *a, **k):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(subprocess.Popen, "communicate", boom)
    with pytest.raises(RuntimeError):
        call_claude(fake_cfg("hang"), "s", "p", SCHEMA)
    assert killed


# ---- final review fixes (I1, I2, M3, M5) ----
import sys
import threading
from pathlib import Path

from probe.atlas.batches import process

PROJECT = Path(__file__).resolve().parents[2]


def test_default_cwd_is_outside_the_project_tree():
    """I1: spec section 5 wants an empty cwd with no CLAUDE.md; the project tree has three above results/."""
    default = Path.home() / ".cache" / "qwen-atlas" / "llm_cwd"
    assert llm.LLM_CWD == default        # this module's tests do not use fake_cfg's override here
    assert not default.resolve().is_relative_to(PROJECT.parent.parent)


def test_cwd_below_a_claude_md_or_not_empty_stops_the_run(tmp_path):
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "CLAUDE.md").write_text("# project memory")
    with pytest.raises(RunStopped, match="CLAUDE.md"):
        clean_cwd(tmp_path / "proj" / "deep" / "llm_cwd")
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "left.txt").write_text("x")
    with pytest.raises(RunStopped, match="not empty"):
        clean_cwd(tmp_path / "busy")
    assert clean_cwd(tmp_path / "fresh" / "llm_cwd").is_dir()      # created when missing


@pytest.mark.parametrize("mode", ["fail", "garbage", "empty", "api_500"])
def test_transport_failures_are_transient(fake_cfg, mode):
    with pytest.raises(TransientFailure):
        call_claude(fake_cfg(mode), "s", "p", SCHEMA)


def test_timeout_is_transient(fake_cfg):
    with pytest.raises(TransientFailure, match="timed out"):
        call_claude(fake_cfg("hang"), "s", "p", SCHEMA, timeout_s=1)


@pytest.mark.parametrize("mode,schema", [("missing_key", {"type": "object", "required": ["ok"]}),
                                         ("schema_retries", SCHEMA), ("context", SCHEMA)])
def test_content_failures_are_not_transient(fake_cfg, mode, schema):
    with pytest.raises(LLMFailed) as e:
        call_claude(fake_cfg(mode), "s", "p", schema)
    assert not isinstance(e.value, TransientFailure)


def test_missing_binary_stops_the_run(fake_cfg, tmp_path):
    import dataclasses
    cfg = dataclasses.replace(fake_cfg("ok"), claude_bin=str(tmp_path / "no_such_claude"))
    with pytest.raises(RunStopped, match="cannot run"):
        call_claude(cfg, "s", "p", SCHEMA)


def test_short_outage_waits_and_retries_the_same_call(fake_cfg, status):
    seen = []
    r = call_with_wait(fake_cfg("fail_twice_then_ok"), status, "s", "p", SCHEMA,
                       sleep=lambda s: seen.append((s, read_json(status.path)["pause"])))
    assert r.data == {"ok": True} and [s for s, _ in seen] == [30, 120]
    assert all(p["reason"].startswith("network") and p["since"] > 0 for _, p in seen)
    assert read_json(status.path)["pause"] is None and status.usage_["calls"] == 1


def _fake_time():
    t = [1_000_000.0]
    return t, (lambda s: t.__setitem__(0, t[0] + s)), (lambda: t[0])


def test_outage_over_12_hours_stops_the_run_with_a_clear_message(fake_cfg, status, tmp_path):
    t, sleep, clock = _fake_time()
    with pytest.raises(RunStopped, match="unreachable for 12 h.*rerun") as e:
        call_with_wait(fake_cfg("fail"), status, "s", "p", SCHEMA, sleep=sleep, clock=clock)
    assert not isinstance(e.value, LLMFailed)
    assert 12 * 3600 <= t[0] - 1_000_000.0 < 12 * 3600 + 900
    calls = len(next(tmp_path.glob("fake_state_*")).read_text())
    assert 50 <= calls <= 60          # 30, 120, 300, 600, then every 900 s
    d = read_json(status.path)
    assert d["pause"] is None and "unreachable" in d["note"]


def _answer(cfg, status, sleep, clock):
    """batches work function: one call per chunk through the real call_with_wait and claude path."""
    def work(chunk):
        ids = [f"n{x['nid']}" for x in chunk]
        data = call_with_wait(cfg, status, "s", "p", {"type": "object", "required": ids}, sleep=sleep, clock=clock).data
        return [{"nid": int(k[1:]), "job": data[k]["positive"]["job"]} for k in ids]
    return work


def test_outage_never_splits_or_marks_neurons_failed_and_rerun_pays_once(fake_cfg, status, tmp_path):
    """I2: a 12-hour outage stops the run before any split mark or failed record; the rerun asks each chunk once."""
    items, out = [{"nid": i} for i in range(8)], tmp_path / "out"
    t, sleep, clock = _fake_time()
    with pytest.raises(RunStopped):
        process(items, 4, out, _answer(fake_cfg("fail"), status, sleep, clock))
    assert not out.exists() or not list(out.iterdir())          # no split_*.json, no batch_*.json
    cfg = fake_cfg("explain")
    records = process(items, 4, out, _answer(cfg, status, sleep, clock))
    assert [r["nid"] for r in records] == list(range(8)) and not any("failed" in r for r in records)
    assert len(next(tmp_path.glob("fake_state_2")).read_text().splitlines()) == 2     # two chunks, two calls
    assert not list(out.glob("split_*.json"))


def test_long_limit_never_splits(tiny_cfg, status, monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "call_claude", lambda *a, **k: (_ for _ in ()).throw(LimitReached(None, "usage limit")))
    t, sleep, clock = _fake_time()
    with pytest.raises(RunStopped):
        process([{"nid": i} for i in range(4)], 4, tmp_path / "out", _answer(tiny_cfg, status, sleep, clock))
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def test_transient_failure_from_work_is_never_split(tmp_path):
    def down(chunk):
        raise TransientFailure("network")
    with pytest.raises(TransientFailure):
        process([{"nid": i} for i in range(8)], 8, tmp_path, down)
    assert not list(tmp_path.iterdir())


def test_status_heartbeat_during_a_call(fake_cfg, status, monkeypatch):
    """M3: status.json is rewritten while claude runs, and says a call is in flight."""
    monkeypatch.setattr(llm, "HEARTBEAT_S", 0.1)
    beats = []
    real = status.beat
    monkeypatch.setattr(status, "beat", lambda: (real(), beats.append(read_json(status.path))))
    call_with_wait(fake_cfg("slow"), status, "s", "p", SCHEMA)
    assert len(beats) >= 3 and all(b["call"] and b["call"]["since"] > 0 for b in beats)
    assert len({b["updated_at"] for b in beats}) == len(beats)
    assert read_json(status.path)["call"] is None


def test_status_heartbeat_during_a_pause(fake_cfg, status, monkeypatch):
    monkeypatch.setattr(llm, "HEARTBEAT_S", 0.05)
    beats = []
    real = status.beat
    monkeypatch.setattr(status, "beat", lambda: (real(), beats.append(read_json(status.path)["pause"])))
    call_with_wait(fake_cfg("limit_then_ok"), status, "s", "p", SCHEMA, sleep=lambda s: time.sleep(0.4))
    assert sum(1 for b in beats if b and "limit" in b["reason"]) >= 3      # the 0.4 s wait alone gives about 7


def test_heartbeat_default_is_under_20_seconds():
    assert llm.HEARTBEAT_S <= 20


CHILD = """
import sys
from pathlib import Path
import dataclasses
import probe.atlas.llm as llm
from probe.atlas.__main__ import _install_signal_handlers
from probe.atlas.config import Config

llm.LLM_CWD = Path(sys.argv[2])
_install_signal_handlers()
cfg = dataclasses.replace(Config.tiny(Path(sys.argv[3])), claude_bin=sys.argv[1])
llm.call_claude(cfg, "s", "p", {"type": "object"}, timeout_s=300)
"""


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_sigterm_and_sighup_kill_the_claude_child(fake_cfg, tmp_path, sig):
    """M5: claude runs in its own session, so only Python's cleanup can stop it."""
    cfg = fake_cfg("hang")
    state = tmp_path / "fake_state_1"
    child = subprocess.Popen([sys.executable, "-c", CHILD, cfg.claude_bin, str(tmp_path / "llm_cwd"), str(tmp_path / "root")])
    try:
        for _ in range(200):
            if state.exists() and state.read_text():
                break
            time.sleep(0.05)
        claude_pid = int(state.read_text())
        os.kill(child.pid, sig)
        assert child.wait(timeout=20) == 128 + sig
    finally:
        if child.poll() is None:
            child.kill()
    time.sleep(0.3)
    with pytest.raises(ProcessLookupError):
        os.kill(claude_pid, 0)


# ---- R37: mass refusal stops the run ----

def _test_items(n):
    return [{"nid": i, "pole": 1, "job": "one job", "explanation": f"fires on topic {i}", "context": "plain"} for i in range(n)]


def test_mass_refusal_stops_the_run_quoting_the_details_and_marks_nothing_failed(fake_cfg, status, tmp_path):
    """The v1 pilot: every test call refused by the safeguards. The run must stop, not mark 200 neurons failed."""
    from probe.atlas.evaluate import write_tests

    cfg, out = fake_cfg("safeguard"), tmp_path / "tests"
    with pytest.raises(RunStopped) as e:
        write_tests(cfg, status, _test_items(30), out)
    assert "`[reasoning_extraction]`" in str(e.value) and not isinstance(e.value, LLMFailed)
    assert not list(out.glob("batch_*.json"))                      # no record of any kind, failed or not
    note = read_json(status.path)["note"]
    assert note.startswith("Run stopped: ") and "[reasoning_extraction]" in note
    models = next(tmp_path.glob("fake_state_*")).read_text().split()
    # the test writer's model, then the fallback once per single refused neuron (R40); both refuse here
    assert models and set(models) == {cfg.test_model, cfg.test_fallback_model}
    # it fired at the 20th single refusal: no further item was asked for or failed after that
    assert "refused 20 of the 20 items" in str(e.value)

    records = write_tests(fake_cfg("atlas"), status, _test_items(30), out)     # the rerun resumes and asks again
    assert len(records) == 30 and not any("failed" in r for r in records)


def _refusing(refused):
    def work(chunk):
        if chunk[0]["nid"] in refused:
            raise Refused("declined\n\nDetails: `[tag]`")
        return [{"nid": x["nid"], "ok": True} for x in chunk]
    return work


def test_refusals_below_twenty_are_stored_as_failed(tmp_path):
    out = process([{"nid": i} for i in range(19)], 1, tmp_path, _refusing(set(range(19))))
    assert all("failed" in r for r in out) and len(list(tmp_path.glob("batch_*.json"))) == 19


def test_twenty_refusals_under_half_of_the_items_do_not_stop(tmp_path):
    out = process([{"nid": i} for i in range(41)], 1, tmp_path, _refusing(set(range(2, 42, 2))))   # the 20th is 20 of 41
    assert sum("failed" in r for r in out) == 20


def test_twenty_refusals_at_half_of_the_items_stop_and_remove_their_records(tmp_path, status):
    with pytest.raises(RunStopped, match=r"refused 20 of the 40 items.*`\[tag\]`"):
        process([{"nid": i} for i in range(60)], 1, tmp_path, _refusing(set(range(1, 60, 2))), status=status)
    stored = [read_json(p)[0] for p in tmp_path.glob("batch_*.json")]
    assert stored and not any("failed" in r for r in stored)         # the 20 answered items stay, no refusal does
    assert max(r["nid"] for r in stored) == 38                       # nothing was asked for after the stop
    assert "Run stopped" in read_json(status.path)["note"]


def test_a_smaller_batch_size_reuses_records_stored_in_bigger_batches(tmp_path):
    """R39: after explain_batch drops, items already answered inside bigger batches are not asked for again."""
    asked = []

    def work(chunk):
        asked.extend(x["nid"] for x in chunk)
        return [{"nid": x["nid"], "v": len(chunk)} for x in chunk]

    first = process([{"nid": i} for i in range(6)], 3, tmp_path, work)
    assert asked == list(range(6))
    again = process([{"nid": i} for i in range(8)], 1, tmp_path, work)
    assert asked == list(range(8))                     # only 6 and 7 are new
    assert again[:6] == first and [r["nid"] for r in again] == list(range(8))


def test_failed_records_in_bigger_batches_are_not_reused(tmp_path):
    from probe.atlas.store import atomic_write_json
    atomic_write_json(tmp_path / "batch_0_2.json", [{"nid": 0, "failed": "x"}, {"nid": 1, "v": 1}])
    asked = []
    out = process([{"nid": 0}, {"nid": 1}], 1, tmp_path, lambda c: asked.extend(x["nid"] for x in c) or [{"nid": c[0]["nid"], "v": 9}])
    assert asked == [0] and out == [{"nid": 0, "v": 9}, {"nid": 1, "v": 1}]


def test_reset_time_reads_cli_clock_text():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from probe.atlas.llm import _reset_time
    tz = ZoneInfo("America/Chicago")
    now = datetime(2026, 11, 20, 14, 18, tzinfo=tz).timestamp()
    msg = "You've hit your 5-hour limit · resets 3:50pm (America/Chicago)"
    assert _reset_time(msg, now) == datetime(2026, 11, 20, 15, 50, tzinfo=tz).timestamp()
    late = datetime(2026, 11, 20, 23, 0, tzinfo=tz).timestamp()
    assert _reset_time("limit · resets 5am (America/Chicago)", late) == datetime(2026, 11, 21, 5, 0, tzinfo=tz).timestamp()
    assert _reset_time("weekly limit · resets Nov 21, 5am (America/Chicago)", now) == \
        datetime(2026, 11, 21, 5, 0, tzinfo=tz).timestamp()
    assert _reset_time("limit · resets 12:30am (America/Chicago)", now) == datetime(2026, 11, 21, 0, 30, tzinfo=tz).timestamp()
    assert _reset_time("limit reached|1763500000", now) == 1763500000.0
    assert _reset_time("usage limit reached", now) is None
