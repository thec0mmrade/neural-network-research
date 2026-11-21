### Task 9: Claude Code wrapper

**Files:**
- Create: `src/probe/atlas/llm.py`, `tests/atlas/fixtures/fake_claude.py`, `tests/atlas/fixtures/claude_ok.json`
- Test: `tests/atlas/test_llm.py`

**Interfaces:**
- Produces: `LLMResult(data: dict, input_tokens: int, output_tokens: int)`; exceptions `LimitReached(resets_at: float | None)`, `LLMFailed(reason: str)`, `Refused(reason: str)`; `child_env(environ: Mapping[str, str]) -> dict[str, str]`; `build_argv(cfg, system: str, schema: dict) -> list[str]`; `call_claude(cfg, system: str, prompt: str, schema: dict, timeout_s: int = 1800) -> LLMResult`; `call_with_wait(cfg, status, system, prompt, schema, sleep=time.sleep) -> LLMResult`.

- [ ] **Step 1: Capture one real response as the fixture**

Run from an empty directory, with both API variables unset:

```bash
env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN claude -p --model claude-opus-5-5 --effort low --tools "" \
  --json-schema '{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}' \
  --output-format json --no-session-persistence --strict-mcp-config --disable-slash-commands --setting-sources "" \
  --system-prompt "Answer with the requested JSON." <<< "Return ok=true." > tests/atlas/fixtures/claude_ok.json
```

Expected: valid JSON containing the structured result and token usage. Record the exact field names in a comment at the top of `llm.py`, and use them in the parser.

- [ ] **Step 2: Write the failing tests**

```python
def test_child_env_removes_api_credentials():
    env = child_env({"ANTHROPIC_API_KEY": "x", "ANTHROPIC_AUTH_TOKEN": "y", "HOME": "/h", "PATH": "/b"})
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env and env["HOME"] == "/h"

def test_argv_has_required_flags_and_never_bare(tiny_cfg):
    argv = build_argv(tiny_cfg, "sys", {"type": "object"})
    for flag in ("-p", "--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands"):
        assert flag in argv
    assert argv[argv.index("--model") + 1] == "claude-opus-5-5"
    assert argv[argv.index("--tools") + 1] == "" and "--bare" not in argv

def test_ok_response_parsed(fake_cfg):
    r = call_claude(fake_cfg("ok"), "s", "p", {"type": "object"})
    assert r.data == {"ok": True} and r.input_tokens > 0

def test_limit_message_raises_limit_reached(fake_cfg):
    with pytest.raises(LimitReached):
        call_claude(fake_cfg("limit"), "s", "p", {"type": "object"})

def test_call_with_wait_pauses_then_retries_same_call(fake_cfg, status):
    slept = []
    r = call_with_wait(fake_cfg("limit_then_ok"), status, "s", "p", {"type": "object"}, sleep=slept.append)
    assert r.data == {"ok": True} and slept and read_json(status.path)["pause"] is None

def test_invalid_output_is_failure_not_success(fake_cfg):
    with pytest.raises(LLMFailed):
        call_claude(fake_cfg("garbage"), "s", "p", {"type": "object"})
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `nix-shell --run 'OMP_NUM_THREADS=4 pytest tests/atlas/test_llm.py -v'`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement `llm.py` and `fake_claude.py`**

The prompt goes through stdin, and `cwd` is the empty directory `cfg.root/"llm_cwd"`. Limit detection: a non-zero or `is_error` result whose text matches `(?i)(usage|rate|session|weekly).{0,40}limit|limit.{0,40}(reached|reset)`. Parse a reset time when one is present; with no time, wait 30 minutes and retry. Other failures retry 3 times with 30 s, 120 s and 300 s backoff before raising `LLMFailed`.

- [ ] **Step 5: Run the tests and confirm they pass**

Expected: 6 passed

---

