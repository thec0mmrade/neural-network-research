"""Stand-in for the `claude` binary in tests. Usage: fake_claude.py MODE STATEFILE [claude args...]

Reads the prompt from stdin like `claude -p`. Modes: ok, limit, limit_then_ok, garbage,
refusal, safeguard, missing_key, fail, hang, echo, explain, atlas, atlas_refuse, and for outages (I2):
empty, api_500, schema_retries, fail_twice_then_ok, slow.
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

mode, state = sys.argv[1], Path(sys.argv[2])
prompt = sys.stdin.read()
OK = (Path(__file__).parent / "claude_ok.json").read_text()


def emit(obj, rc=0):
    print(json.dumps(obj))
    sys.exit(rc)


def limit():
    emit({"type": "result", "is_error": True, "subtype": "success",
          "result": "Claude AI usage limit reached|%d" % (int(time.time()) + 3600)}, rc=1)


if mode == "ok":
    print(OK)
elif mode == "limit":
    limit()
elif mode == "limit_then_ok":
    if state.exists():
        print(OK)
    else:
        state.write_text("seen")
        limit()
elif mode == "context":
    emit({"type": "result", "is_error": True,
          "result": "Prompt is too long: context window usage limit reached"}, rc=1)
elif mode == "limit_with_context_word":
    emit({"type": "result", "is_error": True,
          "result": "You've hit your session limit, resets 6pm (context: weekly)"}, rc=1)
elif mode == "garbage_limit_rc0":
    print("usage limit reached|%d" % (int(time.time()) + 3600))
elif mode == "garbage":
    print("this is not json at all")
elif mode == "refusal":
    emit({**json.loads(OK), "stop_reason": "refusal", "structured_output": None})
elif mode == "safeguard":   # the refusal the v1 pilot got for every test call; logs each call's --model to STATEFILE
    with state.open("a") as f:
        f.write(sys.argv[sys.argv.index("--model") + 1] + "\n")
    emit({**json.loads(OK), "stop_reason": "refusal", "structured_output": None,
          "result": "API Error: Opus 5.5's safeguards flagged this message. Try rephrasing the request in a new session "
                    "or change your model.\n\nDetails: `[reasoning_extraction]`\n\nRequest ID: req_fake"})
elif mode == "missing_key":
    emit({**json.loads(OK), "structured_output": {"other": 1}})
elif mode == "fail":
    with state.open("a") as f:
        f.write("x")
    print("boom", file=sys.stderr)
    sys.exit(2)
elif mode == "empty":              # crashed before writing anything
    sys.exit(1)
elif mode == "api_500":            # the CLI's JSON for a server error
    emit({"type": "result", "is_error": True, "subtype": "success", "api_error_status": 500,
          "result": "API Error: 500 Internal server error"}, rc=1)
elif mode == "schema_retries":     # the model never produced output matching the schema: a content failure
    emit({"type": "result", "is_error": True, "subtype": "error_max_structured_output_retries",
          "result": "Failed to provide valid structured output"}, rc=1)
elif mode == "fail_twice_then_ok":  # a short network outage, then the same call answers
    n = len(state.read_text()) if state.exists() else 0
    state.write_text("x" * (n + 1))
    if n < 2:
        print("connect ECONNREFUSED", file=sys.stderr)
        sys.exit(1)
    print(OK)
elif mode == "slow":               # a call that takes a while, for the status heartbeat
    time.sleep(0.8)
    print(OK)
elif mode == "hang":
    state.write_text(str(os.getpid()))
    time.sleep(600)
elif mode == "echo":
    emit({**json.loads(OK), "structured_output": {"ok": True, "prompt": prompt,
                                                 "cwd": os.getcwd(), "argv": sys.argv[3:],
                                                 "env": sorted(os.environ)}})
elif mode == "explain":   # answers every id the schema requires, and logs each prompt to STATEFILE
    with state.open("a") as f:
        f.write(json.dumps(prompt) + "\n")
    ids = json.loads(sys.argv[sys.argv.index("--json-schema") + 1])["required"]
    pole = {"explanation": "fires on the word cat", "context": "plain", "job": "one job"}
    emit({**json.loads(OK), "structured_output": {k: {"positive": pole, "negative": pole} for k in ids}})
elif mode in ("atlas", "atlas_refuse"):   # schema-valid explain, test and rate replies, derived only from the prompt's text
    REFUSE_ONE_IN = 6     # atlas_refuse declines every call holding such a neuron, so splitting isolates exactly those neurons
    HEADER = re.compile(r"^=== ([ntr]\d+) ===$", re.M)
    parts = HEADER.split(prompt)
    blocks = dict(zip(parts[1::2], parts[2::2]))
    required = json.loads(sys.argv[sys.argv.index("--json-schema") + 1])["required"]
    if not blocks or (required != ["ratings"] and sorted(blocks) != sorted(required)):
        sys.exit(f"prompt ids {sorted(blocks)} do not match schema ids {sorted(required)}")

    def digest(text):
        return hashlib.sha256(text.encode()).hexdigest()[:10]

    def evidence(block):
        """The examples without their numbers: rounding of activations must not change a reply."""
        lines = [l for l in block.split("\n") if not re.match(r"(Positive|Negative) pole", l)]
        return re.sub(r"(?<!\\)\[[^\]]*\]", "", "\n".join(lines))

    def explained(i, block):
        tag = digest(evidence(block))
        job = ("one job", "two related jobs", "two unrelated jobs", "unclear")[int(tag, 16) % 4]
        pole = lambda name: {"explanation": f"fires on pattern {tag} ({name})", "context": "plain", "job": job}
        return {"positive": pole("positive"), "negative": pole("negative")}

    def passages(i, block):
        tag = digest(block)
        texts = {"blind_maximum": "The answer is {n} because the pattern {tag} repeats, so we move on. </think> Done.",
                 "near_miss": "Perhaps the pattern {tag} does not hold for case {n}; let me check that again.",
                 "distant": "The river valley {n} was mapped in spring by surveyors from the {tag} coast."}
        return {c: [{"text": t.format(n=n, tag=tag)} for n in range(10)] for c, t in texts.items()}

    if mode == "atlas_refuse" and any(int(digest(evidence(b)), 16) % REFUSE_ONE_IN == 0 for b in blocks.values()):
        emit({**json.loads(OK), "stop_reason": "refusal", "result": "declined in rehearsal", "structured_output": None})
    if required == ["ratings"]:
        data = {"ratings": [{"id": i, "move_on": int(digest(b), 16) % 4, "reconsider": int(digest(b + "r"), 16) % 4}
                            for i, b in blocks.items()]}
    else:
        make = explained if required[0].startswith("n") else passages
        data = {i: make(i, blocks[i]) for i in required}
    emit({**json.loads(OK), "structured_output": data})
else:
    sys.exit("unknown mode " + mode)
