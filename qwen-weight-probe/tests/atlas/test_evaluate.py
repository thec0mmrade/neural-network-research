from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from probe.atlas import activations, evaluate, prompts
from probe.atlas.config import Config, neuron_id, neuron_name
from probe.atlas.evaluate import example_ids, passage_hit, score_tests, summarize, write_tests
from probe.atlas.llm import LLMResult, Refused
from probe.atlas.status import Status

CATEGORIES = ("blind_maximum", "near_miss", "distant")


def reply_for(ids, count=10):
    return {i: {c: [{"text": f"{i} {c} {n}"} for n in range(count)] for c in CATEGORIES} for i in ids}


@pytest.fixture
def capture_prompt(tmp_path, monkeypatch):
    """Replaces the Claude call: records every prompt and answers each id in the schema with valid passages."""
    seen = SimpleNamespace(status=Status(tmp_path / "status.json"), out=tmp_path / "tests", calls=[], text=None, answer=reply_for)

    def fake(cfg, status, system, prompt, schema, model=None):
        seen.calls.append(SimpleNamespace(system=system, prompt=prompt, schema=schema, model=model))
        seen.text = prompt
        return LLMResult(seen.answer(schema["required"]), 1, 1)

    monkeypatch.setattr(evaluate, "call_with_wait", fake)
    return seen


def refuse_model(capture_prompt, model, nids=None):
    """Make `model` refuse any call (or only calls that mention one of `nids`' explanations)."""
    answer = capture_prompt.answer

    def fake(cfg, status, system, prompt, schema, model=None, refusing=model):
        capture_prompt.calls.append(SimpleNamespace(system=system, prompt=prompt, schema=schema, model=model))
        if model == refusing and (nids is None or any(f"e{n}" + chr(10) in prompt + chr(10) for n in nids)):
            raise Refused("API Error: safeguards flagged this message. Details: `[reasoning_extraction]`")
        return LLMResult(answer(schema["required"]), 1, 1)
    return fake


def test_refused_single_neuron_falls_back_and_is_marked(tiny_cfg, capture_prompt, monkeypatch):
    """R40: Opus refuses one neuron's tests, so they are split down to it and written by the fallback model."""
    monkeypatch.setattr(evaluate, "call_with_wait", refuse_model(capture_prompt, "claude-opus-5-5", nids=[7]))
    items = [{"nid": n, "pole": 1, "explanation": f"e{n}", "context": "plain"} for n in (5, 6, 7, 8)]
    out = write_tests(tiny_cfg, capture_prompt.status, items, capture_prompt.out)
    assert [r["nid"] for r in out] == [5, 6, 7, 8] and not any("failed" in r for r in out)
    assert [r.get("test_model") for r in out] == [None, None, "claude-haiku-5-5", None]
    assert [c.model for c in capture_prompt.calls].count("claude-haiku-5-5") == 1   # only the single refused neuron


def test_refusal_by_both_models_stores_the_neuron_as_failed(tiny_cfg, capture_prompt, monkeypatch):
    seen = []

    def fake(cfg, status, system, prompt, schema, model=None):
        seen.append(model)
        raise Refused("Details: `[reasoning_extraction]`")
    monkeypatch.setattr(evaluate, "call_with_wait", fake)
    out = write_tests(tiny_cfg, capture_prompt.status, [{"nid": 3, "pole": -1, "explanation": "x", "context": "plain"}],
                      capture_prompt.out)
    assert "failed" in out[0] and "reasoning_extraction" in out[0]["failed"] and out[0]["pole"] == -1
    assert seen == ["claude-opus-5-5", "claude-haiku-5-5"]


def test_kill_during_fallback_does_not_ask_the_test_model_again(tiny_cfg, capture_prompt, monkeypatch):
    """The test model's refusal is stored before the fallback call, so a resume goes straight to the fallback."""
    class Killed(Exception):
        pass
    seen = []

    def dying(cfg, status, system, prompt, schema, model=None):
        seen.append(model)
        if model == "claude-opus-5-5":
            raise Refused("declined")
        raise Killed
    monkeypatch.setattr(evaluate, "call_with_wait", dying)
    item = [{"nid": 3, "pole": 1, "explanation": "x", "context": "plain"}]
    with pytest.raises(Killed):
        write_tests(tiny_cfg, capture_prompt.status, item, capture_prompt.out)
    monkeypatch.setattr(evaluate, "call_with_wait", refuse_model(capture_prompt, "claude-opus-5-5"))
    out = write_tests(tiny_cfg, capture_prompt.status, item, capture_prompt.out)
    assert seen == ["claude-opus-5-5", "claude-haiku-5-5"]
    assert [c.model for c in capture_prompt.calls] == ["claude-haiku-5-5"] and out[0]["test_model"] == "claude-haiku-5-5"


def test_placeholder_batch_is_split_and_the_withheld_neuron_falls_back(tiny_cfg, capture_prompt, monkeypatch):
    """R41: Opus answers a batch with "[withheld]" for one neuron; the others keep Opus's passages, that one goes to Haiku."""
    def answer(cfg, status, system, prompt, schema, model=None):
        capture_prompt.calls.append(SimpleNamespace(prompt=prompt, model=model))
        data = reply_for(schema["required"])
        if model == "claude-opus-5-5" and "Description: e7" in prompt:
            key = [k for k in schema["required"] if f"=== {k} ===\nDescription: e7" in prompt][0]
            data[key] = {c: [{"text": "[withheld]"}] * 10 for c in CATEGORIES}
        return LLMResult(data, 1, 1)
    monkeypatch.setattr(evaluate, "call_with_wait", answer)
    items = [{"nid": n, "pole": 1, "explanation": f"e{n}", "context": "plain"} for n in (5, 6, 7, 8)]
    out = write_tests(tiny_cfg, capture_prompt.status, items, capture_prompt.out)
    assert [r.get("test_model") for r in out] == [None, None, "claude-haiku-5-5", None]
    assert not any(t["text"] == "[withheld]" for r in out for c in CATEGORIES for t in r[c])


def test_empty_fallback_model_turns_the_fallback_off(tiny_cfg, capture_prompt, monkeypatch):
    monkeypatch.setattr(evaluate, "call_with_wait", refuse_model(capture_prompt, "claude-opus-5-5"))
    out = write_tests(replace(tiny_cfg, test_fallback_model=""), capture_prompt.status,
                      [{"nid": 3, "pole": 1, "explanation": "x", "context": "plain"}], capture_prompt.out)
    assert "failed" in out[0] and [c.model for c in capture_prompt.calls] == ["claude-opus-5-5"]


def test_summarize_reports_fallback_tested_neurons():
    recs = [rec(0, 10, 1, "one job"), {**rec(1, 0, 1, "one job"), "test_model": "claude-haiku-5-5"}]
    s = summarize(recs, live_all(2))
    assert s["fallback_tests"]["n"] == 1 and s["fallback_tests"]["blind_maximum"] == 0.0 and s["primary"]["n"] == 2


def rec(nid, hits, dom, job):
    return {"nid": nid, "pole": dom, "job": job, "blind_maximum": hits, "near_miss": hits, "distant": hits}


def live_mask(mask, dominant=None):
    n = len(mask)
    return {"L14.live": np.array(mask, bool), "L14.dominant": np.array(dominant or [1] * n, np.int8),
            "L14.pos_extreme": np.ones(n), "L14.neg_extreme": np.ones(n)}


def live_all(n):
    return live_mask([True] * n)


# --- the test writer ---

def test_test_writer_prompt_is_blind(tiny_cfg, capture_prompt):
    items = [{"nid": 12345, "pole": 1, "explanation": "fires on prices", "context": "plain"}]
    write_tests(tiny_cfg, capture_prompt.status, items, capture_prompt.out)
    p = capture_prompt.text
    assert "fires on prices" in p and "12345" not in p and "L15" not in p and "[" + "0." not in p


def test_tests_are_written_by_the_test_model(tiny_cfg, capture_prompt):
    """R36: Opus 5.5's safeguards refused every v1 test call, so the tests go to cfg.test_model."""
    write_tests(tiny_cfg, capture_prompt.status, [{"nid": 1, "pole": 1, "explanation": "x", "context": "plain"}], capture_prompt.out)
    assert [c.model for c in capture_prompt.calls] == [tiny_cfg.test_model] == ["claude-opus-5-5"]


def test_prompt_holds_only_anonymous_ids_and_explanation(tiny_cfg, capture_prompt):
    items = [{"nid": 3584 + 77, "layer": 15, "idx": 77, "pole": -1, "explanation": "fires on prices", "context": "reasoning", "job": "one job"},
             {"nid": 3584 + 78, "layer": 15, "idx": 78, "pole": 1, "explanation": "fires on dates", "context": "plain", "job": "unclear"}]
    write_tests(tiny_cfg, capture_prompt.status, items, capture_prompt.out)
    call = capture_prompt.calls[0]
    assert call.system == prompts.TEST_SYSTEM and call.schema == prompts.test_schema(["t01", "t02"])
    assert call.prompt == "=== t01 ===\nDescription: fires on prices\n\n=== t02 ===\nDescription: fires on dates"


def test_replies_map_back_to_their_neurons(tiny_cfg, capture_prompt):
    items = [{"nid": n, "pole": p, "explanation": f"e{n}", "context": "plain", "job": j}
             for n, p, j in ((5, 1, "one job"), (9, -1, "unclear"))]
    out = write_tests(tiny_cfg, capture_prompt.status, items, capture_prompt.out)
    assert [(r["nid"], r["pole"], r["job"]) for r in out] == [(5, 1, "one job"), (9, -1, "unclear")]
    assert out[0]["near_miss"][3]["text"] == "t01 near_miss 3" and out[1]["distant"][9]["text"] == "t02 distant 9"
    assert all(len(r[c]) == 10 for r in out for c in CATEGORIES)


def test_invalid_reply_splits_the_chunk(tiny_cfg, capture_prompt):
    capture_prompt.answer = lambda ids: reply_for(ids, 9 if len(ids) > 1 else 10)
    items = [{"nid": n, "pole": 1, "explanation": "e", "context": "plain"} for n in (1, 2)]
    out = write_tests(tiny_cfg, capture_prompt.status, items, capture_prompt.out)
    assert len(capture_prompt.calls) == 3 and all("failed" not in r for r in out)


def test_item_that_stays_invalid_is_stored_as_failed(tiny_cfg, capture_prompt):
    capture_prompt.answer = lambda ids: reply_for(ids, 9)
    out = write_tests(tiny_cfg, capture_prompt.status, [{"nid": 1, "pole": 1, "explanation": "e", "context": "plain"}], capture_prompt.out)
    assert out[0]["nid"] == 1 and "failed" in out[0]


def test_failed_record_keeps_the_job_and_pole_of_its_explanation(tiny_cfg, capture_prompt):
    capture_prompt.answer = lambda ids: reply_for(ids, 9)
    items = [{"nid": 1, "pole": 1, "explanation": "e", "context": "plain", "job": "one job"},
             {"nid": 2, "pole": -1, "explanation": "e", "context": "plain"}]
    out = write_tests(tiny_cfg, capture_prompt.status, items, capture_prompt.out)
    assert (out[0]["job"], out[0]["pole"]) == ("one job", 1) and "failed" in out[0]
    assert "job" not in out[1] and out[1]["pole"] == -1 and "failed" in out[1]
    assert write_tests(tiny_cfg, capture_prompt.status, items, capture_prompt.out) == out     # also when read back from disk


def test_finished_batches_are_not_paid_for_twice(tiny_cfg, capture_prompt):
    items = [{"nid": n, "pole": 1, "explanation": "e", "context": "plain"} for n in range(evaluate.TEST_BATCH + 1)]
    first = write_tests(tiny_cfg, capture_prompt.status, items, capture_prompt.out)
    assert len(capture_prompt.calls) == 2
    assert write_tests(tiny_cfg, capture_prompt.status, items, capture_prompt.out) == first and len(capture_prompt.calls) == 2


def test_viewer_payload_follows_ruling_r10(tiny_cfg, capture_prompt):
    items = [{"nid": 12345, "pole": 1, "explanation": "fires on prices", "context": "plain"}]
    write_tests(tiny_cfg, capture_prompt.status, items, capture_prompt.out)
    assert capture_prompt.status.latest_ == {"name": neuron_name(17, 12345 - 3 * 3584), "explanation": "fires on prices",
                                             "score": None, "tokens": []}


# --- what counts as a hit ---

def hit(a, gate=None, eligible=None, start=0, pole=1, extreme=10.0):
    a = torch.tensor(a, dtype=torch.float32)
    gate = torch.ones_like(a) if gate is None else torch.tensor(gate, dtype=torch.float32)
    eligible = torch.ones_like(a, dtype=torch.bool) if eligible is None else torch.tensor(eligible)
    return passage_hit(a, gate, eligible, start, pole, extreme)


def test_hit_needs_30_percent_of_the_extreme():
    assert hit([0.0, 3.0]) and not hit([0.0, 2.99])


def test_negative_pole_uses_the_sign_flipped_activation():
    assert hit([0.0, -3.0], pole=-1) and not hit([0.0, 3.0], pole=-1) and not hit([0.0, -3.0], pole=1)


def test_activation_before_the_passage_does_not_count():
    assert not hit([9.0, 9.0, 1.0], start=2) and hit([9.0, 9.0, 1.0], start=1) and hit([1.0, 1.0, 9.0], start=2)


def test_gate_negative_position_does_not_count():
    assert not hit([9.0, 1.0], gate=[-1.0, 1.0]) and not hit([9.0, 1.0], gate=[0.0, 1.0])


def test_ineligible_position_does_not_count():
    assert not hit([9.0, 1.0], eligible=[False, True])


def test_no_qualifying_position_is_a_miss():
    assert not hit([9.0, 9.0], eligible=[False, False]) and not hit([9.0], start=1)


def test_nan_position_is_ignored_and_does_not_hide_a_real_hit():
    assert hit([float("nan"), 5.0]) and not hit([float("nan"), 1.0]) and not hit([float("inf"), 1.0])


def test_silent_pole_never_hits():
    assert not hit([0.0, 0.0], extreme=0.0) and not hit([1.0, 1.0], extreme=0.0)


# --- passages ---

@pytest.fixture(scope="module")
def real_tok():
    from transformers import AutoTokenizer
    from probe.weights import snapshot_dir
    return AutoTokenizer.from_pretrained(snapshot_dir(), local_files_only=True)


def test_plain_passage_is_bare_and_truncated(real_tok):
    ids, start = example_ids(real_tok, "word " * 400, "plain", Config())
    assert start == 0 and len(ids) == 128 and ids == real_tok("word " * 400, add_special_tokens=False).input_ids[:128]


def test_reasoning_passage_follows_an_open_think_block(real_tok):
    ids, start = example_ids(real_tok, "The answer is 42.", "reasoning", Config())
    prefix = real_tok.apply_chat_template([{"role": "user", "content": "Think about the following."}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=True)
    assert real_tok.decode(ids[:start]) == prefix and prefix.endswith("<think>\n")
    assert real_tok.decode(ids[start:]) == "The answer is 42."


def test_reasoning_passage_truncates_only_the_passage(real_tok):
    ids, start = example_ids(real_tok, "word " * 400, "reasoning", Config())
    assert start > 0 and len(ids) - start == 128


def test_passage_length_and_prefix_come_from_the_hashed_prompt_values(real_tok, monkeypatch):
    before, start = example_ids(real_tok, "word " * 400, "reasoning", Config())
    monkeypatch.setattr(prompts, "PASSAGE_TOKENS", 4)
    monkeypatch.setattr(prompts, "REASONING_USER", "Consider this other request carefully.")
    ids, changed = example_ids(real_tok, "word " * 400, "reasoning", Config())
    assert len(ids) - changed == 4 and ids[:changed] != before[:start]


def test_tokenizer_would_read_a_literal_close_tag_as_the_control_token(real_tok):
    assert real_tok.convert_tokens_to_ids("</think>") in real_tok("a</think>b", add_special_tokens=False).input_ids


@pytest.mark.parametrize("context", prompts.CONTEXTS)
def test_literal_added_tokens_in_a_passage_stay_plain_text(real_tok, context):
    added = real_tok.get_added_vocab()
    assert "</think>" in added and "<|im_start|>" in added
    for token in added:
        for text in (f"a{token}b", f"{token}", f"{token}{token}", f"<{token}>", f"x {token} y"):
            ids, start = example_ids(real_tok, text, context, Config())
            assert not set(ids[start:]) & set(added.values()), text
            assert real_tok.decode(ids[start:]) == text


def test_reasoning_prefix_keeps_its_real_control_tokens(real_tok):
    ids, start = example_ids(real_tok, "a</think>b", "reasoning", Config())
    assert real_tok.convert_tokens_to_ids("<think>") in ids[:start]
    assert real_tok.convert_tokens_to_ids("</think>") not in ids


def test_ordinary_text_tokenizes_as_one_piece(real_tok):
    text = "The cost is <b>$42</b>, isn't it? 价格是四十二。"
    assert example_ids(real_tok, text, "plain", Config())[0] == real_tok(text, add_special_tokens=False).input_ids


def test_unknown_context_is_rejected(real_tok):
    with pytest.raises(ValueError, match="context"):
        example_ids(real_tok, "text", "poem", Config())


# --- scoring, with the model replaced by activations computed from the token ids ---

class NumberTok:
    """Each word is a number and that number is the token id."""
    pad_token_id = 0
    all_special_ids = [99]

    def get_added_vocab(self):
        return {}

    def __call__(self, text, add_special_tokens=True):
        return SimpleNamespace(input_ids=[int(w) for w in text.split()])

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return "90 91 92"

    def convert_tokens_to_ids(self, token):
        return {"<|im_start|>": 98, "<|im_end|>": 97, "<think>": 96, "</think>": 95}[token]


@pytest.fixture
def fake_forward(monkeypatch):
    """Neuron 0: a = token id. Neuron 1: a = -token id. Neuron 2: as neuron 0 with a negative gate. Neuron 3: as neuron 0."""
    batches, hooks = [], {}

    @contextmanager
    def capture(model, layers, on_layer):
        assert list(layers) == [15]
        hooks["on_layer"] = on_layer
        yield

    def run_batch(model, ids, attention):
        batches.append(ids.shape)
        x = ids.float()
        a = torch.stack([x, -x, x, x], -1)
        gate = torch.stack([torch.ones_like(x), torch.ones_like(x), -torch.ones_like(x), torch.ones_like(x)], -1)
        hooks["on_layer"](15, a, gate, a)

    monkeypatch.setattr(activations, "capture", capture)
    monkeypatch.setattr(activations, "run_batch", run_batch)
    return batches


SCORE_SUMMARY = {"L15.pos_extreme": np.array([10.0, 0.0, 10.0, 100.0]), "L15.neg_extreme": np.array([0.0, 10.0, 0.0, 0.0])}


def passages(*spec):
    return [{"text": text, "context": context} for text, context, n in spec for _ in range(n)]


def scored(cfg, idx, pole, blind=(), near=(), distant=()):
    test = {"nid": neuron_id(cfg, 15, idx), "pole": pole, "job": "one job",
            "blind_maximum": passages(*blind), "near_miss": passages(*near), "distant": passages(*distant)}
    return score_tests(cfg, torch.nn.Linear(1, 1), NumberTok(), 15, [test], SCORE_SUMMARY)[0]


def test_score_counts_hits_per_category(tiny_cfg, fake_forward):
    r = scored(tiny_cfg, 0, 1, blind=[("1 5", "plain", 10)], near=[("1 5", "plain", 3), ("1 2", "plain", 7)], distant=[("1 1", "plain", 10)])
    assert (r["blind_maximum"], r["near_miss"], r["distant"]) == (10, 3, 0)
    assert (r["nid"], r["pole"], r["job"]) == (neuron_id(tiny_cfg, 15, 0), 1, "one job")


def test_score_skips_the_first_token_of_a_plain_passage(tiny_cfg, fake_forward):
    assert scored(tiny_cfg, 0, 1, blind=[("5 1", "plain", 10)])["blind_maximum"] == 0


@pytest.mark.skip(reason="R38: scoring runs every passage as plain text, so no reasoning prefix is added")
def test_score_ignores_the_reasoning_prefix(tiny_cfg, fake_forward):
    r = scored(tiny_cfg, 0, 1, blind=[("1 1", "reasoning", 10)], near=[("5 1", "reasoning", 10)])
    assert r["blind_maximum"] == 0     # the prefix's ids are about 90, far above the threshold of 3
    assert r["near_miss"] == 10        # a reasoning passage's first token is not a document's first token


def test_score_uses_the_tested_poles_extreme_and_sign(tiny_cfg, fake_forward):
    assert scored(tiny_cfg, 1, -1, blind=[("1 5", "plain", 10)])["blind_maximum"] == 10
    assert scored(tiny_cfg, 0, -1, blind=[("1 5", "plain", 10)])["blind_maximum"] == 0
    assert scored(tiny_cfg, 3, 1, blind=[("1 5", "plain", 10)])["blind_maximum"] == 0


def test_score_ignores_gate_negative_activation(tiny_cfg, fake_forward):
    assert scored(tiny_cfg, 2, 1, blind=[("1 5", "plain", 10)])["blind_maximum"] == 0


def test_padding_does_not_change_a_score(tiny_cfg, fake_forward):
    r = scored(tiny_cfg, 0, 1, blind=[("1 5", "plain", 5), ("1 1 1 1 1 1 5", "plain", 5)])
    assert r["blind_maximum"] == 10 and fake_forward[0][0] == 10


def test_score_batches_stay_within_both_limits(tiny_cfg, fake_forward):
    cfg = replace(tiny_cfg, batch_tokens=64)
    long, short = " ".join(["1"] * 8), "1 1"
    scored(cfg, 0, 1, blind=[(long, "plain", 10)], near=[(short, "plain", 10)], distant=[(short, "plain", 10)])
    assert sum(b for b, _ in fake_forward) == 30
    assert all(b <= cfg.batch_docs and b * t <= cfg.batch_tokens for b, t in fake_forward)


def test_score_batches_hold_at_most_32_passages(tiny_cfg, fake_forward):
    scored(tiny_cfg, 0, 1, blind=[("1 1", "plain", 40)])
    assert sorted(b for b, _ in fake_forward) == [8, 32]


def test_failed_record_passes_through_scoring(tiny_cfg, fake_forward):
    failed = {"nid": neuron_id(tiny_cfg, 15, 0), "failed": "declined"}
    assert score_tests(tiny_cfg, torch.nn.Linear(1, 1), NumberTok(), 15, [failed], SCORE_SUMMARY) == [failed]
    assert fake_forward == []


def test_score_leaves_the_stored_tests_unchanged(tiny_cfg, fake_forward):
    test = {"nid": neuron_id(tiny_cfg, 15, 0), "pole": 1, "blind_maximum": passages(("1 5", "plain", 1)), "near_miss": [], "distant": []}
    score_tests(tiny_cfg, torch.nn.Linear(1, 1), NumberTok(), 15, [test], SCORE_SUMMARY)
    assert test["blind_maximum"] == passages(("1 5", "plain", 1))


def test_score_rejects_a_neuron_from_another_layer(tiny_cfg, fake_forward):
    test = {"nid": neuron_id(tiny_cfg, 16, 0), "pole": 1, "blind_maximum": [], "near_miss": [], "distant": []}
    with pytest.raises(ValueError, match="layer 15"):
        score_tests(tiny_cfg, torch.nn.Linear(1, 1), NumberTok(), 15, [test], SCORE_SUMMARY)


# --- the headline scores ---

def test_summary_counts_failed_as_zero_in_primary_only():
    recs = [rec(0, hits=10, dom=1, job="one job"), {"nid": 1, "failed": "declined"}, rec(2, hits=10, dom=1, job="unclear")]
    s = summarize(recs, live_all(3))
    assert s["primary"]["n"] == 3 and s["primary"]["blind_maximum"] == pytest.approx(2 / 3)
    assert s["ac_matched"]["n"] == 1 and s["ac_matched"]["blind_maximum"] == 1.0


def test_ac_matched_excludes_negative_dominant_and_dead():
    recs = [rec(0, hits=5, dom=-1, job="one job"), rec(1, hits=5, dom=1, job="one job")]
    assert summarize(recs, live_mask([True, True]))["ac_matched"]["n"] == 1


def test_negative_dominant_neuron_counts_in_primary_but_not_ac_matched():
    recs = [rec(0, hits=4, dom=-1, job="one job"), rec(1, hits=8, dom=1, job="one job")]
    s = summarize(recs, live_mask([True, True], dominant=[-1, 1]))
    assert s["primary"]["n"] == 2 and s["primary"]["blind_maximum"] == pytest.approx(0.6)
    assert s["ac_matched"]["n"] == 1 and s["ac_matched"]["blind_maximum"] == pytest.approx(0.8)


def test_dead_neurons_count_in_neither_score():
    recs = [rec(0, hits=10, dom=1, job="one job"), rec(1, hits=0, dom=1, job="one job")]
    s = summarize(recs, live_mask([True, False]))
    assert s["primary"]["n"] == 1 and s["primary"]["blind_maximum"] == 1.0
    assert s["ac_matched"]["n"] == 1 and s["ac_matched"]["blind_maximum"] == 1.0


def test_live_neuron_without_a_record_is_zero_in_primary_and_absent_from_ac_matched():
    s = summarize([rec(0, hits=10, dom=1, job="one job")], live_all(2))
    assert s["primary"]["n"] == 2 and s["primary"]["blind_maximum"] == pytest.approx(0.5)
    assert s["ac_matched"]["n"] == 1 and s["ac_matched"]["blind_maximum"] == 1.0


def test_record_for_the_other_pole_does_not_score_the_dominant_pole():
    s = summarize([rec(0, hits=10, dom=-1, job="one job")], live_all(1))
    assert s["primary"] == {"n": 1, "blind_maximum": 0.0, "near_miss": 0.0, "distant": 0.0}
    assert s["ac_matched"]["n"] == 0


def test_each_category_has_its_own_share():
    r = {"nid": 0, "pole": 1, "job": "one job", "blind_maximum": 8, "near_miss": 3, "distant": 1}
    assert summarize([r], live_all(1))["primary"] == {"n": 1, "blind_maximum": 0.8, "near_miss": 0.3, "distant": 0.1}


def test_neuron_ids_follow_the_layer_order():
    summary = {f"L15.{k}": v for k, v in (("live", np.ones(1, bool)), ("dominant", np.ones(1, np.int8)))}
    assert summarize([rec(neuron_id(Config(), 15, 0), hits=10, dom=1, job="one job")], summary)["primary"]["blind_maximum"] == 1.0


def test_duplicate_records_are_rejected():
    with pytest.raises(ValueError, match="more than one record"):
        summarize([rec(0, hits=1, dom=1, job="one job"), rec(0, hits=2, dom=1, job="one job")], live_all(1))


def test_empty_score_has_zero_shares():
    assert summarize([], live_mask([False]))["ac_matched"] == {"n": 0, "blind_maximum": 0.0, "near_miss": 0.0, "distant": 0.0,
                                                               "excluded_unknown_job": 0}


def failed(nid, job=None, pole=1):
    return {"nid": nid, "failed": "declined", "pole": pole, **({"job": job} if job else {})}


def test_hand_computed_scores():
    """Neuron by neuron (hits as blind/near/distant):
    0  scored, positive, one job, 8/2/1    primary and AC-matched
    1  failed, job known                   zero in primary and AC-matched
    2  failed, job unknown                 zero in primary, excluded from AC-matched
    3  scored, positive, unclear, 10/4/0   primary only
    4  scored, negative dominant, 6/0/3    primary only
    5  dead                                neither
    6  live, no record                     zero in primary, excluded from AC-matched

    primary: n = 6, blind (8+10+6)/60 = 0.4, near (2+4)/60 = 0.1, distant (1+3)/60 = 1/15
    AC-matched: n = 2 (neurons 0 and 1), blind 8/20 = 0.4, near 2/20 = 0.1, distant 1/20 = 0.05, 2 excluded (neurons 2 and 6)
    """
    def scored_rec(nid, pole, job, blind, near, distant):
        return {"nid": nid, "pole": pole, "job": job, "blind_maximum": blind, "near_miss": near, "distant": distant}

    recs = [scored_rec(0, 1, "one job", 8, 2, 1), failed(1, job="one job"), {"nid": 2, "failed": "declined"},
            scored_rec(3, 1, "unclear", 10, 4, 0), scored_rec(4, -1, "one job", 6, 0, 3)]
    s = summarize(recs, live_mask([True, True, True, True, True, False, True], dominant=[1, 1, 1, 1, -1, 1, 1]))
    assert s["primary"] == pytest.approx({"n": 6, "blind_maximum": 0.4, "near_miss": 0.1, "distant": 1 / 15})
    assert s["ac_matched"] == pytest.approx({"n": 2, "blind_maximum": 0.4, "near_miss": 0.1, "distant": 0.05, "excluded_unknown_job": 2})


def test_failed_neuron_with_a_known_job_is_zero_in_ac_matched():
    s = summarize([rec(0, hits=10, dom=1, job="one job"), failed(1, job="two related jobs")], live_all(2))
    assert s["ac_matched"]["n"] == 2 and s["ac_matched"]["blind_maximum"] == pytest.approx(0.5)
    assert s["ac_matched"]["excluded_unknown_job"] == 0


def test_failed_neuron_labeled_unclear_is_outside_ac_matched_and_not_unknown():
    s = summarize([failed(0, job="unclear")], live_all(1))
    assert s["ac_matched"]["n"] == 0 and s["ac_matched"]["excluded_unknown_job"] == 0


def test_failed_neuron_without_a_job_is_reported_as_excluded():
    s = summarize([{"nid": 0, "failed": "declined"}], live_all(1))
    assert s["ac_matched"]["n"] == 0 and s["ac_matched"]["excluded_unknown_job"] == 1 and s["primary"]["n"] == 1


def test_job_of_the_other_pole_does_not_count_as_known():
    s = summarize([failed(0, job="one job", pole=-1)], live_all(1))
    assert s["ac_matched"]["n"] == 0 and s["ac_matched"]["excluded_unknown_job"] == 1


def test_negative_dominant_neuron_is_never_reported_as_excluded():
    s = summarize([{"nid": 0, "failed": "declined"}], live_mask([True], dominant=[-1]))
    assert s["ac_matched"]["excluded_unknown_job"] == 0


def two_layers():
    one = {"live": np.ones(1, bool), "dominant": np.ones(1, np.int8)}
    return {f"L{layer}.{k}": v for layer in (14, 15) for k, v in one.items()}


def test_layers_restrict_both_scores():
    recs = [rec(neuron_id(Config(), 14, 0), hits=10, dom=1, job="one job")]
    s = summarize(recs, two_layers(), layers=[14])
    assert s["primary"]["n"] == 1 and s["primary"]["blind_maximum"] == 1.0
    assert s["ac_matched"]["n"] == 1 and s["ac_matched"]["excluded_unknown_job"] == 0


def test_all_layers_count_by_default():
    recs = [rec(neuron_id(Config(), 14, 0), hits=10, dom=1, job="one job")]
    s = summarize(recs, two_layers())
    assert s["primary"]["n"] == 2 and s["primary"]["blind_maximum"] == pytest.approx(0.5)
    assert s["ac_matched"]["n"] == 1 and s["ac_matched"]["excluded_unknown_job"] == 1


def test_no_layers_means_no_neurons():
    assert summarize([], two_layers(), layers=[])["primary"]["n"] == 0


# --- the real model ---

@pytest.fixture
def tiny_summary():
    n = Config().n_neurons
    return {"L15.pos_extreme": np.zeros(n), "L15.neg_extreme": np.zeros(n)}


def _best(signed, usable):
    """Per neuron, the largest usable value, or -inf where a neuron has no usable position."""
    return signed.masked_fill(~usable, -torch.inf).amax(0)


@pytest.mark.skip(reason="R38: test passages always run as plain text, so scoring never adds the reasoning prefix")
@pytest.mark.gpu
def test_score_counts_only_passage_positions(base_model, tiny_summary):
    model, tok = base_model
    cfg = Config()
    text = "The answer is 42."
    ids, start = example_ids(tok, text, "reasoning", cfg)
    assert start > 0 and len(ids) > start

    device = next(model.parameters()).device
    batch, attention = activations.pad_right([np.array(ids)], tok.pad_token_id, device)
    eligible = activations.eligible_mask(batch, attention, activations.excluded_ids(tok), tok.convert_tokens_to_ids("<|im_start|>"))[0].cpu()
    seen = {}

    def on_layer(layer, a, gate, value):
        seen["a"], seen["gate"] = a[0].cpu(), gate[0].cpu()

    with activations.capture(model, [15], on_layer):
        activations.run_batch(model, batch, attention)
    a, gate = seen["a"], seen["gate"]

    # A real neuron and pole that fire harder in the prefix than anywhere in the passage, and still fire in the passage.
    usable = eligible[:, None] & (gate > 0)
    choices = []
    for pole in (1, -1):
        signed = a * pole
        in_prefix, in_passage = _best(signed[:start], usable[:start]), _best(signed[start:], usable[start:])
        gap = (in_prefix - in_passage).masked_fill(~(in_passage > 0) | ~in_prefix.isfinite(), -torch.inf)
        idx = int(gap.argmax())
        choices.append((float(gap[idx]), pole, idx, float(in_prefix[idx]), float(in_passage[idx])))
    _, pole, idx, in_prefix, in_passage = max(choices)
    assert in_prefix > 2 * in_passage > 0

    key = "L15.pos_extreme" if pole == 1 else "L15.neg_extreme"  # gitleaks:allow (a summary-array name, not a credential)
    test = {"nid": neuron_id(cfg, 15, idx), "pole": pole, "near_miss": [], "distant": [],
            "blind_maximum": [{"text": text, "context": "reasoning"}] * 10}

    # The threshold sits between the two peaks: only the prefix reaches it.
    tiny_summary[key][idx] = (in_prefix + in_passage) / 2 / 0.3
    assert score_tests(cfg, model, tok, 15, [test], tiny_summary)[0]["blind_maximum"] == 0
    assert passage_hit(a[:, idx], gate[:, idx], eligible, 0, pole, float(tiny_summary[key][idx]))     # counted if the prefix were included

    # Control: with the threshold under the passage's own peak, the same passages hit.
    tiny_summary[key][idx] = in_passage / 2 / 0.3
    assert score_tests(cfg, model, tok, 15, [test], tiny_summary)[0]["blind_maximum"] == 10


# --- R38: test passages always run as plain text ---

def test_reasoning_context_poles_get_their_own_row_and_stay_in_primary():
    recs = [{**rec(0, hits=10, dom=1, job="one job"), "tested_out_of_context": True}, rec(1, hits=0, dom=1, job="one job")]
    s = summarize(recs, live_all(2))
    assert s["primary"]["n"] == 2 and s["primary"]["blind_maximum"] == pytest.approx(0.5)
    assert s["reasoning_context"]["n"] == 1 and s["reasoning_context"]["blind_maximum"] == 1.0


def test_scoring_runs_every_passage_as_plain_text(tiny_cfg, fake_forward, monkeypatch):
    import probe.atlas.evaluate as ev
    seen = []
    real = ev.example_ids
    monkeypatch.setattr(ev, "example_ids", lambda tok, text, context, cfg: seen.append(context) or real(tok, text, context, cfg))
    test = {"nid": neuron_id(tiny_cfg, 15, 0), "pole": 1, "blind_maximum": [{"text": "1 5"}], "near_miss": [], "distant": []}
    score_tests(tiny_cfg, torch.nn.Linear(1, 1), NumberTok(), 15, [test], SCORE_SUMMARY)
    assert seen == ["plain"]


def test_the_test_writer_sees_no_context():
    text = prompts.test_prompt([("t01", {"explanation": "fires on X", "context": "reasoning"})])
    assert "reasoning" not in text and "Context" not in text and "fires on X" in text
    assert "private thinking" not in prompts.TEST_SYSTEM and "reasoning" not in prompts.TEST_SYSTEM
