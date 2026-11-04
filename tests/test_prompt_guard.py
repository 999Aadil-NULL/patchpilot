import json

import pytest

from agent import config, llm
from agent.config import RoleConfig
from agent.llm import Completion
from scripts import audit_context


class FakeProvider:
    def __init__(self):
        self.prompts = []

    def complete(self, messages, temperature, max_tokens):
        self.prompts.append(messages)
        return Completion(text="OK", tokens_in=1, tokens_out=1, finish_reason="stop")


@pytest.fixture
def harness(monkeypatch):
    """complete() with no network or disk: returns the captured events and the fake provider."""
    events, provider = [], FakeProvider()
    monkeypatch.setattr(llm, "emit", lambda run_id, stage, type, **kw: events.append({"type": type, "stage": stage, **kw}))
    monkeypatch.setattr(llm, "_provider_for", lambda cfg: provider)
    monkeypatch.setattr(llm, "role_config", lambda role: RoleConfig("http://x", "k", "m"))
    monkeypatch.setattr(config, "LLM_CACHE", False)
    monkeypatch.setattr(config, "LLM_MAX_PROMPT_TOKENS", 100)
    llm.set_run_context("run", "patching")
    yield events, provider
    llm.set_run_context(None)


def warnings(events):
    return [e for e in events if e["type"] == "error"]


def test_oversized_prompt_emits_event_and_still_sends_it_whole(harness):
    events, provider = harness
    big = "x" * 3500
    assert llm.complete("coder", [{"role": "user", "content": big}]) == "OK"
    assert warnings(events) == [{"type": "error", "stage": "patching", "error": "prompt_near_limit", "warning": True,
                                 "role": "coder", "estimated_tokens": 1000, "limit": 100}]
    assert provider.prompts[0][0]["content"] == big


def test_normal_prompt_emits_nothing(harness):
    events, _ = harness
    llm.complete("planner", [{"role": "user", "content": "hello"}])
    assert warnings(events) == []


def test_estimate_sums_all_messages():
    msgs = [{"role": "system", "content": "a" * 35}, {"role": "user", "content": "b" * 35}, {"role": "user", "content": None}]
    assert llm._estimate_prompt_tokens(msgs) == 20


def test_audit_reports_guard_events(tmp_path, monkeypatch, capsys):
    d = tmp_path / "r1"
    d.mkdir()
    events = [{"type": "error", "error": "prompt_near_limit", "role": "planner", "estimated_tokens": 31000},
              {"type": "error", "error": "boom"},
              {"type": "llm_call", "role": "planner", "model": "m", "tokens_in": 500, "cached": False}]
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    monkeypatch.setattr(audit_context, "RUNS_DIR", tmp_path)
    assert audit_context.main() == 0
    out = capsys.readouterr().out
    assert "prompt_near_limit events: 1" in out and "planner: n=1 max estimate=31000" in out
