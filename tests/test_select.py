import pytest

from agent import config, llm
from agent.config import RoleConfig
from agent.llm import Completion
from agent.orchestrator import Orchestrator
from agent.schemas import ExecResult, Issue, Plan, RunState
from agent.schemas import TestResult as Result
from agent.select import normalize_diff, select


def diff(*changes):
    return "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n context\n" + "\n".join(changes) + "\n"


A = diff("-x = 1", "+x = 2")
A_SAME = diff("-x = 1", "+x  =  2   # fixed", "+", "+# note")
B = diff("-x = 1", "+x = 3", "+y = 4")
C = diff("-x = 1", "+z = 3")


def cand(d=A, **kw):
    return {"diff": d, "clean_apply": True, "lint_clean": True, "repro_passed": True, "regression_passed": True,
            "new_failures": 0, **kw}


def test_normalize_ignores_whitespace_blank_lines_comments_and_headers():
    assert normalize_diff(A) == normalize_diff(A_SAME) == "-x=1\n+x=2"
    assert normalize_diff(A) != normalize_diff(B)


def test_passing_beats_failing():
    idx, _ = select([cand(repro_passed=False, regression_passed=None, new_failures=None), cand(B)])
    assert idx == 1


def test_repro_pass_with_regression_failure_is_not_passing():
    idx, reason = select([cand(regression_passed=False, new_failures=2), cand(B)])
    assert idx == 1 and reason == "only passing candidate"


def test_majority_beats_smaller_diff():
    idx, reason = select([cand(C), cand(B), cand(B)])
    assert idx == 1 and "majority: 2 of 3" in reason


def test_majority_uses_normalized_diff():
    idx, reason = select([cand(C), cand(A), cand(A_SAME)])
    assert idx == 1 and "majority" in reason


def test_tie_takes_smallest_diff():
    idx, reason = select([cand(B), cand(A), cand(C + "\n\n\n\n")])
    assert idx == 1 and "smallest diff" in reason


def test_none_pass_prefers_clean_apply_then_lint_then_fewest_failures():
    bad_apply = cand(clean_apply=False, repro_passed=None, regression_passed=None, new_failures=None)
    bad_lint = cand(lint_clean=False, repro_passed=None, regression_passed=None, new_failures=None)
    many = cand(regression_passed=False, new_failures=5)
    few = cand(regression_passed=False, new_failures=1)
    assert select([bad_apply, bad_lint])[0] == 1
    assert select([bad_lint, many])[0] == 1
    assert select([bad_apply, many, few])[0] == 2
    assert "best failure" in select([bad_apply, many, few])[1]


def test_measured_failures_beat_regression_that_never_ran():
    never_ran = cand(repro_passed=False, regression_passed=None, new_failures=None)
    assert select([never_ran, cand(regression_passed=False, new_failures=9)])[0] == 1


def test_regression_only_mode_uses_regression_and_majority():
    r = dict(repro_passed=None)
    idx, reason = select([cand(C, **r), cand(A, **r), cand(A_SAME, **r)])
    assert idx == 1 and "majority" in reason


def test_no_signal_is_not_passing():
    idx, reason = select([cand(repro_passed=None, regression_passed=None, new_failures=None)])
    assert idx == 0 and "best failure" in reason


def test_selection_never_reads_hidden_fields():
    poisoned = cand(C, FAIL_TO_PASS=["x"], resolved=True, patch="gold")
    assert select([cand(B), cand(B), poisoned]) == select([cand(B), cand(B), cand(C)])


# ---- cache keys
CFG = RoleConfig("http://x", "k", "m")
MSG = [{"role": "user", "content": "hi"}]


def test_sample_zero_keeps_the_v1_cache_key():
    import hashlib
    import json
    blob = json.dumps({"base_url": "http://x", "model": "m", "messages": MSG, "temperature": 0.2, "max_tokens": 10},
                      sort_keys=True, ensure_ascii=False)
    assert llm._cache_key(CFG, MSG, 0.2, 10) == hashlib.sha256(blob.encode()).hexdigest()
    assert llm._cache_key(CFG, MSG, 0.2, 10, 0) == llm._cache_key(CFG, MSG, 0.2, 10)


def test_samples_get_distinct_keys():
    assert len({llm._cache_key(CFG, MSG, 0.7, 10, i) for i in range(3)}) == 3


def test_complete_passes_sample_idx_to_the_cache(monkeypatch, tmp_path):
    class P:
        n = 0

        def complete(self, *a):
            P.n += 1
            return Completion(text=f"reply {P.n}", tokens_in=1, tokens_out=1, finish_reason="stop")
    monkeypatch.setattr(llm, "_provider_for", lambda cfg: P())
    monkeypatch.setattr(llm, "role_config", lambda r: CFG)
    monkeypatch.setattr(llm, "LLM_CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "LLM_CACHE", True)
    first = [llm.complete("coder", MSG, sample_idx=i) for i in range(3)]
    assert first == ["reply 1", "reply 2", "reply 3"]
    assert [llm.complete("coder", MSG, sample_idx=i) for i in range(3)] == first and P.n == 3


# ---- sampled attempt
GOOD = "def add(a, b):\n    return a + b\n"


def reply(new):
    return f"FILE: pkg/mod.py\n<<<<<<< SEARCH\n    return a + b\n=======\n    return {new}\n>>>>>>> REPLACE\n"


class Sandbox:
    def __init__(self):
        self.files = {"pkg/mod.py": GOOD}
        self.resets = 0

    def exec(self, cmd, timeout=300):
        return ExecResult(exit_code=0, stdout="", stderr="", duration_s=0)

    def read_file(self, p):
        return self.files[p]

    def write_file(self, p, c):
        self.files[p] = c

    def reset(self):
        self.files = {"pkg/mod.py": GOOD}
        self.resets += 1

    def diff(self):
        new = self.files["pkg/mod.py"].splitlines()[1].strip()
        return "" if self.files["pkg/mod.py"] == GOOD else f"-    return a + b\n+    {new}\n"

    def run_repro(self, script):
        ok = "a - b" in self.files["pkg/mod.py"]
        return Result(passed=ok, failure_summary="" if ok else "AssertionError",
                      exec=ExecResult(exit_code=0 if ok else 1, stdout="", stderr="", duration_s=0))


@pytest.fixture
def orch(monkeypatch):
    o = Orchestrator.__new__(Orchestrator)
    o.issue = Issue(instance_id="a__b-1", repo="a/b", base_commit="x", version="1", problem_statement="bug")
    o.run_id = "r"
    o.state = RunState(run_id="r", issue=o.issue, repro_script="assert 1")
    o.sb = Sandbox()
    o.context_text = o.patch_context = ""
    o._reflections, o._new_failures, o._repro_fixes, o._baseline = {}, {}, 0, {}
    o.events = []
    monkeypatch.setattr(o, "_stage", lambda s: None, raising=False)
    monkeypatch.setattr(o, "ev", lambda t, **kw: o.events.append({"type": t, **kw}), raising=False)
    monkeypatch.setattr(o, "_regression_files", lambda att: [], raising=False)
    monkeypatch.setattr(o, "_reflect", lambda att, plan: o.events.append({"message": "reflected", "tree": dict(o.sb.files)}),
                        raising=False)
    return o


PLAN = Plan(kind="bug", summary="s", hypothesis="h", steps=[], acceptance_criteria=[], search_terms=[])


def script(monkeypatch, replies):
    calls = []

    def fake(role, messages, **kw):
        calls.append(kw)
        return replies[len(calls) - 1]
    monkeypatch.setattr(llm, "complete", fake)
    return calls


def test_k3_makes_three_calls_and_selects_the_passing_one(orch, monkeypatch):
    monkeypatch.setattr(config, "HARNESS_SAMPLES", 3)
    calls = script(monkeypatch, [reply("a * b"), reply("a - b"), reply("a / b")])
    att = orch._attempt(1, PLAN)
    assert [c["sample_idx"] for c in calls] == [0, 1, 2]
    assert [c["temperature"] for c in calls] == [0.2, 0.7, 0.7]
    assert att.repro.passed and "a - b" in att.diff and len(att.candidates) == 3
    assert att.candidates[1]["selected"] and att.candidates[1]["reason"] == "only passing candidate"
    sel = [e for e in orch.events if e.get("message") == "candidate_selection"]
    assert sel[0]["selected"] == 1 and sel[0]["k"] == 3
    assert orch.sb.files["pkg/mod.py"] == "def add(a, b):\n    return a - b\n"  # chosen one is back on the tree


def test_all_fail_reflects_on_the_chosen_candidate(orch, monkeypatch):
    monkeypatch.setattr(config, "HARNESS_SAMPLES", 2)
    script(monkeypatch, [reply("a * b"), reply("a / b")])
    att = orch._attempt(1, PLAN)
    assert not att.repro.passed
    reflected = [e for e in orch.events if e.get("message") == "reflected"][0]
    assert "a * b" in reflected["tree"]["pkg/mod.py"]


def test_repro_failure_skips_regression(orch, monkeypatch):
    monkeypatch.setattr(config, "HARNESS_SAMPLES", 2)
    monkeypatch.setattr(orch, "_regression_files", lambda att: pytest.fail("regression ran for a repro failure"))
    script(monkeypatch, [reply("a * b"), reply("a / b")])
    orch._attempt(1, PLAN)


def test_unapplied_candidate_is_kept_with_its_errors(orch, monkeypatch):
    monkeypatch.setattr(config, "HARNESS_SAMPLES", 2)
    script(monkeypatch, ["no blocks here", reply("a - b")])
    att = orch._attempt(1, PLAN)
    assert att.candidates[0]["clean_apply"] is False and att.candidates[0]["apply_errors"]
    assert att.repro.passed


def test_k1_takes_the_v1_path(orch, monkeypatch):
    monkeypatch.setattr(config, "HARNESS_SAMPLES", 1)
    calls = script(monkeypatch, [reply("a - b")])
    att = orch._attempt(2, PLAN)
    assert calls == [{"temperature": 0.5, "max_tokens": 8192}]
    assert att.candidates == [] and att.repro.passed
