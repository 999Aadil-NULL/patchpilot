from types import SimpleNamespace

import pytest

from agent import config, llm, orchestrator
from agent.guards import describe_errors, error_region, syntax_check
from agent.orchestrator import Orchestrator
from agent.schemas import Attempt, ExecResult, Issue, Plan, RunState

GOOD = "def add(a, b):\n    return a + b\n"
FILLER = "".join(f"x{i} = {i}\n" for i in range(30))
BROKEN_REPLY = "FILE: pkg/mod.py\n<<<<<<< SEARCH\n    return a + b\n=======\n    return (a + b\n>>>>>>> REPLACE\n"
FIXED_REPLY = "FILE: pkg/mod.py\n<<<<<<< SEARCH\n    return a + b\n=======\n    return a - b\n>>>>>>> REPLACE\n"
NO_MATCH_REPLY = "FILE: pkg/mod.py\n<<<<<<< SEARCH\n    not in the file\n=======\n    x\n>>>>>>> REPLACE\n"


class FakeSandbox:
    """In-memory tree. exec compiles with the host Python, in py_compile's output format."""

    def __init__(self, files=None):
        self.original = dict(files or {"pkg/mod.py": GOOD})
        self.files = dict(self.original)
        self.commands = []

    def exec(self, cmd, timeout=300):
        self.commands.append((cmd, timeout))
        path = cmd.split()[-1].strip("'")
        try:
            compile(self.files[path], path, "exec")
        except SyntaxError as e:
            msg = f'  File "{path}", line {e.lineno}\n    {e.text}\nSyntaxError: {e.msg}'
            return ExecResult(exit_code=1, stdout="", stderr=msg, duration_s=0.1)
        return ExecResult(exit_code=0, stdout="", stderr="", duration_s=0.1)

    def read_file(self, path):
        return self.files[path]

    def write_file(self, path, content):
        self.files[path] = content

    def reset(self):
        self.files = dict(self.original)

    def diff(self):
        return "" if self.files == self.original else "diff --git a/pkg/mod.py b/pkg/mod.py\n"

    def run_repro(self, script):
        raise AssertionError("not used")


def test_syntax_check_passes_good_and_skips_non_python():
    sb = FakeSandbox({"a.py": GOOD, "notes.txt": "((("})
    assert syntax_check(sb, ["a.py", "notes.txt"]) == []
    assert len(sb.commands) == 1 and sb.commands[0][1] == 30


def test_syntax_check_reports_each_broken_file():
    sb = FakeSandbox({"a.py": "def f(:\n", "b.py": GOOD, "c.py": "x = (\n"})
    errors = syntax_check(sb, ["a.py", "b.py", "c.py"])
    assert [e.split(":")[0] for e in errors] == ["a.py", "c.py"]


def test_syntax_check_reports_timeout():
    sb = SimpleNamespace(exec=lambda cmd, timeout: ExecResult(exit_code=124, stdout="", stderr="", duration_s=30,
                                                              timed_out=True))
    assert "timed out" in syntax_check(sb, ["a.py"])[0]


def test_syntax_check_quotes_paths():
    seen = []

    def run(cmd, timeout):
        seen.append(cmd)
        return ExecResult(exit_code=0, stdout="", stderr="", duration_s=0.1)
    syntax_check(SimpleNamespace(exec=run), ["a b.py"])
    assert seen == ["python -m py_compile 'a b.py'"]


def test_error_region_is_ten_lines_each_side():
    region = error_region(FILLER, 'File "a.py", line 15').splitlines()
    assert region[0] == "x4 = 4" and region[-1] == "x24 = 24" and len(region) == 21


def test_error_region_uses_last_line_mention_and_clamps():
    assert error_region("a\nb\n", 'pkg/line 1/x.py: File "x", line 99').splitlines()[-1] == "b"
    assert error_region("a\n", "no location here") == ""


def test_describe_errors_shows_compiler_output_and_code():
    sb = FakeSandbox({"a.py": FILLER + "def f(:\n"})
    text = describe_errors(sb, syntax_check(sb, ["a.py"]))
    assert "SyntaxError" in text and "def f(:" in text and "x29 = 29" in text


# ---- the gate inside an attempt
@pytest.fixture
def orch(monkeypatch):
    o = Orchestrator.__new__(Orchestrator)
    o.issue = Issue(instance_id="a__b-1", repo="a/b", base_commit="x", version="1", problem_statement="bug")
    o.run_id = "r"
    o.state = RunState(run_id="r", issue=o.issue)
    o.sb = FakeSandbox()
    o.context_text = ""
    o._reflections, o._new_failures, o._repro_fixes, o._baseline = {}, {}, 0, {}
    o.events = []
    monkeypatch.setattr(o, "_stage", lambda s: None, raising=False)
    monkeypatch.setattr(o, "ev", lambda t, **kw: o.events.append({"type": t, **kw}), raising=False)
    monkeypatch.setattr(o, "_regression_files", lambda att: [], raising=False)
    return o


PLAN = Plan(kind="bug", summary="s", hypothesis="h", steps=[], acceptance_criteria=[], search_terms=[])


def script_llm(monkeypatch, replies):
    calls = []

    def fake(role, messages, **kw):
        calls.append(messages)
        return replies[len(calls) - 1]
    monkeypatch.setattr(llm, "complete", fake)
    return calls


def lint_events(o):
    return [e for e in o.events if str(e.get("message", "")).startswith("lint_")]


def test_broken_edit_is_caught_and_repair_prompt_built(orch, monkeypatch):
    monkeypatch.setattr(config, "HARNESS_LINT", True)
    calls = script_llm(monkeypatch, [BROKEN_REPLY, FIXED_REPLY])
    att = orch._attempt(1, PLAN)
    assert len(calls) == 2
    repair = calls[1]
    assert [m["role"] for m in repair] == ["system", "user", "assistant", "user"]
    assert repair[2]["content"] == BROKEN_REPLY
    assert "does not compile" in repair[3]["content"] and "return (a + b" in repair[3]["content"]
    assert [e["message"] for e in lint_events(orch)] == ["lint_fail", "lint_repaired"]
    assert att.n == 1 and att.apply_errors == [] and att.edits[0].replace == "    return a - b\n"
    assert orch.sb.files["pkg/mod.py"] == "def add(a, b):\n    return a - b\n"  # re-applied on a reset tree


def test_repair_still_broken_is_recorded_as_apply_errors(orch, monkeypatch):
    monkeypatch.setattr(config, "HARNESS_LINT", True)
    calls = script_llm(monkeypatch, [BROKEN_REPLY, BROKEN_REPLY])
    att = orch._attempt(1, PLAN)
    assert len(calls) == 2  # exactly one repair turn
    assert len(att.apply_errors) == 1 and "pkg/mod.py" in att.apply_errors[0]
    assert [e.get("after_repair") for e in lint_events(orch)] == [None, True]


@pytest.mark.parametrize("reply", [NO_MATCH_REPLY, "I cannot fix this."])
def test_repair_that_does_not_apply_is_recorded(orch, monkeypatch, reply):
    monkeypatch.setattr(config, "HARNESS_LINT", True)
    script_llm(monkeypatch, [BROKEN_REPLY, reply])
    att = orch._attempt(1, PLAN)
    assert len(att.apply_errors) == 2 and "SyntaxError" in att.apply_errors[0]
    assert orch.sb.files["pkg/mod.py"] == GOOD


def test_clean_edit_with_flag_on_makes_no_repair_call(orch, monkeypatch):
    monkeypatch.setattr(config, "HARNESS_LINT", True)
    calls = script_llm(monkeypatch, [FIXED_REPLY])
    orch._attempt(1, PLAN)
    assert len(calls) == 1 and lint_events(orch) == []


def test_apply_error_skips_lint(orch, monkeypatch):
    monkeypatch.setattr(config, "HARNESS_LINT", True)
    calls = script_llm(monkeypatch, [NO_MATCH_REPLY])
    orch._attempt(1, PLAN)
    assert len(calls) == 1 and orch.sb.commands == []


def test_flag_off_never_lints(orch, monkeypatch):
    monkeypatch.setattr(config, "HARNESS_LINT", False)
    monkeypatch.setattr(orchestrator, "syntax_check", lambda *a: pytest.fail("lint ran with flag off"))
    calls = script_llm(monkeypatch, [BROKEN_REPLY])
    att = orch._attempt(1, PLAN)
    assert len(calls) == 1 and orch.sb.commands == [] and lint_events(orch) == []
    assert att.apply_errors == [] and "return (a + b" in orch.sb.files["pkg/mod.py"]  # v1 lets it through to verify


def test_repair_does_not_add_an_attempt(orch, monkeypatch):
    monkeypatch.setattr(config, "HARNESS_LINT", True)
    script_llm(monkeypatch, [BROKEN_REPLY, FIXED_REPLY])
    assert isinstance(orch._attempt(1, PLAN), Attempt) and orch.state.attempts == []


def test_run_config_records_flags_and_models(monkeypatch):
    monkeypatch.setattr(config, "role_config", lambda role: SimpleNamespace(model=f"{role}-m"))
    cfg = config.run_config()
    assert set(cfg) == {"HARNESS_LINT", "HARNESS_FOCUS", "HARNESS_SAMPLES", "HARNESS_SBFL", "planner_model", "coder_model"}
    assert cfg["coder_model"] == "coder-m"
