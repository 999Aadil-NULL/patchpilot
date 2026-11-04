import json
import subprocess
import sys

import pytest

from agent import config
from agent.localize import trace as tr
from agent.localize.repomap import chunk_file
from agent.localize.trace import TracedCall, refine_context, top_functions, trace_repro
from agent.orchestrator import Orchestrator
from agent.schemas import ContextBundle, ExecResult, Issue, RunState

SRC = '''def helper(x):
    return x + 1


class Calc:
    def add(self, a):
        return helper(a)


class Other:
    def add(self, a):
        return a
'''


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(SRC)
    return tmp_path


def call(name, line, count=1, path="pkg/mod.py"):
    return TracedCall(path, name, line, count)


def test_runner_is_valid_and_records_calls(tmp_path):
    """Runs the real runner with the host Python, pointed at a temp root."""
    (tmp_path / "pkg.py").write_text("def hit():\n    return 1\n")
    (tmp_path / "repro_issue.py").write_text("import pkg\npkg.hit()\npkg.hit()\nassert False\n")
    out = tmp_path / "out.json"
    src = tr.RUNNER.replace('"/testbed/"', repr(str(tmp_path) + "/")).replace('"/testbed"', repr(str(tmp_path))) \
        .replace(tr.OUTPUT_PATH, str(out))
    (tmp_path / "runner.py").write_text(src)
    subprocess.run([sys.executable, str(tmp_path / "runner.py")], cwd=tmp_path, timeout=30, check=True)
    rows = json.loads(out.read_text())
    assert ["pkg.py", "hit", 1, 2] in rows
    assert not [r for r in rows if r[0].startswith("<")]  # importlib's "<frozen ...>" is not repo code


def test_runner_avoids_syntax_newer_than_python_36():
    assert 'f"' not in tr.RUNNER and ":=" not in tr.RUNNER


class Box:
    def __init__(self, rows):
        self.rows = rows
        self.written = {}

    def exec(self, cmd, timeout=300):
        return ExecResult(exit_code=1, stdout="", stderr="", duration_s=0)  # the repro fails: that is expected

    def write_file(self, path, content):
        self.written[path] = content

    def read_file(self, path):
        return json.dumps(self.rows)


def test_trace_excludes_tests_and_the_repro():
    rows = [["pkg/mod.py", "f", 1, 3], ["pkg/tests/test_x.py", "t", 1, 9], ["repro_issue.py", "main", 1, 1]]
    assert trace_repro(Box(rows)) == [TracedCall("pkg/mod.py", "f", 1, 3)]


def test_top_functions_prefers_candidate_files_then_counts():
    t = [call("hot", 1, 99, "other.py"), call("a", 1, 2), call("b", 5, 7)]
    assert [c.function for c in top_functions(t, ["pkg/mod.py"], 2)] == ["b", "a"]


def test_refine_adds_at_front_and_matches_method_by_line(repo):
    ctx = ContextBundle(candidate_files=["pkg/mod.py"], chunks=[c for c in chunk_file(repo, "pkg/mod.py")
                                                                if c.symbol == "helper"])
    added = refine_context(ctx, [call("add", 11), call("helper", 1)], repo)
    assert added == ["pkg/mod.py::Other.add"]  # same method name in two classes: the line decides
    assert ctx.chunks[0].symbol == "Other.add" and ctx.chunks[0].score == 1.0 and len(ctx.chunks) == 2


def test_refine_ignores_unknown_functions(repo):
    ctx = ContextBundle(candidate_files=[], chunks=[])
    assert refine_context(ctx, [call("gone", 1), call("x", 1, 1, "missing.py")], repo) == [] and ctx.chunks == []


@pytest.fixture
def orch(repo, monkeypatch):
    o = Orchestrator.__new__(Orchestrator)
    o.issue = Issue(instance_id="", repo="a/b", base_commit="x", version="1", problem_statement="bug")
    o.run_id = "r"
    o.state = RunState(run_id="r", issue=o.issue, repro_script="assert 0")
    o.state.context = ContextBundle(candidate_files=["pkg/mod.py"], chunks=[])
    o.repo_dir = repo
    o.context_text = "OLD"
    o.events = []
    monkeypatch.setattr(o, "ev", lambda t, **kw: o.events.append(kw), raising=False)
    return o


def test_no_repro_is_a_noop(orch):
    orch.state.repro_script = ""
    orch.sb = Box([])
    orch._refine_from_trace()
    assert orch.sb.written == {} and orch.events[-1]["skipped"] == "no valid repro"


def test_trace_feeds_the_context(orch):
    orch.sb = Box([["pkg/mod.py", "helper", 1, 4]])
    orch._refine_from_trace()
    assert orch.events[-1]["added"] == ["pkg/mod.py::helper"] and "def helper" in orch.context_text


def test_missing_trace_output_is_a_noop(orch):
    class Broken(Box):
        def read_file(self, path):
            raise FileNotFoundError(path)
    orch.sb = Broken([])
    orch._refine_from_trace()
    assert "FileNotFoundError" in orch.events[-1]["skipped"] and orch.context_text == "OLD"


def test_flag_defaults_off():
    assert config.HARNESS_SBFL is False
