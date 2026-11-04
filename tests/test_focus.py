import pytest

from agent import config, llm
from agent.localize.focus import (EditLocations, Location, find_symbol, focus_context, symbol_source,
                                  validate_locations)
from agent.localize.repomap import build_repomap, chunk_file
from agent.orchestrator import Orchestrator
from agent.schemas import ContextBundle, Issue, Plan, RunState

SRC = '''import os


def helper(x):
    return x + 1


class Calc:
    limit = 3

    @staticmethod
    def add(a, b):
        return a + b

    def sub(self, a, b):
        return a - b


def add(a):
    return a
'''


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(SRC)
    return tmp_path


@pytest.fixture
def repomap(repo):
    return build_repomap(repo)


def test_find_symbol_forms(repomap):
    syms = repomap["pkg/mod.py"]
    assert find_symbol(syms, "Calc.sub")["parent"] == "Calc"
    assert find_symbol(syms, "sub")["parent"] == "Calc"
    assert find_symbol(syms, "helper()")["name"] == "helper"
    assert find_symbol(syms, "add")["parent"] is None  # top-level wins over Calc.add
    assert find_symbol(syms, "Other.sub") is None
    assert find_symbol(syms, "Calc.limit")["name"] == "Calc"  # class attribute → its class
    assert find_symbol(syms, "missing") is None


def test_validate_drops_unknown_files_and_symbols(repomap):
    valid, dropped = validate_locations([Location(file_path="pkg/mod.py", symbol="sub"),
                                         Location(file_path="pkg/mod.py", symbol="nope"),
                                         Location(file_path="pkg/gone.py", symbol="sub")], repomap)
    assert valid == [Location(file_path="pkg/mod.py", symbol="Calc.sub")]
    assert dropped == ["pkg/mod.py::nope", "pkg/gone.py::sub"]


def test_validate_dedupes_and_caps_at_three(repomap):
    names = ["sub", "Calc.sub", "helper", "Calc"]
    picked = [Location(file_path="./pkg/mod.py", symbol=n) for n in names]
    picked.append(Location(file_path="pkg/mod.py", symbol="add"))
    valid, _ = validate_locations(picked, repomap)
    assert [v.symbol for v in valid] == ["Calc.sub", "helper", "Calc"]


def test_symbol_source_is_full_and_keeps_decorators(repo, repomap):
    sym = find_symbol(repomap["pkg/mod.py"], "Calc.add")
    assert symbol_source(repo, "pkg/mod.py", sym) == "    @staticmethod\n    def add(a, b):\n        return a + b"


def test_focus_context_full_source_and_signatures_without_line_numbers(repo, repomap):
    chunks = chunk_file(repo, "pkg/mod.py")
    text = focus_context(repo, repomap, [Location(file_path="pkg/mod.py", symbol="Calc.sub")], chunks)
    assert "        return a - b" in text           # chosen symbol in full
    assert "- pkg/mod.py: def helper(x)" in text    # other chunk as a signature
    assert "return x + 1" not in text               # ...without its body
    assert "pkg/mod.py:" + str(chunks[0].start_line) not in text


def test_two_attributes_of_one_class_collapse_to_one_location(repomap):
    valid, dropped = validate_locations([Location(file_path="pkg/mod.py", symbol="Calc.limit"),
                                         Location(file_path="pkg/mod.py", symbol="Calc.other")], repomap)
    assert valid == [Location(file_path="pkg/mod.py", symbol="Calc")] and dropped == []


# ---- orchestrator wiring


@pytest.fixture
def orch(repo, monkeypatch):
    o = Orchestrator.__new__(Orchestrator)
    o.issue = Issue(instance_id="", repo="a/b", base_commit="x", version="1", problem_statement="bug")
    o.run_id = "r"
    o.state = RunState(run_id="r", issue=o.issue)
    o.state.context = ContextBundle(candidate_files=["pkg/mod.py"], chunks=chunk_file(repo, "pkg/mod.py"),
                                    notes="DOES NOT EXIST: _foo")
    o.repo_dir = repo
    o.context_text = o.patch_context = "BROAD"
    o.events = []
    monkeypatch.setattr(o, "ev", lambda t, **kw: o.events.append({"type": t, **kw}), raising=False)
    return o


PLAN = Plan(kind="bug", summary="s", hypothesis="h", steps=[], acceptance_criteria=[], search_terms=[])


def answer(monkeypatch, locations):
    monkeypatch.setattr(llm, "complete_json", lambda role, msgs, schema, **kw: EditLocations(locations=locations))


def test_focus_narrows_patch_context_only(orch, monkeypatch):
    answer(monkeypatch, [Location(file_path="pkg/mod.py", symbol="Calc.sub")])
    orch._focus(PLAN)
    assert "return a - b" in orch.patch_context and "DOES NOT EXIST: _foo" in orch.patch_context
    assert orch.context_text == "BROAD"
    assert orch.state.context.edit_locations == ["pkg/mod.py::Calc.sub"]
    assert orch.events[-1]["locations"] == ["pkg/mod.py::Calc.sub"]


def test_no_valid_location_falls_back_to_v1(orch, monkeypatch):
    answer(monkeypatch, [Location(file_path="pkg/mod.py", symbol="made_up")])
    orch._focus(PLAN)
    assert orch.patch_context == "BROAD" and orch.state.context.edit_locations == []
    assert orch.events[-1]["dropped"] == ["pkg/mod.py::made_up"]


def test_llm_failure_falls_back_to_v1(orch, monkeypatch):
    def boom(*a, **k):
        raise llm.LLMError("bad json")
    monkeypatch.setattr(llm, "complete_json", boom)
    orch._focus(PLAN)
    assert orch.patch_context == "BROAD" and "bad json" in orch.events[-1]["fallback"]


def test_flag_defaults_off():
    assert config.HARNESS_FOCUS is False
