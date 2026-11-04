from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

APP = Path(__file__).resolve().parent.parent / "ui" / "app.py"
HELPERS = {"instance_label", "run_tag", "run_label", "md_escape"}
CONSTANTS = {"PROJECT_NAMES", "LABEL_TITLE_CHARS"}


@pytest.fixture(scope="module")
def ui():
    """The helpers alone: importing ui/app.py would run the whole page."""
    def wanted(node):
        if isinstance(node, ast.FunctionDef):
            return node.name in HELPERS
        return isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in CONSTANTS

    body = [n for n in ast.parse(APP.read_text()).body if wanted(n)]
    ns = {"re": re}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(APP), "exec", flags=annotations.compiler_flag), ns)
    return SimpleNamespace(**ns)


@pytest.mark.parametrize("iid, expected", [
    ("django__django-11099", "Django PR 11099"),
    ("sympy__sympy-12481", "SymPy PR 12481"),
    ("psf__requests-2317", "Requests PR 2317"),
    ("pytest-dev__pytest-7432", "pytest PR 7432"),
    ("pallets__flask-4992", "Flask PR 4992"),
    ("scikit-learn__scikit-learn-13241", "scikit-learn PR 13241"),
])
def test_project_names(ui, iid, expected):
    assert ui.instance_label(iid) == expected


def test_uses_first_non_empty_line(ui):
    assert ui.instance_label("psf__requests-1", "\n  Title here \nbody") == "Requests PR 1 — Title here"


def test_long_title_is_cut_to_about_60_chars(ui):
    label = ui.instance_label("psf__requests-1", "x" * 200)
    assert label == "Requests PR 1 — " + "x" * 59 + "…"


def test_title_of_exactly_60_chars_is_kept(ui):
    assert ui.instance_label("psf__requests-1", "x" * 60).endswith("x" * 60)


@pytest.mark.parametrize("run_id, tag", [
    ("qwen-v2-full-django__django-11099", "qwen-v2-full"),
    ("batch-django__django-11099", "batch"),
    ("h2-live-django-11099", "h2-live"),
    ("django__django-11099-20261104-171822", ""),
    ("something-else", ""),
])
def test_run_tag(ui, run_id, tag):
    assert ui.run_tag(run_id, "django__django-11099") == tag


def state(iid="django__django-11099", text="Title"):
    return SimpleNamespace(issue=SimpleNamespace(instance_id=iid, problem_statement=text))


def test_run_label_with_tag(ui):
    assert ui.run_label("qwen-v2-full-django__django-11099", state()) == "qwen-v2-full · Django PR 11099 — Title"


def test_run_label_without_tag(ui):
    assert ui.run_label("django__django-11099-20261104-171822", state()) == "Django PR 11099 — Title"


def test_run_without_state_shows_its_name(ui):
    assert ui.run_label("qwen-v2-full-django__django-11099", None) == "qwen-v2-full-django__django-11099"


def test_md_escape(ui):
    assert ui.md_escape("`a_b` | [c]: d") == r"\`a\_b\` \| \[c\]\: d"
