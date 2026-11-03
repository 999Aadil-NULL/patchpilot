import pytest

from agent.patcher import PatchError, apply_edits, apply_to_text, parse_edits
from agent.schemas import EditBlock

SRC = """\
def add(a, b):
    return a + b


def sub(a, b):
    result = a - b
    return result


class Calc:
    def mul(self, a, b):
        return a * b
"""


class FakeSandbox:
    def __init__(self, files: dict[str, str]):
        self.files = dict(files)

    def read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def write_file(self, path, content):
        self.files[path] = content


# ------------------------------------------------------------------ parse
def test_parse_multiple_blocks_and_file_reuse():
    text = """Here is the fix:
```
FILE: pkg/calc.py
<<<<<<< SEARCH
def add(a, b):
    return a + b
=======
def add(a, b):
    return b + a
>>>>>>> REPLACE
<<<<<<< SEARCH
        return a * b
=======
        return b * a
>>>>>>> REPLACE

FILE: `pkg/new.py`
<<<<<<< SEARCH
=======
X = 1
>>>>>>> REPLACE
```
"""
    edits = parse_edits(text)
    assert [e.file_path for e in edits] == ["pkg/calc.py", "pkg/calc.py", "pkg/new.py"]
    assert edits[0].search == "def add(a, b):\n    return a + b\n"
    assert edits[1].replace == "        return b * a\n"
    assert edits[2].search == "" and edits[2].replace == "X = 1\n"


def test_parse_docstring_heading_underlines_are_not_dividers():
    # sympy/numpydoc docstrings: "Examples\n========" (8) and "Example\n=======" (7)
    text = (
        "FILE: sympy/p.py\n<<<<<<< SEARCH\n"
        '    """\n    Doc.\n\n    Examples\n    ========\n\n    Example\n    =======\n    """\n'
        "    x = 1\n=======\n"
        '    """\n    Doc.\n\n    Examples\n    ========\n    """\n    x = 2\n'
        ">>>>>>> REPLACE\n"
    )
    [e] = parse_edits(text)
    assert e.search.endswith('    """\n    x = 1\n')
    assert "    Example\n    =======\n" in e.search
    assert e.replace.endswith("    x = 2\n")


def test_parse_ignores_unterminated_block():
    assert parse_edits("FILE: a.py\n<<<<<<< SEARCH\nfoo\n=======\nbar\n") == []


# ------------------------------------------------------------------ tier 1: exact
def test_exact_match():
    new, tier = apply_to_text(SRC, "    result = a - b\n", "    result = b - a\n")
    assert tier == "exact"
    assert "result = b - a" in new and "result = a - b" not in new


# ------------------------------------------------------------------ tier 2: normalized
def test_normalized_match_trailing_whitespace():
    search = "def sub(a, b):   \n    result = a - b\t\n"  # trailing junk the file doesn't have
    new, tier = apply_to_text(SRC, search, "def sub(a, b):\n    result = a - b - 0\n")
    assert tier == "normalized"
    assert "result = a - b - 0" in new
    assert new.endswith("return a * b\n")


# ------------------------------------------------------------------ tier 3: fuzzy
def test_fuzzy_match_small_typo():
    # model mis-copied one character inside a longer block
    search = ("class Calc:\n    def mul(self, a, b):\n        return a * b\n")
    src = SRC.replace("def mul(self, a, b):", "def mul(self, a, b) :")
    new, tier = apply_to_text(src, search, "class Calc:\n    def mul(self, a, b):\n        return b * a\n")
    assert tier.startswith("fuzzy")
    assert "return b * a" in new and "return a * b" not in new


def test_fuzzy_below_threshold_errors_with_hint():
    with pytest.raises(PatchError, match="not found"):
        apply_to_text(SRC, "def divide(x, y):\n    return x / y\n", "pass\n", "calc.py")


# ------------------------------------------------------------------ ambiguity
def test_ambiguous_exact_match_errors():
    src = "x = 1\ny = 2\nx = 1\n"
    with pytest.raises(PatchError, match="ambiguous"):
        apply_to_text(src, "x = 1\n", "x = 3\n")


def test_ambiguous_fuzzy_match_errors():
    block = "def handler(request):\n    data = request.json()\n    return process(data)\n"
    src = block + "\n\n" + block.replace("handler", "handlers") + "\n\n" + block.replace("handler", "handlerz")
    search = block.replace("handler", "handlerX")  # equally close to the 2nd and 3rd copies
    src = src.replace(block, "def other():\n    pass\n", 1)
    with pytest.raises(PatchError, match="ambiguous"):
        apply_to_text(src, search, "pass\n")


def test_ambiguous_normalized_match_errors():
    src = "if a:\n    pass  \nif a:\n    pass\n"
    with pytest.raises(PatchError, match="ambiguous"):
        apply_to_text(src, "if a:\n    pass \n", "if a:\n    return\n")


# ------------------------------------------------------------------ apply_edits on a store
def test_apply_edits_create_edit_and_errors():
    sb = FakeSandbox({"pkg/calc.py": SRC})
    errs = apply_edits(sb, [
        EditBlock(file_path="pkg/calc.py", search="    return a + b\n", replace="    return a + b + 0\n"),
        EditBlock(file_path="pkg/new.py", search="", replace="X = 1\n"),
        EditBlock(file_path="pkg/missing.py", search="foo\n", replace="bar\n"),
        EditBlock(file_path="../etc/passwd", search="", replace="x"),
        EditBlock(file_path="pkg/calc.py", search="", replace="clobber\n"),
    ])
    assert "return a + b + 0" in sb.files["pkg/calc.py"]
    assert sb.files["pkg/new.py"] == "X = 1\n"
    assert len(errs) == 3
    assert any("does not exist" in e for e in errs)
    assert any("invalid path" in e for e in errs)
    assert any("already exists" in e for e in errs)


def test_sequential_edits_same_file():
    sb = FakeSandbox({"c.py": SRC})
    errs = apply_edits(sb, [
        EditBlock(file_path="c.py", search="    return a + b\n", replace="    return a + b  # 1\n"),
        EditBlock(file_path="c.py", search="    return a + b  # 1\n", replace="    return a + b  # 2\n"),
    ])
    assert errs == []
    assert "# 2" in sb.files["c.py"]
