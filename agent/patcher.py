"""SEARCH/REPLACE edit parsing + application.

Coder output format (possibly multiple blocks; a block without a FILE: line reuses the previous file):

    FILE: path/relative/to/repo.py
    <<<<<<< SEARCH
    exact existing lines
    =======
    replacement lines
    >>>>>>> REPLACE

Match tiers: exact unique → line-wise rstrip-normalized unique → difflib window ratio ≥ 0.92 →
error message (fed back to the model on retry). Empty SEARCH = create a new file.
"""
from __future__ import annotations

import difflib
import re
from typing import Protocol

from agent.schemas import EditBlock

FUZZY_THRESHOLD = 0.92
AMBIGUITY_MARGIN = 0.01

_FILE_RE = re.compile(r"^\s*(?:FILE|File|file)\s*:\s*`?([^`\s]+)`?\s*$")
_SEARCH_RE = re.compile(r"^\s*<{5,9}\s*SEARCH\s*$")
# Divider is exactly 7 '='. Docstrings (numpydoc/sympy) underline headings with '=' runs, so a run
# whose length equals the previous line's heading text is treated as an underline, not a divider.
_DIVIDER_RE = re.compile(r"^\s*={7}\s*$")
_REPLACE_RE = re.compile(r"^\s*>{5,9}\s*REPLACE\s*$")
_OPEN_FENCE_RE = re.compile(r"^\s*```\w*\s*$")
_CLOSE_FENCE_RE = re.compile(r"^\s*```\s*$")


def _is_divider(lines: list[str], idx: int) -> bool:
    if not _DIVIDER_RE.match(lines[idx]):
        return False
    prev = lines[idx - 1].strip() if idx > 0 else ""
    is_underline = bool(prev) and len(prev) == 7 and re.fullmatch(r"[A-Za-z][A-Za-z ]*", prev) is not None \
        and lines[idx].strip() == "=" * len(prev)
    return not is_underline


class PatchError(Exception):
    pass


class FileStore(Protocol):  # Sandbox satisfies this
    def read_file(self, path: str) -> str: ...
    def write_file(self, path: str, content: str) -> None: ...


# ------------------------------------------------------------------ parsing
def _clean_path(p: str) -> str:
    p = p.strip().strip("`'\"").replace("\\", "/")
    p = p.removeprefix("/testbed/").removeprefix("./")
    return p


def _join(lines: list[str]) -> str:
    return "".join(ln + "\n" for ln in lines)


def parse_edits(text: str) -> list[EditBlock]:
    lines = text.splitlines()
    edits: list[EditBlock] = []
    current_file: str | None = None
    i = 0
    while i < len(lines):
        m = _FILE_RE.match(lines[i])
        if m:
            current_file = _clean_path(m.group(1))
            i += 1
            continue
        if _SEARCH_RE.match(lines[i]):
            # tolerate the path on the line just above (e.g. inside a code fence) without "FILE:"
            if current_file is None and i > 0 and lines[i - 1].strip().endswith(".py"):
                current_file = _clean_path(lines[i - 1])
            j = i + 1
            search: list[str] = []
            while j < len(lines) and not _is_divider(lines, j) and not _REPLACE_RE.match(lines[j]):
                search.append(lines[j])
                j += 1
            k = j + 1
            replace: list[str] = []
            while k < len(lines) and not _REPLACE_RE.match(lines[k]):
                replace.append(lines[k])
                k += 1
            if j >= len(lines) or k >= len(lines) or _REPLACE_RE.match(lines[j]):
                break  # unterminated / malformed block: ignore the remainder
            if current_file:
                edits.append(EditBlock(file_path=current_file, search=_join(search), replace=_join(replace)))
            i = k + 1
            continue
        i += 1
    return edits


# ------------------------------------------------------------------ code fences
def _strip_fences(text: str) -> str:
    """Remove a markdown fence on the first and/or last line of a SEARCH or REPLACE section."""
    lines = text.splitlines()
    if lines and _OPEN_FENCE_RE.match(lines[0]):
        lines = lines[1:]
    if lines and _CLOSE_FENCE_RE.match(lines[-1]):
        lines = lines[:-1]
    return _join(lines)


def _unfence(content: str, search: str, replace: str) -> tuple[str, str]:
    """Models may wrap a section in a fence. Keep it only if the fenced text really is in the file."""
    if search != _strip_fences(search) and search in content:
        return search, replace
    return _strip_fences(search), _strip_fences(replace)


# ------------------------------------------------------------------ matching
def _line_matches(hay: list[str], needle: list[str]) -> list[int]:
    n = len(needle)
    if n == 0 or n > len(hay):
        return []
    norm_needle = [ln.rstrip() for ln in needle]
    norm_hay = [ln.rstrip() for ln in hay]
    return [i for i in range(len(hay) - n + 1) if norm_hay[i:i + n] == norm_needle]


def _fuzzy_best(hay: list[str], needle: list[str]) -> list[tuple[float, int, int]]:
    """Scored windows (ratio, start, length), best first. Window sizes len(needle) ± 1."""
    target = "\n".join(ln.rstrip() for ln in needle)
    n = len(needle)
    scored: list[tuple[float, int, int]] = []
    for size in {n, n - 1, n + 1}:
        if size <= 0 or size > len(hay):
            continue
        for i in range(len(hay) - size + 1):
            window = "\n".join(ln.rstrip() for ln in hay[i:i + size])
            sm = difflib.SequenceMatcher(None, window, target, autojunk=False)
            if sm.real_quick_ratio() < FUZZY_THRESHOLD or sm.quick_ratio() < FUZZY_THRESHOLD:
                continue
            scored.append((sm.ratio(), i, size))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored


def _closest_hint(hay: list[str], needle: list[str]) -> str:
    target = "\n".join(needle)
    n = max(1, len(needle))
    best = (0.0, 0)
    for i in range(0, max(1, len(hay) - n + 1)):
        r = difflib.SequenceMatcher(None, "\n".join(hay[i:i + n]), target).quick_ratio()
        if r > best[0]:
            best = (r, i)
    if best[0] < 0.5:
        return ""
    i = best[1]
    snippet = "\n".join(hay[i:i + n])
    return f"\nClosest region (lines {i + 1}-{i + n}, similarity≈{best[0]:.2f}):\n{snippet}"


def apply_to_text(content: str, search: str, replace: str, path: str = "<file>") -> tuple[str, str]:
    """Apply one edit to text. Returns (new_content, tier). Raises PatchError."""
    search, replace = _unfence(content, search, replace)
    if not search.strip():
        raise PatchError(f"{path}: empty SEARCH block on an existing file (empty SEARCH only creates new files)")

    # tier 1: exact
    count = content.count(search)
    if count == 1:
        return content.replace(search, replace, 1), "exact"
    if count > 1:
        raise PatchError(f"{path}: SEARCH block is ambiguous — matches {count} places exactly. "
                         "Include more surrounding lines to make it unique.")

    hay = content.splitlines()
    needle = search.splitlines()
    while needle and not needle[-1].strip():
        needle.pop()
    while needle and not needle[0].strip():
        needle.pop(0)
    rep_lines = replace.splitlines()
    trailing_nl = content.endswith("\n")

    def splice(start: int, length: int) -> str:
        new = hay[:start] + rep_lines + hay[start + length:]
        return "\n".join(new) + ("\n" if trailing_nl or not new else "")

    # tier 2: line-wise rstrip-normalized
    hits = _line_matches(hay, needle)
    if len(hits) == 1:
        return splice(hits[0], len(needle)), "normalized"
    if len(hits) > 1:
        raise PatchError(f"{path}: SEARCH block is ambiguous — matches {len(hits)} places "
                         f"(lines {', '.join(str(h + 1) for h in hits[:5])}). Include more context.")

    # tier 3: fuzzy window
    scored = _fuzzy_best(hay, needle)
    if scored and scored[0][0] >= FUZZY_THRESHOLD:
        best = scored[0]
        # a rival is an equally good window that does NOT overlap the best one
        rivals = [s for s in scored[1:] if s[0] >= best[0] - AMBIGUITY_MARGIN
                  and (s[1] >= best[1] + best[2] or s[1] + s[2] <= best[1])]
        if rivals:
            raise PatchError(f"{path}: SEARCH block is ambiguous — fuzzy-matches near lines "
                             f"{best[1] + 1} and {rivals[0][1] + 1}. Include more context.")
        return splice(best[1], best[2]), f"fuzzy({best[0]:.3f})"

    raise PatchError(f"{path}: SEARCH block not found. It must copy existing lines exactly."
                     f"{_closest_hint(hay, needle)}")


# ------------------------------------------------------------------ apply to sandbox
def _safe_path(p: str) -> bool:
    return bool(p) and not p.startswith("/") and ".." not in p.split("/")


def apply_edits(sandbox: FileStore, edits: list[EditBlock]) -> list[str]:
    """Apply edits (grouped per file, in order). Successful edits are written; returns error messages."""
    errors: list[str] = []
    contents: dict[str, str | None] = {}
    dirty: set[str] = set()
    for e in edits:
        path = e.file_path
        if not _safe_path(path):
            errors.append(f"{path}: invalid path (must be relative to the repo root)")
            continue
        if path not in contents:
            try:
                contents[path] = sandbox.read_file(path)
            except (FileNotFoundError, IsADirectoryError):
                contents[path] = None
        cur = contents[path]
        if not e.search.strip():
            if cur is not None and cur.strip():
                errors.append(f"{path}: empty SEARCH but file already exists — use a SEARCH block to edit it")
                continue
            contents[path] = _strip_fences(e.replace)
            dirty.add(path)
            continue
        if cur is None:
            errors.append(f"{path}: file does not exist")
            continue
        try:
            contents[path], _tier = apply_to_text(cur, e.search, e.replace, path)
            dirty.add(path)
        except PatchError as pe:
            errors.append(str(pe))
    for path in dirty:
        sandbox.write_file(path, contents[path] or "")
    return errors
