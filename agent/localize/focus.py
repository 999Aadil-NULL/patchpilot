"""H3: narrow the patch prompt to ≤3 edit locations (full source) plus signatures of other candidates."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from agent.schemas import CodeChunk

MAX_LOCATIONS = 3
MAX_SYMBOL_CHARS = 12_000


class Location(BaseModel):
    file_path: str
    symbol: str


class EditLocations(BaseModel):
    locations: list[Location]


def _qualified(sym: dict) -> str:
    return f"{sym['parent']}.{sym['name']}" if sym.get("parent") else sym["name"]


def find_symbol(symbols: list[dict], name: str) -> dict | None:
    """Match 'Class.method', or a bare name (top-level definitions first, then the first method).

    'Class.attribute' resolves to the class: the repo map has no attributes, but the class source holds them.
    """
    name = name.strip().removesuffix("()")
    exact = [s for s in symbols if _qualified(s) == name]
    if exact:
        return exact[0]
    if "." in name:
        owner = name.rsplit(".", 1)[0]
        return next((s for s in symbols if s["kind"] == "class" and s["name"] == owner), None)
    bare = sorted((s for s in symbols if s["name"] == name), key=lambda s: s.get("parent") is not None)
    return bare[0] if bare else None


def validate_locations(picked: list[Location], repomap: dict[str, list[dict]]) -> tuple[list[Location], list[str]]:
    """Keep locations that exist in the repo map, in canonical form. Returns (valid, dropped descriptions)."""
    valid: list[Location] = []
    dropped: list[str] = []
    for loc in picked:
        path = loc.file_path.strip().removeprefix("./")
        sym = find_symbol(repomap.get(path, []), loc.symbol)
        if sym is None:
            dropped.append(f"{loc.file_path}::{loc.symbol}")
            continue
        canonical = Location(file_path=path, symbol=_qualified(sym))
        if canonical not in valid and len(valid) < MAX_LOCATIONS:
            valid.append(canonical)
    return valid, dropped


def symbol_source(repo_dir: Path, path: str, sym: dict) -> str:
    lines = (Path(repo_dir) / path).read_text(encoding="utf-8", errors="replace").splitlines()
    code = "\n".join(lines[sym["start_line"] - 1:sym["end_line"]])
    return code if len(code) <= MAX_SYMBOL_CHARS else code[:MAX_SYMBOL_CHARS] + "\n    # ... (truncated)"


def focus_context(repo_dir: Path, repomap: dict[str, list[dict]], locations: list[Location],
                  chunks: list[CodeChunk]) -> str:
    """Patch context. No line numbers in the shown code: they would leak into SEARCH blocks."""
    parts = ["## Code to edit (full source; copy SEARCH lines from here)"]
    for loc in locations:
        sym = find_symbol(repomap[loc.file_path], loc.symbol)
        assert sym is not None  # validate_locations kept it
        parts.append(f"### {loc.file_path} — {loc.symbol}\n```python\n{symbol_source(repo_dir, loc.file_path, sym)}\n```")
    chosen = {(loc.file_path, loc.symbol) for loc in locations}
    others = []
    for c in chunks:
        sym = find_symbol(repomap.get(c.file_path, []), c.symbol)
        if sym is not None and (c.file_path, _qualified(sym)) not in chosen:
            line = f"- {c.file_path}: {sym['signature']}"
            if line not in others:
                others.append(line)
    if others:
        parts.append("## Other related code (signatures only)\n" + "\n".join(others))
    return "\n\n".join(parts) + "\n\n"
