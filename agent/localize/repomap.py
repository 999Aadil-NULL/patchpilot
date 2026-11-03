"""tree-sitter repo map: file → top-level classes/functions (+ methods one level deep) with signatures.
Also provides function/method/class chunking used by ast_index and read_symbol."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import tree_sitter_python as tsp
from tree_sitter import Language, Node, Parser

from agent.config import CACHE_DIR
from agent.schemas import CodeChunk

REPOMAP_DIR = CACHE_DIR / "repomap"
SKIP_DIRS = {".git", "build", "dist", "docs", "doc", ".tox", ".eggs", "node_modules", "__pycache__",
             "tests", "testing", "test", "examples", "benchmarks", "bench", "release"}
MAX_FILE_BYTES = 1_500_000
CLASS_HEADER_MAX_LINES = 40


@lru_cache(maxsize=1)
def _parser() -> Parser:
    return Parser(Language(tsp.language()))


def is_test_path(rel: str) -> bool:
    parts = rel.split("/")
    name = parts[-1]
    return (any(p in {"tests", "testing", "test"} for p in parts[:-1])
            or name.startswith("test_") or name.endswith("_test.py") or name.endswith("_tests.py")
            or name == "conftest.py")


def iter_source_files(repo_dir: Path) -> list[str]:
    """Non-test .py files, repo-relative POSIX paths, sorted."""
    repo_dir = Path(repo_dir)
    out = []
    for p in repo_dir.rglob("*.py"):
        rel = p.relative_to(repo_dir).as_posix()
        parts = rel.split("/")
        if any(d in SKIP_DIRS or d.startswith(".") for d in parts[:-1]):
            continue
        if is_test_path(rel) or parts[-1] == "setup.py":
            continue
        out.append(rel)
    return sorted(out)


# ------------------------------------------------------------------ parsing helpers
def _unwrap(node: Node) -> Node:
    if node.type == "decorated_definition":
        d = node.child_by_field_name("definition")
        return d if d is not None else node
    return node


def _signature(src: bytes, node: Node) -> str:
    body = node.child_by_field_name("body")
    end = body.start_byte if body is not None else node.end_byte
    sig = src[node.start_byte:end].decode("utf-8", errors="replace")
    sig = re.sub(r"\s+", " ", sig).strip().rstrip(":").strip()
    return sig[:300]


def _name(src: bytes, node: Node) -> str:
    n = node.child_by_field_name("name")
    return src[n.start_byte:n.end_byte].decode("utf-8", errors="replace") if n is not None else "?"


def _defs(node: Node):
    for ch in node.children:
        d = _unwrap(ch)
        if d.type in ("class_definition", "function_definition"):
            yield ch, d  # (outer incl. decorators, definition)


def parse_symbols(src: bytes) -> list[dict]:
    """[{name, kind, signature, start_line, end_line, parent}] — 1-based inclusive lines."""
    tree = _parser().parse(src)
    out: list[dict] = []
    for outer, d in _defs(tree.root_node):
        name = _name(src, d)
        kind = "class" if d.type == "class_definition" else "function"
        out.append(dict(name=name, kind=kind, signature=_signature(src, d),
                        start_line=outer.start_point[0] + 1, end_line=outer.end_point[0] + 1, parent=None))
        if kind == "class":
            body = d.child_by_field_name("body")
            if body is not None:
                for mo, md in _defs(body):
                    if md.type != "function_definition":
                        continue
                    out.append(dict(name=_name(src, md), kind="method", signature=_signature(src, md),
                                    start_line=mo.start_point[0] + 1, end_line=mo.end_point[0] + 1,
                                    parent=name))
    return out


def _read(repo_dir: Path, rel: str) -> bytes | None:
    p = Path(repo_dir) / rel
    try:
        if p.stat().st_size > MAX_FILE_BYTES:
            return None
        return p.read_bytes()
    except OSError:
        return None


# ------------------------------------------------------------------ repo map
def build_repomap(repo_dir: Path, cache_key: str | None = None) -> dict[str, list[dict]]:
    """{rel_path: symbols}. Cached per instance in cache/repomap/<key>.json."""
    cache_fp = REPOMAP_DIR / f"{cache_key}.json" if cache_key else None
    if cache_fp and cache_fp.exists():
        return json.loads(cache_fp.read_text(encoding="utf-8"))
    rm: dict[str, list[dict]] = {}
    for rel in iter_source_files(repo_dir):
        src = _read(repo_dir, rel)
        if src is None:
            continue
        try:
            rm[rel] = parse_symbols(src)
        except Exception:
            rm[rel] = []
    if cache_fp:
        REPOMAP_DIR.mkdir(parents=True, exist_ok=True)
        cache_fp.write_text(json.dumps(rm), encoding="utf-8")
    return rm


def render(repomap: dict[str, list[dict]], files: list[str], max_symbols_per_file: int = 40) -> str:
    """Compact text map: path, then indented signatures (methods nested)."""
    lines: list[str] = []
    for f in files:
        syms = repomap.get(f, [])
        lines.append(f"{f}")
        for s in syms[:max_symbols_per_file]:
            indent = "    " if s["kind"] == "method" else "  "
            lines.append(f"{indent}{s['signature']}  # L{s['start_line']}")
        if len(syms) > max_symbols_per_file:
            lines.append(f"  ... ({len(syms) - max_symbols_per_file} more symbols)")
    return "\n".join(lines)


# ------------------------------------------------------------------ chunking
def chunk_file(repo_dir: Path, rel: str) -> list[CodeChunk]:
    """function / method / class(header up to first method) chunks; module chunk if no defs."""
    src = _read(repo_dir, rel)
    if src is None:
        return []
    text_lines = src.decode("utf-8", errors="replace").splitlines()
    syms = parse_symbols(src)

    def code(a: int, b: int) -> str:
        return "\n".join(text_lines[a - 1:b])

    chunks: list[CodeChunk] = []
    for s in syms:
        if s["kind"] == "class":
            methods = [m for m in syms if m["parent"] == s["name"] and s["start_line"] <= m["start_line"] <= s["end_line"]]
            end = (methods[0]["start_line"] - 1) if methods else s["end_line"]
            end = min(end, s["start_line"] + CLASS_HEADER_MAX_LINES - 1)
            chunks.append(CodeChunk(file_path=rel, symbol=s["name"], kind="class",
                                    start_line=s["start_line"], end_line=end, code=code(s["start_line"], end)))
        elif s["kind"] == "method":
            chunks.append(CodeChunk(file_path=rel, symbol=f"{s['parent']}.{s['name']}", kind="method",
                                    start_line=s["start_line"], end_line=s["end_line"],
                                    code=code(s["start_line"], s["end_line"])))
        else:
            chunks.append(CodeChunk(file_path=rel, symbol=s["name"], kind="function",
                                    start_line=s["start_line"], end_line=s["end_line"],
                                    code=code(s["start_line"], s["end_line"])))
    if not syms and text_lines:
        n = min(len(text_lines), 120)
        chunks.append(CodeChunk(file_path=rel, symbol="<module>", kind="module", start_line=1, end_line=n,
                                code=code(1, n)))
    return chunks
