"""localize(issue, plan, repo_dir) -> ContextBundle

1. BM25 over (path + symbol names) with plan.search_terms + issue text → top 40 files,
   boosted for files named in tracebacks / paths / dotted modules in the issue.
2. Planner picks ≤5 candidate files from the repo map of those 40.
3. ast_index ranks chunks of those files by embedding similarity to summary + hypothesis → top 8.
4. Exploration loop (≤3 turns): read_symbol / grep / done.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from agent import llm, prompts
from agent.events import emit
from agent.localize.ast_index import rank_chunks
from agent.localize.repomap import build_repomap, chunk_file, render
from agent.schemas import CodeChunk, ContextBundle, Issue, Plan

TOP_BM25 = 40
MAX_FILES = 5
MIN_FILES = 3
TOP_CHUNKS = 8
EXPLORE_TURNS = 3
CHUNK_PROMPT_CHARS = 2500
GREP_TIMEOUT_S = 20

_STOP = set("the a an and or of to in is it this that for on with as be are was by not if at from "
            "but we i you can should when def class self return none true false import".split())


class FilePick(BaseModel):
    files: list[str]
    reason: str = ""


class ExploreAction(BaseModel):
    action: Literal["read_symbol", "grep", "done"]
    name: Optional[str] = None
    pattern: Optional[str] = None


# ------------------------------------------------------------------ tokenization
def _split_ident(tok: str) -> list[str]:
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", tok).replace("_", " ").split()
    return [p.lower() for p in parts if p]


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        low = tok.lower()
        if len(low) < 2 or low in _STOP:
            continue
        out.append(low)
        sub = _split_ident(tok)
        if len(sub) > 1:
            out.extend(s for s in sub if len(s) > 1 and s not in _STOP)
    return out


# ------------------------------------------------------------------ issue-mentioned files
def mentioned_files(text: str, files: list[str]) -> dict[str, float]:
    """Files referenced by traceback lines, explicit paths, or dotted module names. Value = boost weight."""
    boosts: dict[str, float] = {}
    by_suffix = {f: f for f in files}

    def hit(candidate: str, w: float) -> None:
        candidate = candidate.replace("\\", "/").lstrip("./")
        for f in by_suffix:
            if f == candidate or f.endswith("/" + candidate) or candidate.endswith("/" + f):
                boosts[f] = max(boosts.get(f, 0.0), w)

    for m in re.finditer(r'File "([^"]+\.py)", line \d+', text):  # traceback frames
        hit(m.group(1), 3.0)
    for m in re.finditer(r"([\w./\\-]+\.py)\b", text):  # explicit paths
        hit(m.group(1), 2.0)
    for m in re.finditer(r"\b([a-z_][\w]*(?:\.[a-z_][\w]*){1,6})\b", text):  # dotted modules
        dotted = m.group(1)
        for k in range(len(dotted.split(".")), 1, -1):
            path = "/".join(dotted.split(".")[:k])
            for cand in (path + ".py", path + "/__init__.py"):
                if cand in by_suffix:
                    boosts[cand] = max(boosts.get(cand, 0.0), 1.0)
    return boosts


def bm25_files(issue: Issue, plan: Plan, repomap: dict[str, list[dict]], top_n: int = TOP_BM25) -> list[tuple[str, float]]:
    files = list(repomap.keys())
    docs = []
    for f in files:
        toks = tokenize(f.replace("/", " ").replace(".py", ""))
        for s in repomap[f]:
            toks += tokenize(s["name"])
        docs.append(toks or ["_empty_"])
    bm25 = BM25Okapi(docs)
    terms_q = []
    for t in plan.search_terms:
        terms_q += tokenize(t) * 3  # plan terms weigh more than raw issue text
    query = terms_q + tokenize(issue.problem_statement[:6000])
    scores = bm25.get_scores(query)
    max_s = max(scores) if len(scores) else 1.0
    boosts = mentioned_files(issue.problem_statement + "\n" + "\n".join(plan.search_terms), files)
    ranked = []
    for f, s in zip(files, scores):
        ranked.append((f, float(s) + boosts.get(f, 0.0) * max_s))
    ranked.sort(key=lambda x: -x[1])
    return ranked[:top_n]


# ------------------------------------------------------------------ exploration tools
def find_symbol(repo_dir: Path, repomap: dict[str, list[dict]], name: str, prefer: list[str]) -> list[CodeChunk]:
    name = name.strip().strip("`").replace("()", "")
    leaf = name.split(".")[-1]
    parent = name.split(".")[-2] if "." in name else None
    order = prefer + [f for f in repomap if f not in prefer]
    found: list[CodeChunk] = []
    for f in order:
        syms = repomap.get(f, [])
        if not any(s["name"] == leaf and (parent is None or s["parent"] == parent or s["kind"] == "class") for s in syms):
            continue
        for c in chunk_file(repo_dir, f):
            sym_leaf = c.symbol.split(".")[-1]
            if sym_leaf == leaf and (parent is None or c.symbol == f"{parent}.{leaf}" or c.kind != "method"):
                found.append(c)
        if len(found) >= 2:
            break
    return found[:2]


def grep_repo(repo_dir: Path, pattern: str, max_lines: int = 30) -> str:
    try:
        r = subprocess.run(["git", "grep", "-n", "-I", "-E", "-e", pattern, "--", "*.py"],
                           cwd=repo_dir, capture_output=True, text=True, timeout=GREP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return "(grep timed out)"
    if r.returncode not in (0, 1):
        return f"(grep error: {r.stderr.strip()[:200]})"
    lines = [ln[:200] for ln in r.stdout.splitlines()]
    if not lines:
        return "(no matches)"
    more = f"\n... ({len(lines) - max_lines} more)" if len(lines) > max_lines else ""
    return "\n".join(lines[:max_lines]) + more


def _chunks_text(chunks: list[CodeChunk], max_chars: int = CHUNK_PROMPT_CHARS) -> str:
    parts = []
    for c in chunks:
        code = c.code if len(c.code) <= max_chars else c.code[:max_chars] + "\n    # ... (truncated)"
        parts.append(f"### {c.file_path}:{c.start_line}-{c.end_line}  {c.kind} {c.symbol}\n```python\n{code}\n```")
    return "\n\n".join(parts)


# ------------------------------------------------------------------ main entry
def localize(issue: Issue, plan: Plan, repo_dir: Path, *, run_id: str | None = None) -> ContextBundle:
    repo_dir = Path(repo_dir)

    def ev(type_, **kw):
        if run_id:
            emit(run_id, "localizing", type_, **kw)

    repomap = build_repomap(repo_dir, cache_key=issue.instance_id)
    ranked = bm25_files(issue, plan, repomap)
    top_files = [f for f, _ in ranked]
    ev("info", step="bm25", n_files=len(repomap), top=[{"file": f, "score": round(s, 2)} for f, s in ranked[:15]])

    # 2. planner picks files from the repo map
    notes: list[str] = []
    try:
        pick = llm.complete_json("planner", [
            {"role": "system", "content": prompts.LOCALIZE_SYSTEM},
            {"role": "user", "content": prompts.localize_files_user(
                issue, plan.summary, plan.hypothesis, render(repomap, top_files), MAX_FILES)},
        ], FilePick)
        cand = []
        for f in pick.files:
            f = f.strip().lstrip("./")
            if f in repomap and f not in cand:
                cand.append(f)
        notes.append(f"file pick: {pick.reason}")
    except llm.LLMError as e:
        cand = []
        notes.append(f"file pick failed ({e}); using BM25 order")
    for f in top_files:  # backfill from BM25 ranking if the pick was thin (or failed)
        if len(cand) >= MIN_FILES:
            break
        if f not in cand:
            cand.append(f)
    cand = cand[:MAX_FILES]
    ev("info", step="file_pick", candidate_files=cand)

    # 3. embedding rank
    query = f"{plan.summary}\n{plan.hypothesis}\n{' '.join(plan.search_terms)}"
    chunks = rank_chunks(repo_dir, cand, query, top_k=TOP_CHUNKS)
    ev("info", step="embed_rank", chunks=[f"{c.file_path}:{c.symbol} ({c.score})" for c in chunks])

    # 4. exploration loop
    history: list[dict] = [{"role": "system", "content": prompts.EXPLORE_SYSTEM}]
    seen = {(c.file_path, c.symbol) for c in chunks}
    observation = ""
    prev_actions: set[str] = set()
    for turn in range(EXPLORE_TURNS):
        user = prompts.explore_user(issue, plan.summary, plan.hypothesis, cand, _chunks_text(chunks))
        if observation:
            user += f"\n\n## Result of your last action\n{observation}"
        try:
            act = llm.complete_json("planner", history + [{"role": "user", "content": user}], ExploreAction)
        except llm.LLMError as e:
            notes.append(f"explore turn {turn + 1} failed: {e}")
            break
        if act.action == "done":
            break
        sig = f"{act.action}:{act.name or act.pattern}"
        if sig in prev_actions:  # small models loop on the same request — stop instead of wasting turns
            ev("info", step="explore", turn=turn + 1, action=act.model_dump(), observation="repeated action; stopping")
            break
        prev_actions.add(sig)
        if act.action == "read_symbol" and act.name:
            found = find_symbol(repo_dir, repomap, act.name, cand)
            new = [c for c in found if (c.file_path, c.symbol) not in seen]
            for c in new:
                seen.add((c.file_path, c.symbol))
                chunks.append(c)
                if c.file_path not in cand:
                    cand.append(c.file_path)
            if new:
                observation = f"read_symbol {act.name}: added {[c.file_path + ':' + c.symbol for c in new]}"
            elif found:
                observation = (f"read_symbol {act.name}: already shown above "
                               f"({found[0].file_path}:{found[0].start_line}). Request something else or say done.")
            else:
                observation = (f"read_symbol {act.name}: not found — this symbol does not exist in the repository. "
                               "Do not assume it exists.")
                notes.append(f"DOES NOT EXIST: `{act.name}` (searched the whole repository)")
        elif act.action == "grep" and act.pattern:
            res = grep_repo(repo_dir, act.pattern)
            observation = f"grep {act.pattern!r}:\n{res}"
            notes.append(observation[:1500])
        else:
            observation = "invalid action (missing name/pattern)"
        ev("info", step="explore", turn=turn + 1, action=act.model_dump(), observation=observation[:500])

    bundle = ContextBundle(candidate_files=cand, chunks=chunks, notes="\n".join(notes))
    ev("result", candidate_files=cand, n_chunks=len(chunks))
    return bundle
