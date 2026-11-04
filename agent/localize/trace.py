"""H5: which repo functions does the failing repro execute? Used to add them to the context.

The runner uses only the standard library and must stay valid on Python 3.6 (oldest testbed).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple, Protocol

from agent.localize.repomap import chunk_file, is_test_path
from agent.schemas import CodeChunk, ContextBundle, ExecResult

RUNNER_PATH = "/tmp/pp_trace.py"
OUTPUT_PATH = "/tmp/pp_trace.json"
TRACE_TIMEOUT_S = 180
TOP_FUNCTIONS = 5

RUNNER = '''\
import json, os, runpy, sys, threading

ROOT = "/testbed/"
counts = {}


def tracer(frame, event, arg):
    if event == "call":
        code = frame.f_code
        path = os.path.abspath(code.co_filename)
        if not code.co_filename.startswith("<") and path.startswith(ROOT) and not code.co_name.startswith("<"):
            key = (path[len(ROOT):], code.co_name, code.co_firstlineno)
            counts[key] = counts.get(key, 0) + 1
    return None


sys.path[0] = "/testbed"  # as if run as `python repro_issue.py`
sys.argv = ["repro_issue.py"]
threading.settrace(tracer)
sys.settrace(tracer)
try:
    runpy.run_path("repro_issue.py", run_name="__main__")
except BaseException:
    pass
finally:
    sys.settrace(None)
    rows = [[k[0], k[1], k[2], v] for k, v in counts.items()]
    with open("%s", "w") as f:
        json.dump(rows, f)
''' % OUTPUT_PATH


class TracedCall(NamedTuple):
    file_path: str
    function: str
    line: int
    count: int


class Tracer(Protocol):  # Sandbox satisfies this
    def exec(self, cmd: str, timeout: int = 300) -> ExecResult: ...
    def read_file(self, path: str) -> str: ...
    def write_file(self, path: str, content: str) -> None: ...


def trace_repro(sandbox: Tracer) -> list[TracedCall]:
    """Run repro_issue.py under the tracer. Tests and the repro itself are excluded."""
    sandbox.write_file(RUNNER_PATH, RUNNER)
    sandbox.exec(f"rm -f {OUTPUT_PATH}; python {RUNNER_PATH}", timeout=TRACE_TIMEOUT_S)
    rows = json.loads(sandbox.read_file(OUTPUT_PATH))
    return [TracedCall(*r) for r in rows if not is_test_path(r[0]) and r[0] != "repro_issue.py"]


def top_functions(trace: list[TracedCall], candidate_files: list[str], n: int = TOP_FUNCTIONS) -> list[TracedCall]:
    """Candidate files first, then most-called: raw counts alone favour generic helpers."""
    return sorted(trace, key=lambda t: (t.file_path not in candidate_files, -t.count, t.file_path, t.line))[:n]


def _chunk_for(repo_dir: Path, call: TracedCall) -> CodeChunk | None:
    for c in chunk_file(repo_dir, call.file_path):
        if c.symbol.rsplit(".", 1)[-1] == call.function and c.start_line <= call.line <= c.end_line:
            return c
    return None


def refine_context(context: ContextBundle, trace: list[TracedCall], repo_dir: Path) -> list[str]:
    """Put the top traced functions at the front of the chunks if absent. Returns 'path::symbol' of those added."""
    have = {(c.file_path, c.symbol) for c in context.chunks}
    added: list[CodeChunk] = []
    for call in top_functions(trace, context.candidate_files):
        chunk = _chunk_for(repo_dir, call)
        if chunk is not None and (chunk.file_path, chunk.symbol) not in have:
            have.add((chunk.file_path, chunk.symbol))
            added.append(chunk.model_copy(update={"score": 1.0}))
    context.chunks = added + context.chunks
    return [f"{c.file_path}::{c.symbol}" for c in added]
