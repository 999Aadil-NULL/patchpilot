"""Post-edit guardrails. syntax_check rejects edits that leave a modified file unable to compile."""
from __future__ import annotations

import re
import shlex
from typing import Protocol

from agent.schemas import ExecResult

LINT_TIMEOUT_S = 30
CONTEXT_LINES = 10
_LINE_RE = re.compile(r"line (\d+)")


class Compiler(Protocol):  # Sandbox satisfies this
    def exec(self, cmd: str, timeout: int = 300) -> ExecResult: ...
    def read_file(self, path: str) -> str: ...


def syntax_check(sandbox: Compiler, files: list[str]) -> list[str]:
    """Compile each .py file in the testbed env. Returns one '<path>: <compiler output>' per failing file."""
    errors = []
    for path in files:
        if not path.endswith(".py"):
            continue
        r = sandbox.exec(f"python -m py_compile {shlex.quote(path)}", timeout=LINT_TIMEOUT_S)
        if r.timed_out:
            errors.append(f"{path}: py_compile timed out after {LINT_TIMEOUT_S}s")
        elif r.exit_code != 0:
            errors.append(f"{path}: {(r.stderr or r.stdout).strip()}")
    return errors


def error_region(source: str, error: str, context: int = CONTEXT_LINES) -> str:
    """The lines around the line an error names. No line numbers: they would leak into SEARCH blocks."""
    m = None
    for m in _LINE_RE.finditer(error):
        pass
    if m is None:
        return ""
    lines = source.splitlines()
    line = min(int(m.group(1)), len(lines))
    return "\n".join(lines[max(0, line - 1 - context):line + context])


def describe_errors(sandbox: Compiler, errors: list[str]) -> str:
    """Compile errors with the surrounding source of the broken file, for the repair prompt."""
    parts = []
    for err in errors:
        path = err.split(":", 1)[0]
        region = error_region(sandbox.read_file(path), err)
        parts.append(f"### {path}\nCompiler output:\n```\n{err}\n```\nAround the error, after your edits:\n"
                     f"```python\n{region}\n```")
    return "\n\n".join(parts)
