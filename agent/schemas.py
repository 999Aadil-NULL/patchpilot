"""Pydantic contracts shared by all modules — single source of truth."""
from typing import Literal, Optional

from pydantic import BaseModel


class Issue(BaseModel):
    instance_id: str
    repo: str
    base_commit: str
    version: str
    problem_statement: str
    hints_text: str = ""


class Plan(BaseModel):
    kind: Literal["bug", "feature"]
    summary: str
    hypothesis: str
    steps: list[str]
    acceptance_criteria: list[str]
    search_terms: list[str]  # identifiers/files/strings to grep for


class CodeChunk(BaseModel):
    file_path: str
    symbol: str
    kind: Literal["function", "class", "method", "module"]
    start_line: int
    end_line: int
    code: str
    score: float = 0.0


class ContextBundle(BaseModel):
    candidate_files: list[str]
    chunks: list[CodeChunk]
    notes: str = ""


class EditBlock(BaseModel):
    file_path: str
    search: str
    replace: str  # empty search = create new file


class ExecResult(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False


class TestResult(BaseModel):
    passed: bool
    exec: ExecResult
    failure_summary: str = ""  # trimmed traceback


class Attempt(BaseModel):
    n: int
    edits: list[EditBlock] = []
    apply_errors: list[str] = []
    diff: str = ""
    repro: Optional[TestResult] = None
    regression: Optional[TestResult] = None
    regression_files: list[str] = []


class OracleResult(BaseModel):
    resolved: bool
    fail_to_pass: dict[str, str] = {}
    pass_to_pass: dict[str, str] = {}
    log_tail: str = ""


class RunState(BaseModel):
    run_id: str
    issue: Issue
    status: Literal[
        "planning", "localizing", "reproducing", "patching", "verifying",
        "oracle", "awaiting_review", "approved", "failed",
    ] = "planning"
    plan: Optional[Plan] = None
    context: Optional[ContextBundle] = None
    repro_script: str = ""
    repro_base: Optional[TestResult] = None
    attempts: list[Attempt] = []
    oracle: Optional[OracleResult] = None
    failure_reason: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    verification_mode: Literal["repro+regression", "regression_only"] = "repro+regression"

    @property
    def regression_only(self) -> bool:
        """True when no valid repro backs the verdict. Runs saved before this field existed have no mode set."""
        return self.verification_mode == "regression_only" or not self.repro_script
