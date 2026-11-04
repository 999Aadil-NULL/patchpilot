"""State machine: PLAN → LOCALIZE → REPRODUCE → [PATCH → VERIFY → (REFLECT)]×≤MAX_ATTEMPTS → ORACLE → AWAITING_REVIEW

Writes runs/<run_id>/events.jsonl (via emit) and state.json after every stage. Never raises:
any exception ends the run with status "failed" + failure_reason.
"""
from __future__ import annotations

import re
import subprocess
import time
import traceback
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from agent import config, llm, prompts
from agent.config import CACHE_DIR
from agent.dataset import get_issue
from agent.guards import describe_errors, syntax_check
from agent.events import emit, write_state
from agent.localize.focus import EditLocations, focus_context, validate_locations
from agent.localize.repomap import build_repomap, is_test_path, render
from agent.localize.retrieve import _chunks_text, localize
from agent.localize.trace import refine_context, trace_repro
from agent.oracle import grade
from agent.patcher import apply_edits, parse_edits
from agent.sandbox import REPRO_FILE, Sandbox, failure_summary
from agent.schemas import Attempt, ContextBundle, ExecResult, Plan, RunState, TestResult
from agent.select import select

INVALID_REPRO_EXC = {"ImportError", "ModuleNotFoundError", "SyntaxError", "IndentationError", "NameError",
                     "TabError", "ImproperlyConfigured"}
_EXC_LINE_RE = re.compile(r"^([A-Za-z_][\w.]*)(?::|$)")
REPRO_REWRITES = 2
MAX_REGRESSION_FILES = 3
FULL_FILE_MAX_LINES = 500
CONTEXT_MAX_CHARS = 60_000
CODER_CHUNK_CHARS = 12_000
REGRESSION_PARSE_CHARS = 5_000_000   # parse the full test log...
REGRESSION_STORE_CHARS = 20_000      # ...but store only a tail in state.json
REPRO_FIXES = 1                      # times REFLECT may replace a repro judged wrong


class Reflection(BaseModel):
    repro_verdict: Literal["patch_wrong", "repro_wrong", "passed"] = "patch_wrong"
    expected_failures: list[str] = []
    analysis: str = ""


# ------------------------------------------------------------------ helpers
def _extract_python(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return text.strip() + "\n"


def _final_exception(tr: TestResult) -> str | None:
    """Exception class name from the last traceback (e.g. 'AssertionError'), whatever its suffix, if any.

    Frames are indented, so the first unindented line after the start is the exception. Without a
    'Traceback' header (a script that does not compile, or trimmed output) start at the last frame.
    """
    lines = ((tr.exec.stdout or "") + "\n" + (tr.exec.stderr or "")).splitlines()
    starts = ([i for i, ln in enumerate(lines) if ln.strip() == "Traceback (most recent call last):"]
              or [i for i, ln in enumerate(lines) if ln.lstrip().startswith('File "')])
    if not starts:
        return None
    for line in lines[starts[-1] + 1:]:
        if line.strip() and not line[0].isspace():
            m = _EXC_LINE_RE.match(line)
            return m.group(1).split(".")[-1] if m else None
    return None


def _module_names(path: str) -> list[str]:
    """sympy/core/power.py → ['sympy.core.power']; src/_pytest/skipping.py → ['_pytest.skipping']."""
    p = path.removesuffix(".py").removesuffix("/__init__")
    if p.startswith("src/"):
        p = p[4:]
    return [p.replace("/", ".")]


def _parse_statuses(repo: str, output: str) -> dict[str, str]:
    from swebench.harness.log_parsers.python import MAP_REPO_TO_PARSER_PY

    parser = MAP_REPO_TO_PARSER_PY.get(repo)
    if parser is None:
        return {}
    try:
        return parser(output, None)
    except Exception:
        return {}


def _ok(status: str) -> bool:
    return status in ("PASSED", "XFAIL", "SKIPPED")


class Orchestrator:
    def __init__(self, instance_id: str, run_id: str | None = None):
        self.issue = get_issue(instance_id)
        self.run_id = run_id or f"{instance_id}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.state = RunState(run_id=self.run_id, issue=self.issue)
        self.sb: Sandbox | None = None
        self.repo_dir: Path = CACHE_DIR / "repos" / instance_id
        self.context_text = ""
        self.patch_context = ""
        self._baseline: dict[str, dict[str, str]] = {}  # test file → status map at base commit
        self._new_failures: dict[int, list[str]] = {}    # attempt n → "file::test: STATUS"
        self._reflections: dict[int, str] = {}           # attempt n → reflection analysis
        self._repro_fixes = 0

    # -------------------------------------------------------------- bookkeeping
    def _stage(self, status: str) -> None:
        self.state.status = status  # type: ignore[assignment]
        llm.set_run_context(self.run_id, status)
        self._save()

    def _save(self) -> None:
        u = llm.get_usage()
        self.state.tokens_in, self.state.tokens_out = u.tokens_in, u.tokens_out
        write_state(self.state)

    def ev(self, type_: str, **kw) -> None:
        emit(self.run_id, self.state.status, type_, **kw)

    # -------------------------------------------------------------- main
    def run(self) -> RunState:
        llm.reset_usage()
        emit(self.run_id, "run", "start", instance_id=self.issue.instance_id, repo=self.issue.repo)
        try:
            self._run()
        except Exception as e:  # never crash the caller / UI
            self.state.failure_reason = self.state.failure_reason or f"{type(e).__name__}: {e}"
            stage = self.state.status
            self.state.status = "failed"
            emit(self.run_id, stage, "error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])
        finally:
            if self.sb is not None:
                self.sb.stop()
            self._save()
            emit(self.run_id, "run", "end", status=self.state.status, failure_reason=self.state.failure_reason,
                 resolved=self.state.oracle.resolved if self.state.oracle else None,
                 tokens_in=self.state.tokens_in, tokens_out=self.state.tokens_out)
        return self.state

    def _run(self) -> None:
        # sandbox + host repo copy
        self._stage("planning")
        self.ev("start")
        self.state.run_config = config.run_config()
        self.ev("info", message="run_config", **self.state.run_config)
        self.sb = Sandbox(self.issue.instance_id, self.run_id)
        self.sb.start()
        self.repo_dir = self.sb.export_repo(self.repo_dir)

        # PLAN
        plan = llm.complete_json("planner", [
            {"role": "system", "content": prompts.PLAN_SYSTEM},
            {"role": "user", "content": prompts.plan_user(self.issue)},
        ], Plan)
        self.state.plan = plan
        self.ev("result", plan=plan.model_dump())
        self.ev("end")

        # LOCALIZE
        self._stage("localizing")
        self.ev("start")
        self.state.context = localize(self.issue, plan, self.repo_dir, run_id=self.run_id)
        self.context_text = self._build_context(self.state.context)
        self.ev("end")

        # REPRODUCE
        self._stage("reproducing")
        self.ev("start")
        self._reproduce(plan)
        self.ev("end")

        # narrowing happens after REPRODUCE so that it can see functions the repro traced
        if config.HARNESS_SBFL:
            self._refine_from_trace()
        self.patch_context = self.context_text
        if config.HARNESS_FOCUS:
            self._stage("localizing")
            self.ev("start")
            self._focus(plan)
            self.ev("end")

        # PATCH / VERIFY loop
        verified: Attempt | None = None
        for n in range(1, config.MAX_ATTEMPTS + 1):
            att = self._attempt(n, plan)
            self.state.attempts.append(att)
            self._save()
            if self._attempt_ok(att):
                verified = att
                break

        self.state.verification_mode = "repro+regression" if self.state.repro_script else "regression_only"

        # ORACLE (on success, else on the best failed attempt for the record)
        target = verified or self._best_failed()
        if target is not None and target.diff.strip():
            self._stage("oracle")
            self.ev("start", attempt=target.n, verified=verified is not None)
            self.state.oracle = grade(self.issue.instance_id, target.diff)
            self.ev("result", resolved=self.state.oracle.resolved,
                    fail_to_pass=self.state.oracle.fail_to_pass,
                    n_pass_to_pass=len(self.state.oracle.pass_to_pass),
                    p2p_failed=[k for k, v in self.state.oracle.pass_to_pass.items() if v != "PASSED"])
            self.ev("end")

        if verified is not None:
            self._stage("awaiting_review")
            self.ev("info", message="verified patch ready for human review", attempt=verified.n)
        else:
            self.state.failure_reason = self.state.failure_reason or \
                f"no attempt passed verification after {len(self.state.attempts)} attempt(s)"
            self.state.status = "failed"
            self._save()

    # -------------------------------------------------------------- context
    def _build_context(self, ctx: ContextBundle) -> str:
        files_text = []
        budget = CONTEXT_MAX_CHARS
        full_files: set[str] = set()
        for f in ctx.candidate_files[:2]:  # whole file for the top candidates if small
            p = self.repo_dir / f
            try:
                src = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(src.splitlines()) <= FULL_FILE_MAX_LINES and len(src) < budget // 2:
                files_text.append(f"### {f}\n```python\n{src}\n```")
                budget -= len(src)
                full_files.add(f)
        chunks = [c for c in ctx.chunks if c.file_path not in full_files]
        # coder needs complete chunks to copy SEARCH lines verbatim → generous per-chunk limit
        chunks_text = _chunks_text(chunks, max_chars=CODER_CHUNK_CHARS)[:max(budget, 5000)]
        missing = [ln for ln in ctx.notes.splitlines() if ln.startswith("DOES NOT EXIST:")]
        out = prompts.context_block(chunks_text, "\n\n".join(files_text))
        if missing:
            out += "## Verified facts\n" + "\n".join(missing) + "\n\n"
        return out

    def _refine_from_trace(self) -> None:
        """Add functions the failing repro executed to the context. No valid repro, or any failure: no-op."""
        assert self.sb is not None and self.state.context is not None
        if not self.state.repro_script:
            self.ev("info", message="trace", skipped="no valid repro")
            return
        try:
            self.sb.write_file(REPRO_FILE, self.state.repro_script)
            trace = trace_repro(self.sb)
            added = refine_context(self.state.context, trace, self.repo_dir)
        except (OSError, ValueError, RuntimeError, KeyError) as e:
            self.ev("info", message="trace", skipped=f"{type(e).__name__}: {e}"[:300])
            return
        self.ev("info", message="trace", n_functions=len(trace), added=added)
        if added:
            self.context_text = self._build_context(self.state.context)

    def _focus(self, plan: Plan) -> None:
        """Pick edit locations for the patch prompt. Any failure leaves the v1 context in place."""
        ctx = self.state.context
        assert ctx is not None
        repomap = build_repomap(self.repo_dir, cache_key=self.issue.instance_id)
        try:
            picked = llm.complete_json("planner", [
                {"role": "system", "content": prompts.FOCUS_SYSTEM},
                {"role": "user", "content": prompts.focus_user(self.issue, plan, self.context_text,
                                                               render(repomap, ctx.candidate_files))},
            ], EditLocations)
        except llm.LLMError as e:
            self.ev("info", message="edit_locations", locations=[], fallback=f"{type(e).__name__}: {e}"[:300])
            return
        valid, dropped = validate_locations(picked.locations, repomap)
        ctx.edit_locations = [f"{loc.file_path}::{loc.symbol}" for loc in valid]
        self.ev("info", message="edit_locations", locations=ctx.edit_locations, dropped=dropped,
                fallback="" if valid else "no valid locations")
        if valid:
            missing = [ln for ln in ctx.notes.splitlines() if ln.startswith("DOES NOT EXIST:")]
            facts = "## Verified facts\n" + "\n".join(missing) + "\n\n" if missing else ""
            self.patch_context = focus_context(self.repo_dir, repomap, valid, ctx.chunks) + facts

    # -------------------------------------------------------------- reproduce
    def _reproduce(self, plan: Plan) -> None:
        assert self.sb is not None
        feedback = ""
        for i in range(REPRO_REWRITES + 1):
            text = llm.complete("coder", [
                {"role": "system", "content": prompts.REPRO_SYSTEM},
                {"role": "user", "content": prompts.repro_user(self.issue, plan, self.context_text, feedback)},
            ], temperature=0.2 if i == 0 else 0.4)
            script = _extract_python(text)
            res = self.sb.run_repro(script)
            exc = _final_exception(res)
            valid, why = self._repro_valid(res, exc)
            self.ev("info", try_n=i + 1, valid=valid, reason=why, exception=exc, exit_code=res.exec.exit_code,
                    script=script[:4000], output_tail=res.failure_summary[-1500:])
            if valid:
                self.state.repro_script = script
                self.state.repro_base = res
                self.ev("result", valid=True, exception=exc, script=script, failure=res.failure_summary)
                self._save()
                return
            out = (res.exec.stdout + "\n" + res.exec.stderr).strip()
            feedback = (f"Reason: {why}\nScript:\n```python\n{script}\n```\nOutput (exit {res.exec.exit_code}):\n```\n"
                        f"{out[-2500:]}\n```")
        self.state.repro_script = ""
        self.state.repro_base = None
        self.ev("result", valid=False,
                message="no valid reproduction after rewrites; continuing with regression-only verification")
        self._save()

    @staticmethod
    def _repro_valid(res: TestResult, exc: str | None) -> tuple[bool, str]:
        if res.exec.timed_out:
            return False, "script timed out"
        if res.passed:
            return False, "script exited 0 on the buggy code — it does not reproduce the issue"
        if exc in INVALID_REPRO_EXC:
            return False, f"script failed with {exc}, which is a script bug, not the reported issue"
        if exc is None:
            return False, "script exited non-zero without a traceback; use assert or let the exception propagate"
        return True, f"fails with {exc}"

    # -------------------------------------------------------------- attempts
    def _history(self) -> str:
        parts = []
        for a in self.state.attempts:
            lines = [f"### Attempt {a.n}"]
            if a.edits:
                lines.append("Edits:\n" + "\n".join(
                    f"FILE: {e.file_path}\n<<<<<<< SEARCH\n{e.search}=======\n{e.replace}>>>>>>> REPLACE"
                    for e in a.edits)[:6000])
            if a.apply_errors:
                lines.append("Apply errors:\n" + "\n".join(a.apply_errors)[:3000])
            if a.repro and not a.repro.passed:
                lines.append(f"Reproduction still failing:\n```\n{a.repro.failure_summary[-2500:]}\n```")
            if a.regression and not a.regression.passed:
                lines.append(f"Regression failures in {a.regression_files}:\n```\n"
                             f"{a.regression.failure_summary[-2500:]}\n```")
            if a.n in self._reflections:
                lines.append(f"Reviewer analysis: {self._reflections[a.n]}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts[-2:])  # last two attempts are enough context

    def _attempt(self, n: int, plan: Plan) -> Attempt:
        assert self.sb is not None
        self._stage("patching")
        self.ev("start", attempt=n)
        self.sb.reset()
        att = Attempt(n=n)
        repro_summary = self.state.repro_base.failure_summary if self.state.repro_base else ""
        messages = [
            {"role": "system", "content": prompts.PATCH_SYSTEM},
            {"role": "user", "content": prompts.patch_user(self.issue, plan, self.patch_context,
                                                           self.state.repro_script, repro_summary, self._history())},
        ]
        if config.HARNESS_SAMPLES > 1:
            return self._sampled_attempt(n, plan, messages)
        temperature = 0.2 if n == 1 else 0.5
        text = llm.complete("coder", messages, temperature=temperature, max_tokens=8192)
        self._apply(text, att)
        if config.HARNESS_LINT and att.edits and not att.apply_errors:
            self._lint_gate(att, messages, text, temperature)
        att.diff = self.sb.diff()
        self.ev("result", attempt=n, n_edits=len(att.edits), apply_errors=att.apply_errors, diff=att.diff)
        self.ev("end", attempt=n)

        if not att.diff.strip() or att.apply_errors:
            # partial/failed application: verifying a half-applied patch gives no useful signal —
            # the apply errors themselves are the feedback for the next attempt
            if not att.diff.strip() and not att.apply_errors:
                att.apply_errors = ["Edits produced no change to the code."]
            return att

        # VERIFY
        self._stage("verifying")
        self.ev("start", attempt=n)
        if self.state.repro_script:
            att.repro = self.sb.run_repro(self.state.repro_script)
            self.ev("info", attempt=n, check="repro", passed=att.repro.passed)
        att.regression_files = self._regression_files(att)
        if att.regression_files:
            att.regression = self._run_regression(att.regression_files, n)
            self.ev("info", attempt=n, check="regression", files=att.regression_files, passed=att.regression.passed,
                    new_failures=self._new_failures.get(n, [])[:20])
        if not self._attempt_ok(att) and (att.repro is not None or att.regression is not None):
            self._reflect(att, plan)
        self.ev("result", attempt=n, ok=self._attempt_ok(att))
        self.ev("end", attempt=n)
        return att

    def _candidate(self, i: int, n: int, messages: list[dict]) -> tuple[Attempt, dict]:
        """Generate one candidate on a fresh tree and check it. Regression runs only if repro did not fail."""
        assert self.sb is not None
        self._stage("patching")
        self.sb.reset()
        att = Attempt(n=n)
        temperature = 0.2 if i == 0 else 0.7
        text = llm.complete("coder", messages, temperature=temperature, max_tokens=8192, sample_idx=i)
        self._apply(text, att)
        applied = not att.apply_errors
        lint_clean = True
        if config.HARNESS_LINT and att.edits and applied:
            lint_clean = self._lint_gate(att, messages, text, temperature)
        att.diff = self.sb.diff()
        if not att.diff.strip() and not att.apply_errors:
            att.apply_errors = ["Edits produced no change to the code."]
        self._new_failures.pop(n, None)
        if att.diff.strip() and not att.apply_errors:
            self._stage("verifying")
            if self.state.repro_script:
                att.repro = self.sb.run_repro(self.state.repro_script)
            if att.repro is None or att.repro.passed:
                att.regression_files = self._regression_files(att)
                if att.regression_files:
                    att.regression = self._run_regression(att.regression_files, n)
        failures = self._new_failures.get(n, []) if att.regression is not None else None
        summary = {"index": i, "temperature": temperature, "diff": att.diff,
                   "clean_apply": applied and not (att.apply_errors and lint_clean), "lint_clean": lint_clean,
                   "repro_passed": att.repro.passed if att.repro else None,
                   "regression_passed": att.regression.passed if att.regression else None,
                   "new_failures": len(failures) if failures is not None else None,
                   "apply_errors": [e[:300] for e in att.apply_errors]}
        self.ev("info", message="candidate", attempt=n, **{k: v for k, v in summary.items() if k != "diff"})
        return att, summary

    def _sampled_attempt(self, n: int, plan: Plan, messages: list[dict]) -> Attempt:
        """K candidates, one chosen by execution. Its feedback, not the others', goes into the next attempt."""
        assert self.sb is not None
        results = []
        for i in range(config.HARNESS_SAMPLES):
            att, summary = self._candidate(i, n, messages)
            results.append((att, summary, list(self._new_failures.get(n, []))))
        summaries = [r[1] for r in results]
        idx, reason = select(summaries)
        chosen, summary, failures = results[idx]
        summary.update(selected=True, reason=reason)
        chosen.candidates = summaries
        self._new_failures[n] = failures
        self.ev("info", message="candidate_selection", attempt=n, selected=idx, reason=reason, k=len(results))
        self.sb.reset()  # the tree holds the last candidate; reflection needs the chosen one
        if chosen.edits:
            apply_edits(self.sb, chosen.edits)
        self._stage("patching")
        self.ev("result", attempt=n, n_edits=len(chosen.edits), apply_errors=chosen.apply_errors, diff=chosen.diff)
        self.ev("end", attempt=n)
        self._stage("verifying")
        self.ev("start", attempt=n)
        if not self._attempt_ok(chosen) and (chosen.repro is not None or chosen.regression is not None):
            self._reflect(chosen, plan)
        self.ev("result", attempt=n, ok=self._attempt_ok(chosen))
        self.ev("end", attempt=n)
        return chosen

    def _apply(self, text: str, att: Attempt) -> None:
        """Parse the coder's reply and apply it to the tree, recording edits and errors on the attempt."""
        assert self.sb is not None
        att.edits = [e for e in parse_edits(text) if not is_test_path(e.file_path) and e.file_path != "repro_issue.py"]
        if not att.edits:
            att.apply_errors = ["No valid edit blocks found in your response (test files and repro_issue.py "
                                "cannot be edited). Use the exact FILE/SEARCH/REPLACE format."]
        else:
            att.apply_errors = apply_edits(self.sb, att.edits)

    def _lint_gate(self, att: Attempt, messages: list[dict], text: str, temperature: float) -> bool:
        """Compile the edited files. On failure give the coder ONE repair turn, which is not a new attempt.

        Returns False when the result still does not compile.
        """
        assert self.sb is not None
        errors = syntax_check(self.sb, sorted({e.file_path for e in att.edits}))
        if not errors:
            return True
        self.ev("info", message="lint_fail", attempt=att.n, errors=errors)
        repair = prompts.lint_repair_user(describe_errors(self.sb, errors))
        fixed = llm.complete("coder", messages + [{"role": "assistant", "content": text},
                                                  {"role": "user", "content": repair}],
                             temperature=temperature, max_tokens=8192)
        self.sb.reset()
        self._apply(fixed, att)
        if att.apply_errors:
            att.apply_errors = errors + att.apply_errors  # the next attempt still needs the compile error
            self.ev("info", message="lint_fail", attempt=att.n, after_repair=True, errors=att.apply_errors)
            return False
        errors = syntax_check(self.sb, sorted({e.file_path for e in att.edits}))
        if errors:
            att.apply_errors = errors
            self.ev("info", message="lint_fail", attempt=att.n, after_repair=True, errors=errors)
            return False
        self.ev("info", message="lint_repaired", attempt=att.n)
        return True

    # -------------------------------------------------------------- reflect
    def _reflect(self, att: Attempt, plan: Plan) -> None:
        """Judge failures: patch wrong vs repro wrong vs old-behaviour tests. May fix the repro once and may
        accept regression failures that only assert behaviour the issue asks to change."""
        assert self.sb is not None
        repro_failed = att.repro is not None and not att.repro.passed
        new_fail = self._new_failures.get(att.n, [])
        try:
            ref = llm.complete_json("planner", [
                {"role": "system", "content": prompts.REFLECT_SYSTEM},
                {"role": "user", "content": prompts.reflect_user(
                    self.issue, plan, att.diff, self.state.repro_script,
                    att.repro.failure_summary if repro_failed else "", "\n".join(new_fail))},
            ], Reflection)
        except llm.QuotaExhausted:
            raise
        except llm.LLMError as e:
            self.ev("error", attempt=att.n, error=f"reflection failed: {e}")
            return
        self._reflections[att.n] = ref.analysis
        self.ev("info", attempt=att.n, check="reflect", verdict=ref.repro_verdict,
                expected_failures=ref.expected_failures, analysis=ref.analysis)

        # (a) repro judged wrong → rewrite it once; new script must still fail on base and is re-run on the patch
        if repro_failed and ref.repro_verdict == "repro_wrong" and self._repro_fixes < REPRO_FIXES:
            self._repro_fixes += 1
            try:
                self._fix_repro(att, plan, ref.analysis)
            except llm.QuotaExhausted:
                raise
            except llm.LLMError as e:
                self.ev("error", attempt=att.n, error=f"repro fix failed: {e}")

        # (b) regression failures judged to be old-behaviour assertions → accept only if ALL are covered
        if att.regression is not None and not att.regression.passed and new_fail and ref.expected_failures:
            expected = set(ref.expected_failures)
            uncovered = [f for f in new_fail
                         if not any(x and (x in f or f.split(":")[0].endswith(x)) for x in expected)]
            if not uncovered:
                att.regression = TestResult(
                    passed=True, exec=att.regression.exec,
                    failure_summary="ACCEPTED by reflection — these tests assert the old behaviour the issue asks "
                                    "to change:\n" + "\n".join(new_fail) + f"\nAnalysis: {ref.analysis}")
                self.ev("info", attempt=att.n, check="regression_accepted", accepted=new_fail)

    def _fix_repro(self, att: Attempt, plan: Plan, analysis: str) -> None:
        assert self.sb is not None and att.repro is not None
        text = llm.complete("coder", [
            {"role": "system", "content": prompts.REPRO_SYSTEM},
            {"role": "user", "content": prompts.repro_fix_user(self.issue, plan, self.context_text,
                                                               self.state.repro_script, att.repro.failure_summary,
                                                               analysis)},
        ], temperature=0.3)
        script = _extract_python(text)
        # validate on the base commit (patch stashed)
        self._stash()
        try:
            base = self.sb.run_repro(script)
        finally:
            self._unstash()
        valid, why = self._repro_valid(base, _final_exception(base))
        self.ev("info", attempt=att.n, check="repro_fix", valid_on_base=valid, reason=why, script=script[:4000])
        if not valid:
            return
        self.state.repro_script = script
        self.state.repro_base = base
        att.repro = self.sb.run_repro(script)
        self.ev("info", attempt=att.n, check="repro", passed=att.repro.passed, note="re-run with fixed repro")

    def _stash(self) -> None:
        assert self.sb is not None
        st = self.sb.exec("git stash push -u -q", timeout=60)
        if st.exit_code != 0:
            raise RuntimeError(f"git stash failed: {st.stderr[-300:]}")

    def _unstash(self) -> None:
        assert self.sb is not None
        # repro_issue.py is untracked and stashed with -u; a newer copy written meanwhile blocks the pop
        pop = self.sb.exec(f"rm -f {REPRO_FILE} && git stash pop -q", timeout=60)
        if pop.exit_code != 0:
            raise RuntimeError(f"git stash pop failed: {pop.stderr[-300:]}")

    @staticmethod
    def _attempt_ok(att: Attempt) -> bool:
        if not att.diff.strip() or att.apply_errors:
            return False
        if att.repro is not None and not att.repro.passed:
            return False
        if att.regression is not None and not att.regression.passed:
            return False
        return att.repro is not None or att.regression is not None  # need at least one real signal

    def _best_failed(self) -> Attempt | None:
        def score(a: Attempt) -> tuple:
            return (bool(a.diff.strip()), not a.apply_errors, bool(a.repro and a.repro.passed),
                    bool(a.regression and a.regression.passed), a.n)
        cands = [a for a in self.state.attempts if a.diff.strip()]
        return max(cands, key=score) if cands else None

    # -------------------------------------------------------------- regression
    def _regression_files(self, att: Attempt) -> list[str]:
        modified = sorted({e.file_path for e in att.edits if e.file_path.endswith(".py")})
        scores: dict[str, float] = {}
        for path in modified:
            leaf = Path(path).stem if Path(path).stem != "__init__" else Path(path).parent.name
            for mod in _module_names(path):
                pattern = rf"(from|import)\s+{re.escape(mod)}\b|{re.escape(mod)}"
                try:
                    r = subprocess.run(["git", "grep", "-l", "-E", "-e", pattern, "--", "*.py"],
                                       cwd=self.repo_dir, capture_output=True, text=True, timeout=30)
                    hits = [h for h in r.stdout.splitlines() if is_test_path(h)]
                except subprocess.TimeoutExpired:
                    hits = []
                for h in hits:
                    scores[h] = scores.get(h, 0) + 1
            # test file named after the module, anywhere (e.g. sympy/core/tests/test_power.py)
            try:
                r = subprocess.run(["git", "ls-files", f"*test_{leaf}.py", f"*tests_{leaf}.py", f"*{leaf}_test.py"],
                                   cwd=self.repo_dir, capture_output=True, text=True, timeout=30)
                for h in r.stdout.splitlines():
                    if is_test_path(h):
                        scores[h] = scores.get(h, 0) + 5
            except subprocess.TimeoutExpired:
                pass
        ranked = sorted(scores, key=lambda f: (-scores[f], len(f), f))
        if self.issue.repo == "django/django":
            ranked = [f for f in ranked if f.startswith("tests/")]
        return ranked[:MAX_REGRESSION_FILES]

    def _run_file(self, f: str) -> tuple[TestResult, dict[str, str]]:
        """Run ONE test file (test names collide across files) and parse the FULL log before trimming."""
        assert self.sb is not None
        r = self.sb.run_tests([f], max_output=REGRESSION_PARSE_CHARS)
        statuses = _parse_statuses(self.issue.repo, r.exec.stdout + "\n" + r.exec.stderr)
        ex = r.exec.model_copy(update={"stdout": r.exec.stdout[-REGRESSION_STORE_CHARS:],
                                       "stderr": r.exec.stderr[-REGRESSION_STORE_CHARS:]})
        return TestResult(passed=r.passed, exec=ex, failure_summary=r.failure_summary), statuses

    def _run_regression(self, files: list[str], n: int) -> TestResult:
        missing = [f for f in files if f not in self._baseline]
        if missing:  # baseline on the base commit: stash the patch, run, restore
            self._stash()
            try:
                for f in missing:
                    self._baseline[f] = self._run_file(f)[1]
            finally:
                self._unstash()
            self.ev("info", check="regression_baseline", files=missing,
                    counts={f: len(self._baseline[f]) for f in missing})

        new_failures: list[str] = []
        summaries: list[str] = []
        all_passed_exit = True
        last: TestResult | None = None
        total_dur = 0.0
        for f in files:
            res, now = self._run_file(f)
            last = res
            total_dur += res.exec.duration_s
            all_passed_exit &= res.passed
            base = self._baseline[f]
            if not base and not now:  # unparseable both times → rely on exit status
                if not res.passed:
                    new_failures.append(f"{f}: run failed (exit {res.exec.exit_code}, no parseable results)")
                    summaries.append(res.failure_summary)
                continue
            def _fails(cur_map: dict[str, str]) -> dict[str, str]:
                return {t: cur_map.get(t, "MISSING") for t, st in base.items()
                        if _ok(st) and not _ok(cur_map.get(t, "MISSING"))}

            failing = _fails(now)
            if failing:  # confirm with a second patched run: flaky tests must fail both times
                res2, now2 = self._run_file(f)
                total_dur += res2.exec.duration_s
                again = _fails(now2)
                flaky = sorted(set(failing) - set(again))
                if flaky:
                    self.ev("info", check="regression_flaky_ignored", file=f, tests=flaky[:20])
                failing = {t: failing[t] for t in failing if t in again}
            file_fail = []
            for test, cur in failing.items():
                note = cur if cur != "MISSING" else "MISSING (not run — collection/import error?)"
                file_fail.append(f"{f}::{test}: {note}")
            if file_fail:
                new_failures += file_fail
                summaries.append(failure_summary(res.exec.stdout, res.exec.stderr, 30))
        self._new_failures[n] = new_failures
        summary = ""
        if new_failures:
            summary = "NEW failures vs base commit:\n" + "\n".join(new_failures[:40]) + "\n\n" + \
                      "\n\n".join(summaries)[-4000:]
        assert last is not None
        ex = last.exec.model_copy(update={"duration_s": round(total_dur, 3),
                                          "exit_code": 0 if all_passed_exit else last.exec.exit_code})
        return TestResult(passed=not new_failures, exec=ex, failure_summary=summary)


def run(instance_id: str, run_id: str | None = None) -> RunState:
    return Orchestrator(instance_id, run_id).run()
