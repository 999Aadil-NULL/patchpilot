"""All prompt templates. Keep them here so they're easy to tune in one place."""
from __future__ import annotations

from agent.schemas import Issue

ISSUE_CHARS = 12000


def _issue_block(issue: Issue) -> str:
    hints = f"\n\n## Hints (from the issue discussion)\n{issue.hints_text[:3000]}" if issue.hints_text.strip() else ""
    return (f"Repository: {issue.repo} (version {issue.version})\n\n"
            f"## Issue\n{issue.problem_statement[:ISSUE_CHARS]}{hints}")


# ------------------------------------------------------------------ PLAN
PLAN_SYSTEM = (
    "You are a senior software engineer triaging a GitHub issue in a large Python codebase. "
    "You do not have the code yet; reason from the issue text and your knowledge of the project. "
    "Be concrete and concise."
)


def plan_user(issue: Issue) -> str:
    return (
        f"{_issue_block(issue)}\n\n"
        "Produce a fix plan:\n"
        "- kind: 'bug' or 'feature'\n"
        "- summary: one or two sentences on what is wrong / requested\n"
        "- hypothesis: where in the code the problem most likely lives and why (module/class/function)\n"
        "- steps: 3-6 implementation steps\n"
        "- acceptance_criteria: observable behaviours that prove the fix (what a test would assert)\n"
        "- search_terms: 5-15 identifiers, function/class names, module paths (e.g. 'sympy/core/power.py'),"
        " error-message fragments to grep for. Prefer exact identifiers from the issue."
    )


# ------------------------------------------------------------------ LOCALIZE: file pick
LOCALIZE_SYSTEM = (
    "You are localizing the source files that must be edited to resolve an issue. "
    "You see a repository map: file paths with their top-level classes/functions and method signatures."
)


def localize_files_user(issue: Issue, plan_summary: str, hypothesis: str, repo_map: str, max_files: int) -> str:
    return (
        f"{_issue_block(issue)}\n\n## Plan\nSummary: {plan_summary}\nHypothesis: {hypothesis}\n\n"
        f"## Repository map (candidate files)\n{repo_map}\n\n"
        f"Pick at most {max_files} files (from the map, exact paths) most likely to need edits, "
        "most likely first. Exclude tests. Return JSON: "
        '{"files": ["path/a.py", ...], "reason": "short justification"}'
    )


# ------------------------------------------------------------------ LOCALIZE: exploration loop
EXPLORE_SYSTEM = (
    "You are inspecting code to find exactly where an issue must be fixed. Each turn you may take ONE action:\n"
    '  {"action": "read_symbol", "name": "ClassName.method_name or function_name"}  — show a definition\n'
    '  {"action": "grep", "pattern": "regex"}  — search the repo (python files) for a pattern\n'
    '  {"action": "done"}  — when the shown code is enough to write the fix\n'
    "Reply with exactly one JSON object in a ```json fence."
)


def explore_user(issue: Issue, plan_summary: str, hypothesis: str, files: list[str], chunks_text: str) -> str:
    return (
        f"{_issue_block(issue)}\n\n## Plan\nSummary: {plan_summary}\nHypothesis: {hypothesis}\n\n"
        f"## Candidate files\n" + "\n".join(files) + "\n\n"
        f"## Code retrieved so far\n{chunks_text}\n\n"
        "Is the code needed for the fix shown? If something important is missing (a called helper, the "
        "method that actually raises, a definition referenced in the issue), request it. Otherwise say done."
    )


# ------------------------------------------------------------------ shared: code context
def context_block(chunks_text: str, files_text: str) -> str:
    out = ""
    if files_text:
        out += f"## Full source of key files\n{files_text}\n\n"
    if chunks_text:
        out += f"## Relevant code (file:line ranges)\n{chunks_text}\n\n"
    return out


def plan_block(plan) -> str:
    steps = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan.steps))
    acc = "\n".join(f"  - {a}" for a in plan.acceptance_criteria)
    return (f"## Plan ({plan.kind})\nSummary: {plan.summary}\nHypothesis: {plan.hypothesis}\n"
            f"Steps:\n{steps}\nAcceptance criteria:\n{acc}\n\n")


# ------------------------------------------------------------------ REPRODUCE
REPRO_SYSTEM = (
    "You write minimal standalone Python reproduction scripts for bug reports. The script is saved as "
    "/testbed/repro_issue.py and run with `python repro_issue.py` from the repo root, where the package is "
    "installed in development mode. The Python version may be old (3.6–3.9): avoid walrus, match, "
    "f-string '=' specifiers and other new syntax."
)

_DJANGO_NOTE = (
    "- This is Django: at the top call `from django.conf import settings; settings.configure(...)` inline "
    "(INSTALLED_APPS as needed, DATABASES with sqlite ':memory:', USE_TZ etc. if relevant), then "
    "`import django; django.setup()` before importing models/forms. Do not rely on a settings module.\n"
)


def repro_user(issue: Issue, plan, context: str, feedback: str = "") -> str:
    django = _DJANGO_NOTE if issue.repo == "django/django" else ""
    fb = f"\n## Your previous script was rejected\n{feedback}\nWrite a corrected script.\n" if feedback else ""
    return (
        f"{_issue_block(issue)}\n\n{plan_block(plan)}{context}"
        "Write `repro_issue.py` that checks the EXPECTED (fixed) behaviour described in the issue:\n"
        "- On the current buggy code it must FAIL: via a failing `assert` (with a helpful message) or by the "
        "exception the issue describes propagating uncaught.\n"
        "- After a correct fix it must exit 0 and print 'OK'.\n"
        "- Use only the repo's public API and the standard library. No pytest, no network, no files outside /tmp.\n"
        "- Keep it short and deterministic. Do not try/except the bug away.\n"
        f"{django}{fb}\n"
        "Return ONLY the script in a single ```python fenced block."
    )


# ------------------------------------------------------------------ PATCH
PATCH_SYSTEM = (
    "You are an expert software engineer fixing an issue in a Python repository. You make minimal, correct, "
    "general fixes (not special-cased to the reproduction script). Never edit test files or repro_issue.py.\n\n"
    "Output ONLY edit blocks in this exact format (one or more):\n"
    "FILE: path/relative/to/repo.py\n"
    "<<<<<<< SEARCH\n"
    "exact existing lines (copy verbatim, including indentation; include enough lines to be unique)\n"
    "=======\n"
    "replacement lines\n"
    ">>>>>>> REPLACE\n\n"
    "Rules: SEARCH must match the current file exactly. Keep blocks small (a few to ~30 lines). "
    "An empty SEARCH creates a new file. You may write a brief analysis before the blocks.\n"
    "IMPORTANT: Only edit code that appears VERBATIM in the code context you are given — copy SEARCH lines "
    "from it character for character. The plan was written before reading the code and may name functions "
    "that do not exist; trust the code, not the plan. Never invent functions or lines in SEARCH."
)


def patch_user(issue: Issue, plan, context: str, repro_script: str, repro_base_summary: str,
               history: str = "") -> str:
    repro = ""
    if repro_script:
        repro = (f"## Reproduction script (fails on current code; must pass after your fix)\n```python\n"
                 f"{repro_script}\n```\nCurrent failure:\n```\n{repro_base_summary[-3000:]}\n```\n\n")
    hist = ""
    if history:
        hist = (f"## Previous attempts (the tree has been reset to the original code)\n{history}\n\n"
                "First, in 2-4 sentences, explain why the previous attempt failed. Then give a corrected "
                "full set of edit blocks against the ORIGINAL code.\n\n")
    return f"{_issue_block(issue)}\n\n{plan_block(plan)}{context}{repro}{hist}Write the fix now."


def lint_repair_user(errors: str) -> str:
    return ("Your edits applied, but the result does not compile:\n\n"
            f"{errors}\n\n"
            "The tree has been reset to the original code. Give a corrected full set of edit blocks against the "
            "ORIGINAL code, in the same format.")


# ------------------------------------------------------------------ REFLECT
REFLECT_SYSTEM = (
    "You review a failed verification of a candidate patch. Be skeptical and precise. Decide, based on the "
    "issue text, whether each failure is the PATCH's fault, or whether the verification itself is wrong: "
    "(a) the reproduction script asserts something the issue does not actually require (e.g. wrong expected "
    "value/repr/size), or (b) an existing test asserts the OLD behaviour that the issue explicitly asks to "
    "change. Only call a failure expected when the issue clearly demands the new behaviour."
)


def reflect_user(issue: Issue, plan, diff: str, repro_script: str, repro_out: str, new_failures: str) -> str:
    repro = ""
    if repro_script:
        repro = (f"## Reproduction script\n```python\n{repro_script}\n```\n"
                 f"## Reproduction result after patch\n```\n{repro_out[-3000:] or 'PASSED'}\n```\n\n")
    reg = f"## Existing tests that newly fail after the patch\n```\n{new_failures[-5000:]}\n```\n\n" if new_failures else ""
    return (
        f"{_issue_block(issue)}\n\n{plan_block(plan)}## Candidate patch\n```diff\n{diff[:8000]}\n```\n\n{repro}{reg}"
        "Return JSON with:\n"
        '- "repro_verdict": "patch_wrong" | "repro_wrong" | "passed"  (passed if the repro passed)\n'
        '- "expected_failures": list of the newly failing test ids above that fail ONLY because they assert the '
        "old behaviour the issue asks to change (empty if none or unsure)\n"
        '- "analysis": 2-5 sentences: what is wrong and what the next patch attempt should do differently'
    )


def repro_fix_user(issue: Issue, plan, context: str, script: str, patched_out: str, analysis: str) -> str:
    return (
        f"{repro_user(issue, plan, context)}\n\n"
        f"## Your previous script was judged INCORRECT (it asserts something the issue does not require)\n"
        f"```python\n{script}\n```\nOutput with a candidate fix applied:\n```\n{patched_out[-2500:]}\n```\n"
        f"Reviewer analysis: {analysis}\n"
        "Write a corrected script that checks only what the issue actually requires."
    )
