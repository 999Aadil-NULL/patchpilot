# CLAUDE.md — PatchPilot (Meta Global AI Developer Hackathon 2026)

Autonomous SWE agent: takes a SWE-bench Lite issue → plans → localizes code → writes a
failing reproduction → patches → verifies in Docker → self-corrects (≤3) → independent
oracle grading → human approval. Deadline: **Nov 5 2026, 06:30 IST**. Reliability > features.

## How to work in this repo (read first)
- Do ONLY the build step you are asked for. Run its acceptance check. Stop and report.
- Never refactor other modules' public interfaces; all cross-module data uses `agent/schemas.py`.
- Environment: WSL2 Ubuntu, Python 3.11, `uv` for deps (`uv add <pkg>`, `uv run ...`).
  Project lives on the Linux FS. LF line endings only (`.gitattributes: * text=auto eol=lf`).
- Every external call (LLM, docker exec) has a timeout. Every stage emits events (see Events).
- Prefer simple, explicit code over frameworks. No LangChain/LlamaIndex. No function-calling
  APIs (providers differ) — models return text/JSON we parse ourselves.
- **Integrity rule:** the agent pipeline must NEVER see `patch`, `test_patch`, `FAIL_TO_PASS`,
  or `PASS_TO_PASS` from the dataset. Only `oracle.py` may touch those.

## Stack
openai (SDK, used for every provider via base_url), pydantic v2, python-dotenv, tenacity,
docker (SDK), swebench, datasets, tree-sitter + tree-sitter-python, rank-bm25, fastembed,
qdrant-client (in-memory), streamlit, rich.

## Layout
```
agent/
  schemas.py      # pydantic contracts (below) — single source of truth
  config.py       # loads .env; role configs
  llm.py          # provider-agnostic client, JSON extraction, disk cache, cost log
  events.py       # emit() → runs/<run_id>/events.jsonl
  dataset.py      # load SWE-bench Lite instance → Issue (strips hidden fields)
  sandbox.py      # Docker container over SWE-bench instance image
  localize/
    repomap.py    # tree-sitter: file → top-level classes/functions + signatures
    ast_index.py  # tree-sitter chunks → FastEmbed → Qdrant(":memory:")
    retrieve.py   # BM25 prefilter → LLM file pick → embed rank → read_symbol/grep loop
  patcher.py      # SEARCH/REPLACE parse + apply (exact → normalized → fuzzy)
  prompts.py      # all prompt templates
  orchestrator.py # state machine
  oracle.py       # fresh container, official test_patch + FAIL_TO_PASS/PASS_TO_PASS
scripts/
  select_instances.py  # filter Lite → data/candidates.json
  run_one.py           # CLI: uv run python -m scripts.run_one --instance <id>
ui/app.py              # Streamlit dashboard
cache/repos/<instance_id>/   # host copy of /testbed (for localization)
runs/<run_id>/               # events.jsonl, state.json, final.patch
```

## Config (.env) — model is swappable; Meta plugged in at the end
```
PLANNER_BASE_URL=   PLANNER_API_KEY=   PLANNER_MODEL=
CODER_BASE_URL=     CODER_API_KEY=     CODER_MODEL=
LLM_CACHE=1         # cache responses by hash(model+messages) in cache/llm/
LLM_TIMEOUT_S=120
MAX_ATTEMPTS=3
```
`.env.example` presets (OpenAI-compatible endpoints):
- Gemini: `https://generativelanguage.googleapis.com/v1beta/openai/` (model id from AI Studio)
- Groq: `https://api.groq.com/openai/v1`
- Ollama (installed inside WSL): `http://localhost:11434/v1`, key `ollama`, model `qwen2.5-coder:7b`
- Meta: `<fill at final stage>` — if Meta's API is not OpenAI-compatible, add ONE adapter
  class in llm.py implementing the same `complete()` signature. Nothing else changes.

## Schemas (agent/schemas.py)
```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class Issue(BaseModel):
    instance_id: str; repo: str; base_commit: str; version: str
    problem_statement: str; hints_text: str = ""

class Plan(BaseModel):
    kind: Literal["bug", "feature"]
    summary: str; hypothesis: str
    steps: list[str]; acceptance_criteria: list[str]
    search_terms: list[str]          # identifiers/files/strings to grep for

class CodeChunk(BaseModel):
    file_path: str; symbol: str; kind: Literal["function","class","method","module"]
    start_line: int; end_line: int; code: str; score: float = 0.0

class ContextBundle(BaseModel):
    candidate_files: list[str]; chunks: list[CodeChunk]; notes: str = ""

class EditBlock(BaseModel):
    file_path: str; search: str; replace: str   # empty search = create new file

class ExecResult(BaseModel):
    exit_code: int; stdout: str; stderr: str; duration_s: float; timed_out: bool = False

class TestResult(BaseModel):
    passed: bool; exec: ExecResult; failure_summary: str = ""   # trimmed traceback

class Attempt(BaseModel):
    n: int; edits: list[EditBlock] = []; apply_errors: list[str] = []
    diff: str = ""; repro: Optional[TestResult] = None
    regression: Optional[TestResult] = None; regression_files: list[str] = []

class OracleResult(BaseModel):
    resolved: bool; fail_to_pass: dict[str, str] = {}; pass_to_pass: dict[str, str] = {}
    log_tail: str = ""

class RunState(BaseModel):
    run_id: str; issue: Issue
    status: Literal["planning","localizing","reproducing","patching","verifying",
                    "oracle","awaiting_review","approved","failed"] = "planning"
    plan: Optional[Plan] = None; context: Optional[ContextBundle] = None
    repro_script: str = ""; repro_base: Optional[TestResult] = None
    attempts: list[Attempt] = []; oracle: Optional[OracleResult] = None
    failure_reason: str = ""; tokens_in: int = 0; tokens_out: int = 0
```

## Events (agent/events.py)
`emit(run_id, stage, type, **payload)` appends `{"ts","stage","type",...}` to
`runs/<run_id>/events.jsonl`. Types: `start`, `llm_call` (role, model, tokens, latency,
cached), `info`, `result`, `error`, `end`. Orchestrator also writes `state.json` after each
stage. The UI reads only these files.

## Module specs

### llm.py
`complete(role: Literal["planner","coder"], messages, *, temperature=0.2, max_tokens=4096) -> str`
- openai SDK with role's base_url/key/model; tenacity retry (3, exp backoff) on 429/5xx/timeouts.
- Disk cache when LLM_CACHE=1 (makes demo replays deterministic + saves credits).
- Emits `llm_call` event with token usage (if returned) and latency.
- `complete_json(role, messages, schema: type[BaseModel])`: ask for JSON in a ```json fence,
  extract (fenced → first `{...}` balanced), validate with pydantic; on failure, one repair
  retry that includes the validation error.

### dataset.py
Load `princeton-nlp/SWE-bench_Lite` (split `test`) once, cache to `data/lite.json`.
`get_issue(id) -> Issue` (hidden fields stripped). `get_raw(id) -> dict` (oracle only).

### sandbox.py
SWE-bench instance images: repo at `/testbed`, conda env `testbed`.
Image name: derive via the installed `swebench` harness (TestSpec / instance image key;
namespace `swebench`, `__` in instance_id becomes `_1776_`). Verify by `docker pull`.
```python
class Sandbox:
    def __init__(self, instance_id: str, run_id: str): ...
    def start(self) -> None          # long-lived container, network disabled, mem_limit="6g"
    def exec(self, cmd: str, timeout: int = 300) -> ExecResult
        # bash -lc "source /opt/miniconda3/bin/activate && conda activate testbed &&
        #           cd /testbed && timeout {timeout} {cmd}"
    def read_file(self, path: str) -> str
    def write_file(self, path: str, content: str) -> None   # via put_archive (tar in memory)
    def reset(self) -> None          # git checkout -- . && git clean -fdq (keep nothing)
    def diff(self) -> str            # git diff (excluding repro_issue.py)
    def export_repo(self, dest: Path) -> None  # get_archive /testbed → cache/repos/<id>/ once
    def run_repro(self, script: str) -> TestResult   # writes /testbed/repro_issue.py, runs it
    def run_tests(self, test_files: list[str]) -> TestResult
        # use the repo's test command from swebench specs for this repo/version + files
    def stop(self) -> None
```
`failure_summary` = last ~60 lines of combined output, prioritizing the final traceback.

### oracle.py (the ONLY consumer of hidden fields)
`grade(instance_id, diff: str) -> OracleResult`: FRESH container from the same image, apply
`diff` with `git apply`, then run the swebench TestSpec eval script (applies official
`test_patch` + runs tests), parse with the swebench log parser/grading for that repo, compute
resolved = all FAIL_TO_PASS pass AND all PASS_TO_PASS pass.

### localize/
- repomap.py: tree-sitter-python; for each non-test `.py` file → path + top-level
  `class`/`def` signatures (+ methods one level deep). Cache per instance.
- retrieve.py `localize(issue, plan, repo_dir) -> ContextBundle`:
  1. BM25 over (path + symbol names) using plan.search_terms + issue text → top 40 files.
     Also boost files/lines named in tracebacks in the issue.
  2. Planner sees repo map of those 40 → returns ≤5 candidate files (JSON).
  3. ast_index: chunk those files (function/method/class), embed with FastEmbed
     (`jinaai/jina-embeddings-v2-base-code`; fallback `BAAI/bge-small-en-v1.5`), Qdrant
     `:memory:`, query = summary + hypothesis → top 8 chunks.
  4. Exploration loop ≤3 turns: model replies JSON
     `{"action":"read_symbol","name":"X"}` | `{"action":"grep","pattern":"..."}` | `{"action":"done"}`.

### patcher.py
Coder output format (only this, possibly multiple blocks):
```
FILE: path/relative/to/repo.py
<<<<<<< SEARCH
exact existing lines
=======
replacement lines
>>>>>>> REPLACE
```
`parse_edits(text) -> list[EditBlock]`; `apply_edits(sandbox, edits) -> list[str] errors`:
exact unique match → else line-wise rstrip-normalized match → else difflib window with
ratio ≥ 0.92 → else error message (shown to model on retry). Empty SEARCH = new file.
Unit tests in `tests/test_patcher.py` for all three match tiers + ambiguous match error.

### orchestrator.py — state machine
```
PLAN → LOCALIZE → REPRODUCE → [PATCH → VERIFY → (REFLECT)]×≤3 → ORACLE → AWAITING_REVIEW
```
- REPRODUCE: coder writes standalone `repro_issue.py` (asserts expected behaviour; for
  django, `settings.configure()` inline). Must FAIL on base commit with AssertionError or the
  exception the issue describes — NOT ImportError/SyntaxError/NameError. Up to 2 rewrites.
  If no valid repro after that: continue with regression-only verification, note it.
- PATCH: `sandbox.reset()` first (fresh tree every attempt). Prompt = plan + context chunks +
  repro script + (prior attempt edits + failure_summary + apply_errors).
- VERIFY: repro passes AND regression passes. Regression files = existing test files that
  import/mention the modified modules (grep), max 3 files; capture base-commit result of those
  files once and only count NEW failures.
- ORACLE runs after verify success (and also on the best failed attempt, for the record).
- Any exception → status "failed" with failure_reason; never crash the UI.

### ui/app.py (Streamlit)
Sidebar: instance picker (data/candidates.json) + "Run" (spawns `scripts/run_one.py` subprocess)
+ past runs. Main: stage timeline from events.jsonl (auto-refresh 2s), plan card, localized
chunks, repro script + base failure, attempts tabs (edits, diff, test output), two badges:
"Agent verification" and "Official SWE-bench grading", token/cost counter, **Approve** →
writes runs/<id>/final.patch and commits it on branch `agent/<instance_id>` in cache/repos copy.

### scripts/select_instances.py
Filter Lite to repos {sympy/sympy, psf/requests, pytest-dev/pytest, pallets/flask,
django/django}; gold patch touches 1 file and < 15 changed lines → data/candidates.json.

## Build steps (do one at a time)
0. Scaffold: uv project, deps, layout, .gitattributes, .gitignore (runs/, cache/, .env, data/),
   .env.example, schemas.py, config.py, events.py. Check: `uv run python -c "import agent.schemas"`.
1. llm.py + smoke script calling planner and coder with "say OK" and a complete_json test.
2. dataset.py + select_instances.py + sandbox.py + oracle.py.
   **Check (critical):** for one sympy candidate, `grade(id, gold_patch)` → resolved=True and
   `grade(id, "")` → resolved=False. Also run_repro of a trivial failing script works.
3. localize/ — check: for 3 candidates, gold patch's file is in candidate_files (print hit rate).
4. patcher.py + tests/test_patcher.py passing.
5. orchestrator.py + scripts/run_one.py — check: one instance end-to-end, events.jsonl populated.
6. ui/app.py.
7. Batch run 10–15 candidates, record resolve rate in README.