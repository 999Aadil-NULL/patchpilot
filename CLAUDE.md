# CLAUDE.md — PatchPilot (Meta Global AI Developer Hackathon 2026)

Autonomous SWE agent: takes a SWE-bench Lite issue → plans → localizes code → writes a
failing reproduction → patches → verifies in Docker → self-corrects (≤3) → independent
oracle grading → human approval. Deadline: **Nov 5 2026, 06:30 IST**. Reliability > features.

## How to work in this repo (read first)
- Do ONLY the build step you are asked for. Run its acceptance check. Stop and report.
- Never refactor other modules' public interfaces; all cross-module data uses `agent/schemas.py`.
  Additive changes are allowed: new OPTIONAL schema fields with defaults, new OPTIONAL keyword
  arguments with defaults. Existing call sites must keep working unchanged.
- Environment: WSL2 Ubuntu, Python 3.11, `uv` for deps (`uv add <pkg>`, `uv run ...`).
  Project lives on the Linux FS. LF line endings only (`.gitattributes: * text=auto eol=lf`).
- Every external call (LLM, docker exec) has a timeout. Every stage emits events (see Events).
- Prefer simple, explicit code over frameworks. No LangChain/LlamaIndex. No function-calling
  APIs (providers differ) — models return text/JSON we parse ourselves.
- **Integrity rule:** the agent pipeline must NEVER see `patch`, `test_patch`, `FAIL_TO_PASS`,
  or `PASS_TO_PASS` from the dataset. Only `oracle.py` may touch those. `scripts/run_batch.py`
  may read the gold patch ONLY after a run finishes, to score localization.
- Machine: ~7.8 GB WSL RAM, 8 GB RTX 4060. Runs are strictly sequential.

## Stack
openai (SDK, used for every provider via base_url), pydantic v2, python-dotenv, tenacity,
docker (SDK), swebench, datasets, tree-sitter + tree-sitter-python, rank-bm25, fastembed,
qdrant-client (in-memory), streamlit, rich.

## Layout
```
agent/
  schemas.py      # pydantic contracts (below) — single source of truth
  config.py       # loads .env; role configs; harness feature flags (Phase 2)
  llm.py          # provider-agnostic client, JSON extraction, disk cache, cost log
  events.py       # emit() → runs/<run_id>/events.jsonl
  dataset.py      # load SWE-bench Lite instance → Issue (strips hidden fields)
  sandbox.py      # Docker container over SWE-bench instance image
  localize/
    repomap.py    # tree-sitter: file → top-level classes/functions + signatures
    ast_index.py  # tree-sitter chunks → FastEmbed → Qdrant(":memory:")
    retrieve.py   # BM25 prefilter → LLM file pick → embed rank → read_symbol/grep loop
    trace.py      # (Phase 2, H5) repro execution trace → fault-localization boost
  patcher.py      # SEARCH/REPLACE parse + apply (exact → normalized → fuzzy)
  guards.py       # (Phase 2, H2) post-edit syntax guardrail
  select.py       # (Phase 2, H4) execution-based candidate patch selection
  prompts.py      # all prompt templates
  orchestrator.py # state machine
  oracle.py       # fresh container, official test_patch + FAIL_TO_PASS/PASS_TO_PASS
scripts/
  select_instances.py  # filter Lite → data/candidates.json
  run_one.py           # CLI: uv run python -m scripts.run_one --instance <id>
  run_batch.py         # batch runner (resumable; skips ids already in results file)
  audit_context.py     # (Phase 2, H0) context-window truncation audit
ui/app.py              # Streamlit dashboard
cache/repos/<instance_id>/   # host copy of /testbed (for localization)
runs/<run_id>/               # events.jsonl, state.json, final.patch
results/                     # batch results; Phase 2 writes results/<tag>/
```

## Config (.env) — model is swappable; Meta plugged in at the end
```
PLANNER_BASE_URL=   PLANNER_API_KEY=   PLANNER_MODEL=
CODER_BASE_URL=     CODER_API_KEY=     CODER_MODEL=
LLM_CACHE=1         # cache responses by hash(model+messages[+sample_idx]) in cache/llm/
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

## Module specs (v1 — implemented)

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
    def read_file(self, path: str) -> str
    def write_file(self, path: str, content: str) -> None   # via put_archive (tar in memory)
    def reset(self) -> None          # git checkout -- . && git clean -fdq (keep nothing)
    def diff(self) -> str            # git diff (excluding repro_issue.py)
    def export_repo(self, dest: Path) -> None  # get_archive /testbed → cache/repos/<id>/ once
    def run_repro(self, script: str) -> TestResult   # writes /testbed/repro_issue.py, runs it
    def run_tests(self, test_files: list[str]) -> TestResult
    def stop(self) -> None
```
`failure_summary` = last ~60 lines of combined output, prioritizing the final traceback.

### oracle.py (the ONLY consumer of hidden fields)
`grade(instance_id, diff: str) -> OracleResult`: FRESH container from the same image, apply
`diff` with `git apply`, then run the swebench TestSpec eval script, parse with the swebench
log parser for that repo; resolved = all FAIL_TO_PASS pass AND all PASS_TO_PASS pass.

### localize/
- repomap.py: tree-sitter-python; non-test `.py` file → path + top-level class/def signatures
  (+ methods one level deep). Cached per instance.
- retrieve.py `localize(issue, plan, repo_dir) -> ContextBundle`:
  1. BM25 over (path + symbol names) using plan.search_terms + issue text → top 40 files,
     boosted by files/lines named in tracebacks in the issue.
  2. Planner sees repo map of those 40 → returns ≤5 candidate files (JSON).
  3. ast_index: chunk → FastEmbed (`jinaai/jina-embeddings-v2-base-code`; fallback
     `BAAI/bge-small-en-v1.5`) → Qdrant `:memory:` → top 8 chunks for summary + hypothesis.
  4. Exploration loop ≤3 turns: `read_symbol` | `grep` | `done` (JSON actions).

### patcher.py
SEARCH/REPLACE blocks (`FILE:` header, `<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE`).
Apply tiers: exact unique → rstrip-normalized → difflib window ratio ≥ 0.92 → error message.
Empty SEARCH = new file. Tests in `tests/test_patcher.py`.

### orchestrator.py — state machine
```
PLAN → LOCALIZE → REPRODUCE → [PATCH → VERIFY → (REFLECT)]×≤3 → ORACLE → AWAITING_REVIEW
```
- REPRODUCE: standalone `repro_issue.py`, must FAIL on base with AssertionError or the issue's
  exception — not ImportError/SyntaxError/NameError. ≤2 rewrites, else regression-only mode.
- PATCH: `sandbox.reset()` first. Prompt = plan + chunks + repro + prior attempt feedback.
- VERIFY: repro passes AND regression (≤3 test files mentioning modified modules; only NEW
  failures vs base count).
- ORACLE after verify success, and on the best failed attempt for the record.
- Any exception → status "failed" with failure_reason; never crash the UI.

### ui/app.py (Streamlit)
Instance picker + Run (spawns run_one) + past runs; stage timeline from events.jsonl; plan,
chunks, repro, attempt tabs, badges "Agent verification" / "Official SWE-bench grading",
token counter, Approve → final.patch + commit on `agent/<instance_id>` in cache/repos copy.

## v1 build steps — ALL DONE
0 scaffold · 1 llm · 2 dataset/sandbox/oracle · 3 localize · 4 patcher · 5 orchestrator ·
6 UI · 7 batch (13 instances, qwen2.5-coder:7b: 0/12 resolved fair, 9/13 gold file localized,
8/13 valid repro, 10/13 clean apply, 3 verifier false positives, 0 crashes).

---

# PHASE 2 — Harness v2: make a small local model useful

## Claim we want to be able to make (honestly)
"Structural harness improvements raise a 7B local model's stage metrics (and ideally
resolves) on the same instances, and those same improvements carry over to the Meta model."
The deliverable is an **ablation table**, not a single headline number. With 12 instances,
report it as diagnostic evidence, never as a benchmark score.

## Where the ideas come from (structural techniques only)
| Technique | Inspired by | What we take |
|---|---|---|
| Check the model actually sees the prompt | General small-model serving practice | Audit effective context length; silent truncation can masquerade as "the model is dumb" |
| Lint guardrail on edits | SWE-agent's agent–computer interface (edits that fail syntax checks are rejected); Aider's lint-after-edit | `py_compile` after every apply; cheap repair turn that does not burn an attempt |
| Narrow edit scope | Agentless hierarchical localization (file → function → edit location) | Model picks ≤3 edit locations first; patch prompt shows only those full symbols |
| Sample many, select by execution | Agentless (multiple candidate patches, filter by repro + regression, pick by majority over normalized diffs) | K candidates per attempt; selection uses ONLY agent-visible signals |
| Fault localization from the repro | AutoCodeRover (spectrum-based fault localization) | Trace which repo functions the failing repro executes; boost them |
| Repo map + SEARCH/REPLACE | Aider | Already in v1 — keep |

## Phase 2 principles
1. **Every technique is a flag, default OFF.** With all flags off, behaviour equals v1 (+ fixes
   from H1). A flag is only turned ON in the demo config after its acceptance check passes.
2. **Structural, not model-specific.** No prompt wording tuned to qwen quirks. Each technique
   must make sense for any model, including Meta's.
3. **Selection never touches hidden data.** Candidate selection uses repro, regression, lint
   and diff shape only. Oracle stays grading-only.
4. **Don't break the submission.** Work on branch `harness-v2`. `master` keeps the tagged v1.

## Flags (agent/config.py reads these; record ALL of them into every run's state/events)
```
HARNESS_LINT=0          # H2 syntax guardrail
HARNESS_FOCUS=0         # H3 edit-location narrowing
HARNESS_SAMPLES=1       # H4 candidates per attempt (1 = v1 behaviour)
HARNESS_SBFL=0          # H5 repro-trace fault localization (stretch)
```

## Steps (one at a time; acceptance check; stop and report)

### H0 — Safety + context-window audit (≤30 min)
- Confirm step-7 files are committed on `master`; `git tag v1-baseline`; create branch
  `harness-v2`. Archive `results/batch_results.json` → `results/qwen-v1/` and
  `runs/batch-*` → `runs/archive/qwen-v1/` (move, don't delete).
- `scripts/audit_context.py`: scan all `runs/**/events.jsonl` `llm_call` events; print
  per-role distribution of tokens_in (max, p50, p95) and how many calls cluster at a suspicious
  ceiling (e.g. within 2% of 2048/4096/8192).
- Determine the serving backend in use (Ollama vs llama.cpp `llama-server`) and its EFFECTIVE
  context length for `qwen2.5-coder:7b`. Report it; do not change it without asking.
- If truncation is likely: propose (do not auto-apply) the fix — for Ollama, a Modelfile with
  `PARAMETER num_ctx 16384` exposed as a new model tag, or the server's context-length setting.
  16k KV cache for this model is on the order of 1 GB, so it should fit on 8 GB VRAM — verify.
- **Acceptance:** audit printed; backend + effective context reported; tag + branch exist.

### H1 — Correctness fixes (≤30 min)
- Badge: when no valid repro exists, UI shows "PASS (regression only)" and batch scoring does
  NOT count it as agent-verified. Add `verification_mode: Literal["repro+regression",
  "regression_only"] = "repro+regression"` as an optional field where appropriate (additive).
- `_final_exception`: parse the LAST traceback's final line with a regex for a dotted
  exception name followed by `:` or end of line, regardless of suffix. Treat
  `ImportError, ModuleNotFoundError, SyntaxError, IndentationError, NameError,
  ImproperlyConfigured` as INVALID repro exceptions.
- **Acceptance:** unit tests in `tests/test_repro_validity.py` covering `ImproperlyConfigured`,
  a dotted exception (`django.core.exceptions.ImproperlyConfigured: ...`), a custom exception
  with no Error suffix, AssertionError (valid), no traceback. `uv run pytest -q` passes.

### H2 — Lint guardrail (≤40 min) — flag HARNESS_LINT
- `agent/guards.py`: `syntax_check(sandbox, files: list[str]) -> list[str]` runs
  `python -m py_compile <file>` in the testbed env for each modified `.py` file (timeout 30s).
- In PATCH: after `apply_edits`, if errors → ONE repair turn showing the compile error and the
  offending region (±10 lines), re-apply on a reset tree. Repair turns do not count toward
  MAX_ATTEMPTS. Still failing → record as apply_errors and move on as today.
- Emit `info` events `lint_fail` / `lint_repaired`.
- **Acceptance:** test with a deliberately broken edit → caught, repair prompt built; flag off
  → identical code path to v1.

### H3 — Narrow edit scope (≤60 min) — flag HARNESS_FOCUS
- After LOCALIZE (and before PATCH), one `complete_json` call: given plan + chunks, return
  `{"locations":[{"file_path":..., "symbol":...}]}` with ≤3 entries that MUST exist in the
  repo map (validate; drop unknowns; if none valid, fall back to v1 context).
- Patch prompt then contains the FULL source of those symbols (via tree-sitter, from
  cache/repos) plus a short list of other candidate chunks as signatures only. No line
  numbers inside code shown to the model (they leak into SEARCH blocks).
- New optional field `ContextBundle.edit_locations: list[str] = []`.
- **Acceptance:** on 2 instances, prompt tokens for PATCH drop vs v1, and the chosen locations
  are logged. (Whether the gold function was chosen is scored post hoc by run_batch only.)

### H4 — Sample K, select by execution (≤75 min) — flag HARNESS_SAMPLES
- `llm.complete` gets an OPTIONAL kwarg `sample_idx: int = 0` included in the cache key
  (otherwise all samples hit the same cache entry). Default keeps v1 cache keys unchanged.
- Per attempt, generate K candidates: sample 0 at temperature 0.2, others at 0.7.
  For each: reset → apply → (lint if HARNESS_LINT) → run repro. Run regression ONLY for
  candidates that pass repro (regression is the slow part).
- `agent/select.py` `select(candidates) -> int`:
  1. Prefer candidates passing repro AND regression.
  2. Among them, majority vote by normalized diff (strip whitespace, blank lines, comments).
  3. Tie → smallest diff.
  4. If none pass: pick the best failure (applied cleanly > lint clean > fewest new
     regression failures) and feed ITS failure_summary into the next attempt.
- Store all candidates on the Attempt (optional field `candidates: list[dict] = []` holding
  diff + per-candidate results summary). UI: show "K candidates, selected #i, why".
- In regression-only mode (no valid repro), selection uses regression + majority vote and the
  badge still says "regression only".
- **Acceptance:** K=3 on one instance end to end; events show 3 coder calls with distinct
  cache keys and a selection event with the reason. K=1 → v1 behaviour.
- Budget note: K multiplies patch-stage time. Use K=3 locally; on the Meta API mind credits.

### H5 — STRETCH: repro-trace fault localization (≤60 min) — flag HARNESS_SBFL
Only start if H0–H4 are done before 02:00 IST.
- `agent/localize/trace.py`: wrap the repro with a tiny `sys.settrace` runner (no new
  dependency in the testbed) that records `(file, function)` calls under `/testbed`, excluding
  tests and `repro_issue.py`; return counts.
- After a VALID failing repro exists, `refine_context(context, trace)` adds the top-5 traced
  functions as chunks (score boost) if not already present. No valid repro → no-op.
- **Acceptance:** on one instance with a valid repro, trace is non-empty and the refined
  chunks are logged.

### H6 — Ablation batch + report
- `run_batch.py` gets `--tag <name>` → writes `results/<tag>/batch_results.json`, and records
  all HARNESS_* flags + model ids in every row. Same 13 instances, same seed, as qwen-v1.
- Run order (sequential, highest value first; stop when the freeze time arrives):
  1. `qwen-v2-full` — every flag whose acceptance passed, ON.
  2. If time: single-flag ablations on a 6-instance subset.
- Report table per tag: resolved (fair, /12), agent-verified (repro+regression only),
  false positives, gold file in candidates, valid repro, clean apply, lint repairs, mean
  wall time, tokens. Note clearly that qwen-v1 predates H1 fixes.

### H7 — Meta API (as soon as the key arrives; can interleave with any step)
- Fill `.env` (never commit it). Smoke test `say OK` + one `complete_json`.
- If not OpenAI-compatible: ONE adapter class in llm.py, same `complete()` signature.
- Run `meta-v2` with the same flags as `qwen-v2-full` on the same instances.
- Demo instance: run it live once (fills cache), record the video from the cached replay.

## Timeline (IST) and freeze rules
| Time | Work |
|---|---|
| 23:00–23:30 | H0 |
| 23:30–00:00 | H1 |
| 00:00–00:40 | H2 |
| 00:40–01:40 | H3 |
| 01:40–02:55 | H4 |
| 02:00 | Decision: H5 only if H0–H4 already done |
| ~03:00–04:15 | H6 `qwen-v2-full` batch (and/or H7 `meta-v2`); write README + demo meanwhile |
| **04:30** | **Code freeze.** Only flags with passing acceptance are ON. Merge `harness-v2` → `master` only if its batch ran without crashes; otherwise submit `v1-baseline` + fixes. |
| 05:30 | Submit. Last hour is buffer. |

If a step misses its timebox: leave its flag OFF, commit what exists, move on.

## Out of scope tonight
Fine-tuning, LLM-as-judge rerankers, new retrieval models, multi-agent debate, parallel runs,
any change to oracle.py grading logic, any prompt tuning aimed at one model's quirks.