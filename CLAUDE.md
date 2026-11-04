# CLAUDE.md — PatchPilot

Entry for the **Meta Global AI Developer Hackathon 2026** (Indie track, solo).
**Submission deadline: Thu Nov 5 2026, 06:30 IST. Code freeze: 04:30 IST.**
Reliability beats features. An honest result beats an impressive-looking one.

---

## 1. Vision

PatchPilot is an autonomous software-engineering agent for real GitHub issues (SWE-bench Lite):

```
plan → localize → write failing repro → patch → verify in Docker → self-correct (≤3)
     → independent oracle grading → human approval (Streamlit)
```

**The thesis.** A SWE-bench score measures *model + harness*, never the model alone. The
harness is our contribution. We want to show that **structural harness engineering** —
not prompt tricks tuned to one model — makes a small local model (qwen2.5-coder:7b on an
8 GB laptop GPU) measurably more capable, and that the same harness carries over unchanged
to Meta's hosted models.

**What makes PatchPilot different from a demo agent**
1. **Integrity by construction.** The agent never sees the gold patch, the official tests,
   or the FAIL_TO_PASS / PASS_TO_PASS lists. Grading is done by a separate oracle.
2. **Self-verification is not trusted.** The oracle already caught 3 false positives that the
   agent's own verification passed. We report both numbers side by side.
3. **Model-agnostic.** Every provider goes through one OpenAI-compatible client; switching
   models is a `.env` change.
4. **Measured, not claimed.** Every harness technique is a feature flag, so each one can be
   ablated on the same instances.

**What we will claim in the submission** — only what the ablation table supports. With
~12 instances, results are diagnostic evidence, never a benchmark score.

---

## 2. Working protocol (Claude Code: read this every time)

1. Do exactly the step(s) named in the prompt, in order. Nothing else.
2. For each step: implement → run its acceptance check → if it passes, **commit** with message
   `H<n>: <short summary>` and **update §6 Status in this file in the same commit**.
3. If an acceptance check fails or something is ambiguous: stop, do not commit that step,
   report what happened and what you propose.
4. Report at the end: what changed, acceptance results, anything surprising, anything
   left undone. Be honest about what was not exercised (e.g. "not run live").
5. Do not edit sections 1, 2, 3 or 7 of this file unless the prompt asks. §6 is yours to
   keep current; §8 (gotchas) you may append to.

**Code rules**
- Never refactor another module's public interface. Cross-module data goes through
  `agent/schemas.py`. Additive changes are allowed: new OPTIONAL fields / kwargs with defaults.
  Code that reads saved runs must tolerate those fields being absent.
- Every external call (LLM, docker exec, subprocess) has a timeout. Every stage emits events.
- No LangChain / LlamaIndex. No function-calling APIs. Models return text/JSON we parse.
- No prompt wording tuned to one model's quirks. Techniques must make sense for any model.
- Simple, explicit code. LF line endings. Never commit `.env`.

---

## 3. Integrity rule (non-negotiable)

The agent pipeline must NEVER read `patch`, `test_patch`, `FAIL_TO_PASS` or `PASS_TO_PASS`.
- Only `agent/oracle.py` may use them, for grading.
- `scripts/run_batch.py` may read the gold patch only AFTER a run, to score localization.
- Any candidate selection, reranking or stopping decision uses only agent-visible signals:
  repro result, regression result, lint result, diff shape.

---

## 4. Environment and how to run

- WSL2 Ubuntu, Python 3.11, `uv` (`uv add <pkg>`, `uv run ...`). Project on the Linux FS.
- Machine: ~7.8 GB WSL RAM, RTX 4060 8 GB. **Runs are strictly sequential.**
- **Local model: always start Ollama with `scripts/ollama.sh`** (sets 32,768 context). Plain
  `ollama serve` does not apply it. Model uses ~5.4 GB VRAM at 32k.
- Docker must be running; SWE-bench instance images are pulled per instance.

| Task | Command |
|---|---|
| Tests | `uv run pytest -q` (testpaths = `tests/`) |
| One instance | `uv run python -m scripts.run_one --instance <id>` |
| Dashboard | `uv run streamlit run ui/app.py` |
| Batch | `uv run python -m scripts.run_batch ...` (see `--help`; resumable, skips ids already in its results file) |
| Context audit | `uv run python -m scripts.audit_context` |

**Config (`.env`)**
```
PLANNER_BASE_URL=  PLANNER_API_KEY=  PLANNER_MODEL=
CODER_BASE_URL=    CODER_API_KEY=    CODER_MODEL=
LLM_CACHE=1             # disk cache keyed by model + messages (+ sample_idx once H4 lands)
LLM_TIMEOUT_S=120
MAX_ATTEMPTS=3
LLM_MAX_PROMPT_TOKENS=30000   # prompt-size warning threshold
HARNESS_LINT=0  HARNESS_FOCUS=0  HARNESS_SAMPLES=1  HARNESS_SBFL=0   # Phase 2 flags
```
Presets in `.env.example`: Ollama `http://localhost:11434/v1` (key `ollama`,
`qwen2.5-coder:7b`), Gemini and Groq OpenAI-compatible endpoints, Meta `<fill>`. If Meta's
API is not OpenAI-compatible: ONE adapter class in `llm.py` with the same `complete()`
signature. Nothing else changes.

---

## 5. Architecture

**Pipeline (orchestrator state machine)**
```
PLAN → LOCALIZE → REPRODUCE → [PATCH → VERIFY → (REFLECT)]×≤MAX_ATTEMPTS → ORACLE → AWAITING_REVIEW
```
- REPRODUCE: standalone `repro_issue.py` that must fail on the base commit with an
  AssertionError or the issue's exception. Invalid: ImportError, ModuleNotFoundError,
  SyntaxError, IndentationError, NameError, ImproperlyConfigured, or no traceback. ≤2 rewrites,
  else `verification_mode = "regression_only"`.
- PATCH: fresh tree (`sandbox.reset()`) every attempt; SEARCH/REPLACE edits.
- VERIFY: repro passes AND regression passes (≤3 related test files; only NEW failures vs
  base count). Regression-only runs are shown as "PASS (regression only)" and are NOT counted
  as agent-verified.
- ORACLE: fresh container, official test patch, swebench grading. Also run on the best
  failed attempt, for the record.
- Any exception → status `failed` with a reason. The UI never crashes.

**Modules** (`agent/schemas.py` is the single source of truth for data contracts)

| Module | Role |
|---|---|
| `schemas.py` | Pydantic contracts: Issue, Plan, CodeChunk, ContextBundle, EditBlock, ExecResult, TestResult, Attempt, OracleResult, RunState |
| `config.py` | `.env` loading, role configs, harness flags |
| `llm.py` | `complete(role, messages, *, temperature, max_tokens)`, `complete_json(...)`, retry, disk cache, token/latency events |
| `events.py` | `emit()` → `runs/<run_id>/events.jsonl`; orchestrator also writes `state.json` |
| `dataset.py` | SWE-bench Lite loader; `get_issue` strips hidden fields; `get_raw` is oracle-only |
| `sandbox.py` | Long-lived Docker container on the instance image (`/testbed`, conda `testbed`), network off, 6g |
| `localize/` | repo map (tree-sitter) → BM25 top-40 → LLM picks ≤5 files → FastEmbed + in-memory Qdrant top-8 chunks → ≤3-turn read_symbol/grep loop |
| `patcher.py` | Parse + apply SEARCH/REPLACE: exact → whitespace-normalized → fuzzy (ratio ≥ 0.92) → error |
| `prompts.py` | All prompt templates |
| `orchestrator.py` | State machine above |
| `oracle.py` | The only consumer of hidden fields |
| `ui/app.py` | Streamlit: instance picker, live stage timeline, plan/chunks/repro/attempts, two badges (agent verification vs official grading), Approve → `final.patch` |
| `scripts/` | `select_instances`, `run_one`, `run_batch`, `audit_context`, `ollama.sh` |

---

## 6. Status (Claude Code keeps this current)

**v1 — complete** (tag `v1-baseline` at `1f5921c` on `master`)

Baseline batch `qwen-v1` (13 instances, qwen2.5-coder:7b, 59 min, archived in
`results/qwen-v1/` and `runs/archive/qwen-v1/`). Predates H1 fixes.

| Metric | Result |
|---|---|
| Resolved (official, fair denominator) | 0/12 (`psf__requests-2317` excluded: gold fails, needs live httpbin) |
| Agent-verified | 3/13 originally → 0/13 under H1 rules (all 3 were regression-only false positives) |
| Gold file in candidates | 9/13 |
| Valid repro | 8/13 |
| ≥1 clean patch apply | 10/13 |
| Crashes / timeouts | 0 |
| Grader sanity | gold patches resolve 12/13 |

Caveats: 4 of 13 instances were fixed (images already pulled), 9 seeded-random; some LLM
responses replayed from cache. Only historical resolve: `sympy__sympy-12481` on an earlier
Gemini run.

**Phase 2 — branch `harness-v2`**

| Step | State | Commit | Notes |
|---|---|---|---|
| H0 context audit | done | 4571f3f | No truncation in qwen-v1 (max prompt 26,925 = 82% of 32k) |
| H1 correctness fixes | done | 8d921c9, b30be0d | Badge, repro exception parsing, prompt-size guard. Ran live on django-11099 (repro+regression path). The regression-only path and a guard firing have only been tested offline. |
| H2 lint guardrail | done, flag default OFF | (this commit) | 44 tests pass. `py_compile` checked in Docker on all 5 repos (Python 3.6–3.11). Flag off replays qwen-v1 django-11099 identically (13/13 cached). Live flag on: caught a real broken edit, but the repair turn did not apply, so `lint_repaired` has only been seen in unit tests. |
| H3 narrow edit scope | todo | | |
| H4 sample K + select | todo | | |
| H5 repro-trace localization | stretch | | Only if H0–H4 done by 02:00 |
| H6 ablation batch | todo | | |
| H7 Meta API | waiting for key | | |

---

## 7. Phase 2 roadmap

Inspiration (structural techniques from published systems):

| Technique | From | Ours |
|---|---|---|
| Reject syntactically broken edits | SWE-agent interface design; Aider lint-after-edit | H2 |
| Hierarchical localization, narrow edit scope | Agentless | H3 |
| Sample many patches, select by execution + majority | Agentless | H4 |
| Fault localization from the failing test | AutoCodeRover | H5 |
| Repo map + SEARCH/REPLACE | Aider | v1 |

Every technique is a flag, **default OFF**. All flags off = v1 behaviour plus H1 fixes. A flag
is ON in the final config only if its acceptance check passed. Record all flags and model ids
in every run's state and in every batch row.

### H1 (remaining) — prompt-size guard
In `llm.py`, before each call estimate prompt tokens (chars / 3.5) against
`LLM_MAX_PROMPT_TOKENS` (default 30000). Over → emit `error` event `prompt_near_limit` with
role and estimate. Do not raise, do not truncate. `audit_context.py` reports these events.
Reason: Ollama truncates over-limit prompts silently.
**Acceptance:** unit test with an oversized fake prompt emits the event; normal calls emit
nothing; full test suite passes.

### H2 — Lint guardrail (`HARNESS_LINT`)
- `agent/guards.py`: `syntax_check(sandbox, files) -> list[str]` runs `python -m py_compile`
  in the testbed env per modified `.py` file (timeout 30 s).
- After `apply_edits`: on errors, ONE repair turn showing the compile error and ±10 lines
  around it; re-apply on a reset tree. Repair turns do not count toward MAX_ATTEMPTS. Still
  failing → record as apply_errors, continue as v1.
- Events: `lint_fail`, `lint_repaired`.
- **Acceptance:** test with a deliberately broken edit → caught and repair prompt built;
  flag off → v1 code path unchanged; suite passes.

### H3 — Narrow edit scope (`HARNESS_FOCUS`)
- After LOCALIZE: one `complete_json` call returns `{"locations":[{"file_path","symbol"}]}`,
  ≤3 entries, validated against the repo map (drop unknowns; none valid → v1 context).
- Patch prompt shows FULL source of those symbols (tree-sitter, from `cache/repos`) plus
  other candidate chunks as signatures only. No line numbers inside shown code.
- Optional field `ContextBundle.edit_locations: list[str] = []`.
- **Acceptance:** on 2 instances, PATCH prompt tokens drop vs v1 and locations are logged.
  (Gold-function hit is scored post hoc by run_batch only.)

### H4 — Sample K, select by execution (`HARNESS_SAMPLES`)
- `llm.complete` gains optional `sample_idx: int = 0`, included in the cache key only when
  non-zero (v1 cache keys unchanged).
- Per attempt K candidates: sample 0 at temperature 0.2, others at 0.7. Each: reset → apply →
  lint (if H2 on) → repro. Regression only for repro-passing candidates.
- `agent/select.py` `select(candidates) -> (index, reason)`:
  1. prefer repro + regression pass; 2. majority by normalized diff (whitespace, blank lines,
  comments stripped); 3. tie → smallest diff; 4. none pass → best failure (clean apply >
  lint clean > fewest new regression failures), whose feedback goes into the next attempt.
- Regression-only mode: regression + majority; badge stays "regression only".
- Optional `Attempt.candidates: list[dict] = []` (diff + result summary). UI shows
  "K candidates, selected #i, reason".
- **Acceptance:** K=3 on one instance end to end: 3 coder calls with distinct cache keys and a
  selection event with reason. K=1 → v1 behaviour. Suite passes.

### H5 — STRETCH: repro-trace localization (`HARNESS_SBFL`)
- `agent/localize/trace.py`: `sys.settrace` runner (no new testbed deps) recording
  `(file, function)` calls under `/testbed`, excluding tests and `repro_issue.py`.
- After a valid failing repro: add top-5 traced functions to context if absent. No valid
  repro → no-op.
- **Acceptance:** one instance with a valid repro gives a non-empty trace and logged refined chunks.

### H6 — Ablation batch and report
- `run_batch.py --tag <name>` → `results/<tag>/batch_results.json`; every row records flags +
  model ids. `--report` must tolerate rows missing newer fields.
- Same instances and seed as qwen-v1. Archive old run folders before reruns.
- Run order: `qwen-v2-full` (all passing flags ON), then single-flag ablations on a
  6-instance subset if time allows.
- Report per tag: resolved (/12), agent-verified (repro+regression only), false positives,
  gold file in candidates, valid repro, clean apply, lint repairs, mean wall time, tokens.
  State that qwen-v1 predates H1.

### H7 — Meta API
- Fill `.env` (never commit). Smoke test plain completion + one `complete_json`.
- Adapter only if not OpenAI-compatible.
- `meta-v2` batch with the same flags as `qwen-v2-full`, same instances; watch credits (K
  multiplies calls).
- Demo: run the chosen instance live once (fills cache), record from the cached replay, say so.

### Submission checklist
README (vision, architecture, how to run, ablation table with caveats, integrity design,
future work: larger-sample ablation), demo video, `harness-v2` merged to `master` only if its
batch ran crash-free (else submit `v1-baseline` + H1).

### Timeline (IST)
| Time | Work |
|---|---|
| now → 00:30 | H1 remainder, H2 |
| 00:30 → 01:30 | H3 |
| 01:30 → 02:45 | H4 (H5 only if ahead) |
| 02:45 → 04:15 | H6 batch (+ H7 if key available); README + demo in parallel |
| **04:30** | **Freeze** |
| 05:30 | Submit; final hour is buffer |
A step that overruns its slot: flag stays OFF, commit what works, move on.

---

## 8. Gotchas and decisions (append-only)
- Ollama truncates over-limit prompts silently (~42k test prompt cut to 16,386). Never lower
  `num_ctx` below 32k; planner prompts reach ~27k.
- `pytest` must use `testpaths = ["tests"]`; otherwise it collects `cache/repos/` (2,755 errors).
- Repro exception parsing: last traceback only; compile failures fall back to the last
  `File "..."` line; a printed exception name without a traceback does not count.
- Saved runs from before H1 lack `verification_mode`; no repro script ⇒ treat as regression-only.
- Moving `runs/batch-*` to the archive hides them from the dashboard; copy a folder back to view it.
- `run_batch.py` skips ids already in its results file — archive before rerunning.
- Models sometimes put ```` ```python ```` fence lines INSIDE SEARCH/REPLACE (3/40 edit blocks in qwen-v1, all
  django-11099). The fuzzy tier still matches SEARCH, so the fence is written into the source file. H2 catches
  it as a SyntaxError; `patcher.py` does not strip it yet.

## 9. Out of scope tonight
Fine-tuning, LLM-as-judge rerankers, new retrieval models, multi-agent debate, parallel runs,
changes to oracle grading logic, prompt tuning aimed at one model.