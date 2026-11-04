# PatchPilot

**An autonomous software-engineering agent that fixes real GitHub issues, checked by grading it cannot cheat.**
Entry for the Meta Global AI Developer Hackathon 2026 (Indie track).

PatchPilot takes an issue from [SWE-bench Lite](https://www.swebench.com/), plans a fix, finds the relevant code, writes a script that reproduces the bug, patches the repository, checks the patch in Docker, corrects itself, and hands the result to a person for approval. A separate oracle grades every patch against the official SWE-bench tests. The agent never sees those tests.

## Why this project

A SWE-bench score measures a **model and its harness together**, never the model alone. Handed only the issue text, a raw LLM cannot find the right file in a large repository, cannot run code, and cannot find out that its patch broke something. The harness supplies all of that, so the harness is this project's contribution.

The thesis: **structural harness engineering**, rather than prompt tricks tuned to one model, can make a small local model (qwen2.5-coder 7B on an 8 GB laptop GPU) measurably more capable. The same harness then runs unchanged on any stronger model.

## What makes it different

1. **Integrity by construction.** The agent pipeline never reads the gold patch, the official test patch or the FAIL_TO_PASS / PASS_TO_PASS lists. Only `agent/oracle.py` touches them, in a fresh container.
2. **Self-verification is not trusted.** The agent's own "it passes" is reported next to the official grade. In the baseline run, the oracle caught 3 patches the agent believed were correct. All of them were *regression-only* passes, where no valid reproduction existed. The dashboard now labels those runs honestly instead of showing a green PASS.
3. **Model-agnostic.** Every provider is reached through one OpenAI-compatible client. Switching models is a `.env` change, or one adapter class if a provider isn't OpenAI-compatible.
4. **Measured, not claimed.** Each harness technique sits behind a feature flag, so each one can be turned off and compared on the same instances.

## Architecture

```
PLAN → LOCALIZE → REPRODUCE → [PATCH → LINT → VERIFY → REFLECT] ×≤3 → ORACLE → HUMAN REVIEW
```

| Stage | What it does |
|---|---|
| Plan | Turns the issue into a hypothesis, steps, acceptance criteria and search terms (JSON) |
| Localize | tree-sitter repo map → BM25 over 40 files → the LLM picks ≤5 files → FastEmbed + in-memory Qdrant rank code chunks → short read_symbol / grep exploration loop |
| Reproduce | Writes `repro_issue.py`, which must fail on the base commit with the issue's real error. Import, syntax and configuration errors are rejected as invalid. |
| Patch | SEARCH/REPLACE edits on a fresh tree each attempt, applied exact → whitespace-normalized → fuzzy |
| Verify | The repro must pass and related existing tests must not newly fail, in the instance's Docker image with networking disabled |
| Reflect | Failure output feeds the next attempt (up to 3) |
| Oracle | Fresh container, official test patch, official SWE-bench grading |
| Review | Streamlit dashboard: live stage timeline, plan, retrieved code, repro, every attempt's diff and test output, the two badges side by side, and an Approve button |

### Harness techniques (Phase 2, all feature-flagged)

Each idea is borrowed from a published system and kept structural, so it applies to any model.

| Flag | Technique | Inspired by |
|---|---|---|
| `HARNESS_LINT` | Compile-check every edited file; one cheap repair turn that doesn't use up an attempt | SWE-agent's agent–computer interface; Aider's lint-after-edit |
| `HARNESS_FOCUS` | Model picks ≤3 edit locations first; the patch prompt shows only those full functions | Agentless hierarchical localization |
| `HARNESS_SAMPLES` | K candidate patches per attempt; selected by repro + regression results, then majority over normalized diffs | Agentless |
| `HARNESS_SBFL` | Trace which repository functions the failing repro executes, and add them to the context | AutoCodeRover fault localization |

Candidate selection uses only signals the agent is allowed to see. The oracle is never consulted.

## Results

All results are on **qwen2.5-coder:7b running locally** (Ollama, 32k context, RTX 4060 8 GB), on the same 13 SWE-bench Lite instances from Django, SymPy, Requests, pytest and Flask. Instances are named by project and the pull request that fixed them (for example Django PR 11099), with the SWE-bench ID in brackets.

The **fair denominator is 12**: Requests PR 2317 (`psf__requests-2317`) is excluded because even the gold patch fails there, since its tests need live network access. The grader itself was validated first: the gold patches resolve 12/13 instances.

| Metric | v1 baseline | v2 (all flags) |
|---|---|---|
| Resolved (official grading, /12) | 0 | <!-- FILL --> |
| Agent-verified (repro + regression) | 0 † | <!-- FILL --> |
| Regression-only passes (false positives) | 3 | <!-- FILL --> |
| Gold file among candidates | 9/13 | <!-- FILL --> |
| Valid reproduction | 8/13 | <!-- FILL --> |
| At least one clean patch apply | 10/13 | <!-- FILL --> |
| Instances completed before time limit | 13/13 | <!-- FILL --> |
| Crashes | 0 | <!-- FILL --> |
| Mean wall time per instance | ~4.5 min | <!-- FILL --> |

† The v1 run originally reported 3 agent-verified patches. Under the stricter rule they are reclassified as regression-only, and the official grader rejected all three.

**Highlight:** Django PR 11099 (`django__django-11099`) is resolved under official grading on the 7B model. This is the first resolve on this model.

**How to read this honestly**
- 12 instances is a diagnostic sample, not a benchmark score: one resolve moves the rate by about 8 points.
- v1 numbers predate the correctness fixes in Phase 2 (repro-validity parsing and the regression-only badge).
- 4 of the 13 instances were chosen because their Docker images were already pulled; the other 9 were a seeded random draw.
- Per-flag ablations were limited by time. <!-- FILL: what was / wasn't ablated -->

## Model choice

Development and every reported result use a local open model, so the whole system runs offline on a laptop. Access to the Meta Model API did not arrive before the deadline. <!-- FILL if a Llama model was run: describe it here --> Because every model is reached through the same client, Meta's models drop in through `.env`, with one adapter class in `agent/llm.py` if the API isn't OpenAI-compatible. No other code changes.

## Run it

Requirements: WSL2 or Linux, Python 3.11, [uv](https://docs.astral.sh/uv/), Docker, and either a local Ollama install or any OpenAI-compatible API key.

```bash
uv sync
cp .env.example .env              # fill planner/coder base URL, key, model
scripts/ollama.sh                 # local model only: starts Ollama with a 32k context
uv run pytest -q                  # unit tests
uv run python -m scripts.run_one --instance django__django-11099
uv run streamlit run ui/app.py    # dashboard
uv run python -m scripts.run_batch --help
```

`LLM_CACHE=1` caches every model response on disk, so a finished run can be replayed exactly. The demo video uses a replay of a real run.

## Limitations

- Patch quality is the bottleneck for the 7B model. Localization and reproduction succeed far more often than patches resolve.
- A reproduction can count as valid without exercising the actual bug site. This is detected in logs but not yet prevented.
- K-sample selection multiplies runtime (up to ~21 min on one instance locally).
- Runs are sequential because of local memory limits.

## Future work

- Per-technique ablation on 50+ instances, to estimate the gain from each part of the harness with real confidence intervals.
- Check that each reproduction actually reaches the localized code before accepting it.
- The same ablation on Meta's hosted models.

## Repository layout

```
agent/        schemas, llm client, sandbox, localize/, patcher, guards, select, orchestrator, oracle
scripts/      run_one, run_batch, select_instances, audit_context, ollama.sh
ui/app.py     Streamlit dashboard
tests/        unit tests
results/      batch results per tag (qwen-v1, qwen-v2-full, ...)
```