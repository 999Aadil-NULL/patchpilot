# PatchPilot

An autonomous software engineering agent. Given a SWE-bench Lite issue, it plans a fix, finds the relevant code,
writes a failing reproduction, patches the code, verifies the patch in Docker and corrects itself up to 3 times.
The patch is then graded independently with the official SWE-bench tests and waits for human approval.

```
PLAN → LOCALIZE → REPRODUCE → [PATCH → VERIFY → REFLECT] ×3 → ORACLE → HUMAN REVIEW
```

**Integrity:** the agent never sees the gold patch, the official tests or the test lists from the dataset.
Only `agent/oracle.py` reads them, and its result is never fed back to the agent.

## Setup

Requires WSL2 or Linux, Docker, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env                       # set PLANNER_* and CODER_* for your provider
uv run python -m scripts.select_instances  # builds data/candidates.json
```

The model is swappable: any OpenAI-compatible endpoint works through `.env`. For local development:

```bash
scripts/ollama.sh pull    # qwen2.5-coder:7b
scripts/ollama.sh         # serves on :11434 with a 32k context
```

## Usage

```bash
uv run python -m scripts.run_one --instance sympy__sympy-12481   # one issue from the CLI
uv run streamlit run ui/app.py                                   # dashboard: run, watch, approve
uv run python -m scripts.run_batch                               # batch; writes results/
uv run pytest tests                                              # unit tests
```

## Results

Batch run on 2026-11-04 with **qwen2.5-coder:7b**, a small local model used for development. These numbers
describe that model, not the final demo model, and will be re-measured when the final model is plugged in.

The 13 instances are 4 django, 4 sympy, 3 pytest, 1 requests and 1 flask. Four were kept because their images
were already downloaded; the other nine are a seeded random draw per repository, made without looking at gold
patches. All candidates have a gold patch touching 1 file and fewer than 15 lines.

| Metric | Result |
|---|---|
| **Resolved (official SWE-bench grading)** | **0/13 (0%)** |
| Passed agent verification | 3/13 (23%) |
| Verified and resolved | 0/13 (0%) |
| Verified but NOT resolved (false positive) | 3 |
| Resolved but not verified (false negative) | 0 |
| Gold file in candidate files | 9/13 (69%) |
| Valid reproduction script | 8/13 (62%) |
| At least one patch applied cleanly | 10/13 (77%) |
| Crashed / timed out | 0 |

| Instance | Localized | Repro | Applied | Verified | Resolved | Time |
|---|---|---|---|---|---|---|
| django__django-11099 | yes | yes | 3/3 | no | no | 166s |
| sympy__sympy-12481 | yes | yes | 1/3 | no | no | 793s |
| pytest-dev__pytest-7432 | yes | yes | 2/3 | no | no | 203s |
| psf__requests-2317 | yes | yes | 2/3 | no | no | 340s |
| django__django-13551 | yes | no | 1/1 | yes | no | 108s |
| django__django-12184 | no | yes | 0/3 | no | not graded | 374s |
| django__django-14016 | yes | yes | 3/3 | no | no | 161s |
| sympy__sympy-18698 | yes | no | 2/2 | yes | no | 469s |
| sympy__sympy-12171 | yes | yes | 3/3 | no | no | 223s |
| sympy__sympy-12236 | no | no | 0/3 | no | not graded | 91s |
| pytest-dev__pytest-7220 | no | no | 1/1 | yes | no | 119s |
| pytest-dev__pytest-11148 | no | no | 0/3 | no | not graded | 128s |
| pallets__flask-4992 | yes | yes | 3/3 | no | no | 385s |

### How to read these numbers

- **Grading is sound.** The gold patches for 12 of the 13 instances grade as resolved
  (`results/oracle_gold_check.json`), so 0/13 reflects the model and not a broken grader.
- **`psf__requests-2317` cannot be resolved in this environment.** Its gold patch also fails, on tests that
  appear to need the live httpbin service. Excluding it, the result is 0/12.
- **All 3 "verified" results are false positives, and none had a valid reproduction.** Without a reproduction,
  verification only shows that existing tests still pass, not that the issue is fixed.
- **The pipeline can resolve issues with a stronger model.** In an earlier run, `sympy__sympy-12481` was resolved
  with Gemini 3.5 Flash as coder. That was one instance, and the agent's own verifier rejected the patch at the
  time; the verifier was fixed afterwards.
- **Sample size:** 13 instances is small, so treat the percentages as rough.
- **Replays:** LLM responses are cached, so `sympy__sympy-12481` and `psf__requests-2317` partly replayed earlier runs.
- **Memory:** Ollama's model process was killed twice for lack of memory during the batch. No run recorded an LLM
  error, because the retries covered it, but some calls took several minutes.
