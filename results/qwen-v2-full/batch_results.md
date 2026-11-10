Instances: 5  |  model: qwen2.5-coder:7b  |  max attempts: 3
Flags: {"HARNESS_LINT": true, "HARNESS_FOCUS": true, "HARNESS_SAMPLES": 3, "HARNESS_SBFL": true, "planner_model": "qwen2.5-coder:7b", "coder_model": "qwen2.5-coder:7b"}

| Metric | Result |
|---|---|
| **Resolved (official grading, requests-2317 excluded)** | **1/4** |
| Passed agent verification (repro + regression) | 0/5 (0%) |
| Passed regression only (not counted as verified) | 1/5 (20%) |
| Passed either check but NOT resolved (false positive) | 1 |
| Resolved but not verified (false negative) | 1 |
| Gold file in candidate files | 5/5 (100%) |
| Valid reproduction script | 4/5 (80%) |
| At least one patch applied cleanly | 5/5 (100%) |
| Lint repairs | 0 |
| Crashed / timed out | 0 |
| Mean wall time | 422s |
| Tokens (in + out) | 190,642 |

| Instance | Localized | Repro | Applied | Verified | Resolved | Time |
|---|---|---|---|---|---|---|
| django__django-11099 | yes | yes | 3/3 | no | yes | 629s |
| sympy__sympy-12481 | yes | yes | 3/3 | no | no | 662s |
| pytest-dev__pytest-7432 | yes | yes | 3/3 | no | no | 267s |
| psf__requests-2317 | yes | yes | 3/3 | no | no | 470s |
| django__django-13551 | yes | no | 1/1 | no | no | 81s |
