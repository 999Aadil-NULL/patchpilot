Instances: 13  |  model: qwen2.5-coder:7b  |  max attempts: 3

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
