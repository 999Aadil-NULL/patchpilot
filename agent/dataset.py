"""SWE-bench Lite loader.

INTEGRITY: the agent pipeline may only use `get_issue()`, `get_image()` and `get_test_cmd()`.
`get_raw()` returns hidden fields (patch, test_patch, FAIL_TO_PASS, PASS_TO_PASS, eval_script)
and must ONLY be called from agent/oracle.py (and offline scripts that never feed the agent).

We use `SWE-bench/SWE-bench_Lite` (same 300 instances as princeton-nlp/SWE-bench_Lite) because
swebench>=5 expects the per-instance `image`, `eval_script`, `log_parser`, `eval_type` columns
that only this copy carries.
"""
from __future__ import annotations

import json
import shlex
from collections import defaultdict
from functools import lru_cache

from agent.config import DATA_DIR
from agent.schemas import Issue

DATASET_NAME = "SWE-bench/SWE-bench_Lite"
LITE_PATH = DATA_DIR / "lite.json"
TEST_CMDS_PATH = DATA_DIR / "test_cmds.json"

HIDDEN_FIELDS = {"patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "eval_script"}

_START = ": '>>>>> Start Test Output'"
_END = ": '>>>>> End Test Output'"


@lru_cache(maxsize=1)
def _load() -> dict[str, dict]:
    if not LITE_PATH.exists():
        from datasets import load_dataset  # heavy import, only on first run

        ds = load_dataset(DATASET_NAME, split="test")
        rows = [dict(r) for r in ds]
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = LITE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows), encoding="utf-8")
        tmp.replace(LITE_PATH)
    rows = json.loads(LITE_PATH.read_text(encoding="utf-8"))
    return {r["instance_id"]: r for r in rows}


def all_ids() -> list[str]:
    return list(_load().keys())


def _row(instance_id: str) -> dict:
    try:
        return _load()[instance_id]
    except KeyError:
        raise KeyError(f"unknown SWE-bench Lite instance: {instance_id}") from None


def get_issue(instance_id: str) -> Issue:
    """Agent-safe view: hidden fields stripped."""
    r = _row(instance_id)
    return Issue(
        instance_id=r["instance_id"], repo=r["repo"], base_commit=r["base_commit"],
        version=str(r["version"]), problem_statement=r["problem_statement"],
        hints_text=r.get("hints_text") or "",
    )


def get_image(instance_id: str) -> str:
    """Docker image for the instance (not hidden: it's just the environment)."""
    return _row(instance_id)["image"]


def get_raw(instance_id: str) -> dict:
    """FULL row including hidden fields. ORACLE ONLY."""
    return dict(_row(instance_id))


# ------------------------------------------------------------------ repo-level test command
def _test_line(eval_script: str) -> str:
    seg = eval_script.split(_START + "\n", 1)[1].split("\n" + _END, 1)[0]
    return seg.strip()


@lru_cache(maxsize=1)
def _test_cmd_prefixes() -> dict[str, str]:
    """Per repo: longest common token prefix of the official test command across ALL Lite
    instances of that repo. Instance-specific test files are dropped, so nothing hidden leaks."""
    if TEST_CMDS_PATH.exists():
        return json.loads(TEST_CMDS_PATH.read_text(encoding="utf-8"))
    by_repo: dict[str, list[list[str]]] = defaultdict(list)
    for r in _load().values():
        by_repo[r["repo"]].append(shlex.split(_test_line(r["eval_script"])))
    out: dict[str, str] = {}
    for repo, cmds in by_repo.items():
        prefix = cmds[0]
        for toks in cmds[1:]:
            n = 0
            while n < min(len(prefix), len(toks)) and prefix[n] == toks[n]:
                n += 1
            prefix = prefix[:n]
        # never end on a path-looking token (single-instance repos)
        while prefix and ("/" in prefix[-1] or prefix[-1].endswith(".py")) and not prefix[-1].startswith("-") \
                and "=" not in prefix[-1]:
            prefix = prefix[:-1]
        out[repo] = shlex.join(prefix)
    TEST_CMDS_PATH.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def get_test_cmd(repo: str) -> str:
    """Generic test command prefix for a repo, e.g. 'pytest -rA'. Append test files/labels."""
    try:
        return _test_cmd_prefixes()[repo]
    except KeyError:
        raise KeyError(f"no test command known for repo {repo}") from None
