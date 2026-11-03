"""Filter SWE-bench Lite → data/candidates.json.

uv run python -m scripts.select_instances

Offline script: reads the gold patch only to *select* instances; the output file carries no
hidden fields (just id/repo/version/title), so it is safe for the UI/agent to read.
"""
import json
import re

from agent.config import DATA_DIR
from agent.dataset import _load

REPOS = {"sympy/sympy", "psf/requests", "pytest-dev/pytest", "pallets/flask", "django/django"}
MAX_CHANGED = 15


def patch_stats(patch: str) -> tuple[int, int]:
    files = set(re.findall(r"^diff --git a/(\S+) b/", patch, re.M))
    changed = sum(
        1 for ln in patch.splitlines()
        if (ln.startswith("+") and not ln.startswith("+++")) or (ln.startswith("-") and not ln.startswith("---"))
    )
    return len(files), changed


def main() -> None:
    out = []
    for r in _load().values():
        if r["repo"] not in REPOS:
            continue
        nfiles, changed = patch_stats(r["patch"])
        if nfiles == 1 and changed < MAX_CHANGED:
            title = (r["problem_statement"].strip().splitlines() or [""])[0][:120]
            out.append({"instance_id": r["instance_id"], "repo": r["repo"],
                        "version": str(r["version"]), "title": title})
    out.sort(key=lambda x: (x["repo"], x["instance_id"]))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "candidates.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    by = {}
    for x in out:
        by[x["repo"]] = by.get(x["repo"], 0) + 1
    print(f"{len(out)} candidates -> data/candidates.json  {by}")


if __name__ == "__main__":
    main()
