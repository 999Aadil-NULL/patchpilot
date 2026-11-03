"""Step-3 acceptance check: uv run python -m scripts.check_step3 [ids...]

For each instance: export repo (once) → plan (planner) → localize → is the gold patch's file in
candidate_files? Gold file is read here ONLY to score; it is never shown to the agent.
"""
import re
import sys
import time

from agent import llm, prompts
from agent.config import CACHE_DIR
from agent.dataset import get_issue, get_raw
from agent.localize.retrieve import localize
from agent.sandbox import Sandbox
from agent.schemas import Plan

DEFAULT = ["sympy__sympy-12481", "pytest-dev__pytest-7432", "django__django-11099"]


def repo_dir_for(iid: str):
    dest = CACHE_DIR / "repos" / iid
    if not (dest / ".pp_export_ok").exists():
        with Sandbox(iid, run_id="export") as sb:
            sb.export_repo(dest)
    return dest


def main() -> int:
    ids = sys.argv[1:] or DEFAULT
    hits = 0
    for iid in ids:
        t0 = time.time()
        issue = get_issue(iid)
        repo = repo_dir_for(iid)
        run_id = f"_check3_{iid}"
        llm.set_run_context(run_id, "planning")
        plan = llm.complete_json("planner", [
            {"role": "system", "content": prompts.PLAN_SYSTEM},
            {"role": "user", "content": prompts.plan_user(issue)},
        ], Plan)
        llm.set_run_context(run_id, "localizing")
        bundle = localize(issue, plan, repo, run_id=run_id)
        gold = set(re.findall(r"^diff --git a/(\S+) b/", get_raw(iid)["patch"], re.M))
        hit = bool(gold & set(bundle.candidate_files))
        rank = next((i + 1 for i, f in enumerate(bundle.candidate_files) if f in gold), None)
        chunk_hit = any(c.file_path in gold for c in bundle.chunks)
        hits += hit
        print(f"[{'HIT ' if hit else 'MISS'}] {iid}  gold={sorted(gold)} rank={rank} "
              f"chunks_in_gold={chunk_hit}  ({time.time() - t0:.0f}s)")
        print(f"       candidates={bundle.candidate_files}")
        short = [c.file_path.split("/")[-1] + ":" + c.symbol for c in bundle.chunks]
        print(f"       chunks={short}")
    print(f"\nfile hit rate: {hits}/{len(ids)}  usage={llm.get_usage()}")
    return 0 if hits == len(ids) else 1


if __name__ == "__main__":
    sys.exit(main())
