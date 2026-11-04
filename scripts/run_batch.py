"""Step 7: batch run → results/batch_results.json + results/batch_results.md

uv run python -m scripts.run_batch --select            # choose instances, print them, write data/batch_ids.json
uv run python -m scripts.run_batch                     # run them sequentially (resumable)
uv run python -m scripts.run_batch --report            # rebuild the report only

Selection is a seeded stratified draw from data/candidates.json made without looking at gold patches.
Each instance runs in its own subprocess: an OOM kill or hang costs one instance, not the batch.
The gold patch is read here ONLY to score localization afterwards; it never reaches the agent.
"""
import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import docker

from agent.config import DATA_DIR, ROOT, RUNS_DIR
from agent.dataset import get_raw
from agent.schemas import RunState

IDS_PATH = DATA_DIR / "batch_ids.json"
OUT_DIR = ROOT / "results"
OUT_JSON = OUT_DIR / "batch_results.json"
OUT_MD = OUT_DIR / "batch_results.md"
SEED = 7
QUOTA = {"django/django": 4, "sympy/sympy": 4, "pytest-dev/pytest": 3, "psf/requests": 1, "pallets/flask": 1}
ALREADY_PULLED = ["django__django-11099", "sympy__sympy-12481", "pytest-dev__pytest-7432", "psf__requests-2317"]
INSTANCE_TIMEOUT_S = 40 * 60


def select() -> list[str]:
    cands = json.loads((DATA_DIR / "candidates.json").read_text())
    rng = random.Random(SEED)
    ids = list(ALREADY_PULLED)
    for repo, quota in QUOTA.items():
        pool = sorted(c["instance_id"] for c in cands if c["repo"] == repo and c["instance_id"] not in ids)
        have = sum(1 for c in cands if c["repo"] == repo and c["instance_id"] in ids)
        ids += rng.sample(pool, quota - have)
    IDS_PATH.write_text(json.dumps(ids, indent=1))
    return ids


def cleanup_containers(run_id: str) -> None:
    client = docker.from_env(timeout=60)
    for c in client.containers.list(all=True, filters={"label": f"run_id={run_id}"}):
        c.remove(force=True)


def run_instance(iid: str) -> dict:
    run_id = f"batch-{iid}"
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    note = ""
    with open(d / "stdout.log", "w") as log:
        p = subprocess.Popen([sys.executable, "-m", "scripts.run_one", "--instance", iid, "--run-id", run_id],
                             cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (d / "pid").write_text(str(p.pid))
        try:
            rc = p.wait(timeout=INSTANCE_TIMEOUT_S)
            if rc < 0:
                note = f"process killed by signal {-rc}"
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            note = f"timed out after {INSTANCE_TIMEOUT_S}s"
    cleanup_containers(run_id)
    return summarize(iid, run_id, time.time() - t0, note)


def summarize(iid: str, run_id: str, duration: float, note: str) -> dict:
    gold = set(re.findall(r"^diff --git a/(\S+) b/", get_raw(iid)["patch"], re.M))
    row = {"instance_id": iid, "repo": iid.split("__")[0], "run_id": run_id, "duration_s": round(duration),
           "note": note, "status": "crashed", "verified": False, "regression_only_pass": False,
           "resolved": False, "oracle_ran": False,
           "gold_file_found": False, "repro_valid": False, "attempts": 0, "attempts_applied": 0,
           "tokens_in": 0, "tokens_out": 0, "failure_reason": note}
    p = RUNS_DIR / run_id / "state.json"
    if not p.exists():
        return row
    s = RunState.model_validate_json(p.read_text(encoding="utf-8"))
    row.update(
        status=s.status, verified=s.status == "awaiting_review" and not s.regression_only,
        regression_only_pass=s.status == "awaiting_review" and s.regression_only,
        resolved=bool(s.oracle and s.oracle.resolved), oracle_ran=s.oracle is not None,
        gold_file_found=bool(s.context and gold & set(s.context.candidate_files)),
        repro_valid=bool(s.repro_script), attempts=len(s.attempts),
        attempts_applied=sum(1 for a in s.attempts if a.diff.strip() and not a.apply_errors),
        tokens_in=s.tokens_in, tokens_out=s.tokens_out, failure_reason=s.failure_reason or note,
    )
    return row


def report(rows: list[dict]) -> str:
    n = len(rows)
    c = lambda k: sum(1 for r in rows if r[k])  # noqa: E731
    pct = lambda a: f"{a}/{n} ({100 * a / n:.0f}%)" if n else "0/0"  # noqa: E731
    both = sum(1 for r in rows if r["verified"] and r["resolved"])
    false_pos = sum(1 for r in rows if r["verified"] and not r["resolved"])
    false_neg = sum(1 for r in rows if r["resolved"] and not r["verified"])
    lines = [
        f"Instances: {n}  |  model: {os.getenv('CODER_MODEL', '?')}  |  max attempts: {os.getenv('MAX_ATTEMPTS', '3')}",
        "",
        "| Metric | Result |", "|---|---|",
        f"| **Resolved (official SWE-bench grading)** | **{pct(c('resolved'))}** |",
        f"| Passed agent verification (repro + regression) | {pct(c('verified'))} |",
        f"| Passed regression only (not counted as verified) | {pct(c('regression_only_pass'))} |",
        f"| Verified and resolved | {pct(both)} |",
        f"| Verified but NOT resolved (false positive) | {false_pos} |",
        f"| Resolved but not verified (false negative) | {false_neg} |",
        f"| Gold file in candidate files | {pct(c('gold_file_found'))} |",
        f"| Valid reproduction script | {pct(c('repro_valid'))} |",
        f"| At least one patch applied cleanly | {pct(sum(1 for r in rows if r['attempts_applied']))} |",
        f"| Crashed / timed out | {sum(1 for r in rows if r['status'] == 'crashed')} |",
        "",
        "| Instance | Localized | Repro | Applied | Verified | Resolved | Time |", "|---|---|---|---|---|---|---|",
    ]
    yn = lambda b: "yes" if b else "no"  # noqa: E731
    for r in rows:
        lines.append(f"| {r['instance_id']} | {yn(r['gold_file_found'])} | {yn(r['repro_valid'])} | "
                     f"{r['attempts_applied']}/{r['attempts']} | {yn(r['verified'])} | "
                     f"{yn(r['resolved']) if r['oracle_ran'] else 'not graded'} | {r['duration_s']}s |")
    return "\n".join(lines) + "\n"


def save(rows: list[dict]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, indent=1))
    OUT_MD.write_text(report(rows))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.select:
        print("\n".join(select()))
        return 0
    rows = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else []
    if args.report:
        save(rows)
        print(report(rows))
        return 0
    ids = json.loads(IDS_PATH.read_text()) if IDS_PATH.exists() else select()
    done = {r["instance_id"] for r in rows}
    for i, iid in enumerate(ids, 1):
        if iid in done:
            continue
        print(f"[{i}/{len(ids)}] {iid}", flush=True)
        row = run_instance(iid)
        rows.append(row)
        save(rows)  # after every instance, so a crash keeps partial results
        print(f"    status={row['status']} verified={row['verified']} resolved={row['resolved']} "
              f"{row['duration_s']}s {row['note']}", flush=True)
    print(report(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
