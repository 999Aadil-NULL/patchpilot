"""Step 7: batch run → results/batch_results.json + results/batch_results.md

uv run python -m scripts.run_batch --select            # choose instances, print them, write data/batch_ids.json
uv run python -m scripts.run_batch                     # run them sequentially (resumable)
uv run python -m scripts.run_batch --report            # rebuild the report only
uv run python -m scripts.run_batch --tag qwen-v2-full --stop-at 04:15   # results/<tag>/, hard stop (IST)
uv run python -m scripts.run_batch --compare           # one table across results/*/ → results/ablation.md

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
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import docker

from agent.config import DATA_DIR, ROOT, RUNS_DIR
from agent.dataset import get_raw
from agent.schemas import RunState

IDS_PATH = DATA_DIR / "batch_ids.json"
OUT_DIR = ROOT / "results"
SEED = 7
QUOTA = {"django/django": 4, "sympy/sympy": 4, "pytest-dev/pytest": 3, "psf/requests": 1, "pallets/flask": 1}
ALREADY_PULLED = ["django__django-11099", "sympy__sympy-12481", "pytest-dev__pytest-7432", "psf__requests-2317"]
INSTANCE_TIMEOUT_S = 40 * 60
EXCLUDED_FROM_RESOLVED = {"psf__requests-2317"}  # its gold patch fails too: tests need live httpbin
IST = ZoneInfo("Asia/Kolkata")
BASELINE_TAG = "qwen-v1"


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


def seconds_until(hhmm: str, now: datetime | None = None) -> float:
    """Seconds to the next HH:MM in IST."""
    now = now or datetime.now(IST)
    h, m = hhmm.split(":")
    target = now.replace(hour=int(h), minute=int(m), second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def run_instance(iid: str, tag: str = "batch", timeout_s: float = INSTANCE_TIMEOUT_S) -> dict:
    run_id = f"{tag}-{iid}"
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    note = ""
    with open(d / "stdout.log", "w") as log:
        p = subprocess.Popen([sys.executable, "-m", "scripts.run_one", "--instance", iid, "--run-id", run_id],
                             cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (d / "pid").write_text(str(p.pid))
        try:
            rc = p.wait(timeout=timeout_s)
            if rc < 0:
                note = f"process killed by signal {-rc}"
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            note = f"timed out after {int(timeout_s)}s"
    cleanup_containers(run_id)
    return summarize(iid, run_id, time.time() - t0, note)


def summarize(iid: str, run_id: str, duration: float, note: str) -> dict:
    gold = set(re.findall(r"^diff --git a/(\S+) b/", get_raw(iid)["patch"], re.M))
    row = {"instance_id": iid, "repo": iid.split("__")[0], "run_id": run_id, "duration_s": round(duration),
           "note": note, "status": "crashed", "verified": False, "regression_only_pass": False,
           "resolved": False, "oracle_ran": False,
           "gold_file_found": False, "repro_valid": False, "attempts": 0, "attempts_applied": 0,
           "tokens_in": 0, "tokens_out": 0, "failure_reason": note, "lint_repairs": 0, "run_config": {}}
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
        lint_repairs=count_lint_repairs(run_id), run_config=s.run_config,
    )
    return row


def count_lint_repairs(run_id: str) -> int:
    p = RUNS_DIR / run_id / "events.jsonl"
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text(encoding="utf-8").splitlines() if '"lint_repaired"' in line)


def normalize(row: dict) -> dict:
    """Rows from before H1 counted regression-only passes as verified; score every row under the current rules."""
    r = dict(row)
    if "regression_only_pass" not in r:
        r["regression_only_pass"] = bool(r.get("verified")) and not r.get("repro_valid")
        r["verified"] = bool(r.get("verified")) and bool(r.get("repro_valid"))
    return r


def metrics(rows: list[dict]) -> dict:
    rows = [normalize(r) for r in rows]
    fair = [r for r in rows if r["instance_id"] not in EXCLUDED_FROM_RESOLVED]
    n = len(rows)
    return {
        "n": n, "fair_n": len(fair),
        "resolved": sum(1 for r in fair if r.get("resolved")),
        "verified": sum(1 for r in rows if r.get("verified")),
        "regression_only": sum(1 for r in rows if r.get("regression_only_pass")),
        "false_positives": sum(1 for r in rows if (r.get("verified") or r.get("regression_only_pass"))
                               and not r.get("resolved")),
        "false_negatives": sum(1 for r in fair if r.get("resolved") and not r.get("verified")),
        "gold_file": sum(1 for r in rows if r.get("gold_file_found")),
        "repro": sum(1 for r in rows if r.get("repro_valid")),
        "clean_apply": sum(1 for r in rows if r.get("attempts_applied")),
        "lint_repairs": sum(r.get("lint_repairs", 0) for r in rows),
        "crashed": sum(1 for r in rows if r.get("status") == "crashed"),
        "mean_s": round(sum(r.get("duration_s", 0) for r in rows) / n) if n else 0,
        "tokens": sum(r.get("tokens_in", 0) + r.get("tokens_out", 0) for r in rows),
    }


def report(rows: list[dict]) -> str:
    rows = [normalize(r) for r in rows]
    m = metrics(rows)
    n = m["n"]
    pct = lambda a: f"{a}/{n} ({100 * a / n:.0f}%)" if n else "0/0"  # noqa: E731
    flags = (rows[-1].get("run_config") or {}) if rows else {}
    lines = [
        f"Instances: {n}  |  model: {os.getenv('CODER_MODEL', '?')}  |  max attempts: {os.getenv('MAX_ATTEMPTS', '3')}",
        f"Flags: {json.dumps(flags) if flags else 'not recorded'}",
        "",
        "| Metric | Result |", "|---|---|",
        f"| **Resolved (official grading, requests-2317 excluded)** | **{m['resolved']}/{m['fair_n']}** |",
        f"| Passed agent verification (repro + regression) | {pct(m['verified'])} |",
        f"| Passed regression only (not counted as verified) | {pct(m['regression_only'])} |",
        f"| Passed either check but NOT resolved (false positive) | {m['false_positives']} |",
        f"| Resolved but not verified (false negative) | {m['false_negatives']} |",
        f"| Gold file in candidate files | {pct(m['gold_file'])} |",
        f"| Valid reproduction script | {pct(m['repro'])} |",
        f"| At least one patch applied cleanly | {pct(m['clean_apply'])} |",
        f"| Lint repairs | {m['lint_repairs']} |",
        f"| Crashed / timed out | {m['crashed']} |",
        f"| Mean wall time | {m['mean_s']}s |",
        f"| Tokens (in + out) | {m['tokens']:,} |",
        "",
        "| Instance | Localized | Repro | Applied | Verified | Resolved | Time |", "|---|---|---|---|---|---|---|",
    ]
    yn = lambda b: "yes" if b else "no"  # noqa: E731
    for r in rows:
        lines.append(f"| {r['instance_id']} | {yn(r.get('gold_file_found'))} | {yn(r.get('repro_valid'))} | "
                     f"{r.get('attempts_applied', 0)}/{r.get('attempts', 0)} | {yn(r.get('verified'))} | "
                     f"{yn(r.get('resolved')) if r.get('oracle_ran') else 'not graded'} | "
                     f"{r.get('duration_s', 0)}s |")
    return "\n".join(lines) + "\n"


def paths(tag: str | None) -> tuple[Path, Path]:
    d = OUT_DIR / tag if tag else OUT_DIR
    return d / "batch_results.json", d / "batch_results.md"


def save(rows: list[dict], tag: str | None = None) -> None:
    out_json, out_md = paths(tag)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=1))
    out_md.write_text(report(rows))


def compare(root: Path = OUT_DIR) -> str:
    """One row per tag. Partial batches show their own n, so tags are only comparable on equal instance sets."""
    lines = [f"Note: {BASELINE_TAG} predates the H1 fixes; its rows are rescored under H1 rules here.", "",
             "| Tag | Instances | Resolved (requests-2317 excluded) | Agent-verified | Regression only | "
             "False positives | False negatives | Gold file | Valid repro | Clean apply | Lint repairs | "
             "Mean time | Tokens |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in sorted(root.glob("*/batch_results.json")):
        m = metrics(json.loads(f.read_text()))
        lines.append(f"| {f.parent.name} | {m['n']} | {m['resolved']}/{m['fair_n']} | {m['verified']} | "
                     f"{m['regression_only']} | {m['false_positives']} | {m['false_negatives']} | {m['gold_file']} | "
                     f"{m['repro']} | {m['clean_apply']} | {m['lint_repairs']} | {m['mean_s']}s | {m['tokens']:,} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--tag", default=None, help="write to results/<tag>/ and name runs <tag>-<instance>")
    ap.add_argument("--only", default="", help="comma-separated instance ids (subset of the batch)")
    ap.add_argument("--stop-at", default="", help="HH:MM IST: no new instance after this, and running ones are cut off")
    args = ap.parse_args()
    if args.compare:
        text = compare()
        (OUT_DIR / "ablation.md").write_text(text)
        print(text)
        return 0
    if args.select:
        print("\n".join(select()))
        return 0
    out_json, _ = paths(args.tag)
    rows = json.loads(out_json.read_text()) if out_json.exists() else []
    if args.report:
        save(rows, args.tag)
        print(report(rows))
        return 0
    ids = json.loads(IDS_PATH.read_text()) if IDS_PATH.exists() else select()
    if args.only:
        ids = [i for i in ids if i in set(args.only.split(","))]
    done = {r["instance_id"] for r in rows}
    deadline = time.time() + seconds_until(args.stop_at) if args.stop_at else None
    for i, iid in enumerate(ids, 1):
        if iid in done:
            continue
        timeout_s = float(INSTANCE_TIMEOUT_S)
        if deadline is not None:
            left = deadline - time.time()
            if left < 120:
                print(f"stop time {args.stop_at} reached; {len(ids) - len(rows)} instances not run", flush=True)
                break
            timeout_s = min(timeout_s, left)
        print(f"[{i}/{len(ids)}] {iid}", flush=True)
        row = run_instance(iid, args.tag or "batch", timeout_s)
        rows.append(row)
        save(rows, args.tag)  # after every instance, so a crash keeps partial results
        print(f"    status={row['status']} verified={row['verified']} resolved={row['resolved']} "
              f"{row['duration_s']}s {row['note']}", flush=True)
    print(report(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
