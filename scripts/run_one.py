"""uv run python -m scripts.run_one --instance <id> [--run-id <id>]"""
import argparse
import sys

from rich.console import Console
from rich.table import Table

from agent.orchestrator import run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    con = Console()
    con.print(f"[bold]PatchPilot[/] → {args.instance}")
    st = run(args.instance, args.run_id)

    t = Table(title=f"run {st.run_id}", show_header=False)
    t.add_row("status", st.status)
    if st.failure_reason:
        t.add_row("failure_reason", st.failure_reason[:300])
    t.add_row("candidate files", ", ".join(st.context.candidate_files) if st.context else "-")
    t.add_row("repro", "valid" if st.repro_script else "none (regression-only)")
    for a in st.attempts:
        t.add_row(f"attempt {a.n}",
                  f"edits={len(a.edits)} apply_errors={len(a.apply_errors)} "
                  f"repro={'-' if a.repro is None else a.repro.passed} "
                  f"regression={'-' if a.regression is None else a.regression.passed} {a.regression_files}")
    t.add_row("agent verification", "PASS" if st.status in ("awaiting_review", "approved") else "FAIL")
    t.add_row("official grading", "-" if st.oracle is None else ("RESOLVED" if st.oracle.resolved else "NOT RESOLVED"))
    t.add_row("tokens", f"in={st.tokens_in} out={st.tokens_out}")
    con.print(t)
    if st.attempts and st.attempts[-1].diff:
        con.print("[bold]final diff:[/]")
        con.print(next((a.diff for a in reversed(st.attempts) if a.diff), ""), markup=False, highlight=False)
    con.print(f"events: runs/{st.run_id}/events.jsonl")
    return 0 if st.status == "awaiting_review" else 1


if __name__ == "__main__":
    sys.exit(main())
