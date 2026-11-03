"""Step-2 acceptance check: uv run python -m scripts.check_step2 [--instance ID]

1. Sandbox: exec, write/read file, run_repro on a trivial failing script, run_tests, diff, reset.
2. Oracle: grade(id, gold_patch) → resolved=True; grade(id, "") → resolved=False.
(Gold patch is read here only for the oracle self-test; this script never feeds the agent.)
"""
import argparse
import sys
import time

from agent.dataset import get_raw
from agent.oracle import grade
from agent.sandbox import Sandbox


def check(name: str, cond: bool, extra: str = "") -> bool:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")
    return cond


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="sympy__sympy-12481")
    args = ap.parse_args()
    iid = args.instance
    ok = True

    t0 = time.time()
    with Sandbox(iid, run_id="check2") as sb:
        r = sb.exec("python -c 'import sys; print(sys.version.split()[0])'", timeout=60)
        ok &= check("exec in conda env", r.exit_code == 0, r.stdout.strip())
        sb.write_file("pp_tmp/hello.txt", "héllo\n")
        ok &= check("write_file/read_file roundtrip", sb.read_file("pp_tmp/hello.txt") == "héllo\n")
        rr = sb.run_repro("def f():\n    return 1\nassert f() == 2, 'expected 2'\n")
        ok &= check("run_repro trivial failing script fails w/ AssertionError",
                    not rr.passed and "AssertionError" in rr.failure_summary,
                    rr.failure_summary.splitlines()[-1] if rr.failure_summary else "")
        rp = sb.run_repro("print('fine')\n")
        ok &= check("run_repro passing script passes", rp.passed)
        t = sb.exec("timeout_probe() { sleep 5; }; timeout_probe", timeout=2)
        ok &= check("exec timeout enforced", t.timed_out, f"exit={t.exit_code} {t.duration_s:.1f}s")
        sb.write_file("sympy/pp_new_module.py", "X = 1\n")
        d = sb.diff()
        ok &= check("diff includes new file, excludes repro", "pp_new_module.py" in d and "repro_issue" not in d)
        sb.reset()
        ok &= check("reset gives clean tree", sb.diff() == "")
        tr = sb.run_tests(["sympy/combinatorics/tests/test_permutations.py"])
        ok &= check("run_tests (base commit) ran", "tests finished" in tr.exec.stdout or tr.exec.exit_code in (0, 1),
                    f"exit={tr.exec.exit_code} {tr.exec.duration_s:.0f}s cmd={sb.test_command(['…'])!r}")
        dest = sb.export_repo()
        ok &= check("export_repo", (dest / ".git").exists() and (dest / "setup.py").exists(), str(dest))
    print(f"sandbox checks: {time.time() - t0:.0f}s")

    t0 = time.time()
    gold = grade(iid, get_raw(iid)["patch"])
    ok &= check("oracle gold patch → resolved=True", gold.resolved,
                f"F2P={gold.fail_to_pass} P2P pass={sum(v == 'PASSED' for v in gold.pass_to_pass.values())}"
                f"/{len(gold.pass_to_pass)}")
    if not gold.resolved:
        print(gold.log_tail)
    empty = grade(iid, "")
    ok &= check("oracle empty patch → resolved=False", not empty.resolved, f"F2P={empty.fail_to_pass}")
    print(f"oracle checks: {time.time() - t0:.0f}s")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
