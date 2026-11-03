"""Independent official grading. The ONLY module allowed to touch hidden dataset fields.

grade(instance_id, diff): FRESH container from the instance image → apply diff (same fallback
chain as the swebench harness) → run the official eval script (applies test_patch, runs tests)
→ parse with the swebench log parser → resolved = all FAIL_TO_PASS pass AND all PASS_TO_PASS pass.
"""
from __future__ import annotations

import hashlib
import time

from swebench.harness.constants import FAIL_TO_PASS, PASS_TO_PASS, EvalType, ResolvedStatus
from swebench.harness.docker_utils import exec_run_with_timeout
from swebench.harness.grading import get_eval_tests_report, get_logs_eval, get_resolution_status
from swebench.harness.utils import make_test_spec

from agent.config import CACHE_DIR
from agent.dataset import get_raw
from agent.sandbox import Sandbox
from agent.schemas import OracleResult

ORACLE_DIR = CACHE_DIR / "oracle"
EVAL_TIMEOUT_S = 1800
PATCH_PATH = "/tmp/pp_patch.diff"
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --3way",
    "git apply --verbose --reject",
    "patch --batch --forward --fuzz=5 -p1 -i",
]


def _tail(s: str, n: int = 80) -> str:
    return "\n".join(s.rstrip().splitlines()[-n:])


def grade(instance_id: str, diff: str) -> OracleResult:
    raw = get_raw(instance_id)
    spec = make_test_spec(raw)
    key = hashlib.sha256(diff.encode()).hexdigest()[:16]
    out_dir = ORACLE_DIR / instance_id / key
    out_dir.mkdir(parents=True, exist_ok=True)
    log_fp = out_dir / "test_output.txt"

    # network on, like the official harness (eval script runs `pip install -e .`)
    sb = Sandbox(instance_id, run_id=f"oracle-{key}", network=True, name_suffix=f".{int(time.time())}")
    try:
        sb.start()
        c = sb._c()
        if diff.strip():
            sb.write_file(PATCH_PATH, diff if diff.endswith("\n") else diff + "\n")
            applied, last = False, ""
            for i, cmd in enumerate(GIT_APPLY_CMDS):
                if i:
                    c.exec_run(["bash", "-c", "git checkout -- . ; git clean -fdq"], workdir="/testbed")
                r = c.exec_run(f"{cmd} {PATCH_PATH}", workdir="/testbed")
                last = r.output.decode("utf-8", errors="replace")
                if r.exit_code == 0:
                    applied = True
                    break
            if not applied:
                rc = c.exec_run(f"git apply --check --reverse {PATCH_PATH}", workdir="/testbed")
                applied = rc.exit_code == 0
            if not applied:
                return OracleResult(resolved=False, log_tail=f"PATCH APPLY FAILED:\n{_tail(last)}")

        sb.write_file("/eval.sh", spec.eval_script)
        output, timed_out, runtime = exec_run_with_timeout(c, "/bin/bash /eval.sh", EVAL_TIMEOUT_S)
        log_fp.write_text(output, encoding="utf-8")
        if timed_out:
            return OracleResult(resolved=False, log_tail=f"EVAL TIMED OUT after {EVAL_TIMEOUT_S}s\n{_tail(output)}")

        status_map, found = get_logs_eval(spec, str(log_fp))
        f2p_ids, p2p_ids = spec.FAIL_TO_PASS, spec.PASS_TO_PASS
        if not found:
            return OracleResult(
                resolved=False,
                fail_to_pass={t: "NOT_RUN" for t in f2p_ids},
                pass_to_pass={t: "NOT_RUN" for t in p2p_ids},
                log_tail=f"NO PARSEABLE TEST OUTPUT\n{_tail(output)}",
            )
        report = get_eval_tests_report(
            status_map, {"instance_id": instance_id, FAIL_TO_PASS: f2p_ids, PASS_TO_PASS: p2p_ids},
            eval_type=EvalType(spec.eval_type),
        )

        def _statuses(section: str) -> dict[str, str]:
            d = {t: "PASSED" for t in report[section]["success"]}
            d.update({t: status_map.get(t, "FAILED") for t in report[section]["failure"]})
            return d

        resolved = get_resolution_status(report) == ResolvedStatus.FULL.value
        return OracleResult(
            resolved=resolved,
            fail_to_pass=_statuses(FAIL_TO_PASS),
            pass_to_pass=_statuses(PASS_TO_PASS),
            log_tail=_tail(output.split(">>>>> End Test Output")[0], 60) + f"\n[eval runtime {runtime:.1f}s]",
        )
    except Exception as e:  # never crash the caller
        return OracleResult(resolved=False, log_tail=f"ORACLE ERROR: {type(e).__name__}: {e}")
    finally:
        sb.stop()
