"""Step-1 smoke test: uv run python -m scripts.smoke_llm"""
import sys
import time

from agent import llm
from agent.config import role_config
from agent.events import read_events
from agent.schemas import Plan

RUN_ID = f"_smoke_llm_{int(time.time())}"


def main() -> int:
    llm.set_run_context(RUN_ID, "smoke")
    ok = True
    for role in ("planner", "coder"):
        cfg = role_config(role)
        out = llm.complete(role, [{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=512)
        good = "OK" in out.upper()
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {role} ({cfg.model}): {out.strip()[:80]!r}")

    plan = llm.complete_json("planner", [{"role": "user", "content":
        "Issue: `sympy.sqrt(-1)` should return `I` but raises ValueError in sympy/core/power.py. "
        "Produce a fix plan."}], Plan)
    print(f"[PASS] complete_json -> Plan(kind={plan.kind!r}, steps={len(plan.steps)}, "
          f"search_terms={plan.search_terms[:4]})")

    # extractor unit checks (no network)
    assert llm.extract_json('blah ```json\n{"a": 1}\n``` tail') == {"a": 1}
    assert llm.extract_json('prefix {"a": "x}{", "b": {"c": 2}} suffix') == {"a": "x}{", "b": {"c": 2}}
    print("[PASS] extract_json fenced + balanced")

    u = llm.get_usage()
    evs = [e for e in read_events(RUN_ID) if e["type"] == "llm_call"]
    print(f"usage: calls={u.calls} cached={u.cached_calls} in={u.tokens_in} out={u.tokens_out}; "
          f"llm_call events={len(evs)}")
    return 0 if ok and evs else 1


if __name__ == "__main__":
    sys.exit(main())
