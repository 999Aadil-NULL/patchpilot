"""H0: context-window truncation audit.

uv run python -m scripts.audit_context

Scans runs/**/events.jsonl (including runs/archive) for llm_call events and prints the prompt-size
distribution per role. A server that silently truncates prompts shows up as many calls piled up just
under one limit. Cached calls are excluded: they repeat a real call and would double count it.
"""
import json
import math
import sys
from collections import defaultdict

from agent.config import RUNS_DIR

CEILINGS = [2048, 4096, 8192, 16384, 32768]
TOLERANCE = 0.02


def percentile(sorted_vals: list[int], p: float) -> int:
    return sorted_vals[max(0, math.ceil(p / 100 * len(sorted_vals)) - 1)]


def load_calls() -> tuple[list[dict], int, int]:
    calls, cached, files = [], 0, 0
    for path in sorted(RUNS_DIR.rglob("events.jsonl")):
        files += 1
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") != "llm_call":
                continue
            if ev.get("cached"):
                cached += 1
            elif ev.get("tokens_in"):
                calls.append(ev)
    return calls, cached, files


def print_group(name: str, tokens: list[int]) -> None:
    vals = sorted(tokens)
    print(f"{name:<34} n={len(vals):<5} p50={percentile(vals, 50):<7} p95={percentile(vals, 95):<7} max={vals[-1]}")
    for c in CEILINGS:
        near = sum(1 for t in vals if abs(t - c) <= c * TOLERANCE)
        print(f"    within {TOLERANCE:.0%} of {c:<6} {near}")


def main() -> int:
    calls, cached, files = load_calls()
    print(f"{files} event files, {len(calls)} uncached calls with token counts, {cached} cached calls skipped\n")
    if not calls:
        return 1
    by_role, by_model = defaultdict(list), defaultdict(list)
    for ev in calls:
        by_role[ev.get("role", "?")].append(ev["tokens_in"])
        by_model[f"{ev.get('role', '?')} / {ev.get('model', '?')}"].append(ev["tokens_in"])
    print("tokens_in per role")
    for name in sorted(by_role):
        print_group(name, by_role[name])
    print("\ntokens_in per role and model")
    for name in sorted(by_model):
        print_group(name, by_model[name])
    return 0


if __name__ == "__main__":
    sys.exit(main())
