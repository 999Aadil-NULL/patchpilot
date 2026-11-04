"""H4: choose one candidate patch using agent-visible signals only (repro, regression, lint, diff shape).

A candidate is a dict: diff, clean_apply, lint_clean, repro_passed, regression_passed, new_failures.
repro_passed / regression_passed / new_failures are None when that check did not run.
"""
from __future__ import annotations

import re
from collections import defaultdict

_TRAILING_COMMENT_RE = re.compile(r"\s+#.*$")


def normalize_diff(diff: str) -> str:
    """Changed lines only, with whitespace, blank lines and comments removed, so equivalent patches compare equal."""
    out = []
    for line in diff.splitlines():
        if line.startswith(("+++", "---")) or line[:1] not in ("+", "-"):
            continue
        body = _TRAILING_COMMENT_RE.sub("", line[1:]).strip()
        if body and not body.startswith("#"):
            out.append(line[0] + re.sub(r"\s+", "", body))
    return "\n".join(out)


def _passing(c: dict) -> bool:
    ran = c.get("repro_passed") is not None or c.get("regression_passed") is not None
    return (ran and c.get("clean_apply", False) and c.get("lint_clean", True) and bool(c.get("diff", "").strip())
            and c.get("repro_passed") is not False and c.get("regression_passed") is not False)


def _failure_rank(c: dict) -> tuple:
    """Lower is better. A regression that never ran is worse than any measured count."""
    failures = c.get("new_failures")
    return (not c.get("clean_apply", False), not c.get("lint_clean", True), failures is None, failures or 0)


def select(candidates: list[dict]) -> tuple[int, str]:
    """Returns (index, reason). Assumes at least one candidate."""
    passing = [i for i, c in enumerate(candidates) if _passing(c)]
    if not passing:
        best = min(range(len(candidates)), key=lambda i: (_failure_rank(candidates[i]), i))
        c = candidates[best]
        return best, (f"no candidate passed; best failure (clean_apply={c.get('clean_apply', False)}, "
                      f"lint_clean={c.get('lint_clean', True)}, new_failures={c.get('new_failures')})")
    groups: dict[str, list[int]] = defaultdict(list)
    for i in passing:
        groups[normalize_diff(candidates[i]["diff"])].append(i)
    top = max(len(g) for g in groups.values())
    tied = [g for g in groups.values() if len(g) == top]
    best = min((g[0] for g in tied), key=lambda i: (len(candidates[i]["diff"]), i))
    if len(passing) == 1:
        return best, "only passing candidate"
    if len(tied) == 1 and top > 1:
        return best, f"majority: {top} of {len(passing)} passing candidates agree"
    return best, f"tie among {len(passing)} passing candidates; smallest diff"
