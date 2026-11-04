"""PatchPilot dashboard — uv run streamlit run ui/app.py

Reads ONLY runs/<run_id>/events.jsonl + state.json (and data/candidates.json for the picker).
"Run" spawns scripts/run_one.py as a subprocess; "Approve" writes final.patch and commits it on
branch agent/<instance_id> in the cache/repos copy.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.config import CACHE_DIR, DATA_DIR, RUNS_DIR  # noqa: E402
from agent.events import emit, read_events  # noqa: E402
from agent.schemas import RunState  # noqa: E402

STAGES = [
    ("planning", "Plan"), ("localizing", "Localize"), ("reproducing", "Reproduce"),
    ("patching", "Patch"), ("verifying", "Verify"), ("oracle", "Oracle"), ("awaiting_review", "Review"),
]
TERMINAL = {"awaiting_review", "approved", "failed"}
STALE_S = 15 * 60
# optional cost estimate, USD per 1M tokens (local Ollama = 0)
PRICE_IN = float(os.getenv("PRICE_IN_PER_M", "0") or 0)
PRICE_OUT = float(os.getenv("PRICE_OUT_PER_M", "0") or 0)

st.set_page_config(page_title="PatchPilot", page_icon="🛠️", layout="wide")


# ------------------------------------------------------------------ data access
@st.cache_data(ttl=60)
def load_candidates() -> list[dict]:
    p = DATA_DIR / "candidates.json"
    return json.loads(p.read_text()) if p.exists() else []


def load_state(run_id: str) -> RunState | None:
    p = RUNS_DIR / run_id / "state.json"
    if not p.exists():
        return None
    try:
        return RunState.model_validate_json(p.read_text(encoding="utf-8"))
    except Exception:
        return None  # mid-write; next refresh picks it up


def list_runs() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    runs = [d for d in RUNS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")
            and ((d / "state.json").exists() or (d / "events.jsonl").exists())]
    runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [d.name for d in runs]


def pid_alive(run_id: str) -> bool:
    p = RUNS_DIR / run_id / "pid"
    if not p.exists():
        return False
    try:
        pid = int(p.read_text().strip())
        os.kill(pid, 0)
        # zombie check (child we spawned that already exited)
        stat = Path(f"/proc/{pid}/stat")
        return not (stat.exists() and stat.read_text().split()[2] == "Z")
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        return False


def run_phase(run_id: str, events: list[dict], state: RunState | None) -> str:
    """'running' | 'done' | 'stalled'"""
    if any(e.get("stage") == "run" and e.get("type") == "end" for e in events):
        return "done"
    if state and state.status in ("approved",):
        return "done"
    if pid_alive(run_id):
        return "running"
    last = events[-1]["ts"] if events else 0
    return "running" if time.time() - last < 30 else "stalled"


def active_run() -> str | None:
    for rid in list_runs()[:10]:
        if pid_alive(rid) and run_phase(rid, read_events(rid), load_state(rid)) == "running":
            return rid
    return None


def spawn_run(instance_id: str) -> str:
    run_id = f"{instance_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    log = open(d / "stdout.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "scripts.run_one", "--instance", instance_id, "--run-id", run_id],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    (d / "pid").write_text(str(proc.pid))
    return run_id


def approve(state: RunState, diff: str) -> tuple[bool, str]:
    """Write final.patch and commit it on branch agent/<instance_id> in cache/repos/<id> (tree restored after)."""
    run_dir = RUNS_DIR / state.run_id
    (run_dir / "final.patch").write_text(diff, encoding="utf-8")
    repo = CACHE_DIR / "repos" / state.issue.instance_id
    if not (repo / ".git").exists():
        return False, f"repo copy not found at {repo}"
    branch = f"agent/{state.issue.instance_id}"
    base = state.issue.base_commit
    git = ["git", "-C", str(repo), "-c", "user.name=PatchPilot", "-c", "user.email=patchpilot@localhost"]
    msg = (f"PatchPilot: fix {state.issue.instance_id}\n\nrun: {state.run_id}\n"
           f"official grading: {'RESOLVED' if state.oracle and state.oracle.resolved else 'not resolved'}")
    steps = [
        git + ["checkout", "-q", "-B", branch, base],
        git + ["apply", "--index", str(run_dir / "final.patch")],   # index only: untracked cache files stay out
        git + ["commit", "-q", "-m", msg],
    ]
    try:
        for cmd in steps:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                subprocess.run(git + ["reset", "-q", "--hard"], capture_output=True, timeout=60)
                subprocess.run(git + ["checkout", "-q", "--detach", base], capture_output=True, timeout=60)
                return False, f"`{' '.join(cmd[5:])}` failed: {r.stderr.strip()[:400]}"
        sha = subprocess.run(git + ["rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    finally:
        # localization reads this copy — always return it to the pristine base commit
        subprocess.run(git + ["checkout", "-q", "--detach", base], capture_output=True, timeout=60)
    state.status = "approved"
    tmp = run_dir / "state.json.tmp"
    tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(run_dir / "state.json")
    emit(state.run_id, "review", "result", approved=True, branch=branch, commit=sha)
    return True, f"committed {sha} on `{branch}` in {repo}"


# ------------------------------------------------------------------ rendering helpers
def badge(label: str, value: str, kind: str) -> str:
    colors = {"ok": ("#0f5132", "#d1e7dd"), "bad": ("#842029", "#f8d7da"),
              "wait": ("#664d03", "#fff3cd"), "none": ("#41464b", "#e2e3e5")}
    fg, bg = colors[kind]
    return (f"<div style='display:inline-block;margin:0 12px 8px 0;padding:10px 16px;border-radius:10px;"
            f"background:{bg};color:{fg};font-weight:600'>{label}<br>"
            f"<span style='font-size:1.25em'>{value}</span></div>")


def stage_timeline(events: list[dict], state: RunState | None, phase: str) -> None:
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    counts: dict[str, int] = {}
    errors: set[str] = set()
    for e in events:
        s, t = e.get("stage"), e.get("type")
        if t == "start":
            starts.setdefault(s, e["ts"])
            counts[s] = counts.get(s, 0) + 1
        elif t == "end":
            ends[s] = e["ts"]
        elif t == "error" and not e.get("warning"):
            errors.add(s)
    current = state.status if state else "planning"
    cols = st.columns(len(STAGES))
    for col, (key, label) in zip(cols, STAGES):
        if key in errors:
            icon = "❌"
        elif key == "awaiting_review":
            icon = "✅" if current in ("awaiting_review", "approved") else "⬜"
        elif key == current and phase == "running":
            icon = "⏳"
        elif key in ends or (key in starts and phase != "running"):
            icon = "✅"
        else:
            icon = "⬜"
        dur = ""
        if key in starts:
            end = ends.get(key) or (time.time() if key == current and phase == "running" else starts[key])
            dur = f"{end - starts[key]:.0f}s"
        n = f" ×{counts[key]}" if counts.get(key, 0) > 1 else ""
        col.markdown(f"<div style='text-align:center'><div style='font-size:1.6em'>{icon}</div>"
                     f"<b>{label}{n}</b><br><small>{dur}</small></div>", unsafe_allow_html=True)


def render_run(run_id: str) -> None:
    events = read_events(run_id)
    state = load_state(run_id)
    phase = run_phase(run_id, events, state)

    head = st.columns([3, 1])
    head[0].subheader(run_id)
    head[1].caption({"running": "⏳ running (auto-refresh 2s)", "done": "finished",
                     "stalled": "⚠️ stalled (process gone without end event)"}[phase])
    if state is None:
        st.info("Waiting for the run to start…")
        return

    stage_timeline(events, state, phase)

    # badges
    if state.status in ("awaiting_review", "approved"):
        agent_b = (badge("Agent verification", "PASS (regression only)", "wait") if state.regression_only
                   else badge("Agent verification", "PASS", "ok"))
    elif state.status == "failed" or phase == "stalled":
        agent_b = badge("Agent verification", "FAIL", "bad")
    else:
        agent_b = badge("Agent verification", "running…", "wait")
    if state.oracle is not None:
        oracle_b = badge("Official SWE-bench grading", "RESOLVED" if state.oracle.resolved else "NOT RESOLVED",
                         "ok" if state.oracle.resolved else "bad")
    elif state.status == "oracle":
        oracle_b = badge("Official SWE-bench grading", "grading…", "wait")
    else:
        oracle_b = badge("Official SWE-bench grading", "—", "none")
    cost = state.tokens_in / 1e6 * PRICE_IN + state.tokens_out / 1e6 * PRICE_OUT
    n_calls = sum(1 for e in events if e.get("type") == "llm_call")
    n_cached = sum(1 for e in events if e.get("type") == "llm_call" and e.get("cached"))
    tok_b = badge("Tokens (in / out)", f"{state.tokens_in:,} / {state.tokens_out:,}", "none")
    calls_b = badge("LLM calls", f"{n_calls} ({n_cached} cached)" + (f" · ${cost:.4f}" if cost else ""), "none")
    st.markdown(agent_b + oracle_b + tok_b + calls_b, unsafe_allow_html=True)
    if state.failure_reason:
        st.error(f"Failure: {state.failure_reason}")

    with st.expander("📝 Issue", expanded=False):
        st.caption(f"{state.issue.repo} @ {state.issue.base_commit[:10]} (v{state.issue.version})")
        st.markdown(state.issue.problem_statement[:6000])

    # plan
    if state.plan:
        p = state.plan
        with st.container(border=True):
            st.markdown(f"### 🧭 Plan  `{p.kind}`")
            st.markdown(f"**Summary:** {p.summary}\n\n**Hypothesis:** {p.hypothesis}")
            c1, c2 = st.columns(2)
            c1.markdown("**Steps**\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(p.steps)))
            c2.markdown("**Acceptance criteria**\n" + "\n".join(f"- {a}" for a in p.acceptance_criteria))
            st.caption("search terms: " + ", ".join(f"`{t}`" for t in p.search_terms))

    # localization
    if state.context:
        ctx = state.context
        with st.expander(f"🔎 Localization — {len(ctx.candidate_files)} files, {len(ctx.chunks)} chunks", expanded=False):
            st.markdown("**Candidate files:** " + ", ".join(f"`{f}`" for f in ctx.candidate_files))
            if ctx.edit_locations:
                st.markdown("**Edit locations:** " + ", ".join(f"`{x}`" for x in ctx.edit_locations))
            for c in ctx.chunks:
                st.markdown(f"`{c.file_path}:{c.start_line}-{c.end_line}` · {c.kind} **{c.symbol}**"
                            + (f" · score {c.score:.3f}" if c.score else ""))
                st.code(c.code[:3000], language="python")
            if ctx.notes:
                st.caption("Notes")
                st.text(ctx.notes[:3000])

    # repro
    if state.status not in ("planning", "localizing"):
        with st.expander("🧪 Reproduction", expanded=not state.attempts):
            if state.repro_script:
                st.code(state.repro_script, language="python")
                if state.repro_base:
                    st.markdown("**Base commit result** (expected to fail):")
                    st.code(state.repro_base.failure_summary or "(no output)", language="text")
            elif state.status != "reproducing":
                st.warning("No valid reproduction — verification used regression tests only.")
            else:
                st.info("Writing reproduction script…")

    # attempts
    if state.attempts:
        st.markdown("### 🔧 Attempts")
        tabs = st.tabs([f"Attempt {a.n} {'✅' if _attempt_ok(a) else '❌'}" for a in state.attempts])
        for tab, a in zip(tabs, state.attempts):
            with tab:
                if a.apply_errors:
                    st.error("Apply errors:\n\n" + "\n\n".join(a.apply_errors))
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Diff**")
                    st.code(a.diff or "(no changes)", language="diff")
                with c2:
                    st.markdown(f"**Edits ({len(a.edits)})**")
                    for e in a.edits:
                        st.caption(e.file_path)
                        st.code(f"<<<<<<< SEARCH\n{e.search}=======\n{e.replace}>>>>>>> REPLACE", language="text")
                if a.repro:
                    st.markdown(f"**Reproduction after patch:** {'✅ passed' if a.repro.passed else '❌ failed'}")
                    if not a.repro.passed:
                        st.code(a.repro.failure_summary, language="text")
                if a.regression:
                    st.markdown(f"**Regression** ({', '.join(f'`{f}`' for f in a.regression_files)}): "
                                f"{'✅ no new failures' if a.regression.passed else '❌ new failures'}"
                                f" · {a.regression.exec.duration_s:.0f}s")
                    if a.regression.failure_summary:
                        st.code(a.regression.failure_summary[-4000:], language="text")
                for e in events:
                    if e.get("check") == "reflect" and e.get("attempt") == a.n:
                        st.info(f"**Reflection** ({e.get('verdict')}): {e.get('analysis')}")

    # oracle detail
    if state.oracle:
        with st.expander("🏛️ Official grading detail", expanded=False):
            o = state.oracle
            st.markdown("**FAIL_TO_PASS**")
            st.table([{"test": k, "status": v} for k, v in o.fail_to_pass.items()] or [{"test": "-", "status": "-"}])
            bad = {k: v for k, v in o.pass_to_pass.items() if v != "PASSED"}
            st.markdown(f"**PASS_TO_PASS:** {len(o.pass_to_pass) - len(bad)}/{len(o.pass_to_pass)} passed")
            if bad:
                st.table([{"test": k, "status": v} for k, v in bad.items()])
            st.code(o.log_tail[-4000:], language="text")

    # review
    if state.status in ("awaiting_review", "approved"):
        final = next((a for a in reversed(state.attempts) if _attempt_ok(a)), None)
        with st.container(border=True):
            st.markdown("### 👤 Human review")
            if state.status == "approved":
                st.success(f"Approved — final patch at `runs/{run_id}/final.patch`, "
                           f"branch `agent/{state.issue.instance_id}` in `cache/repos/{state.issue.instance_id}`")
            elif final is not None:
                st.code(final.diff, language="diff")
                if st.button("✅ Approve & commit", type="primary", key=f"approve-{run_id}"):
                    ok, msg = approve(state, final.diff)
                    (st.success if ok else st.error)(msg)

    with st.expander(f"📜 Event log ({len(events)})", expanded=False):
        rows = []
        for e in events[-200:]:
            extra = {k: v for k, v in e.items() if k not in ("ts", "stage", "type", "script", "diff", "traceback",
                                                             "plan", "top", "chunks", "failure", "output_tail")}
            rows.append({"time": datetime.fromtimestamp(e["ts"]).strftime("%H:%M:%S"), "stage": e.get("stage"),
                         "type": e.get("type"), "detail": json.dumps(extra, default=str)[:300]})
        st.dataframe(rows, width="stretch", hide_index=True)


def _attempt_ok(a) -> bool:
    if not a.diff.strip() or a.apply_errors:
        return False
    if a.repro is not None and not a.repro.passed:
        return False
    if a.regression is not None and not a.regression.passed:
        return False
    return a.repro is not None or a.regression is not None


# ------------------------------------------------------------------ layout
st.title("🛠️ PatchPilot")
st.caption("Autonomous SWE agent · plan → localize → reproduce → patch → verify → official grading → human review")

with st.sidebar:
    st.header("New run")
    cands = load_candidates()
    if not cands:
        st.warning("No data/candidates.json — run `uv run python -m scripts.select_instances`.")
    repos = sorted({c["repo"] for c in cands})
    repo_f = st.selectbox("Repository", ["(all)"] + repos)
    shown = [c for c in cands if repo_f == "(all)" or c["repo"] == repo_f]
    labels = {c["instance_id"]: f"{c['instance_id']} — {c['title'][:60]}" for c in shown}
    inst = st.selectbox("Instance", list(labels), format_func=lambda i: labels[i]) if shown else None
    busy = active_run()
    if busy:
        st.info(f"Run in progress: `{busy}` (one at a time — GPU/RAM)")
    if st.button("▶ Run", type="primary", disabled=inst is None or busy is not None, width="stretch"):
        st.session_state["run_id"] = spawn_run(inst)
        st.rerun()

    st.header("Runs")
    runs = list_runs()
    if runs:
        default = st.session_state.get("run_id")
        idx = runs.index(default) if default in runs else 0

        def _fmt(r: str) -> str:
            s = load_state(r)
            icon = {"awaiting_review": "✅", "approved": "🟢", "failed": "❌"}.get(s.status if s else "", "⏳")
            if s and s.status == "awaiting_review" and s.regression_only:
                icon = "⚠️"
            res = " · 🏛️" if s and s.oracle and s.oracle.resolved else ""
            return f"{icon} {r}{res}"

        st.session_state["run_id"] = st.radio("Past runs", runs, index=idx, format_func=_fmt,
                                              label_visibility="collapsed")
    else:
        st.caption("No runs yet.")

run_id = st.session_state.get("run_id")
if not run_id:
    st.info("Pick an instance and press **Run**, or select a past run.")
else:
    _events = read_events(run_id)
    _live = run_phase(run_id, _events, load_state(run_id)) == "running"

    @st.fragment(run_every="2s" if _live else None)
    def _panel():
        render_run(run_id)
        # run just finished → one full rerun so the sidebar updates and polling stops
        if _live and run_phase(run_id, read_events(run_id), load_state(run_id)) != "running":
            st.rerun(scope="app")

    _panel()
