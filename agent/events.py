"""Event log: emit() appends JSON lines to runs/<run_id>/events.jsonl. The UI reads only these files."""
import json
import threading
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from agent.config import RUNS_DIR

EventType = Literal["start", "llm_call", "info", "result", "error", "end"]

_lock = threading.Lock()


def run_dir(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _default(o: Any) -> Any:
    if isinstance(o, BaseModel):
        return o.model_dump()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def emit(run_id: str, stage: str, type: EventType, **payload: Any) -> dict:
    event = {"ts": time.time(), "stage": stage, "type": type, **payload}
    line = json.dumps(event, default=_default, ensure_ascii=False)
    with _lock:
        with open(run_dir(run_id) / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return event


def write_state(state: BaseModel) -> None:
    """Atomically write runs/<run_id>/state.json."""
    d = run_dir(state.run_id)  # type: ignore[attr-defined]
    tmp = d / "state.json.tmp"
    tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(d / "state.json")


def read_events(run_id: str) -> list[dict]:
    p = RUNS_DIR / run_id / "events.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate partially-written trailing line
    return out
