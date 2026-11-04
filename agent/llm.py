"""Provider-agnostic LLM client: text completion, JSON extraction/validation, disk cache, usage log.

All providers go through the openai SDK via base_url. If a provider is not OpenAI-compatible
(e.g. Meta at the final stage), add ONE adapter class with the same `complete()` signature as
OpenAICompatProvider and select it in `_provider_for()`. Nothing else changes.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Literal, TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from agent import config
from agent.config import CACHE_DIR, RoleConfig, role_config
from agent.events import emit

Role = Literal["planner", "coder"]
T = TypeVar("T", bound=BaseModel)

LLM_CACHE_DIR = CACHE_DIR / "llm"


class LLMError(RuntimeError):
    pass


# ---------------------------------------------------------------- run context + usage
# Set by the orchestrator so llm_call events land in the right run without changing signatures.
_run_ctx: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar("_run_ctx", default=None)


def set_run_context(run_id: str | None, stage: str = "llm") -> None:
    _run_ctx.set((run_id, stage) if run_id else None)


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0
    calls: int = 0
    cached_calls: int = 0


_usage = Usage()
_usage_lock = threading.Lock()


def get_usage() -> Usage:
    with _usage_lock:
        return Usage(**_usage.__dict__)


def reset_usage() -> None:
    global _usage
    with _usage_lock:
        _usage = Usage()


# ---------------------------------------------------------------- providers
@dataclass
class Completion:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    finish_reason: str = ""


class QuotaExhausted(LLMError):
    """Hard quota (e.g. requests-per-day) — retrying won't help."""


_HARD_QUOTA_MARKERS = ("PerDay", "per day", "insufficient_quota", "billing")


def _is_hard_quota(exc: BaseException) -> bool:
    if not isinstance(exc, openai.APIStatusError) or exc.status_code != 429:
        return False
    body = str(getattr(exc, "body", "")) + str(exc)
    return any(m in body for m in _HARD_QUOTA_MARKERS)


def _retryable(exc: BaseException) -> bool:
    if _is_hard_quota(exc):
        return False
    if isinstance(exc, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                        openai.InternalServerError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


class OpenAICompatProvider:
    def __init__(self, cfg: RoleConfig):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key,
                             timeout=config.LLM_TIMEOUT_S, max_retries=0)

    # per-minute rate limits on free tiers need long waits: 5s, 10s, 20s, 40s, 60s
    @retry(retry=retry_if_exception(_retryable), stop=stop_after_attempt(6),
           wait=wait_exponential(multiplier=5, min=5, max=60), reraise=True)
    def complete(self, messages: list[dict], temperature: float, max_tokens: int) -> Completion:
        kw = {}
        if self.cfg.reasoning_effort:
            kw["reasoning_effort"] = self.cfg.reasoning_effort
        resp = self.client.chat.completions.create(
            model=self.cfg.model, messages=messages,
            temperature=temperature, max_tokens=max_tokens, **kw,
        )
        choice = resp.choices[0] if resp.choices else None
        text = (choice.message.content or "") if choice else ""
        u = resp.usage
        return Completion(text=text,
                          tokens_in=getattr(u, "prompt_tokens", 0) or 0,
                          tokens_out=getattr(u, "completion_tokens", 0) or 0,
                          finish_reason=(choice.finish_reason or "") if choice else "")


_providers: dict[tuple, OpenAICompatProvider] = {}


def _provider_for(cfg: RoleConfig) -> OpenAICompatProvider:
    key = (cfg.base_url, cfg.api_key, cfg.model)
    if key not in _providers:
        _providers[key] = OpenAICompatProvider(cfg)
    return _providers[key]


# ---------------------------------------------------------------- cache
def _cache_key(cfg: RoleConfig, messages: list[dict], temperature: float, max_tokens: int,
               sample_idx: int = 0) -> str:
    fields = {"base_url": cfg.base_url, "model": cfg.model, "messages": messages,
              "temperature": temperature, "max_tokens": max_tokens}
    if sample_idx:  # only when non-zero, so existing cache entries stay valid
        fields["sample_idx"] = sample_idx
    blob = json.dumps(fields, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _cache_get(key: str) -> Completion | None:
    p = LLM_CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return Completion(text=d["text"], tokens_in=d.get("tokens_in", 0), tokens_out=d.get("tokens_out", 0))
    except (json.JSONDecodeError, KeyError):
        return None


def _cache_put(key: str, c: Completion, model: str) -> None:
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LLM_CACHE_DIR / f"{key}.tmp"
    tmp.write_text(json.dumps({"model": model, "text": c.text, "tokens_in": c.tokens_in,
                               "tokens_out": c.tokens_out, "ts": time.time()}), encoding="utf-8")
    tmp.replace(LLM_CACHE_DIR / f"{key}.json")


# ---------------------------------------------------------------- prompt-size guard
CHARS_PER_TOKEN = 3.5


def _estimate_prompt_tokens(messages: list[dict]) -> int:
    return int(sum(len(m.get("content") or "") for m in messages) / CHARS_PER_TOKEN)


def _warn_if_prompt_large(role: Role, messages: list[dict]) -> None:
    """Warn only: the call still goes out unchanged. `warning` keeps the UI from marking the stage as failed."""
    estimate = _estimate_prompt_tokens(messages)
    ctx = _run_ctx.get()
    if ctx and estimate > config.LLM_MAX_PROMPT_TOKENS:
        emit(ctx[0], ctx[1], "error", error="prompt_near_limit", warning=True, role=role,
             estimated_tokens=estimate, limit=config.LLM_MAX_PROMPT_TOKENS)


# ---------------------------------------------------------------- public API
def complete(role: Role, messages: list[dict], *, temperature: float = 0.2, max_tokens: int = 4096,
             sample_idx: int = 0) -> str:
    cfg = role_config(role)
    key = _cache_key(cfg, messages, temperature, max_tokens, sample_idx)
    t0 = time.time()
    cached = False
    comp = _cache_get(key) if config.LLM_CACHE else None
    if comp is not None:
        cached = True
    else:
        _warn_if_prompt_large(role, messages)
        # thinking models spend hidden reasoning tokens from the same budget → enforce a floor,
        # and on finish_reason=length retry with a doubled budget (truncated output is never cached)
        budget = max(max_tokens, config.LLM_MAX_TOKENS_FLOOR)
        comp = None
        for _ in range(3):
            try:
                comp = _provider_for(cfg).complete(messages, temperature, budget)
            except Exception as e:
                _emit_call(role, cfg.model, 0, 0, time.time() - t0, False, error=f"{type(e).__name__}: {e}")
                cls = QuotaExhausted if _is_hard_quota(e) else LLMError
                raise cls(f"{role} call to {cfg.model} failed: {type(e).__name__}: {str(e)[:400]}") from e
            if comp.finish_reason != "length" or budget >= 65536:
                break
            budget *= 2
        assert comp is not None
        if not comp.text.strip():
            _emit_call(role, cfg.model, comp.tokens_in, comp.tokens_out, time.time() - t0, False,
                       error=f"empty response (finish_reason={comp.finish_reason})")
            raise LLMError(f"{role} model {cfg.model} returned an empty response")
        if config.LLM_CACHE and comp.finish_reason != "length":
            _cache_put(key, comp, cfg.model)

    with _usage_lock:
        _usage.calls += 1
        if cached:
            _usage.cached_calls += 1
        else:
            _usage.tokens_in += comp.tokens_in
            _usage.tokens_out += comp.tokens_out
    _emit_call(role, cfg.model, comp.tokens_in, comp.tokens_out, time.time() - t0, cached, cache_key=key[:12])
    return comp.text


def _emit_call(role, model, tin, tout, latency, cached, error: str | None = None, cache_key: str = "") -> None:
    ctx = _run_ctx.get()
    if not ctx:
        return
    run_id, stage = ctx
    payload = dict(role=role, model=model, tokens_in=tin, tokens_out=tout,
                   latency_s=round(latency, 3), cached=cached)
    if error:
        payload["error"] = error
    if cache_key:
        payload["cache_key"] = cache_key
    emit(run_id, stage, "llm_call", **payload)


# ---------------------------------------------------------------- JSON helpers
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n(.*?)```", re.DOTALL)


def _first_balanced(text: str, open_ch: str = "{", close_ch: str = "}") -> str | None:
    """Return the first balanced {...} substring, respecting JSON string literals."""
    start = text.find(open_ch)
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find(open_ch, start + 1)
    return None


def extract_json(text: str) -> dict:
    """Fenced ```json block first, else first balanced {...}. Raises ValueError."""
    candidates = [m.strip() for m in _FENCE_RE.findall(text)]
    bal = _first_balanced(text)
    if bal:
        candidates.append(bal)
    last_err: Exception | None = None
    for c in candidates:
        for s in (c, _first_balanced(c) or ""):
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError as e:
                last_err = e
    raise ValueError(f"no valid JSON object found in response ({last_err or 'no braces'})")


def _schema_instructions(schema: type[BaseModel]) -> str:
    return (
        "Respond with a single JSON object inside a ```json fenced code block, and nothing else "
        "after it. It must validate against this JSON Schema:\n"
        f"```json\n{json.dumps(schema.model_json_schema(), indent=1)}\n```"
    )


def complete_json(role: Role, messages: list[dict], schema: type[T], *,
                  temperature: float = 0.2, max_tokens: int = 4096) -> T:
    msgs = [dict(m) for m in messages]
    instr = _schema_instructions(schema)
    if msgs and msgs[-1]["role"] == "user":
        msgs[-1]["content"] = f"{msgs[-1]['content']}\n\n{instr}"
    else:
        msgs.append({"role": "user", "content": instr})

    text = complete(role, msgs, temperature=temperature, max_tokens=max_tokens)
    try:
        return schema.model_validate(extract_json(text))
    except (ValueError, ValidationError) as e:
        err = str(e)[:2000]

    repair = msgs + [
        {"role": "assistant", "content": text},
        {"role": "user", "content": f"Your previous response was invalid:\n{err}\n\n"
                                    f"Return ONLY the corrected JSON object in a ```json fence."},
    ]
    text2 = complete(role, repair, temperature=0.0, max_tokens=max_tokens)
    try:
        return schema.model_validate(extract_json(text2))
    except (ValueError, ValidationError) as e:
        raise LLMError(f"complete_json({schema.__name__}) failed after repair: {str(e)[:500]}") from e
