"""Loads .env and exposes per-role LLM configs + global settings."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

Role = Literal["planner", "coder"]

DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
RUNS_DIR = ROOT / "runs"


@dataclass(frozen=True)
class RoleConfig:
    base_url: str
    api_key: str
    model: str
    reasoning_effort: str = ""  # optional: low|medium|high for thinking models (OpenAI-compat param)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_DEFAULT_MODEL = "gemini-3-flash-preview"


def role_config(role: Role) -> RoleConfig:
    """Role settings from <ROLE>_* env vars; unset fields fall back to Gemini if GEMINI_API_KEY is set."""
    p = role.upper()
    base_url = _env(f"{p}_BASE_URL")
    api_key = _env(f"{p}_API_KEY")
    model = _env(f"{p}_MODEL")
    gemini_key = _env("GEMINI_API_KEY")
    if gemini_key and (not base_url or base_url == GEMINI_BASE_URL):
        base_url = base_url or GEMINI_BASE_URL
        api_key = api_key or gemini_key
        model = model or GEMINI_DEFAULT_MODEL
    if not (base_url and api_key and model):
        raise RuntimeError(
            f"LLM role '{role}' not configured: set {p}_BASE_URL, {p}_API_KEY, {p}_MODEL in .env"
        )
    return RoleConfig(base_url=base_url, api_key=api_key, model=model,
                      reasoning_effort=_env(f"{p}_REASONING_EFFORT"))


LLM_CACHE: bool = _env("LLM_CACHE", "1") == "1"
LLM_TIMEOUT_S: int = int(_env("LLM_TIMEOUT_S", "120"))
MAX_ATTEMPTS: int = int(_env("MAX_ATTEMPTS", "3"))
# minimum max_tokens sent to the provider (thinking models consume hidden tokens from this budget)
LLM_MAX_TOKENS_FLOOR: int = int(_env("LLM_MAX_TOKENS_FLOOR", "16384"))
