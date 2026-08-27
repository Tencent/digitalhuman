"""
Unified LLM client.

``call_LLM(prompt, model_name)`` sends ``model_name`` unchanged as the
Chat Completions ``model`` field. Configure the endpoint in ``.env``.

    response_text, sys_thinking = call_LLM(messages_or_prompt, model_name)
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

PromptType = Union[str, List[Dict[str, str]]]

TOKEN_COST: Dict[str, Dict[str, float]] = {}
LOG_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
MODEL_LOG_PATH = Path(os.environ.get("POET_LOG_DIR", "output/logs")) / f"model_calls_{LOG_TIMESTAMP}.log"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def get_token_cost_summary() -> Dict[str, Dict[str, float]]:
    return TOKEN_COST


def format_token_cost_summary() -> str:
    if not TOKEN_COST:
        return "No token usage recorded."
    lines = ["Token / cost summary:"]
    for model, stats in TOKEN_COST.items():
        lines.append(
            f"  - {model}: prompt={stats.get('prompt', 0)}, "
            f"completion={stats.get('completion', 0)}, cost={stats.get('cost', 0)}"
        )
    return "\n".join(lines)


def _normalize_messages(prompt: PromptType) -> List[Dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, list):
        return prompt
    raise TypeError(f"Unsupported prompt type: {type(prompt)}")


def _normalize_response(raw: str, current_sys_thinking: str = "none") -> Tuple[str, str]:
    response = (raw or "").strip()
    sys_thinking = current_sys_thinking or "none"
    if "<think>" in response and "</think>" in response:
        thinking_part, *rest = response.split("</think>", maxsplit=1)
        thinking_text = thinking_part.replace("<think>", "").strip()
        remaining = rest[0] if rest else ""
        response = remaining.strip() or response
        if sys_thinking in ("none", "", None):
            sys_thinking = thinking_text or sys_thinking
    return response, sys_thinking


def _log_interaction(model_name: str, prompt: PromptType, response: str, sys_thinking: str) -> None:
    if _env("POET_DISABLE_MODEL_LOG", "0") in {"1", "true", "True"}:
        return
    try:
        MODEL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        prompt_repr = json.dumps(prompt, ensure_ascii=False) if isinstance(prompt, (list, dict)) else str(prompt)
        with open(MODEL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(
                f"[{datetime.now().isoformat()}] model={model_name}\n"
                f"PROMPT: {prompt_repr}\n"
                f"RESPONSE: {response}\n"
                f"SYS_THINKING: {sys_thinking}\n"
                "---\n"
            )
    except Exception:
        pass


def _openai_client() -> Any:
    if OpenAI is None:
        raise RuntimeError("Please install openai: pip install openai")
    api_key = _env("POET_API_KEY") or _env("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set POET_API_KEY or OPENAI_API_KEY "
            "(copy .env.example -> .env)."
        )
    base_url = _env("POET_API_BASE") or _env("OPENAI_BASE_URL") or None
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def call_LLM(
    prompt: PromptType,
    model_name: str,
    thinking_pattern: str = "none-first",
    character: str = "NSP",
) -> Tuple[str, str]:
    """
    Project-wide LLM entrypoint.

    ``model_name`` is sent as-is in the request ``model`` field.

    Returns:
        (response_text, sys_thinking)
    """
    _ = (thinking_pattern, character)
    messages = _normalize_messages(prompt)
    model_id = (model_name or "").strip()
    if not model_id:
        raise ValueError("model_name is empty")

    retry_limit = int(_env("POET_RETRY_LIMIT", "5") or "5")
    last_error: Optional[Exception] = None

    for attempt in range(1, retry_limit + 1):
        try:
            client = _openai_client()
            completion = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=float(_env("POET_TEMPERATURE", "1.0") or "1.0"),
            )
            raw = completion.choices[0].message.content or ""
            response, sys_thinking = _normalize_response(raw)

            usage = getattr(completion, "usage", None)
            if usage is not None:
                stats = TOKEN_COST.setdefault(
                    model_name, {"prompt": 0.0, "completion": 0.0, "cost": 0.0}
                )
                stats["prompt"] += float(getattr(usage, "prompt_tokens", 0) or 0)
                stats["completion"] += float(getattr(usage, "completion_tokens", 0) or 0)

            _log_interaction(model_name, messages, response, sys_thinking)
            return response, sys_thinking
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(2 * attempt, 8))

    raise RuntimeError(f"call_LLM failed for model={model_name}: {last_error}")
