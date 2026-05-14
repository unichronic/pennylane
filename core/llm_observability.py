from __future__ import annotations

import json
import re
import signal
import threading
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from config import get_config


class LLMCallTimeout(TimeoutError):
    pass


class LLMProviderError(RuntimeError):
    def __init__(self, message, status_code=None, retryable=False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


def raise_if_agno_error_content(content):
    text = "" if content is None else str(content).strip()
    lowered = text.lower()
    if lowered.startswith("llm call exceeded"):
        raise LLMCallTimeout(text)
    if "status 429" in lowered or "rate limit exceeded" in lowered:
        raise LLMProviderError(text, status_code=429, retryable=True)
    if "temporary failure in name resolution" in lowered or lowered.startswith("[errno "):
        raise LLMProviderError(text, retryable=True)
    if lowered.startswith("api error occurred"):
        status_code = None
        match = re.search(r"status\s+(\d{3})", lowered)
        if match:
            status_code = int(match.group(1))
        retryable = status_code is None or status_code in {408, 409, 425, 429, 500, 502, 503, 504}
        raise LLMProviderError(text, status_code=status_code, retryable=retryable)
    if lowered.startswith("error in agent run"):
        raise RuntimeError(text)


def llm_trace_path():
    return Path(get_config().get("llm_trace_path", "data_cache/llm_calls.jsonl"))


def _now():
    return datetime.now(UTC).isoformat()


def _compact(value, limit=1200):
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _status_code(exc):
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if value:
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        return getattr(response, "status_code", None) or getattr(response, "status", None)
    return None


def append_llm_trace(event):
    if not get_config().get("llm_trace_enabled", True):
        return
    path = llm_trace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


class traced_llm_call:
    def __init__(
        self,
        *,
        agent_name,
        provider,
        model,
        kind=None,
        key_slot=None,
        prompt=None,
        call_site=None,
        timeout_seconds=None,
    ):
        cfg = get_config()
        self.agent_name = agent_name
        self.provider = provider
        self.model = model
        self.kind = kind
        self.key_slot = key_slot
        self.prompt = "" if prompt is None else str(prompt)
        self.call_site = call_site
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else cfg.get("llm_call_timeout_seconds", 120)
        self.started = None
        self.response_text = ""
        self._previous_handler = None
        self._previous_timer = None

    def __enter__(self):
        self.started = perf_counter()
        append_llm_trace(self._event("started"))
        if self.timeout_seconds and hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread():
            self._previous_handler = signal.getsignal(signal.SIGALRM)
            self._previous_timer = signal.setitimer(signal.ITIMER_REAL, float(self.timeout_seconds))
            signal.signal(signal.SIGALRM, self._timeout_handler)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._clear_alarm()
        if exc is None:
            append_llm_trace(self._event("succeeded"))
            return False
        status = "timeout" if isinstance(exc, LLMCallTimeout) else "failed"
        append_llm_trace(
            self._event(
                status,
                error_type=type(exc).__name__,
                error=_compact(exc),
                status_code=_status_code(exc),
            )
        )
        return False

    def set_response(self, response):
        self.response_text = "" if response is None else str(response)

    def _timeout_handler(self, signum, frame):
        raise LLMCallTimeout(f"LLM call exceeded {self.timeout_seconds} seconds")

    def _clear_alarm(self):
        if self._previous_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._previous_handler)
            if self._previous_timer and self._previous_timer[0] > 0:
                remaining, interval = self._previous_timer
                signal.setitimer(signal.ITIMER_REAL, remaining, interval)
            self._previous_handler = None
            self._previous_timer = None

    def _event(self, status, **extra):
        elapsed = 0 if self.started is None else perf_counter() - self.started
        event = {
            "timestamp": _now(),
            "status": status,
            "agent_name": self.agent_name,
            "provider": self.provider,
            "model": self.model,
            "kind": self.kind,
            "key_slot": self.key_slot,
            "call_site": self.call_site,
            "timeout_seconds": self.timeout_seconds,
            "duration_seconds": round(elapsed, 3),
            "prompt_chars": len(self.prompt),
            "prompt_preview": _compact(self.prompt, 500),
            "response_chars": len(self.response_text),
        }
        event.update({key: value for key, value in extra.items() if value is not None})
        return event
