import json
import time

import pytest

from agents.agno_runtime import AgnoLLMAdapter
from core.llm_observability import LLMCallTimeout, LLMProviderError, traced_llm_call


def _trace_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_llm_adapter_logs_successful_local_call(tmp_path, monkeypatch):
    trace_path = tmp_path / "llm_calls.jsonl"
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("TRADEAGE_LLM_TRACE_PATH", str(trace_path))

    response = AgnoLLMAdapter(name="Trace Test").invoke("say hold")

    rows = _trace_rows(trace_path)
    assert response.content
    assert rows[0]["status"] == "started"
    assert rows[-1]["status"] == "succeeded"
    assert rows[-1]["agent_name"] == "Trace Test"
    assert rows[-1]["provider"] == "local"
    assert rows[-1]["prompt_chars"] == len("say hold")
    assert rows[-1]["response_chars"] > 0


def test_llm_adapter_logs_failures(tmp_path, monkeypatch):
    trace_path = tmp_path / "llm_calls.jsonl"
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("TRADEAGE_LLM_TRACE_PATH", str(trace_path))

    class BrokenAgent:
        def run(self, prompt):
            raise RuntimeError("provider broke")

    adapter = AgnoLLMAdapter(name="Broken Agent")
    adapter.agent = BrokenAgent()

    with pytest.raises(RuntimeError, match="provider broke"):
        adapter.invoke("fail now")

    rows = _trace_rows(trace_path)
    assert rows[-1]["status"] == "failed"
    assert rows[-1]["error_type"] == "RuntimeError"
    assert "provider broke" in rows[-1]["error"]


def test_llm_adapter_turns_agno_error_content_into_logged_timeout(tmp_path, monkeypatch):
    trace_path = tmp_path / "llm_calls.jsonl"
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("TRADEAGE_LLM_TRACE_PATH", str(trace_path))

    class ErrorContent:
        content = "LLM call exceeded 60 seconds"

    class ErrorContentAgent:
        def run(self, prompt):
            return ErrorContent()

    adapter = AgnoLLMAdapter(name="Agno Error Agent")
    adapter.agent = ErrorContentAgent()

    with pytest.raises(LLMCallTimeout):
        adapter.invoke("slow")

    rows = _trace_rows(trace_path)
    assert rows[-1]["status"] == "timeout"
    assert rows[-1]["response_chars"] == len("LLM call exceeded 60 seconds")


def test_llm_adapter_turns_agno_429_content_into_logged_provider_error(tmp_path, monkeypatch):
    trace_path = tmp_path / "llm_calls.jsonl"
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("TRADEAGE_LLM_TRACE_PATH", str(trace_path))

    class ErrorContent:
        content = 'API error occurred: Status 429. Body: {"message":"Rate limit exceeded"}'

    class ErrorContentAgent:
        def run(self, prompt):
            return ErrorContent()

    adapter = AgnoLLMAdapter(name="Agno 429 Agent")
    adapter.agent = ErrorContentAgent()

    with pytest.raises(LLMProviderError):
        adapter.invoke("rate limited")

    rows = _trace_rows(trace_path)
    assert rows[-1]["status"] == "failed"
    assert rows[-1]["status_code"] == 429


def test_llm_adapter_retries_429_with_visible_trace(tmp_path, monkeypatch):
    trace_path = tmp_path / "llm_calls.jsonl"
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("TRADEAGE_LLM_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("TRADEAGE_LLM_RATE_LIMIT_RETRIES", "1")
    monkeypatch.setenv("TRADEAGE_LLM_RATE_LIMIT_BACKOFF_SECONDS", "0")

    class Response:
        def __init__(self, content):
            self.content = content

    class FlakyAgent:
        def __init__(self):
            self.calls = 0

        def run(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return Response('API error occurred: Status 429. Body: {"message":"Rate limit exceeded"}')
            return Response("retry succeeded")

    adapter = AgnoLLMAdapter(name="Retry Agent")
    adapter.agent = FlakyAgent()

    response = adapter.invoke("retry")

    rows = _trace_rows(trace_path)
    assert response.content == "retry succeeded"
    assert [row["status"] for row in rows] == ["started", "failed", "started", "succeeded"]
    assert rows[1]["status_code"] == 429


def test_llm_adapter_retries_transient_network_error_content(tmp_path, monkeypatch):
    trace_path = tmp_path / "llm_calls.jsonl"
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("TRADEAGE_LLM_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("TRADEAGE_LLM_RATE_LIMIT_RETRIES", "1")
    monkeypatch.setenv("TRADEAGE_LLM_RATE_LIMIT_BACKOFF_SECONDS", "0")

    class Response:
        def __init__(self, content):
            self.content = content

    class FlakyAgent:
        def __init__(self):
            self.calls = 0

        def run(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return Response("[Errno -3] Temporary failure in name resolution")
            return Response("network retry succeeded")

    adapter = AgnoLLMAdapter(name="Network Retry Agent")
    adapter.agent = FlakyAgent()

    response = adapter.invoke("retry network")

    rows = _trace_rows(trace_path)
    assert response.content == "network retry succeeded"
    assert [row["status"] for row in rows] == ["started", "failed", "started", "succeeded"]
    assert rows[1]["error_type"] == "LLMProviderError"


def test_traced_llm_call_times_out_and_logs(tmp_path, monkeypatch):
    trace_path = tmp_path / "llm_calls.jsonl"
    monkeypatch.setenv("TRADEAGE_LLM_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("TRADEAGE_LLM_CALL_TIMEOUT_SECONDS", "1")

    with pytest.raises(LLMCallTimeout):
        with traced_llm_call(
            agent_name="Slow Agent",
            provider="mistral",
            model="mistral-large-2512",
            prompt="slow prompt",
            call_site="test",
        ):
            time.sleep(2)

    rows = _trace_rows(trace_path)
    assert rows[-1]["status"] == "timeout"
    assert rows[-1]["agent_name"] == "Slow Agent"
    assert rows[-1]["timeout_seconds"] == 1


def test_mistral_model_receives_timeout_and_retry_settings(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("MISTRAL_QUICK_MODEL", "mistral-small-2603")
    monkeypatch.setenv("TRADEAGE_LLM_CALL_TIMEOUT_SECONDS", "77")
    monkeypatch.setenv("TRADEAGE_LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("TRADEAGE_LLM_MAX_TOKENS", "999")
    monkeypatch.setenv("TRADEAGE_QUICK_LLM_MAX_TOKENS", "999")

    adapter = AgnoLLMAdapter(name="Mistral Config")

    assert adapter.model.id == "mistral-small-2603"
    assert adapter.model.client_params["timeout_ms"] == 77000
    assert adapter.model.retries == 0
    assert adapter.model.max_tokens == 999


def test_mistral_role_specific_token_budgets(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.delenv("TRADEAGE_LLM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TRADEAGE_TRADER_LLM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TRADEAGE_RESEARCH_MANAGER_LLM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TRADEAGE_PORTFOLIO_MANAGER_LLM_MAX_TOKENS", raising=False)

    trader = AgnoLLMAdapter(kind="quick", name="trader")
    research_manager = AgnoLLMAdapter(kind="deep", name="research_manager")
    portfolio_manager = AgnoLLMAdapter(kind="deep", name="portfolio_manager")

    assert trader.model.max_tokens == 1800
    assert research_manager.model.max_tokens == 4500
    assert portfolio_manager.model.max_tokens == 5000


def test_mistral_role_specific_token_budget_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("TRADEAGE_ANALYST_LLM_MAX_TOKENS", "3333")
    monkeypatch.setenv("TRADEAGE_DEBATE_LLM_MAX_TOKENS", "3444")
    monkeypatch.setenv("TRADEAGE_PORTFOLIO_MANAGER_LLM_MAX_TOKENS", "5555")

    market = AgnoLLMAdapter(kind="quick", name="Market Analyst")
    bull = AgnoLLMAdapter(kind="quick", name="bull_researcher")
    portfolio_manager = AgnoLLMAdapter(kind="deep", name="portfolio_manager")

    assert market.model.max_tokens == 3333
    assert bull.model.max_tokens == 3444
    assert portfolio_manager.model.max_tokens == 5555
