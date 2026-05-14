import json
import time
from pathlib import Path

import pytest

from core.agno_workflow import run_agno_pipeline


@pytest.fixture(autouse=True)
def force_local_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")


def test_agno_workflow_mirrors_graph_contract_with_state_db_logs_and_memory(tmp_path, monkeypatch):
    db_path = tmp_path / "workflows.db"
    memory_path = tmp_path / "memory.jsonl"
    log_path = tmp_path / "run.jsonl"
    monkeypatch.setenv("TRADEAGE_WORKFLOW_DB", str(db_path))
    monkeypatch.setenv("TRADEAGE_MEMORY_PATH", str(memory_path))

    result = run_agno_pipeline(
        "AAPL",
        start_date="2024-03-01",
        end_date="2024-04-15",
        data_provider="yfinance",
        log_path=log_path,
        tool_reports=False,
    )
    trace = result["trace"]

    assert trace["workflow"]["runtime"] == "agno"
    assert trace["workflow"]["checkpoint_db"] == str(db_path)
    assert db_path.exists()
    assert {"investment_debate_loop", "risk_debate_loop", "execution_condition"} <= set(
        trace["workflow"]["steps"]
    )
    assert {"market", "news", "sentiment", "fundamentals"} == set(trace["reports"])
    assert trace["state_snapshot"]["investment_debate_state"]["count"] == 2
    assert trace["state_snapshot"]["risk_debate_state"]["count"] == 3
    assert trace["portfolio_manager"]["action"] == trace["final_trader"]["action"]
    assert memory_path.exists()

    stages = [json.loads(line)["stage"] for line in log_path.read_text().splitlines()]
    assert stages[:4] == [
        "market_analyst_report",
        "news_analyst_report",
        "sentiment_analyst_report",
        "fundamentals_analyst_report",
    ]
    assert stages[-1] == "evaluation"


def test_tool_report_steps_are_called_when_enabled(monkeypatch, tmp_path):
    called = []

    def report(name):
        def inner(*args, **kwargs):
            called.append(name)
            return f"{name} report"

        return inner

    monkeypatch.setenv("TRADEAGE_WORKFLOW_DB", str(tmp_path / "workflows.db"))
    monkeypatch.setenv("TRADEAGE_MEMORY_PATH", str(tmp_path / "memory.jsonl"))
    monkeypatch.setattr("core.agno_workflow.run_market_analyst_report", report("market"))
    monkeypatch.setattr("core.agno_workflow.run_news_analyst_report", report("news"))
    monkeypatch.setattr("core.agno_workflow.run_sentiment_analyst_report", report("sentiment"))
    monkeypatch.setattr("core.agno_workflow.run_fundamentals_analyst_report", report("fundamentals"))

    run_agno_pipeline(
        "AAPL",
        start_date="2024-03-01",
        end_date="2024-04-15",
        data_provider="yfinance",
        log_path=tmp_path / "run.jsonl",
        tool_reports=True,
    )

    assert called == ["market", "news", "sentiment", "fundamentals"]


def test_nonlocal_analyst_reports_run_concurrently(monkeypatch):
    from core.agno_workflow import _analyst_reports_parallel

    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("TRADEAGE_PARALLEL_ANALYSTS", "1")
    monkeypatch.setenv("MISTRAL_API_KEY1", "key-one")
    monkeypatch.setenv("MISTRAL_API_KEY2", "key-two")

    def delayed(name):
        def inner(*args, **kwargs):
            time.sleep(0.2)
            return {"report": f"{name} report", "tool_calls": [], "react_steps": []}

        return inner

    monkeypatch.setattr("core.agno_workflow.run_market_analyst_report", delayed("market"))
    monkeypatch.setattr("core.agno_workflow.run_news_analyst_report", delayed("news"))
    monkeypatch.setattr("core.agno_workflow.run_sentiment_analyst_report", delayed("sentiment"))
    monkeypatch.setattr("core.agno_workflow.run_fundamentals_analyst_report", delayed("fundamentals"))

    session_state = {
        "symbol": "AAPL",
        "cash": 10000,
        "rows": [
            {"date": "2024-03-01", "close": 100},
            {"date": "2024-03-04", "close": 101},
        ],
        "market": {"date": "2024-03-04", "close": 101},
        "trade_date": "2024-03-04",
        "data_provider": "yfinance",
        "tool_reports": True,
        "tool_trace": [],
        "react_trace": [],
        "state": {"market_report": "", "investment_debate_state": {}, "risk_debate_state": {}},
        "log_path": None,
    }

    started = time.monotonic()
    _analyst_reports_parallel(None, session_state=session_state)
    elapsed = time.monotonic() - started

    assert elapsed < 0.55
    assert session_state["report_compression"]["market"]["full_chars"] == len("market report")
