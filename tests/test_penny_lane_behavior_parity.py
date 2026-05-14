import json
from pathlib import Path

import pandas as pd
import pytest
from types import SimpleNamespace

from agents.analyst_reports import run_market_analyst_report, run_news_analyst_report
from backtest.paper_methodology import PaperBacktestConfig, aggregate_experiment_results
from core.agno_workflow import run_agno_pipeline
from core.checkpointing import checkpoint_path, clear_checkpoint, load_checkpoint, save_checkpoint
from core.reflection import (
    load_lessons,
    load_memory_entries,
    memory_vector_path,
    record_decision,
    reflect_outcomes,
    semantic_search_lessons,
)
from data import trading_tools


@pytest.fixture(autouse=True)
def force_local_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")


def test_workflow_exposes_penny_lane_behavior_traces_and_full_state_log(tmp_path, monkeypatch):
    db_path = tmp_path / "workflows.db"
    memory_path = tmp_path / "memory.jsonl"
    results_dir = tmp_path / "results"
    monkeypatch.setenv("TRADEAGE_WORKFLOW_DB", str(db_path))
    monkeypatch.setenv("TRADEAGE_MEMORY_PATH", str(memory_path))
    monkeypatch.setenv("TRADEAGE_RESULTS_DIR", str(results_dir))

    result = run_agno_pipeline(
        "AAPL",
        start_date="2024-03-01",
        end_date="2024-04-15",
        data_provider="yfinance",
        tool_reports=False,
        workflow_session_id="behavior-parity-session",
    )
    trace = result["trace"]

    assert any(item["from"] == "Bull Researcher" for item in trace["routing_trace"])
    assert any(item["from"] == "Bear Researcher" for item in trace["routing_trace"])
    assert any(item["from"] == "Aggressive Analyst" for item in trace["routing_trace"])
    assert Path(trace["full_state_log"]).exists()
    full_state = json.loads(Path(trace["full_state_log"]).read_text())
    assert full_state["company_of_interest"] == "AAPL"
    assert "investment_debate_state" in full_state
    assert "final_trade_decision" in full_state
    assert not checkpoint_path("behavior-parity-session").exists()


def test_report_tool_trace_uses_penny_lane_tool_surface(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADEAGE_WORKFLOW_DB", str(tmp_path / "workflows.db"))
    monkeypatch.setenv("TRADEAGE_MEMORY_PATH", str(tmp_path / "memory.jsonl"))
    monkeypatch.setenv("TRADEAGE_RESULTS_DIR", str(tmp_path / "results"))

    def fake_report(name, tools):
        return {
            "report": f"{name} report",
            "tool_calls": [{"tool": tool, "args": {}} for tool in tools],
        }

    monkeypatch.setattr(
        "core.agno_workflow.run_market_analyst_report",
        lambda *args, **kwargs: fake_report("market", ["get_stock_data", "get_indicators"]),
    )
    monkeypatch.setattr(
        "core.agno_workflow.run_sentiment_analyst_report",
        lambda *args, **kwargs: fake_report("social", ["get_news", "sentiment_from_news"]),
    )
    monkeypatch.setattr(
        "core.agno_workflow.run_news_analyst_report",
        lambda *args, **kwargs: fake_report("news", ["get_news", "get_global_news", "get_insider_transactions"]),
    )
    monkeypatch.setattr(
        "core.agno_workflow.run_fundamentals_analyst_report",
        lambda *args, **kwargs: fake_report(
            "fundamentals",
            ["get_fundamentals", "get_balance_sheet", "get_cashflow", "get_income_statement"],
        ),
    )

    result = run_agno_pipeline(
        "AAPL",
        start_date="2024-03-01",
        end_date="2024-04-15",
        data_provider="yfinance",
        tool_reports=True,
    )
    tools = {item["tool"] for item in result["trace"]["tool_trace"]}

    assert {
        "get_stock_data",
        "get_indicators",
        "get_news",
        "sentiment_from_news",
        "get_global_news",
        "get_insider_transactions",
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
    } <= tools
    assert "react_trace" in result["trace"]


def test_checkpoint_save_load_and_clear_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEAGE_DATA_CACHE_DIR", str(tmp_path))
    session_id = "AAPL-2024-03-01"

    save_checkpoint(session_id, "analyst", {"symbol": "AAPL"}, {"report": "ok"})
    checkpoint = load_checkpoint(session_id)

    assert checkpoint["completed"] == ["analyst"]
    assert checkpoint["session_state"]["symbol"] == "AAPL"
    assert checkpoint["outputs"]["analyst"]["report"] == "ok"

    clear_checkpoint(session_id)
    assert not checkpoint_path(session_id).exists()


def test_memory_reflection_records_raw_and_alpha_returns(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEAGE_MEMORY_PATH", str(tmp_path / "memory.jsonl"))
    rows = [
        {"date": "2024-01-01", "close": 100},
        {"date": "2024-01-02", "close": 102},
        {"date": "2024-01-03", "close": 104},
        {"date": "2024-01-04", "close": 106},
        {"date": "2024-01-05", "close": 108},
        {"date": "2024-01-08", "close": 110},
    ]
    spy = [
        {"date": "2024-01-01", "close": 100},
        {"date": "2024-01-02", "close": 101},
        {"date": "2024-01-03", "close": 102},
        {"date": "2024-01-04", "close": 103},
        {"date": "2024-01-05", "close": 104},
        {"date": "2024-01-08", "close": 105},
    ]

    record_decision(
        "AAPL",
        "2024-01-01",
        "Buy",
        "buy",
        100,
        final_decision="Buy because momentum is improving.",
        analyst={"trend": "bullish"},
        debate={"consensus_bias": "bullish"},
        reports={"market": "market report"},
    )
    updates = reflect_outcomes("AAPL", rows, benchmark_rows=spy, holding_days=5)

    assert updates
    assert updates[0]["raw_return"] == pytest.approx(0.10)
    assert updates[0]["benchmark_return"] == pytest.approx(0.05)
    assert updates[0]["alpha_return"] == pytest.approx(0.05)
    lessons = load_lessons("AAPL").lower()
    assert "past analyses of aapl" in lessons
    assert "patterns that generated positive alpha" in lessons


def test_memory_reflection_scores_hold_as_no_new_position(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEAGE_MEMORY_PATH", str(tmp_path / "memory.jsonl"))
    rows = [
        {"date": "2024-01-01", "close": 100},
        {"date": "2024-01-02", "close": 98},
        {"date": "2024-01-03", "close": 96},
    ]
    spy = [
        {"date": "2024-01-01", "close": 100},
        {"date": "2024-01-02", "close": 99},
        {"date": "2024-01-03", "close": 98},
    ]

    record_decision("AAPL", "2024-01-01", "Hold", "hold", 100)
    updates = reflect_outcomes("AAPL", rows, benchmark_rows=spy, holding_days=2)

    assert updates
    assert updates[0]["raw_return"] == 0
    assert updates[0]["benchmark_return"] == pytest.approx(-0.02)
    assert updates[0]["alpha_return"] == pytest.approx(0.02)


def test_memory_depth_includes_cross_ticker_and_negative_alpha_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEAGE_MEMORY_PATH", str(tmp_path / "memory.jsonl"))
    dates = [f"2024-01-0{idx}" for idx in range(1, 7)]
    rows = [{"date": date, "close": 100 + idx} for idx, date in enumerate(dates)]
    falling = [{"date": date, "close": 100 - idx} for idx, date in enumerate(dates)]
    flat_spy = [{"date": date, "close": 100} for date in dates]

    record_decision("MSFT", "2024-01-01", "Sell", "sell", 100, final_decision="Sell weak trend", analyst={"trend": "bearish"})
    reflect_outcomes("MSFT", rows, benchmark_rows=flat_spy, holding_days=5)
    record_decision("AAPL", "2024-01-01", "Buy", "buy", 100, final_decision="Buy weakly", analyst={"trend": "bullish"})
    reflect_outcomes("AAPL", falling, benchmark_rows=flat_spy, holding_days=5)

    lessons = load_lessons("AAPL")

    assert "Recent cross-ticker lessons" in lessons
    assert "Patterns that generated negative alpha" in lessons
    assert "Decision:" in lessons


def test_memory_embedding_index_retrieves_semantic_lessons(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEAGE_MEMORY_PATH", str(tmp_path / "memory.jsonl"))
    monkeypatch.setenv("TRADEAGE_MEMORY_VECTOR_PATH", str(tmp_path / "memory.vectors.jsonl"))
    monkeypatch.setenv("TRADEAGE_MEMORY_EMBEDDING_PROVIDER", "local")
    dates = [f"2024-01-0{idx}" for idx in range(1, 7)]
    up_rows = [{"date": date, "close": 100 + idx} for idx, date in enumerate(dates)]
    flat_spy = [{"date": date, "close": 100} for date in dates]

    record_decision(
        "NVDA",
        "2024-01-01",
        "Buy",
        "buy",
        100,
        final_decision="Buy because accelerator demand, data center orders, and AI margins are improving.",
        analyst={"trend": "bullish", "summary": "AI accelerator momentum"},
        reports={"market": "semiconductor momentum and data center demand"},
    )
    reflect_outcomes("NVDA", up_rows, benchmark_rows=flat_spy, holding_days=5)

    matches = semantic_search_lessons("AAPL", query_context="AI accelerator data center semiconductor margins", limit=1)
    lessons = load_lessons("AAPL", query_context="AI accelerator data center semiconductor margins")

    assert memory_vector_path().exists()
    assert matches[0]["symbol"] == "NVDA"
    assert matches[0]["memory_similarity"] > 0
    assert "Semantically similar memory lessons" in lessons
    assert "NVDA" in lessons


def test_mistral_memory_embeddings_fail_loudly_without_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEAGE_MEMORY_PATH", str(tmp_path / "memory.jsonl"))
    monkeypatch.setenv("TRADEAGE_MEMORY_VECTOR_PATH", str(tmp_path / "memory.vectors.jsonl"))
    monkeypatch.setenv("TRADEAGE_MEMORY_EMBEDDING_PROVIDER", "mistral")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    for idx in range(1, 17):
        monkeypatch.delenv(f"MISTRAL_API_KEY{idx}", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEYS", raising=False)
    dates = [f"2024-01-0{idx}" for idx in range(1, 7)]
    rows = [{"date": date, "close": 100 + idx} for idx, date in enumerate(dates)]

    record_decision("AAPL", "2024-01-01", "Buy", "buy", 100)

    with pytest.raises(RuntimeError, match="Missing Mistral API key for memory embeddings"):
        reflect_outcomes("AAPL", rows, holding_days=5)


def test_memory_rotation_keeps_pending_and_recent_resolved(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEAGE_MEMORY_PATH", str(tmp_path / "memory.jsonl"))
    monkeypatch.setenv("TRADEAGE_MEMORY_MAX_ENTRIES", "1")
    dates = [f"2024-01-0{idx}" for idx in range(1, 7)]
    rows = [{"date": date, "close": 100 + idx} for idx, date in enumerate(dates)]

    record_decision("AAPL", "2024-01-01", "Buy", "buy", 100)
    reflect_outcomes("AAPL", rows, holding_days=5)
    record_decision("MSFT", "2024-01-01", "Buy", "buy", 100)
    reflect_outcomes("MSFT", rows, holding_days=5)
    record_decision("GOOGL", "2024-01-01", "Hold", "hold", 100)

    entries = load_memory_entries()

    assert len([item for item in entries if item.get("outcome")]) == 1
    assert any(item["symbol"] == "GOOGL" and not item.get("outcome") for item in entries)


def test_missing_financial_statement_tools_return_real_yfinance_tables(monkeypatch):
    frame = pd.DataFrame({"2023-12-31": [1, 2], "2024-06-30": [3, 4]}, index=["Cash", "Debt"])

    class FakeTicker:
        balance_sheet = frame
        cashflow = frame
        financials = frame

    monkeypatch.setattr(trading_tools.yf, "Ticker", lambda symbol: FakeTicker())
    monkeypatch.setattr(trading_tools, "yf_retry", lambda fn: fn())

    balance_sheet = trading_tools.get_balance_sheet("AAPL", curr_date="2024-01-01")
    cashflow = trading_tools.get_cashflow("AAPL", curr_date="2024-01-01")
    income_statement = trading_tools.get_income_statement("AAPL", curr_date="2024-01-01")

    assert "2023-12-31" in balance_sheet
    assert "2024-06-30" not in balance_sheet
    assert "Debt" in cashflow
    assert "Cash" in income_statement


def test_alpha_vantage_tool_vendor_filters_statement_reports(monkeypatch):
    monkeypatch.setenv("TRADEAGE_FUNDAMENTALS_VENDOR", "alpha_vantage")
    calls = []

    def fake_request(function_name, params):
        calls.append((function_name, params))
        return {
            "quarterlyReports": [
                {"fiscalDateEnding": "2023-12-31", "cash": "1"},
                {"fiscalDateEnding": "2024-06-30", "cash": "2"},
            ]
        }

    monkeypatch.setattr(trading_tools, "_alpha_vantage_request", fake_request)

    result = trading_tools.get_balance_sheet("AAPL", curr_date="2024-01-01")

    assert calls == [("BALANCE_SHEET", {"symbol": "AAPL"})]
    assert result["quarterlyReports"] == [{"fiscalDateEnding": "2023-12-31", "cash": "1"}]


def test_news_report_returns_react_steps(monkeypatch):
    monkeypatch.setattr("agents.analyst_reports.get_news", lambda *args, **kwargs: "company news")
    monkeypatch.setattr("agents.analyst_reports.get_global_news", lambda *args, **kwargs: "global news")
    monkeypatch.setattr("agents.analyst_reports.get_insider_transactions", lambda *args, **kwargs: "insider rows")

    result = run_news_analyst_report("AAPL", curr_date="2024-01-05")

    assert [step["action"] for step in result["react_steps"]] == [
        "get_news",
        "get_global_news",
        "get_insider_transactions",
    ]
    assert "Thought 1" in result["observations"]


def test_model_selected_analyst_tools_use_agno_agent(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("TRADEAGE_ANALYST_TOOL_MODE", "model")
    monkeypatch.setattr("agents.analyst_reports.create_agno_model", lambda *args, **kwargs: object())

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            self.tools = kwargs["tools"]

        def run(self, prompt):
            first = self.tools[0]()
            return SimpleNamespace(
                content=f"final report from model after {first}",
                tools=[
                    SimpleNamespace(
                        tool_name=self.tools[0].__name__,
                        tool_args={},
                        result=first,
                    )
                ],
            )

    monkeypatch.setattr("agents.analyst_reports.Agent", FakeAgent)
    monkeypatch.setattr("agents.analyst_reports.get_stock_data", lambda *args, **kwargs: "stock rows")
    monkeypatch.setattr("agents.analyst_reports.get_indicators", lambda *args, **kwargs: {"rsi": 51})

    result = run_market_analyst_report("AAPL", "2024-01-01", "2024-02-01")

    assert result["report"] == "final report from model after stock rows"
    assert result["tool_calls"] == [{"tool": "get_stock_data", "args": {}}]
    assert result["react_steps"][0]["thought"] == "Model selected get_stock_data for evidence gathering."


def test_paper_experiment_aggregate_combines_equal_capital_curves():
    results = {
        "AAPL": {
            "PennyLaneCapital": {"account": {"equity_curve": [100, 110, 120]}},
            "macd": {"account": {"equity_curve": [100, 90, 95]}},
        },
        "GOOGL": {
            "PennyLaneCapital": {"account": {"equity_curve": [100, 105, 115]}},
            "macd": {"account": {"equity_curve": [100, 100, 110]}},
        },
    }

    aggregate = aggregate_experiment_results(results, config=PaperBacktestConfig(initial_cash=100))

    assert aggregate["PennyLaneCapital"]["symbols"] == 2
    assert aggregate["PennyLaneCapital"]["equity_curve"] == [200, 215, 235]
    assert aggregate["PennyLaneCapital"]["metrics"]["cumulative_return_pct"] == 17.5


def test_news_filter_excludes_articles_after_current_date(monkeypatch):
    class FakeTicker:
        news = [
            {"content": {"title": "Future", "provider": {"displayName": "X"}, "pubDate": "2024-02-10T00:00:00Z"}},
            {"content": {"title": "Current", "provider": {"displayName": "X"}, "pubDate": "2024-01-05T00:00:00Z"}},
            {"content": {"title": "Old", "provider": {"displayName": "X"}, "pubDate": "2023-12-20T00:00:00Z"}},
        ]

    monkeypatch.setattr(trading_tools.yf, "Ticker", lambda symbol: FakeTicker())
    monkeypatch.setattr(trading_tools, "yf_retry", lambda fn: fn())

    report = trading_tools.get_news("AAPL", curr_date="2024-01-05", look_back_days=7)

    assert "Current" in report
    assert "Future" not in report
    assert "Old" not in report


def test_yfinance_news_error_is_reported_not_fallback(monkeypatch):
    class FakeTicker:
        @property
        def news(self):
            raise RuntimeError("provider down")

    monkeypatch.setattr(trading_tools.yf, "Ticker", lambda symbol: FakeTicker())

    report = trading_tools.get_news("AAPL", curr_date="2024-01-05")

    assert report == "Error fetching yfinance news for AAPL: provider down"
