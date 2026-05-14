import json
from datetime import date
from pathlib import Path

import pytest

from backtest.engine import run_backtest
from core.pipeline import run_pipeline
from main import resolve_live_date_range

REAL_SYMBOL = "AAPL"
REAL_START = "2024-03-01"
REAL_END = "2024-04-15"


@pytest.fixture(autouse=True)
def force_local_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")


def test_project_architecture_contract_exists():
    required = [
        "agents/analyst.py",
        "agents/bull_researcher.py",
        "agents/bear_researcher.py",
        "agents/agno_runtime.py",
        "agents/llm_factory.py",
        "agents/research_manager.py",
        "agents/trader.py",
        "agents/aggressive_debator.py",
        "agents/conservative_debator.py",
        "agents/neutral_debator.py",
        "agents/risk_debate.py",
        "agents/risk_manager.py",
        "agents/final_decision.py",
        "agents/portfolio_manager.py",
        "agents/rating.py",
        "agents/analyst_reports.py",
        "core/agno_workflow.py",
        "core/pipeline.py",
        "core/state.py",
        "core/workflow_state.py",
        "core/reflection.py",
        "core/logger.py",
        "core/conditional_logic.py",
        "core/signal_processing.py",
        "data/loader.py",
        "data/y_finance.py",
        "data/indicators.py",
        "data/trading_tools.py",
        "backtest/engine.py",
        "backtest/paper_methodology.py",
        "config.py",
        ".env.example",
    ]

    for path in required:
        assert Path(path).exists(), path


def test_live_date_range_uses_tomorrow_as_default_exclusive_end():
    start_date, end_date = resolve_live_date_range(today=date(2026, 5, 4))

    assert end_date == "2026-05-05"
    assert start_date == "2026-01-05"


def test_project_pipeline_contract_definitive(tmp_path):
    log_path = tmp_path / "contract.jsonl"
    result = run_pipeline(REAL_SYMBOL, cash=10000, log_path=log_path, start_date=REAL_START, end_date=REAL_END)

    stages = [json.loads(x)["stage"] for x in log_path.read_text().splitlines()]
    assert stages == [
        "market_analyst_report",
        "news_analyst_report",
        "sentiment_analyst_report",
        "fundamentals_analyst_report",
        "analyst",
        "debate",
        "trader",
        "risk_debate",
        "portfolio_manager",
        "risk",
        "execution",
        "evaluation",
    ]
    assert result["trace"]["analyst"].keys() == {"trend", "confidence", "signals", "summary"}
    assert result["trace"]["workflow"]["runtime"] == "agno"
    assert "market_analyst_report" in result["trace"]["workflow"]["steps"]
    assert "execution_condition" in result["trace"]["workflow"]["steps"]
    assert "investment_debate_loop" in result["trace"]["workflow"]["steps"]
    assert "risk_debate_loop" in result["trace"]["workflow"]["steps"]
    assert result["trace"]["state_snapshot"]["symbol"] == REAL_SYMBOL
    assert result["trace"]["reports"].keys() == {"market", "news", "sentiment", "fundamentals"}
    assert result["trace"]["debate"].keys() == {"bull_case", "bear_case", "key_risks", "consensus_bias"}
    assert result["trace"]["trader"].keys() == {"action", "confidence", "reasoning", "position_size"}
    assert result["trace"]["portfolio_manager"].keys() == {"rating", "action", "final_trade_decision"}
    assert result["trace"]["risk"].keys() == {"approved", "adjusted_position", "stop_loss", "risk_notes"}
    assert result["trace"]["investment_debate_state"]["count"] == 2
    assert result["trace"]["risk_debate_state"]["count"] == 3
    assert result["trace"]["execution"]["trade"] is None or result["portfolio"]["trades"]
    assert len(result["portfolio"]["equity_curve"]) >= 2


def test_project_backtest_contract_definitive(tmp_path):
    result = run_backtest(REAL_SYMBOL, cash=10000, log_path=tmp_path / "bt.jsonl", start_date=REAL_START, end_date=REAL_END)

    assert result["metrics"].keys() == {
        "cumulative_return_pct",
        "annualized_return_pct",
        "sharpe_ratio",
        "max_drawdown_pct",
    }
    assert result["metrics"]["cumulative_return_pct"] != 0
    assert result["trades"]
    assert all("risk_debate_state" in x for x in result["trace"])
