import pytest

from agents.analyst_reports import run_market_analyst_report, run_sentiment_analyst_report
from backtest.paper_methodology import (
    PaperBacktestConfig,
    execute_paper_signal,
    run_penny_lane_paper_backtest,
)
from core.agno_workflow import _load_market, _market_analyst_report, _risk_gate
from core.state import execute_trade
from data import trading_tools
from data.market_data import load_market_ohlcv


@pytest.fixture(autouse=True)
def force_local_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")


def test_market_analyst_report_uses_preloaded_backtest_window(monkeypatch):
    calls = []
    rows = [
        {"date": "2024-03-01", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100},
        {"date": "2024-03-04", "open": 11, "high": 12, "low": 10, "close": 11, "volume": 110},
    ]

    def fail_load(*args, **kwargs):
        raise AssertionError("market report should not refetch future provider rows")

    monkeypatch.setattr("data.trading_tools.load_ohlcv", fail_load)
    result = run_market_analyst_report(
        "AAPL",
        "2024-03-01",
        "2024-03-05",
        provider="alpha_vantage",
        rows=rows,
    )

    calls.extend(result["tool_calls"])
    assert "2024-03-04" in result["observations"]
    assert "2024-03-05," not in result["observations"]
    assert calls[0]["args"]["provider"] == "alpha_vantage"


def test_news_strict_historical_mode_excludes_undated_articles(monkeypatch):
    class FakeTicker:
        news = [
            {"content": {"title": "Undated", "provider": {"displayName": "X"}}},
            {"content": {"title": "Current", "provider": {"displayName": "X"}, "pubDate": "2024-01-05T00:00:00Z"}},
        ]

    monkeypatch.setattr(trading_tools.yf, "Ticker", lambda symbol: FakeTicker())
    monkeypatch.setattr(trading_tools, "yf_retry", lambda fn: fn())

    report = trading_tools.get_news("AAPL", curr_date="2024-01-05", look_back_days=7)

    assert "Current" in report
    assert "Undated" not in report


def test_insider_transactions_filters_by_explicit_transaction_date(monkeypatch):
    class FakeTicker:
        insider_transactions = trading_tools.pd.DataFrame(
            {
                "Value": [20260101, 1000],
                "Start Date": ["2026-01-01", "2024-05-01"],
                "Transaction": ["Future sale", "Current sale"],
            }
        )

    monkeypatch.setattr(trading_tools.yf, "Ticker", lambda symbol: FakeTicker())
    monkeypatch.setattr(trading_tools, "yf_retry", lambda fn: fn())

    report = trading_tools.get_insider_transactions("AAPL", curr_date="2024-05-01")

    assert "Current sale" in report
    assert "Future sale" not in report


def test_insider_transactions_fail_closed_without_date_column(monkeypatch):
    class FakeTicker:
        insider_transactions = trading_tools.pd.DataFrame(
            {
                "Value": [20260101],
                "Transaction": ["Ambiguous future row"],
            }
        )

    monkeypatch.setattr(trading_tools.yf, "Ticker", lambda symbol: FakeTicker())
    monkeypatch.setattr(trading_tools, "yf_retry", lambda fn: fn())

    report = trading_tools.get_insider_transactions("AAPL", curr_date="2024-05-01")

    assert "No yfinance insider transactions returned for AAPL by 2024-05-01" in report
    assert "Ambiguous future row" not in report


def test_fundamentals_and_statements_label_timing(monkeypatch):
    class FakeTicker:
        def get_info(self):
            return {"longName": "Apple", "marketCap": 1}

    monkeypatch.setattr(trading_tools.yf, "Ticker", lambda symbol: FakeTicker())
    monkeypatch.setattr(trading_tools, "yf_retry", lambda fn: fn())

    fundamentals = trading_tools.get_fundamentals("AAPL", curr_date="2024-01-05")

    assert "Data timing: current_snapshot" in fundamentals


def test_alpha_vantage_adjusts_ohlc_to_adjusted_close(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test")

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "Time Series (Daily)": {
                    "2024-03-01": {
                        "1. open": "100",
                        "2. high": "110",
                        "3. low": "90",
                        "4. close": "100",
                        "5. adjusted close": "50",
                        "6. volume": "1000",
                    },
                    "2024-03-04": {
                        "1. open": "50",
                        "2. high": "55",
                        "3. low": "45",
                        "4. close": "50",
                        "5. adjusted close": "50",
                        "6. volume": "1000",
                    },
                }
            }

    monkeypatch.setattr("data.market_data.requests.get", lambda *args, **kwargs: FakeResponse())

    rows = load_market_ohlcv("AAPL", "2024-03-01", "2024-03-05", provider="alpha_vantage")

    assert rows[0]["open"] == 50
    assert rows[0]["high"] == 55
    assert rows[0]["low"] == 45
    assert rows[0]["close"] == 50


def test_paper_backtest_passes_signed_short_state_to_workflow(monkeypatch):
    rows = [
        {"date": "2024-01-01", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000},
        {"date": "2024-01-02", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000},
        {"date": "2024-01-03", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000},
    ]
    seen_shares = []
    seen_cost_basis = []

    monkeypatch.setattr("backtest.paper_methodology.load_ohlcv", lambda *args, **kwargs: rows)

    def fake_workflow(*args, **kwargs):
        seen_shares.append(kwargs["initial_portfolio"]["shares"])
        seen_cost_basis.append(kwargs["initial_portfolio"]["cost_basis"])
        action = "sell" if len(seen_shares) == 1 else "hold"
        return {
            "trace": {
                "workflow": {"runtime": "agno", "steps": []},
                "reports": {},
                "analyst": {},
                "debate": {},
                "investment_debate_state": {},
                "investment_plan": "",
                "trader": {},
                "risk": {"approved": True},
                "portfolio_manager": {"action": action},
                "final_trader": {"action": action},
                "risk_debate_state": {},
            }
        }

    monkeypatch.setattr("backtest.paper_methodology.run_agno_pipeline", fake_workflow)

    run_penny_lane_paper_backtest("AAPL", start_date="2024-01-01", end_date="2024-01-04")

    assert seen_shares[0] == 0
    assert seen_shares[1] < 0
    assert seen_cost_basis[1] == 100


def test_paper_backtest_decision_cadence_marks_to_market_between_workflows(monkeypatch):
    rows = [
        {"date": "2024-01-01", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000},
        {"date": "2024-01-02", "open": 95, "high": 96, "low": 94, "close": 95, "volume": 1000},
        {"date": "2024-01-03", "open": 90, "high": 91, "low": 89, "close": 90, "volume": 1000},
        {"date": "2024-01-04", "open": 85, "high": 86, "low": 84, "close": 85, "volume": 1000},
    ]
    calls = []

    monkeypatch.setattr("backtest.paper_methodology.load_ohlcv", lambda *args, **kwargs: rows)

    def fake_workflow(*args, **kwargs):
        calls.append(kwargs["trade_date"])
        return {
            "trace": {
                "workflow": {"runtime": "agno", "steps": []},
                "reports": {},
                "analyst": {},
                "debate": {},
                "investment_debate_state": {},
                "investment_plan": "",
                "trader": {},
                "risk": {"approved": True},
                "portfolio_manager": {"action": "sell"},
                "final_trader": {"action": "sell"},
                "risk_debate_state": {},
            }
        }

    monkeypatch.setattr("backtest.paper_methodology.run_agno_pipeline", fake_workflow)

    result = run_penny_lane_paper_backtest(
        "AAPL",
        start_date="2024-01-01",
        end_date="2024-01-05",
        config=PaperBacktestConfig(initial_cash=10000, decision_cadence_days=2),
    )

    assert calls == ["2024-01-01", "2024-01-03"]
    assert [item["decision_skipped"] for item in result["trace"]] == [False, True, False, True]
    assert [item["signal"] for item in result["trace"]] == ["sell", "hold", "sell", "hold"]
    assert result["account"]["equity_curve"] == pytest.approx([
        10000,
        10000,
        10500,
        11000,
        11611.111111111111,
    ])


def test_workflow_market_report_reuses_loaded_rows_even_without_preload(monkeypatch):
    rows = [
        {"date": "2024-03-01", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100},
        {"date": "2024-03-04", "open": 11, "high": 12, "low": 10, "close": 11, "volume": 110},
    ]
    seen = {}

    def fake_report(symbol, start_date, end_date, **kwargs):
        seen.update({
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "rows": kwargs["rows"],
        })
        return {"report": "market", "tool_calls": [], "react_steps": []}

    monkeypatch.setattr("core.agno_workflow.run_market_analyst_report", fake_report)
    session_state = {
        "symbol": "AAPL",
        "cash": 10000,
        "start_date": "2024-01-01",
        "end_date": "2024-04-01",
        "trade_date": "2024-03-04",
        "rows": rows,
        "market": rows[-1],
        "state": {"investment_debate_state": {}, "risk_debate_state": {}},
        "tool_reports": True,
        "log_path": None,
    }

    _market_analyst_report(None, session_state=session_state)

    assert seen["rows"] is rows
    assert seen["start_date"] == "2024-03-01"
    assert seen["end_date"] == "2024-03-05"


def test_preloaded_workflow_benchmark_uses_loaded_row_window(monkeypatch):
    rows = [
        {"date": "2024-03-01", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100},
        {"date": "2024-03-04", "open": 11, "high": 12, "low": 10, "close": 11, "volume": 110},
    ]
    calls = []

    def fake_load_ohlcv(symbol, start_date, end_date, provider=None):
        calls.append((symbol, start_date, end_date, provider))
        return rows

    monkeypatch.setattr("core.agno_workflow.load_ohlcv", fake_load_ohlcv)
    monkeypatch.setattr("core.agno_workflow.reflect_outcomes", lambda *args, **kwargs: [])
    monkeypatch.setattr("core.agno_workflow.load_lessons", lambda *args, **kwargs: "")
    session_state = {
        "symbol": "AAPL",
        "cash": 10000,
        "start_date": "2024-01-01",
        "end_date": "2024-04-01",
        "data_provider": "yfinance",
        "preloaded_rows": rows,
    }

    _load_market(None, session_state=session_state)

    assert calls == [("SPY", "2024-03-01", "2024-03-05", "yfinance")]


def test_risk_gate_preserves_signed_short_shares(monkeypatch):
    seen = {}

    def fake_risk_manager(decision, market, cash, shares, risk_debate):
        seen["shares"] = shares
        return {"approved": True, "adjusted_position": 0.2, "stop_loss": 105, "risk_notes": "ok"}

    monkeypatch.setattr("core.agno_workflow.run_risk_manager", fake_risk_manager)
    session_state = {
        "symbol": "AAPL",
        "cash": 10000,
        "start_date": "2024-03-01",
        "end_date": "2024-03-05",
        "trade_date": "2024-03-04",
        "rows": [{"date": "2024-03-04", "close": 100}],
        "market": {"date": "2024-03-04", "close": 100},
        "portfolio": {"cash": 12000, "shares": -20},
        "final_trader": {"action": "sell", "confidence": 0.8, "reasoning": "test", "position_size": 0.2},
        "state": {"investment_debate_state": {}, "risk_debate_state": {"count": 1}},
        "log_path": None,
    }

    _risk_gate(None, session_state=session_state)

    assert seen["shares"] == -20


def test_execute_trade_opens_short_and_covers_with_signed_accounting():
    portfolio = {
        "cash": 10000.0,
        "shares": 0.0,
        "cost_basis": 0.0,
        "position": "flat",
        "equity": 10000.0,
        "pnl": 0.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "stop_loss": None,
        "trades": [],
        "equity_curve": [10000.0],
    }
    sell = {"action": "sell", "confidence": 0.8, "reasoning": "test", "position_size": 0.5}
    buy = {"action": "buy", "confidence": 0.8, "reasoning": "test", "position_size": 1.0}
    open_risk = {"approved": True, "adjusted_position": 0.5, "stop_loss": 105, "risk_notes": "ok"}
    cover_risk = {"approved": True, "adjusted_position": 1.0, "stop_loss": None, "risk_notes": "ok"}

    opened = execute_trade(portfolio, sell, open_risk, {"date": "2024-03-04", "close": 100})
    covered = execute_trade(opened["portfolio"], buy, cover_risk, {"date": "2024-03-05", "close": 90})

    assert opened["portfolio"]["shares"] < 0
    assert opened["trade"]["action"] == "sell_short"
    assert opened["portfolio"]["cost_basis"] == 100
    assert covered["portfolio"]["shares"] == 0
    assert covered["portfolio"]["realized_pnl"] == 500
    assert covered["portfolio"]["equity"] == 10500


def test_paper_signal_tracks_cost_basis_across_short_position():
    account = {
        "cash": 10000.0,
        "shares": 0.0,
        "position": "flat",
        "equity": 10000.0,
        "trades": [],
        "equity_curve": [10000.0],
    }

    long_result = execute_paper_signal(account, "buy", {"date": "2024-01-02", "close": 100})
    short_result = execute_paper_signal(long_result["account"], "sell", {"date": "2024-01-03", "close": 110})

    assert long_result["account"]["cost_basis"] == 100
    assert short_result["account"]["position"] == "short"
    assert short_result["account"]["cost_basis"] == 110
    assert short_result["account"]["realized_pnl"] == 1000


def test_sentiment_report_can_use_provided_news_without_refetch(monkeypatch):
    def fail_news(*args, **kwargs):
        raise AssertionError("provided news report should not be refetched")

    monkeypatch.setattr("agents.analyst_reports.get_news", fail_news)

    result = run_sentiment_analyst_report("AAPL", "Apple reports strong demand and analyst upgrades.")

    assert "Apple reports strong demand" in result["observations"]
    assert result["tool_calls"][0]["args"]["source"] == "provided_news_report"
