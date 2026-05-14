import pytest

from backtest.paper_methodology import (
    PaperBacktestConfig,
    baseline_signals,
    execute_paper_signal,
    paper_metrics,
    run_paper_experiment,
    run_signal_backtest,
    run_penny_lane_paper_backtest,
)


@pytest.fixture(autouse=True)
def force_local_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")


def sample_rows():
    closes = [100, 102, 101, 105, 108, 106, 110, 112, 111, 115, 117, 116, 119, 121, 120, 123, 125, 124, 126, 128, 127, 130]
    rows = []
    for idx, close in enumerate(closes, start=1):
        rows.append(
            {
                "date": f"2024-01-{idx:02d}",
                "open": close - 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1000 + idx,
            }
        )
    return rows


def test_paper_signal_execution_supports_long_short_and_hold():
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
    hold_result = execute_paper_signal(short_result["account"], "hold", {"date": "2024-01-04", "close": 100})

    assert long_result["account"]["position"] == "long"
    assert short_result["account"]["position"] == "short"
    assert hold_result["account"]["position"] == "short"
    assert hold_result["account"]["equity"] > short_result["account"]["equity"]


def test_paper_signal_short_policy_can_reduce_without_opening_short():
    account = {
        "cash": 10000.0,
        "shares": 0.0,
        "position": "flat",
        "equity": 10000.0,
        "trades": [],
        "equity_curve": [10000.0],
    }

    result = execute_paper_signal(
        account,
        "sell",
        {"date": "2024-01-02", "close": 100},
        config=PaperBacktestConfig(short_policy="reduce_only"),
    )

    assert result["trade"] is None
    assert result["account"]["position"] == "flat"


def test_paper_signal_trend_confirmed_short_blocks_uptrend_short():
    account = {
        "cash": 10000.0,
        "shares": 0.0,
        "position": "flat",
        "equity": 10000.0,
        "trades": [],
        "equity_curve": [10000.0],
    }
    market = {
        "date": "2024-01-02",
        "close": 110,
        "close_20_sma": 100,
        "macd_hist": 1,
        "rsi": 70,
    }

    result = execute_paper_signal(
        account,
        "sell",
        market,
        config=PaperBacktestConfig(short_policy="trend_confirmed_short"),
    )

    assert result["trade"] is None
    assert result["account"]["position"] == "flat"


def test_paper_signal_trend_confirmed_short_allows_confirmed_downtrend():
    account = {
        "cash": 10000.0,
        "shares": 0.0,
        "position": "flat",
        "equity": 10000.0,
        "trades": [],
        "equity_curve": [10000.0],
    }
    market = {
        "date": "2024-01-02",
        "close": 90,
        "close_20_sma": 100,
        "macd_hist": -1,
        "rsi": 45,
    }

    result = execute_paper_signal(
        account,
        "sell",
        market,
        config=PaperBacktestConfig(short_policy="trend_confirmed_short"),
    )

    assert result["trade"]["action"] == "short"
    assert result["account"]["position"] == "short"


def test_paper_metrics_match_formulas():
    metrics = paper_metrics([100, 110, 105], risk_free_rate=0, trading_days_per_year=252)

    assert metrics["cumulative_return_pct"] == 5.0
    assert metrics["annualized_return_pct"] > 0
    assert metrics["max_drawdown_pct"] == pytest.approx(4.545455)
    assert "sharpe_ratio" in metrics


def test_paper_baselines_emit_daily_signals():
    rows = sample_rows()
    config = PaperBacktestConfig()

    for name in ["buy_and_hold", "macd", "kdj_rsi", "zmr", "sma"]:
        signals = baseline_signals(name, rows, config)
        assert len(signals) == len(rows)
        assert set(signals) <= {"buy", "sell", "hold"}


def test_signal_backtest_returns_paper_metric_set():
    rows = sample_rows()
    signals = ["buy"] + ["hold"] * (len(rows) - 1)
    result = run_signal_backtest(rows, signals)

    assert result["metrics"].keys() == {
        "cumulative_return_pct",
        "annualized_return_pct",
        "sharpe_ratio",
        "max_drawdown_pct",
    }
    assert result["trades"][0]["action"] == "long"


def test_penny_lane_paper_backtest_uses_only_past_rows(monkeypatch):
    rows = sample_rows()[:6]
    seen_lengths = []

    def fake_load_ohlcv(symbol, start_date, end_date, provider=None):
        return rows

    def fake_run_agno_pipeline(*args, **kwargs):
        seen_lengths.append(len(kwargs["preloaded_rows"]))
        assert kwargs["tool_reports"] is True
        return {
            "trace": {
                "workflow": {"runtime": "agno", "steps": ["load_market", "execution_condition"]},
                "reports": {"market": "test", "news": "test", "sentiment": "test", "fundamentals": "test"},
                "analyst": {
                    "trend": "bullish",
                    "confidence": 0.8,
                    "signals": {"rsi": 55, "macd": 1, "macd_signal": 0.5, "macd_hist": 0.5},
                    "summary": "test",
                },
                "debate": {
                    "bull_case": "test",
                    "bear_case": "test",
                    "key_risks": ["test"],
                    "consensus_bias": "bullish",
                },
                "investment_debate_state": {"count": 2},
                "investment_plan": "buy",
                "trader": {"action": "buy", "confidence": 0.8, "reasoning": "test", "position_size": 1},
                "risk": {"approved": True, "adjusted_position": 1, "stop_loss": None, "risk_notes": "test"},
                "portfolio_manager": {"rating": "Buy", "action": "buy", "final_trade_decision": "Buy"},
                "final_trader": {"action": "buy", "confidence": 0.8, "reasoning": "test", "position_size": 1},
                "risk_debate_state": {"count": 3},
            },
            "portfolio": {},
            "metrics": {},
        }

    monkeypatch.setattr("backtest.paper_methodology.load_ohlcv", fake_load_ohlcv)
    monkeypatch.setattr("backtest.paper_methodology.run_agno_pipeline", fake_run_agno_pipeline)

    result = run_penny_lane_paper_backtest("AAPL", start_date="2024-01-01", end_date="2024-01-10")

    assert seen_lengths == [1, 2, 3, 4, 5, 6]
    assert len(result["trace"]) == len(rows)
    assert all(item["workflow"]["runtime"] == "agno" for item in result["trace"])


def test_paper_experiment_keeps_one_log_and_disables_inner_truncation(monkeypatch, tmp_path):
    truncate_flags = []

    def fake_penny_lane(symbol, **kwargs):
        truncate_flags.append(kwargs["truncate_log"])
        return {
            "symbol": symbol,
            "strategy": "PennyLaneCapital",
            "account": {"equity_curve": [10000.0, 10100.0]},
            "portfolio": {"equity_curve": [10000.0, 10100.0]},
            "metrics": {"cumulative_return_pct": 1.0},
            "trace": [],
            "reflections": [],
        }

    def fake_baseline(symbol, baseline, **kwargs):
        return {
            "symbol": symbol,
            "strategy": baseline,
            "account": {"equity_curve": [10000.0, 10050.0]},
            "portfolio": {"equity_curve": [10000.0, 10050.0]},
            "metrics": {"cumulative_return_pct": 0.5},
            "trace": [],
        }

    monkeypatch.setattr("backtest.paper_methodology.run_penny_lane_paper_backtest", fake_penny_lane)
    monkeypatch.setattr("backtest.paper_methodology.run_baseline_backtest", fake_baseline)

    log_path = tmp_path / "paper-experiment.jsonl"
    result = run_paper_experiment(
        symbols=["AAPL", "MSFT"],
        baselines=("buy_and_hold",),
        config=PaperBacktestConfig(initial_cash=10000),
        log_path=str(log_path),
    )

    lines = [line for line in log_path.read_text().splitlines() if line.strip()]

    assert truncate_flags == [False, False]
    assert '"stage": "paper_experiment:start"' in lines[0]
    assert any('"stage": "paper_experiment:symbol_start"' in line and '"symbol": "AAPL"' in line for line in lines)
    assert any('"stage": "paper_experiment:symbol_start"' in line and '"symbol": "MSFT"' in line for line in lines)
    assert lines[-1].startswith('{"stage": "paper_experiment:aggregate"')
    assert set(result["results"]) == {"AAPL", "MSFT"}
