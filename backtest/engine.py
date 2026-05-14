from agents.analyst import run_analyst
from agents.debate import run_debate_round
from agents.risk_manager import run_risk_manager
from agents.risk_debate import run_risk_debate
from agents.schemas import validate_analyst, validate_risk, validate_trader
from agents.trader import run_trader
from backtest.paper_methodology import (
    PAPER_END_DATE_EXCLUSIVE,
    PAPER_START_DATE,
    PaperBacktestConfig,
    run_penny_lane_paper_backtest,
)
from core.logger import log_stage
from core.state import execute_trade, make_portfolio, metrics
from data.indicators import add_indicators
from data.loader import load_ohlcv
from pathlib import Path


def run_legacy_backtest(
    symbol="AAPL",
    cash=10000,
    log_path=None,
    warmup=20,
    start_date=None,
    end_date=None,
    data_provider=None,
):
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        open(log_path, "w").close()

    rows = add_indicators(load_ohlcv(symbol, start_date, end_date, provider=data_provider))
    portfolio = make_portfolio(cash)
    trace = []

    for idx in range(max(2, warmup), len(rows)):
        window = rows[: idx + 1]
        market = window[-1]
        analyst = validate_analyst(run_analyst(window))
        debate_result = run_debate_round(analyst, market)
        debate = debate_result["debate"]
        trader = validate_trader(run_trader(analyst, debate, debate_result["investment_plan"]))
        risk_debate = run_risk_debate(analyst, trader, market)
        risk = validate_risk(run_risk_manager(trader, market, portfolio["cash"], portfolio["shares"], risk_debate))
        execution = execute_trade(portfolio, trader, risk, market)
        portfolio = execution["portfolio"]
        item = {
            "date": market.get("date"),
            "analyst": analyst,
            "debate": debate,
            "investment_debate_state": debate_result["investment_debate_state"],
            "investment_plan": debate_result["investment_plan"],
            "trader": trader,
            "risk": risk,
            "risk_debate_state": risk_debate,
            "execution": execution,
        }
        trace.append(item)
        log_stage(log_path, "step", item)

    final_metrics = metrics(portfolio["equity_curve"])
    result = {
        "portfolio": portfolio,
        "trades": portfolio["trades"],
        "metrics": final_metrics,
        "trace": trace,
    }
    log_stage(log_path, "evaluation", final_metrics)
    return result


def run_backtest(
    symbol="AAPL",
    cash=10000,
    log_path=None,
    warmup=None,
    start_date=PAPER_START_DATE,
    end_date=PAPER_END_DATE_EXCLUSIVE,
    data_provider="yfinance",
    decision_cadence_days=1,
    short_policy="allow_short",
    record_memory=False,
):
    """Main backtest entrypoint.

    ``warmup`` is still accepted because a few older call sites pass it in,
    even though the walk-forward setup does not really use it now.
    """
    return run_penny_lane_paper_backtest(
        symbol,
        start_date=start_date or PAPER_START_DATE,
        end_date=end_date or PAPER_END_DATE_EXCLUSIVE,
        data_provider=data_provider or "yfinance",
        config=PaperBacktestConfig(
            initial_cash=cash,
            decision_cadence_days=decision_cadence_days or 1,
            short_policy=short_policy or "allow_short",
            record_memory=record_memory,
        ),
        log_path=log_path,
    )
