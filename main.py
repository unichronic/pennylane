import argparse
import json
from datetime import date, timedelta

from backtest.engine import run_backtest
from backtest.paper_methodology import (
    PAPER_END_DATE_EXCLUSIVE,
    PAPER_START_DATE,
    PaperBacktestConfig,
    run_baseline_backtest,
    run_paper_experiment,
)
from core.pipeline import run_pipeline


def resolve_live_date_range(start=None, end=None, today=None):
    """Figure out a default live window.

    yfinance treats ``end`` as exclusive on daily bars, so using tomorrow here
    usually pulls in the newest daily candle that is already up.
    """
    today = today or date.today()
    end_s = end or (today + timedelta(days=1)).strftime("%Y-%m-%d")
    start_s = start or (date.fromisoformat(end_s) - timedelta(days=120)).strftime("%Y-%m-%d")
    return start_s, end_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", nargs="?", default="AAPL")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--cash", type=float, default=10000)
    parser.add_argument("--log", default="runs/latest.jsonl")
    parser.add_argument(
        "--data-provider",
        choices=["auto", "yfinance", "twelvedata", "alpha_vantage"],
        help="Market data provider. Defaults to TRADEAGE_DATA_PROVIDER or auto.",
    )
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument(
        "--paper-backtest",
        action="store_true",
        help="Deprecated alias for --backtest; backtests now use the paper methodology.",
    )
    parser.add_argument(
        "--paper-experiment",
        action="store_true",
        help="Run the multi-symbol paper benchmark table.",
    )
    parser.add_argument(
        "--symbols",
        default="AAPL,GOOGL,AMZN",
        help="Comma-separated symbols for --paper-experiment.",
    )
    parser.add_argument(
        "--baseline",
        choices=["buy_and_hold", "macd", "kdj_rsi", "zmr", "sma"],
        help="Run one paper baseline instead of the full committee.",
    )
    args = parser.parse_args()

    if args.backtest or args.paper_backtest or args.paper_experiment or args.baseline:
        end_date = args.end or PAPER_END_DATE_EXCLUSIVE
        start_date = args.start or PAPER_START_DATE
    else:
        start_date, end_date = resolve_live_date_range(args.start, args.end)

    if args.paper_experiment:
        result = run_paper_experiment(
            symbols=[item.strip() for item in args.symbols.split(",") if item.strip()],
            start_date=start_date,
            end_date=end_date,
            data_provider=args.data_provider or "yfinance",
            include_penny_lane=args.baseline is None,
            baselines=([args.baseline] if args.baseline else ("buy_and_hold", "macd", "kdj_rsi", "zmr", "sma")),
            config=PaperBacktestConfig(initial_cash=args.cash),
            log_path=args.log,
        )
        print(json.dumps(result, indent=2))
        return
    if args.baseline:
        result = run_baseline_backtest(
            args.symbol,
            args.baseline,
            start_date=start_date,
            end_date=end_date,
            data_provider=args.data_provider or "yfinance",
            config=PaperBacktestConfig(initial_cash=args.cash),
            log_path=args.log,
        )
    elif args.backtest or args.paper_backtest:
        result = run_backtest(
            args.symbol,
            cash=args.cash,
            log_path=args.log,
            start_date=start_date,
            end_date=end_date,
            data_provider=args.data_provider,
        )
    else:
        result = run_pipeline(
            args.symbol,
            cash=args.cash,
            log_path=args.log,
            start_date=start_date,
            end_date=end_date,
            data_provider=args.data_provider,
        )

    print(json.dumps({
        "portfolio": result.get("portfolio", result.get("account")),
        "metrics": result["metrics"],
        "trades": result.get("portfolio", result.get("account", {})).get("trades", result.get("trades", [])),
    }, indent=2))


if __name__ == "__main__":
    main()
