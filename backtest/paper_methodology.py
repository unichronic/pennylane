from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from core.logger import log_stage
from core.reflection import reflect_outcomes
from core.agno_workflow import run_agno_pipeline
from data.indicators import add_indicators, ema, rsi_at
from data.loader import load_ohlcv

PAPER_START_DATE = "2024-01-01"
# yfinance end dates are exclusive, so this still includes 2024-03-29.
PAPER_END_DATE_EXCLUSIVE = "2024-03-30"
PAPER_RESULT_SYMBOLS = ("AAPL", "GOOGL", "AMZN")
PAPER_TECH_SYMBOLS = ("AAPL", "NVDA", "MSFT", "META", "GOOGL", "AMZN")


@dataclass(frozen=True)
class PaperBacktestConfig:
    initial_cash: float = 10000
    risk_free_rate: float = 0.0
    trading_days_per_year: int = 252
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    kdj_period: int = 9
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    sma_short: int = 5
    sma_long: int = 20
    zmr_window: int = 20
    zmr_threshold: float = 1.0
    decision_cadence_days: int = 1
    short_policy: str = "allow_short"
    record_memory: bool = False


def _empty_account(cash: float) -> dict:
    return {
        "cash": float(cash),
        "shares": 0.0,
        "cost_basis": 0.0,
        "position": "flat",
        "equity": float(cash),
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "trades": [],
        "equity_curve": [float(cash)],
    }


def _position_label(shares: float) -> str:
    if shares > 0:
        return "long"
    if shares < 0:
        return "short"
    return "flat"


def _short_confirmed(market: dict) -> bool:
    close = float(market["close"])
    sma20 = market.get("close_20_sma")
    hist = market.get("macd_hist")
    rsi = market.get("rsi")
    below_sma = sma20 is not None and close < float(sma20)
    bearish_macd = hist is not None and float(hist) < 0
    weak_rsi = rsi is not None and float(rsi) < 55
    return sum([below_sma, bearish_macd, weak_rsi]) >= 2


def _target_from_signal(
    signal: str,
    current_shares: float,
    *,
    market: dict | None = None,
    config: PaperBacktestConfig | None = None,
) -> int | None:
    signal = str(signal).lower()
    if signal == "buy":
        return 1
    if signal == "sell":
        short_policy = str((config or PaperBacktestConfig()).short_policy or "allow_short").lower()
        if short_policy in {"reduce_only", "long_only", "no_short"}:
            return 0 if current_shares > 0 else None
        if short_policy in {"trend_confirmed", "trend_confirmed_short"}:
            if current_shares > 0:
                return 0
            return -1 if market and _short_confirmed(market) else None
        return -1
    return None


def execute_paper_signal(
    account: dict,
    signal: str,
    market: dict,
    *,
    config: PaperBacktestConfig | None = None,
) -> dict:
    """Apply a buy/sell/hold signal to the paper account.

    This is a plain close-to-close model. Buy means fully long, sell means
    fully short, hold means keep whatever the account already has.
    """
    p = dict(account)
    p["trades"] = list(account["trades"])
    p["equity_curve"] = list(account["equity_curve"])
    p.setdefault("cost_basis", 0.0)
    p.setdefault("realized_pnl", 0.0)
    p.setdefault("unrealized_pnl", 0.0)
    price = float(market["close"])
    eq0 = p["cash"] + p["shares"] * price
    target = _target_from_signal(signal, p["shares"], market=market, config=config)
    want_shares = p["shares"] if target is None else target * eq0 / price if price else 0.0
    delta = want_shares - p["shares"]
    trade = None

    if abs(delta) > 1e-12:
        old_shares = p["shares"]
        old_cost = p["cost_basis"]
        if old_shares > 0 and delta < 0:
            closed_qty = min(old_shares, abs(delta))
            p["realized_pnl"] += (price - old_cost) * closed_qty
        elif old_shares < 0 and delta > 0:
            closed_qty = min(abs(old_shares), delta)
            p["realized_pnl"] += (old_cost - price) * closed_qty

        p["cash"] -= delta * price
        p["shares"] = want_shares
        if abs(want_shares) <= 1e-12:
            p["shares"] = 0.0
            p["cost_basis"] = 0.0
        elif old_shares == 0 or old_shares * want_shares < 0:
            p["cost_basis"] = price
        elif abs(want_shares) > abs(old_shares):
            add_qty = abs(want_shares) - abs(old_shares)
            p["cost_basis"] = ((abs(old_shares) * old_cost) + (add_qty * price)) / abs(want_shares)
        else:
            p["cost_basis"] = old_cost
        trade = {
            "date": market.get("date"),
            "signal": signal,
            "action": _position_label(want_shares),
            "price": price,
            "shares_delta": delta,
            "target_shares": want_shares,
        }
        p["trades"].append(trade)

    p["equity"] = p["cash"] + p["shares"] * price
    p["position"] = _position_label(p["shares"])
    if p["shares"] > 0:
        p["unrealized_pnl"] = (price - p["cost_basis"]) * p["shares"]
    elif p["shares"] < 0:
        p["unrealized_pnl"] = (p["cost_basis"] - price) * abs(p["shares"])
    else:
        p["unrealized_pnl"] = 0.0
    p["equity_curve"].append(p["equity"])
    return {"account": p, "trade": trade, "price": price}


def paper_metrics(equity_curve: list[float], *, risk_free_rate=0.0, trading_days_per_year=252) -> dict:
    if not equity_curve or len(equity_curve) < 2 or equity_curve[0] == 0:
        return {
            "cumulative_return_pct": 0,
            "annualized_return_pct": 0,
            "sharpe_ratio": 0,
            "max_drawdown_pct": 0,
        }

    rets = [
        (cur - prev) / prev if prev else 0
        for prev, cur in zip(equity_curve, equity_curve[1:])
    ]
    ret = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    years = len(rets) / trading_days_per_year
    ann = (equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1 if years else 0

    daily_rf = risk_free_rate / trading_days_per_year
    excess = [day_ret - daily_rf for day_ret in rets]
    avg = mean(excess)
    var = sum((day_ret - avg) ** 2 for day_ret in excess) / len(excess)
    std = var ** 0.5
    sharpe = (avg / std) * (trading_days_per_year ** 0.5) if std else 0

    peak = equity_curve[0]
    max_dd = 0
    for value in equity_curve:
        peak = max(peak, value)
        dd = (peak - value) / peak if peak else 0
        max_dd = max(max_dd, dd)

    return {
        "cumulative_return_pct": round(ret * 100, 6),
        "annualized_return_pct": round(ann * 100, 6),
        "sharpe_ratio": round(sharpe, 6),
        "max_drawdown_pct": round(max_dd * 100, 6),
    }


def _sma(values: list[float], idx: int, period: int) -> float | None:
    if idx + 1 < period:
        return None
    return sum(values[idx - period + 1 : idx + 1]) / period


def _macd_signals(rows: list[dict], config: PaperBacktestConfig) -> list[str]:
    closes = [float(row["close"]) for row in rows]
    fast = ema(closes, config.macd_fast)
    slow = ema(closes, config.macd_slow)
    macd = [a - b for a, b in zip(fast, slow)]
    signal = ema(macd, config.macd_signal)
    out = []
    for idx in range(len(rows)):
        if idx == 0:
            out.append("hold")
        elif macd[idx] > signal[idx] and macd[idx - 1] <= signal[idx - 1]:
            out.append("buy")
        elif macd[idx] < signal[idx] and macd[idx - 1] >= signal[idx - 1]:
            out.append("sell")
        else:
            out.append("hold")
    return out


def _kdj_values(rows: list[dict], period: int) -> list[tuple[float | None, float | None, float | None]]:
    k = 50.0
    d = 50.0
    values = []
    for idx, row in enumerate(rows):
        if idx + 1 < period:
            values.append((None, None, None))
            continue
        window = rows[idx - period + 1 : idx + 1]
        low = min(float(item["low"]) for item in window)
        high = max(float(item["high"]) for item in window)
        rsv = 50.0 if high == low else (float(row["close"]) - low) / (high - low) * 100
        k = (2 / 3) * k + (1 / 3) * rsv
        d = (2 / 3) * d + (1 / 3) * k
        j = 3 * k - 2 * d
        values.append((k, d, j))
    return values


def _kdj_rsi_signals(rows: list[dict], config: PaperBacktestConfig) -> list[str]:
    closes = [float(row["close"]) for row in rows]
    kdj = _kdj_values(rows, config.kdj_period)
    out = []
    for idx, (_, _, j) in enumerate(kdj):
        rsi = rsi_at(closes, idx, config.rsi_period)
        if j is None or rsi is None:
            out.append("hold")
        elif j < 20 and rsi < config.rsi_oversold:
            out.append("buy")
        elif j > 80 and rsi > config.rsi_overbought:
            out.append("sell")
        else:
            out.append("hold")
    return out


def _sma_signals(rows: list[dict], config: PaperBacktestConfig) -> list[str]:
    closes = [float(row["close"]) for row in rows]
    out = []
    for idx in range(len(rows)):
        short_now = _sma(closes, idx, config.sma_short)
        long_now = _sma(closes, idx, config.sma_long)
        short_prev = _sma(closes, idx - 1, config.sma_short) if idx else None
        long_prev = _sma(closes, idx - 1, config.sma_long) if idx else None
        if None in {short_now, long_now, short_prev, long_prev}:
            out.append("hold")
        elif short_now > long_now and short_prev <= long_prev:
            out.append("buy")
        elif short_now < long_now and short_prev >= long_prev:
            out.append("sell")
        else:
            out.append("hold")
    return out


def _zmr_signals(rows: list[dict], config: PaperBacktestConfig) -> list[str]:
    closes = [float(row["close"]) for row in rows]
    out = []
    for idx, close in enumerate(closes):
        if idx + 1 < config.zmr_window:
            out.append("hold")
            continue
        window = closes[idx - config.zmr_window + 1 : idx + 1]
        avg = mean(window)
        variance = sum((value - avg) ** 2 for value in window) / len(window)
        std = variance ** 0.5
        z = (close - avg) / std if std else 0
        if z <= -config.zmr_threshold:
            out.append("buy")
        elif z >= config.zmr_threshold:
            out.append("sell")
        else:
            out.append("hold")
    return out


def baseline_signals(name: str, rows: list[dict], config: PaperBacktestConfig) -> list[str]:
    normalized = name.lower().replace("_", "").replace("+", "").replace("&", "")
    if normalized in {"buyhold", "buyandhold", "bh"}:
        return ["buy"] + ["hold"] * (len(rows) - 1)
    if normalized == "macd":
        return _macd_signals(rows, config)
    if normalized in {"kdjrsi", "kdjandrsi"}:
        return _kdj_rsi_signals(rows, config)
    if normalized == "zmr":
        return _zmr_signals(rows, config)
    if normalized == "sma":
        return _sma_signals(rows, config)
    raise ValueError(f"unknown paper baseline: {name}")


def run_signal_backtest(
    rows: list[dict],
    signals: list[str],
    *,
    config: PaperBacktestConfig | None = None,
    log_path=None,
    strategy_name="strategy",
) -> dict:
    config = config or PaperBacktestConfig()
    if len(signals) != len(rows):
        raise ValueError("signals length must match rows length")

    account = _empty_account(config.initial_cash)
    trace = []
    for market, signal in zip(rows, signals):
        execution = execute_paper_signal(account, signal, market, config=config)
        account = execution["account"]
        item = {"date": market.get("date"), "signal": signal, "execution": execution}
        trace.append(item)
        log_stage(log_path, strategy_name, item)

    return {
        "account": account,
        "trades": account["trades"],
        "metrics": paper_metrics(
            account["equity_curve"],
            risk_free_rate=config.risk_free_rate,
            trading_days_per_year=config.trading_days_per_year,
        ),
        "trace": trace,
    }


def run_baseline_backtest(
    symbol: str,
    baseline: str,
    *,
    start_date=PAPER_START_DATE,
    end_date=PAPER_END_DATE_EXCLUSIVE,
    data_provider="yfinance",
    config: PaperBacktestConfig | None = None,
    log_path=None,
) -> dict:
    config = config or PaperBacktestConfig()
    rows = load_ohlcv(symbol, start_date, end_date, provider=data_provider)
    signals = baseline_signals(baseline, rows, config)
    result = run_signal_backtest(
        rows,
        signals,
        config=config,
        log_path=log_path,
        strategy_name=f"{symbol}:{baseline}",
    )
    result["symbol"] = symbol.upper()
    result["strategy"] = baseline
    result["period"] = {"start": start_date, "end_exclusive": end_date}
    return result


def run_penny_lane_paper_backtest(
    symbol: str,
    *,
    start_date=PAPER_START_DATE,
    end_date=PAPER_END_DATE_EXCLUSIVE,
    data_provider="yfinance",
    config: PaperBacktestConfig | None = None,
    log_path=None,
    truncate_log=True,
) -> dict:
    config = config or PaperBacktestConfig()
    if log_path and truncate_log:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        open(log_path, "w").close()

    rows = add_indicators(load_ohlcv(symbol, start_date, end_date, provider=data_provider))
    try:
        benchmark_rows = add_indicators(load_ohlcv("SPY", start_date, end_date, provider=data_provider))
    except Exception:
        benchmark_rows = None
    account = _empty_account(config.initial_cash)
    trace = []

    last_workflow_trace = None
    cadence = max(1, int(config.decision_cadence_days or 1))

    for idx, market in enumerate(rows):
        window = rows[: idx + 1]
        should_decide = idx == 0 or idx % cadence == 0
        if should_decide:
            workflow_result = run_agno_pipeline(
                symbol,
                cash=account["equity"],
                log_path=log_path,
                start_date=start_date,
                end_date=end_date,
                data_provider=data_provider,
                preloaded_rows=window,
                initial_portfolio={
                    "cash": account["cash"],
                    "shares": account["shares"],
                    "cost_basis": account["cost_basis"],
                    "equity": account["equity"],
                    "pnl": account["equity"] - config.initial_cash,
                    "realized_pnl": account["realized_pnl"],
                    "unrealized_pnl": account["unrealized_pnl"],
                    "stop_loss": None,
                    "position": account["position"],
                    "trades": list(account["trades"]),
                    "equity_curve": list(account["equity_curve"]),
                },
                trade_date=market.get("date"),
                workflow_session_id=f"{symbol.upper()}-paper-{market.get('date')}",
                truncate_log=False,
                record_memory=config.record_memory,
                tool_reports=True,
            )
            workflow_trace = workflow_result["trace"]
            last_workflow_trace = workflow_trace
            signal = (
                workflow_trace["portfolio_manager"]["action"]
                if workflow_trace["risk"]["approved"]
                else "hold"
            )
        else:
            workflow_trace = last_workflow_trace or {}
            signal = "hold"
        execution = execute_paper_signal(account, signal, market, config=config)
        account = execution["account"]
        item = {
            "date": market.get("date"),
            "decision_cadence_days": cadence,
            "decision_skipped": not should_decide,
            "workflow": workflow_trace.get("workflow", {}),
            "reports": workflow_trace.get("reports", {}),
            "analyst": workflow_trace.get("analyst", {}),
            "debate": workflow_trace.get("debate", {}),
            "investment_debate_state": workflow_trace.get("investment_debate_state", {}),
            "investment_plan": workflow_trace.get("investment_plan", ""),
            "trader": workflow_trace.get("trader", {}),
            "risk": workflow_trace.get("risk", {"approved": True}),
            "portfolio_manager": workflow_trace.get("portfolio_manager", {"action": signal}),
            "final_trader": workflow_trace.get("final_trader", {"action": signal}),
            "risk_debate_state": workflow_trace.get("risk_debate_state", {}),
            "signal": signal,
            "execution": execution,
        }
        trace.append(item)
        log_stage(log_path, f"{symbol}:PennyLaneCapital", item)

    final_metrics = paper_metrics(
        account["equity_curve"],
        risk_free_rate=config.risk_free_rate,
        trading_days_per_year=config.trading_days_per_year,
    )
    result = {
        "symbol": symbol.upper(),
        "strategy": "PennyLaneCapital",
        "period": {"start": start_date, "end_exclusive": end_date},
        "decision_cadence_days": cadence,
        "account": account,
        "portfolio": account,
        "trades": account["trades"],
        "metrics": final_metrics,
        "trace": trace,
        "reflections": reflect_outcomes(symbol, rows, benchmark_rows=benchmark_rows),
    }
    log_stage(log_path, f"{symbol}:evaluation", final_metrics)
    return result


def run_paper_experiment(
    symbols=PAPER_RESULT_SYMBOLS,
    *,
    start_date=PAPER_START_DATE,
    end_date=PAPER_END_DATE_EXCLUSIVE,
    data_provider="yfinance",
    include_penny_lane=True,
    baselines=("buy_and_hold", "macd", "kdj_rsi", "zmr", "sma"),
    config: PaperBacktestConfig | None = None,
    log_path=None,
) -> dict:
    config = config or PaperBacktestConfig()
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        open(log_path, "w").close()
        log_stage(
            log_path,
            "paper_experiment:start",
            {
                "symbols": [str(item).upper() for item in symbols],
                "start": start_date,
                "end_exclusive": end_date,
                "include_penny_lane": include_penny_lane,
                "baselines": list(baselines),
            },
        )
    results = {}
    for symbol in symbols:
        log_stage(log_path, "paper_experiment:symbol_start", {"symbol": str(symbol).upper()})
        symbol_results = {}
        if include_penny_lane:
            symbol_results["PennyLaneCapital"] = run_penny_lane_paper_backtest(
                symbol,
                start_date=start_date,
                end_date=end_date,
                data_provider=data_provider,
                config=config,
                log_path=log_path,
                truncate_log=False,
            )
        for baseline in baselines:
            symbol_results[baseline] = run_baseline_backtest(
                symbol,
                baseline,
                start_date=start_date,
                end_date=end_date,
                data_provider=data_provider,
                config=config,
                log_path=log_path,
            )
        results[symbol.upper()] = symbol_results
        log_stage(
            log_path,
            "paper_experiment:symbol_complete",
            {"symbol": str(symbol).upper(), "strategies": list(symbol_results)},
        )

    aggregate = aggregate_experiment_results(results, config=config)
    log_stage(log_path, "paper_experiment:aggregate", aggregate)
    return {
        "methodology": {
            "source": "Original multi-agent trading paper (arXiv:2412.20138v7)",
            "period": {
                "paper_start": PAPER_START_DATE,
                "paper_end_inclusive": "2024-03-29",
                "loader_end_exclusive": end_date,
            },
            "metrics": ["CR", "AR", "SR", "MDD"],
            "baselines": list(baselines),
            "data_provider": data_provider,
            "short_policy": config.short_policy,
            "record_memory": config.record_memory,
        },
        "results": results,
        "aggregate": aggregate,
    }


def aggregate_experiment_results(results: dict, *, config: PaperBacktestConfig | None = None) -> dict:
    """Roll per-symbol paper runs into one curve per strategy.

    Nothing fancy here. It just lines up the curves and adds them with equal
    starting capital for each symbol.
    """
    config = config or PaperBacktestConfig()
    names = sorted({strategy for symbol_results in results.values() for strategy in symbol_results})
    aggregate = {}
    for strategy in names:
        curves = [
            symbol_results[strategy].get("account", symbol_results[strategy].get("portfolio", {})).get("equity_curve", [])
            for symbol_results in results.values()
            if strategy in symbol_results
        ]
        curves = [curve for curve in curves if curve]
        if not curves:
            continue
        length = min(len(curve) for curve in curves)
        combo = [sum(curve[idx] for curve in curves) for idx in range(length)]
        aggregate[strategy] = {
            "symbols": len(curves),
            "initial_capital": round(combo[0], 6),
            "ending_equity": round(combo[-1], 6),
            "metrics": paper_metrics(
                combo,
                risk_free_rate=config.risk_free_rate,
                trading_days_per_year=config.trading_days_per_year,
            ),
            "equity_curve": combo,
        }
    return aggregate
