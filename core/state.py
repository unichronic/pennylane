try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    messages: list
    investment_debate_state: dict
    risk_debate_state: dict


def make_portfolio(cash):
    return {
        "cash": float(cash),
        "shares": 0.0,
        "cost_basis": 0.0,
        "position": "flat",
        "equity": float(cash),
        "pnl": 0.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "stop_loss": None,
        "trades": [],
        "equity_curve": [float(cash)],
    }


def make_investment_debate_state():
    return {
        "bull_history": "",
        "bear_history": "",
        "history": "",
        "current_response": "",
        "judge_decision": "",
        "count": 0,
        "bull_argument": None,
        "bear_argument": None,
        "bull_validation": None,
        "bear_validation": None,
        "subagent_reviews": [],
    }


def make_risk_debate_state():
    return {
        "aggressive_history": "",
        "conservative_history": "",
        "neutral_history": "",
        "history": "",
        "latest_speaker": "",
        "current_aggressive_response": "",
        "current_conservative_response": "",
        "current_neutral_response": "",
        "judge_decision": "",
        "count": 0,
    }


def make_pipeline_state(cash):
    return {
        "market_report": "",
        "investment_debate_state": make_investment_debate_state(),
        "risk_debate_state": make_risk_debate_state(),
        "portfolio": make_portfolio(cash),
        "trace": {},
    }


def execute_trade(portfolio, decision, risk, market):
    p = dict(portfolio)
    p["trades"] = list(portfolio["trades"])
    p["equity_curve"] = list(portfolio["equity_curve"])
    price = float(market["close"])
    trade = None
    eps = 1e-12

    if p["shares"] > 0 and p.get("stop_loss") is not None and float(market.get("low", price)) <= p["stop_loss"]:
        qty = p["shares"]
        fill = p["stop_loss"]
        p["cash"] += qty * fill
        p["realized_pnl"] += (fill - p["cost_basis"]) * qty
        p["shares"] = 0
        p["cost_basis"] = 0
        trade = {"date": market.get("date"), "action": "stop_loss", "price": fill, "shares": qty}
        p["stop_loss"] = None
    elif p["shares"] < 0 and p.get("stop_loss") is not None and float(market.get("high", price)) >= p["stop_loss"]:
        qty = abs(p["shares"])
        fill = p["stop_loss"]
        p["cash"] -= qty * fill
        p["realized_pnl"] += (p["cost_basis"] - fill) * qty
        p["shares"] = 0
        p["cost_basis"] = 0
        trade = {"date": market.get("date"), "action": "short_stop_loss", "price": fill, "shares": qty}
        p["stop_loss"] = None

    if risk["approved"] and risk["adjusted_position"] > 0:
        size = risk["adjusted_position"]
        if decision["action"] == "buy" and p["shares"] < -eps:
            qty = abs(p["shares"]) * size
            p["cash"] -= qty * price
            p["realized_pnl"] += (p["cost_basis"] - price) * qty
            p["shares"] += qty
            trade = {"date": market.get("date"), "action": "cover", "price": price, "shares": qty}
            if abs(p["shares"]) <= eps:
                p["shares"] = 0
                p["cost_basis"] = 0
                p["stop_loss"] = None
        elif decision["action"] == "buy" and p["cash"] > 0:
            spend = p["cash"] * size
            qty = spend / price
            old_value = p["shares"] * p["cost_basis"]
            p["cash"] -= spend
            p["cost_basis"] = (old_value + spend) / (p["shares"] + qty)
            p["shares"] += qty
            trade = {"date": market.get("date"), "action": "buy", "price": price, "shares": qty}
            p["stop_loss"] = risk.get("stop_loss")
        elif decision["action"] == "sell" and p["shares"] > 0:
            qty = p["shares"] * size
            p["cash"] += qty * price
            p["realized_pnl"] += (price - p["cost_basis"]) * qty
            p["shares"] -= qty
            trade = {"date": market.get("date"), "action": "sell", "price": price, "shares": qty}
            if p["shares"] == 0:
                p["cost_basis"] = 0
                p["stop_loss"] = None
        elif decision["action"] == "sell":
            equity = p["cash"] + p["shares"] * price
            short_notional = max(0, equity) * size
            qty = short_notional / price if price else 0
            if qty > eps:
                old_value = abs(p["shares"]) * p["cost_basis"]
                old_abs = abs(p["shares"])
                p["cash"] += qty * price
                p["shares"] -= qty
                p["cost_basis"] = (old_value + qty * price) / (old_abs + qty)
                trade = {"date": market.get("date"), "action": "sell_short", "price": price, "shares": qty}
                p["stop_loss"] = risk.get("stop_loss")

    p["equity"] = p["cash"] + p["shares"] * price
    if p["shares"] > 0:
        p["unrealized_pnl"] = (price - p["cost_basis"]) * p["shares"]
        p["position"] = "long"
    elif p["shares"] < 0:
        p["unrealized_pnl"] = (p["cost_basis"] - price) * abs(p["shares"])
        p["position"] = "short"
    else:
        p["unrealized_pnl"] = 0
        p["position"] = "flat"
    p["pnl"] = p["equity"] - p["equity_curve"][0]
    p["equity_curve"].append(p["equity"])
    if trade:
        p["trades"].append(trade)

    return {
        "portfolio": p,
        "trade": trade,
        "approved": risk["approved"],
        "price": price,
    }


def metrics(equity_curve):
    if not equity_curve or len(equity_curve) < 2:
        return {"cumulative_return": 0, "sharpe_ratio": 0, "max_drawdown": 0}

    returns = []
    for prev, cur in zip(equity_curve, equity_curve[1:]):
        returns.append((cur - prev) / prev if prev else 0)

    cumulative = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] if equity_curve[0] else 0
    mean = sum(returns) / len(returns)
    variance = sum((x - mean) ** 2 for x in returns) / len(returns)
    std = variance ** 0.5
    sharpe = (mean / std) * (252 ** 0.5) if std else 0

    peak = equity_curve[0]
    max_dd = 0
    for val in equity_curve:
        peak = max(peak, val)
        dd = (val - peak) / peak if peak else 0
        max_dd = min(max_dd, dd)

    return {
        "cumulative_return": round(cumulative, 6),
        "sharpe_ratio": round(sharpe, 6),
        "max_drawdown": round(max_dd, 6),
    }
