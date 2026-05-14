def run_risk_manager(decision, market, cash=10000, shares=0, risk_debate=None):
    action = decision["action"]
    size = decision["position_size"]
    close = float(market["close"])
    rsi = market.get("rsi")

    if action == "hold" or size <= 0:
        return {
            "approved": True,
            "adjusted_position": 0,
            "stop_loss": None,
            "risk_notes": "no execution needed",
        }

    if size > 0.6:
        return {
            "approved": False,
            "adjusted_position": 0,
            "stop_loss": None,
            "risk_notes": "rejected because requested position exceeds risk limit",
        }

    if action == "buy" and rsi is not None and rsi > 92:
        return {
            "approved": False,
            "adjusted_position": 0,
            "stop_loss": None,
            "risk_notes": "rejected because buy signal is too overbought",
        }

    adjusted = min(size, 0.35)
    if action == "buy" and rsi is not None and rsi > 75:
        adjusted = min(adjusted, 0.18)
    if decision["confidence"] < 0.7:
        adjusted = min(adjusted, 0.25)

    stop = round(close * 0.95, 2) if action == "buy" else round(close * 1.05, 2)

    notes = "approved with capped position sizing"
    if action == "sell" and shares <= 0:
        notes = "approved as short-side exposure with capped position sizing"
    if risk_debate:
        notes = notes + f"; risk debate speakers={risk_debate.get('count', 0)}"

    return {
        "approved": True,
        "adjusted_position": round(adjusted, 2),
        "stop_loss": stop,
        "risk_notes": notes,
    }
