def run_analyst(rows):
    if not rows:
        return {
            "trend": "neutral",
            "confidence": 0,
            "signals": {"rsi": None, "macd": None},
            "summary": "no market data available",
        }

    row = rows[-1]
    close = float(row["close"])
    rsi = row.get("rsi")
    macd = row.get("macd")
    signal = row.get("macd_signal")
    hist = row.get("macd_hist")

    prices = [float(x["close"]) for x in rows]
    base = prices[-6] if len(prices) > 5 else prices[0]
    move = (close - base) / base if base else 0
    score = 0

    if move > 0.015:
        score += 1
    elif move < -0.015:
        score -= 1

    if hist is not None and hist > 0:
        score += 1
    elif hist is not None and hist < 0:
        score -= 1

    if rsi is not None and rsi > 58:
        score += 1
    elif rsi is not None and rsi < 42:
        score -= 1

    if score >= 2:
        trend = "bullish"
    elif score <= -2:
        trend = "bearish"
    else:
        trend = "neutral"

    confidence = min(0.95, round(0.35 + abs(score) * 0.18 + min(abs(move), 0.08), 2))
    if trend == "neutral":
        confidence = min(confidence, 0.55)

    bits = []
    bits.append(f"price move {move:.2%}")
    if rsi is not None:
        bits.append(f"rsi {rsi:.2f}")
    if macd is not None and signal is not None:
        bits.append(f"macd {macd:.4f} vs signal {signal:.4f}")

    return {
        "trend": trend,
        "confidence": confidence,
        "signals": {
            "rsi": rsi,
            "macd": macd,
            "macd_signal": signal,
            "macd_hist": hist,
        },
        "summary": "; ".join(bits),
    }
