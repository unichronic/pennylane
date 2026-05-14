def ema(values, period):
    if not values:
        return []
    out = []
    k = 2 / (period + 1)
    prev = values[0]
    for val in values:
        prev = val * k + prev * (1 - k)
        out.append(prev)
    return out


def sma_at(values, idx, period):
    if idx + 1 < period:
        return None
    return round(sum(values[idx - period + 1 : idx + 1]) / period, 4)


def rsi_at(values, idx, period=14):
    if idx < period:
        return None
    gains = []
    losses = []
    start = idx - period + 1
    for i in range(start, idx + 1):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(diff))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def bollinger_at(values, idx, period=20, width=2):
    if idx + 1 < period:
        return None, None, None
    window = values[idx - period + 1 : idx + 1]
    mid = sum(window) / period
    variance = sum((x - mid) ** 2 for x in window) / period
    std = variance ** 0.5
    return round(mid, 4), round(mid + width * std, 4), round(mid - width * std, 4)


def atr_at(rows, idx, period=14):
    if idx == 0 or idx + 1 < period:
        return None
    trs = []
    start = idx - period + 1
    for i in range(start, idx + 1):
        high = float(rows[i]["high"])
        low = float(rows[i]["low"])
        prev_close = float(rows[i - 1]["close"])
        trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    return round(sum(trs) / period, 4)


def stochastic_at(rows, idx, period=14):
    if idx + 1 < period:
        return None
    window = rows[idx - period + 1 : idx + 1]
    low = min(float(x["low"]) for x in window)
    high = max(float(x["high"]) for x in window)
    if high == low:
        return 50
    return round((float(rows[idx]["close"]) - low) / (high - low) * 100, 2)


def mfi_at(rows, idx, period=14):
    if idx == 0 or idx + 1 < period:
        return None
    positive = 0
    negative = 0
    start = idx - period + 1
    for i in range(start, idx + 1):
        typical = (float(rows[i]["high"]) + float(rows[i]["low"]) + float(rows[i]["close"])) / 3
        prev = (float(rows[i - 1]["high"]) + float(rows[i - 1]["low"]) + float(rows[i - 1]["close"])) / 3
        flow = typical * float(rows[i]["volume"])
        if typical >= prev:
            positive += flow
        else:
            negative += flow
    if negative == 0:
        return 100
    ratio = positive / negative
    return round(100 - (100 / (1 + ratio)), 2)


def obv_values(rows):
    out = []
    current = 0
    for idx, row in enumerate(rows):
        if idx == 0:
            out.append(current)
            continue
        close = float(row["close"])
        prev = float(rows[idx - 1]["close"])
        vol = float(row["volume"])
        if close > prev:
            current += vol
        elif close < prev:
            current -= vol
        out.append(current)
    return out


def add_indicators(rows):
    closes = [float(x["close"]) for x in rows]
    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    ema10 = ema(closes, 10)
    ema20 = ema(closes, 20)
    macd = [a - b for a, b in zip(ema12, ema26)]
    sig = ema(macd, 9)
    stoch_values = [stochastic_at(rows, idx) for idx in range(len(rows))]
    stoch_signal = ema([x if x is not None else 50 for x in stoch_values], 3)
    obv = obv_values(rows)
    out = []
    for idx, row in enumerate(rows):
        new = dict(row)
        new["rsi"] = rsi_at(closes, idx)
        new["macd"] = round(macd[idx], 4)
        new["macd_signal"] = round(sig[idx], 4)
        new["macd_hist"] = round(macd[idx] - sig[idx], 4)
        new["close_5_sma"] = sma_at(closes, idx, 5)
        new["close_10_sma"] = sma_at(closes, idx, 10)
        new["close_20_sma"] = sma_at(closes, idx, 20)
        new["close_50_sma"] = sma_at(closes, idx, 50)
        new["close_10_ema"] = round(ema10[idx], 4)
        new["close_20_ema"] = round(ema20[idx], 4)
        boll, boll_ub, boll_lb = bollinger_at(closes, idx)
        new["boll"] = boll
        new["boll_ub"] = boll_ub
        new["boll_lb"] = boll_lb
        new["atr"] = atr_at(rows, idx)
        new["stoch_k"] = stoch_values[idx]
        new["stoch_d"] = round(stoch_signal[idx], 4)
        new["mfi"] = mfi_at(rows, idx)
        new["obv"] = obv[idx]
        if idx == 0:
            new["daily_return"] = 0
        else:
            prev = closes[idx - 1]
            new["daily_return"] = round((closes[idx] - prev) / prev, 6) if prev else 0
        if new["close_20_sma"] and float(row["volume"]):
            window = rows[max(0, idx - 19) : idx + 1]
            vol_sum = sum(float(x["volume"]) for x in window)
            new["vwma"] = round(
                sum(float(x["close"]) * float(x["volume"]) for x in window) / vol_sum,
                4,
            ) if vol_sum else None
        else:
            new["vwma"] = None
        out.append(new)
    return out
