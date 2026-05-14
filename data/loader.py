import csv
from pathlib import Path

from data.market_data import load_market_ohlcv


def load_ohlcv_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        needed = {"date", "open", "high", "low", "close", "volume"}
        names = {x.strip().lower() for x in reader.fieldnames or []}
        missing = needed - names
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
        for raw in reader:
            row = {k.strip().lower(): v for k, v in raw.items()}
            item = {
                "date": row["date"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            if item["high"] < item["low"]:
                raise ValueError("high cannot be lower than low")
            if item["close"] <= 0:
                raise ValueError("close must be positive")
            rows.append(item)
    if len(rows) < 2:
        raise ValueError("need at least two OHLCV rows")
    return rows


def load_ohlcv(source, start_date=None, end_date=None, provider=None):
    path = Path(str(source))
    if path.exists() and start_date is None and end_date is None:
        return load_ohlcv_csv(path)
    if start_date is None or end_date is None:
        raise ValueError("real market data requires both start_date and end_date")
    return load_market_ohlcv(str(source), start_date, end_date, provider=provider)
