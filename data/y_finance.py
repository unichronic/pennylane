import csv
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from config import get_config

logger = logging.getLogger(__name__)

_TICKER_PATH_RE = re.compile(r"^[A-Za-z0-9._\-\^]+$")


def safe_ticker_component(value: str, *, max_len: int = 32) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"ticker must be a non-empty string, got {value!r}")
    if len(value) > max_len:
        raise ValueError(f"ticker exceeds {max_len} chars: {value!r}")
    if not _TICKER_PATH_RE.fullmatch(value):
        raise ValueError(
            f"ticker contains characters not allowed in a filesystem path: {value!r}"
        )
    if set(value) == {"."}:
        raise ValueError(f"ticker cannot consist solely of dots: {value!r}")
    return value


def yf_retry(func, max_retries=3, base_delay=2.0):
    """Retry a yfinance call when Yahoo rate-limits it."""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except YFRateLimitError:
            if attempt < max_retries:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Yahoo Finance rate limited, retrying in %.0fs (attempt %s/%s)",
                    delay,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(delay)
            else:
                raise


def _validate_date(value: str, name: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{name} must be YYYY-mm-dd, got {value!r}") from exc


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    data[price_cols] = data[price_cols].ffill().bfill()

    return data


def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    ticker = yf.Ticker(symbol.upper())
    data = yf_retry(lambda: ticker.history(start=start_date, end=end_date))
    if data.empty:
        return (
            f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
        )

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    num_cols = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in num_cols:
        if col in data.columns:
            data[col] = data[col].round(2)

    csv_txt = data.to_csv()
    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_txt


def _data_cache_dir() -> Path:
    cfg = get_config()
    return Path(os.getenv("TRADEAGE_DATA_CACHE_DIR", cfg.get("data_cache_dir", "data_cache")))


def load_yfinance_ohlcv(symbol: str, start_date: str, end_date: str) -> list[dict]:
    """Load yfinance OHLCV rows and cache them on disk."""
    start_dt = _validate_date(start_date, "start_date")
    end_dt = _validate_date(end_date, "end_date")
    if start_dt >= end_dt:
        raise ValueError("start_date must be before end_date")

    safe_symbol = safe_ticker_component(symbol.upper())
    cache_dir = _data_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{safe_symbol}-YFin-data-{start_date}-{end_date}.csv"

    if cache_file.exists():
        data = pd.read_csv(cache_file, on_bad_lines="skip", encoding="utf-8")
    else:
        data = yf_retry(
            lambda: yf.download(
                safe_symbol,
                start=start_date,
                end=end_date,
                multi_level_index=False,
                progress=False,
                auto_adjust=True,
            )
        )
        if data.empty:
            raise RuntimeError(
                f"No yfinance OHLCV data found for {safe_symbol} from {start_date} to {end_date}"
            )
        data = data.reset_index()
        data.to_csv(cache_file, index=False, encoding="utf-8")

    data = _clean_dataframe(data)
    data = data[(data["Date"] >= pd.Timestamp(start_date)) & (data["Date"] < pd.Timestamp(end_date))]
    if len(data) < 2:
        raise ValueError(
            f"need at least two real OHLCV rows for {safe_symbol} from {start_date} to {end_date}"
        )

    rows = []
    for _, raw in data.iterrows():
        item = {
            "date": raw["Date"].strftime("%Y-%m-%d"),
            "open": float(raw["Open"]),
            "high": float(raw["High"]),
            "low": float(raw["Low"]),
            "close": float(raw["Close"]),
            "volume": float(raw["Volume"]),
        }
        if item["high"] < item["low"]:
            raise ValueError("high cannot be lower than low")
        if item["close"] <= 0:
            raise ValueError("close must be positive")
        rows.append(item)
    return rows


def parse_yfinance_csv_text(text: str) -> list[dict]:
    lines = []
    capture = False
    for line in text.splitlines():
        if line.startswith("Date,"):
            capture = True
        if capture:
            if line.startswith("#"):
                break
            lines.append(line)
    if not lines:
        raise ValueError("no yfinance CSV block found")

    reader = csv.DictReader(lines)
    rows = []
    for raw in reader:
        rows.append(
            {
                "date": raw["Date"],
                "open": float(raw["Open"]),
                "high": float(raw["High"]),
                "low": float(raw["Low"]),
                "close": float(raw["Close"]),
                "volume": float(raw["Volume"]),
            }
        )
    if len(rows) < 2:
        raise ValueError("need at least two OHLCV rows")
    return rows
