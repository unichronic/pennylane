import logging
import os
from datetime import datetime
from typing import Callable

import requests

from config import get_config
from data.y_finance import load_yfinance_ohlcv

logger = logging.getLogger(__name__)

PROVIDER_ALIASES = {
    "yahoo": "yfinance",
    "yf": "yfinance",
    "yfinance": "yfinance",
    "twelve": "twelvedata",
    "twelve_data": "twelvedata",
    "twelvedata": "twelvedata",
    "alpha": "alpha_vantage",
    "alphavantage": "alpha_vantage",
    "alpha_vantage": "alpha_vantage",
}


def _validate_date(value: str, name: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{name} must be YYYY-mm-dd, got {value!r}") from exc


def _normalize_provider(name: str) -> str:
    provider = PROVIDER_ALIASES.get(str(name or "").strip().lower())
    if not provider:
        allowed = ", ".join(sorted(set(PROVIDER_ALIASES.values())))
        raise ValueError(f"unknown market data provider {name!r}; expected one of: {allowed}")
    return provider


def _provider_order(provider: str | None = None) -> list[str]:
    selected = provider or os.getenv("TRADEAGE_DATA_PROVIDER") or get_config().get(
        "market_data_provider", "auto"
    )
    selected = str(selected).strip().lower()
    if selected and selected != "auto":
        return [_normalize_provider(selected)]

    raw = os.getenv("TRADEAGE_DATA_PROVIDERS") or get_config().get(
        "market_data_providers", "yfinance,twelvedata,alpha_vantage"
    )
    providers = []
    for item in str(raw).split(","):
        item = item.strip()
        if item:
            normalized = _normalize_provider(item)
            if normalized not in providers:
                providers.append(normalized)
    return providers or ["yfinance"]


def _float(value, field: str) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field} value from market data provider: {value!r}") from exc


def _validate_rows(rows: list[dict], symbol: str, start_date: str, end_date: str) -> list[dict]:
    start = _validate_date(start_date, "start_date")
    end = _validate_date(end_date, "end_date")
    if start >= end:
        raise ValueError("start_date must be before end_date")

    cleaned = []
    for raw in rows:
        date = str(raw.get("date", ""))[:10]
        if not date:
            continue
        row_dt = _validate_date(date, "date")
        if not (start <= row_dt < end):
            continue
        item = {
            "date": date,
            "open": _float(raw.get("open"), "open"),
            "high": _float(raw.get("high"), "high"),
            "low": _float(raw.get("low"), "low"),
            "close": _float(raw.get("close"), "close"),
            "volume": _float(raw.get("volume", 0), "volume"),
        }
        if item["high"] < item["low"]:
            raise ValueError("high cannot be lower than low")
        if item["close"] <= 0:
            raise ValueError("close must be positive")
        cleaned.append(item)

    cleaned.sort(key=lambda row: row["date"])
    if len(cleaned) < 2:
        raise ValueError(
            f"need at least two OHLCV rows for {symbol.upper()} from {start_date} to {end_date}"
        )
    return cleaned


def _request_json(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("market data provider returned a non-object JSON response")
    return data


def _load_twelvedata(symbol: str, start_date: str, end_date: str) -> list[dict]:
    api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        raise RuntimeError("TWELVE_DATA_API_KEY is required for Twelve Data")

    data = _request_json(
        "https://api.twelvedata.com/time_series",
        {
            "symbol": symbol.upper(),
            "interval": "1day",
            "start_date": start_date,
            "end_date": end_date,
            "order": "ASC",
            "outputsize": 5000,
            "apikey": api_key,
        },
    )
    if data.get("status") == "error" or "values" not in data:
        raise RuntimeError(data.get("message") or data.get("code") or "Twelve Data returned no values")

    rows = [
        {
            "date": item.get("datetime"),
            "open": item.get("open"),
            "high": item.get("high"),
            "low": item.get("low"),
            "close": item.get("close"),
            "volume": item.get("volume", 0),
        }
        for item in data["values"]
    ]
    return _validate_rows(rows, symbol, start_date, end_date)


def _load_alpha_vantage(symbol: str, start_date: str, end_date: str) -> list[dict]:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY is required for Alpha Vantage")

    data = _request_json(
        "https://www.alphavantage.co/query",
        {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol.upper(),
            "outputsize": "full",
            "apikey": api_key,
        },
    )
    if "Error Message" in data or "Note" in data or "Information" in data:
        raise RuntimeError(data.get("Error Message") or data.get("Note") or data.get("Information"))

    series = data.get("Time Series (Daily)")
    if not isinstance(series, dict):
        raise RuntimeError("Alpha Vantage returned no daily time series")

    rows = []
    for day, values in series.items():
        close = _float(values.get("4. close"), "close")
        adjusted_close = _float(values.get("5. adjusted close", close), "adjusted close")
        factor = adjusted_close / close if close else 1.0
        rows.append(
            {
                "date": day,
                "open": _float(values.get("1. open"), "open") * factor,
                "high": _float(values.get("2. high"), "high") * factor,
                "low": _float(values.get("3. low"), "low") * factor,
                "close": adjusted_close,
                "volume": values.get("6. volume"),
            }
        )
    return _validate_rows(rows, symbol, start_date, end_date)


def _load_yfinance(symbol: str, start_date: str, end_date: str) -> list[dict]:
    return load_yfinance_ohlcv(symbol, start_date, end_date)


PROVIDER_LOADERS: dict[str, Callable[[str, str, str], list[dict]]] = {
    "yfinance": _load_yfinance,
    "twelvedata": _load_twelvedata,
    "alpha_vantage": _load_alpha_vantage,
}


def load_market_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
    provider: str | None = None,
) -> list[dict]:
    """Load OHLCV rows from the configured data providers.

    ``auto`` just walks the provider list in order. Missing optional keys are
    skipped. Real failures get collected and raised at the end.
    """
    errs = []
    for name in _provider_order(provider):
        loader = PROVIDER_LOADERS[name]
        try:
            return loader(symbol, start_date, end_date)
        except RuntimeError as exc:
            missing_optional_key = (
                provider is None
                and name in {"twelvedata", "alpha_vantage"}
                and "API_KEY is required" in str(exc)
            )
            if missing_optional_key:
                logger.info("Skipping %s because its API key is not configured", name)
                continue
            errs.append(f"{name}: {exc}")
        except Exception as exc:
            errs.append(f"{name}: {exc}")

    msg = "; ".join(errs) if errs else "no configured providers were usable"
    raise RuntimeError(
        f"Unable to load market data for {symbol.upper()} from {start_date} to {end_date}: {msg}"
    )
