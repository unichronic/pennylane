from datetime import datetime, timedelta
from io import StringIO
import os

import pandas as pd

import requests
import yfinance as yf

from config import get_config
from data.indicators import add_indicators
from data.loader import load_ohlcv
from data.y_finance import get_YFin_data_online, yf_retry


def _rows_to_stock_csv(symbol, start_date, end_date, rows, source):
    buf = StringIO()
    buf.write(f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n")
    buf.write(f"# Total records: {len(rows)}\n")
    buf.write(f"# Source: {source}\n")
    buf.write("# Data retrieved from preloaded/provider rows; no future rows beyond this window are included.\n\n")
    buf.write("Date,Open,High,Low,Close,Volume\n")
    for row in rows:
        buf.write(
            f"{row['date']},{float(row['open']):.2f},{float(row['high']):.2f},"
            f"{float(row['low']):.2f},{float(row['close']):.2f},{float(row.get('volume', 0)):.0f}\n"
        )
    return buf.getvalue()


def get_stock_data(symbol, start_date, end_date, provider=None, rows=None):
    if rows is not None:
        return _rows_to_stock_csv(symbol, start_date, end_date, rows, "preloaded_rows")
    normalized = str(provider or "yfinance").lower()
    if normalized in {"", "auto", "yfinance", "yf", "yahoo"}:
        return get_YFin_data_online(symbol, start_date, end_date)
    provider_rows = load_ohlcv(symbol, start_date, end_date, provider=provider)
    return _rows_to_stock_csv(symbol, start_date, end_date, provider_rows, normalized)


def get_indicators(symbol, start_date, end_date, indicators=None, provider=None, rows=None):
    rows = add_indicators(rows if rows is not None else load_ohlcv(symbol, start_date, end_date, provider=provider))
    latest = rows[-1]
    names = indicators or [
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "close_20_sma",
        "close_50_sma",
        "close_10_ema",
        "boll",
        "boll_ub",
        "boll_lb",
        "atr",
        "stoch_k",
        "stoch_d",
        "mfi",
        "obv",
        "vwma",
        "daily_return",
    ]
    return {name: latest.get(name) for name in names}


class AlphaVantageRateLimitError(RuntimeError):
    pass


def _tool_vendor(method):
    return get_config().get("tool_vendors", {}).get(method, "yfinance").strip().lower()


def _alpha_vantage_request(function_name, params):
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY is required for Alpha Vantage tools")
    payload = dict(params)
    payload.update({"function": function_name, "apikey": api_key, "source": "trading_agents"})
    response = requests.get("https://www.alphavantage.co/query", params=payload, timeout=30)
    response.raise_for_status()
    text = response.text
    try:
        data = response.json()
    except ValueError:
        return text
    message = str(data.get("Information") or data.get("Note") or "")
    if "rate limit" in message.lower() or "api key" in message.lower():
        raise AlphaVantageRateLimitError(message)
    return data


def _format_datetime_for_alpha(date_value):
    if not date_value:
        return None
    return datetime.strptime(date_value, "%Y-%m-%d").strftime("%Y%m%dT0000")


def _start_date(curr_date, look_back_days):
    if not curr_date:
        return None
    return (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)).strftime("%Y-%m-%d")


def _alpha_news(symbol, curr_date=None, look_back_days=7, limit=10):
    start = _start_date(curr_date, look_back_days)
    params = {"tickers": symbol.upper(), "limit": str(limit)}
    if start:
        params["time_from"] = _format_datetime_for_alpha(start)
    if curr_date:
        params["time_to"] = _format_datetime_for_alpha(curr_date)
    return _alpha_vantage_request("NEWS_SENTIMENT", params)


def _alpha_global_news(curr_date=None, look_back_days=7, limit=10):
    start = _start_date(curr_date, look_back_days)
    params = {"topics": "financial_markets,economy_macro,economy_monetary", "limit": str(limit)}
    if start:
        params["time_from"] = _format_datetime_for_alpha(start)
    if curr_date:
        params["time_to"] = _format_datetime_for_alpha(curr_date)
    return _alpha_vantage_request("NEWS_SENTIMENT", params)


def _filter_alpha_reports_by_date(result, curr_date):
    if not curr_date or not isinstance(result, dict):
        return result
    out = dict(result)
    for key in ("annualReports", "quarterlyReports"):
        if key in out:
            out[key] = [row for row in out[key] if row.get("fiscalDateEnding", "") <= curr_date]
    return out


def _alpha_statement(symbol, function_name, curr_date=None):
    result = _filter_alpha_reports_by_date(
        _alpha_vantage_request(function_name, {"symbol": symbol.upper()}),
        curr_date,
    )
    if isinstance(result, dict):
        result = dict(result)
        result["data_timing"] = (
            "fiscal_period_filtered; reports after curr_date are removed, "
            "but fiscal period end date is not necessarily filing/publication date."
        )
    return result


def get_news(symbol, curr_date=None, look_back_days=7, limit=10):
    if _tool_vendor("get_news") == "alpha_vantage":
        return _alpha_news(symbol, curr_date=curr_date, look_back_days=look_back_days, limit=limit)

    try:
        ticker = yf.Ticker(symbol.upper())
        news = yf_retry(lambda: ticker.news)
        if not news:
            return f"No yfinance news returned for {symbol.upper()}"
    except Exception as exc:
        return f"Error fetching yfinance news for {symbol.upper()}: {exc}"

    cutoff = None
    current = None
    if curr_date:
        current = datetime.strptime(curr_date, "%Y-%m-%d")
        cutoff = current - timedelta(days=look_back_days)

    items = []
    for article in news[:limit]:
        content = article.get("content", article)
        title = content.get("title") or article.get("title") or "Untitled"
        publisher = content.get("provider", {}).get("displayName") or article.get("publisher") or "Unknown"
        pub_date = content.get("pubDate") or article.get("providerPublishTime")
        if isinstance(pub_date, int):
            pub_dt = datetime.fromtimestamp(pub_date)
        elif isinstance(pub_date, str):
            try:
                pub_dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                pub_dt = None
        else:
            pub_dt = None
        if current and pub_dt is None:
            continue
        if cutoff and pub_dt and pub_dt < cutoff:
            continue
        if current and pub_dt and pub_dt > current + timedelta(days=1):
            continue
        summary = content.get("summary") or article.get("summary") or ""
        items.append(f"- {title} ({publisher}) {summary}".strip())

    return "\n".join(items) if items else f"No recent yfinance news returned for {symbol.upper()}"


def get_global_news(curr_date=None, look_back_days=7, limit=10):
    if _tool_vendor("get_global_news") == "alpha_vantage":
        return _alpha_global_news(curr_date=curr_date, look_back_days=look_back_days, limit=limit)

    tickers = ["SPY", "QQQ", "DIA"]
    reports = []
    for ticker in tickers:
        reports.append(f"## {ticker}\n{get_news(ticker, curr_date=curr_date, look_back_days=look_back_days, limit=limit)}")
    return "\n\n".join(reports)


def get_fundamentals(symbol, curr_date=None):
    if _tool_vendor("get_fundamentals") == "alpha_vantage":
        result = _alpha_vantage_request("OVERVIEW", {"symbol": symbol.upper()})
        if isinstance(result, dict):
            result = dict(result)
            result["data_timing"] = "current_snapshot; provider overview fields are not guaranteed point-in-time."
        return result

    try:
        ticker = yf.Ticker(symbol.upper())
        info = yf_retry(lambda: ticker.get_info())
        if not info:
            return f"No yfinance fundamentals returned for {symbol.upper()}"
    except Exception as exc:
        return f"Error retrieving yfinance fundamentals for {symbol.upper()}: {exc}"
    lines = [
        "Data timing: current_snapshot; provider overview fields may be revised/current and are not guaranteed point-in-time.",
    ]
    keys = [
        "longName",
        "sector",
        "industry",
        "marketCap",
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "profitMargins",
        "revenueGrowth",
        "debtToEquity",
        "returnOnEquity",
    ]
    lines.extend(f"{key}: {info.get(key)}" for key in keys if info.get(key) is not None)
    return "\n".join(lines)


def _filter_financials_by_date(data, curr_date):
    if not curr_date or data is None or getattr(data, "empty", True):
        return data
    cutoff = pd.Timestamp(curr_date)
    parsed = pd.to_datetime(data.columns, errors="coerce")
    mask = parsed <= cutoff
    return data.loc[:, mask]


def _statement_to_text(symbol, attr, limit=6, curr_date=None):
    try:
        ticker = yf.Ticker(symbol.upper())
        data = yf_retry(lambda: getattr(ticker, attr))
        data = _filter_financials_by_date(data, curr_date)
        if data is None or getattr(data, "empty", True):
            return f"No yfinance {attr.replace('_', ' ')} returned for {symbol.upper()}"
        label = (
            "Data timing: fiscal_period_filtered; columns after curr_date are removed, "
            "but fiscal period end date is not necessarily filing/publication date."
        )
        return label + "\n" + data.iloc[:, :limit].to_string()
    except Exception as exc:
        return f"Error retrieving yfinance {attr.replace('_', ' ')} for {symbol.upper()}: {exc}"


def get_balance_sheet(symbol, limit=6, curr_date=None):
    if _tool_vendor("get_balance_sheet") == "alpha_vantage":
        return _alpha_statement(symbol, "BALANCE_SHEET", curr_date=curr_date)
    return _statement_to_text(symbol, "balance_sheet", limit=limit, curr_date=curr_date)


def get_cashflow(symbol, limit=6, curr_date=None):
    if _tool_vendor("get_cashflow") == "alpha_vantage":
        return _alpha_statement(symbol, "CASH_FLOW", curr_date=curr_date)
    return _statement_to_text(symbol, "cashflow", limit=limit, curr_date=curr_date)


def get_income_statement(symbol, limit=6, curr_date=None):
    if _tool_vendor("get_income_statement") == "alpha_vantage":
        return _alpha_statement(symbol, "INCOME_STATEMENT", curr_date=curr_date)
    return _statement_to_text(symbol, "financials", limit=limit, curr_date=curr_date)


def get_insider_transactions(symbol, limit=10, curr_date=None):
    if _tool_vendor("get_insider_transactions") == "alpha_vantage":
        return _alpha_vantage_request("INSIDER_TRANSACTIONS", {"symbol": symbol.upper()})

    try:
        ticker = yf.Ticker(symbol.upper())
        data = yf_retry(lambda: ticker.insider_transactions)
        if data is None or getattr(data, "empty", True):
            return f"No yfinance insider transactions returned for {symbol.upper()}"
        if curr_date:
            cutoff = pd.Timestamp(curr_date)
            date_columns = [
                column
                for column in data.columns
                if "date" in str(column).lower() or str(column).lower() in {"start", "end"}
            ]
            for column in date_columns:
                parsed = pd.to_datetime(data[column], errors="coerce")
                if parsed.notna().any():
                    data = data[parsed <= cutoff]
                    break
            else:
                return f"No yfinance insider transactions returned for {symbol.upper()} by {curr_date}"
        if getattr(data, "empty", True):
            return f"No yfinance insider transactions returned for {symbol.upper()} by {curr_date}"
        return data.head(limit).to_string()
    except Exception as exc:
        return f"Error retrieving yfinance insider transactions for {symbol.upper()}: {exc}"


def sentiment_from_news(news_report):
    positive_words = {"beat", "growth", "upgrade", "strong", "record", "surge", "profit", "gain"}
    negative_words = {"miss", "downgrade", "weak", "loss", "fall", "drop", "risk", "lawsuit"}
    text = news_report.lower()
    positive = sum(text.count(word) for word in positive_words)
    negative = sum(text.count(word) for word in negative_words)
    if positive > negative:
        label = "positive"
    elif negative > positive:
        label = "negative"
    else:
        label = "neutral"
    return f"News-derived sentiment: {label} (positive_terms={positive}, negative_terms={negative})"
