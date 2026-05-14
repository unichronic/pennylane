import pytest

from data.loader import load_ohlcv
from data.market_data import load_market_ohlcv


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_twelvedata_provider_parses_daily_rows(monkeypatch):
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "test-key")

    def fake_get(url, params, timeout):
        assert url == "https://api.twelvedata.com/time_series"
        assert params["symbol"] == "AAPL"
        assert params["interval"] == "1day"
        assert timeout == 30
        return FakeResponse(
            {
                "values": [
                    {
                        "datetime": "2024-03-01",
                        "open": "180.00",
                        "high": "182.00",
                        "low": "179.50",
                        "close": "181.25",
                        "volume": "1000",
                    },
                    {
                        "datetime": "2024-03-04",
                        "open": "181.25",
                        "high": "183.00",
                        "low": "180.00",
                        "close": "182.50",
                        "volume": "1100",
                    },
                ]
            }
        )

    monkeypatch.setattr("data.market_data.requests.get", fake_get)

    rows = load_market_ohlcv("aapl", "2024-03-01", "2024-03-05", provider="twelvedata")

    assert rows == [
        {
            "date": "2024-03-01",
            "open": 180.0,
            "high": 182.0,
            "low": 179.5,
            "close": 181.25,
            "volume": 1000.0,
        },
        {
            "date": "2024-03-04",
            "open": 181.25,
            "high": 183.0,
            "low": 180.0,
            "close": 182.5,
            "volume": 1100.0,
        },
    ]


def test_alpha_vantage_provider_parses_and_sorts_daily_rows(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")

    def fake_get(url, params, timeout):
        assert url == "https://www.alphavantage.co/query"
        assert params["function"] == "TIME_SERIES_DAILY_ADJUSTED"
        assert params["symbol"] == "AAPL"
        return FakeResponse(
            {
                "Time Series (Daily)": {
                    "2024-03-04": {
                        "1. open": "181.25",
                        "2. high": "183.00",
                        "3. low": "180.00",
                        "4. close": "182.50",
                        "6. volume": "1100",
                    },
                    "2024-03-01": {
                        "1. open": "180.00",
                        "2. high": "182.00",
                        "3. low": "179.50",
                        "4. close": "181.25",
                        "6. volume": "1000",
                    },
                }
            }
        )

    monkeypatch.setattr("data.market_data.requests.get", fake_get)

    rows = load_market_ohlcv("AAPL", "2024-03-01", "2024-03-05", provider="alpha_vantage")

    assert [row["date"] for row in rows] == ["2024-03-01", "2024-03-04"]
    assert rows[-1]["close"] == 182.5


def test_explicit_keyed_provider_fails_loudly_without_key(monkeypatch):
    monkeypatch.delenv("TWELVE_DATA_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="TWELVE_DATA_API_KEY"):
        load_market_ohlcv("AAPL", "2024-03-01", "2024-03-05", provider="twelvedata")


def test_loader_keeps_csv_input_without_provider(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text(
        "date,open,high,low,close,volume\n"
        "2024-03-01,1,2,1,1.5,100\n"
        "2024-03-04,1.5,2.5,1.4,2,200\n"
    )

    rows = load_ohlcv(path)

    assert len(rows) == 2
    assert rows[-1]["close"] == 2
