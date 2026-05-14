# India Market Data Design

## Goal

Define an India-focused data stack for the portfolio committee simulator:

- NSE/BSE OHLCV for daily and intraday replay
- Indian company fundamentals and filings
- Indian business news and sentiment
- India macro context
- Broker/live data alternatives for future paper/live workflows
- Point-in-time rules for time-travel simulation

This document complements `docs/time_travel_simulator_design.md`.

## Recommended Stack

Start simple, then add stricter/paid sources only when needed.

```text
Phase 1:
  Market/technical: yfinance .NS/.BO + NSE/BSE bhavcopy fallback
  News/sentiment: GDELT + company IR/RSS + yfinance news
  Filings/events: NSE/BSE corporate announcements + MCA where practical
  Macro: RBI, MOSPI, FRED/World Bank fallback

Phase 2:
  Market/technical: DhanHQ adapter for Indian OHLCV/intraday
  News/sentiment: Indian finance RSS adapters + optional paid news API
  Fundamentals: Screener-style manual/import path or paid API after licensing review

Phase 3:
  Production/live: TrueData or GlobalDataFeeds
  Broker integration: Zerodha Kite / DhanHQ / Upstox / ICICI Breeze
```

## Symbol Conventions

Use canonical internal symbols and provider-specific mappings.

Examples:

```text
internal: RELIANCE
yfinance NSE: RELIANCE.NS
yfinance BSE: RELIANCE.BO
DhanHQ: securityId + exchangeSegment=NSE_EQ
Zerodha: instrument_token
Upstox: instrument_key
```

Create an instrument master:

```text
data/instruments/india_instruments.csv
  symbol
  company_name
  isin
  nse_symbol
  bse_code
  yfinance_nse
  yfinance_bse
  dhan_security_id
  zerodha_instrument_token
  upstox_instrument_key
  sector
  industry
```

Provider adapters should never guess identifiers at runtime if the instrument
master has a mapping.

## Market / Technical Analyst

Purpose:

- Daily and intraday OHLCV
- Volume
- Technical indicators computed locally
- Index context: NIFTY 50, BANKNIFTY, SENSEX, sector indices

### Sources

| Source | Fit | Cost/key | Use |
|---|---:|---:|---|
| `yfinance` `.NS` / `.BO` | Good for quick daily OHLCV | No key | Default research/backtest source. |
| NSE/BSE bhavcopy | Good for official EOD | No key, file-based | Daily backtests and validation. |
| DhanHQ historical/intraday | Strong free broker API candidate | Dhan credentials | Best next adapter for India OHLCV/intraday. |
| Zerodha Kite Connect | Strong broker API | Zerodha account, paid Connect for data | Good quality live/historical data. |
| Upstox | Usable broker API | Upstox credentials | Alternative broker data path; validate candle quality. |
| ICICI Breeze | Usable for ICICI users | ICICI credentials | Broker-dependent historical/live data. |
| TrueData | Production-grade | Paid/licensed | Authorized vendor for serious live systems. |
| GlobalDataFeeds | Production-grade | Paid/licensed | Authorized vendor for live + historical. |

### Recommendation

For this repo:

```text
Default: yfinance .NS/.BO
First India upgrade: DhanHQ
Official EOD validation: NSE/BSE bhavcopy
Production data: TrueData or GlobalDataFeeds
```

Implementation rules:

- Compute RSI/MACD/Bollinger/ATR/etc locally from visible OHLCV.
- Do not call provider-side indicator APIs in time-travel mode unless the
  provider call is cached and filtered as-of `sim_now`.
- For NSE/BSE daily bars, use exchange holidays/trading calendar, not calendar days.
- If using yfinance for Indian symbols, support `.NS`, `.BO`, and index symbols:

```text
^NSEI     NIFTY 50
^NSEBANK  BANKNIFTY, if available
^BSESN    SENSEX
```

## Fundamentals Analyst

Purpose:

- Financial statements
- Ratios
- Company profile
- Sector/peer context
- Shareholding and corporate actions where available

### Sources

| Source | Fit | Cost/key | Use |
|---|---:|---:|---|
| `yfinance` | Quick fallback | No key | Current snapshot fundamentals; not strict point-in-time. |
| NSE/BSE company filings/financial results | Strong official source | No key, adapter required | Point-in-time filings/results where available. |
| MCA / company filings | Official corporate filings | No/limited public access; adapter complexity | Longer-term official source. |
| Screener-style exports/imports | Very useful manually | Manual/export/licensing review | Good local ingestion path if user supplies exports. |
| Financial Modeling Prep / market data vendors | Structured API | Key/paid depending plan | Use only after India coverage/licensing review. |
| TrueData / GlobalDataFeeds corporate data | Production-quality | Paid/licensed | Best for commercial/product-grade fundamentals. |

### Recommendation

For the simulator:

```text
Strict mode:
  NSE/BSE corporate financial results and exchange announcements by published/accepted date

Fallback:
  yfinance fundamentals marked current_snapshot

Manual/import:
  CSV/JSON importer for fundamentals from licensed or user-provided datasets
```

Implementation rules:

- Label fundamentals as:

```text
strict_point_in_time
current_snapshot
manual_import
```

- If a balance sheet or ratio comes from current yfinance data during a 2022
  simulation, the agent report must say it is a current/revised fallback.
- For serious historical India fundamentals, prefer exchange filings/results
  or a paid vendor with point-in-time support.

## Filings / Corporate Events / Insider Activity

Purpose:

- Exchange announcements
- Board meetings
- Results announcements
- Corporate actions
- Shareholding disclosures
- Insider/promoter transactions where available

### Sources

| Source | Fit | Cost/key | Use |
|---|---:|---:|---|
| NSE corporate announcements | High | No key, adapter required | Primary event source for NSE listed companies. |
| BSE corporate announcements | High | No key, adapter required | Primary event source for BSE and broad coverage. |
| NSE/BSE corporate actions | High | No key, adapter required | Splits, dividends, bonuses, rights, etc. |
| Company investor relations RSS/pages | Medium/high | No key | Official company updates and press releases. |
| MCA | Medium | Access complexity | Corporate filings beyond exchange announcements. |
| Vendor corporate events APIs | High | Paid/licensed | Production-grade option. |

### Recommendation

Build exchange-announcement adapters before trying broad web scraping:

```text
NSEAnnouncementsProvider
BSEAnnouncementsProvider
CompanyIRProvider
```

Implementation rules:

- Store announcement timestamp, exchange, category, subject, company, symbol,
  attachment URL, and extracted text/snippet.
- Use only announcements with `published_at <= sim_now`.
- Corporate actions must adjust OHLCV when necessary or be logged as context.

## News / Sentiment Analyst

Purpose:

- Company news
- India market news
- Sector news
- Macro/regulatory news
- Sentiment signals

### Sources

| Source | Fit | Cost/key | Use |
|---|---:|---:|---|
| GDELT DOC | Good broad/free historical search | No key | First historical broad-news source. |
| yfinance news | Recent/current ticker news | No key | Recent capture/fallback, weak historical archive. |
| NSE/BSE announcements | Very high signal | No key | Treat as official news/events. |
| Company IR RSS/pages | High signal | No key | Official company source. |
| Curated finance RSS | Good live capture | Usually no key | Moneycontrol/ET/LiveMint/etc only if terms/RSS permit. |
| Alpha Vantage NEWS_SENTIMENT | Possible fallback | Key, low free quota | Check India ticker coverage before relying. |
| Polygon/Massive News | Low/uncertain for India equities | Key | Strong for US; only use if India coverage proves useful. |
| Paid Indian news/data providers | High | Paid/licensed | Best for product-grade news/sentiment. |

### Recommendation

For India:

```text
Historical/backfilled:
  GDELT + NSE/BSE announcements + company IR pages

Live capture:
  yfinance news + GDELT + RSS/IR + exchange announcements

Strict future replay:
  our own captured archive with observed_at timestamps
```

Do not rely on Polygon as a primary India news source unless testing confirms
coverage for NSE/BSE symbols. Polygon is excellent for US stocks; India coverage
is not the main reason to use it.

Implementation rules:

- Prefer source diversity:

```text
official exchange/company source > finance news API > GDELT/RSS > yfinance
```

- Store only metadata/snippets unless license permits full article storage.
- Dedupe aggressively by URL/title/publisher/date.
- Normalize sentiment internally:

```text
bullish, bearish, neutral, mixed
score: -1.0 to +1.0
```

## Macro / Global Context

Purpose:

- RBI policy/rates
- CPI/WPI/inflation
- GDP/IIP
- INR/USD
- crude oil/gold
- FII/DII flows if available
- global risk context

### Sources

| Source | Fit | Cost/key | Use |
|---|---:|---:|---|
| RBI Database / RBI releases | High | No key | Rates, liquidity, monetary policy, macro releases. |
| MOSPI | High | No key | CPI, GDP, IIP and official statistics. |
| FRED | Medium | No key/key depending usage | INR/USD, rates/global macro proxies. |
| World Bank / IMF APIs | Medium | No key | Slow macro context, not daily trading signal. |
| Yahoo/yfinance | Medium | No key | USDINR, crude, gold, global indices proxies. |
| GDELT | Medium | No key | Macro/news context. |
| Paid vendors | High | Paid | Better if we need clean historical release calendars. |

### Recommendation

For early India simulator:

```text
Macro prices/proxies:
  yfinance for USDINR, crude/gold proxies, global indices

Official macro context:
  RBI + MOSPI adapters later

News macro:
  GDELT
```

Implementation rules:

- Macro releases are point-in-time only if we know release date/time.
- Observation dates are not release dates. For example, March CPI may be
  published in April; it must not be visible before publication.
- If release timestamps are unavailable, label the series as `current_snapshot`
  or `approx_publication_time`.

## Broker / Live Integration Candidates

This is separate from historical simulation but relevant for future paper/live product work.

| Broker/API | Fit | Notes |
|---|---:|---|
| DhanHQ | High | Good candidate for India because it exposes historical/intraday data and broker APIs. |
| Zerodha Kite Connect | High | Mature and widely used; data access on paid Connect tier. |
| Upstox | Medium/high | Accessible APIs; validate data quality. |
| ICICI Breeze | Medium | Good if the user is already in ICICI ecosystem. |
| TrueData/GlobalDataFeeds | High for market data, not broker execution | Use for licensed production data. |

Recommended path:

```text
Data-first:
  yfinance -> DhanHQ -> TrueData/GlobalDataFeeds

Execution later:
  DhanHQ or Zerodha, depending account/API availability
```

## Time-Travel Labels

Every India data payload must carry one label:

```text
strict_point_in_time
publication_time
approx_publication_time
current_snapshot
manual_import
simulated_or_derived
```

Examples:

```text
NSE announcement with timestamp <= sim_now:
  strict_point_in_time

GDELT article published before sim_now but fetched later:
  publication_time

yfinance balance sheet used for 2021 replay:
  current_snapshot

RSI computed from visible OHLCV:
  simulated_or_derived
```

## Adapter Priority

Build order for India support:

1. `YFinanceIndiaProvider`
   - `.NS`, `.BO`, Indian indices
   - already mostly supported by current provider layer

2. `BhavcopyProvider`
   - EOD file ingestion/cache for NSE/BSE
   - good validation against yfinance

3. `DhanHQProvider`
   - daily and intraday OHLCV
   - requires user credentials

4. `ExchangeAnnouncementsProvider`
   - NSE/BSE announcements and corporate actions
   - core for event/fundamental context

5. `GDELTIndiaNewsProvider`
   - company and macro news search

6. `CompanyIRProvider`
   - configurable RSS/pages per symbol

7. `RBI/MOSPIProvider`
   - India macro context

8. Production data adapters
   - TrueData / GlobalDataFeeds

## User Inputs Needed

No keys needed for the first India mode if we use:

```text
yfinance
bhavcopy files
GDELT
NSE/BSE public announcements
RBI/MOSPI public data
```

Optional credentials later:

```env
DHAN_CLIENT_ID=
DHAN_ACCESS_TOKEN=
ZERODHA_API_KEY=
ZERODHA_ACCESS_TOKEN=
UPSTOX_ACCESS_TOKEN=
ICICI_BREEZE_API_KEY=
TRUEDATA_USERNAME=
TRUEDATA_PASSWORD=
GLOBALDATAFEEDS_API_KEY=
```

For our current repo, the most useful optional India credential would be DhanHQ.

## Recommendation

For an India-market version of the simulator, do this:

```text
First:
  yfinance .NS/.BO price replay
  GDELT + NSE/BSE announcements for news/events
  local technical indicators
  explicit time-travel labels

Next:
  DhanHQ OHLCV/intraday adapter
  Bhavcopy EOD validator
  exchange announcement store

Later:
  TrueData/GlobalDataFeeds for licensed production data
  broker integration through DhanHQ or Zerodha
```

This gives us a practical Indian market path without pretending unofficial
free sources are production-grade.
