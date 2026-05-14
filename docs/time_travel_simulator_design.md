# Time Travel Simulator Design

## Goal

Build a mode where the system treats a historical date as "now", receives only the market/news/fundamental data that would have been visible by that simulated time, makes one decision, advances the clock, and repeats as if it were running live.

The target user experience:

```bash
LLM_PROVIDER=local python main.py AAPL --time-travel \
  --sim-start 2024-03-01 \
  --sim-end 2024-04-15 \
  --step 1d \
  --lookback-days 120
```

At each simulated day:

```text
sim_now = 2024-03-01
visible market window = rows <= 2024-03-01
visible news window = articles published/known <= 2024-03-01
agent decision -> risk gate -> simulated fill -> audit log
advance sim_now to next trading day
```

This is different from the current `run_backtest` only in architecture and data boundaries. The existing backtest already uses a rolling price window. The new simulator makes that boundary explicit for every data type and adds a live-like event loop.

## Research Summary

I reviewed these references:

- `reference/SEXTANT`, cloned from `https://github.com/raphaub-hub/SEXTANT.git`. It is the closest local reference: a strict bar-by-bar event loop, `MarketEvent -> SignalEvent -> OrderEvent -> FillEvent`, and a JSON audit trail.
- NautilusTrader: strong model for research-to-live parity, deterministic event-driven replay, custom/raw market data loading, and a single core for backtest/live.
- Zipline: classic Python event-driven backtesting API where historical data is streamed through an algorithm over a date range.
- `trade-replay` on PyPI: a simpler candle-by-candle replay concept focused on avoiding hindsight bias.
- SEC EDGAR APIs: free, no-key company filings/submissions updated throughout the day in real time.
- FRED/ALFRED: relevant for macro data; ALFRED-style vintage data is the correct pattern when avoiding look-ahead from revised economic data.

Design choice: use SEXTANT's event-loop shape as inspiration, but keep our system's smaller modules and agent pipeline. Do not adopt NautilusTrader or Zipline as dependencies now; they are heavier than this repo needs.

## Hard Constraint: Historical News

Historical price replay is easy because `yfinance` can fetch dated OHLCV.

Historical news is harder:

- Live websites show today's web, not the web as it existed on `2024-03-01`.
- Scraping Yahoo/Google/news sites today and filtering by article publish date is useful, but not fully point-in-time correct because articles can be edited, removed, republished, or discovered later.
- Strict historical simulation needs an archive with article `published_at` and ideally `observed_at` or "first seen" time.

Therefore news must support modes:

1. `strict`: only use articles whose `observed_at <= sim_now`. This is unbiased but requires our own captured archive or a provider that exposes first-seen timestamps.
2. `publication_time`: use articles where `published_at <= sim_now`, even if we discovered them later. Useful for experiments, but must be flagged as backfilled.
3. `disabled`: price-only replay, which is honest and deterministic.

For the current repo, implement `disabled` and `publication_time` first. Add `strict` once we have enough captured articles.

## Current System Fit

Existing pieces:

- `data.loader.load_ohlcv(symbol, start, end, provider)` loads daily bars.
- `data.indicators.add_indicators(rows)` enriches bars.
- `core.pipeline.run_pipeline(...)` runs one latest-window decision.
- `backtest.engine.run_backtest(...)` already walks forward day by day using `rows[:idx + 1]`.
- `core.state.execute_trade(...)` simulates execution.
- `core.logger.log_stage(...)` writes JSONL stage logs.

Missing pieces:

- No explicit simulated clock.
- No event schema.
- No news store or news ingestion.
- No "as-of" data portal for all data types.
- No audit trail proving what the agent could see at each decision.
- No agent prompt context that says "simulated current date is X".

## Proposed Architecture

```text
simulator/
  clock.py           SimClock, trading-day calendar, step size
  events.py          MarketEvent, NewsEvent, DecisionEvent, OrderEvent, FillEvent
  data_portal.py     AsOfDataPortal: price/news/fundamental access bounded by sim_now
  news_store.py      SQLite-backed article store
  news_providers.py  yfinance/GDELT/SEC/RSS adapters
  runner.py          TimeTravelRunner orchestration
  audit.py           JSONL audit writer
  cli.py             optional command grouping later
```

### Event Flow

```text
Clock.tick()
  -> MarketEvent(symbol, sim_now, latest_bar, visible_window)
  -> NewsEvent(symbol, sim_now, visible_articles)
  -> AgentDecisionEvent(trace from analyst/debate/trader/risk)
  -> OrderEvent(action, requested_size, approved_size)
  -> FillEvent(price, shares, portfolio_after)
  -> AuditEvent(all input hashes + output)
```

The loop should be deterministic when news ingestion is disabled or when using a frozen news DB snapshot.

## Analyst Data Source Plan

The simulator should expand beyond the original paper's default of mostly
`yfinance`. Every data family should flow through `AsOfDataPortal` so agents
cannot access future or current-revised data by accident.

### Market / Technical Analyst

Purpose:

- Price action
- OHLCV windows
- Technical indicators
- Volatility and volume context

Sources:

| Source | Role | Notes |
|---|---|---|
| `yfinance` | Default OHLCV source | Enough for the first simulator; already integrated and cached. |
| Twelve Data | Optional OHLCV fallback | Good free-key fallback if we later need official API behavior. |
| Alpha Vantage | Optional OHLCV/indicator fallback | Free quota is small; useful as a second fallback. |
| Massive/Polygon Aggregates | Later production-grade OHLCV | Better API and historical bars if we use Polygon more broadly. |

Implementation rule:

- Compute indicators locally from the visible OHLCV window whenever possible.
- Do not let technical indicators call a provider directly in time-travel mode.
- The only bars visible at `sim_now` are rows with `date <= sim_now`.

### Fundamentals Analyst

Purpose:

- Company profile
- Balance sheet
- Income statement
- Cash flow
- Fundamental ratios
- Financial history and trend context

Sources:

| Source | Role | Notes |
|---|---|---|
| SEC EDGAR | Primary point-in-time source | Free, official, strong for 10-K/10-Q/8-K and company facts. Use acceptance timestamps. |
| `yfinance` | Fast fallback/preview source | Convenient, but can expose current/revised fundamentals; not strict historical by default. |
| Alpha Vantage Fundamentals | Optional structured fallback | Needs key; free quota is small. |
| Financial Modeling Prep | Possible later structured fundamentals | Useful API shape, but licensing/plan must be reviewed before product use. |

Implementation rule:

- In strict time-travel mode, prefer SEC filings accepted on or before `sim_now`.
- Mark yfinance/FMP/Alpha Vantage fundamentals as `current_snapshot` unless the source proves point-in-time behavior.
- The agent-facing report should explicitly say whether fundamentals are strict point-in-time or current/revised fallback data.

### Filings / Corporate Events / Insider Activity

Purpose:

- 8-K events
- 10-Q / 10-K filings
- Form 4 insider transactions
- Corporate actions and major official disclosures

Sources:

| Source | Role | Notes |
|---|---|---|
| SEC EDGAR submissions/companyfacts | Primary | No key; strongest point-in-time foundation. |
| SEC Form 4 | Primary insider data | Better than yfinance for strict replay. |
| `yfinance` insider transactions | Fallback | Convenient but not enough for strict historical replay. |
| Alpha Vantage insider transactions | Optional fallback | Keyed; use if available and cached. |

Implementation rule:

- Filter by SEC `acceptanceDateTime <= sim_now`.
- Store filing metadata, accession number, form type, accepted timestamp, document URL, and extracted snippet/summary.
- Treat filings as first-class events, separate from general news articles.

### Macro / Global Context

Purpose:

- Rates
- Inflation
- Employment
- GDP/industrial production
- Liquidity and recession indicators
- Macro regime context

Sources:

| Source | Role | Notes |
|---|---|---|
| ALFRED | Primary strict macro source | Vintage data avoids revised-data look-ahead bias. |
| FRED | Current/latest macro source | Good for live/current analysis; revised historical values can leak hindsight. |
| GDELT | Global macro news context | Free, broad, noisy; useful after filtering. |
| Alpha Vantage global news/topics | Optional macro news source | Keyed, rate-limited. |

Implementation rule:

- For strict replay, use release/vintage dates, not only observation dates.
- If using latest FRED values for a historical simulation, mark them as revised/current-snapshot data.

### News / Sentiment Analyst

Purpose:

- Ticker-specific news
- Macro news
- Public sentiment
- Publisher/source diversity
- Source-level sentiment signals where available

Sources:

| Source | Role | Notes |
|---|---|---|
| Massive/Polygon News | Primary ticker news API | Clean date filters and ticker mappings. |
| Alpha Vantage `NEWS_SENTIMENT` | Secondary news/sentiment | Good fields, small free quota. |
| GDELT | Broad news | Good for diversity/macro, requires filtering. |
| SEC EDGAR | Official company event source | Not general news but crucial for market-moving disclosures. |
| `yfinance` news | Recent Yahoo news fallback | Weak historical behavior. |
| Company IR/RSS | Official company updates | Adapter-by-adapter, terms-aware. |
| Curated finance RSS | Live capture | Good going forward; not enough for old dates unless archived. |

Implementation rule:

- Normalize sentiment into one internal scale.
- Keep source-provided sentiment separately.
- Never hide whether an article is strict observed-time data or backfilled by publication date.

### Data Quality Labels

Every source payload entering an agent report should carry one of these labels:

```text
strict_point_in_time      observed/accepted/released by sim_now
publication_time         published by sim_now but discovered/backfilled later
current_snapshot         current/revised data used as fallback
simulated_or_derived     locally computed from visible data
```

The audit log must store these labels. Agent prompts should mention them when
non-strict data is present.

### AsOfDataPortal

The data portal is the anti-lookahead boundary. Agents must never read provider APIs directly in time-travel mode.

Interface:

```python
class AsOfDataPortal:
    def get_price_window(self, symbol: str, sim_now: date, lookback_days: int) -> list[dict]: ...
    def get_latest_bar(self, symbol: str, sim_now: date) -> dict: ...
    def get_news(self, symbol: str, sim_now: datetime, lookback_days: int) -> list[Article]: ...
    def get_filings(self, symbol: str, sim_now: datetime, lookback_days: int) -> list[Filing]: ...
```

Rules:

- `get_price_window`: only rows with `date <= sim_now`.
- `get_news` strict mode: `published_at <= sim_now` and `observed_at <= sim_now`.
- `get_news` publication-time mode: `published_at <= sim_now`, `observed_at` may be later; audit must include `backfilled=true`.
- `get_filings`: SEC `acceptanceDateTime <= sim_now`.
- Macro data later: use vintage release date, not revised observation date.

### News Store

Use SQLite first. It is enough for local simulation, deterministic tests, and portable snapshots.

Tables:

```sql
news_articles(
  id text primary key,
  symbol text,
  source text,
  title text not null,
  summary text,
  url text,
  publisher text,
  published_at text,
  observed_at text not null,
  fetched_at text not null,
  content_hash text,
  raw_json text
);

news_symbol_map(
  article_id text,
  symbol text,
  relevance real,
  primary key(article_id, symbol)
);

sim_runs(
  run_id text primary key,
  symbol text,
  sim_start text,
  sim_end text,
  mode text,
  created_at text,
  config_json text
);

sim_events(
  run_id text,
  seq integer,
  sim_time text,
  event_type text,
  payload_json text,
  primary key(run_id, seq)
);
```

### News and Sentiment Providers

Use a multi-source news/sentiment collector. The goal is broad coverage, but
each source must be a registered adapter with known fields, timestamps,
rate limits, provenance, and license/compliance notes.

Initial source set:

| Source | Role | Historical usefulness | Key needed | Notes |
|---|---|---:|---:|---|
| Massive/Polygon News | Primary ticker news API | High within plan history | Yes | Clean ticker filter, `published_utc`, publisher metadata, sentiment/insights. Best candidate for `publication_time` replay. |
| Alpha Vantage `NEWS_SENTIMENT` | Secondary market news + sentiment | High | Yes | Supports `tickers`, `topics`, `time_from`, `time_to`; free quota is small, so use with cache and narrow windows. |
| GDELT DOC | Broad global/macro/company news | Medium/high | No | Free and broad, but noisy. Use for macro context, company-name queries, and source diversity. Needs relevance scoring. |
| SEC EDGAR | Official filings/events | High | No | Not news, but essential point-in-time company event data: 8-K, 10-Q, 10-K, etc. |
| yfinance news | Recent Yahoo Finance news | Low/medium | No | Useful for live/recent capture, weak for historical replay. |
| Company IR RSS/pages | Company primary-source updates | Medium | No | Good for official press releases; each domain should be an explicit adapter or RSS config. |
| Curated finance RSS | Live capture feed | Low for old history, high going forward | Usually no | Use only where terms/RSS allow. Store metadata/snippets, not full copyrighted articles unless licensed. |

Do not scrape arbitrary websites aggressively. Prefer official APIs, RSS feeds,
SEC, and provider metadata. If HTML scraping is added, require per-domain
adapters, robots/terms review, low rate limits, and content snippets rather
than full article storage unless licensed.

### Multi-Source Aggregation

The simulator should query all enabled sources, normalize the results, dedupe
them, score relevance, then produce a compact news/sentiment report.

```text
Provider adapters
  -> NormalizedArticle[]
  -> dedupe by canonical_url/title_hash/source_id
  -> entity/ticker relevance scoring
  -> sentiment normalization
  -> time-travel visibility filter
  -> news_report for agents
```

Normalized article schema:

```python
class NormalizedArticle:
    id: str
    source: str
    source_type: str        # news_api, filing, rss, yahoo, gdelt
    symbol: str | None
    title: str
    summary: str
    url: str
    publisher: str
    published_at: datetime | None
    observed_at: datetime
    fetched_at: datetime
    sentiment_label: str | None  # bullish, bearish, neutral, mixed
    sentiment_score: float | None
    relevance_score: float
    backfilled: bool
    raw: dict
```

Visibility rules by replay mode:

- `strict`: `observed_at <= sim_now`; this is unbiased and works for our own captured archive.
- `publication_time`: `published_at <= sim_now`; useful for historical APIs, but log `backfilled=true` when `observed_at > sim_now`.
- `disabled`: no news context.

Aggregation rules:

- Keep source provenance in every agent-facing report.
- Cap articles per source and per day so one feed cannot dominate.
- Prefer primary sources in tie-breaks: SEC/company IR > paid news API > GDELT/RSS > yfinance.
- Dedupe by canonical URL first, then normalized title + publisher + date.
- Score ticker relevance using explicit ticker tags first; fall back to company-name/entity matching.
- Store source-specific sentiment, but normalize to one internal sentiment scale.
- Summarize only snippets/metadata unless we have rights to store full article text.

### Periodic Real-Time Capture

This is how we make future strict replay possible:

```bash
python main.py --news-capture --symbols AAPL,MSFT,NVDA --interval 15m
```

Behavior:

- Every interval, call configured providers.
- Store articles with `observed_at = wall_clock_now`.
- Dedupe by canonical URL and title hash.
- Keep raw provider metadata for audit.
- Later, a time-travel run can use strict mode for dates after our archive started.

Important: if we start capture on `2026-05-04`, then strict replay before `2026-05-04` still does not have true first-seen news unless a provider supplies that metadata.

## Agent Integration

Create a single-day function that accepts bounded context rather than calling global data loaders:

```python
def run_agent_day(
    symbol: str,
    visible_rows: list[dict],
    portfolio: dict,
    sim_now: str,
    news_report: str = "",
    sentiment_report: str = "",
    log_path: str | None = None,
) -> dict:
    ...
```

It will reuse:

- `run_analyst(visible_rows)`
- `run_debate_round(...)`
- `run_trader(...)`
- `run_risk_debate(...)`
- `run_risk_manager(...)`
- `execute_trade(...)`

The analyst and decision prompts should include:

```text
The current simulated date is YYYY-MM-DD. Treat this as the present.
Do not refer to data after this date. If news context is empty, say so.
```

The local deterministic LLM path should also accept `sim_now` for repeatable tests.

## Runner Algorithm

```python
rows = add_indicators(load_ohlcv(symbol, preload_start, sim_end_exclusive))
portfolio = make_portfolio(cash)
clock = SimClock(sim_start, sim_end, trading_dates=[row["date"] for row in rows])

for sim_now in clock:
    visible_rows = portal.get_price_window(symbol, sim_now, lookback_days)
    if len(visible_rows) < warmup:
        continue

    articles = portal.get_news(symbol, sim_now, news_lookback_days)
    news_report = summarize_articles(articles, sim_now)

    decision = run_agent_day(
        symbol=symbol,
        visible_rows=visible_rows,
        portfolio=portfolio,
        sim_now=sim_now,
        news_report=news_report,
    )

    execution = execute_trade(portfolio, decision["trader"], decision["risk"], visible_rows[-1])
    portfolio = execution["portfolio"]
    audit.write(...)
```

## CLI Design

Add flags:

```bash
python main.py AAPL --time-travel
python main.py AAPL --time-travel --sim-start 2024-03-01 --sim-end 2024-04-15
python main.py AAPL --time-travel --news-mode disabled
python main.py AAPL --time-travel --news-mode publication_time
python main.py AAPL --news-capture --symbols AAPL,MSFT --interval 15m
```

Recommended defaults:

```text
--time-travel                  false
--sim-start                    required in time-travel mode
--sim-end                      latest available market date if omitted
--lookback-days                120
--warmup-days                  20
--step                         1d
--news-mode                    disabled initially
--news-lookback-days           7
--audit-log                    runs/time-travel-<timestamp>.jsonl
```

## Audit Requirements

Every simulated decision must log:

- `run_id`
- `sim_now`
- `symbol`
- `price_window_start`, `price_window_end`, `n_price_rows`
- `latest_bar`
- `news_mode`
- `article_ids`
- `backfilled_article_count`
- `analyst`, `debate`, `trader`, `risk_debate`, `risk`
- `execution`
- `portfolio_after`
- hashes of visible price/news inputs

This is not optional. Without this, agentic backtests are hard to debug and easy to fool with accidental look-ahead.

## Implementation Phases

### Phase 1: Price-Only Time Travel

Scope:

- Add `simulator/clock.py`, `simulator/data_portal.py`, `simulator/runner.py`, `simulator/audit.py`.
- Add `run_agent_day(...)` wrapper.
- Add `--time-travel`, `--sim-start`, `--sim-end`, `--lookback-days`, `--news-mode disabled`.
- Reuse current yfinance OHLCV and existing simulated execution.

Acceptance tests:

- A run at `sim_now=2024-03-15` cannot see price rows after `2024-03-15`.
- Re-running with same cached data and `LLM_PROVIDER=local` produces identical audit logs except timestamps/run id.
- The final portfolio matches current `run_backtest` for equivalent price-only logic or has explained differences.

### Phase 2: News Store + Publication-Time Replay

Scope:

- SQLite `news_store.py`.
- Provider adapter interface plus `MassivePolygonNewsProvider`, `AlphaVantageNewsProvider`, `GDELTProvider`, `SECProvider`, `YFinanceNewsProvider`, and `RSSProvider`.
- Multi-source normalization, dedupe, relevance scoring, and sentiment normalization.
- `summarize_articles(...)` into a compact `news_report`.
- `--news-mode publication_time`.
- Source toggles such as `--news-sources polygon,alpha_vantage,gdelt,sec,yfinance,rss`.

Acceptance tests:

- Articles after `sim_now` are excluded.
- Articles discovered after `sim_now` are included only in `publication_time` mode and logged as backfilled.
- Empty news produces a clear "No available news in window" report, not fabricated news.
- Duplicate articles from multiple sources appear once in the agent report but retain all source references in audit.
- A source failure is logged and does not fail the day if at least one source succeeds.

### Phase 3: Real-Time News Capture

Scope:

- `--news-capture --symbols ... --interval ...`
- Dedupe, source throttling, provider error handling.
- Store `observed_at` and `fetched_at`.

Acceptance tests:

- Two capture runs with the same provider payload create one article.
- Captured articles become visible in strict mode only after `observed_at`.
- Per-provider failures do not crash the capture loop unless all providers fail.

### Phase 4: Strict News Replay + Filings

Scope:

- `--news-mode strict`.
- SEC filings provider using official SEC APIs.
- Optional GDELT backfill provider.

Acceptance tests:

- SEC filings are visible only after accepted timestamp.
- Strict mode never uses articles whose `observed_at > sim_now`.
- Audit clearly separates article news from regulatory filings.

### Phase 5: Live/Replay Parity

Scope:

- Use the same `AsOfDataPortal` interface for live and simulated modes.
- Live mode sets `sim_now = wall_clock_now`.
- Replay mode sets `sim_now = SimClock.current`.

Acceptance tests:

- Agent-day code is identical between live and replay.
- Only the clock and data portal implementation differ.

## Risks and Mitigations

- Look-ahead bias from news: enforce `sim_now` filtering in `AsOfDataPortal`; audit every article ID.
- LLM nondeterminism: use `LLM_PROVIDER=local` for tests and support low-temperature remote model config later.
- yfinance gaps/rate limits: cache price data and keep provider layer already added in `data/market_data.py`.
- News licensing/scraping: prefer APIs/RSS/metadata; do not store full copyrighted article bodies unless source license allows it.
- Slow agent loop: cache article summaries per `(symbol, sim_now, article_ids_hash)` and allow `--max-days`.

## Recommended Next Build

Start with Phase 1. It gives us the core time-travel mode without the news ambiguity. Then add the news store in Phase 2 with explicit `publication_time` labeling. That creates a useful simulator quickly while keeping the correctness boundary honest.

The smallest valuable first PR:

- `simulator/clock.py`
- `simulator/data_portal.py`
- `simulator/runner.py`
- `core/pipeline.py::run_agent_day`
- CLI flags in `main.py`
- tests for no future rows and deterministic local replay

The smallest useful news PR after that:

- `simulator/news_store.py`
- `simulator/news_providers/base.py`
- `simulator/news_providers/polygon.py`
- `simulator/news_providers/sec.py`
- `simulator/news_aggregator.py`
- `--news-mode publication_time --news-sources polygon,sec`
- tests for date filtering, dedupe, and audit provenance
