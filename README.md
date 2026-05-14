# Penny Lane Capital

Penny Lane Capital is a multi-agent trading research project. It was inspired by the original multi-agent trading paper and explores what that kind of system looks like as working code: analyst roles, research debate, a trader, risk review, and portfolio approval around market data.

This is research software. It is not financial advice, not a production trading system, and not connected to a live brokerage account.

## What it does

The project runs a decision pipeline for a ticker symbol:

```text
OHLCV data
  -> market / sentiment / news / fundamentals analyst reports
  -> bull and bear research debate
  -> research manager
  -> trader
  -> risk debate
  -> portfolio manager
  -> paper execution / evaluation
```

The main focus is agent orchestration, prompt handoffs, traceability, and paper-style evaluation in a financial setting. Outputs should be read as structured research notes, not as buy or sell recommendations.

## What works now

- Runs one-off ticker analyses from the command line.
- Supports paper-style walk-forward backtests over daily bars.
- Includes baseline strategies such as buy-and-hold, MACD, KDJ+RSI, ZMR, and SMA.
- Uses Agno Workflow for the main decision path with typed state snapshots and checkpoints.
- Produces analyst reports, debate traces, routing traces, final decisions, and full state logs.
- Supports Mistral-backed agents for real LLM runs and a local provider for deterministic tests.
- Persists workflow state, decision memory, optional vector memory, and LLM call traces under local cache/output paths.
- Provides a small React/Vite frontend that talks to the Python orchestration API.
- Includes pytest coverage for orchestration behavior, market data providers, report context, LLM observability, and regression cases.

## Architecture

The backend is organized around role-specific agents and shared workflow state:

- `agents/` contains the analyst, researcher, trader, debate, risk, portfolio, and LLM runtime code.
- `core/` contains workflow orchestration, conditional routing, checkpointing, state logging, memory reflection, and prompt/report context handling.
- `data/` contains OHLCV loading, indicators, yfinance integration, and market-data provider routing.
- `backtest/` contains the paper-style backtest engine and benchmark methodology.
- `api/` exposes the orchestration flow for the frontend.
- `frontend/` contains the React/Vite interface.
- `tests/` covers expected behavior and several no-silent-fallback checks.

Each agent role stays visible in the runtime instead of being folded into one large prompt. That makes a run easier to inspect: reports, debates, risk arguments, and final approvals can be traced separately.

## Data and runtime

The default market data path uses `yfinance`, with optional Twelve Data and Alpha Vantage fallbacks when keys are configured. No market-data key is required for the default yfinance path.

The default LLM provider is Mistral:

```env
LLM_PROVIDER=mistral
MISTRAL_API_KEY=...
MISTRAL_QUICK_MODEL=mistral-small-2603
MISTRAL_DEEP_MODEL=mistral-large-2512
```

`MISTRAL_MODEL` is still accepted as a backward-compatible fallback, but the
role-specific variables are preferred. Quick roles produce analyst reports,
debate arguments, and trader drafts; deep roles make the research-manager and
portfolio-manager synthesis decisions.

For longer backtests, keep `decision_cadence_days=1` when you need a strict
paper-style daily committee run. Increase it to run the full committee every N
trading days and mark-to-market between decisions.

For deterministic local runs and tests, set:

```env
LLM_PROVIDER=local
```

Useful runtime paths:

```env
TRADEAGE_WORKFLOW_DB=data_cache/workflows.db
TRADEAGE_MEMORY_PATH=data_cache/decision_memory.jsonl
TRADEAGE_MEMORY_VECTOR_PATH=data_cache/decision_memory.vectors.jsonl
TRADEAGE_LLM_TRACE_PATH=data_cache/llm_calls.jsonl
TRADEAGE_RESULTS_DIR=results
```

LLM calls fail when the configured provider is missing required credentials. There is no automatic fallback from Mistral to local mode.

## Frontend

The frontend is a React/Vite app in `frontend/`. It calls the Python API through the Vite development proxy at `/api`.

Typical development flow:

```bash
python -m api.server

cd frontend
npm install
npm run dev
```

Or start both together:

```bash
./start.sh
```

The UI is for inspecting runs and orchestration output. It does not place trades.

## Quickstart

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Create local environment settings:

```bash
cp .env.example .env
```

Run a deterministic local decision:

```bash
LLM_PROVIDER=local python main.py AAPL
```

Start backend + frontend together:

```bash
./start.sh
```

Run with a date window:

```bash
LLM_PROVIDER=local python main.py AAPL --start 2024-03-01 --end 2024-04-15
```

Run a paper-style backtest:

```bash
LLM_PROVIDER=local python main.py AAPL --backtest
```

Run a baseline:

```bash
python main.py AAPL --baseline macd
```

Run tests:

```bash
python -m pytest -q
```

Run the validation harness:

```bash
python scripts/validation_harness.py --llm-provider local --skip-large-experiment
```

Run a reward-scored research loop:

```bash
python scripts/reward_loop.py \
  --run-id mistral-reward-30 \
  --llm-provider mistral \
  --loops 30 \
  --self-improve \
  --continue-on-error
```

The reward loop runs end-to-end committee backtests across multiple symbols,
periods, decision cadences, and short-execution policies. Each trial is scored
against buy-and-hold, MACD, KDJ+RSI, ZMR, and SMA baselines, then written to
`runs/reward_loop/<run-id>/trials.jsonl`. It also maintains
`leaderboard.json` and `best_config.json`. The default 30-loop Mistral run is
long-running by design; use `--llm-provider local --loops 1` for a smoke test.

## Limitations

- This is a research project, not an investment product.
- The system does not execute live trades.
- Results depend on market-data availability, LLM behavior, prompt design, and configured date windows.
- Free data providers can return missing, delayed, revised, or rate-limited data.
- Backtests are paper-style simulations and should not be treated as evidence of future performance.
- The agent architecture is built for inspection and evaluation, not low-latency execution.
- Fundamentals and news coverage are limited by the selected data vendors.
- Local LLM mode is useful for repeatable tests, but it is not a substitute for real model reasoning.

## Disclaimer

Penny Lane Capital is educational research software. It does not provide financial, investment, tax, or legal advice. Outputs are experimental model-generated research notes and may be incomplete or wrong. Do not use this project as the basis for real trading or investment decisions.
