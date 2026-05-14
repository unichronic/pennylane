from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from backtest.engine import run_backtest
from backtest.paper_methodology import (
    PAPER_END_DATE_EXCLUSIVE,
    PAPER_START_DATE,
    PaperBacktestConfig,
    run_baseline_backtest,
    run_paper_experiment,
)
from core.pipeline import run_pipeline
from main import resolve_live_date_range

BASE_AGENTS = [
    {"id": "load_market", "label": "Market Loader", "group": "data", "summary": "Fetches visible OHLCV rows"},
    {"id": "market", "label": "Market Analyst", "group": "analysis", "summary": "Price, indicators, volatility, volume"},
    {"id": "news", "label": "News Analyst", "group": "analysis", "summary": "Company, macro, insider context"},
    {"id": "sentiment", "label": "Sentiment Analyst", "group": "analysis", "summary": "Public sentiment and discussion proxies"},
    {"id": "fundamentals", "label": "Fundamentals Analyst", "group": "analysis", "summary": "Financial statements and company quality"},
    {"id": "bull", "label": "Bull Researcher", "group": "research", "summary": "Constructs pro-trade thesis"},
    {"id": "bear", "label": "Bear Researcher", "group": "research", "summary": "Challenges thesis and downside risk"},
    {"id": "research_manager", "label": "Research Manager", "group": "research", "summary": "Synthesizes investment plan"},
    {"id": "trader", "label": "Trader", "group": "trading", "summary": "Builds transaction proposal"},
    {"id": "risk_debate", "label": "Risk Debate", "group": "risk", "summary": "Aggressive, conservative, neutral review"},
    {"id": "portfolio_manager", "label": "Portfolio Manager", "group": "risk", "summary": "Final rating and action"},
    {"id": "execution", "label": "Paper Execution", "group": "execution", "summary": "Risk gate, fill, portfolio metrics"},
]

BASELINE_AGENTS = [
    {"id": "load_market", "label": "Market Loader", "group": "data", "summary": "Fetches the historical window for the baseline run"},
    {"id": "baseline", "label": "Baseline Strategy", "group": "trading", "summary": "Applies the selected rule-based benchmark"},
    {"id": "execution", "label": "Paper Execution", "group": "execution", "summary": "Tracks fills, equity, and drawdown over the window"},
]

EXPERIMENT_AGENTS = [
    {"id": "experiment_inputs", "label": "Experiment Loader", "group": "data", "summary": "Loads the market windows for each symbol in the batch"},
    {"id": "batch_runner", "label": "Benchmark Runner", "group": "execution", "summary": "Runs Penny Lane and the selected baselines across the batch"},
    {"id": "aggregate", "label": "Aggregate Results", "group": "execution", "summary": "Rolls per-symbol curves into one comparison table"},
]

_RUN_LOCK = threading.Lock()
_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}

_EXPERIMENT_STRATEGIES = ("PennyLaneCapital", "buy_and_hold", "macd", "kdj_rsi", "zmr", "sma")


def _agents_for_mode(mode: str) -> list[dict[str, Any]]:
    if mode == "baseline":
        return BASELINE_AGENTS
    if mode == "paper_experiment":
        return EXPERIMENT_AGENTS
    return BASE_AGENTS


def default_config() -> dict[str, Any]:
    start_date, end_date = resolve_live_date_range(today=date.today())
    sim_present = (date.fromisoformat(end_date) - timedelta(days=1)).strftime("%Y-%m-%d")
    return {
        "symbol": "AAPL",
        "symbols": "AAPL,GOOGL,AMZN",
        "mode": "live",
        "startDate": start_date,
        "endDate": end_date,
        "cash": 10000,
        "dataProvider": os.getenv("TRADEAGE_DATA_PROVIDER", "yfinance"),
        "baseline": "buy_and_hold",
        "llmProvider": os.getenv("LLM_PROVIDER", "local"),
        "newsMode": "disabled",
        "newsSources": "yfinance,alpha_vantage",
        "debateRounds": int(os.getenv("MAX_DEBATE_ROUNDS", "1") or "1"),
        "toolReports": True,
        "checkpointing": os.getenv("TRADEAGE_CHECKPOINT_ENABLED", "1") not in {"0", "false", "False"},
        "clearCheckpoints": False,
        "resumeCheckpoint": False,
        "simulatedPresentDate": sim_present,
        "playbackSpeed": 1,
        "stepInterval": "daily",
        "marketSession": "regular",
        "autoAdvance": True,
        "logPath": "runs/latest.jsonl",
    }


def normalize_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    config = default_config()
    if raw:
        config.update({key: value for key, value in raw.items() if value is not None})
    config["symbol"] = str(config.get("symbol") or "AAPL").upper().strip()
    config["symbols"] = str(config.get("symbols") or config["symbol"]).upper().strip()
    config["cash"] = float(config.get("cash") or 10000)
    config["debateRounds"] = int(config.get("debateRounds") or 1)
    config["dataProvider"] = str(config.get("dataProvider") or "yfinance")
    config["llmProvider"] = str(config.get("llmProvider") or "local")
    config["logPath"] = str(config.get("logPath") or "runs/latest.jsonl")
    return config


def cli_equivalent(config: dict[str, Any]) -> str:
    args = ["python", "main.py"]
    if config["mode"] != "paper_experiment":
        args.append(config["symbol"])
    args.extend(["--start", config["startDate"]])
    args.extend(["--end", config["endDate"]])
    args.extend(["--cash", str(config["cash"])])
    args.extend(["--log", config["logPath"]])
    args.extend(["--data-provider", config["dataProvider"]])
    if config["mode"] in {"backtest", "time_travel"}:
        args.append("--backtest")
    if config["mode"] == "paper_experiment":
        args.extend(["--paper-experiment", "--symbols", config["symbols"]])
    if config["mode"] == "baseline":
        args.extend(["--baseline", config["baseline"]])
    return " ".join(args)


def initial_snapshot(config: dict[str, Any], *, status: str = "idle", run_id: str | None = None) -> dict[str, Any]:
    agents = _agents_for_mode(config["mode"])
    return {
        "runId": run_id or f"run-{uuid.uuid4().hex}",
        "status": status,
        "config": config,
        "agents": [
            {**agent, "status": "idle", "progress": 0}
            for agent in agents
        ],
        "events": [],
        "toolCalls": _tool_calls(config, "queued"),
        "reports": {
            "market": "No market report yet.",
            "news": "News context is not loaded.",
            "sentiment": "Sentiment report is not loaded.",
            "fundamentals": "Fundamentals report is not loaded.",
        },
        "portfolio": {
            "cash": config["cash"],
            "shares": 0,
            "equity": config["cash"],
            "position": "flat",
            "pnl": 0,
            "stopLoss": None,
        },
        "signal": {
            "rating": "Pending",
            "action": "hold",
            "confidence": 0,
            "riskNotes": "Run has not started.",
        },
        "audit": {
            "fullStateLog": "",
            "jsonlLog": config["logPath"],
            "checkpointDb": os.getenv("TRADEAGE_WORKFLOW_DB", "data_cache/workflows.db"),
            "cliEquivalent": cli_equivalent(config),
            "rowsVisible": 0,
            "latestMarketDate": "-",
        },
    }


def running_snapshot(config: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    snapshot = initial_snapshot(config, status="running", run_id=run_id)
    running_id = snapshot["agents"][0]["id"] if snapshot["agents"] else None
    snapshot["agents"] = [
        {
            **agent,
            "status": "running" if agent["id"] == running_id else "queued",
            "progress": 35 if agent["id"] == running_id else 0,
        }
        for agent in snapshot["agents"]
    ]
    snapshot["events"] = [
        _event(
            "started",
            "Backend run started",
            "The Python API created a background run job and is executing the configured workflow.",
            "info",
            running_id,
        )
    ]
    return snapshot


def start_run_job(raw_config: dict[str, Any] | None) -> dict[str, Any]:
    config = normalize_config(raw_config)
    run_id = f"run-{uuid.uuid4().hex}"
    snapshot = running_snapshot(config, run_id=run_id)
    job = {
        "run_id": run_id,
        "config": config,
        "snapshot": snapshot,
        "cancelled": False,
        "started_at": time.time(),
        "finished_at": None,
    }
    with _JOBS_LOCK:
        _JOBS[run_id] = job

    thread = threading.Thread(target=_run_job_worker, args=(run_id,), daemon=True)
    thread.start()
    return snapshot


def get_run_snapshot(run_id: str) -> dict[str, Any] | None:
    with _JOBS_LOCK:
        job = _JOBS.get(run_id)
        if not job:
            return None
        if job.get("snapshot", {}).get("status") == "running":
            job["snapshot"] = _refresh_running_snapshot(job)
        return job["snapshot"]


def stop_run_job(run_id: str) -> dict[str, Any] | None:
    with _JOBS_LOCK:
        job = _JOBS.get(run_id)
        if not job:
            return None
        job["cancelled"] = True
        snapshot = dict(job["snapshot"])
        snapshot["status"] = "idle"
        snapshot["events"] = [
            _event(
                "stopped",
                "Run stopped",
                "The backend job was marked stopped. In-flight Python work may finish in the background.",
                "info",
            ),
            *snapshot.get("events", []),
        ]
        job["snapshot"] = snapshot
        job["finished_at"] = time.time()
        return snapshot


def run_orchestration(raw_config: dict[str, Any] | None) -> dict[str, Any]:
    config = normalize_config(raw_config)
    run_id = f"run-{uuid.uuid4().hex}"
    started = time.time()
    with _RUN_LOCK, _run_environment(config):
        try:
            result = _execute(config)
            snapshot = snapshot_from_result(config, result, run_id=run_id)
            snapshot["events"].insert(0, _event("complete", "Run complete", f"Backend run finished in {time.time() - started:.2f}s.", "decision"))
            return snapshot
        except Exception as exc:
            snapshot = initial_snapshot(config, status="failed", run_id=run_id)
            snapshot["events"].insert(0, _event("failed", "Run failed", _exception_detail(exc), "error"))
            return snapshot


def _run_job_worker(run_id: str) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(run_id)
        if not job:
            return
        config = job["config"]
        started = job["started_at"]

    with _RUN_LOCK, _run_environment(config):
        try:
            result = _execute(config)
            snapshot = snapshot_from_result(config, result, run_id=run_id)
            snapshot["events"].insert(
                0,
                _event(
                    "complete",
                    "Run complete",
                    f"Backend run finished in {time.time() - started:.2f}s.",
                    "decision",
                ),
            )
        except Exception as exc:
            with _JOBS_LOCK:
                current_job = _JOBS.get(run_id)
                base_snapshot = _refresh_running_snapshot(current_job) if current_job else initial_snapshot(config, status="running", run_id=run_id)
            snapshot = dict(base_snapshot)
            snapshot["status"] = "failed"
            snapshot["events"].insert(0, _event("failed", "Run failed", _exception_detail(exc), "error"))

    with _JOBS_LOCK:
        job = _JOBS.get(run_id)
        if not job or job.get("cancelled"):
            return
        job["snapshot"] = snapshot
        job["finished_at"] = time.time()


def snapshot_from_result(config: dict[str, Any], result: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    snapshot = initial_snapshot(config, status="complete", run_id=run_id)
    trace = result.get("trace")
    trace_item = _last_trace_item(trace)
    reports = _extract_reports(config, trace, result)
    metrics = result.get("metrics") or result.get("aggregate", {})
    latest_date = _latest_date(trace, config)
    portfolio_source = result.get("portfolio") or result.get("account") or {}

    snapshot["agents"] = [{**agent, "status": "complete", "progress": 100} for agent in snapshot["agents"]]
    snapshot["toolCalls"] = _tool_calls(config, "complete", result=result)
    snapshot["reports"] = reports
    snapshot["portfolio"] = _portfolio_state(config, result)
    snapshot["signal"] = _signal_state(config, trace_item, result, metrics)
    snapshot["audit"] = {
        **snapshot["audit"],
        "fullStateLog": _full_state_log(trace, result),
        "rowsVisible": _rows_visible(trace, portfolio_source),
        "latestMarketDate": latest_date,
    }
    snapshot["events"] = _events_from_result(config, result, trace_item)
    return snapshot


def _execute(config: dict[str, Any]) -> dict[str, Any]:
    mode = config["mode"]
    if mode == "paper_experiment":
        symbols = [item.strip() for item in config["symbols"].split(",") if item.strip()]
        return run_paper_experiment(
            symbols=symbols,
            start_date=config["startDate"] or PAPER_START_DATE,
            end_date=config["endDate"] or PAPER_END_DATE_EXCLUSIVE,
            data_provider=config["dataProvider"] or "yfinance",
            config=PaperBacktestConfig(initial_cash=config["cash"]),
            log_path=config["logPath"],
        )
    if mode == "baseline":
        return run_baseline_backtest(
            config["symbol"],
            config["baseline"],
            start_date=config["startDate"] or PAPER_START_DATE,
            end_date=config["endDate"] or PAPER_END_DATE_EXCLUSIVE,
            data_provider=config["dataProvider"] or "yfinance",
            config=PaperBacktestConfig(initial_cash=config["cash"]),
            log_path=config["logPath"],
        )
    if mode in {"backtest", "time_travel"}:
        return run_backtest(
            config["symbol"],
            cash=config["cash"],
            log_path=config["logPath"],
            start_date=config["startDate"] or PAPER_START_DATE,
            end_date=config["endDate"] or PAPER_END_DATE_EXCLUSIVE,
            data_provider=config["dataProvider"],
        )
    return run_pipeline(
        config["symbol"],
        cash=config["cash"],
        log_path=config["logPath"],
        start_date=config["startDate"],
        end_date=config["endDate"],
        data_provider=config["dataProvider"],
    )


@contextmanager
def _run_environment(config: dict[str, Any]):
    keys = {
        "LLM_PROVIDER": config["llmProvider"],
        "MAX_DEBATE_ROUNDS": str(config["debateRounds"]),
        "TRADEAGE_CHECKPOINT_ENABLED": "1" if config["checkpointing"] else "0",
    }
    old = {key: os.environ.get(key) for key in keys}
    os.environ.update(keys)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _exception_detail(exc: Exception) -> str:
    parts: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).strip() or current.__class__.__name__
        if message not in parts:
            parts.append(message)
        current = current.__cause__ or current.__context__
    return " | caused by: ".join(parts)


def _append_job_log_records(job: dict[str, Any]) -> list[dict[str, Any]]:
    config = job["config"]
    path = Path(config["logPath"])
    watch = job.setdefault("log_watch", {"position": 0, "records": []})
    if not path.exists():
        return watch["records"]

    size = path.stat().st_size
    if size < watch["position"]:
        watch["position"] = 0
        watch["records"] = []

    with path.open() as handle:
        handle.seek(watch["position"])
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                watch["records"].append(json.loads(line))
            except json.JSONDecodeError:
                continue
        watch["position"] = handle.tell()
    watch["records"] = watch["records"][-400:]
    return watch["records"]


def _refresh_running_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    snapshot = dict(job["snapshot"])
    records = _append_job_log_records(job)
    if not records:
        return snapshot

    config = job["config"]
    if config["mode"] == "paper_experiment":
        return _refresh_experiment_running_snapshot(config, snapshot, records)
    return _refresh_generic_running_snapshot(config, snapshot, records)


def _refresh_generic_running_snapshot(config: dict[str, Any], snapshot: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    latest = records[-1]
    stage = str(latest.get("stage") or "running")
    payload = latest.get("payload")
    detail = f"Latest stage: {stage}"
    if isinstance(payload, dict) and payload.get("date"):
        detail += f" · {payload['date']}"
    events = list(snapshot.get("events", []))
    marker = f"event-progress-{stage}"
    event = _event("progress", "Run in progress", detail, "info", event_id=marker)
    if not events or events[0].get("id") != marker:
        events.insert(0, event)
    else:
        events[0] = event
    snapshot["events"] = events[:20]
    return snapshot


def _refresh_experiment_running_snapshot(config: dict[str, Any], snapshot: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    symbols = [item.strip().upper() for item in str(config.get("symbols") or "").split(",") if item.strip()]
    expected_units = max(1, len(symbols) * len(_EXPERIMENT_STRATEGIES))
    completed_units: set[tuple[str, str]] = set()
    latest_stage = ""
    latest_detail = ""

    for record in records:
        stage = str(record.get("stage") or "")
        payload = record.get("payload") or {}
        latest_stage = stage or latest_stage
        if stage == "paper_experiment:symbol_start" and isinstance(payload, dict):
            latest_detail = f"Running {payload.get('symbol', '')}"
        elif stage == "paper_experiment:symbol_complete" and isinstance(payload, dict):
            latest_detail = f"Completed {payload.get('symbol', '')}"
        elif ":" in stage:
            symbol, strategy = stage.split(":", 1)
            if strategy in _EXPERIMENT_STRATEGIES:
                completed_units.add((symbol.upper(), strategy))
                if isinstance(payload, dict) and payload.get("date"):
                    latest_detail = f"{symbol.upper()} · {strategy} · {payload['date']}"
                else:
                    latest_detail = f"{symbol.upper()} · {strategy}"
        elif stage == "paper_experiment:aggregate":
            latest_detail = "Aggregating experiment results"

    progress = min(99, int((len(completed_units) / expected_units) * 100))
    agents = [dict(agent) for agent in snapshot.get("agents", [])]
    for agent in agents:
        if agent["id"] == "experiment_inputs":
            agent["status"] = "complete" if records else "running"
            agent["progress"] = 100 if records else 35
        elif agent["id"] == "batch_runner":
            agent["status"] = "running"
            agent["progress"] = max(8, progress)
        elif agent["id"] == "aggregate":
            agent["status"] = "queued"
            agent["progress"] = 0
            if latest_stage == "paper_experiment:aggregate":
                agent["status"] = "running"
                agent["progress"] = 95
    snapshot["agents"] = agents

    detail = latest_detail or f"Completed {len(completed_units)} of {expected_units} strategy runs."
    events = list(snapshot.get("events", []))
    marker = f"event-progress-{len(completed_units)}"
    event = _event("progress", "Experiment in progress", detail, "info", "batch_runner", event_id=marker)
    if events and events[0].get("id") == marker:
        events[0] = event
    else:
        events.insert(0, event)
    snapshot["events"] = events[:20]
    snapshot["toolCalls"] = _experiment_running_tool_calls(config, len(completed_units), expected_units, latest_detail or latest_stage)
    snapshot["signal"] = {
        "rating": "Experiment running",
        "action": "summary",
        "confidence": 0,
        "riskNotes": detail,
    }
    snapshot["reports"] = {
        **snapshot.get("reports", {}),
        "market": "\n".join(
            [
                "Paper experiment is still running.",
                "",
                f"Completed strategy runs: {len(completed_units)} / {expected_units}",
                f"Latest activity: {detail}",
            ]
        ),
    }
    return snapshot


def _experiment_running_tool_calls(config: dict[str, Any], completed_units: int, expected_units: int, latest_detail: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "tool-1",
            "agentId": "experiment_inputs",
            "tool": "load_symbol_windows",
            "args": {"symbols": config["symbols"], "start_date": config["startDate"], "end_date": config["endDate"]},
            "source": config["dataProvider"],
            "dataQuality": "strict_point_in_time",
            "status": "complete",
            "observation": "Loaded one historical window per symbol for the experiment batch.",
        },
        {
            "id": "tool-2",
            "agentId": "batch_runner",
            "tool": "run_strategy_batch",
            "args": {"symbols": config["symbols"], "strategies": ",".join(_EXPERIMENT_STRATEGIES)},
            "source": "backend paper engine",
            "dataQuality": "simulated_or_derived",
            "status": "running",
            "observation": f"Completed {completed_units} of {expected_units} strategy runs. Latest activity: {latest_detail or 'running batch'}",
        },
        {
            "id": "tool-3",
            "agentId": "aggregate",
            "tool": "aggregate_equity_curves",
            "args": {"symbols": config["symbols"]},
            "source": "backend aggregator",
            "dataQuality": "simulated_or_derived",
            "status": "queued",
            "observation": "Waiting for the full batch before building aggregate comparisons.",
        },
    ]


def _tool_calls(config: dict[str, Any], status: str, *, result: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if config["mode"] == "baseline":
        return _baseline_tool_calls(config, status, result=result)
    if config["mode"] == "paper_experiment":
        return _experiment_tool_calls(config, status, result=result)

    trace = (result or {}).get("trace")
    report_trace = {}
    if isinstance(trace, dict):
        candidate = trace.get("report_tool_traces") or trace.get("tool_trace") or {}
        report_trace = candidate if isinstance(candidate, dict) else {}
    return [
        {
            "id": "tool-1",
            "agentId": "market",
            "tool": "get_stock_data",
            "args": {"symbol": config["symbol"], "start_date": config["startDate"], "end_date": config["endDate"]},
            "source": config["dataProvider"],
            "dataQuality": "strict_point_in_time",
            "status": status,
            "observation": _safe_json(report_trace.get("market")) if report_trace else "Market rows loaded by backend.",
        },
        {
            "id": "tool-2",
            "agentId": "market",
            "tool": "get_indicators",
            "args": {"indicators": "rsi,macd,boll,atr,vwma"},
            "source": "local",
            "dataQuality": "simulated_or_derived",
            "status": status,
            "observation": "Indicators derived from backend-visible rows.",
        },
        {
            "id": "tool-3",
            "agentId": "news",
            "tool": "get_news",
            "args": {"news_mode": config["newsMode"], "sources": config["newsSources"]},
            "source": config["newsSources"],
            "dataQuality": "publication_time" if config["newsMode"] != "disabled" else "strict_point_in_time",
            "status": status,
            "observation": _safe_json(report_trace.get("news")) if report_trace else "News report generated by backend tools.",
        },
        {
            "id": "tool-4",
            "agentId": "fundamentals",
            "tool": "get_fundamentals",
            "args": {"label": "current_snapshot"},
            "source": "configured backend vendor",
            "dataQuality": "current_snapshot",
            "status": status,
            "observation": "Fundamentals generated by backend tools and labeled by timing quality.",
        },
    ]


def _baseline_tool_calls(config: dict[str, Any], status: str, *, result: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    trace = (result or {}).get("trace") or []
    last_item = trace[-1] if isinstance(trace, list) and trace else {}
    signal = last_item.get("signal", "hold")
    calls = [
        {
            "id": "tool-1",
            "agentId": "load_market",
            "tool": "get_stock_data",
            "args": {"symbol": config["symbol"], "start_date": config["startDate"], "end_date": config["endDate"]},
            "source": config["dataProvider"],
            "dataQuality": "strict_point_in_time",
            "status": status,
            "observation": "Historical bars loaded for the selected baseline window.",
        },
        {
            "id": "tool-2",
            "agentId": "baseline",
            "tool": "run_baseline_strategy",
            "args": {"strategy": config["baseline"], "symbol": config["symbol"]},
            "source": "backend baseline engine",
            "dataQuality": "simulated_or_derived",
            "status": status,
            "observation": f"Baseline strategy {config['baseline']} finished. Last visible signal: {signal}.",
        },
    ]
    if config["baseline"] != "buy_and_hold":
        calls.insert(
            1,
            {
                "id": "tool-1b",
                "agentId": "baseline",
                "tool": "get_indicators",
                "args": {"strategy": config["baseline"], "inputs": "price-derived indicators"},
                "source": "backend indicators",
                "dataQuality": "simulated_or_derived",
                "status": status,
                "observation": "Indicator inputs were derived from the visible market window.",
            },
        )
    return calls


def _experiment_tool_calls(config: dict[str, Any], status: str, *, result: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    aggregate = (result or {}).get("aggregate") or {}
    return [
        {
            "id": "tool-1",
            "agentId": "experiment_inputs",
            "tool": "load_symbol_windows",
            "args": {"symbols": config["symbols"], "start_date": config["startDate"], "end_date": config["endDate"]},
            "source": config["dataProvider"],
            "dataQuality": "strict_point_in_time",
            "status": status,
            "observation": "Loaded one historical window per symbol for the experiment batch.",
        },
        {
            "id": "tool-2",
            "agentId": "batch_runner",
            "tool": "run_strategy_batch",
            "args": {"symbols": config["symbols"], "strategies": "PennyLaneCapital,buy_and_hold,macd,kdj_rsi,zmr,sma"},
            "source": "backend paper engine",
            "dataQuality": "simulated_or_derived",
            "status": status,
            "observation": "Completed the symbol-by-symbol benchmark batch.",
        },
        {
            "id": "tool-3",
            "agentId": "aggregate",
            "tool": "aggregate_equity_curves",
            "args": {"strategies": len(aggregate), "symbols": len((result or {}).get("results", {}))},
            "source": "backend aggregator",
            "dataQuality": "simulated_or_derived",
            "status": status,
            "observation": "Combined the per-symbol equity curves into aggregate strategy totals.",
        },
    ]


def _extract_reports(config: dict[str, Any], trace: Any, result: dict[str, Any]) -> dict[str, str]:
    if config["mode"] == "baseline":
        return _baseline_reports(config, result)
    if config["mode"] == "paper_experiment":
        return _experiment_reports(result)
    if isinstance(trace, dict):
        reports = trace.get("reports") or {}
        return {
            "market": _stringify_report(reports.get("market", "No market report returned.")),
            "news": _stringify_report(reports.get("news", "No news report returned.")),
            "sentiment": _stringify_report(reports.get("sentiment", "No sentiment report returned.")),
            "fundamentals": _stringify_report(reports.get("fundamentals", "No fundamentals report returned.")),
        }
    item = _last_trace_item(trace)
    if item and item.get("reports"):
        reports = item["reports"]
        return {
            "market": _stringify_report(reports.get("market", "No market report returned.")),
            "news": _stringify_report(reports.get("news", "No news report returned.")),
            "sentiment": _stringify_report(reports.get("sentiment", "No sentiment report returned.")),
            "fundamentals": _stringify_report(reports.get("fundamentals", "No fundamentals report returned.")),
        }
    if "aggregate" in result:
        return {
            "market": _safe_json(result.get("aggregate")),
            "news": "Paper experiment does not produce one consolidated news report.",
            "sentiment": "Paper experiment does not produce one consolidated sentiment report.",
            "fundamentals": "Paper experiment does not produce one consolidated fundamentals report.",
        }
    return {
        "market": _safe_json(result.get("metrics", {})),
        "news": "No news report returned.",
        "sentiment": "No sentiment report returned.",
        "fundamentals": "No fundamentals report returned.",
    }

def _baseline_reports(config: dict[str, Any], result: dict[str, Any]) -> dict[str, str]:
    trace = result.get("trace") or []
    last_signal = ""
    if isinstance(trace, list) and trace:
        last_signal = str(trace[-1].get("signal") or "")
    return {
        "market": "\n".join([
            f"Baseline strategy: {result.get('strategy', config['baseline'])}",
            f"Symbol: {result.get('symbol', config['symbol'])}",
            f"Window: {config['startDate']} to {config['endDate']}",
            f"Last signal: {last_signal or 'n/a'}",
            "",
            "Metrics:",
            _safe_json(result.get("metrics", {})),
        ]),
        "news": "News analysis is not used in baseline mode.",
        "sentiment": "Sentiment analysis is not used in baseline mode.",
        "fundamentals": "Fundamentals analysis is not used in baseline mode.",
    }


def _experiment_reports(result: dict[str, Any]) -> dict[str, str]:
    return {
        "market": "\n".join([
            "Paper experiment summary",
            "",
            "Methodology:",
            _safe_json(result.get("methodology", {})),
            "",
            "Aggregate results:",
            _safe_json(result.get("aggregate", {})),
        ]),
        "news": "The experiment view aggregates strategy outcomes rather than one combined news report.",
        "sentiment": "The experiment view aggregates strategy outcomes rather than one combined sentiment report.",
        "fundamentals": "The experiment view aggregates strategy outcomes rather than one combined fundamentals report.",
    }


def _portfolio_state(config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    if config["mode"] == "paper_experiment":
        aggregate = result.get("aggregate") or {}
        _, summary = _best_aggregate_strategy(aggregate)
        initial = float(summary.get("initial_capital", config["cash"]) or config["cash"])
        ending = float(summary.get("ending_equity", initial) or initial)
        symbol_count = float(summary.get("symbols", 0) or 0)
        return {
            "cash": initial,
            "shares": symbol_count,
            "equity": ending,
            "position": "flat",
            "pnl": ending - initial,
            "stopLoss": None,
        }

    portfolio = result.get("portfolio") or result.get("account") or {}
    cash = float(portfolio.get("cash", config["cash"]) or 0)
    shares = float(portfolio.get("shares", 0) or 0)
    equity = float(portfolio.get("equity", cash) or cash)
    pnl = float(portfolio.get("pnl", portfolio.get("realized_pnl", 0) or 0) or 0)
    return {
        "cash": cash,
        "shares": shares,
        "equity": equity,
        "position": portfolio.get("position") or ("long" if shares > 0 else "short" if shares < 0 else "flat"),
        "pnl": pnl,
        "stopLoss": portfolio.get("stop_loss") or portfolio.get("stopLoss"),
    }


def _signal_state(config: dict[str, Any], trace_item: dict[str, Any] | None, result: dict[str, Any], metrics: Any) -> dict[str, Any]:
    if config["mode"] == "baseline":
        last_signal = str((trace_item or {}).get("signal") or "hold")
        return {
            "rating": f"Baseline · {result.get('strategy', config['baseline'])}",
            "action": last_signal,
            "confidence": 0,
            "riskNotes": (
                f"Cumulative return {result.get('metrics', {}).get('cumulative_return_pct', 0)}% · "
                f"max drawdown {result.get('metrics', {}).get('max_drawdown_pct', 0)}%."
            ),
        }
    if config["mode"] == "paper_experiment":
        aggregate = result.get("aggregate") or {}
        best_name, best_summary = _best_aggregate_strategy(aggregate)
        symbol_count = len(result.get("results", {}))
        return {
            "rating": "Experiment complete",
            "action": "summary",
            "confidence": 0,
            "riskNotes": (
                f"Compared {len(aggregate)} strategies across {symbol_count} symbols. "
                f"Top aggregate result: {best_name} ending at {best_summary.get('ending_equity', 'n/a')}."
            ),
        }

    pm = (trace_item or {}).get("portfolio_manager") or {}
    risk = (trace_item or {}).get("risk") or {}
    action = pm.get("action") or (trace_item or {}).get("signal") or result.get("strategy") or "hold"
    rating = pm.get("rating") or result.get("strategy") or "Complete"
    confidence = pm.get("confidence", 0)
    return {
        "rating": str(rating),
        "action": str(action),
        "confidence": float(confidence or 0),
        "riskNotes": risk.get("risk_notes") or f"Metrics: {_safe_json(metrics)}",
    }


def _events_from_result(config: dict[str, Any], result: dict[str, Any], trace_item: dict[str, Any] | None) -> list[dict[str, Any]]:
    if config["mode"] == "baseline":
        return [
            _event("load", "Baseline data loaded", f"{config['symbol']} from {config['startDate']} to {config['endDate']}.", "tool", "load_market"),
            _event("baseline", "Baseline strategy evaluated", f"Ran {config['baseline']} over the selected historical window.", "decision", "baseline"),
            _event("execution", "Paper execution updated", f"Strategy metrics: {_safe_json(result.get('metrics', {}))}.", "risk", "execution"),
        ]
    if config["mode"] == "paper_experiment":
        return [
            _event("inputs", "Experiment inputs loaded", f"{config['symbols']} across {config['startDate']} to {config['endDate']}.", "tool", "experiment_inputs"),
            _event("batch", "Benchmark batch finished", "Completed the symbol-by-symbol strategy runs for the experiment batch.", "decision", "batch_runner"),
            _event("aggregate", "Aggregate comparison compiled", _safe_json(result.get("aggregate", {}))[:500], "risk", "aggregate"),
        ]

    events = [
        _event("load", "Backend data loaded", f"{config['symbol']} from {config['startDate']} to {config['endDate']}.", "tool", "load_market"),
        _event("reports", "Analyst reports generated", "Backend generated market/news/sentiment/fundamentals report bundle.", "tool", "market"),
        _event("decision", "Decision state resolved", f"Mode {config['mode']} returned metrics {result.get('metrics', {})}.", "decision", "portfolio_manager"),
    ]
    if trace_item and trace_item.get("execution"):
        events.append(_event("execution", "Execution state updated", _safe_json(trace_item["execution"])[:500], "risk", "execution"))
    return events


def _event(
    suffix: str,
    title: str,
    detail: str,
    level: str,
    agent_id: str | None = None,
    *,
    event_id: str | None = None,
) -> dict[str, Any]:
    return {
        "id": event_id or f"event-{suffix}-{uuid.uuid4().hex[:8]}",
        "ts": time.strftime("%H:%M:%S"),
        "level": level,
        "agentId": agent_id,
        "title": title,
        "detail": detail,
    }


def _last_trace_item(trace: Any) -> dict[str, Any] | None:
    if isinstance(trace, list) and trace:
        return trace[-1]
    if isinstance(trace, dict):
        return trace
    return None


def _latest_date(trace: Any, config: dict[str, Any]) -> str:
    item = _last_trace_item(trace)
    return str((item or {}).get("date") or config.get("simulatedPresentDate") or config["endDate"])


def _rows_visible(trace: Any, portfolio: dict[str, Any]) -> int:
    if isinstance(trace, list):
        return len(trace)
    curve = portfolio.get("equity_curve")
    if isinstance(curve, list):
        return len(curve)
    return 1 if trace else 0


def _best_aggregate_strategy(aggregate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not aggregate:
        return "n/a", {}
    name, summary = max(
        aggregate.items(),
        key=lambda item: float(item[1].get("ending_equity", item[1].get("initial_capital", 0)) or 0),
    )
    return str(name), summary


def _full_state_log(trace: Any, result: dict[str, Any]) -> str:
    if isinstance(trace, dict):
        path = trace.get("full_state_log")
        if path:
            return str(path)
    symbol = result.get("symbol")
    if symbol:
        directory = Path("results") / str(symbol).upper() / "PennyLaneCapital_logs"
        return str(directory)
    return ""


def _stringify_report(value: Any) -> str:
    if isinstance(value, str):
        return value
    return _safe_json(value)


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, indent=2, default=str)
    except TypeError:
        return str(value)
