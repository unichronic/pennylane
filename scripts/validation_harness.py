from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.engine import run_backtest
from backtest.paper_methodology import PAPER_END_DATE_EXCLUSIVE, PAPER_START_DATE, run_paper_experiment
from config import get_config, get_mistral_api_keys
from core.pipeline import run_pipeline
from core.reflection import load_lessons, load_memory_entries, load_memory_vectors


def _json_safe(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return str(value)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True))


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _summarize_log(path: Path):
    rows = _read_jsonl(path)
    stages = [row.get("stage") for row in rows]
    return {
        "path": str(path),
        "exists": path.exists(),
        "line_count": len(rows),
        "stages": stages,
        "stage_counts": {stage: stages.count(stage) for stage in sorted(set(stages)) if stage},
    }


def _summarize_trace(result):
    trace = result.get("trace", {}) if isinstance(result, dict) else {}
    if isinstance(trace, list):
        return {
            "trace_items": len(trace),
            "sample_keys": sorted(trace[0].keys()) if trace and isinstance(trace[0], dict) else [],
        }
    if not isinstance(trace, dict):
        return {"trace_type": type(trace).__name__}
    reports = trace.get("reports", {})
    tool_trace = trace.get("tool_trace", [])
    react_trace = trace.get("react_trace", [])
    routing_trace = trace.get("routing_trace", [])
    return {
        "workflow": trace.get("workflow"),
        "report_keys": sorted(reports.keys()) if isinstance(reports, dict) else [],
        "tool_count": len(tool_trace),
        "tools": sorted({item.get("tool") for item in tool_trace if item.get("tool")}),
        "react_steps": len(react_trace),
        "routing_steps": len(routing_trace),
        "full_state_log": trace.get("full_state_log"),
        "risk": trace.get("risk"),
        "portfolio_manager": trace.get("portfolio_manager"),
    }


def _run_step(name, fn, out_dir: Path, summary: dict):
    started = perf_counter()
    result_path = out_dir / f"{name}.json"
    traceback_path = out_dir / f"{name}.traceback.txt"
    timeout = summary.get("step_timeout_seconds")
    item = {"name": name, "status": "running"}
    summary["steps"].append(item)
    ctx = get_context("fork")
    queue = ctx.Queue()
    process = ctx.Process(target=_run_step_child, args=(fn, result_path, traceback_path, queue), daemon=False)
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
        item.update({
            "status": "failed",
            "duration_seconds": round(perf_counter() - started, 3),
            "error": f"TimeoutError: step exceeded {timeout} seconds",
            "traceback_file": str(traceback_path),
        })
        traceback_path.write_text(f"Step {name} exceeded timeout of {timeout} seconds and was terminated.\n")
        return None

    message = queue.get() if not queue.empty() else {"status": "failed", "error": f"child exited with code {process.exitcode}"}
    item["duration_seconds"] = round(perf_counter() - started, 3)
    if message.get("status") == "passed":
        item.update({"status": "passed", "result_file": str(result_path)})
        return json.loads(result_path.read_text()) if result_path.exists() else None
    item.update({
        "status": "failed",
        "error": message.get("error", "unknown child failure"),
        "traceback_file": str(traceback_path),
    })
    return None


def _run_step_child(fn, result_path: Path, traceback_path: Path, queue):
    try:
        result = fn()
        _write_json(result_path, result)
        queue.put({"status": "passed"})
    except Exception as exc:
        traceback_path.write_text(traceback.format_exc())
        queue.put({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})


def _configure_isolated_env(run_id: str, out_dir: Path, args):
    if args.llm_provider:
        os.environ["LLM_PROVIDER"] = args.llm_provider
    if args.memory_embedding_provider:
        os.environ["TRADEAGE_MEMORY_EMBEDDING_PROVIDER"] = args.memory_embedding_provider

    cache_dir = Path("data_cache") / "validation" / run_id
    results_dir = Path("results") / "validation" / run_id
    os.environ["TRADEAGE_WORKFLOW_DB"] = str(cache_dir / "workflows.db")
    os.environ["TRADEAGE_MEMORY_PATH"] = str(cache_dir / "decision_memory.jsonl")
    os.environ["TRADEAGE_MEMORY_VECTOR_PATH"] = str(cache_dir / "decision_memory.vectors.jsonl")
    os.environ["TRADEAGE_LLM_TRACE_PATH"] = str(cache_dir / "llm_calls.jsonl")
    os.environ["TRADEAGE_RESULTS_DIR"] = str(results_dir)
    os.environ.setdefault("TRADEAGE_MEMORY_VECTOR_ENABLED", "1")
    os.environ.setdefault("TRADEAGE_ANALYST_TOOL_MODE", "model")

    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)


def _inspect_artifacts(out_dir: Path, run_id: str, symbols: list[str]):
    memory_entries = load_memory_entries()
    vectors = load_memory_vectors()
    results_dir = Path(os.environ["TRADEAGE_RESULTS_DIR"])
    full_state_logs = sorted(str(path) for path in results_dir.glob("**/full_states_log_*.json"))
    lessons = {
        symbol.upper(): load_lessons(
            symbol,
            query_context=f"{symbol} validation market trend risk decision memory",
        )
        for symbol in symbols
    }
    logs = {
        path.name: _summarize_log(path)
        for path in sorted(out_dir.glob("*.jsonl"))
    }
    llm_trace = Path(os.environ.get("TRADEAGE_LLM_TRACE_PATH", "data_cache/llm_calls.jsonl"))
    llm_calls = _read_jsonl(llm_trace)
    result_files = {
        path.name: json.loads(path.read_text())
        for path in sorted(out_dir.glob("*.json"))
        if path.name not in {"summary.json", "artifact_inspection.json"}
    }
    trace_summaries = {
        name: _summarize_trace(payload)
        for name, payload in result_files.items()
        if isinstance(payload, dict) and "trace" in payload
    }
    return {
        "run_id": run_id,
        "logs": logs,
        "trace_summaries": trace_summaries,
        "memory": {
            "path": os.environ["TRADEAGE_MEMORY_PATH"],
            "entry_count": len(memory_entries),
            "resolved_count": len([item for item in memory_entries if item.get("outcome")]),
            "pending_count": len([item for item in memory_entries if not item.get("outcome")]),
            "symbols": sorted({item.get("symbol") for item in memory_entries if item.get("symbol")}),
        },
        "vectors": {
            "path": os.environ["TRADEAGE_MEMORY_VECTOR_PATH"],
            "count": len(vectors),
            "providers": sorted({item.get("embedding_provider") for item in vectors if item.get("embedding_provider")}),
            "models": sorted({item.get("embedding_model") for item in vectors if item.get("embedding_model")}),
        },
        "llm_calls": {
            "path": str(llm_trace),
            "count": len(llm_calls),
            "status_counts": {
                status: len([item for item in llm_calls if item.get("status") == status])
                for status in sorted({item.get("status") for item in llm_calls if item.get("status")})
            },
            "slowest": sorted(
                [
                    {
                        "agent_name": item.get("agent_name"),
                        "status": item.get("status"),
                        "duration_seconds": item.get("duration_seconds"),
                        "model": item.get("model"),
                        "key_slot": item.get("key_slot"),
                        "error": item.get("error"),
                    }
                    for item in llm_calls
                    if item.get("status") in {"succeeded", "failed", "timeout"}
                ],
                key=lambda item: item.get("duration_seconds") or 0,
                reverse=True,
            )[:10],
        },
        "full_state_logs": full_state_logs,
        "lessons_preview": {symbol: text[:1800] for symbol, text in lessons.items()},
    }


def _symbols(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run non-stopping real-data validation for the project.")
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--llm-provider", choices=["mistral", "local"], default="")
    parser.add_argument("--memory-embedding-provider", choices=["local", "mistral"], default="")
    parser.add_argument("--data-provider", choices=["auto", "yfinance", "twelvedata", "alpha_vantage"], default="yfinance")
    parser.add_argument("--live-symbol", default="AAPL")
    parser.add_argument("--live-start", default="2024-03-01")
    parser.add_argument("--live-end", default="2024-04-15")
    parser.add_argument("--short-symbols", default="AAPL,GOOGL,AMZN")
    parser.add_argument("--short-start", default="2024-03-01")
    parser.add_argument("--short-end", default="2024-03-15")
    parser.add_argument("--experiment-symbols", default="AAPL,GOOGL,AMZN")
    parser.add_argument("--experiment-start", default=PAPER_START_DATE)
    parser.add_argument("--experiment-end", default=PAPER_END_DATE_EXCLUSIVE)
    parser.add_argument("--cash", type=float, default=10000)
    parser.add_argument(
        "--decision-cadence-days",
        type=int,
        default=1,
        help="Run the full committee every N trading days in short backtests; 1 preserves paper-style daily decisions.",
    )
    parser.add_argument(
        "--short-policy",
        choices=["allow_short", "trend_confirmed_short", "reduce_only"],
        default="allow_short",
        help="How paper backtests interpret a final sell signal.",
    )
    parser.add_argument("--step-timeout-seconds", type=int, default=900)
    parser.add_argument("--skip-large-experiment", action="store_true")
    parser.add_argument(
        "--fail-exit-code",
        action="store_true",
        help="Exit with code 1 if any step failed. By default the harness records failures but exits 0.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / "validation" / args.run_id
    _configure_isolated_env(args.run_id, out_dir, args)
    cfg = get_config()
    short_symbols = _symbols(args.short_symbols)
    experiment_symbols = _symbols(args.experiment_symbols)
    all_symbols = sorted({args.live_symbol.upper(), *short_symbols, *experiment_symbols})
    summary = {
        "run_id": args.run_id,
        "out_dir": str(out_dir),
        "status": "running",
        "step_timeout_seconds": args.step_timeout_seconds,
        "config": {
            "llm_provider": cfg["llm_provider"],
            "mistral_key_count": len(get_mistral_api_keys()),
            "mistral_quick_model": cfg["quick_think_llm"],
            "mistral_deep_model": cfg["deep_think_llm"],
            "data_provider": args.data_provider,
            "decision_cadence_days": args.decision_cadence_days,
            "short_policy": args.short_policy,
            "memory_embedding_provider": cfg["memory_embedding_provider"],
            "memory_vector_enabled": cfg["memory_vector_enabled"],
            "workflow_db": os.environ["TRADEAGE_WORKFLOW_DB"],
            "memory_path": os.environ["TRADEAGE_MEMORY_PATH"],
            "memory_vector_path": os.environ["TRADEAGE_MEMORY_VECTOR_PATH"],
            "llm_trace_path": os.environ["TRADEAGE_LLM_TRACE_PATH"],
            "results_dir": os.environ["TRADEAGE_RESULTS_DIR"],
        },
        "steps": [],
    }

    def live_decision():
        return run_pipeline(
            args.live_symbol,
            cash=args.cash,
            log_path=out_dir / "01_live_decision.jsonl",
            start_date=args.live_start,
            end_date=args.live_end,
            data_provider=args.data_provider,
        )

    _run_step("01_live_decision", live_decision, out_dir, summary)

    for symbol in short_symbols:
        def short_backtest(symbol=symbol):
            return run_backtest(
                symbol,
                cash=args.cash,
                log_path=out_dir / f"02_short_backtest_{symbol}.jsonl",
                start_date=args.short_start,
                end_date=args.short_end,
                data_provider=args.data_provider,
                decision_cadence_days=args.decision_cadence_days,
                short_policy=args.short_policy,
            )

        _run_step(f"02_short_backtest_{symbol}", short_backtest, out_dir, summary)

    def inspect_before_experiment():
        return _inspect_artifacts(out_dir, args.run_id, all_symbols)

    _run_step("03_artifact_inspection", inspect_before_experiment, out_dir, summary)

    if not args.skip_large_experiment:
        def paper_experiment():
            return run_paper_experiment(
                symbols=experiment_symbols,
                start_date=args.experiment_start,
                end_date=args.experiment_end,
                data_provider=args.data_provider,
                log_path=out_dir / "04_paper_experiment.jsonl",
            )

        _run_step("04_paper_experiment", paper_experiment, out_dir, summary)

        def final_inspection():
            return _inspect_artifacts(out_dir, args.run_id, all_symbols)

        _run_step("05_final_artifact_inspection", final_inspection, out_dir, summary)

    failures = [step for step in summary["steps"] if step["status"] == "failed"]
    summary["status"] = "failed" if failures else "passed"
    summary["failure_count"] = len(failures)
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failures and args.fail_exit_code else 0


if __name__ == "__main__":
    raise SystemExit(main())
