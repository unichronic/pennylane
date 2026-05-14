from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.paper_methodology import (
    PaperBacktestConfig,
    run_baseline_backtest,
    run_penny_lane_paper_backtest,
)
from config import get_config, get_mistral_api_keys
from core.reflection import load_memory_entries, load_memory_vectors


DEFAULT_SYMBOLS = "AAPL,MSFT,NVDA,AMZN,GOOGL"
DEFAULT_PERIODS = ",".join(
    [
        "2024-01-02:2024-02-01",
        "2024-03-01:2024-04-01",
        "2024-05-01:2024-06-01",
        "2024-08-01:2024-09-03",
        "2024-11-01:2024-12-02",
        "2025-02-03:2025-03-03",
    ]
)
DEFAULT_BASELINES = ("buy_and_hold", "macd", "kdj_rsi", "zmr", "sma")


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


def _append_jsonl(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(payload), sort_keys=True) + "\n")


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _symbols(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _periods(raw: str) -> list[dict]:
    periods = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        start, end = item.split(":", 1)
        periods.append({"start": start.strip(), "end": end.strip()})
    return sorted(periods, key=lambda item: (item["start"], item["end"]))


def _configure_env(run_id: str, out_dir: Path, args):
    os.environ["LLM_PROVIDER"] = args.llm_provider
    os.environ["TRADEAGE_MEMORY_EMBEDDING_PROVIDER"] = args.memory_embedding_provider
    os.environ["TRADEAGE_LLM_CALL_TIMEOUT_SECONDS"] = str(args.llm_timeout_seconds)
    os.environ.setdefault("TRADEAGE_ANALYST_TOOL_MODE", "model")
    os.environ.setdefault("TRADEAGE_MEMORY_VECTOR_ENABLED", "1")

    cache_dir = Path("data_cache") / "reward_loop" / run_id
    results_dir = Path("results") / "reward_loop" / run_id
    os.environ["TRADEAGE_WORKFLOW_DB"] = str(cache_dir / "workflows.db")
    os.environ["TRADEAGE_MEMORY_PATH"] = str(cache_dir / "decision_memory.jsonl")
    os.environ["TRADEAGE_MEMORY_VECTOR_PATH"] = str(cache_dir / "decision_memory.vectors.jsonl")
    os.environ["TRADEAGE_LLM_TRACE_PATH"] = str(cache_dir / "llm_calls.jsonl")
    os.environ["TRADEAGE_RESULTS_DIR"] = str(results_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)


def _setting_key(trial: dict) -> str:
    return f"cadence={trial['decision_cadence_days']}|short_policy={trial['short_policy']}"


def _setting_stats(completed: list[dict]) -> dict:
    grouped = defaultdict(list)
    for item in completed:
        if item.get("status") == "passed":
            grouped[_setting_key(item)].append(float(item.get("reward", 0)))
    return {
        key: {
            "count": len(values),
            "avg_reward": round(sum(values) / len(values), 6),
            "best_reward": round(max(values), 6),
        }
        for key, values in grouped.items()
    }


def _choose_setting(settings: list[dict], completed: list[dict], idx: int) -> dict:
    if idx < len(settings):
        return settings[idx]
    stats = _setting_stats(completed)
    if not stats:
        return settings[idx % len(settings)]
    ranked = sorted(
        settings,
        key=lambda item: (
            stats.get(f"cadence={item['decision_cadence_days']}|short_policy={item['short_policy']}", {}).get("avg_reward", -10**9),
            -stats.get(f"cadence={item['decision_cadence_days']}|short_policy={item['short_policy']}", {}).get("count", 0),
        ),
        reverse=True,
    )
    return ranked[0]


def _trial_plan(symbols: list[str], periods: list[dict], settings: list[dict], loops: int, completed: list[dict]) -> list[dict]:
    plan = []
    completed_count = len(completed)
    contexts = [{"symbol": symbol, **period} for period in periods for symbol in symbols]
    for offset in range(loops):
        idx = completed_count + offset
        context = contexts[idx % len(contexts)]
        setting = _choose_setting(settings, completed + plan, idx)
        plan.append({
            "trial_index": idx + 1,
            **context,
            **setting,
        })
    return plan


def _score(agent_result: dict, baseline_results: dict) -> dict:
    agent_metrics = agent_result["metrics"]
    agent_cr = float(agent_metrics.get("cumulative_return_pct", 0))
    agent_mdd = float(agent_metrics.get("max_drawdown_pct", 0))
    agent_sharpe = float(agent_metrics.get("sharpe_ratio", 0))
    baseline_crs = {
        name: float(result["metrics"].get("cumulative_return_pct", 0))
        for name, result in baseline_results.items()
    }
    best_baseline_name = max(baseline_crs, key=baseline_crs.get)
    best_baseline_cr = baseline_crs[best_baseline_name]
    buy_hold_cr = baseline_crs.get("buy_and_hold", 0.0)
    alpha_vs_best = agent_cr - best_baseline_cr
    alpha_vs_buy_hold = agent_cr - buy_hold_cr
    reward = alpha_vs_best - (0.5 * agent_mdd) + (0.25 * agent_sharpe)
    return {
        "reward": round(reward, 6),
        "agent_cumulative_return_pct": round(agent_cr, 6),
        "agent_max_drawdown_pct": round(agent_mdd, 6),
        "agent_sharpe_ratio": round(agent_sharpe, 6),
        "best_baseline": best_baseline_name,
        "best_baseline_cumulative_return_pct": round(best_baseline_cr, 6),
        "buy_hold_cumulative_return_pct": round(buy_hold_cr, 6),
        "alpha_vs_best_baseline_pct": round(alpha_vs_best, 6),
        "alpha_vs_buy_hold_pct": round(alpha_vs_buy_hold, 6),
        "profitable": agent_cr > 0,
        "beat_best_baseline": alpha_vs_best > 0,
    }


def _run_trial(trial: dict, out_dir: Path, args) -> dict:
    trial_id = (
        f"{trial['trial_index']:03d}_{trial['symbol']}_{trial['start']}_{trial['end']}"
        f"_c{trial['decision_cadence_days']}_{trial['short_policy']}"
    ).replace(":", "-")
    trial_dir = out_dir / "trials" / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    config = PaperBacktestConfig(
        initial_cash=args.cash,
        decision_cadence_days=trial["decision_cadence_days"],
        short_policy=trial["short_policy"],
        record_memory=args.self_improve,
    )
    started = perf_counter()
    baseline_results = {}
    for baseline in DEFAULT_BASELINES:
        baseline_results[baseline] = run_baseline_backtest(
            trial["symbol"],
            baseline,
            start_date=trial["start"],
            end_date=trial["end"],
            data_provider=args.data_provider,
            config=config,
            log_path=trial_dir / f"{baseline}.jsonl",
        )

    agent_result = run_penny_lane_paper_backtest(
        trial["symbol"],
        start_date=trial["start"],
        end_date=trial["end"],
        data_provider=args.data_provider,
        config=config,
        log_path=trial_dir / "PennyLaneCapital.jsonl",
    )
    scoring = _score(agent_result, baseline_results)
    result = {
        **trial,
        **scoring,
        "trial_id": trial_id,
        "status": "passed",
        "duration_seconds": round(perf_counter() - started, 3),
        "agent_result_path": str(trial_dir / "PennyLaneCapital.json"),
        "baseline_result_paths": {
            name: str(trial_dir / f"{name}.json")
            for name in baseline_results
        },
        "trades": agent_result.get("trades", []),
        "signals": [
            {
                "date": item.get("date"),
                "signal": item.get("signal"),
                "decision_skipped": item.get("decision_skipped"),
            }
            for item in agent_result.get("trace", [])
        ],
    }
    _write_json(trial_dir / "PennyLaneCapital.json", agent_result)
    for name, payload in baseline_results.items():
        _write_json(trial_dir / f"{name}.json", payload)
    _write_json(trial_dir / "score.json", result)
    return result


def _write_leaderboard(out_dir: Path, trials: list[dict], args):
    passed = [item for item in trials if item.get("status") == "passed"]
    ranked = sorted(passed, key=lambda item: item.get("reward", -10**9), reverse=True)
    leaderboard = {
        "updated_at": datetime.now(UTC).isoformat(),
        "run_id": args.run_id,
        "trial_count": len(trials),
        "passed_count": len(passed),
        "failed_count": len([item for item in trials if item.get("status") == "failed"]),
        "setting_stats": _setting_stats(passed),
        "top_trials": ranked[:10],
        "memory": {
            "path": os.environ["TRADEAGE_MEMORY_PATH"],
            "entry_count": len(load_memory_entries()),
            "vector_path": os.environ["TRADEAGE_MEMORY_VECTOR_PATH"],
            "vector_count": len(load_memory_vectors()),
        },
        "llm_trace_path": os.environ["TRADEAGE_LLM_TRACE_PATH"],
    }
    _write_json(out_dir / "leaderboard.json", leaderboard)
    if ranked:
        best = {
            key: ranked[0][key]
            for key in [
                "decision_cadence_days",
                "short_policy",
                "reward",
                "agent_cumulative_return_pct",
                "alpha_vs_best_baseline_pct",
                "alpha_vs_buy_hold_pct",
            ]
        }
        _write_json(out_dir / "best_config.json", best)
    return leaderboard


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reward-scored committee trials across symbols and periods.")
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--loops", type=int, default=30)
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--periods", default=DEFAULT_PERIODS, help="Comma-separated start:end windows.")
    parser.add_argument("--llm-provider", choices=["mistral", "local"], default="mistral")
    parser.add_argument("--memory-embedding-provider", choices=["local", "mistral"], default="local")
    parser.add_argument("--data-provider", choices=["auto", "yfinance", "twelvedata", "alpha_vantage"], default="yfinance")
    parser.add_argument("--cash", type=float, default=10000)
    parser.add_argument("--cadences", default="2,3")
    parser.add_argument("--short-policies", default="allow_short,trend_confirmed_short,reduce_only")
    parser.add_argument("--self-improve", action="store_true", help="Record/refelect decisions into shared chronological memory.")
    parser.add_argument("--llm-timeout-seconds", type=int, default=180)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / "reward_loop" / args.run_id
    _configure_env(args.run_id, out_dir, args)
    cfg = get_config()
    settings = [
        {"decision_cadence_days": int(cadence), "short_policy": policy.strip()}
        for cadence in args.cadences.split(",")
        for policy in args.short_policies.split(",")
        if cadence.strip() and policy.strip()
    ]
    symbols = _symbols(args.symbols)
    periods = _periods(args.periods)
    trials_path = out_dir / "trials.jsonl"
    completed = _read_jsonl(trials_path)
    plan = _trial_plan(symbols, periods, settings, args.loops, completed)

    manifest = {
        "run_id": args.run_id,
        "out_dir": str(out_dir),
        "self_improve": args.self_improve,
        "symbols": symbols,
        "periods": periods,
        "settings": settings,
        "config": {
            "llm_provider": cfg["llm_provider"],
            "mistral_key_count": len(get_mistral_api_keys()),
            "mistral_quick_model": cfg["quick_think_llm"],
            "mistral_deep_model": cfg["deep_think_llm"],
            "data_provider": args.data_provider,
            "memory_embedding_provider": cfg["memory_embedding_provider"],
            "memory_path": os.environ["TRADEAGE_MEMORY_PATH"],
            "memory_vector_path": os.environ["TRADEAGE_MEMORY_VECTOR_PATH"],
            "llm_trace_path": os.environ["TRADEAGE_LLM_TRACE_PATH"],
        },
    }
    _write_json(out_dir / "manifest.json", manifest)

    all_trials = list(completed)
    for trial in plan:
        try:
            result = _run_trial(trial, out_dir, args)
        except Exception as exc:
            result = {
                **trial,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            if not args.continue_on_error:
                _append_jsonl(trials_path, result)
                all_trials.append(result)
                _write_leaderboard(out_dir, all_trials, args)
                raise
        _append_jsonl(trials_path, result)
        all_trials.append(result)
        leaderboard = _write_leaderboard(out_dir, all_trials, args)
        print(json.dumps({
            "trial": result.get("trial_id"),
            "status": result["status"],
            "reward": result.get("reward"),
            "agent_return": result.get("agent_cumulative_return_pct"),
            "alpha_vs_best": result.get("alpha_vs_best_baseline_pct"),
            "best_so_far": leaderboard["top_trials"][0]["trial_id"] if leaderboard["top_trials"] else None,
        }, sort_keys=True))

    _write_leaderboard(out_dir, all_trials, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
