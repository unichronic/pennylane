import time
import json

from api.orchestrator import (
    _JOBS,
    _JOBS_LOCK,
    default_config,
    get_run_snapshot,
    initial_snapshot,
    run_orchestration,
    running_snapshot,
    start_run_job,
    stop_run_job,
)


def test_api_default_snapshot_matches_frontend_contract(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    config = default_config()
    snapshot = initial_snapshot(config)

    assert snapshot["status"] == "idle"
    assert snapshot["config"]["symbol"] == "AAPL"
    assert snapshot["agents"]
    assert snapshot["toolCalls"]
    assert snapshot["audit"]["cliEquivalent"].startswith("python main.py")


def test_api_run_orchestration_returns_real_backend_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    config = default_config()
    config.update(
        {
            "mode": "baseline",
            "symbol": "AAPL",
            "startDate": "2024-03-01",
            "endDate": "2024-03-08",
            "llmProvider": "local",
            "dataProvider": "yfinance",
            "baseline": "buy_and_hold",
            "logPath": str(tmp_path / "api-run.jsonl"),
        }
    )

    snapshot = run_orchestration(config)

    assert snapshot["status"] == "complete"
    assert [agent["label"] for agent in snapshot["agents"]] == [
        "Market Loader",
        "Baseline Strategy",
        "Paper Execution",
    ]
    assert snapshot["portfolio"]["equity"] > 0
    assert snapshot["audit"]["latestMarketDate"] == "2024-03-07"
    assert any(event["title"] == "Run complete" for event in snapshot["events"])


def test_api_run_job_publishes_snapshots(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    config = default_config()
    config.update(
        {
            "mode": "baseline",
            "symbol": "AAPL",
            "startDate": "2024-03-01",
            "endDate": "2024-03-08",
            "llmProvider": "local",
            "dataProvider": "yfinance",
            "baseline": "buy_and_hold",
            "logPath": str(tmp_path / "api-job.jsonl"),
        }
    )

    started = start_run_job(config)

    assert started["status"] == "running"
    run_id = started["runId"]
    deadline = time.time() + 10
    snapshot = started
    while time.time() < deadline:
        snapshot = get_run_snapshot(run_id)
        if snapshot["status"] == "complete":
            break
        time.sleep(0.05)

    assert snapshot["status"] == "complete"
    assert snapshot["portfolio"]["equity"] > 0


def test_api_stop_run_job_marks_snapshot_idle(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    config = default_config()
    config.update(
        {
            "mode": "baseline",
            "symbol": "AAPL",
            "startDate": "2024-03-01",
            "endDate": "2024-03-08",
            "llmProvider": "local",
            "dataProvider": "yfinance",
            "baseline": "buy_and_hold",
            "logPath": str(tmp_path / "api-stop.jsonl"),
        }
    )

    started = start_run_job(config)
    stopped = stop_run_job(started["runId"])

    assert stopped["status"] == "idle"
    assert stopped["events"][0]["title"] == "Run stopped"


def test_api_run_orchestration_surfaces_nested_failure_detail(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    config = default_config()
    config.update(
        {
            "mode": "baseline",
            "symbol": "AAPL",
            "startDate": "2024-03-01",
            "endDate": "2024-03-08",
            "llmProvider": "local",
            "dataProvider": "yfinance",
            "baseline": "buy_and_hold",
            "logPath": str(tmp_path / "api-fail.jsonl"),
        }
    )

    def fake_execute(_config):
        try:
            raise RuntimeError("Status 503: Service unavailable")
        except RuntimeError as inner:
            raise RuntimeError("Fundamentals analyst report failed") from inner

    monkeypatch.setattr("api.orchestrator._execute", fake_execute)

    snapshot = run_orchestration(config)

    assert snapshot["status"] == "failed"
    assert snapshot["events"][0]["title"] == "Run failed"
    assert "Fundamentals analyst report failed" in snapshot["events"][0]["detail"]
    assert "Status 503: Service unavailable" in snapshot["events"][0]["detail"]


def test_api_get_run_snapshot_refreshes_paper_experiment_progress(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    log_path = tmp_path / "paper-progress.jsonl"
    config = default_config()
    config.update(
        {
            "mode": "paper_experiment",
            "symbols": "AAPL,MSFT",
            "startDate": "2024-03-01",
            "endDate": "2024-03-08",
            "llmProvider": "local",
            "dataProvider": "yfinance",
            "logPath": str(log_path),
        }
    )

    records = [
        {"stage": "paper_experiment:start", "payload": {"symbols": ["AAPL", "MSFT"]}},
        {"stage": "paper_experiment:symbol_start", "payload": {"symbol": "AAPL"}},
        {"stage": "AAPL:PennyLaneCapital", "payload": {"date": "2024-03-01"}},
        {"stage": "AAPL:buy_and_hold", "payload": {"date": "2024-03-07"}},
    ]
    log_path.write_text("".join(json.dumps(record) + "\n" for record in records))

    run_id = "run-progress-test"
    with _JOBS_LOCK:
        _JOBS[run_id] = {
            "run_id": run_id,
            "config": config,
            "snapshot": running_snapshot(config, run_id=run_id),
            "cancelled": False,
            "started_at": time.time(),
            "finished_at": None,
        }

    try:
        first = get_run_snapshot(run_id)
        second = get_run_snapshot(run_id)
    finally:
        with _JOBS_LOCK:
            _JOBS.pop(run_id, None)

    assert first["status"] == "running"
    assert first["agents"][0]["status"] == "complete"
    assert first["agents"][1]["status"] == "running"
    assert first["toolCalls"][1]["status"] == "running"
    assert first["events"][0]["title"] == "Experiment in progress"
    assert first["events"][0]["id"] == "event-progress-2"
    assert "Completed strategy runs: 2 / 12" in first["reports"]["market"]
    assert len(second["events"]) == len(first["events"])
    assert second["events"][0]["id"] == "event-progress-2"
