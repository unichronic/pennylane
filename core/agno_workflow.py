from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

from agno.db.sqlite import SqliteDb
from agno.workflow.condition import Condition
from agno.workflow.loop import Loop
from agno.workflow.step import Step
from agno.workflow.types import StepOutput
from agno.workflow.workflow import Workflow

from agents.analyst import run_analyst
from agents.analyst_reports import (
    run_fundamentals_analyst_report,
    run_market_analyst_report,
    run_news_analyst_report,
    run_sentiment_analyst_report,
)
from agents.bear_researcher import create_bear_researcher
from agents.bull_researcher import create_bull_researcher, market_report_from_signal
from agents.debate import summarize_debate
from agents.final_decision import apply_final_portfolio_decision
from agents.aggressive_debator import run_aggressive_debator
from agents.conservative_debator import run_conservative_debator
from agents.llm_factory import get_llm
from agents.neutral_debator import run_neutral_debator
from agents.research_manager import run_research_manager
from agents.risk_manager import run_risk_manager
from agents.schemas import validate_analyst, validate_debate, validate_risk, validate_trader
from agents.trader import run_trader
from agents.trader import render_trader_context
from config import get_config, get_mistral_api_keys
from core.checkpointing import clear_checkpoint, load_checkpoint, save_checkpoint
from core.conditional_logic import ConditionalLogic
from core.evidence_ledger import build_evidence_ledger
from core.full_state_log import write_full_state_log
from core.logger import log_stage
from core.reflection import load_lessons, record_decision, reflect_outcomes
from core.report_context import build_compressed_reports, report_context
from core.state import execute_trade, make_pipeline_state, metrics
from core.workflow_state import workflow_snapshot
from data.indicators import add_indicators
from data.loader import load_ohlcv


WORKFLOW_STEPS = [
    "load_market",
    "analyst_reports_parallel",
    "market_analyst_report",
    "news_analyst_report",
    "sentiment_analyst_report",
    "fundamentals_analyst_report",
    "analyst",
    "investment_debate_loop",
    "research_manager",
    "trader",
    "risk_debate_loop",
    "risk_debate_log",
    "portfolio_manager",
    "execution_condition",
    "risk_gate",
    "execution",
    "evaluation",
]


def _step(name, executor):
    def wrapped(step_input, session_state=None):
        if session_state and session_state.get("checkpoint_enabled"):
            session_id = session_state["workflow_session_id"]
            checkpoint = load_checkpoint(session_id)
            if session_state.get("resume_checkpoint") and name in checkpoint.get("completed", []):
                session_state.update(checkpoint.get("session_state", {}))
                return StepOutput(content=checkpoint.get("outputs", {}).get(name))
        output = executor(step_input, session_state=session_state)
        if session_state and session_state.get("checkpoint_enabled") and name != "evaluation":
            save_checkpoint(session_state["workflow_session_id"], name, session_state, output.content)
        return output

    return Step(name=name, executor=wrapped, max_retries=0, on_error="fail")


def _log(session_state, stage, payload):
    log_stage(session_state.get("log_path"), stage, payload)


def _snapshot(session_state):
    snap = workflow_snapshot(session_state)
    session_state["state_snapshot"] = snap.model_dump()
    return session_state["state_snapshot"]


def _report_enabled(session_state):
    return bool(session_state.get("tool_reports", True))


def _report_unavailable(name):
    return f"{name} report skipped for this deterministic workflow run."


def _report_content(result):
    if isinstance(result, dict):
        return result.get("report", ""), result.get("tool_calls", []), result.get("react_steps", [])
    return str(result), [], []


def _store_report(session_state, name, report, calls=None, react_steps=None):
    session_state[f"{name}_report"] = report
    if name == "market":
        session_state["state"]["market_report"] = report
    session_state.setdefault("report_tool_traces", {})[name] = list(calls or [])
    session_state.setdefault("report_react_traces", {})[name] = list(react_steps or [])
    session_state.setdefault("tool_trace", []).extend(calls or [])
    session_state.setdefault("react_trace", []).extend(react_steps or [])
    _log(session_state, f"{name}_analyst_report", {"report": report})


def _window_dates_from_rows(rows):
    start = rows[0].get("date")
    last = rows[-1].get("date")
    end = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    return start, end


def _load_market(step_input, session_state=None):
    if session_state.get("preloaded_rows") is not None:
        rows = add_indicators(session_state["preloaded_rows"])
    else:
        rows = add_indicators(
            load_ohlcv(
                session_state["symbol"],
                session_state["start_date"],
                session_state["end_date"],
                provider=session_state.get("data_provider"),
            )
        )
    if not rows:
        raise ValueError("Agno workflow cannot run without market rows")

    state = make_pipeline_state(session_state["cash"])
    if session_state.get("initial_portfolio"):
        state["portfolio"] = dict(session_state["initial_portfolio"])

    session_state["rows"] = rows
    session_state["state"] = state
    session_state["portfolio"] = state["portfolio"]
    session_state["market"] = rows[-1]
    session_state["trade_date"] = session_state.get("trade_date") or rows[-1].get("date")
    benchmark_rows = None
    benchmark_start, benchmark_end = _window_dates_from_rows(rows)
    try:
        benchmark_rows = add_indicators(
            load_ohlcv(
                "SPY",
                benchmark_start,
                benchmark_end,
                provider=session_state.get("data_provider"),
            )
        )
    except Exception:
        benchmark_rows = None
    reflect_outcomes(session_state["symbol"], rows, benchmark_rows=benchmark_rows)
    memory_query = (
        f"{session_state['symbol']} market context on {session_state['trade_date']}: "
        f"close={session_state['market'].get('close')} rsi={session_state['market'].get('rsi')} "
        f"macd={session_state['market'].get('macd')} trend inputs from loaded OHLCV window."
    )
    session_state["past_context"] = load_lessons(session_state["symbol"], query_context=memory_query)
    _snapshot(session_state)
    return StepOutput(content={"rows": len(rows), "market_date": rows[-1].get("date")})


def _market_analyst_report(step_input, session_state=None):
    if _report_enabled(session_state):
        report_start, report_end = _window_dates_from_rows(session_state["rows"])
        result = run_market_analyst_report(
            session_state["symbol"],
            report_start,
            report_end,
            provider=session_state.get("data_provider"),
            trade_date=session_state.get("trade_date"),
            rows=session_state["rows"],
        )
        report, calls, react_steps = _report_content(result)
    else:
        calls = []
        react_steps = []
        report = market_report_from_signal(run_analyst(session_state["rows"]), session_state["market"])
    _store_report(session_state, "market", report, calls, react_steps)
    build_compressed_reports(session_state)
    _snapshot(session_state)
    return StepOutput(content=report)


def _news_analyst_report(step_input, session_state=None):
    if _report_enabled(session_state):
        result = run_news_analyst_report(session_state["symbol"], session_state.get("trade_date"))
        report, calls, react_steps = _report_content(result)
    else:
        calls = []
        react_steps = []
        report = _report_unavailable("News")
    _store_report(session_state, "news", report, calls, react_steps)
    build_compressed_reports(session_state)
    _snapshot(session_state)
    return StepOutput(content=report)


def _sentiment_analyst_report(step_input, session_state=None):
    if _report_enabled(session_state):
        result = run_sentiment_analyst_report(session_state["symbol"], session_state.get("trade_date"))
        report, calls, react_steps = _report_content(result)
    else:
        calls = []
        react_steps = []
        report = _report_unavailable("Sentiment")
    _store_report(session_state, "sentiment", report, calls, react_steps)
    build_compressed_reports(session_state)
    _snapshot(session_state)
    return StepOutput(content=report)


def _fundamentals_analyst_report(step_input, session_state=None):
    if _report_enabled(session_state):
        result = run_fundamentals_analyst_report(session_state["symbol"], session_state.get("trade_date"))
        report, calls, react_steps = _report_content(result)
    else:
        calls = []
        react_steps = []
        report = _report_unavailable("Fundamentals")
    _store_report(session_state, "fundamentals", report, calls, react_steps)
    build_compressed_reports(session_state)
    _snapshot(session_state)
    return StepOutput(content=report)


def _run_report_task(name, session_state):
    if not _report_enabled(session_state):
        if name == "market":
            return name, market_report_from_signal(run_analyst(session_state["rows"]), session_state["market"]), [], []
        title = "Sentiment" if name == "sentiment" else name.capitalize()
        return name, _report_unavailable(title), [], []

    if name == "market":
        report_start, report_end = _window_dates_from_rows(session_state["rows"])
        result = run_market_analyst_report(
            session_state["symbol"],
            report_start,
            report_end,
            provider=session_state.get("data_provider"),
            trade_date=session_state.get("trade_date"),
            rows=session_state["rows"],
        )
    elif name == "news":
        result = run_news_analyst_report(session_state["symbol"], session_state.get("trade_date"))
    elif name == "sentiment":
        result = run_sentiment_analyst_report(session_state["symbol"], session_state.get("trade_date"))
    elif name == "fundamentals":
        result = run_fundamentals_analyst_report(session_state["symbol"], session_state.get("trade_date"))
    else:
        raise ValueError(f"unknown analyst report task: {name}")
    report, calls, react_steps = _report_content(result)
    return name, report, calls, react_steps


def _analyst_reports_parallel(step_input, session_state=None):
    order = ["market", "news", "sentiment", "fundamentals"]
    cfg = get_config()
    use_parallel = cfg["parallel_analysts"] and cfg["llm_provider"].lower() != "local"
    if use_parallel:
        configured_workers = cfg.get("parallel_analyst_max_workers") or 0
        if configured_workers > 0:
            max_workers = configured_workers
        elif cfg["llm_provider"].lower() == "mistral":
            max_workers = max(1, len(get_mistral_api_keys()))
        else:
            max_workers = len(order)
        max_workers = max(1, min(len(order), max_workers))
    else:
        max_workers = 1
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tradeage-analyst") as executor:
        futures = {executor.submit(_run_report_task, name, session_state): name for name in order}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result_name, report, calls, react_steps = future.result()
            except Exception as exc:
                raise RuntimeError(f"{name} analyst report failed") from exc
            results[result_name] = (report, calls, react_steps)

    for name in order:
        report, calls, react_steps = results[name]
        _store_report(session_state, name, report, calls, react_steps)

    build_compressed_reports(session_state)
    build_evidence_ledger(session_state)
    _snapshot(session_state)
    return StepOutput(content={name: results[name][0] for name in order})


def _analyst(step_input, session_state=None):
    analyst = validate_analyst(run_analyst(session_state["rows"]))
    session_state["analyst"] = analyst
    if not session_state.get("market_report"):
        session_state["market_report"] = market_report_from_signal(analyst, session_state["market"])
    _log(session_state, "analyst", analyst)
    _snapshot(session_state)
    return StepOutput(content=analyst)


def _research_full_state(session_state, debate_state):
    reports = report_context(session_state)
    return {
        "company_of_interest": session_state["symbol"],
        "investment_debate_state": debate_state,
        "market_report": reports.get("market", ""),
        "sentiment_report": reports.get("sentiment", ""),
        "news_report": reports.get("news", ""),
        "fundamentals_report": reports.get("fundamentals", ""),
        "evidence_ledger": session_state.get("evidence_ledger", []),
        "evidence_prompt": session_state.get("evidence_prompt", ""),
    }


def _bull_researcher(step_input, session_state=None):
    state = session_state["state"]
    full_state = _research_full_state(session_state, state["investment_debate_state"])
    result = create_bull_researcher(get_llm("quick", "bull_researcher"))(full_state)
    state["investment_debate_state"] = result["investment_debate_state"]
    logic = ConditionalLogic(get_config()["max_debate_rounds"], get_config()["max_debate_rounds"])
    session_state.setdefault("routing_trace", []).append({
        "from": "Bull Researcher",
        "to": logic.should_continue_debate(state),
    })
    _snapshot(session_state)
    return StepOutput(content=state["investment_debate_state"])


def _bear_researcher(step_input, session_state=None):
    state = session_state["state"]
    full_state = _research_full_state(session_state, state["investment_debate_state"])
    result = create_bear_researcher(get_llm("quick", "bear_researcher"))(full_state)
    state["investment_debate_state"] = result["investment_debate_state"]
    logic = ConditionalLogic(get_config()["max_debate_rounds"], get_config()["max_debate_rounds"])
    session_state.setdefault("routing_trace", []).append({
        "from": "Bear Researcher",
        "to": logic.should_continue_debate(state),
    })
    _snapshot(session_state)
    return StepOutput(content=state["investment_debate_state"])


def _research_manager(step_input, session_state=None):
    state = session_state["state"]
    debate = validate_debate(
        summarize_debate(
            session_state["analyst"],
            session_state["market"],
            state["investment_debate_state"],
        )
    )
    manager_result = run_research_manager(
        {
            "company_of_interest": session_state["symbol"],
            "investment_debate_state": state["investment_debate_state"],
            "evidence_ledger": session_state.get("evidence_ledger", []),
            "evidence_prompt": session_state.get("evidence_prompt", ""),
        }
    )
    state["investment_debate_state"] = manager_result["investment_debate_state"]
    if "**Recommendation**: Buy" in manager_result["investment_plan"]:
        state["investment_debate_state"]["judge_decision"] = "bullish"
        debate["consensus_bias"] = "bullish"
    elif "**Recommendation**: Sell" in manager_result["investment_plan"]:
        state["investment_debate_state"]["judge_decision"] = "bearish"
        debate["consensus_bias"] = "bearish"
    else:
        state["investment_debate_state"]["judge_decision"] = "uncertain"
        debate["consensus_bias"] = "uncertain"
    session_state["debate"] = debate
    session_state["investment_plan"] = manager_result["investment_plan"]
    _log(session_state, "debate", debate)
    _snapshot(session_state)
    return StepOutput(content={"debate": debate, "investment_plan": manager_result["investment_plan"]})


def _trader(step_input, session_state=None):
    trader = validate_trader(
        run_trader(
            session_state["analyst"],
            session_state["debate"],
            session_state["investment_plan"],
            company=session_state["symbol"],
        )
    )
    session_state["trader"] = trader
    _log(session_state, "trader", trader)
    _snapshot(session_state)
    return StepOutput(content=trader)


def _risk_full_state(session_state):
    reports = report_context(session_state)
    return {
        "company_of_interest": session_state["symbol"],
        "risk_debate_state": session_state["state"]["risk_debate_state"],
        "market_report": reports.get("market", ""),
        "sentiment_report": reports.get("sentiment", ""),
        "news_report": reports.get("news", ""),
        "fundamentals_report": reports.get("fundamentals", ""),
        "trader_investment_plan": render_trader_context(session_state["trader"]),
    }


def _risk_debate_round(step_input, session_state=None):
    state = session_state["state"]
    full_state = _risk_full_state(session_state)
    logic = ConditionalLogic(get_config()["max_debate_rounds"], get_config()["max_debate_rounds"])
    full_state["risk_debate_state"] = run_aggressive_debator(full_state)
    session_state.setdefault("routing_trace", []).append({
        "from": "Aggressive Analyst",
        "to": logic.should_continue_risk_analysis({"risk_debate_state": full_state["risk_debate_state"]}),
    })
    full_state["risk_debate_state"] = run_conservative_debator(full_state)
    session_state.setdefault("routing_trace", []).append({
        "from": "Conservative Analyst",
        "to": logic.should_continue_risk_analysis({"risk_debate_state": full_state["risk_debate_state"]}),
    })
    full_state["risk_debate_state"] = run_neutral_debator(full_state)
    session_state.setdefault("routing_trace", []).append({
        "from": "Neutral Analyst",
        "to": logic.should_continue_risk_analysis({"risk_debate_state": full_state["risk_debate_state"]}),
    })
    state["risk_debate_state"] = full_state["risk_debate_state"]
    _snapshot(session_state)
    return StepOutput(content=state["risk_debate_state"])


def _risk_debate_log(step_input, session_state=None):
    risk_debate = session_state["state"]["risk_debate_state"]
    _log(session_state, "risk_debate", risk_debate)
    return StepOutput(content=risk_debate)


def _portfolio_manager(step_input, session_state=None):
    final = apply_final_portfolio_decision(
        company=session_state["symbol"],
        investment_plan=session_state["investment_plan"],
        trader=session_state["trader"],
        risk_debate_state=session_state["state"]["risk_debate_state"],
        past_context=session_state.get("past_context", ""),
        bull_validation=session_state["state"]["investment_debate_state"].get("bull_validation"),
        bear_validation=session_state["state"]["investment_debate_state"].get("bear_validation"),
        evidence_prompt=session_state.get("evidence_prompt", ""),
    )
    session_state["state"]["risk_debate_state"] = final["risk_debate_state"]
    session_state["portfolio_manager"] = final["portfolio_manager"]
    session_state["final_trader"] = validate_trader(final["trader"])
    _log(session_state, "portfolio_manager", final["portfolio_manager"])
    _snapshot(session_state)
    return StepOutput(content=final["portfolio_manager"])


def _should_execute(step_input):
    content = step_input.previous_step_content or {}
    if not isinstance(content, dict):
        return False
    return content.get("action") in {"buy", "sell"}


def _risk_gate(step_input, session_state=None):
    risk = validate_risk(
        run_risk_manager(
            session_state["final_trader"],
            session_state["market"],
            session_state["portfolio"]["cash"],
            session_state["portfolio"]["shares"],
            session_state["state"]["risk_debate_state"],
        )
    )
    session_state["risk"] = risk
    _log(session_state, "risk", risk)
    _snapshot(session_state)
    return StepOutput(content=risk)


def _hold_risk_gate(step_input, session_state=None):
    risk = validate_risk(
        {
            "approved": True,
            "adjusted_position": 0,
            "stop_loss": None,
            "risk_notes": "Portfolio Manager selected hold; execution branch preserved as a no-trade step.",
        }
    )
    session_state["risk"] = risk
    _log(session_state, "risk", risk)
    _snapshot(session_state)
    return StepOutput(content=risk)


def _execution(step_input, session_state=None):
    execution = execute_trade(
        session_state["portfolio"],
        session_state["final_trader"],
        session_state["risk"],
        session_state["market"],
    )
    session_state["execution"] = execution
    session_state["portfolio"] = execution["portfolio"]
    if session_state.get("record_memory", True):
        record_decision(
            session_state["symbol"],
            session_state.get("trade_date") or session_state["market"].get("date"),
            session_state["portfolio_manager"]["rating"],
            session_state["portfolio_manager"]["action"],
            execution["price"],
            final_decision=session_state["portfolio_manager"].get("final_trade_decision", ""),
            analyst=session_state.get("analyst"),
            debate=session_state.get("debate"),
            trader=session_state.get("final_trader"),
            risk=session_state.get("risk"),
            reports={
                "market": session_state.get("market_report", ""),
                "news": session_state.get("news_report", ""),
                "sentiment": session_state.get("sentiment_report", ""),
                "fundamentals": session_state.get("fundamentals_report", ""),
            },
            market=session_state.get("market"),
        )
    _log(session_state, "execution", execution)
    _snapshot(session_state)
    return StepOutput(content=execution)


def _evaluation(step_input, session_state=None):
    final_metrics = metrics(session_state["portfolio"]["equity_curve"])
    session_state["metrics"] = final_metrics
    _log(session_state, "evaluation", final_metrics)
    result = {
        "trace": {
            "workflow": {
                "runtime": "agno",
                "steps": WORKFLOW_STEPS,
                "session_id": session_state.get("workflow_session_id"),
                "checkpoint_db": session_state.get("workflow_db_path"),
            },
            "state_snapshot": session_state.get("state_snapshot"),
            "routing_trace": session_state.get("routing_trace", []),
            "tool_trace": session_state.get("tool_trace", []),
            "react_trace": session_state.get("react_trace", []),
            "reports": {
                "market": session_state.get("market_report", ""),
                "news": session_state.get("news_report", ""),
                "sentiment": session_state.get("sentiment_report", ""),
                "fundamentals": session_state.get("fundamentals_report", ""),
            },
            "compressed_reports": session_state.get("compressed_reports", {}),
            "report_compression": session_state.get("report_compression", {}),
            "evidence_ledger": session_state.get("evidence_ledger", []),
            "analyst": session_state["analyst"],
            "debate": session_state["debate"],
            "investment_debate_state": session_state["state"]["investment_debate_state"],
            "investment_plan": session_state["investment_plan"],
            "trader": session_state["trader"],
            "portfolio_manager": session_state["portfolio_manager"],
            "final_trader": session_state["final_trader"],
            "risk": session_state["risk"],
            "risk_debate_state": session_state["state"]["risk_debate_state"],
            "execution": session_state["execution"],
        },
        "portfolio": session_state["portfolio"],
        "metrics": final_metrics,
    }
    result["trace"]["full_state_log"] = write_full_state_log(session_state)
    session_state["result"] = result
    if session_state.get("checkpoint_enabled") and session_state.get("clear_checkpoint_on_success", True):
        clear_checkpoint(session_state["workflow_session_id"])
    return StepOutput(content=result)


def create_trading_workflow(session_state):
    debate_rounds = max(1, get_config()["max_debate_rounds"])
    db_path = Path(session_state["workflow_db_path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return Workflow(
        name="Penny Lane Workflow",
        session_id=session_state.get("workflow_session_id"),
        session_state=session_state,
        db=SqliteDb(db_file=str(db_path)),
        cache_session=True,
        store_events=True,
        telemetry=False,
        steps=[
            _step("load_market", _load_market),
            _step("analyst_reports_parallel", _analyst_reports_parallel),
            _step("analyst", _analyst),
            Loop(
                name="investment_debate_loop",
                max_iterations=debate_rounds,
                steps=[
                    _step("bull_researcher", _bull_researcher),
                    _step("bear_researcher", _bear_researcher),
                ],
            ),
            _step("research_manager", _research_manager),
            _step("trader", _trader),
            Loop(
                name="risk_debate_loop",
                max_iterations=debate_rounds,
                steps=[_step("risk_debate_round", _risk_debate_round)],
            ),
            _step("risk_debate_log", _risk_debate_log),
            _step("portfolio_manager", _portfolio_manager),
            Condition(
                name="execution_condition",
                evaluator=_should_execute,
                steps=[
                    _step("risk_gate", _risk_gate),
                    _step("execution", _execution),
                ],
                else_steps=[
                    _step("hold_risk_gate", _hold_risk_gate),
                    _step("execution", _execution),
                ],
            ),
            _step("evaluation", _evaluation),
        ],
    )


def run_agno_pipeline(
    symbol="AAPL",
    cash=10000,
    log_path=None,
    start_date=None,
    end_date=None,
    data_provider=None,
    preloaded_rows=None,
    initial_portfolio=None,
    trade_date=None,
    workflow_session_id=None,
    truncate_log=True,
    record_memory=True,
    tool_reports=True,
):
    if log_path and truncate_log:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        open(log_path, "w").close()

    config = get_config()
    session_state = {
        "symbol": symbol,
        "cash": float(cash),
        "log_path": str(log_path) if log_path else None,
        "start_date": start_date,
        "end_date": end_date,
        "data_provider": data_provider,
        "preloaded_rows": preloaded_rows,
        "initial_portfolio": initial_portfolio,
        "trade_date": trade_date,
        "workflow_session_id": workflow_session_id or f"{symbol.upper()}-{trade_date or end_date or 'live'}-{uuid4().hex}",
        "workflow_db_path": config["workflow_db_path"],
        "record_memory": record_memory,
        "tool_reports": tool_reports,
        "tool_trace": [],
        "react_trace": [],
        "routing_trace": [],
        "checkpoint_enabled": config["checkpoint_enabled"],
        "resume_checkpoint": workflow_session_id is not None,
        "clear_checkpoint_on_success": True,
    }
    workflow = create_trading_workflow(session_state)
    output = workflow.run(input={"symbol": symbol})
    result = output.content
    if not isinstance(result, dict) or "portfolio" not in result:
        raise RuntimeError("Agno workflow did not return a pipeline result")
    return result
