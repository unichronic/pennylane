import json
import re
from pathlib import Path

from config import get_config


_SAFE_TICKER = re.compile(r"^[A-Za-z0-9._\-\^]+$")


def _safe_ticker(symbol: str) -> str:
    value = symbol.upper()
    if not _SAFE_TICKER.fullmatch(value) or set(value) == {"."}:
        raise ValueError(f"unsafe ticker for results path: {symbol!r}")
    return value


def write_full_state_log(session_state: dict) -> str:
    cfg = get_config()
    symbol = _safe_ticker(session_state["symbol"])
    trade_date = str(session_state.get("trade_date") or session_state.get("end_date") or "latest")
    directory = Path(cfg["results_dir"]) / symbol / "PennyLaneCapital_logs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"full_states_log_{trade_date}.json"
    state = session_state.get("state", {})
    payload = {
        "company_of_interest": symbol,
        "trade_date": trade_date,
        "market_report": session_state.get("market_report", ""),
        "sentiment_report": session_state.get("sentiment_report", ""),
        "news_report": session_state.get("news_report", ""),
        "fundamentals_report": session_state.get("fundamentals_report", ""),
        "compressed_reports": session_state.get("compressed_reports", {}),
        "report_compression": session_state.get("report_compression", {}),
        "evidence_ledger": session_state.get("evidence_ledger", []),
        "tool_trace": session_state.get("tool_trace", []),
        "react_trace": session_state.get("react_trace", []),
        "investment_debate_state": state.get("investment_debate_state", {}),
        "trader_investment_decision": session_state.get("trader"),
        "risk_debate_state": state.get("risk_debate_state", {}),
        "investment_plan": session_state.get("investment_plan", ""),
        "final_trade_decision": session_state.get("portfolio_manager", {}).get("final_trade_decision", ""),
        "portfolio_manager": session_state.get("portfolio_manager"),
        "risk": session_state.get("risk"),
        "execution": session_state.get("execution"),
    }
    path.write_text(json.dumps(payload, indent=2))
    return str(path)
