from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DebateStateModel(BaseModel):
    bull_history: str = ""
    bear_history: str = ""
    history: str = ""
    current_response: str = ""
    judge_decision: str = ""
    count: int = 0
    bull_argument: Optional[Dict[str, Any]] = None
    bear_argument: Optional[Dict[str, Any]] = None
    bull_validation: Optional[Dict[str, Any]] = None
    bear_validation: Optional[Dict[str, Any]] = None
    subagent_reviews: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_ledger: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_prompt: str = ""


class RiskDebateStateModel(BaseModel):
    aggressive_history: str = ""
    conservative_history: str = ""
    neutral_history: str = ""
    history: str = ""
    latest_speaker: str = ""
    current_aggressive_response: str = ""
    current_conservative_response: str = ""
    current_neutral_response: str = ""
    judge_decision: str = ""
    count: int = 0


class WorkflowStateSnapshot(BaseModel):
    symbol: str
    cash: float
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    trade_date: Optional[str] = None
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    market: Optional[Dict[str, Any]] = None
    market_report: str = ""
    sentiment_report: str = ""
    news_report: str = ""
    fundamentals_report: str = ""
    evidence_ledger: List[Dict[str, Any]] = Field(default_factory=list)
    analyst: Optional[Dict[str, Any]] = None
    investment_debate_state: DebateStateModel = Field(default_factory=DebateStateModel)
    risk_debate_state: RiskDebateStateModel = Field(default_factory=RiskDebateStateModel)
    investment_plan: str = ""
    trader: Optional[Dict[str, Any]] = None
    portfolio_manager: Optional[Dict[str, Any]] = None
    final_trader: Optional[Dict[str, Any]] = None
    risk: Optional[Dict[str, Any]] = None
    execution: Optional[Dict[str, Any]] = None


def workflow_snapshot(session_state: dict) -> WorkflowStateSnapshot:
    state = session_state.get("state", {})
    return WorkflowStateSnapshot(
        symbol=session_state["symbol"],
        cash=float(session_state["cash"]),
        start_date=session_state.get("start_date"),
        end_date=session_state.get("end_date"),
        trade_date=session_state.get("trade_date"),
        rows=session_state.get("rows", []),
        market=session_state.get("market"),
        market_report=session_state.get("market_report", ""),
        sentiment_report=session_state.get("sentiment_report", ""),
        news_report=session_state.get("news_report", ""),
        fundamentals_report=session_state.get("fundamentals_report", ""),
        evidence_ledger=session_state.get("evidence_ledger", []),
        analyst=session_state.get("analyst"),
        investment_debate_state=state.get("investment_debate_state", {}),
        risk_debate_state=state.get("risk_debate_state", {}),
        investment_plan=session_state.get("investment_plan", ""),
        trader=session_state.get("trader"),
        portfolio_manager=session_state.get("portfolio_manager"),
        final_trader=session_state.get("final_trader"),
        risk=session_state.get("risk"),
        execution=session_state.get("execution"),
    )
