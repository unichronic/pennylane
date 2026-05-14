from agents.bear_researcher import run_bear_researcher
from agents.bull_researcher import run_bull_researcher
from agents.debate_support import RUBRIC_LABELS, render_argument
from agents.research_manager import run_research_manager
from agents.schemas import validate_debate
from config import get_config
from core.state import make_investment_debate_state


def run_debate_round(analyst, market, state=None):
    state = state or make_investment_debate_state()
    for _ in range(max(1, get_config()["max_debate_rounds"])):
        state = run_bull_researcher(state, analyst, market)
        state = run_bear_researcher(state, analyst, market)
    debate = summarize_debate(analyst, market, state)
    manager_state = {
        "company_of_interest": "SAMPLE",
        "investment_debate_state": state,
        "evidence_ledger": state.get("evidence_ledger", []),
        "evidence_prompt": state.get("evidence_prompt", ""),
    }
    manager_result = run_research_manager(manager_state)
    state = manager_result["investment_debate_state"]
    if "**Recommendation**: Buy" in manager_result["investment_plan"]:
        state["judge_decision"] = "bullish"
        debate["consensus_bias"] = "bullish"
    elif "**Recommendation**: Sell" in manager_result["investment_plan"]:
        state["judge_decision"] = "bearish"
        debate["consensus_bias"] = "bearish"
    else:
        state["judge_decision"] = "uncertain"
        debate["consensus_bias"] = "uncertain"
    return {
        "debate": debate,
        "investment_debate_state": state,
        "investment_plan": manager_result["investment_plan"],
    }


def summarize_debate(analyst, market, state):
    trend = analyst["trend"]
    conf = analyst["confidence"]
    rsi = analyst["signals"].get("rsi")

    bull_argument = state.get("bull_argument")
    bear_argument = state.get("bear_argument")

    risks = [
        "indicator signals can lag fast markets",
        "single-symbol backtests can overfit recent price action",
    ]
    if rsi is not None and (rsi > 70 or rsi < 30):
        risks.append("RSI is at an extreme")
    for side, argument in (("bull", bull_argument), ("bear", bear_argument)):
        if not argument:
            continue
        validation = state.get(f"{side}_validation") or {}
        if validation.get("unsupported_claims"):
            risks.append(f"{side} side included unsupported claims that were rejected")
        for field in validation.get("unsupported_dimensions", [])[:2]:
            risks.append(f"{side} {RUBRIC_LABELS.get(field, field)} lacked grounded evidence")

    if conf >= 0.65 and trend in {"bullish", "bearish"}:
        consensus = trend
    else:
        consensus = "uncertain"

    return validate_debate({
        "bull_case": (
            f"Bull Analyst: {render_argument('bull', bull_argument)}"
            if bull_argument
            else state["bull_history"].strip()
        ),
        "bear_case": (
            f"Bear Analyst: {render_argument('bear', bear_argument)}"
            if bear_argument
            else state["bear_history"].strip()
        ),
        "key_risks": list(dict.fromkeys(risks)),
        "consensus_bias": consensus,
    })


def run_debate(analyst, market):
    return run_debate_round(analyst, market)["debate"]
