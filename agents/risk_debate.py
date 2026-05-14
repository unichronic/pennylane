from agents.aggressive_debator import run_aggressive_debator
from agents.bull_researcher import market_report_from_signal
from agents.conservative_debator import run_conservative_debator
from agents.neutral_debator import run_neutral_debator
from config import get_config
from core.state import make_risk_debate_state


def run_risk_debate(analyst, trader, market, state=None):
    risk_state = state or make_risk_debate_state()
    full_state = {
        "risk_debate_state": risk_state,
        "market_report": market_report_from_signal(analyst, market),
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "trader_investment_plan": trader,
    }
    for _ in range(max(1, get_config()["max_debate_rounds"])):
        full_state["risk_debate_state"] = run_aggressive_debator(full_state)
        full_state["risk_debate_state"] = run_conservative_debator(full_state)
        full_state["risk_debate_state"] = run_neutral_debator(full_state)
    return full_state["risk_debate_state"]
