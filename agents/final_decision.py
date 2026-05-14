from agents.portfolio_manager import run_portfolio_manager
from agents.rating import parse_rating
from agents.trader import render_trader_context


BUY_RATINGS = {"Buy", "Overweight"}
SELL_RATINGS = {"Sell", "Underweight"}


def rating_to_action(rating):
    if rating in BUY_RATINGS:
        return "buy"
    if rating in SELL_RATINGS:
        return "sell"
    return "hold"


def apply_final_portfolio_decision(
    *,
    company,
    investment_plan,
    trader,
    risk_debate_state,
    past_context="",
    bull_validation=None,
    bear_validation=None,
    evidence_prompt="",
):
    state = {
        "company_of_interest": company,
        "investment_plan": investment_plan,
        "trader_investment_plan": render_trader_context(trader),
        "risk_debate_state": risk_debate_state,
        "past_context": past_context,
        "bull_validation": bull_validation or {},
        "bear_validation": bear_validation or {},
        "evidence_prompt": evidence_prompt or "",
    }
    result = run_portfolio_manager(state)
    final_decision = result["final_trade_decision"]
    rating = parse_rating(final_decision)
    action = rating_to_action(rating)
    final_trader = dict(trader)
    final_trader["action"] = action
    if action == "hold":
        final_trader["position_size"] = 0
    final_trader["reasoning"] = (
        f"{trader['reasoning']}\n\nPortfolio Manager final rating: {rating}.\n"
        f"{final_decision}"
    )
    return {
        "portfolio_manager": {
            "rating": rating,
            "action": action,
            "final_trade_decision": final_decision,
        },
        "risk_debate_state": result["risk_debate_state"],
        "trader": final_trader,
    }
