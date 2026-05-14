import functools
import re

from langchain_core.messages import AIMessage

from agents.agent_utils import build_instrument_context
from agents.llm_factory import get_llm
from agents.schemas import TraderProposal, render_trader_proposal
from agents.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    "Anchor your reasoning in the analysts' reports and the research plan."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on a comprehensive analysis by a team of analysts, here is an investment "
                    f"plan tailored for {company_name}. {instrument_context} This plan incorporates "
                    f"insights from current technical market trends, macroeconomic indicators, and "
                    f"social media sentiment. Use this plan as a foundation for evaluating your next "
                    f"trading decision.\n\nProposed Investment Plan: {investment_plan}\n\n"
                    f"Leverage these insights to make an informed and strategic decision."
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")


def parse_trader_plan(plan, analyst):
    action = "hold"
    m = re.search(r"FINAL TRANSACTION PROPOSAL: \*\*(BUY|HOLD|SELL)\*\*", plan)
    if m:
        action = m.group(1).lower()
    elif "**Action**: Buy" in plan:
        action = "buy"
    elif "**Action**: Sell" in plan:
        action = "sell"

    conf = analyst["confidence"]

    if action == "hold":
        size = 0
    else:
        size = round(min(0.5, max(0.1, conf * 0.45)), 2)

    reasoning = plan

    return {
        "action": action,
        "confidence": conf,
        "reasoning": reasoning,
        "position_size": size,
    }


def render_trader_context(trader):
    if isinstance(trader, str):
        return trader
    if not isinstance(trader, dict):
        return str(trader)
    lines = [
        f"Action: {trader.get('action', 'unknown')}",
        f"Confidence: {trader.get('confidence', 'unknown')}",
        f"Position size: {trader.get('position_size', 'unknown')}",
    ]
    if trader.get("entry_price") is not None:
        lines.append(f"Entry price: {trader.get('entry_price')}")
    if trader.get("stop_loss") is not None:
        lines.append(f"Stop loss: {trader.get('stop_loss')}")
    reasoning = trader.get("reasoning")
    if reasoning:
        lines.append(f"Reasoning:\n{reasoning}")
    return "\n".join(lines)


def run_trader(analyst, debate, investment_plan=None, company="SAMPLE"):
    if investment_plan is None:
        bias = debate["consensus_bias"]
        if bias == "bullish":
            investment_plan = "**Recommendation**: Buy"
        elif bias == "bearish":
            investment_plan = "**Recommendation**: Sell"
        else:
            investment_plan = "**Recommendation**: Hold"
    state = {"company_of_interest": company, "investment_plan": investment_plan}
    result = create_trader(get_llm("quick", "trader"))(state)
    return parse_trader_plan(result["trader_investment_plan"], analyst)
