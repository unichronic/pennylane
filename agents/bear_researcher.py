from agents.bull_researcher import (
    _ensure_evidence_context,
    _render_specialist_reviews,
    market_report_from_signal,
    maybe_run_specialist_subagents,
)
from agents.agent_utils import build_instrument_context
from agents.debate_support import render_argument, validate_argument_grounding
from agents.llm_factory import get_llm
from agents.schemas import DebateArgument
from agents.structured import bind_structured
from core.prompt_context import compact_text, prompt_limits


def create_bear_researcher(llm):
    structured_llm = bind_structured(llm, DebateArgument, "Bear Researcher")

    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        instrument_context = build_instrument_context(state.get("company_of_interest", "the stock"))
        limits = prompt_limits()
        prompt_history = compact_text(history, limits["investment_history"], "investment debate history")
        prompt_current_response = compact_text(current_response, limits["single_argument"], "last bull argument")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        _, evidence_prompt, evidence_map = _ensure_evidence_context(state)
        specialist_reviews = maybe_run_specialist_subagents(state)
        prompt_specialists = _render_specialist_reviews(specialist_reviews)

        prompt = f"""You are a Bear Analyst making the case against investing in the stock. Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators. Leverage the provided research and data to highlight potential downsides and counter bullish arguments effectively.

{instrument_context}

Key points to focus on:

- Risks and Challenges: Highlight factors like market saturation, financial instability, or macroeconomic threats that could hinder the stock's performance.
- Competitive Weaknesses: Emphasize vulnerabilities such as weaker market positioning, declining innovation, or threats from competitors.
- Negative Indicators: Use evidence from financial data, market trends, or recent adverse news to support your position.
- Bull Counterpoints: Critically analyze the bull argument with specific data and sound reasoning, exposing weaknesses or over-optimistic assumptions.
- Engagement: Present your argument in a conversational style, directly engaging with the bull analyst's points and debating effectively rather than simply listing facts.

Resources available:

        Market research report: {market_research_report}
        Social media sentiment report: {sentiment_report}
        Latest world affairs news: {news_report}
        Company fundamentals report: {fundamentals_report}
        Evidence ledger:
        {evidence_prompt}
        Specialist subagent reviews:
        {prompt_specialists}
        Conversation history of the debate: {prompt_history}
        Last bull argument: {prompt_current_response}
        Use this information to deliver a compelling bear argument, refute the bull's claims, and engage in a dynamic debate that demonstrates the risks and weaknesses of investing in the stock.

        Return a structured score-based argument. Every score dimension and every key claim must cite evidence IDs from the evidence ledger.
        If a claim cannot be grounded in the evidence ledger, do not include it.
        """

        response = structured_llm.invoke(prompt)
        validation = validate_argument_grounding(response, evidence_map)
        argument = f"Bear Analyst: {render_argument('bear', response)}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
            "bull_argument": investment_debate_state.get("bull_argument"),
            "bull_validation": investment_debate_state.get("bull_validation"),
            "bear_argument": response.model_dump(),
            "bear_validation": validation,
            "subagent_reviews": specialist_reviews,
            "evidence_ledger": state.get("evidence_ledger", []),
            "evidence_prompt": state.get("evidence_prompt", ""),
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node


def run_bear_researcher(state, analyst, market):
    full_state = {
        "investment_debate_state": state,
        "market_report": market_report_from_signal(analyst, market),
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "evidence_ledger": state.get("evidence_ledger", []),
        "evidence_prompt": state.get("evidence_prompt", ""),
    }
    return create_bear_researcher(get_llm("quick", "bear_researcher"))(full_state)["investment_debate_state"]
