from agents.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from agents.debate_support import (
    enforce_portfolio_decision_grounding,
    render_validation_summary,
)
from agents.llm_factory import get_llm
from agents.schemas import PortfolioDecision, render_pm_decision
from agents.structured import (
    bind_structured,
)
from core.prompt_context import compact_text, prompt_limits


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]
        limits = prompt_limits()
        prompt_history = compact_text(history, limits["risk_history"], "risk debate history")
        prompt_research_plan = compact_text(research_plan, limits["plan"], "research manager investment plan")
        prompt_trader_plan = compact_text(trader_plan, limits["plan"], "trader transaction proposal")
        bull_validation = state.get("bull_validation") or {}
        bear_validation = state.get("bear_validation") or {}
        evidence_prompt = compact_text(state.get("evidence_prompt", ""), limits["plan"], "evidence ledger")

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
        - Research Manager's investment plan: **{prompt_research_plan}**
        - Trader's transaction proposal: **{prompt_trader_plan}**
        {lessons_line}
        - Validated evidence ledger:
        {evidence_prompt}
        - Bull validation summary:
        {render_validation_summary("bull", bull_validation)}
        - Bear validation summary:
        {render_validation_summary("bear", bear_validation)}
        **Risk Analysts Debate History:**
        {prompt_history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""

        final_decision = structured_llm.invoke(prompt)
        final_decision = enforce_portfolio_decision_grounding(
            final_decision,
            {"bull": bull_validation, "bear": bear_validation},
        )
        final_trade_decision = render_pm_decision(final_decision)

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node


def run_portfolio_manager(state):
    return create_portfolio_manager(get_llm("deep", "portfolio_manager"))(state)
