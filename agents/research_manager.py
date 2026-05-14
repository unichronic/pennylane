from agents.llm_factory import get_llm
from agents.debate_support import (
    enforce_research_plan_grounding,
    render_validation_summary,
)
from agents.schemas import ResearchPlan, render_research_plan
from agents.agent_utils import build_instrument_context
from agents.structured import (
    bind_structured,
)
from core.prompt_context import compact_text, prompt_limits


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])
        history = state["investment_debate_state"].get("history", "")
        prompt_history = compact_text(
            history,
            prompt_limits()["investment_history"],
            "investment debate history",
        )

        investment_debate_state = state["investment_debate_state"]
        bull_validation = investment_debate_state.get("bull_validation") or {}
        bear_validation = investment_debate_state.get("bear_validation") or {}
        evidence_prompt = compact_text(
            state.get("evidence_prompt", ""),
            prompt_limits()["plan"],
            "evidence ledger",
        )

        prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction in the bull thesis; recommend taking or growing the position
- **Overweight**: Constructive view; recommend gradually increasing exposure
- **Hold**: Balanced view; recommend maintaining the current position
- **Underweight**: Cautious view; recommend trimming exposure
- **Sell**: Strong conviction in the bear thesis; recommend exiting or avoiding the position

Commit to a clear stance whenever the debate's strongest arguments warrant one; reserve Hold for situations where the evidence on both sides is genuinely balanced.

---

        **Debate History:**
        {prompt_history}

        **Validated Evidence Ledger:**
        {evidence_prompt}

        **Bull Validation Summary:**
        {render_validation_summary("bull", bull_validation)}

        **Bear Validation Summary:**
        {render_validation_summary("bear", bear_validation)}

        Ignore unsupported claims. If the stronger side does not retain at least two validated evidence IDs across at least two score dimensions, return Hold.
        Cite only validated evidence IDs in the final recommendation."""

        plan = structured_llm.invoke(prompt)
        plan = enforce_research_plan_grounding(
            plan,
            {"bull": bull_validation, "bear": bear_validation},
        )
        investment_plan = render_research_plan(plan)

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
            "bull_argument": investment_debate_state.get("bull_argument"),
            "bear_argument": investment_debate_state.get("bear_argument"),
            "bull_validation": bull_validation,
            "bear_validation": bear_validation,
            "subagent_reviews": investment_debate_state.get("subagent_reviews", []),
            "evidence_ledger": investment_debate_state.get("evidence_ledger", []),
            "evidence_prompt": investment_debate_state.get("evidence_prompt", ""),
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node


def run_research_manager(state):
    node = create_research_manager(get_llm("deep", "research_manager"))
    return node(state)
