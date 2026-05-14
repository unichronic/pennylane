from core.state import AgentState


class ConditionalLogic:
    """Small router for the graph."""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Store round caps."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def should_continue_market(self, state: AgentState):
        """Route the market branch."""
        msgs = state["messages"]
        last_msg = msgs[-1]
        if last_msg.tool_calls:
            return "tools_market"
        return "Msg Clear Market"

    def should_continue_social(self, state: AgentState):
        """Route the social branch."""
        msgs = state["messages"]
        last_msg = msgs[-1]
        if last_msg.tool_calls:
            return "tools_social"
        return "Msg Clear Social"

    def should_continue_news(self, state: AgentState):
        """Route the news branch."""
        msgs = state["messages"]
        last_msg = msgs[-1]
        if last_msg.tool_calls:
            return "tools_news"
        return "Msg Clear News"

    def should_continue_fundamentals(self, state: AgentState):
        """Route the fundamentals branch."""
        msgs = state["messages"]
        last_msg = msgs[-1]
        if last_msg.tool_calls:
            return "tools_fundamentals"
        return "Msg Clear Fundamentals"

    def should_continue_debate(self, state: AgentState) -> str:
        """Pick the next investment debate speaker."""

        if (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Pick the next risk debate speaker."""
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):
            return "Portfolio Manager"
        if state["risk_debate_state"]["latest_speaker"].startswith("Aggressive"):
            return "Conservative Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Conservative"):
            return "Neutral Analyst"
        return "Aggressive Analyst"
