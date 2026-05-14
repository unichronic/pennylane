from agents.bull_researcher import create_bull_researcher
from agents.debate_support import (
    enforce_portfolio_decision_grounding,
    enforce_research_plan_grounding,
)
from agents.llm_factory import get_llm
from agents.schemas import PortfolioDecision, PortfolioRating, ResearchPlan
from core.evidence_ledger import build_evidence_ledger
from core.state import make_investment_debate_state


def test_build_evidence_ledger_uses_react_steps():
    session_state = {
        "market_report": "market report",
        "news_report": "news report",
        "sentiment_report": "sentiment report",
        "fundamentals_report": "fundamentals report",
        "report_react_traces": {
            "market": [{"action": "get_indicators", "observation": "RSI=62 MACD=1.2"}],
            "news": [{"action": "get_news", "observation": "Product launch and upbeat guidance"}],
        },
    }

    ledger = build_evidence_ledger(session_state)

    assert any(item["evidence_id"] == "market:1" for item in ledger)
    assert any(item["evidence_id"] == "news:1" for item in ledger)
    assert session_state["evidence_prompt"]


def test_research_plan_is_downgraded_without_cited_validated_evidence():
    plan = ResearchPlan(
        recommendation=PortfolioRating.BUY,
        rationale="Bullish",
        strategic_actions="Buy now",
        supporting_evidence_ids=["market:1"],
    )

    grounded = enforce_research_plan_grounding(
        plan,
        {
            "bull": {"supported_ids": ["market:1", "news:1"], "supported_dimensions": ["trend_momentum"]},
            "bear": {"supported_ids": [], "supported_dimensions": []},
        },
    )

    assert grounded.recommendation == PortfolioRating.HOLD


def test_portfolio_decision_is_downgraded_without_cited_validated_evidence():
    decision = PortfolioDecision(
        rating=PortfolioRating.SELL,
        executive_summary="Sell",
        investment_thesis="Bearish",
        supporting_evidence_ids=["news:1"],
    )

    grounded = enforce_portfolio_decision_grounding(
        decision,
        {
            "bull": {"supported_ids": [], "supported_dimensions": []},
            "bear": {"supported_ids": ["news:1", "market:1"], "supported_dimensions": ["news_event_risk"]},
        },
    )

    assert grounded.rating == PortfolioRating.HOLD


def test_bull_researcher_populates_scorecard_validation_and_subagent_reviews(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    state = {
        "company_of_interest": "AAPL",
        "investment_debate_state": make_investment_debate_state(),
        "market_report": "trend=bullish confidence=0.7 close=180 rsi=61 macd=1.4 macd_hist=0.8 macd_hist_positive=True",
        "news_report": "AAPL launches new products and guidance remains constructive.",
        "sentiment_report": "News-derived sentiment: positive",
        "fundamentals_report": "revenueGrowth: 0.12\nprofitMargins: 0.24\nreturnOnEquity: 0.38",
        "evidence_ledger": [
            {"evidence_id": "market:1", "report": "market", "tool": "get_indicators", "snippet": "trend bullish"},
            {"evidence_id": "news:1", "report": "news", "tool": "get_news", "snippet": "constructive guidance"},
            {"evidence_id": "fundamentals:1", "report": "fundamentals", "tool": "get_fundamentals", "snippet": "strong margins"},
            {"evidence_id": "sentiment:1", "report": "sentiment", "tool": "sentiment_from_news", "snippet": "positive sentiment"},
        ],
        "evidence_prompt": "\n".join(
            [
                "[market:1] report=market tool=get_indicators snippet=trend bullish",
                "[news:1] report=news tool=get_news snippet=constructive guidance",
                "[fundamentals:1] report=fundamentals tool=get_fundamentals snippet=strong margins",
                "[sentiment:1] report=sentiment tool=sentiment_from_news snippet=positive sentiment",
            ]
        ),
    }

    result = create_bull_researcher(get_llm("quick", "bull_researcher"))(state)["investment_debate_state"]

    assert result["bull_argument"]["scorecard"]["trend_momentum"]["support_score"] >= 0
    assert result["bull_validation"]["supported_ids"]
    assert len(result["subagent_reviews"]) == 3
