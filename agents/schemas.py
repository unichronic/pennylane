from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PortfolioRating(str, Enum):
    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


class DebateEvidence(BaseModel):
    evidence_id: str = Field(description="Evidence ledger ID, for example market:1 or news:2.")
    claim: str = Field(description="Claim supported by the cited evidence.")
    why_it_matters: str = Field(description="Short explanation of why the evidence matters for the thesis.")


class RubricDimension(BaseModel):
    support_score: int = Field(
        ge=0,
        le=10,
        description="0-10 score for how strongly this dimension supports the side's thesis.",
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence in the score for this dimension.",
    )
    rationale: str = Field(description="Short explanation tied directly to the evidence.")
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Evidence ledger IDs that support this dimension score.",
    )


class DebateScorecard(BaseModel):
    trend_momentum: RubricDimension
    valuation_quality: RubricDimension
    news_event_risk: RubricDimension
    macro_sensitivity: RubricDimension
    downside_asymmetry: RubricDimension
    counterargument_strength: RubricDimension


class DebateArgument(BaseModel):
    thesis: str = Field(description="Main thesis for the side, written as concise plain prose.")
    scorecard: DebateScorecard
    key_claims: list[DebateEvidence] = Field(
        default_factory=list,
        description="Three to five concrete claims, each grounded in one or more evidence IDs.",
    )
    counterargument_response: str = Field(
        description="Direct response to the opposing side's strongest argument."
    )
    overall_conviction: float = Field(
        ge=0,
        le=1,
        description="Overall conviction in the side's thesis after weighing the scorecard.",
    )


class ResearchPlan(BaseModel):
    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )
    supporting_evidence_ids: list[str] = Field(
        default_factory=list,
        description="Validated evidence ledger IDs that directly support the recommendation.",
    )


def render_research_plan(plan: ResearchPlan) -> str:
    parts = [
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ]
    if plan.supporting_evidence_ids:
        parts.extend(["", f"**Supporting Evidence IDs**: {', '.join(plan.supporting_evidence_ids)}"])
    return "\n".join(parts)


class TraderProposal(BaseModel):
    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


class PortfolioDecision(BaseModel):
    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: Optional[float] = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )
    supporting_evidence_ids: list[str] = Field(
        default_factory=list,
        description="Validated evidence ledger IDs that directly support the final rating.",
    )


def render_pm_decision(decision: PortfolioDecision) -> str:
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    if decision.supporting_evidence_ids:
        parts.extend(["", f"**Supporting Evidence IDs**: {', '.join(decision.supporting_evidence_ids)}"])
    return "\n".join(parts)


TRENDS = {"bullish", "bearish", "neutral"}
BIASES = {"bullish", "bearish", "uncertain"}
ACTIONS = {"buy", "sell", "hold"}


def clamp(val, low=0, high=1):
    return max(low, min(high, val))


def require_keys(name, obj, keys):
    missing = [x for x in keys if x not in obj]
    if missing:
        raise ValueError(f"{name} missing keys: {', '.join(missing)}")
    return obj


def check_prob(name, val):
    if val < 0 or val > 1:
        raise ValueError(f"{name} must be between 0 and 1")


def validate_analyst(x):
    require_keys("analyst", x, ["trend", "confidence", "signals", "summary"])
    if x["trend"] not in TRENDS:
        raise ValueError("analyst trend is invalid")
    check_prob("analyst confidence", x["confidence"])
    require_keys("analyst signals", x["signals"], ["rsi", "macd"])
    return x


def validate_debate(x):
    require_keys("debate", x, ["bull_case", "bear_case", "key_risks", "consensus_bias"])
    if x["consensus_bias"] not in BIASES:
        raise ValueError("debate consensus is invalid")
    if not isinstance(x["key_risks"], list):
        raise ValueError("debate key_risks must be a list")
    return x


def validate_trader(x):
    require_keys("trader", x, ["action", "confidence", "reasoning", "position_size"])
    if x["action"] not in ACTIONS:
        raise ValueError("trader action is invalid")
    check_prob("trader confidence", x["confidence"])
    check_prob("trader position_size", x["position_size"])
    return x


def validate_risk(x):
    require_keys("risk", x, ["approved", "adjusted_position", "stop_loss", "risk_notes"])
    if not isinstance(x["approved"], bool):
        raise ValueError("risk approved must be bool")
    check_prob("risk adjusted_position", x["adjusted_position"])
    return x
