from __future__ import annotations

from typing import Any, Mapping

from agents.rating import RATINGS_5_TIER
from agents.schemas import DebateArgument, PortfolioDecision, PortfolioRating, ResearchPlan


RUBRIC_LABELS = {
    "trend_momentum": "Trend/Momentum",
    "valuation_quality": "Valuation/Quality",
    "news_event_risk": "News/Event Risk",
    "macro_sensitivity": "Macro Sensitivity",
    "downside_asymmetry": "Downside Asymmetry",
    "counterargument_strength": "Counterargument Strength",
}


BUYISH_RATINGS = {PortfolioRating.BUY, PortfolioRating.OVERWEIGHT}
SELLISH_RATINGS = {PortfolioRating.SELL, PortfolioRating.UNDERWEIGHT}


def _all_argument_evidence_ids(argument: DebateArgument) -> list[str]:
    ids = []
    for field in RUBRIC_LABELS:
        ids.extend(getattr(argument.scorecard, field).evidence_ids)
    for item in argument.key_claims:
        ids.extend(item.evidence_id for item in [item] if item.evidence_id)
    seen = []
    for evidence_id in ids:
        if evidence_id and evidence_id not in seen:
            seen.append(evidence_id)
    return seen


def _coerce_argument(argument: DebateArgument | Mapping[str, Any]) -> DebateArgument:
    if isinstance(argument, DebateArgument):
        return argument
    return DebateArgument.model_validate(argument)


def render_scorecard_table(argument: DebateArgument) -> str:
    argument = _coerce_argument(argument)
    lines = [
        "| Dimension | Score | Confidence | Evidence IDs |",
        "| --- | ---: | ---: | --- |",
    ]
    for field, label in RUBRIC_LABELS.items():
        dimension = getattr(argument.scorecard, field)
        evidence_ids = ", ".join(dimension.evidence_ids) if dimension.evidence_ids else "-"
        lines.append(
            f"| {label} | {dimension.support_score}/10 | {dimension.confidence:.2f} | {evidence_ids} |"
        )
    return "\n".join(lines)


def render_argument(side: str, argument: DebateArgument) -> str:
    argument = _coerce_argument(argument)
    title = "Bull" if side.lower().startswith("bull") else "Bear"
    claims = []
    for item in argument.key_claims:
        ids = f"[{', '.join([item.evidence_id] if item.evidence_id else [])}]"
        claims.append(f"- {item.claim} {ids} {item.why_it_matters}".strip())
    claims_block = "\n".join(claims) if claims else "- No grounded claims returned."
    return "\n".join(
        [
            f"{title} Thesis: {argument.thesis}",
            "",
            render_scorecard_table(argument),
            "",
            "**Grounded Claims**:",
            claims_block,
            "",
            f"**Counterargument Response**: {argument.counterargument_response}",
            "",
            f"**Overall Conviction**: {argument.overall_conviction:.2f}",
        ]
    )


def validate_argument_grounding(argument: DebateArgument, evidence_index: Mapping[str, Any]) -> dict[str, Any]:
    argument = _coerce_argument(argument)
    supported_ids = []
    missing_ids = []
    unsupported_claims = []
    supported_dimensions = []
    unsupported_dimensions = []

    for field in RUBRIC_LABELS:
        dimension = getattr(argument.scorecard, field)
        valid_ids = [item for item in dimension.evidence_ids if item in evidence_index]
        if valid_ids:
            supported_dimensions.append(field)
        else:
            unsupported_dimensions.append(field)
        for evidence_id in dimension.evidence_ids:
            if evidence_id in evidence_index:
                if evidence_id not in supported_ids:
                    supported_ids.append(evidence_id)
            elif evidence_id not in missing_ids:
                missing_ids.append(evidence_id)

    for item in argument.key_claims:
        if item.evidence_id in evidence_index:
            if item.evidence_id not in supported_ids:
                supported_ids.append(item.evidence_id)
        else:
            if item.evidence_id and item.evidence_id not in missing_ids:
                missing_ids.append(item.evidence_id)
            unsupported_claims.append(item.claim)

    return {
        "all_evidence_ids": _all_argument_evidence_ids(argument),
        "supported_ids": supported_ids,
        "missing_ids": missing_ids,
        "unsupported_claims": unsupported_claims,
        "supported_dimensions": supported_dimensions,
        "unsupported_dimensions": unsupported_dimensions,
        "grounded_claim_count": len(argument.key_claims) - len(unsupported_claims),
    }


def render_validation_summary(side: str, validation: Mapping[str, Any]) -> str:
    label = "Bull" if side.lower().startswith("bull") else "Bear"
    supported = ", ".join(validation.get("supported_ids", [])) or "none"
    missing = ", ".join(validation.get("missing_ids", [])) or "none"
    unsupported_claims = validation.get("unsupported_claims", [])
    unsupported = "; ".join(unsupported_claims) if unsupported_claims else "none"
    unsupported_dimensions = validation.get("unsupported_dimensions", [])
    unsupported_dims = (
        ", ".join(RUBRIC_LABELS.get(name, name) for name in unsupported_dimensions)
        if unsupported_dimensions
        else "none"
    )
    return "\n".join(
        [
            f"{label} validated evidence IDs: {supported}",
            f"{label} missing evidence IDs: {missing}",
            f"{label} unsupported claims: {unsupported}",
            f"{label} unsupported score dimensions: {unsupported_dims}",
        ]
    )


def recommendation_winner(value: PortfolioRating | str) -> str | None:
    if isinstance(value, str):
        matched = next((item for item in PortfolioRating if item.value.lower() == value.lower()), None)
        if matched is None:
            return None
        value = matched
    if value in BUYISH_RATINGS:
        return "bull"
    if value in SELLISH_RATINGS:
        return "bear"
    return None


def _enough_support(validation: Mapping[str, Any]) -> bool:
    return len(validation.get("supported_ids", [])) >= 2 and len(validation.get("supported_dimensions", [])) >= 2


def enforce_research_plan_grounding(
    plan: ResearchPlan,
    validations: Mapping[str, Mapping[str, Any]],
) -> ResearchPlan:
    winner = recommendation_winner(plan.recommendation)
    if winner is None:
        plan.supporting_evidence_ids = []
        return plan
    validation = validations.get(winner, {})
    if not _enough_support(validation):
        return ResearchPlan(
            recommendation=PortfolioRating.HOLD,
            rationale=(
                "Unsupported claims were rejected because the decisive side did not carry "
                "enough validated evidence across the scoring model."
            ),
            strategic_actions="Hold and wait for broader validated support before committing capital.",
            supporting_evidence_ids=[],
        )
    filtered_ids = [
        item for item in plan.supporting_evidence_ids if item in validation.get("supported_ids", [])
    ]
    if len(filtered_ids) < 2:
        return ResearchPlan(
            recommendation=PortfolioRating.HOLD,
            rationale=(
                "Hold because the recommendation was not backed by enough cited validated evidence IDs."
            ),
            strategic_actions="Wait for a cleaner setup with explicit evidence-backed support.",
            supporting_evidence_ids=[],
        )
    plan.supporting_evidence_ids = filtered_ids
    return plan


def enforce_portfolio_decision_grounding(
    decision: PortfolioDecision,
    validations: Mapping[str, Mapping[str, Any]],
) -> PortfolioDecision:
    winner = recommendation_winner(decision.rating)
    if winner is None:
        decision.supporting_evidence_ids = []
        return decision
    validation = validations.get(winner, {})
    if not _enough_support(validation):
        return PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary=(
                "Hold because the decisive side did not retain enough validated evidence "
                "after unsupported claims were removed."
            ),
            investment_thesis=(
                "The final rating was downgraded to Hold because the validated evidence set "
                "was too thin to support a decisive allocation change."
            ),
            price_target=None,
            time_horizon=decision.time_horizon,
            supporting_evidence_ids=[],
        )
    filtered_ids = [
        item for item in decision.supporting_evidence_ids if item in validation.get("supported_ids", [])
    ]
    if len(filtered_ids) < 2:
        return PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="Hold because the final rating did not cite enough validated evidence IDs.",
            investment_thesis=(
                "The portfolio decision was downgraded because it did not cite a sufficient validated evidence set."
            ),
            price_target=None,
            time_horizon=decision.time_horizon,
            supporting_evidence_ids=[],
        )
    decision.supporting_evidence_ids = filtered_ids
    return decision


def ratings_5_tier_text() -> str:
    return ", ".join(RATINGS_5_TIER)
