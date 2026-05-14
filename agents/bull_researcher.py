from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.agent_utils import build_instrument_context
from agents.debate_support import render_argument, validate_argument_grounding
from agents.llm_factory import get_llm
from agents.schemas import DebateArgument
from agents.structured import bind_structured
from config import get_config
from core.prompt_context import compact_text, prompt_limits
from core.evidence_ledger import evidence_index, render_evidence_prompt


SPECIALIST_SPECS = [
    {
        "name": "Trend and Asymmetry Specialist",
        "focus": "trend/momentum and downside asymmetry",
        "reports": ("market_report", "news_report"),
    },
    {
        "name": "Quality and Macro Specialist",
        "focus": "valuation/quality and macro sensitivity",
        "reports": ("fundamentals_report", "news_report"),
    },
    {
        "name": "Event and Counterargument Specialist",
        "focus": "news/event risk and the strongest likely counterarguments",
        "reports": ("news_report", "sentiment_report", "market_report"),
    },
]


def _fallback_evidence_ledger(state):
    report_map = {
        "market": state.get("market_report", ""),
        "sentiment": state.get("sentiment_report", ""),
        "news": state.get("news_report", ""),
        "fundamentals": state.get("fundamentals_report", ""),
    }
    ledger = []
    for report_name, text in report_map.items():
        lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
        for idx, line in enumerate(lines[:3], start=1):
            ledger.append({
                "evidence_id": f"{report_name}:report:{idx}",
                "report": report_name,
                "tool": "report_excerpt",
                "snippet": line[:220],
            })
    return ledger


def _ensure_evidence_context(state):
    ledger = list(state.get("evidence_ledger") or [])
    if not ledger:
        ledger = _fallback_evidence_ledger(state)
        state["evidence_ledger"] = ledger
    prompt = state.get("evidence_prompt") or render_evidence_prompt(ledger)
    state["evidence_prompt"] = prompt
    return ledger, prompt, evidence_index(ledger)


def _subagent_prompt(spec, state, evidence_prompt):
    limits = prompt_limits()
    sections = []
    for key in spec["reports"]:
        value = compact_text(state.get(key, ""), limits["plan"], key.replace("_", " "))
        if value:
            sections.append(f"{key}:\n{value}")
    context = "\n\n".join(sections) or "No specialist report context provided."
    return f"""You are the {spec["name"]}.

Focus only on {spec["focus"]}. Read the supplied evidence and produce 3-5 concise bullets.
Each bullet must name the dimension it informs, state whether it favors longs, shorts, or is mixed,
and cite one or more evidence IDs from the evidence ledger.

Evidence ledger:
{evidence_prompt}

Relevant reports:
{context}
"""


def maybe_run_specialist_subagents(state):
    investment_debate_state = state.get("investment_debate_state", {})
    existing = investment_debate_state.get("subagent_reviews") or []
    if existing or not get_config().get("debate_subagents_enabled", True):
        return existing

    _, evidence_prompt, _ = _ensure_evidence_context(state)
    provider = get_config()["llm_provider"].lower()
    worker_count = 1 if provider == "local" else min(len(SPECIALIST_SPECS), 3)
    reviews = []

    def run_spec(spec):
        prompt = _subagent_prompt(spec, state, evidence_prompt)
        content = get_llm("quick", spec["name"]).invoke(prompt).content
        return {
            "name": spec["name"],
            "focus": spec["focus"],
            "summary": str(content).strip(),
        }

    if worker_count == 1:
        for spec in SPECIALIST_SPECS:
            reviews.append(run_spec(spec))
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="tradeage-subagent") as executor:
            futures = {executor.submit(run_spec, spec): spec for spec in SPECIALIST_SPECS}
            for future in as_completed(futures):
                reviews.append(future.result())
        reviews.sort(key=lambda item: next(idx for idx, spec in enumerate(SPECIALIST_SPECS) if spec["name"] == item["name"]))

    investment_debate_state["subagent_reviews"] = reviews
    return reviews


def _render_specialist_reviews(reviews):
    if not reviews:
        return "No specialist subagent reviews were generated."
    parts = []
    for item in reviews:
        parts.append(f"{item['name']} ({item['focus']}):\n{item['summary']}")
    return "\n\n".join(parts)


def create_bull_researcher(llm):
    structured_llm = bind_structured(llm, DebateArgument, "Bull Researcher")

    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        instrument_context = build_instrument_context(state.get("company_of_interest", "the stock"))
        limits = prompt_limits()
        prompt_history = compact_text(history, limits["investment_history"], "investment debate history")
        prompt_current_response = compact_text(current_response, limits["single_argument"], "last bear argument")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        ledger, evidence_prompt, evidence_map = _ensure_evidence_context(state)
        specialist_reviews = maybe_run_specialist_subagents(state)
        prompt_specialists = _render_specialist_reviews(specialist_reviews)

        prompt = f"""You are a Bull Analyst advocating for investing in the stock. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators. Leverage the provided research and data to address concerns and counter bearish arguments effectively.

        {instrument_context}

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the bull perspective holds stronger merit.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.

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
        Last bear argument: {prompt_current_response}
        Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position.

        Return a structured score-based argument. Every score dimension and every key claim must cite evidence IDs from the evidence ledger.
        If a claim cannot be grounded in the evidence ledger, do not include it.
        """

        response = structured_llm.invoke(prompt)
        validation = validate_argument_grounding(response, evidence_map)
        argument = f"Bull Analyst: {render_argument('bull', response)}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
            "bull_argument": response.model_dump(),
            "bull_validation": validation,
            "bear_argument": investment_debate_state.get("bear_argument"),
            "bear_validation": investment_debate_state.get("bear_validation"),
            "subagent_reviews": specialist_reviews,
            "evidence_ledger": ledger,
            "evidence_prompt": evidence_prompt,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node


def market_report_from_signal(analyst, market):
    hist = analyst["signals"].get("macd_hist")
    rsi = analyst["signals"].get("rsi")
    return (
        f"trend={analyst['trend']} confidence={analyst['confidence']} "
        f"close={market['close']} rsi={rsi} macd={analyst['signals'].get('macd')} "
        f"macd_hist={hist} macd_hist_positive={hist is not None and hist > 0} "
        f"rsi_extreme={rsi is not None and rsi > 70} "
        f"boll={market.get('boll')} boll_ub={market.get('boll_ub')} boll_lb={market.get('boll_lb')} "
        f"atr={market.get('atr')} mfi={market.get('mfi')} vwma={market.get('vwma')}"
    )


def run_bull_researcher(state, analyst, market):
    full_state = {
        "investment_debate_state": state,
        "market_report": market_report_from_signal(analyst, market),
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "evidence_ledger": state.get("evidence_ledger", []),
        "evidence_prompt": state.get("evidence_prompt", ""),
    }
    return create_bull_researcher(get_llm("quick", "bull_researcher"))(full_state)["investment_debate_state"]
