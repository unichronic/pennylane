from config import get_config


def compact_text(text, max_chars, label="context"):
    value = "" if text is None else str(text).strip()
    limit = int(max_chars or 0)
    if not value or limit <= 0 or len(value) <= limit:
        return value

    marker = (
        f"\n\n[{label} compacted from {len(value)} to about {limit} characters; "
        "full content is retained in workflow state and logs.]\n\n"
    )
    remaining = max(80, limit - len(marker))
    head_len = max(40, int(remaining * 0.35))
    tail_len = max(40, remaining - head_len)
    return f"{value[:head_len].rstrip()}{marker}{value[-tail_len:].lstrip()}"


def prompt_limits():
    cfg = get_config()
    return {
        "investment_history": cfg.get("investment_debate_prompt_max_chars", 12000),
        "risk_history": cfg.get("risk_debate_prompt_max_chars", 14000),
        "single_argument": cfg.get("single_argument_prompt_max_chars", 6000),
        "plan": cfg.get("plan_prompt_max_chars", 6000),
    }
