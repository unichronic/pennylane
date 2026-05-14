import os


def load_env(path=".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


load_env()


def _clean_key(key):
    key = (key or "").strip()
    if not key or "replace" in key.lower():
        return ""
    return key


def get_mistral_api_keys():
    raw = []
    if os.getenv("MISTRAL_API_KEYS"):
        raw.extend(os.getenv("MISTRAL_API_KEYS", "").split(","))
    raw.append(os.getenv("MISTRAL_API_KEY"))
    for idx in range(1, 17):
        raw.append(os.getenv(f"MISTRAL_API_KEY{idx}"))

    keys = []
    for key in raw:
        key = _clean_key(key)
        if key and key not in keys:
            keys.append(key)
    return keys


def select_mistral_api_key(kind="quick", name=""):
    keys = get_mistral_api_keys()
    if not keys:
        return "", None
    if len(keys) == 1:
        return keys[0], 1

    label = f"{kind}:{name}".lower()
    role_order = [
        "bull_researcher",
        "bear_researcher",
        "trader",
        "research_manager",
        "aggressive_debator",
        "conservative_debator",
        "neutral_debator",
        "portfolio_manager",
        "market analyst",
        "news analyst",
        "social media analyst",
        "fundamentals analyst",
    ]
    matched = next((slot for slot, role in enumerate(role_order) if role in label), None)
    if matched is not None:
        idx = matched % len(keys)
    elif kind == "deep" or "manager" in label or "portfolio" in label:
        idx = 1
    else:
        idx = sum(ord(ch) for ch in label) % len(keys)
    idx = idx % len(keys)
    return keys[idx], idx + 1


def _int_env(name, default):
    return int(os.getenv(name, str(default)) or str(default))


def llm_max_tokens_for(kind="quick", name=""):
    fallback = _int_env("TRADEAGE_LLM_MAX_TOKENS", 2500)
    label = f"{kind}:{name}".lower()
    role_defaults = {
        "trader": _int_env("TRADEAGE_TRADER_LLM_MAX_TOKENS", 1800),
        "research_manager": _int_env("TRADEAGE_RESEARCH_MANAGER_LLM_MAX_TOKENS", 4500),
        "portfolio_manager": _int_env("TRADEAGE_PORTFOLIO_MANAGER_LLM_MAX_TOKENS", 5000),
        "market analyst": _int_env("TRADEAGE_ANALYST_LLM_MAX_TOKENS", 2800),
        "news analyst": _int_env("TRADEAGE_ANALYST_LLM_MAX_TOKENS", 2800),
        "social media analyst": _int_env("TRADEAGE_ANALYST_LLM_MAX_TOKENS", 2800),
        "fundamentals analyst": _int_env("TRADEAGE_ANALYST_LLM_MAX_TOKENS", 3200),
        "bull_researcher": _int_env("TRADEAGE_DEBATE_LLM_MAX_TOKENS", 2800),
        "bear_researcher": _int_env("TRADEAGE_DEBATE_LLM_MAX_TOKENS", 2800),
        "aggressive_debator": _int_env("TRADEAGE_RISK_DEBATE_LLM_MAX_TOKENS", 2800),
        "conservative_debator": _int_env("TRADEAGE_RISK_DEBATE_LLM_MAX_TOKENS", 2800),
        "neutral_debator": _int_env("TRADEAGE_RISK_DEBATE_LLM_MAX_TOKENS", 2800),
    }
    for role, value in role_defaults.items():
        if role in label:
            return value
    if kind == "deep":
        return _int_env("TRADEAGE_DEEP_LLM_MAX_TOKENS", 4500)
    if kind == "quick":
        return _int_env("TRADEAGE_QUICK_LLM_MAX_TOKENS", fallback)
    return fallback


def get_config():
    return {
        "llm_provider": os.getenv("LLM_PROVIDER", "mistral"),
        "quick_think_llm": os.getenv(
            "MISTRAL_QUICK_MODEL",
            os.getenv("MISTRAL_MODEL", "mistral-small-2603"),
        ),
        "deep_think_llm": os.getenv(
            "MISTRAL_DEEP_MODEL",
            os.getenv("MISTRAL_MODEL", "mistral-large-2512"),
        ),
        "backend_url": os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1"),
        "mistral_key_count": len(get_mistral_api_keys()),
        "max_debate_rounds": _int_env("MAX_DEBATE_ROUNDS", 1),
        "output_language": os.getenv("OUTPUT_LANGUAGE", "English"),
        "data_cache_dir": os.getenv("TRADEAGE_DATA_CACHE_DIR", "data_cache"),
        "results_dir": os.getenv("TRADEAGE_RESULTS_DIR", "results"),
        "workflow_db_path": os.getenv("TRADEAGE_WORKFLOW_DB", "data_cache/workflows.db"),
        "decision_memory_path": os.getenv("TRADEAGE_MEMORY_PATH", "data_cache/decision_memory.jsonl"),
        "memory_log_max_entries": _int_env("TRADEAGE_MEMORY_MAX_ENTRIES", 0) or None,
        "memory_vector_enabled": os.getenv("TRADEAGE_MEMORY_VECTOR_ENABLED", "1") not in {"0", "false", "False"},
        "memory_vector_path": os.getenv("TRADEAGE_MEMORY_VECTOR_PATH", ""),
        "memory_embedding_provider": os.getenv("TRADEAGE_MEMORY_EMBEDDING_PROVIDER", "local"),
        "memory_embedding_model": os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed"),
        "memory_embedding_dims": _int_env("TRADEAGE_MEMORY_EMBEDDING_DIMS", 256),
        "llm_trace_enabled": os.getenv("TRADEAGE_LLM_TRACE_ENABLED", "1") not in {"0", "false", "False"},
        "llm_trace_path": os.getenv("TRADEAGE_LLM_TRACE_PATH", "data_cache/llm_calls.jsonl"),
        "llm_call_timeout_seconds": _int_env("TRADEAGE_LLM_CALL_TIMEOUT_SECONDS", 120),
        "llm_max_retries": _int_env("TRADEAGE_LLM_MAX_RETRIES", 0),
        "llm_rate_limit_retries": _int_env("TRADEAGE_LLM_RATE_LIMIT_RETRIES", 2),
        "llm_rate_limit_backoff_seconds": float(os.getenv("TRADEAGE_LLM_RATE_LIMIT_BACKOFF_SECONDS", "30") or "30"),
        "llm_max_tokens": _int_env("TRADEAGE_LLM_MAX_TOKENS", 2500),
        "quick_llm_max_tokens": _int_env("TRADEAGE_QUICK_LLM_MAX_TOKENS", 2500),
        "deep_llm_max_tokens": _int_env("TRADEAGE_DEEP_LLM_MAX_TOKENS", 4500),
        "analyst_llm_max_tokens": _int_env("TRADEAGE_ANALYST_LLM_MAX_TOKENS", 2800),
        "debate_llm_max_tokens": _int_env("TRADEAGE_DEBATE_LLM_MAX_TOKENS", 2800),
        "risk_debate_llm_max_tokens": _int_env("TRADEAGE_RISK_DEBATE_LLM_MAX_TOKENS", 2800),
        "trader_llm_max_tokens": _int_env("TRADEAGE_TRADER_LLM_MAX_TOKENS", 1800),
        "research_manager_llm_max_tokens": _int_env("TRADEAGE_RESEARCH_MANAGER_LLM_MAX_TOKENS", 4500),
        "portfolio_manager_llm_max_tokens": _int_env("TRADEAGE_PORTFOLIO_MANAGER_LLM_MAX_TOKENS", 5000),
        "parallel_analysts": os.getenv("TRADEAGE_PARALLEL_ANALYSTS", "1") not in {"0", "false", "False"},
        "parallel_analyst_max_workers": _int_env("TRADEAGE_PARALLEL_ANALYST_MAX_WORKERS", 0),
        "report_context_max_chars": _int_env("TRADEAGE_REPORT_CONTEXT_MAX_CHARS", 1800),
        "debate_subagents_enabled": os.getenv("TRADEAGE_DEBATE_SUBAGENTS_ENABLED", "1") not in {"0", "false", "False"},
        "investment_debate_prompt_max_chars": _int_env("TRADEAGE_INVESTMENT_DEBATE_PROMPT_MAX_CHARS", 12000),
        "risk_debate_prompt_max_chars": _int_env("TRADEAGE_RISK_DEBATE_PROMPT_MAX_CHARS", 14000),
        "single_argument_prompt_max_chars": _int_env("TRADEAGE_SINGLE_ARGUMENT_PROMPT_MAX_CHARS", 6000),
        "plan_prompt_max_chars": _int_env("TRADEAGE_PLAN_PROMPT_MAX_CHARS", 6000),
        "checkpoint_enabled": os.getenv("TRADEAGE_CHECKPOINT_ENABLED", "1") not in {"0", "false", "False"},
        "analyst_tool_mode": os.getenv("TRADEAGE_ANALYST_TOOL_MODE", "model"),
        "market_data_provider": os.getenv("TRADEAGE_DATA_PROVIDER", "auto"),
        "market_data_providers": os.getenv(
            "TRADEAGE_DATA_PROVIDERS",
            "yfinance,twelvedata,alpha_vantage",
        ),
        "tool_vendors": {
            "get_news": os.getenv("TRADEAGE_NEWS_VENDOR", "yfinance"),
            "get_global_news": os.getenv("TRADEAGE_NEWS_VENDOR", "yfinance"),
            "get_insider_transactions": os.getenv("TRADEAGE_INSIDER_VENDOR", "yfinance"),
            "get_fundamentals": os.getenv("TRADEAGE_FUNDAMENTALS_VENDOR", "yfinance"),
            "get_balance_sheet": os.getenv("TRADEAGE_FUNDAMENTALS_VENDOR", "yfinance"),
            "get_cashflow": os.getenv("TRADEAGE_FUNDAMENTALS_VENDOR", "yfinance"),
            "get_income_statement": os.getenv("TRADEAGE_FUNDAMENTALS_VENDOR", "yfinance"),
        },
    }
