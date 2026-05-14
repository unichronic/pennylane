from agno.agent import Agent
import time

from agents.agent_utils import build_instrument_context
from agents.agno_runtime import create_agno_model
from agents.llm_factory import get_llm
from config import get_config
from core.llm_observability import LLMProviderError, raise_if_agno_error_content, traced_llm_call
from data.trading_tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_global_news,
    get_income_statement,
    get_indicators,
    get_insider_transactions,
    get_news,
    get_stock_data,
    sentiment_from_news,
)


MARKET_INDICATORS = [
    "close_50_sma",
    "close_200_sma",
    "close_10_ema",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi",
    "boll",
    "boll_ub",
    "boll_lb",
    "atr",
    "vwma",
]


def _stringify(value):
    return value if isinstance(value, str) else str(value)


def _tool_mode():
    cfg = get_config()
    if cfg["llm_provider"].lower() == "local":
        return "deterministic"
    return str(cfg.get("analyst_tool_mode", "model")).lower()


def _scoped_tool(name, func, description):
    def tool():
        return func()

    tool.__name__ = name
    tool.__doc__ = description
    return tool


def _run_react_plan(plan):
    observations = []
    calls = []
    react_steps = []
    for idx, item in enumerate(plan, start=1):
        tool = item["tool"]
        args = item["args"]
        result = item["call"]()
        result_text = _stringify(result)
        calls.append({"tool": tool, "args": args})
        react_steps.append({
            "thought": item["thought"],
            "action": tool,
            "action_input": args,
            "observation": result_text,
        })
        observations.append(
            f"Thought {idx}: {item['thought']}\n"
            f"Action {idx}: {tool}\n"
            f"Observation {idx}:\n{result_text}"
        )
    return "\n\n".join(observations), calls, react_steps


def _run_model_selected_tools(role, symbol, trade_date, plan, instruction):
    tools = [
        _scoped_tool(item["tool"], item["call"], item["thought"])
        for item in plan
    ]
    model = create_agno_model("quick", role, role=role)
    agent = Agent(
        name=role,
        role=role,
        model=model,
        tools=tools,
        tool_choice="auto",
        tool_call_limit=len(tools),
        instructions=[
            "Use the available tools before writing your report.",
            "Do not fabricate data or cite evidence you did not retrieve.",
            "Synthesize the evidence into a clear analyst report and end with a concise markdown summary table.",
        ],
        markdown=True,
        telemetry=False,
    )
    prompt = f"""Analyze {symbol.upper()} for trading date {trade_date}.

{build_instrument_context(symbol)}

Task:
{instruction}

The tools already have the correct ticker and date context wired in. Use the relevant tools, inspect the outputs carefully, and then write the report."""
    cfg = get_config()
    max_attempts = 1 + max(0, cfg.get("llm_rate_limit_retries", 0))
    for attempt in range(1, max_attempts + 1):
        try:
            with traced_llm_call(
                agent_name=role,
                provider=getattr(model, "provider", None) or cfg["llm_provider"],
                model=getattr(model, "id", None) or getattr(model, "name", None),
                kind="quick",
                key_slot=getattr(model, "tradeage_mistral_key_slot", None),
                prompt=prompt,
                call_site=f"analyst_reports.model_selected_tools attempt={attempt}",
            ) as trace:
                run = agent.run(prompt)
                content = getattr(run, "content", "")
                trace.set_response(content)
                raise_if_agno_error_content(content)
            break
        except LLMProviderError as exc:
            if not getattr(exc, "retryable", False) or attempt >= max_attempts:
                raise
            time.sleep(float(cfg.get("llm_rate_limit_backoff_seconds", 30)) * attempt)
    executions = list(getattr(run, "tools", None) or [])
    if not executions:
        raise RuntimeError(f"{role} did not call any analyst tools")

    calls = []
    react_steps = []
    observations = []
    for idx, execution in enumerate(executions, start=1):
        name = getattr(execution, "tool_name", None) or getattr(execution, "name", None) or "unknown_tool"
        args = getattr(execution, "tool_args", None) or {}
        result = getattr(execution, "result", "")
        calls.append({"tool": name, "args": args})
        thought = f"Model selected {name} for evidence gathering."
        react_steps.append({
            "thought": thought,
            "action": name,
            "action_input": args,
            "observation": _stringify(result),
        })
        observations.append(
            f"Thought {idx}: {thought}\n"
            f"Action {idx}: {name}\n"
            f"Observation {idx}:\n{_stringify(result)}"
        )
    return {
        "report": _stringify(getattr(run, "content", "")),
        "tool_calls": calls,
        "react_steps": react_steps,
        "observations": "\n\n".join(observations),
    }


def _run_analyst_plan(role, symbol, trade_date, plan, instruction):
    if _tool_mode() == "model":
        return _run_model_selected_tools(role, symbol, trade_date, plan, instruction)
    observations, calls, react_steps = _run_react_plan(plan)
    report = _synthesize(role, symbol, trade_date, observations, instruction)
    return {
        "report": report,
        "tool_calls": calls,
        "react_steps": react_steps,
        "observations": observations,
    }


def _synthesize(role, symbol, trade_date, tool_observations, instruction):
    prompt = f"""You are the {role} for Penny Lane Capital.

{build_instrument_context(symbol)}
Current trading date: {trade_date}

Use the observations below to write a clear, evidence-based analyst report tied directly to trading implications.
End with a concise markdown table that summarizes the key takeaways.

Tool observations:
{tool_observations}

Task:
{instruction}
"""
    return get_llm("quick", role).invoke(prompt).content


def run_market_analyst_report(symbol, start_date, end_date, provider=None, trade_date=None, rows=None):
    plan = [
        {
            "thought": "Retrieve historical price data for the analysis window.",
            "tool": "get_stock_data",
            "args": {"symbol": symbol, "start_date": start_date, "end_date": end_date, "provider": provider},
            "call": lambda: get_stock_data(symbol, start_date, end_date, provider=provider, rows=rows),
        },
        {
            "thought": "Compute the technical indicator snapshot for the same window.",
            "tool": "get_indicators",
            "args": {"symbol": symbol, "indicators": MARKET_INDICATORS},
            "call": lambda: "\n".join(
                f"- {key}: {value}"
                for key, value in get_indicators(
                    symbol,
                    start_date,
                    end_date,
                    indicators=MARKET_INDICATORS,
                    provider=provider,
                    rows=rows,
                ).items()
            ),
        },
    ]
    return _run_analyst_plan(
        "Market Analyst",
        symbol,
        trade_date or end_date,
        plan,
        "Assess price action, trend, momentum, volatility, and volume conditions.",
    )


def run_social_media_analyst_report(symbol, curr_date=None):
    cached = {}
    plan = [
        {
            "thought": "Collect recent company-specific news and public discussion signals.",
            "tool": "get_news",
            "args": {"symbol": symbol, "curr_date": curr_date},
            "call": lambda: cached.setdefault("news", get_news(symbol, curr_date=curr_date)),
        },
        {
            "thought": "Convert the collected discussion into a sentiment summary.",
            "tool": "sentiment_from_news",
            "args": {"news_report": "get_news output"},
            "call": lambda: sentiment_from_news(
                cached.setdefault("news", get_news(symbol, curr_date=curr_date))
            ),
        },
    ]
    return _run_analyst_plan(
        "Social Media Analyst",
        symbol,
        curr_date,
        plan,
        "Assess company-specific sentiment using recent public discussion and news.",
    )


def run_news_analyst_report(symbol, curr_date=None):
    plan = [
        {
            "thought": "Review company-specific news within the target date range.",
            "tool": "get_news",
            "args": {"symbol": symbol, "curr_date": curr_date},
            "call": lambda: get_news(symbol, curr_date=curr_date),
        },
        {
            "thought": "Review broader macroeconomic and market developments for the same period.",
            "tool": "get_global_news",
            "args": {"curr_date": curr_date},
            "call": lambda: get_global_news(curr_date=curr_date),
        },
        {
            "thought": "Inspect insider transaction activity for relevant signals.",
            "tool": "get_insider_transactions",
            "args": {"symbol": symbol, "curr_date": curr_date},
            "call": lambda: get_insider_transactions(symbol, curr_date=curr_date),
        },
    ]
    return _run_analyst_plan(
        "News Analyst",
        symbol,
        curr_date,
        plan,
        "Assess company news, macro news, and insider activity for trading relevance.",
    )


def run_fundamentals_analyst_report(symbol, curr_date=None):
    plan = [
        {
            "thought": "Retrieve the company fundamentals overview and key snapshot metrics.",
            "tool": "get_fundamentals",
            "args": {"symbol": symbol, "curr_date": curr_date},
            "call": lambda: get_fundamentals(symbol, curr_date=curr_date),
        },
        {
            "thought": "Review balance sheet data available by the trade date.",
            "tool": "get_balance_sheet",
            "args": {"symbol": symbol, "curr_date": curr_date},
            "call": lambda: get_balance_sheet(symbol, curr_date=curr_date),
        },
        {
            "thought": "Review cash flow data available by the trade date.",
            "tool": "get_cashflow",
            "args": {"symbol": symbol, "curr_date": curr_date},
            "call": lambda: get_cashflow(symbol, curr_date=curr_date),
        },
        {
            "thought": "Review income statement data available by the trade date.",
            "tool": "get_income_statement",
            "args": {"symbol": symbol, "curr_date": curr_date},
            "call": lambda: get_income_statement(symbol, curr_date=curr_date),
        },
    ]
    return _run_analyst_plan(
        "Fundamentals Analyst",
        symbol,
        curr_date,
        plan,
        "Assess the financial statements and overall quality, leverage, and growth profile of the company.",
    )


def _looks_like_date(value):
    return (
        isinstance(value, str)
        and len(value) == 10
        and value[4] == "-"
        and value[7] == "-"
        and value[:4].isdigit()
        and value[5:7].isdigit()
        and value[8:].isdigit()
    )


def run_sentiment_analyst_report(symbol, curr_date=None, news_report=None):
    if news_report is None and curr_date is not None and not _looks_like_date(curr_date):
        news_report = curr_date
        curr_date = None
    if news_report is None:
        return run_social_media_analyst_report(symbol, curr_date=curr_date)

    plan = [
        {
            "thought": "reuse the news report we already have",
            "tool": "get_news",
            "args": {"symbol": symbol, "curr_date": curr_date, "source": "provided_news_report"},
            "call": lambda: news_report,
        },
        {
            "thought": "turn that into a sentiment read",
            "tool": "sentiment_from_news",
            "args": {"news_report": "provided_news_report"},
            "call": lambda: sentiment_from_news(news_report),
        },
    ]
    return _run_analyst_plan(
        "Social Media Analyst",
        symbol,
        curr_date,
        plan,
        "Read company-specific sentiment from public discussion and news.",
    )
