import json
from pathlib import Path

import pytest

from agents.analyst import run_analyst
from agents.bear_researcher import run_bear_researcher
from agents.bull_researcher import run_bull_researcher
from agents.debate import run_debate, run_debate_round
from agents.agno_runtime import AgnoLLMAdapter
from agents.llm_factory import get_llm
from agents.risk_manager import run_risk_manager
from agents.risk_debate import run_risk_debate
from agents.trader import run_trader
from backtest.engine import run_backtest
from config import get_config, get_mistral_api_keys, select_mistral_api_key
from core.pipeline import run_pipeline
from core.state import make_investment_debate_state
from data.indicators import add_indicators
from data.loader import load_ohlcv
from data.y_finance import get_YFin_data_online

REAL_SYMBOL = "AAPL"
REAL_START = "2024-03-01"
REAL_END = "2024-04-15"


@pytest.fixture(autouse=True)
def force_local_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")


def test_indicator_and_analyst_schema():
    rows = load_ohlcv(REAL_SYMBOL, REAL_START, REAL_END)
    with_indicators = add_indicators(rows)
    signal = run_analyst(with_indicators)

    assert len(with_indicators) == len(rows)
    assert {"rsi", "macd", "macd_signal", "macd_hist"}.issubset(with_indicators[-1])
    assert {
        "close_20_sma",
        "close_50_sma",
        "close_10_ema",
        "boll",
        "boll_ub",
        "boll_lb",
        "atr",
        "stoch_k",
        "stoch_d",
        "mfi",
        "obv",
        "vwma",
    }.issubset(with_indicators[-1])
    assert signal["trend"] in {"bullish", "bearish", "neutral"}
    assert 0 <= signal["confidence"] <= 1
    assert set(signal["signals"]) >= {"rsi", "macd"}
    assert isinstance(signal["summary"], str) and signal["summary"]


def test_mistral_env_defaults():
    cfg = get_config()

    assert cfg["quick_think_llm"] == "mistral-small-2603"
    assert cfg["deep_think_llm"] == "mistral-large-2512"
    assert cfg["backend_url"] == "https://api.mistral.ai/v1"
    assert cfg["mistral_key_count"] >= 2


def test_mistral_keys_are_distributed_by_agent_role(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEYS", "k1,k2")
    for idx in range(1, 5):
        monkeypatch.delenv(f"MISTRAL_API_KEY{idx}", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    keys = get_mistral_api_keys()
    quick_key, quick_slot = select_mistral_api_key("quick", "bull_researcher")
    bear_key, bear_slot = select_mistral_api_key("quick", "bear_researcher")
    deep_key, deep_slot = select_mistral_api_key("deep", "research_manager")

    assert len(keys) >= 2
    assert quick_key in keys
    assert bear_key in keys
    assert deep_key in keys
    assert quick_slot == 1
    assert bear_slot == 2
    assert deep_slot == 2


def test_mistral_keys_can_distribute_across_more_than_two_orgs(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEYS", "k1,k2,k3,k4")
    for idx in range(1, 5):
        monkeypatch.delenv(f"MISTRAL_API_KEY{idx}", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    slots = {
        select_mistral_api_key("quick", "bull_researcher")[1],
        select_mistral_api_key("quick", "bear_researcher")[1],
        select_mistral_api_key("quick", "trader")[1],
        select_mistral_api_key("deep", "research_manager")[1],
    }

    assert slots == {1, 2, 3, 4}


def test_agentic_runtime_uses_agno():
    llm = get_llm()

    assert isinstance(llm, AgnoLLMAdapter)
    assert llm.agent.__class__.__module__.startswith("agno.")
    assert llm.mistral_key_slot in {None, 1, 2}


def test_debate_trader_and_risk_schemas():
    rows = add_indicators(load_ohlcv(REAL_SYMBOL, REAL_START, REAL_END))
    analyst = run_analyst(rows)
    debate = run_debate(analyst, rows[-1])
    trader = run_trader(analyst, debate)
    risk = run_risk_manager(trader, rows[-1], cash=10000, shares=0)
    risk_debate = run_risk_debate(analyst, trader, rows[-1])

    assert set(debate) == {"bull_case", "bear_case", "key_risks", "consensus_bias"}
    assert isinstance(debate["key_risks"], list) and len(debate["key_risks"]) >= 2
    assert debate["consensus_bias"] in {"bullish", "bearish", "uncertain"}
    assert trader["action"] in {"buy", "sell", "hold"}
    assert 0 <= trader["confidence"] <= 1
    assert 0 <= trader["position_size"] <= 1
    assert isinstance(trader["reasoning"], str) and trader["reasoning"]
    assert set(risk) == {"approved", "adjusted_position", "stop_loss", "risk_notes"}
    assert isinstance(risk["approved"], bool)
    assert 0 <= risk["adjusted_position"] <= 1
    assert risk_debate["count"] == 3


def test_run_trader_uses_actual_company_in_prompt(monkeypatch):
    from agents.schemas import TraderProposal

    seen = {}

    class FakeStructuredLLM:
        def invoke(self, prompt):
            seen["prompt"] = prompt
            return TraderProposal(
                action="Hold",
                confidence=0.5,
                reasoning="Holding",
                position_size=0,
                entry_price=None,
                stop_loss=None,
            )

    class FakeLLM:
        def with_structured_output(self, schema):
            return FakeStructuredLLM()

    monkeypatch.setattr("agents.trader.get_llm", lambda *args, **kwargs: FakeLLM())

    analyst = {"confidence": 0.5}
    debate = {"consensus_bias": "uncertain"}
    run_trader(analyst, debate, "**Recommendation**: Hold", company="AAPL")

    user_content = seen["prompt"][1]["content"]
    assert "tailored for AAPL" in user_content
    assert "SAMPLE" not in user_content


def test_trader_context_renderer_avoids_dict_repr():
    from agents.trader import render_trader_context

    rendered = render_trader_context(
        {
            "action": "buy",
            "confidence": 0.8,
            "position_size": 0.3,
            "reasoning": "Evidence-backed plan",
        }
    )

    assert "Action: buy" in rendered
    assert "Evidence-backed plan" in rendered
    assert "{'action'" not in rendered


def test_debate_is_two_structured_researchers():
    rows = add_indicators(load_ohlcv(REAL_SYMBOL, REAL_START, REAL_END))
    analyst = run_analyst(rows)
    state = make_investment_debate_state()

    bull_state = run_bull_researcher(state, analyst, rows[-1])
    bear_state = run_bear_researcher(bull_state, analyst, rows[-1])
    debate_result = run_debate_round(analyst, rows[-1], state)

    assert bull_state["count"] == 1
    assert bear_state["count"] == 2
    assert "Bull Analyst:" in bear_state["bull_history"]
    assert "Bear Analyst:" in bear_state["bear_history"]
    assert debate_result["investment_debate_state"]["count"] == 2
    assert debate_result["investment_debate_state"]["history"].count("Analyst:") == 2
    assert debate_result["debate"]["bull_case"].startswith("Bull Analyst:")
    assert debate_result["debate"]["bear_case"].startswith("Bear Analyst:")
    assert "**Recommendation**:" in debate_result["investment_plan"]


def test_debate_rounds_are_configurable(monkeypatch):
    monkeypatch.setenv("MAX_DEBATE_ROUNDS", "2")
    rows = add_indicators(load_ohlcv(REAL_SYMBOL, REAL_START, REAL_END))
    analyst = run_analyst(rows)
    debate_result = run_debate_round(analyst, rows[-1])
    trader = run_trader(analyst, debate_result["debate"], debate_result["investment_plan"])
    risk_debate = run_risk_debate(analyst, trader, rows[-1])

    assert debate_result["investment_debate_state"]["count"] == 4
    assert risk_debate["count"] == 6


def test_risk_manager_rejects_unsafe_trades():
    decision = {
        "action": "buy",
        "confidence": 0.92,
        "reasoning": "oversized test decision",
        "position_size": 0.95,
    }
    market = {"close": 100, "rsi": 83, "macd": 2.2, "macd_signal": 1.5}

    risk = run_risk_manager(decision, market, cash=10000, shares=0)

    assert risk["approved"] is False
    assert risk["adjusted_position"] == 0
    assert risk["stop_loss"] is None


def test_pipeline_logs_execution_and_metrics(tmp_path):
    out = tmp_path / "run.jsonl"
    result = run_pipeline(REAL_SYMBOL, cash=10000, log_path=out, start_date=REAL_START, end_date=REAL_END)

    assert result["portfolio"]["equity"] > 0
    assert result["portfolio"].keys() >= {"realized_pnl", "unrealized_pnl", "stop_loss", "cost_basis"}
    assert result["metrics"].keys() >= {"cumulative_return", "sharpe_ratio", "max_drawdown"}
    assert result["trace"].keys() >= {"analyst", "debate", "trader", "risk", "execution"}
    assert result["trace"]["portfolio_manager"]["rating"] in {
        "Buy",
        "Overweight",
        "Hold",
        "Underweight",
        "Sell",
    }
    assert result["trace"]["final_trader"]["action"] in {"buy", "sell", "hold"}
    assert result["trace"]["investment_debate_state"]["count"] == 2
    assert result["trace"]["investment_debate_state"]["judge_decision"] in {
        "bullish",
        "bearish",
        "uncertain",
    }
    assert out.exists()

    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert [x["stage"] for x in lines] == [
        "market_analyst_report",
        "news_analyst_report",
        "sentiment_analyst_report",
        "fundamentals_analyst_report",
        "analyst",
        "debate",
        "trader",
        "risk_debate",
        "portfolio_manager",
        "risk",
        "execution",
        "evaluation",
    ]


def test_backtest_runs_multiple_steps(tmp_path):
    out = tmp_path / "backtest.jsonl"
    result = run_backtest(REAL_SYMBOL, cash=10000, log_path=out, start_date=REAL_START, end_date=REAL_END)

    assert result["metrics"]["cumulative_return_pct"] != 0
    assert result["metrics"]["max_drawdown_pct"] >= 0
    assert result["metrics"].keys() == {
        "cumulative_return_pct",
        "annualized_return_pct",
        "sharpe_ratio",
        "max_drawdown_pct",
    }
    assert len(result["trades"]) >= 1
    assert Path(out).exists()


def test_yfinance_loader_uses_real_market_rows():
    rows = load_ohlcv(REAL_SYMBOL, REAL_START, REAL_END)
    text = get_YFin_data_online(REAL_SYMBOL, REAL_START, REAL_END)

    assert len(rows) >= 20
    assert rows[0]["date"] >= REAL_START
    assert rows[-1]["date"] < REAL_END
    assert rows[-1]["volume"] > 0
    assert "# Stock data for AAPL" in text
    assert "Date,Open,High,Low,Close" in text
