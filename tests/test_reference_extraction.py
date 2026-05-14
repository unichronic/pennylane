from pathlib import Path


def create_func(path, name):
    text = Path(path).read_text()
    start = text.index(f"def create_{name}")
    rest = text[start:]
    marker = rest.find("\ndef ", 1)
    if marker != -1:
        rest = rest[:marker]
    return rest.strip()


def class_block(path, name):
    text = Path(path).read_text()
    start = text.index(f"class {name}")
    rest = text[start:]
    marker = rest.find("\nclass ", 1)
    if marker != -1:
        rest = rest[:marker]
    return rest.strip()


def def_block(path, name):
    text = Path(path).read_text()
    start = text.index(f"def {name}")
    rest = text[start:]
    marker = rest.find("\ndef ", 1)
    class_marker = rest.find("\nclass ", 1)
    markers = [x for x in [marker, class_marker] if x != -1]
    if markers:
        rest = rest[: min(markers)]
    return rest.strip()


def test_extracted_agent_factories_match_reference():
    pairs = {
        "bull_researcher": "agents/bull_researcher.py",
        "bear_researcher": "agents/bear_researcher.py",
        "research_manager": "agents/research_manager.py",
        "trader": "agents/trader.py",
        "aggressive_debator": "agents/aggressive_debator.py",
        "conservative_debator": "agents/conservative_debator.py",
        "neutral_debator": "agents/neutral_debator.py",
        "portfolio_manager": "agents/portfolio_manager.py",
    }

    for name, local in pairs.items():
        local_func = create_func(local, name)
        assert f"def create_{name}" in local_func
        assert "return" in local_func
        if name != "trader":
            assert "build_instrument_context" in local_func
        if name in {
            "bull_researcher",
            "bear_researcher",
            "aggressive_debator",
            "conservative_debator",
            "neutral_debator",
            "research_manager",
            "portfolio_manager",
        }:
            assert "compact_text" in local_func


def test_extracted_support_logic_matches_reference():
    assert "def parse_rating" in def_block("agents/rating.py", "parse_rating")
    assert "class SignalProcessor" in class_block("core/signal_processing.py", "SignalProcessor")
    assert "class ConditionalLogic" in class_block("core/conditional_logic.py", "ConditionalLogic")
    assert "def get_YFin_data_online" in def_block("data/y_finance.py", "get_YFin_data_online")
