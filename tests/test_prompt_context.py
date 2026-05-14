from core.prompt_context import compact_text


def test_compact_text_keeps_short_text_unchanged():
    assert compact_text("short evidence", 100, "evidence") == "short evidence"


def test_compact_text_preserves_opening_and_latest_context():
    text = "OPENING " + ("middle " * 200) + "LATEST"

    compacted = compact_text(text, 240, "debate history")

    assert len(compacted) <= 260
    assert compacted.startswith("OPENING")
    assert compacted.endswith("LATEST")
    assert "debate history compacted" in compacted
