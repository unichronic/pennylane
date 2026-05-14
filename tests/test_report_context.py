from core.report_context import build_compressed_reports, compress_report


def test_compress_report_preserves_evidence_and_marks_compression(monkeypatch):
    monkeypatch.setenv("TRADEAGE_REPORT_CONTEXT_MAX_CHARS", "700")
    report = "\n".join(
        [
            "# Fundamentals",
            "Revenue growth accelerated 12% while margin expanded.",
            "Debt risk remains manageable.",
        ]
        + [f"low value filler line {idx}" for idx in range(80)]
        + ["RSI 71 and MACD histogram positive support momentum."]
    )

    compressed = compress_report(report)

    assert len(compressed) <= 700
    assert "Revenue growth" in compressed
    assert "RSI 71" in compressed
    assert "Report compressed from" in compressed


def test_build_compressed_reports_keeps_full_reports_separate(monkeypatch):
    monkeypatch.setenv("TRADEAGE_REPORT_CONTEXT_MAX_CHARS", "500")
    full = "Market trend is positive with MACD support.\n" * 80
    session_state = {
        "market_report": full,
        "news_report": "News report",
        "sentiment_report": "Sentiment report",
        "fundamentals_report": "Fundamentals report",
    }

    compressed = build_compressed_reports(session_state)

    assert session_state["market_report"] == full
    assert compressed["market"] != full
    assert session_state["report_compression"]["market"]["full_chars"] == len(full)
    assert session_state["report_compression"]["market"]["compressed_chars"] <= 500
