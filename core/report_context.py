import re

from config import get_config


REPORT_KEYS = ("market", "news", "sentiment", "fundamentals")


def _normalize(text):
    return re.sub(r"\n{3,}", "\n\n", str(text or "").strip())


def _line_score(line):
    s = line.strip()
    lo = s.lower()
    score = 0
    if s.startswith(("#", "-", "*", "|")):
        score += 3
    if any(token in lo for token in ("buy", "sell", "hold", "risk", "growth", "margin", "revenue", "trend")):
        score += 2
    if any(token in lo for token in ("rsi", "macd", "volume", "earnings", "cash", "debt", "insider", "sentiment")):
        score += 2
    if any(ch.isdigit() for ch in s):
        score += 1
    return score


def compress_report(report, max_chars=None):
    """Shrink a report without losing the useful bits."""
    cfg = get_config()
    limit = max(400, int(max_chars or cfg["report_context_max_chars"]))
    text = _normalize(report)
    if len(text) <= limit:
        return text

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text[:limit].rstrip()

    budget = max(0, limit - 180)
    picked = []
    used_chars = 0
    for line in lines[:6]:
        line = line[: min(len(line), 500)]
        if used_chars + len(line) + 1 > budget:
            break
        picked.append(line)
        used_chars += len(line) + 1

    scored = sorted(
        enumerate(lines[6:], start=6),
        key=lambda item: (-_line_score(item[1]), item[0]),
    )
    seen = set(picked)
    for _, line in scored:
        if _line_score(line) <= 0:
            continue
        line = line[: min(len(line), 500)]
        if line in seen:
            continue
        if used_chars + len(line) + 1 > budget:
            continue
        picked.append(line)
        seen.add(line)
        used_chars += len(line) + 1
        if used_chars >= budget:
            break

    summary = "\n".join(picked).strip()
    if not summary:
        summary = text[:budget].rstrip()
    suffix = f"\n\n[Report compressed from {len(text)} to {len(summary)} characters; full report is stored in trace/logs/memory.]"
    return (summary + suffix)[:limit].rstrip()


def build_compressed_reports(session_state):
    full_reports = {
        "market": session_state.get("market_report", ""),
        "news": session_state.get("news_report", ""),
        "sentiment": session_state.get("sentiment_report", ""),
        "fundamentals": session_state.get("fundamentals_report", ""),
    }
    compressed = {key: compress_report(value) for key, value in full_reports.items()}
    stats = {
        key: {
            "full_chars": len(str(full_reports[key] or "")),
            "compressed_chars": len(str(compressed[key] or "")),
        }
        for key in REPORT_KEYS
    }
    session_state["compressed_reports"] = compressed
    session_state["report_compression"] = stats
    return compressed


def report_context(session_state):
    return session_state.get("compressed_reports") or build_compressed_reports(session_state)
