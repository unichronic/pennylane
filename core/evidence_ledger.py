from __future__ import annotations

from typing import Any

from core.report_context import REPORT_KEYS


def _compact(text: Any, limit: int = 220) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


def _report_line_items(report_name: str, report_text: str) -> list[dict[str, Any]]:
    items = []
    for idx, line in enumerate(str(report_text or "").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        items.append(
            {
                "evidence_id": f"{report_name}:report:{idx}",
                "report": report_name,
                "tool": None,
                "snippet": _compact(stripped),
            }
        )
        if len(items) >= 3:
            break
    return items


def build_evidence_ledger(session_state: dict[str, Any]) -> list[dict[str, Any]]:
    ledger = []
    report_react_traces = session_state.get("report_react_traces", {})
    report_texts = {
        "market": session_state.get("market_report", ""),
        "news": session_state.get("news_report", ""),
        "sentiment": session_state.get("sentiment_report", ""),
        "fundamentals": session_state.get("fundamentals_report", ""),
    }

    for report_name in REPORT_KEYS:
        react_steps = report_react_traces.get(report_name) or []
        if react_steps:
            for idx, step in enumerate(react_steps, start=1):
                ledger.append(
                    {
                        "evidence_id": f"{report_name}:{idx}",
                        "report": report_name,
                        "tool": step.get("action"),
                        "snippet": _compact(step.get("observation")),
                    }
                )
        else:
            ledger.extend(_report_line_items(report_name, report_texts.get(report_name, "")))

    session_state["evidence_ledger"] = ledger
    session_state["evidence_prompt"] = render_evidence_prompt(ledger)
    return ledger


def render_evidence_prompt(ledger: list[dict[str, Any]]) -> str:
    if not ledger:
        return "No evidence ledger items are available."
    lines = []
    for item in ledger:
        tool = item.get("tool") or "report_excerpt"
        lines.append(
            f"[{item['evidence_id']}] report={item['report']} tool={tool} snippet={item['snippet']}"
        )
    return "\n".join(lines)


def evidence_index(ledger: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["evidence_id"]: item for item in ledger if item.get("evidence_id")}
