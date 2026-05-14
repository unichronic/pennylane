import json
import os
import re
from hashlib import sha256
from pathlib import Path

from config import get_config, select_mistral_api_key


def memory_path():
    return Path(os.getenv("TRADEAGE_MEMORY_PATH", "data_cache/decision_memory.jsonl"))


def memory_vector_path():
    configured = get_config().get("memory_vector_path")
    if configured:
        return Path(configured)
    return memory_path().with_suffix(".vectors.jsonl")


def load_memory_entries():
    path = memory_path()
    if not path.exists():
        return []
    return [_ensure_memory_id(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def write_memory_entries(items):
    path = memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [_ensure_memory_id(dict(item)) for item in items]
    max_entries = get_config().get("memory_log_max_entries")
    if max_entries:
        pending = [item for item in items if not item.get("outcome")]
        resolved = [item for item in items if item.get("outcome")]
        resolved = resolved[-max_entries:]
        items = pending + resolved
    path.write_text("\n".join(json.dumps(item) for item in items) + ("\n" if items else ""))
    if get_config().get("memory_vector_enabled"):
        rebuild_memory_index(items)


def _ensure_memory_id(item):
    item.setdefault(
        "id",
        "|".join(
            [
                str(item.get("symbol", "")).upper(),
                str(item.get("date", "")),
                str(item.get("rating", "")),
                str(item.get("action", "")),
            ]
        ),
    )
    return item


def get_pending_entries(symbol=None):
    entries = [item for item in load_memory_entries() if not item.get("outcome")]
    if symbol:
        entries = [item for item in entries if item.get("symbol") == symbol.upper()]
    return entries


def _compact_text(value, limit=1200):
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _memory_tags(*, analyst=None, debate=None, rating="", action=""):
    tags = [str(rating).lower(), str(action).lower()]
    if analyst:
        tags.append(str(analyst.get("trend", "")).lower())
    if debate:
        tags.append(str(debate.get("consensus_bias", "")).lower())
    return sorted({tag for tag in tags if tag})


def load_lessons(symbol, limit=5, query_context=None):
    entries = [item for item in load_memory_entries() if item.get("outcome")]
    if not entries:
        return ""

    ticker = symbol.upper()
    same = [item for item in entries if item.get("symbol") == ticker]
    cross = [item for item in entries if item.get("symbol") != ticker]
    recent_same = list(reversed(same[-limit:]))
    recent_cross = list(reversed(cross[-3:]))

    latest_tags = set()
    if same:
        latest_tags.update(same[-1].get("tags", []))
    similar = [
        item for item in reversed(entries)
        if item.get("symbol") != ticker and latest_tags.intersection(item.get("tags", []))
    ][:3]
    failures = [
        item for item in reversed(entries)
        if item.get("alpha_return") is not None and item.get("alpha_return") < 0
    ][:3]
    successes = [
        item for item in reversed(entries)
        if item.get("alpha_return") is not None and item.get("alpha_return") > 0
    ][:3]
    semantic = semantic_search_lessons(ticker, query_context=query_context, limit=3)

    parts = []
    if recent_same:
        parts.append(f"Past analyses of {ticker} (most recent first):")
        parts.extend(_format_full_lesson(item) for item in recent_same)
    if semantic:
        parts.append("Semantically similar memory lessons:")
        parts.extend(_format_reflection_only(item) for item in semantic)
    if similar:
        parts.append("Similar tagged lessons across instruments:")
        parts.extend(_format_reflection_only(item) for item in similar)
    if recent_cross:
        parts.append("Recent cross-ticker lessons:")
        parts.extend(_format_reflection_only(item) for item in recent_cross)
    if successes:
        parts.append("Patterns that generated positive alpha:")
        parts.extend(_format_reflection_only(item) for item in successes)
    if failures:
        parts.append("Patterns that generated negative alpha:")
        parts.extend(_format_reflection_only(item) for item in failures)
    return "\n\n".join(parts)


def _format_full_lesson(item):
    decision = item.get("decision_text") or f"{item.get('rating')} / {item.get('action')}"
    return (
        f"{item['date']} {item['symbol']}: {item['outcome']}\n"
        f"Decision: {_compact_text(decision, 420)}\n"
        f"Reflection: {_compact_text(item.get('reflection') or item.get('outcome'), 420)}"
    )


def _format_reflection_only(item):
    return (
        f"{item['date']} {item['symbol']} {item.get('rating')} / {item.get('action')}: "
        f"{item.get('outcome')}. {_compact_text(item.get('reflection'), 360)}"
    )


def _memory_entry_text(item):
    reports = item.get("reports_summary", {}) or {}
    analyst = item.get("analyst", {}) or {}
    debate = item.get("debate", {}) or {}
    trader = item.get("trader", {}) or {}
    risk = item.get("risk", {}) or {}
    market = item.get("market", {}) or {}
    parts = [
        f"symbol {item.get('symbol')} date {item.get('date')}",
        f"rating {item.get('rating')} action {item.get('action')}",
        f"tags {' '.join(item.get('tags', []))}",
        f"decision {item.get('decision_text', '')}",
        f"outcome {item.get('outcome', '')}",
        f"reflection {item.get('reflection', '')}",
        f"analyst trend {analyst.get('trend', '')} summary {analyst.get('summary', '')}",
        f"debate bias {debate.get('consensus_bias', '')} risks {' '.join(debate.get('key_risks', []) or [])}",
        f"trader {trader.get('reasoning', '')}",
        f"risk {risk.get('risk_notes', '')}",
        f"market close {market.get('close', '')} rsi {market.get('rsi', '')} macd {market.get('macd', '')}",
    ]
    parts.extend(f"{key} report {value}" for key, value in reports.items())
    return _compact_text("\n".join(parts), 6000)


def _tokenize(text):
    return re.findall(r"[a-z0-9][a-z0-9_.$%-]*", str(text).lower())


def _local_embedding(text, dims=256):
    tokens = _tokenize(text)
    tokens.extend(f"{left}_{right}" for left, right in zip(tokens, tokens[1:]))
    if not tokens:
        return {}
    vector = {}
    for token in tokens:
        digest = sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dims
        sign = 1 if digest[4] % 2 == 0 else -1
        vector[str(idx)] = vector.get(str(idx), 0.0) + sign
    norm = sum(value * value for value in vector.values()) ** 0.5
    if not norm:
        return {}
    return {key: round(value / norm, 8) for key, value in vector.items()}


def _mistral_embedding(text):
    try:
        from mistralai import Mistral
    except ImportError as exc:
        raise RuntimeError("mistralai is required for TRADEAGE_MEMORY_EMBEDDING_PROVIDER=mistral") from exc

    key, _slot = select_mistral_api_key("quick", "memory_embeddings")
    if not key:
        raise RuntimeError(
            "Missing Mistral API key for memory embeddings. Set MISTRAL_API_KEY1/MISTRAL_API_KEY2 or use TRADEAGE_MEMORY_EMBEDDING_PROVIDER=local."
        )
    client = Mistral(api_key=key)
    response = client.embeddings.create(
        model=get_config().get("memory_embedding_model", "mistral-embed"),
        inputs=[text],
    )
    embedding = response.data[0].embedding
    norm = sum(float(value) * float(value) for value in embedding) ** 0.5
    if not norm:
        return {}
    return {str(idx): round(float(value) / norm, 8) for idx, value in enumerate(embedding)}


def embed_memory_text(text):
    provider = str(get_config().get("memory_embedding_provider", "local")).strip().lower()
    if provider in {"", "local", "hash", "deterministic"}:
        return _local_embedding(text, dims=get_config().get("memory_embedding_dims", 256))
    if provider == "mistral":
        return _mistral_embedding(text)
    raise RuntimeError(f"Unsupported TRADEAGE_MEMORY_EMBEDDING_PROVIDER={provider}")


def rebuild_memory_index(entries=None):
    entries = entries if entries is not None else load_memory_entries()
    vectors = []
    for item in entries:
        if not item.get("outcome"):
            continue
        item = _ensure_memory_id(dict(item))
        text = _memory_entry_text(item)
        vectors.append({
            "id": item["id"],
            "symbol": item.get("symbol"),
            "date": item.get("date"),
            "text_hash": sha256(text.encode("utf-8")).hexdigest(),
            "embedding_provider": get_config().get("memory_embedding_provider", "local"),
            "embedding_model": (
                get_config().get("memory_embedding_model")
                if str(get_config().get("memory_embedding_provider", "local")).lower() == "mistral"
                else f"local-hash-{get_config().get('memory_embedding_dims', 256)}"
            ),
            "vector": embed_memory_text(text),
        })
    path = memory_vector_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(item) for item in vectors) + ("\n" if vectors else ""))
    return vectors


def load_memory_vectors():
    path = memory_vector_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _dot_sparse(left, right):
    if len(left) > len(right):
        left, right = right, left
    return sum(float(value) * float(right.get(key, 0.0)) for key, value in left.items())


def semantic_search_lessons(symbol, query_context=None, limit=5):
    if not get_config().get("memory_vector_enabled"):
        return []
    entries = {item["id"]: item for item in load_memory_entries() if item.get("outcome")}
    if not entries:
        return []
    vectors = load_memory_vectors()
    if not vectors:
        vectors = rebuild_memory_index(entries.values())

    query_text = query_context or f"trading decision memory for {symbol.upper()}"
    query_vector = embed_memory_text(query_text)
    if not query_vector:
        return []

    scored = []
    for indexed in vectors:
        item = entries.get(indexed.get("id"))
        if not item:
            continue
        score = _dot_sparse(query_vector, indexed.get("vector", {}))
        if score > 0:
            enriched = dict(item)
            enriched["memory_similarity"] = round(score, 6)
            scored.append(enriched)
    scored.sort(key=lambda item: (item["memory_similarity"], item.get("date", "")), reverse=True)
    return scored[:limit]


def record_decision(
    symbol,
    date,
    rating,
    action,
    price,
    *,
    final_decision="",
    analyst=None,
    debate=None,
    trader=None,
    risk=None,
    reports=None,
    market=None,
):
    path = memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = load_memory_entries()
    for existing in entries:
        if (
            existing.get("symbol") == symbol.upper()
            and existing.get("date") == date
            and existing.get("rating") == rating
            and existing.get("action") == action
        ):
            return
    item = {
        "id": "|".join([symbol.upper(), str(date), str(rating), str(action)]),
        "symbol": symbol.upper(),
        "date": date,
        "rating": rating,
        "action": action,
        "price": price,
        "decision_text": _compact_text(final_decision, 2200),
        "analyst": analyst or {},
        "debate": debate or {},
        "trader": trader or {},
        "risk": risk or {},
        "reports_summary": {
            key: _compact_text(value, 700)
            for key, value in (reports or {}).items()
        },
        "market": market or {},
        "tags": _memory_tags(analyst=analyst, debate=debate, rating=rating, action=action),
        "outcome": "",
        "raw_return": None,
        "benchmark_symbol": "SPY",
        "benchmark_return": None,
        "alpha_return": None,
        "holding_days": None,
        "reflection": "",
    }
    entries.append(item)
    write_memory_entries(entries)


def _reflection_text(item, raw_return, alpha_return, benchmark_return):
    try:
        from agents.llm_factory import get_llm

        prompt = (
            "You are a trading analyst reviewing your own past decision now that the outcome is known.\n"
            "Write exactly 2-4 sentences of plain prose. Cover whether the directional call was correct, "
            "what part of the thesis held or failed, and one concrete lesson for next time.\n\n"
            f"Raw return: {raw_return:+.1%}\n"
            f"Benchmark return: {benchmark_return:+.1%}\n"
            f"Alpha vs SPY: {alpha_return:+.1%}\n"
            f"Tags: {', '.join(item.get('tags', []))}\n\n"
            f"Decision:\n{item.get('decision_text') or item.get('rating')}"
        )
        return get_llm("quick", "reflector").invoke(prompt).content
    except Exception:
        direction = "helped" if alpha_return >= 0 else "hurt"
        return (
            f"The decision produced {raw_return:+.1%} raw return and {alpha_return:+.1%} alpha "
            f"versus SPY, which {direction} relative performance. Future similar calls should check "
            "whether the thesis has enough market-relative edge before acting."
        )


def reflect_outcomes(symbol, rows, benchmark_rows=None, holding_days=5):
    entries = load_memory_entries()
    if not entries:
        return []
    by_date = {row["date"]: row for row in rows}
    dates = [row["date"] for row in rows]
    benchmark_by_date = {row["date"]: row for row in benchmark_rows or []}
    benchmark_dates = [row["date"] for row in benchmark_rows or []]
    updates = []

    for item in entries:
        if item.get("symbol") != symbol.upper() or item.get("outcome"):
            continue
        if item["date"] not in dates:
            continue
        idx = dates.index(item["date"])
        out_idx = idx + holding_days
        if out_idx >= len(dates):
            continue

        start = float(by_date[item["date"]]["close"])
        end = float(by_date[dates[out_idx]]["close"])
        action = str(item.get("action", "")).lower()
        raw_return = (end - start) / start if start else 0
        if action == "hold":
            raw_return = 0
        elif action == "sell":
            raw_return = -raw_return

        benchmark_return = 0
        if benchmark_by_date and item["date"] in benchmark_by_date:
            b_idx = benchmark_dates.index(item["date"])
            b_out_idx = min(b_idx + holding_days, len(benchmark_dates) - 1)
            b_start = float(benchmark_by_date[benchmark_dates[b_idx]]["close"])
            b_end = float(benchmark_by_date[benchmark_dates[b_out_idx]]["close"])
            benchmark_return = (b_end - b_start) / b_start if b_start else 0

        alpha = raw_return - benchmark_return
        item["raw_return"] = raw_return
        item["benchmark_return"] = benchmark_return
        item["alpha_return"] = alpha
        item["holding_days"] = holding_days
        item["outcome"] = f"{holding_days}d raw {raw_return:.2%}, SPY {benchmark_return:.2%}, alpha {alpha:.2%}"
        item["reflection"] = _reflection_text(item, raw_return, alpha, benchmark_return)
        updates.append(item)

    if updates:
        write_memory_entries(entries)
    return updates
