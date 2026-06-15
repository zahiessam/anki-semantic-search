"""Bounded local runtime helpers for Smart Retrieval.

This module contains only deterministic orchestration helpers. It does not call
models, web tools, or external medical sources.
"""

import html
import re


ALLOWED_LOCAL_TOOLS = {
    "metadata_filter",
    "bm25_search",
    "hybrid_embedding_search",
    "rerank",
    "mmr_context_selection",
    "evidence_evaluation",
    "session_fact_cache",
}
MAX_MODEL_PLANNER_CALLS = 1
MAX_RETRIEVAL_PASSES = 3
MAX_SUBQUERIES = 6
MAX_ANSWER_CONTEXT_NOTES = 24
SESSION_EXCERPT_CHARS = 150


def validate_required_local_tools(plan):
    plan = dict(plan or {})
    raw_tools = plan.get("required_local_tools") or []
    if not isinstance(raw_tools, list):
        raw_tools = []
    tools = []
    errors = list(plan.get("plan_validation_errors") or [])
    for item in raw_tools:
        tool = str(item or "").strip()
        if not tool:
            continue
        if tool not in ALLOWED_LOCAL_TOOLS:
            errors.append(f"invalid_local_tool:{tool}")
            continue
        if tool not in tools:
            tools.append(tool)
    plan["required_local_tools"] = tools
    plan["plan_validation_errors"] = list(dict.fromkeys(errors))
    if any(str(error).startswith("invalid_local_tool:") for error in errors):
        plan["planner_source"] = "fallback"
        plan["fallback_reason"] = plan.get("fallback_reason") or "invalid_local_tool"
        plan["retrieval_plan"] = "fallback_current_hybrid"
    return plan


def apply_runtime_budgets(plan, context_limit=None, search_config=None):
    plan = dict(plan or {})
    sc = search_config or {}
    try:
        max_subqueries = max(1, min(12, int(sc.get("agentic_max_subqueries", MAX_SUBQUERIES) or MAX_SUBQUERIES)))
    except Exception:
        max_subqueries = MAX_SUBQUERIES
    try:
        max_passes = max(1, min(5, int(sc.get("agentic_max_retrieval_passes", MAX_RETRIEVAL_PASSES) or MAX_RETRIEVAL_PASSES)))
    except Exception:
        max_passes = MAX_RETRIEVAL_PASSES
    subqueries = list(plan.get("subqueries") or [])[:max_subqueries]
    plan["subqueries"] = subqueries
    subquery_plans = list(plan.get("subquery_plans") or [])
    if subquery_plans:
        plan["subquery_plans"] = subquery_plans[:max_subqueries]
    effective_context_limit = MAX_ANSWER_CONTEXT_NOTES
    if context_limit is not None:
        try:
            effective_context_limit = min(effective_context_limit, max(1, int(context_limit)))
        except Exception:
            effective_context_limit = MAX_ANSWER_CONTEXT_NOTES
    plan["runtime_budgets"] = {
        "max_model_planner_calls": MAX_MODEL_PLANNER_CALLS,
        "max_retrieval_passes": max_passes,
        "max_subqueries": max_subqueries,
        "max_answer_context_notes": effective_context_limit,
        "recursive_tool_loop": False,
    }
    plan["budget_reason"] = plan.get("budget_reason") or "local_bounded_v1"
    return plan


def bounded_context_limit(existing_limit):
    try:
        existing_limit = int(existing_limit)
    except Exception:
        existing_limit = MAX_ANSWER_CONTEXT_NOTES
    return max(1, min(MAX_ANSWER_CONTEXT_NOTES, existing_limit))


def normalize_note_excerpt(text, max_chars=SESSION_EXCERPT_CHARS):
    text = html.unescape(str(text or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{\{c\d+::(.*?)(?:::[^}]*)?\}\}", r"\1", text)
    text = re.sub(r"\[sound:[^\]]+\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars].strip()


def note_primary_text(note):
    if not isinstance(note, dict):
        return ""
    for key in ("display_content", "content_preview", "content"):
        value = note.get(key)
        if value:
            return value
    return ""


def extract_session_fact_cache(context_notes, subquery_label="main", max_chars=SESSION_EXCERPT_CHARS):
    cache = []
    seen = set()
    for note in context_notes or []:
        if not isinstance(note, dict):
            continue
        note_id = note.get("id")
        if note_id is None or note_id in seen:
            continue
        seen.add(note_id)
        excerpt = normalize_note_excerpt(note_primary_text(note), max_chars=max_chars)
        if not excerpt:
            continue
        cache.append({
            "note_id": note_id,
            "subquery": subquery_label or "main",
            "excerpt": excerpt,
        })
    return cache
