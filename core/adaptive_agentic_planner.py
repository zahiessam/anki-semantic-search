"""Adaptive Agentic RAG V2 planning helpers.

This module is intentionally side-effect-free: it builds prompts, parses model
responses, validates plans, and chooses deterministic retrieval routing. The UI
owns any actual model call.
"""

import json
import re
import socket
from concurrent.futures import TimeoutError as FuturesTimeoutError
from urllib.error import URLError

try:
    from requests.exceptions import Timeout as RequestsTimeoutError
except Exception:
    RequestsTimeoutError = ()


ALLOWED_RETRIEVAL_MODES = {"bm25_only", "hybrid"}
ALLOWED_PASS2_POLICIES = {"never", "if_weak", "if_coverage_gap"}
MAX_SUBQUERIES = 6
MAX_SUBQUERY_CHARS = 80
MAX_SUBQUERY_TOKENS = 15
MAX_PLANNER_QUERY_CHARS = 1200
AGENTIC_PLANNER_TIMEOUT_SECONDS_DEFAULT = 25

EXACT_FACT_RE = re.compile(
    r"\b(?:what is|what are|which|when|where|who|define|dose|dosage)\b",
    re.IGNORECASE,
)
SEMANTIC_RE = re.compile(
    r"\b(?:why|how|mechanism|pathophysiology|compare|vs|versus|summarize|overview|"
    r"review|features|factors|classification|complications|diagnosis|treatment)\b",
    re.IGNORECASE,
)
DRUG_OR_REGIMEN_RE = re.compile(
    r"\b(?:regimen|drug|antibiotic|treat(?:ment)?|first[-\s]?line|"
    r"[a-z]+(?:cillin|cycline|floxacin|mycin|azole|statin|pril|sartan|olol))\b",
    re.IGNORECASE,
)


def _tokens(text):
    return re.findall(r"\b[\w+-]+\b", text or "")


def _has_exact_quoted_term(text):
    """Detect deliberate exact-match quotes without treating apostrophes as quotes."""
    text = text or ""
    if re.search(r'"[^"\n]{2,120}"', text) or re.search(r"“[^”\n]{2,120}”", text):
        return True
    return bool(re.search(r"(?<!\w)'[^'\n]{2,120}'(?!\w)", text))


def _clean_string(value, max_chars=200):
    value = str(value or "").strip()
    value = re.sub(r"\s+", " ", value)
    return value[:max_chars].strip()


def _compact_planner_query(query):
    query = re.sub(r"\s+", " ", str(query or "")).strip()
    if len(query) <= MAX_PLANNER_QUERY_CHARS:
        return query, False
    head = query[:850].rstrip()
    tail = query[-250:].lstrip()
    return f"{head} ... {tail}", True


def planner_mode(search_config=None):
    value = ((search_config or {}).get("agentic_planner_mode") or "deterministic_v1").strip().lower()
    return "adaptive_v2" if value == "adaptive_v2" else "deterministic_v1"


def is_adaptive_v2_enabled(search_config=None):
    return planner_mode(search_config) == "adaptive_v2"


def should_call_model_planner(deterministic_plan, query, session_context=None, search_config=None):
    if not is_adaptive_v2_enabled(search_config):
        return False
    plan = deterministic_plan or {}
    if plan.get("fallback_reason"):
        return True
    if plan.get("follow_up"):
        return True
    if len(plan.get("subqueries") or []) > 1:
        return True
    if plan.get("intent") in {"compare", "mechanism", "broad_review", "diagnosis", "treatment"}:
        return True
    if SEMANTIC_RE.search(query or "") and float(plan.get("confidence") or 0.0) < 0.85:
        return True
    return False


def embedding_health_snapshot(config=None, notes=None):
    """Return a lightweight embedding health label for routing.

    This avoids a full stale-index scan during search startup. Provider/query
    failures later in the existing embedding worker still force keyword fallback.
    """
    config = config or {}
    sc = config.get("search_config") or {}
    engine = (sc.get("embedding_engine") or "").strip().lower()
    if engine == "unsupported_same_as_answer":
        return {"status": "missing_index", "reason": "unsupported_embedding_provider"}
    if engine in {"openai", "cohere", "voyage"}:
        key_map = {
            "openai": "openai_embedding_api_key",
            "cohere": "cohere_api_key",
            "voyage": "voyage_api_key",
        }
        if not (sc.get(key_map.get(engine, "voyage_api_key")) or "").strip():
            return {"status": "missing_index", "reason": "missing_embedding_api_key"}
    if engine in {"ollama", "local_openai"}:
        url_key = "ollama_base_url" if engine == "ollama" else "local_llm_url"
        if not (sc.get(url_key) or "").strip():
            return {"status": "missing_index", "reason": "missing_embedding_url"}
    partial = False
    if notes is not None:
        try:
            sample = list(notes or [])[:200]
            if sample and any(not note.get("embedding") for note in sample if isinstance(note, dict)):
                partial = True
        except Exception:
            partial = False
    return {"status": "partial" if partial else "healthy", "reason": "sample_missing_embeddings" if partial else ""}


def choose_retrieval_mode(query, deterministic_plan=None, embedding_health=None, search_config=None):
    query = query or ""
    plan = deterministic_plan or {}
    health = (embedding_health or {}).get("status") or "healthy"
    if health in {"stale_engine", "missing_index", "provider_failure", "query_embedding_failure"}:
        return "bm25_only", f"embedding_health_{health}"
    if _has_exact_quoted_term(query):
        return "bm25_only", "quoted_exact_terms"
    tokens = _tokens(query)
    acronym_count = sum(1 for token in tokens if len(token) >= 2 and token.isupper())
    if acronym_count and len(tokens) <= 6:
        return "bm25_only", "short_acronym_entity_query"
    if DRUG_OR_REGIMEN_RE.search(query) and len(tokens) <= 8:
        return "bm25_only", "exact_drug_or_regimen_query"
    if EXACT_FACT_RE.search(query) and len(tokens) <= 8 and plan.get("intent") in {"specific_fact", ""}:
        return "bm25_only", "simple_exact_fact"
    if plan.get("intent") in {"compare", "mechanism", "broad_review", "diagnosis", "treatment"}:
        return "hybrid", f"semantic_intent_{plan.get('intent')}"
    if SEMANTIC_RE.search(query):
        return "hybrid", "semantic_query_terms"
    return "bm25_only", "keyword_precision_default"


def _model_must_respect_bm25(query, deterministic_plan=None, embedding_health=None):
    mode, reason = choose_retrieval_mode(query, deterministic_plan, embedding_health)
    forced_reasons = {
        "quoted_exact_terms",
        "short_acronym_entity_query",
        "exact_drug_or_regimen_query",
        "simple_exact_fact",
    }
    if reason.startswith("embedding_health_"):
        return mode, reason
    if mode == "bm25_only" and reason in forced_reasons:
        return mode, reason
    return "", ""


def build_planner_prompt(query, deterministic_plan, session_context=None, embedding_health=None, memory_snapshot=None):
    planner_query, query_truncated = _compact_planner_query(query)
    schema = {
        "intent": "specific_fact|compare|mechanism|diagnosis|treatment|broad_review",
        "intent_variant": "",
        "confidence": 0.0,
        "retrieval_mode": "bm25_only|hybrid",
        "retrieval_mode_reason": "",
        "use_rerank": True,
        "pass2_policy": "never|if_weak|if_coverage_gap",
        "subqueries": [
            {"label": "primary", "query": "short retrieval query", "retrieval_mode": "bm25_only|hybrid"}
        ],
        "coverage_targets": [
            {"label": "concept_a", "required_terms": ["term"], "min_results": 1}
        ],
        "memory_used": False,
        "required_local_tools": ["bm25_search"],
        "budget_reason": "",
        "fallback_reason": "",
    }
    payload = {
        "query": planner_query,
        "query_truncated_for_planning": query_truncated,
        "original_query_chars": len(str(query or "")),
        "deterministic_pre_plan": deterministic_plan or {},
        "session_context": session_context or {},
        "memory_snapshot": memory_snapshot or {},
        "embedding_health": embedding_health or {},
        "schema": schema,
        "rules": [
            "Return JSON only.",
            "Use at most six subqueries.",
            "Each subquery must be <= 80 characters or <= 15 tokens.",
            "Choose bm25_only for exact terms and hybrid for semantic/multi-step concepts.",
            "Use only local tools from required_local_tools; do not request web, PubMed, browser, guideline, or external-source tools.",
            "Return a single JSON object with no markdown, comments, or explanatory text.",
            "Do not answer the medical question; only plan retrieval.",
        ],
    }
    system_prompt = (
        "You are a retrieval planner for a medical Anki note search system. "
        "Return only valid compact JSON matching the requested schema."
    )
    user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return system_prompt, user_prompt


def parse_model_plan_response(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def build_minimal_retry_planner_prompt(query, deterministic_plan=None, retrieval_mode_hint=""):
    planner_query, _query_truncated = _compact_planner_query(query)
    deterministic_plan = deterministic_plan or {}
    mode = retrieval_mode_hint or deterministic_plan.get("retrieval_mode") or "bm25_only"
    if mode not in ALLOWED_RETRIEVAL_MODES:
        mode = "bm25_only"
    payload = {
        "query": planner_query,
        "intent_hint": deterministic_plan.get("intent") or "specific_fact",
        "retrieval_mode_hint": mode,
        "format": {
            "intent": "specific_fact",
            "subqueries": [
                {"label": "main", "query": "short query", "retrieval_mode": mode}
            ],
        },
        "rules": [
            "Return JSON only.",
            "Use at most three subqueries.",
            "Each subquery must be <= 80 characters or <= 15 tokens.",
        ],
    }
    system_prompt = "Return only compact JSON for an Anki retrieval plan."
    user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return system_prompt, user_prompt


def planner_timeout_seconds(search_config=None, default=AGENTIC_PLANNER_TIMEOUT_SECONDS_DEFAULT):
    search_config = search_config or {}
    try:
        value = int(search_config.get("agentic_planner_timeout_seconds", default) or default)
    except Exception:
        value = default
    return max(3, min(60, value))


def planner_retry_timeout_seconds(search_config=None):
    return min(10, planner_timeout_seconds(search_config))


def is_timeout_exception(exc):
    if isinstance(exc, (TimeoutError, FuturesTimeoutError, socket.timeout)):
        return True
    if RequestsTimeoutError and isinstance(exc, RequestsTimeoutError):
        return True
    if isinstance(exc, URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (TimeoutError, FuturesTimeoutError, socket.timeout)):
            return True
    return "timeout" in str(exc).lower() or "timed out" in str(exc).lower()


def build_model_planner_with_retry(
    query,
    deterministic_plan,
    fallback_plan,
    session_context,
    embedding_health,
    memory_snapshot,
    search_config,
    call_planner,
):
    """Run adaptive model planning once, then a compact retry if needed."""
    deterministic_plan = deterministic_plan or {}
    fallback_plan = fallback_plan or {}
    errors = []
    model_name = ""

    try:
        system_prompt, user_prompt = build_planner_prompt(
            query,
            deterministic_plan,
            session_context,
            embedding_health,
            memory_snapshot=memory_snapshot,
        )
        raw_text, model_name = call_planner(system_prompt, user_prompt)
        raw_plan = parse_model_plan_response(raw_text)
        model_plan, errors = validate_model_plan(raw_plan, deterministic_plan, query, embedding_health)
        if model_plan is not None:
            return {
                "plan": model_plan,
                "errors": errors,
                "model_name": model_name,
                "llm_planner_error": "",
            }
        first_error = "model_plan_parse_failed" if raw_plan is None else "model_plan_invalid"
    except Exception as exc:
        first_error = "timeout" if is_timeout_exception(exc) else "planner_exception"
        errors = [str(exc)]
        if first_error != "timeout":
            errors.append("first_attempt_planner_exception")
            return {
                "plan": None,
                "errors": list(dict.fromkeys(error for error in errors if error)),
                "model_name": model_name,
                "llm_planner_error": "planner_exception",
            }

    retry_mode = fallback_plan.get("retrieval_mode") or deterministic_plan.get("retrieval_mode") or "bm25_only"
    retry_system, retry_user = build_minimal_retry_planner_prompt(query, deterministic_plan, retry_mode)
    retry_timeout = planner_retry_timeout_seconds(search_config)
    try:
        retry_text, retry_model_name = call_planner(
            retry_system,
            retry_user,
            max_tokens_override=200,
            timeout_override=retry_timeout,
        )
        retry_raw_plan = parse_model_plan_response(retry_text)
        retry_plan, retry_errors = validate_model_plan(retry_raw_plan, deterministic_plan, query, embedding_health)
        if retry_plan is not None:
            retry_errors = list(retry_errors or [])
            retry_errors.append(f"first_attempt_{first_error}")
            return {
                "plan": retry_plan,
                "errors": list(dict.fromkeys(retry_errors)),
                "model_name": retry_model_name,
                "llm_planner_error": "",
            }
        retry_error = "model_plan_parse_failed" if retry_raw_plan is None else "model_plan_invalid"
        errors = list(errors or []) + [f"retry_{error}" for error in (retry_errors or [retry_error])]
        model_name = retry_model_name or model_name
        final_error = f"retry_{retry_error}"
    except Exception as exc:
        retry_error = "timeout" if is_timeout_exception(exc) else "planner_exception"
        errors = list(errors or []) + [f"retry_{retry_error}", str(exc)]
        final_error = f"retry_{retry_error}"

    if first_error == "timeout":
        errors.append("first_attempt_timeout")
    elif first_error:
        errors.append(f"first_attempt_{first_error}")
    return {
        "plan": None,
        "errors": list(dict.fromkeys(error for error in errors if error)),
        "model_name": model_name,
        "llm_planner_error": final_error,
    }


def _valid_subquery(item, default_mode):
    if not isinstance(item, dict):
        return None, "subquery_not_object"
    label = _clean_string(item.get("label") or "subquery", 40)
    raw_query = str(item.get("query") or "").strip()
    if len(raw_query) > MAX_SUBQUERY_CHARS or len(_tokens(raw_query)) > MAX_SUBQUERY_TOKENS:
        return None, "subquery_too_long"
    query = _clean_string(raw_query, MAX_SUBQUERY_CHARS)
    if not query:
        return None, "empty_subquery"
    mode = (item.get("retrieval_mode") or default_mode or "bm25_only").strip().lower()
    if mode not in ALLOWED_RETRIEVAL_MODES:
        return None, "invalid_subquery_retrieval_mode"
    return {"label": label, "query": query, "retrieval_mode": mode}, ""


def validate_model_plan(raw_plan, deterministic_plan, query, embedding_health=None):
    errors = []
    if not isinstance(raw_plan, dict):
        return None, ["plan_not_object"]

    retrieval_mode = (raw_plan.get("retrieval_mode") or "").strip().lower()
    if retrieval_mode not in ALLOWED_RETRIEVAL_MODES:
        retrieval_mode, reason = choose_retrieval_mode(query, deterministic_plan, embedding_health)
        errors.append("invalid_retrieval_mode")
    else:
        reason = _clean_string(raw_plan.get("retrieval_mode_reason") or "model_planner", 160)

    if (embedding_health or {}).get("status") in {"stale_engine", "missing_index", "provider_failure", "query_embedding_failure"}:
        retrieval_mode = "bm25_only"
        reason = f"embedding_health_{(embedding_health or {}).get('status')}"
    else:
        forced_mode, forced_reason = _model_must_respect_bm25(query, deterministic_plan, embedding_health)
        if forced_mode:
            retrieval_mode = forced_mode
            reason = forced_reason
            errors.append("model_retrieval_mode_overridden_to_bm25")

    pass2_policy = (raw_plan.get("pass2_policy") or "if_weak").strip().lower()
    if pass2_policy not in ALLOWED_PASS2_POLICIES:
        errors.append("invalid_pass2_policy")
        pass2_policy = "if_weak"

    subqueries = []
    seen = set()
    for item in list(raw_plan.get("subqueries") or [])[:MAX_SUBQUERIES]:
        subquery, error = _valid_subquery(item, retrieval_mode)
        if error:
            errors.append(error)
            continue
        key = subquery["query"].lower()
        if key in seen:
            continue
        seen.add(key)
        subqueries.append(subquery)

    if not subqueries:
        errors.append("zero_usable_subqueries")
        return None, errors

    coverage_targets = []
    for item in raw_plan.get("coverage_targets") or []:
        if not isinstance(item, dict):
            errors.append("coverage_target_not_object")
            continue
        label = _clean_string(item.get("label") or "", 40)
        terms = [_clean_string(term, 40).lower() for term in (item.get("required_terms") or []) if _clean_string(term, 40)]
        try:
            min_results = max(1, int(item.get("min_results", 1)))
        except Exception:
            min_results = 1
            errors.append("invalid_coverage_min_results")
        if not label or not terms:
            errors.append("invalid_coverage_target")
            continue
        coverage_targets.append({"label": label, "required_terms": terms[:6], "min_results": min_results})

    allowed_tools = {
        "metadata_filter",
        "bm25_search",
        "hybrid_embedding_search",
        "rerank",
        "mmr_context_selection",
        "evidence_evaluation",
        "session_fact_cache",
    }
    required_local_tools = []
    for tool in raw_plan.get("required_local_tools") or []:
        tool = _clean_string(tool, 80)
        if not tool:
            continue
        if tool not in allowed_tools:
            errors.append(f"invalid_local_tool:{tool}")
            continue
        if tool not in required_local_tools:
            required_local_tools.append(tool)

    confidence = raw_plan.get("confidence", (deterministic_plan or {}).get("confidence", 0.0))
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except Exception:
        confidence = float((deterministic_plan or {}).get("confidence") or 0.0)
        errors.append("invalid_confidence")

    plan = dict(deterministic_plan or {})
    plan.update({
        "planner_version": "adaptive_v2",
        "planner_source": "model",
        "intent": raw_plan.get("intent") if raw_plan.get("intent") in {"specific_fact", "compare", "mechanism", "diagnosis", "treatment", "broad_review"} else plan.get("intent", "specific_fact"),
        "intent_variant": _clean_string(raw_plan.get("intent_variant") or plan.get("intent_variant") or "", 60),
        "confidence": round(confidence, 3),
        "retrieval_mode": retrieval_mode,
        "retrieval_mode_reason": reason,
        "use_rerank": bool(raw_plan.get("use_rerank", True)),
        "pass2_policy": pass2_policy,
        "memory_used": bool(raw_plan.get("memory_used", False)),
        "required_local_tools": required_local_tools,
        "budget_reason": _clean_string(raw_plan.get("budget_reason") or "", 120),
        "subqueries": [item["query"] for item in subqueries],
        "subquery_plans": subqueries,
        "coverage_targets": coverage_targets,
        "fallback_reason": _clean_string(raw_plan.get("fallback_reason") or "", 80),
        "plan_validation_errors": list(dict.fromkeys(errors)),
    })
    if plan.get("fallback_reason"):
        plan["retrieval_plan"] = "fallback_current_hybrid"
    else:
        plan["retrieval_plan"] = "bounded_2_pass"
    if any(str(error).startswith("invalid_local_tool:") for error in errors):
        plan["fallback_reason"] = plan.get("fallback_reason") or "invalid_local_tool"
        plan["retrieval_plan"] = "fallback_current_hybrid"
        plan["planner_source"] = "fallback"
    return plan, list(dict.fromkeys(errors))


def apply_adaptive_defaults(deterministic_plan, query, embedding_health=None, search_config=None):
    plan = dict(deterministic_plan or {})
    mode, reason = choose_retrieval_mode(query, plan, embedding_health, search_config)
    subqueries = _bounded_fallback_subqueries(plan.get("subqueries") or [query], query, plan.get("intent"))
    plan.update({
        "planner_version": "adaptive_v2" if is_adaptive_v2_enabled(search_config) else "deterministic_v1",
        "planner_source": "deterministic",
        "retrieval_mode": mode,
        "retrieval_mode_reason": reason,
        "use_rerank": bool((search_config or {}).get("enable_rerank", True)),
        "pass2_policy": "if_coverage_gap" if len(plan.get("subqueries") or []) > 1 else "if_weak",
        "memory_used": bool(plan.get("memory_snapshot")),
        "required_local_tools": [
            "bm25_search" if mode == "bm25_only" else "hybrid_embedding_search",
            "evidence_evaluation",
        ],
        "budget_reason": "deterministic_local_bounded",
        "coverage_targets": [],
        "subquery_plans": [
            {"label": "main" if idx == 0 else f"subquery_{idx}", "query": query_text, "retrieval_mode": mode}
            for idx, query_text in enumerate(subqueries)
        ],
        "subqueries": subqueries,
        "plan_validation_errors": [],
    })
    return plan


def _cap_long_subquery(text):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= MAX_SUBQUERY_CHARS:
        return text
    match = re.match(r"(.{1,%d}?[.!?])(?:\s|$)" % MAX_SUBQUERY_CHARS, text)
    if match:
        return match.group(1).strip()
    head = text[:MAX_SUBQUERY_CHARS].rstrip()
    word_boundary = head.rfind(" ")
    if word_boundary > 0:
        return head[:word_boundary].strip()
    return head.strip()


def _bounded_fallback_subqueries(subqueries, original_query, intent):
    base_items = []
    for item in subqueries or []:
        capped = _cap_long_subquery(item)
        if capped and capped not in base_items:
            base_items.append(capped)
    if not base_items:
        base_items = [_cap_long_subquery(original_query)]
    base_items = [item for item in base_items if item]
    if intent == "broad_review" and original_query and len(str(original_query)) > MAX_SUBQUERY_CHARS:
        base = base_items[0] if base_items else _cap_long_subquery(original_query)
        variants = [base]
        for suffix in ("mechanism", "treatment", "complications"):
            value = _cap_long_subquery(f"{base} {suffix}")
            if value not in variants:
                variants.append(value)
        return variants[:MAX_SUBQUERIES]
    return base_items[:MAX_SUBQUERIES]
