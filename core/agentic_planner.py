"""Deterministic Agentic RAG planning and evidence checks."""

import re

from .keyword_scoring import extract_keywords_improved, get_extended_stop_words


DEFAULT_PLANNER_CONFIDENCE_THRESHOLD = 0.6
BROAD_REVIEW_MAX_CONTEXT_NOTES = 12
BROAD_REVIEW_MMR_LAMBDA = 0.6
SINGLE_TERM_WEIGHT = 1.0
HIGH_SIGNAL_PHRASE_WEIGHT = 2.0
RUNNER_UP_INCLUDE_RATIO = 0.7
SPECIFIC_FACT_BASELINE = 0.45
INTENT_CONFIDENCE_SCORE_CEILING = 3.0
MIXED_INTENT_SIGNAL_BONUS = 1.0
EXPLICIT_COMPARE_SIGNAL_BONUS = 2.0

INTENT_PRIORITY = {
    "compare": 0,
    "treatment": 1,
    "diagnosis": 2,
    "mechanism": 3,
    "list_review": 4,
    "broad_review": 4,
    "specific_fact": 5,
}

INTENT_VOCABULARY = {
    "compare": {
        "terms": (
            "vs", "versus", "compare", "differentiate", "distinguish",
            "difference", "contrast", "similarities", "similarity",
        ),
        "phrases": (
            "compared with", "compared to", "difference between",
            "differentiate between",
        ),
    },
    "treatment": {
        "terms": (
            "treat", "treatment", "treatments", "regimen", "regimens",
            "protocol", "therapy", "therapies", "antibiotic", "antibiotics",
            "antimicrobial", "antimicrobials", "drug", "drugs", "medication",
            "medications", "maintenance", "prophylaxis", "prevention",
            "prevent", "management", "manage",
        ),
        "phrases": (
            "first-line", "first line", "second-line", "second line",
            "treatment regimen", "drug regimen", "antibiotic regimen",
            "maintenance therapy", "treatment complications",
        ),
    },
    "diagnosis": {
        "terms": (
            "diagnosis", "diagnostic", "diagnose", "workup", "test", "tests",
            "lab", "labs", "criteria", "screen", "screening", "findings",
            "finding", "evaluation", "evaluate",
        ),
        "phrases": (
            "diagnostic criteria", "workup for", "tests for", "lab findings",
            "screening test",
        ),
    },
    "mechanism": {
        "terms": (
            "why", "how", "mechanism", "pathophysiology", "pathophys",
            "pathogenesis", "cause", "causes", "etiology", "etiologies",
        ),
        "phrases": (
            "leads to", "lead to", "results in", "result in", "due to",
            "pathophysiology of", "caused by",
        ),
    },
    "broad_review": {
        "terms": (
            "summarize", "summary", "overview", "review", "high-yield",
            "explain", "everything",
        ),
        "phrases": (
            "approach to", "high yield", "overview of", "summarize",
            "overview", "review", "explain",
        ),
    },
    "list_review": {
        "terms": (
            "factor", "factors", "feature", "features", "manifestation",
            "manifestations", "complication", "complications", "type",
            "types", "classification", "staging", "signs", "symptoms",
            "toxins", "enzymes", "virulence",
        ),
        "phrases": (
            "risk factors", "virulence factors", "clinical features",
            "signs and symptoms", "types of", "classification of",
            "complications of",
        ),
    },
}

LIST_REVIEW_TRIGGER_TERMS = {
    "factor", "factors", "feature", "features", "manifestation",
    "manifestations", "complication", "complications", "type", "types",
    "classification", "staging", "signs", "symptoms", "toxins", "enzymes",
    "virulence",
}

LIST_REVIEW_PRESERVED_PHRASES = {
    "risk factors", "virulence factors",
}

MEDICAL_ENTITY_TERMS = {
    "acid", "anemia", "artery", "asthma", "bilirubin", "bradycardia", "cancer",
    "cardiac", "cell", "cerebral", "chronic", "deficiency", "diabetes",
    "disease", "enzyme", "failure", "fetal", "glucose", "heart", "hormone",
    "hyperplasia", "hypertension", "hypoxia", "infection", "kidney", "liver",
    "lung", "malformation", "metabolism", "mutation", "neonatal", "nerve",
    "pediatric", "placenta", "pneumonia", "pulmonary", "renal", "respiratory",
    "seizure", "shock", "syndrome", "thyroid", "tumor", "vasopressin",
}

DRUG_SUFFIX_RE = re.compile(
    r"(?:cillin|cycline|floxacin|mycin|azole|statin|pril|sartan|olol|pine|"
    r"prazole|tidine|parin|caine|zepam|lamide|gliptin|gliflozin|mab)$",
    re.IGNORECASE,
)
LAB_VALUE_RE = re.compile(
    r"\b(?:pH|PaO2|PaCO2|HCO3|Na|K|Cl|Ca|Mg|BUN|Cr|AST|ALT|TSH|T3|T4|"
    r"HbA1c|WBC|RBC|platelets?|INR|PTT|LDH|G6PD|Ig[GAME])\b",
    re.IGNORECASE,
)
IMAGE_RE = re.compile(r"\b(?:image|picture|photo|diagram|x[-\s]?ray|ct|mri|ultrasound|ecg|ekg|histology|gross|microscopy|media)\b", re.IGNORECASE)


def planner_config(search_config=None):
    sc = search_config or {}
    try:
        threshold = float(sc.get("planner_confidence_threshold", DEFAULT_PLANNER_CONFIDENCE_THRESHOLD))
    except Exception:
        threshold = DEFAULT_PLANNER_CONFIDENCE_THRESHOLD
    threshold = max(0.0, min(1.0, threshold))
    return {
        "threshold": threshold,
        "broad_review_max_context_notes": int(sc.get("agentic_broad_review_max_context_notes", BROAD_REVIEW_MAX_CONTEXT_NOTES) or BROAD_REVIEW_MAX_CONTEXT_NOTES),
        "broad_review_mmr_lambda": float(sc.get("agentic_broad_review_mmr_lambda", BROAD_REVIEW_MMR_LAMBDA) or BROAD_REVIEW_MMR_LAMBDA),
    }


def _tokens(text):
    return re.findall(r"\b[\w+-]+\b", text or "")


def _dedupe_preserve_order(items):
    out = []
    seen = set()
    for item in items or []:
        key = str(item).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _dedupe_phrase_words(text):
    words = _tokens(text)
    cleaned = []
    for word in words:
        if cleaned and cleaned[-1].lower() == word.lower():
            continue
        cleaned.append(word)
    return " ".join(cleaned).strip() or (text or "").strip()


def _append_unique_suffix(base, suffix):
    base = (base or "").strip()
    suffix = (suffix or "").strip()
    if not base:
        return suffix
    if not suffix:
        return base
    base_words = [word.lower() for word in _tokens(base)]
    suffix_words = [word.lower() for word in _tokens(suffix)]
    if suffix_words and len(base_words) >= len(suffix_words) and base_words[-len(suffix_words):] == suffix_words:
        return base
    return f"{base} {suffix}".strip()


def _repair_compare_parts(parts):
    cleaned = []
    for part in parts:
        part = part.strip(" ?.,;:")
        part = re.sub(r"^(?:compare|differentiate|distinguish|contrast)\s+", "", part, flags=re.IGNORECASE).strip()
        if len(part) > 2:
            cleaned.append(part)
    if len(cleaned) < 2:
        return cleaned
    right_tokens = _tokens(cleaned[-1])
    if len(right_tokens) < 2:
        return cleaned
    shared_suffix = right_tokens[-1]
    suffix_l = shared_suffix.lower()
    if suffix_l in {"syndrome", "disease", "deficiency", "anemia", "diabetes", "infection", "failure", "cancer"}:
        repaired = []
        for part in cleaned[:-1]:
            part_tokens = _tokens(part)
            part_l = [token.lower() for token in part_tokens]
            if suffix_l not in part_l and len(part_tokens) < len(right_tokens):
                repaired.append(f"{part} {shared_suffix}".strip())
            else:
                repaired.append(part)
        repaired.append(cleaned[-1])
        return repaired
    return cleaned


def has_entity_signal(query, search_config=None):
    tokens = _tokens(query)
    if not tokens:
        return False
    stop_words = get_extended_stop_words(search_config)
    for idx, token in enumerate(tokens):
        clean = token.strip("_-").lower()
        if not clean or clean in stop_words:
            continue
        if clean in MEDICAL_ENTITY_TERMS:
            return True
        if idx > 0 and token[:1].isupper() and len(token) > 2:
            return True
        if DRUG_SUFFIX_RE.search(clean):
            return True
        if LAB_VALUE_RE.search(token):
            return True
        if len(clean) > 6:
            return True
    return False


def _term_pattern(term):
    return r"\b" + re.escape(term).replace(r"\-", r"[-\s]?") + r"\b"


def _score_vocabulary(query, vocabulary):
    raw = 0.0
    matched = []
    for phrase in vocabulary.get("phrases", ()):
        if re.search(_term_pattern(phrase), query, re.IGNORECASE):
            raw += HIGH_SIGNAL_PHRASE_WEIGHT
            matched.append(phrase)
    for term in vocabulary.get("terms", ()):
        if re.search(_term_pattern(term), query, re.IGNORECASE):
            raw += SINGLE_TERM_WEIGHT
            matched.append(term)
    return raw, tuple(dict.fromkeys(matched))


def _raw_intent_scores(query):
    q = query or ""
    scores = {"specific_fact": SPECIFIC_FACT_BASELINE}
    matches = {"specific_fact": ()}
    for intent, vocabulary in INTENT_VOCABULARY.items():
        raw, matched = _score_vocabulary(q, vocabulary)
        scores[intent] = raw
        matches[intent] = matched

    matched_intents = [
        intent for intent in INTENT_VOCABULARY
        if scores.get(intent, 0.0) > 0.0
    ]
    if len(matched_intents) > 1:
        for intent in matched_intents:
            scores[intent] += MIXED_INTENT_SIGNAL_BONUS
    if re.search(r"\b(?:vs|versus|compared\s+(?:with|to)|difference\s+between)\b", q, re.IGNORECASE):
        scores["compare"] += EXPLICIT_COMPARE_SIGNAL_BONUS

    token_count = len(_tokens(q))
    if re.search(r"\b(?:what is|define|which|when|where|who)\b", q, re.IGNORECASE):
        scores["specific_fact"] += 0.20
    if token_count >= 10:
        scores["broad_review"] = max(scores["broad_review"], HIGH_SIGNAL_PHRASE_WEIGHT)
    return scores, matches


def _normalize_intent_scores(raw_scores):
    scores = {}
    for intent, raw in raw_scores.items():
        capped_raw = min(INTENT_CONFIDENCE_SCORE_CEILING, float(raw))
        if intent == "specific_fact":
            scores[intent] = min(1.0, capped_raw)
        else:
            scores[intent] = capped_raw / INTENT_CONFIDENCE_SCORE_CEILING
    return scores


def _rank_intents(query):
    raw_scores, matches = _raw_intent_scores(query)
    scores = _normalize_intent_scores(raw_scores)
    ranked = sorted(
        scores.items(),
        key=lambda item: (item[1], -INTENT_PRIORITY.get(item[0], 99)),
        reverse=True,
    )
    return ranked, raw_scores, matches


def _score_intents(query):
    ranked, _raw_scores, _matches = _rank_intents(query)
    return dict(ranked)


def _subqueries(query, intent, search_config=None):
    keywords, _stems, phrases = extract_keywords_improved(query, search_config)
    anchors = _dedupe_preserve_order(phrases[:2] or keywords[:4] or _tokens(query)[:4])
    base = " ".join(anchors).strip() or query
    base = _dedupe_phrase_words(base)
    if intent == "compare":
        parts = re.split(r"\b(?:vs|versus|compared with|compared to|between)\b", query, flags=re.IGNORECASE)
        cleaned = _repair_compare_parts(parts)
        if len(cleaned) >= 2:
            return cleaned[:4]
        return [base, f"{base} differences", f"{base} similarities"]
    if intent == "mechanism":
        return [base, _append_unique_suffix(base, "mechanism"), _append_unique_suffix(base, "pathophysiology")]
    if intent == "diagnosis":
        return [base, _append_unique_suffix(base, "diagnosis"), _append_unique_suffix(base, "labs findings")]
    if intent == "treatment":
        return [base, _append_unique_suffix(base, "treatment"), _append_unique_suffix(base, "management")]
    if intent == "broad_review":
        return [base, _append_unique_suffix(base, "high yield"), _append_unique_suffix(base, "diagnosis treatment")]
    return [query]


def _clean_list_review_topic(text):
    cleaned = re.sub(r"\s+", " ", (text or "").strip(" ?.,;:"))
    cleaned = re.sub(r"^(?:what\s+are|what\s+is|list|show|give|provide)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:the\s+)?", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" ?.,;:")


def _list_review_base(query):
    q = _clean_list_review_topic(query)
    for trigger in ("risk factors", "virulence factors", "clinical features", "types", "classification", "complications"):
        match = re.match(rf"^{_term_pattern(trigger)}\s+(?:of|for|in)\s+(.+)$", q, flags=re.IGNORECASE)
        if match:
            topic = _clean_list_review_topic(match.group(1))
            if trigger in LIST_REVIEW_PRESERVED_PHRASES:
                return f"{topic} {trigger}".strip()
            if trigger == "clinical features":
                return topic
            return f"{topic} {trigger}".strip()

    tokens = _tokens(q)
    if len(tokens) > 1 and tokens[-1].lower() in LIST_REVIEW_TRIGGER_TERMS:
        suffix = tokens[-1].lower()
        phrase = " ".join(token.lower() for token in tokens[-2:])
        if phrase not in LIST_REVIEW_PRESERVED_PHRASES:
            return " ".join(tokens[:-1]).strip() or q
    return q


def _list_review_subqueries(query):
    base = _list_review_base(query)
    subqueries = [query]
    base_lower = base.lower()
    detail_tail = (
        "features findings"
        if re.search(r"\b(?:classification|complications?)\b", base_lower)
        else "classification complications"
    )
    for subquery in (
        f"{base} high yield",
        f"{base} features findings",
        f"{base} {detail_tail}",
    ):
        if subquery not in subqueries:
            subqueries.append(subquery)
    return subqueries


def _mixed_intent_subqueries(query, primary_intent, runner_up_intent, search_config=None):
    primary = list(_list_review_subqueries(query) if primary_intent == "list_review" else _subqueries(query, primary_intent, search_config))
    if not runner_up_intent or runner_up_intent == "specific_fact":
        return primary
    runner = _list_review_subqueries(query) if runner_up_intent == "list_review" else _subqueries(query, runner_up_intent, search_config)
    for subquery in runner:
        if subquery not in primary:
            primary.append(subquery)
            break
    return primary


def build_agentic_plan(query, session_context=None, search_config=None):
    config = planner_config(search_config)
    threshold = config["threshold"]
    query = (query or "").strip()
    session_context = session_context or {}
    token_count = len(_tokens(query))
    prior_query = (session_context.get("prior_query") or "").strip()
    entity_signal = has_entity_signal(query, search_config)
    follow_up = bool(token_count < 15 and prior_query and not entity_signal)
    image_intent = bool(IMAGE_RE.search(query))

    ranked, raw_scores, _matches = _rank_intents(query)
    planned_intent, confidence = ranked[0]
    intent = planned_intent
    intent_variant = ""
    if planned_intent == "list_review":
        intent = "broad_review"
        intent_variant = "list_review"
    runner_up_intent = ""
    if len(ranked) > 1:
        for candidate_intent, _candidate_confidence in ranked[1:]:
            if candidate_intent != "specific_fact":
                runner_up_intent = candidate_intent
                break
    fallback_reason = ""
    if confidence < threshold:
        fallback_reason = "low_planner_confidence"
        intent = "specific_fact"

    include_runner_up = bool(
        runner_up_intent
        and not fallback_reason
        and planned_intent != "specific_fact"
        and raw_scores.get(runner_up_intent, 0.0) >= raw_scores.get(planned_intent, 0.0) * RUNNER_UP_INCLUDE_RATIO
    )
    subqueries = (
        _mixed_intent_subqueries(query, planned_intent, runner_up_intent, search_config)
        if include_runner_up else (
            _list_review_subqueries(query)
            if planned_intent == "list_review" else _subqueries(query, intent, search_config)
        )
    )
    subqueries = _dedupe_preserve_order(subqueries)[:4]
    if fallback_reason:
        subqueries = [query]
        intent_variant = ""

    return {
        "agentic_enabled": True,
        "intent": intent,
        "intent_variant": intent_variant,
        "confidence": round(float(confidence), 3),
        "threshold": threshold,
        "subqueries": subqueries,
        "follow_up": follow_up,
        "image_intent": image_intent,
        "retrieval_plan": "fallback_current_hybrid" if fallback_reason else "bounded_2_pass",
        "fallback_reason": fallback_reason,
        "entity_signal": entity_signal,
        "broad_review_max_context_notes": config["broad_review_max_context_notes"],
        "broad_review_mmr_lambda": config["broad_review_mmr_lambda"],
    }


def _note_text(note):
    return (note.get("content") or "").lower() if isinstance(note, dict) else ""


def _concept_terms(query, search_config=None):
    keywords, _stems, phrases = extract_keywords_improved(query, search_config)
    generic = {"syndrome", "disease", "disorder", "condition", "finding", "findings", "type", "types"}
    specific_keywords = [term for term in keywords if term.lower() not in generic]
    terms = phrases[:2] + specific_keywords[:4]
    return [term.lower() for term in terms if len(term) > 2]


def evaluate_agentic_evidence(plan, pass1_results, search_config=None):
    plan = plan or {}
    results = list(pass1_results or [])
    pass2_policy = (plan.get("pass2_policy") or "if_coverage_gap").strip().lower()
    if pass2_policy == "never":
        return {
            "should_run_pass2": False,
            "evidence_status": "pass2_disabled",
            "coverage_gaps": [],
            "unmet_coverage_targets": [],
            "memory_hint_gaps": [],
            "top_score": 0.0,
            "enough_results": False,
            "strong_top": False,
            "pass2_policy": pass2_policy,
            "pass2_reason": "policy_never",
        }
    threshold = float((search_config or {}).get("relevance_threshold_percent", 65) or 65)
    top = results[:10]
    top_score = float(top[0][0]) if top else 0.0
    enough_results = len([score for score, _note in top if float(score or 0) >= max(20.0, threshold * 0.5)]) >= 3
    strong_top = top_score >= threshold

    coverage_gaps = []
    unmet_coverage_targets = []
    for subquery in plan.get("subqueries") or []:
        terms = _concept_terms(subquery, search_config)
        if not terms:
            continue
        covered = False
        for _score, note in top:
            text = _note_text(note)
            if any(term in text for term in terms):
                covered = True
                break
        if not covered:
            coverage_gaps.append(subquery)

    for target in plan.get("coverage_targets") or []:
        if not isinstance(target, dict):
            continue
        terms = [str(term).lower() for term in (target.get("required_terms") or []) if str(term).strip()]
        if not terms:
            continue
        try:
            min_results = max(1, int(target.get("min_results", 1)))
        except Exception:
            min_results = 1
        matches = 0
        for _score, note in top:
            text = _note_text(note)
            if any(term in text for term in terms):
                matches += 1
        if matches < min_results:
            label = target.get("label") or "coverage_target"
            coverage_gaps.append(label)
            unmet_coverage_targets.append({
                "label": label,
                "required_terms": terms[:6],
                "min_results": min_results,
                "matches": matches,
            })

    memory_hint_gaps = []
    memory_snippets = []
    snapshot = plan.get("memory_snapshot") or {}
    if isinstance(snapshot, dict):
        memory_snippets = snapshot.get("fact_snippets") or []
    for target in unmet_coverage_targets:
        terms = target.get("required_terms") or []
        if not terms:
            continue
        for item in memory_snippets:
            if not isinstance(item, dict):
                continue
            snippet_text = str(item.get("snippet") or item.get("snippet_text") or "").lower()
            if snippet_text and any(term in snippet_text for term in terms):
                memory_hint_gaps.append(target.get("label") or "coverage_target")
                break

    multi_subquery = len(plan.get("subqueries") or []) > 1
    weak_evidence = not (enough_results and strong_top)
    coverage_missing = bool(multi_subquery and coverage_gaps)
    should_run_pass2 = bool(
        not plan.get("fallback_reason")
        and (
            weak_evidence
            if pass2_policy == "if_weak"
            else (weak_evidence or coverage_missing or bool(memory_hint_gaps))
        )
    )
    pass2_reason = ""
    if should_run_pass2:
        pass2_reason = "weak_evidence" if weak_evidence else ("memory_hint_gap" if memory_hint_gaps else "coverage_gap")

    return {
        "should_run_pass2": should_run_pass2,
        "evidence_status": "weak" if weak_evidence else ("coverage_gaps" if coverage_gaps else "strong"),
        "coverage_gaps": coverage_gaps,
        "unmet_coverage_targets": unmet_coverage_targets,
        "memory_hint_gaps": memory_hint_gaps,
        "top_score": round(top_score, 3),
        "enough_results": enough_results,
        "strong_top": strong_top,
        "pass2_policy": pass2_policy,
        "pass2_reason": pass2_reason,
    }


def possible_media_in_note(note):
    text = note.get("content") or ""
    return bool(re.search(r"<\s*img\b|\[sound:|<\s*(video|audio|source)\b|\.(?:png|jpe?g|gif|webp|svg)\b", text, re.IGNORECASE))
