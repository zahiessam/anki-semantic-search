"""Keyword extraction, TF-IDF scoring, and scored-note aggregation helpers."""

# ============================================================================
# Imports
# ============================================================================

from dataclasses import dataclass, field
import math
import re


# ============================================================================
# Keyword Extraction And Scoring
# ============================================================================

@dataclass(frozen=True)
class QueryIntent:
    intent: str = ""
    matched_terms: tuple = field(default_factory=tuple)
    boost_terms: tuple = field(default_factory=tuple)

    @property
    def has_intent(self):
        return bool(self.intent)


_INTENT_PATTERNS = {
    "cause": (
        r"\bcauses?\b",
        r"\bcaused\s+by\b",
        r"\betiolog(?:y|ies|ic|ical)\b",
        r"\bdue\s+to\b",
        r"\brisk\s+factors?\b",
    ),
    "treatment": (
        r"\btreat(?:ment|ments|ing)?\b",
        r"\bmanage(?:ment|d|s|ing)?\b",
        r"\btherap(?:y|ies|eutic)\b",
        r"\bdrugs?\b",
        r"\bmedications?\b",
        r"\bfirst[-\s]?line\b",
    ),
    "diagnosis": (
        r"\bdiagnos(?:is|tic|e|ed|ing)\b",
        r"\bwork[-\s]?up\b",
        r"\b(?:diagnostic|lab|screening)\s+tests?\b",
        r"\btests?\s+for\b",
    ),
    "complication": (
        r"\bcomplications?\b",
        r"\bassociated\s+with\b",
        r"\bleads?\s+to\b",
        r"\bresults?\s+in\b",
        r"\bsequelae?\b",
    ),
    "classification": (
        r"\bclassif(?:y|ies|ication|ications)\b",
        r"\btypes?\s+of\b",
        r"\bstag(?:e|es|ing)\b",
        r"\bseverity\b",
    ),
}

_INTENT_BOOST_PATTERNS = {
    "cause": (
        (r"\bmost\s+common\s+cause\b", 18),
        (r"\bcaused\s+by\b", 18),
        (r"\bdue\s+to\b", 16),
        (r"\bsecondary\s+to\b", 16),
        (r"\brisk\s+factors?\s+for\b", 15),
        (r"\btrigger(?:ed|s)?\s+by\b", 14),
        (r"\bpredispos(?:e|es|ed|ing)\b", 12),
        (r"\betiolog(?:y|ies|ic|ical)\b", 12),
        (r"\bpathogenesis\b", 10),
    ),
    "treatment": (
        (r"\bfirst[-\s]?line\b", 18),
        (r"\btreat(?:ment|ments|ed|ing)?\b", 16),
        (r"\bmanage(?:ment|d|s|ing)?\b", 16),
        (r"\btherap(?:y|ies|eutic)\b", 14),
        (r"\b(?:oral\s+)?feeds?\b", 12),
        (r"\biv\b", 10),
        (r"\bd10w\b", 14),
        (r"\bbolus\b", 10),
        (r"\binfusion\b", 10),
        (r"\bmedications?\b", 10),
        (r"\bdrugs?\b", 8),
    ),
    "diagnosis": (
        (r"\bdiagnos(?:is|tic|e|ed|ing)\b", 18),
        (r"\bwork[-\s]?up\b", 16),
        (r"\b(?:diagnostic|lab|screening)\s+tests?\b", 15),
        (r"\btests?\s+for\b", 14),
        (r"\bfindings?\b", 10),
        (r"\bcriteria\b", 10),
    ),
    "complication": (
        (r"\bcomplications?\b", 18),
        (r"\bleads?\s+to\b", 16),
        (r"\bresults?\s+in\b", 16),
        (r"\bassociated\s+with\b", 14),
        (r"\bincreases?\s+(?:the\s+)?risk\b", 14),
        (r"\bsequelae?\b", 12),
        (r"\bmanifest(?:s|ed|ing|ation)?\b", 8),
    ),
    "classification": (
        (r"\bclassif(?:y|ies|ication|ications)\b", 18),
        (r"\btypes?\b", 14),
        (r"\bstag(?:e|es|ing)\b", 14),
        (r"\bseverity\b", 14),
        (r"\bintermittent\b", 12),
        (r"\bpersistent\b", 12),
        (r"\bmild\b", 10),
        (r"\bmoderate\b", 10),
        (r"\bsevere\b", 10),
    ),
}

_INTENT_EXCLUDED_WORDS = {
    "cause", "causes", "caused", "etiology", "etiologies", "etiologic", "etiological",
    "treat", "treats", "treated", "treating", "treatment", "treatments",
    "manage", "manages", "managed", "managing", "management",
    "therapy", "therapies", "therapeutic", "drug", "drugs", "medication", "medications",
    "diagnosis", "diagnostic", "diagnose", "diagnosed", "diagnosing", "workup",
    "complication", "complications", "sequela", "sequelae",
    "classification", "classifications", "classify", "types", "type", "staging", "stage", "stages",
    "severity",
}


def extract_query_intent(query):
    query_lower = (query or "").lower()
    best_intent = ""
    best_matches = []
    best_count = 0

    for intent, patterns in _INTENT_PATTERNS.items():
        matches = []
        for pattern in patterns:
            for match in re.finditer(pattern, query_lower):
                matches.append(match.group(0).strip())
        if len(matches) > best_count:
            best_intent = intent
            best_matches = matches
            best_count = len(matches)

    if not best_intent:
        return QueryIntent()

    excluded = set(_INTENT_EXCLUDED_WORDS)
    for term in best_matches:
        excluded.update(re.findall(r"\b\w+\b", term.lower()))
    # Preserve "high risk pregnancy" as a searchable topic; only risk-factor phrasing is intent.
    if not any(re.search(r"\brisk\s+factors?\b", term) for term in best_matches):
        excluded.discard("risk")
    return QueryIntent(
        intent=best_intent,
        matched_terms=tuple(dict.fromkeys(best_matches)),
        boost_terms=tuple(sorted(excluded)),
    )


def apply_intent_boost(text, query_intent, anchor_keywords=None):
    if not query_intent:
        return 0.0
    if isinstance(query_intent, dict):
        intent = query_intent.get("intent") or ""
    else:
        intent = getattr(query_intent, "intent", "") or ""
    if not intent:
        return 0.0

    text_lower = (text or "").lower()
    if intent == "classification" and anchor_keywords:
        anchors = [
            str(keyword).strip().lower()
            for keyword in anchor_keywords
            if keyword and len(str(keyword).strip()) > 2
        ]
        if anchors and not any(anchor in text_lower for anchor in anchors):
            return 0.0

    total = 0.0
    matched = 0
    for pattern, weight in _INTENT_BOOST_PATTERNS.get(intent, ()):
        if re.search(pattern, text_lower):
            total += weight
            matched += 1
            if matched >= 2:
                break
    return min(18.0, total)

def _simple_stem(word):
    if len(word) <= 3:
        return word
    suffixes = ["ing", "ed", "er", "est", "ly", "s", "es", "tion", "sion", "ness", "ment"]
    word_lower = word.lower()
    for suffix in suffixes:
        if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
            return word_lower[: -len(suffix)]
    return word_lower


def get_extended_stop_words(search_config=None):
    builtin_stop_words = {
        "what", "is", "are", "the", "a", "an", "how", "why", "when", "where", "who",
        "does", "do", "can", "could", "would", "should", "tell", "me", "about", "explain",
        "describe", "define", "list", "show", "give", "provide", "this", "that", "these",
        "those", "with", "from", "for", "and", "or", "but", "not", "have", "has", "had",
        "been", "being", "was", "were", "will", "may", "might", "must", "shall",
        "which", "work", "works", "working", "use", "uses", "used", "using",
        "overview", "introduction", "review", "study", "case", "cases",
        "difference", "between", "compared", "comparison", "similar", "similarity",
        "different", "same", "other", "another", "each", "every", "both", "either",
        "neither", "like", "such", "common", "generally", "usually", "often", "typically",
        "example", "examples", "including", "involves", "involve", "related", "association",
    }
    stop_words = set(builtin_stop_words)
    if search_config:
        extra = search_config.get("extra_stop_words") or []
        if isinstance(extra, str):
            extra = [extra]
        for item in extra:
            if isinstance(item, str) and item.strip():
                stop_words.add(item.strip().lower())
    return stop_words


def extract_keywords_improved(query, search_config=None, ai_excluded=None):
    query_lower = (query or "").lower()
    query_words = re.findall(r"\b\w+\b", query_lower)
    stop_words = get_extended_stop_words(search_config)
    query_intent = extract_query_intent(query)
    intent_excluded = set(query_intent.boost_terms)
    ai_excluded = ai_excluded or set()

    keywords = []
    stems = {}
    for word in query_words:
        if (
            word not in stop_words
            and word not in intent_excluded
            and word not in ai_excluded
            and len(word) > 2
        ):
            stem = _simple_stem(word)
            keywords.append(word)
            if stem != word:
                stems[stem] = word

    phrases = []
    if len(keywords) > 1:
        for i in range(len(keywords) - 1):
            phrases.append(f"{keywords[i]} {keywords[i + 1]}")
    if len(keywords) > 2:
        for i in range(len(keywords) - 2):
            phrases.append(f"{keywords[i]} {keywords[i + 1]} {keywords[i + 2]}")
    return keywords, stems, phrases


def compute_bm25_scores(notes, query_keywords, k1=1.5, b=0.75):
    """Compute BM25 scores over the loaded candidate notes.

    This intentionally uses the current candidate set for IDF rather than a
    persistent corpus-wide sparse index, keeping Retrieval V2 dependency-free.
    """
    if not notes or not query_keywords:
        return {}, set()

    note_tfs = {}
    doc_freq = {}
    doc_lengths = {}
    total_length = 0
    unique_keywords = list(dict.fromkeys(query_keywords))

    for note in notes:
        note_id = note["id"]
        tokens = re.findall(r"\b\w+\b", (note.get("content") or "").lower())
        doc_len = max(1, len(tokens))
        doc_lengths[note_id] = doc_len
        total_length += doc_len
        text = " ".join(tokens)
        note_tfs[note_id] = {}
        for keyword in unique_keywords:
            count = text.count(keyword)
            if count > 0:
                note_tfs[note_id][keyword] = count
                doc_freq[keyword] = doc_freq.get(keyword, 0) + 1

    total_notes = max(1, len(notes))
    avg_doc_len = max(1.0, total_length / total_notes)
    high_freq_keywords = {
        keyword for keyword, freq in doc_freq.items() if (freq / total_notes) >= 0.65
    }

    scores = {}
    for note_id, tfs in note_tfs.items():
        doc_len = doc_lengths.get(note_id, avg_doc_len)
        score = 0.0
        for keyword, tf in tfs.items():
            df = doc_freq.get(keyword, 0)
            idf = math.log(1 + (total_notes - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
            score += idf * ((tf * (k1 + 1)) / max(denom, 1e-9))
        scores[note_id] = score

    return scores, high_freq_keywords


def aggregate_scored_notes_by_note_id(scored_notes):
    if not scored_notes:
        return []

    best_by_id = {}
    for score, note in scored_notes:
        note_id = note.get("id")
        if note_id is None:
            continue
        prev = best_by_id.get(note_id)
        if prev is None or score > prev[0]:
            best_by_id[note_id] = (score, note)

    out = list(best_by_id.values())
    out.sort(key=lambda item: item[0], reverse=True)
    return out
