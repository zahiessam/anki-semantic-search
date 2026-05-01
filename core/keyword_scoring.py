"""Keyword extraction, TF-IDF scoring, and scored-note aggregation helpers."""

# ============================================================================
# Imports
# ============================================================================

import math
import re


# ============================================================================
# Keyword Extraction And Scoring
# ============================================================================

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
        "cause", "causes", "overview", "introduction", "review", "study", "case", "cases",
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
    ai_excluded = ai_excluded or set()

    keywords = []
    stems = {}
    for word in query_words:
        if word not in stop_words and word not in ai_excluded and len(word) > 2:
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


def compute_tfidf_scores(notes, query_keywords):
    if not notes or not query_keywords:
        return {}, set()

    note_tfs = {}
    doc_freq = {}
    for note in notes:
        content_lower = note["content"].lower()
        note_tfs[note["id"]] = {}
        for keyword in query_keywords:
            count = content_lower.count(keyword)
            if count > 0:
                note_tfs[note["id"]][keyword] = count
                doc_freq[keyword] = doc_freq.get(keyword, 0) + 1

    total_notes = max(1, len(notes))
    high_freq_keywords = {
        keyword for keyword, freq in doc_freq.items() if (freq / total_notes) >= 0.65
    }

    tfidf_scores = {}
    for note_id, tfs in note_tfs.items():
        score = 0.0
        for keyword, tf in tfs.items():
            idf = math.log(total_notes / max(1, doc_freq[keyword]))
            score += tf * idf
        tfidf_scores[note_id] = score

    return tfidf_scores, high_freq_keywords


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
