"""Note content loading and search scoring helpers for the search dialog."""

# ============================================================================
# Imports
# ============================================================================

from aqt import mw
from aqt.qt import QApplication

from ..core.engine import _build_deck_query, get_models_with_fields
from ..core.keyword_scoring import (
    _simple_stem,
    apply_intent_boost,
    compute_bm25_scores,
    extract_keywords_improved,
    extract_query_intent,
    get_extended_stop_words,
)
from ..utils import get_retrieval_config, load_config, log_debug
from ..utils.text import build_searchable_note_chunks, clean_html, reveal_cloze


# ============================================================================
# Note Content Loading
# ============================================================================

def get_all_notes_content(dialog):
    """Load searchable note content for the given search dialog."""
    notes_data = []
    config = load_config()
    ntf = (config or {}).get('note_type_filter', {}) or {}

    if not ntf.get('fields_to_search') and not ntf.get('note_type_fields'):
        try:
            models = get_models_with_fields()
            if models:
                ntf = dict(ntf)
                ntf['note_type_fields'] = {m['name']: m.get('fields', []) for m in models}
        except Exception as e:
            log_debug(f"Could not infer note type fields: {e}")

    if ntf.get('fields_to_search') and not ntf.get('note_type_fields'):
        global_flds = set(f.lower() for f in ntf['fields_to_search'])
        ntf = dict(ntf)
        ntf['note_type_fields'] = {}
        for model in mw.col.models.all():
            field_names = [fld['name'] for fld in model.get('flds', [])]
            ntf['note_type_fields'][model['name']] = [
                field_name for field_name in field_names if field_name.lower() in global_flds
            ]

    if not ntf:
        enabled_set = None
        search_all = False
        ntf_fields = {}
        use_first = False
        fields_description = "Text & Extra"
    else:
        enabled = ntf.get('enabled_note_types')
        enabled_set = set(enabled) if enabled else None
        search_all = bool(ntf.get('search_all_fields', False))
        ntf_fields = ntf.get('note_type_fields') or {}
        use_first = bool(ntf.get('use_first_field_fallback', True))
        fields_description = "all fields" if search_all else "per-type"

    deck_q = _build_deck_query(ntf.get('enabled_decks') if ntf else None)
    note_ids = mw.col.find_notes(deck_q) if deck_q else mw.col.find_notes("")
    total_notes = len(note_ids)
    cache_key = (
        deck_q or '',
        frozenset(enabled_set) if enabled_set is not None else None,
        search_all,
        tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in ntf_fields.items())),
        total_notes,
    )

    cached_key = getattr(dialog, '_cached_notes_key', None)
    cached_notes = getattr(dialog, '_cached_notes', None)
    if cached_key == cache_key and cached_notes is not None:
        log_debug(f"Using cached notes ({len(cached_notes)} chunks/notes)")
        dialog.fields_description = getattr(dialog, 'fields_description', fields_description)
        return cached_notes

    model_map = {}
    for model in mw.col.models.all():
        mid = model['id']
        model_name = model['name']
        if enabled_set is not None and model_name not in enabled_set:
            continue
        flds = model.get('flds', [])
        if search_all:
            indices = list(range(len(flds)))
        else:
            wanted = set(f.lower() for f in (ntf_fields.get(model_name) or []))
            if not wanted and use_first and flds:
                wanted = {field['name'].lower() for field in flds[:2]}
            if not wanted:
                # Default fallback
                wanted = {'text', 'extra'}
            indices = [idx for idx, field in enumerate(flds) if field['name'].lower() in wanted]
        if indices:
            model_map[mid] = (model_name, indices)

    dialog.fields_description = fields_description
    if not note_ids:
        dialog._cached_notes = notes_data
        dialog._cached_notes_key = cache_key
        return notes_data

    progress = getattr(dialog, 'progress_bar', None)
    if progress:
        progress.setMaximum(total_notes)
        progress.setValue(0)

    id_list = ",".join(map(str, note_ids))
    for row_index, (nid, mid, flds_str) in enumerate(
        mw.col.db.execute(f"select id, mid, flds from notes where id in ({id_list})"),
        start=1,
    ):
        model_info = model_map.get(mid)
        if not model_info:
            continue
        model_name, indices = model_info
        fields = (flds_str or "").split("\x1f")
        content_parts = []
        for index in indices:
            if index < len(fields):
                value = fields[index].strip()
                if value:
                    content_parts.append(value)
        if not content_parts:
            continue

        searchable = build_searchable_note_chunks(content_parts)
        if not searchable.get("content"):
            continue
        chunks = searchable.get("chunks") or []
        if len(chunks) <= 1:
            chunk = chunks[0]
            notes_data.append({
                'id': nid,
                'content': chunk['content'],
                'content_hash': chunk['content_hash'],
                'model': model_name,
                'display_content': chunk['display_content'],
            })
        else:
            for chunk in chunks:
                notes_data.append({
                    'id': nid,
                    'content': chunk['content'],
                    'content_hash': chunk['content_hash'],
                    'model': model_name,
                    'display_content': chunk['display_content'],
                    'chunk_index': chunk['chunk_index'],
                    '_full_content': chunk['_full_content'],
                    '_full_display_content': chunk['_full_display_content'],
                })

        if progress and row_index % 100 == 0:
            progress.setValue(row_index)
            QApplication.processEvents()

    if progress:
        progress.setValue(total_notes)

    if not hasattr(dialog, 'fields_description'):
        dialog.fields_description = fields_description

    log_debug(f"Loaded {len(notes_data)} notes from collection")
    dialog._cached_notes = notes_data
    dialog._cached_notes_key = cache_key
    return notes_data


# ============================================================================
# Text And Scoring Helpers
# ============================================================================

def aggregate_scored_notes_by_note_id(scored_notes):
    if not scored_notes:
        return scored_notes

    by_id = {}
    for score, note in scored_notes:
        note_id = note.get('id')
        if note_id is None:
            by_id[id(note)] = (score, note)
            continue
        if note_id not in by_id or score > by_id[note_id][0]:
            representative = dict(note)
            if representative.get('_full_content'):
                representative['display_content'] = representative['content'] = representative['_full_content']
            if representative.get('_full_display_content'):
                representative['display_content'] = representative['_full_display_content']
            by_id[note_id] = (score, representative)

    return sorted(by_id.values(), key=lambda item: -item[0])


def strip_html(text):
    return clean_html(text)


def reveal_cloze_for_display(text):
    return reveal_cloze(text)


def extract_keywords_for_dialog(dialog, query, agent_debug_log=None):
    search_config = (load_config() or {}).get('search_config') or {}
    ai_excluded = getattr(dialog, '_query_ai_excluded_terms', None) or set()
    if not isinstance(ai_excluded, set):
        ai_excluded = set(ai_excluded) if ai_excluded else set()
    query_intent = extract_query_intent(query)
    try:
        dialog._query_intent = query_intent
    except Exception:
        pass
    keywords, stems, phrases = extract_keywords_improved(query, search_config, ai_excluded)

    try:
        if agent_debug_log and "trisom" in (query or "").lower():
            agent_debug_log(
                run_id="pre-fix",
                hypothesis_id="H1",
                location="note_content.extract_keywords_for_dialog",
                message="keywords_extracted",
                data={"query": query, "keywords": keywords, "stems": stems, "phrases": phrases[:10]},
            )
    except Exception:
        pass

    return keywords, stems, phrases


def apply_query_intent_boost_for_dialog(dialog, content, anchor_keywords=None):
    query_intent = getattr(dialog, '_query_intent', None)
    return apply_intent_boost(content, query_intent, anchor_keywords)


def compute_bm25_scores_for_dialog(dialog, notes, query_keywords):
    scores, high_freq_keywords = compute_bm25_scores(notes, query_keywords)
    try:
        dialog._query_high_freq_keywords = high_freq_keywords
        dialog._query_keyword_scoring_method = "bm25"
    except Exception:
        pass
    return scores


class SearchNoteContentMixin:
    """Owns note loading and text-scoring helper methods used by the dialog."""

    get_all_notes_content = get_all_notes_content
    _aggregate_scored_notes_by_note_id = staticmethod(aggregate_scored_notes_by_note_id)
    strip_html = staticmethod(strip_html)
    reveal_cloze_for_display = staticmethod(reveal_cloze_for_display)
    _simple_stem = staticmethod(_simple_stem)

    def _get_extended_stop_words(self):
        cached = getattr(self, "_cached_extended_stop_words", None)
        if cached is not None:
            return cached
        search_config = (load_config() or {}).get('search_config') or {}
        stop_words = get_extended_stop_words(search_config)
        try:
            self._cached_extended_stop_words = stop_words
        except Exception:
            pass
        return stop_words

    def _extract_keywords_improved(self, query):
        cache_key = (
            query or "",
            tuple(sorted(getattr(self, "_query_ai_excluded_terms", None) or [])),
        )
        cached = getattr(self, "_cached_extracted_keywords", None)
        if cached and cached.get("key") == cache_key:
            return cached.get("value")

        from .search_dialog_shared import _agent_debug_log

        value = extract_keywords_for_dialog(self, query, _agent_debug_log)
        try:
            self._cached_extracted_keywords = {"key": cache_key, "value": value}
        except Exception:
            pass
        return value

    def _compute_bm25_scores(self, notes, query_keywords):
        return compute_bm25_scores_for_dialog(self, notes, query_keywords)
