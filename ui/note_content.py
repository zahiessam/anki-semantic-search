"""Note content loading and search scoring helpers for the search dialog."""

# ============================================================================
# Imports
# ============================================================================

import hashlib

from aqt import mw
from aqt.qt import QApplication

from ..core.engine import _build_deck_query, get_models_with_fields
from ..core.keyword_scoring import (
    _simple_stem,
    compute_tfidf_scores,
    extract_keywords_improved,
    get_extended_stop_words,
)
from ..utils import load_config, log_debug
from ..utils.text import clean_html, clean_html_for_display, reveal_cloze, semantic_chunk_text


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

    legacy_fields = None
    if not ntf:
        enabled_set = None
        search_all = False
        ntf_fields = {}
        use_first = False
        legacy_fields = {'text', 'extra'}
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
            if legacy_fields is not None:
                wanted = legacy_fields
            else:
                wanted = set(f.lower() for f in (ntf_fields.get(model_name) or []))
                if not wanted and use_first and flds:
                    wanted = {flds[0]['name'].lower()}
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

    chunk_target = 500
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

        raw_content = " | ".join(content_parts)
        content = clean_html(raw_content).strip()
        if not content:
            continue
        display_parts = []
        for part in content_parts:
            display_part = clean_html_for_display(part).strip()
            if display_part:
                display_parts.append(display_part)
        display_content = " | ".join(display_parts) or content

        chunks = semantic_chunk_text(content, chunk_target)
        if len(chunks) <= 1:
            content_hash = hashlib.md5(content.encode()).hexdigest()
            notes_data.append({
                'id': nid,
                'content': content,
                'content_hash': content_hash,
                'model': model_name,
                'display_content': display_content,
            })
        else:
            for chunk_idx, chunk in enumerate(chunks):
                chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
                notes_data.append({
                    'id': nid,
                    'content': chunk,
                    'content_hash': chunk_hash,
                    'model': model_name,
                    'display_content': clean_html_for_display(chunk).strip() or chunk,
                    'chunk_index': chunk_idx,
                    '_full_content': content,
                    '_full_display_content': display_content,
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


def compute_tfidf_scores_for_dialog(dialog, notes, query_keywords):
    scores, high_freq_keywords = compute_tfidf_scores(notes, query_keywords)
    try:
        dialog._query_high_freq_keywords = high_freq_keywords
    except Exception:
        pass
    return scores
