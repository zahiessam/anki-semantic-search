"""Shared helpers for search dialog orchestration and diagnostics.

Keep this module free of AISearchDialog imports. It exists so mixins can share
collection-loading and debug helpers without importing ui.dialogs.
"""

import json
import os
import time

from ..core.engine import _build_deck_query
from ..utils import log_debug
from ..utils.text import build_searchable_note_chunks


def _prefilter_safety_reason(col, ntf, config):
    """Return None when V2 can safely prefilter before loading full note text."""
    if not ntf:
        return "no note_type_filter configured"
    if config.get("_history_replay"):
        return "history replay is active"

    enabled_decks = ntf.get("enabled_decks") or []
    if enabled_decks:
        try:
            all_decks = set(col.decks.all_names())
        except Exception:
            return "could not resolve deck names"
        missing_decks = [deck for deck in enabled_decks if deck not in all_decks]
        if missing_decks:
            return f"selected deck(s) not found: {missing_decks[:3]}"

    models = col.models.all()
    model_by_name = {model.get("name"): model for model in models}
    enabled_types = ntf.get("enabled_note_types") or []
    if enabled_types:
        missing_types = [name for name in enabled_types if name not in model_by_name]
        if missing_types:
            return f"selected note type(s) not found: {missing_types[:3]}"

    search_all = bool(ntf.get("search_all_fields", False))
    if search_all and not enabled_types:
        return "all-fields search across unknown/mixed note types"

    if not search_all:
        ntf_fields = ntf.get("note_type_fields") or {}
        use_first = bool(ntf.get("use_first_field_fallback", True))
        type_names = enabled_types or list(model_by_name)
        for model_name in type_names:
            model = model_by_name.get(model_name)
            if not model:
                continue
            available = {field.get("name", "").lower() for field in model.get("flds", [])}
            wanted = {field.lower() for field in (ntf_fields.get(model_name) or [])}
            if wanted and not wanted.issubset(available):
                return f"selected field missing on {model_name}"
            if not wanted and not use_first:
                return f"no fields selected for {model_name} and first-two-fields fallback disabled"

    return None


def get_safe_config(value):
    """Return config-like data with secrets redacted for logging."""
    secret_markers = ("api_key", "token", "secret", "password")

    if isinstance(value, dict):
        safe = {}
        for key, item in value.items():
            key_str = str(key).lower()
            if any(marker in key_str for marker in secret_markers):
                safe[key] = "***REDACTED***" if item else item
            else:
                safe[key] = get_safe_config(item)
        return safe

    if isinstance(value, list):
        return [get_safe_config(item) for item in value]

    return value


def get_notes_content_with_col(col, config):
    """Load searchable note content in a background QueryOp using the provided collection."""
    notes_data = []
    ntf = (config or {}).get('note_type_filter', {}) or {}

    if ntf.get('fields_to_search') and not ntf.get('note_type_fields'):
        global_flds = set(f.lower() for f in ntf['fields_to_search'])
        ntf = dict(ntf)
        ntf['note_type_fields'] = {}
        for model in col.models.all():
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

    safety_reason = _prefilter_safety_reason(col, ntf, config)
    if safety_reason:
        log_debug(f"V2 note prefilter fallback: {safety_reason}")
    else:
        log_debug("V2 note prefilter safe: deck/note-type/field filters resolved")

    deck_q = _build_deck_query(ntf.get('enabled_decks') if ntf else None)
    note_ids = col.find_notes(deck_q) if deck_q else col.find_notes("")
    total_notes = len(note_ids)
    cache_key = (
        deck_q or '',
        frozenset(enabled_set) if enabled_set is not None else None,
        search_all,
        tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in ntf_fields.items())),
        total_notes,
    )

    model_map = {}
    for model in col.models.all():
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
                wanted = {'text', 'extra'}
            indices = [i for i, field in enumerate(flds) if field['name'].lower() in wanted]
        if indices:
            model_map[mid] = (model_name, indices)

    if not note_ids:
        return notes_data, fields_description, cache_key

    id_list = ",".join(map(str, note_ids))
    for nid, mid, flds_str in col.db.execute(f"select id, mid, flds from notes where id in ({id_list})"):
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
            continue

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

    return notes_data, fields_description, cache_key


def _session_debug_log(hypothesis_id, location, message, data=None):
    """Write NDJSON to session log file for debug."""
    try:
        entry = {
            "sessionId": "85902e",
            "timestamp": int(time.time() * 1000),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
        }
        addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(addon_dir, "user_files", "debug-85902e.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _agent_debug_log(run_id, hypothesis_id, location, message, data=None):
    """Lightweight debug logger for agent-driven investigations."""
    try:
        entry = {
            "id": f"log_{int(time.time() * 1000)}",
            "timestamp": int(time.time() * 1000),
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
        }
        addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(addon_dir, "user_files", "debug.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
