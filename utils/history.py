"""Search history persistence helpers."""

import datetime
import json
import os

from .log import log_debug
from .paths import get_checkpoint_path

MAX_HISTORY_ITEMS = 25


def _history_path():
    return os.path.join(os.path.dirname(get_checkpoint_path()), "search_history.json")


def _load_history_blob():
    path = _history_path()
    if not os.path.exists(path):
        return {"searches": []}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and isinstance(data.get("searches"), list):
            return data
    except Exception as exc:
        log_debug(f"Error loading search history: {exc}")
    return {"searches": []}


def _save_history_blob(data):
    path = _history_path()
    temp_path = path + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        if os.path.exists(path):
            os.replace(temp_path, path)
        else:
            os.rename(temp_path, path)
        return True
    except Exception as exc:
        log_debug(f"Error saving search history: {exc}")
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        return False


def _normalize_query(query):
    return " ".join((query or "").strip().split()).lower()


def get_search_history_queries():
    searches = _load_history_blob().get("searches", [])
    return [entry.get("query", "") for entry in searches if isinstance(entry, dict) and entry.get("query")]


def load_search_history(query):
    query_key = _normalize_query(query)
    if not query_key:
        return None
    for entry in _load_history_blob().get("searches", []):
        if isinstance(entry, dict) and entry.get("query_key") == query_key:
            return entry
    return None


def save_search_history(query, answer, relevant_note_ids, scored_notes, context_note_ids):
    query_text = (query or "").strip()
    query_key = _normalize_query(query_text)
    if not query_key:
        return False

    searches = [
        entry for entry in _load_history_blob().get("searches", [])
        if isinstance(entry, dict) and entry.get("query_key") != query_key
    ]

    compact_scored = []
    for score, note in scored_notes or []:
        note_id = note.get("id") if isinstance(note, dict) else None
        if note_id is None:
            continue
        compact_scored.append((
            float(score),
            {
                "id": note_id,
                "content": note.get("content", ""),
            },
        ))

    searches.insert(0, {
        "query": query_text,
        "query_key": query_key,
        "answer": answer or "",
        "relevant_note_ids": list(relevant_note_ids or []),
        "context_note_ids": list(context_note_ids or []),
        "scored_notes": compact_scored,
        "timestamp": datetime.datetime.now().isoformat(),
    })

    return _save_history_blob({"searches": searches[:MAX_HISTORY_ITEMS]})


def clear_search_history():
    return _save_history_blob({"searches": []})
