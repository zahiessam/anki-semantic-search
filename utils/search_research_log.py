"""Structured per-search research logging for retrieval/rerank comparisons."""

import datetime
import json
import os
import uuid

from .log import log_debug
from .paths import get_checkpoint_path

MAX_CONTENT_CHARS = 500


def research_log_dir():
    path = os.path.join(os.path.dirname(get_checkpoint_path()), "search_research")
    os.makedirs(path, exist_ok=True)
    return path


def new_search_run_id():
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _scrub_config(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if "key" in lowered or "token" in lowered or "secret" in lowered or "password" in lowered:
                out[key] = "***"
            else:
                out[key] = _scrub_config(item)
        return out
    if isinstance(value, list):
        return [_scrub_config(item) for item in value]
    return value


def compact_note_result(score, note, rank=None):
    note = note or {}
    content = str(note.get("content", "") or "")
    item = {
        "rank": rank,
        "score": float(score) if isinstance(score, (int, float)) else score,
        "note_id": note.get("id"),
        "content_preview": content[:MAX_CONTENT_CHARS],
    }
    for key in (
        "_display_relevance",
        "_answer_relevance_score",
        "_keyword_score",
        "_embedding_score",
        "chunk_index",
        "chunk_count",
    ):
        if key in note:
            item[key] = note.get(key)
    return item


def compact_results(scored_notes, limit=50):
    rows = []
    for idx, pair in enumerate(scored_notes or [], start=1):
        try:
            score, note = pair
        except Exception:
            continue
        rows.append(compact_note_result(score, note, rank=idx))
        if len(rows) >= limit:
            break
    return rows


def write_search_research_report(report):
    """Write latest full report and append a JSONL snapshot for later comparison."""
    try:
        report = dict(report or {})
        report["updated_at"] = datetime.datetime.now().isoformat()
        report["config"] = _scrub_config(report.get("config") or {})
        run_id = report.get("run_id") or new_search_run_id()
        report["run_id"] = run_id

        directory = research_log_dir()
        latest_path = os.path.join(directory, "latest_search_report.json")
        run_path = os.path.join(directory, f"{run_id}.json")
        jsonl_path = os.path.join(directory, "search_runs.jsonl")

        for path in (latest_path, run_path):
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)

        with open(jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(report, ensure_ascii=False) + "\n")
        return run_path
    except Exception as exc:
        log_debug(f"Error writing search research report: {exc}", is_error=True)
        return None
