"""Structured per-search research logging for retrieval/rerank comparisons."""

import datetime
import json
import os
import re
import uuid

from .log import log_debug
from .paths import get_checkpoint_path

MAX_CONTENT_CHARS = 500
MAX_PROMPT_PREVIEW_CHARS = 4000
MAX_ANSWER_PREVIEW_CHARS = 4000
DEFAULT_MAX_RESEARCH_FILES = 50
CITATION_RE = re.compile(r"\[([\d,\s]+)\]")
RELEVANT_NOTES_RE = re.compile(r"RELEVANT_NOTES:\s*([^\n\r]+)", re.IGNORECASE)


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


def _research_settings(report):
    config = (report or {}).get("config") or {}
    search_config = config.get("search_config") or {}
    mode = (search_config.get("research_mode") or "compact").strip().lower()
    if mode not in ("compact", "full"):
        mode = "compact"
    try:
        max_files = int(search_config.get("max_research_files", DEFAULT_MAX_RESEARCH_FILES))
    except Exception:
        max_files = DEFAULT_MAX_RESEARCH_FILES
    return {
        "enabled": bool(search_config.get("research_enabled", False)),
        "mode": mode,
        "max_research_files": max(1, max_files),
    }


def _preview(text, limit=MAX_CONTENT_CHARS):
    text = str(text or "")
    return text[:limit]


def compact_note_result(score, note, rank=None, mode="compact"):
    note = note or {}
    content = str(note.get("content", "") or "")
    item = {
        "rank": rank,
        "score": float(score) if isinstance(score, (int, float)) else score,
        "note_id": note.get("id"),
        "content_preview": content[:MAX_CONTENT_CHARS],
    }
    if mode == "full":
        item["content"] = content
        display_content = note.get("display_content")
        if display_content and display_content != content:
            item["display_content"] = str(display_content)
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


def compact_results(scored_notes, limit=50, mode="compact"):
    rows = []
    for idx, pair in enumerate(scored_notes or [], start=1):
        try:
            score, note = pair
        except Exception:
            continue
        rows.append(compact_note_result(score, note, rank=idx, mode=mode))
        if len(rows) >= limit:
            break
    return rows


def make_prompt_payload(prompt=None, prompt_parts=None, mode="compact"):
    payload = {}
    if prompt is not None:
        text = str(prompt or "")
        payload["prompt_length"] = len(text)
        if mode == "full":
            payload["prompt"] = text
        else:
            payload["prompt_preview"] = _preview(text, MAX_PROMPT_PREVIEW_CHARS)
    if prompt_parts:
        payload["prompt_parts"] = prompt_parts if mode == "full" else _scrub_prompt_parts(prompt_parts)
    return payload


def _scrub_prompt_parts(prompt_parts):
    if isinstance(prompt_parts, dict):
        out = {}
        for key, value in prompt_parts.items():
            if isinstance(value, str):
                out[key] = _preview(value, MAX_PROMPT_PREVIEW_CHARS)
            elif isinstance(value, list):
                out[key] = [_scrub_prompt_parts(item) for item in value]
            else:
                out[key] = value
        return out
    if isinstance(prompt_parts, list):
        return [_scrub_prompt_parts(item) for item in prompt_parts]
    return prompt_parts


def make_answer_payload(answer, context_note_ids=None, final_results=None, mode="compact"):
    text = str(answer or "")
    inline_refs = extract_inline_citation_refs(text)
    relevant_refs = extract_relevant_note_refs(text)
    return {
        "answer_length": len(text),
        "answer": text if mode == "full" else _preview(text, MAX_ANSWER_PREVIEW_CHARS),
        "inline_citation_refs": inline_refs,
        "relevant_notes_refs": relevant_refs,
        "citation_map": build_citation_map(
            sorted(set(inline_refs + relevant_refs)),
            context_note_ids or [],
            final_results or [],
        ),
    }


def extract_inline_citation_refs(answer):
    refs = []
    for match in CITATION_RE.finditer(str(answer or "")):
        for part in match.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                refs.append(int(part))
    return refs


def extract_relevant_note_refs(answer):
    match = RELEVANT_NOTES_RE.search(str(answer or ""))
    if not match:
        return []
    refs = []
    for part in re.findall(r"\d+", match.group(1)):
        refs.append(int(part))
    return refs


def build_citation_map(refs, context_note_ids, final_results):
    by_note_id = {}
    for row in final_results or []:
        note_id = row.get("note_id")
        if note_id is not None:
            by_note_id[note_id] = row
    rows = []
    context = list(context_note_ids or [])
    for ref in refs or []:
        note_id = context[ref - 1] if 1 <= ref <= len(context) else None
        final = by_note_id.get(note_id) if note_id is not None else None
        rows.append({
            "ref": ref,
            "note_id": note_id,
            "valid": note_id is not None,
            "final_rank": final.get("rank") if final else None,
            "content_preview": final.get("content_preview") if final else None,
        })
    return rows


def rank_deltas(before, after, final=None):
    def ranks(rows):
        out = {}
        for row in rows or []:
            note_id = row.get("note_id")
            if note_id is not None:
                out[note_id] = row.get("rank")
        return out

    before_ranks = ranks(before)
    after_ranks = ranks(after)
    final_ranks = ranks(final)
    note_ids = sorted(set(before_ranks) | set(after_ranks) | set(final_ranks))
    return [
        {
            "note_id": note_id,
            "pre_rank": before_ranks.get(note_id),
            "post_rerank_rank": after_ranks.get(note_id),
            "final_rank": final_ranks.get(note_id),
        }
        for note_id in note_ids
    ]


def compact_report_summary(report, run_path=None):
    stages = report.get("stages") or []
    final_results = report.get("final_results") or []
    answer = report.get("answer") or {}
    return {
        "run_id": report.get("run_id"),
        "started_at": report.get("started_at"),
        "completed_at": report.get("completed_at"),
        "query": report.get("query"),
        "final_status": report.get("final_status"),
        "models": report.get("models"),
        "settings": report.get("settings"),
        "used_history": report.get("used_history"),
        "answer_length": answer.get("answer_length") or report.get("final_answer_length"),
        "inline_citation_refs": answer.get("inline_citation_refs", []),
        "relevant_notes_refs": answer.get("relevant_notes_refs", []),
        "cited_note_ids": report.get("cited_note_ids", []),
        "context_note_ids": report.get("context_note_ids", []),
        "top_final_note_ids": [row.get("note_id") for row in final_results[:20]],
        "stage_names": [stage.get("name") for stage in stages],
        "report_path": run_path,
    }


def cleanup_old_research_reports(directory, max_files):
    try:
        candidates = []
        for name in os.listdir(directory):
            if not name.endswith(".json") or name == "latest_search_report.json":
                continue
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                candidates.append((os.path.getmtime(path), path))
        candidates.sort()
        excess = max(0, len(candidates) - int(max_files))
        for _, path in candidates[:excess]:
            try:
                os.remove(path)
            except Exception as exc:
                log_debug(f"Could not remove old search research report {path}: {exc}")
    except Exception as exc:
        log_debug(f"Error cleaning search research reports: {exc}", is_error=True)


def write_search_research_report(report):
    """Write latest full report and append a JSONL snapshot for later comparison."""
    try:
        report = dict(report or {})
        settings = _research_settings(report)
        if not settings["enabled"]:
            return None
        report["updated_at"] = datetime.datetime.now().isoformat()
        report["config"] = _scrub_config(report.get("config") or {})
        run_id = report.get("run_id") or new_search_run_id()
        report["run_id"] = run_id
        report["research"] = {
            "mode": settings["mode"],
            "max_research_files": settings["max_research_files"],
        }

        directory = research_log_dir()
        latest_path = os.path.join(directory, "latest_search_report.json")
        run_path = os.path.join(directory, f"{run_id}.json")
        jsonl_path = os.path.join(directory, "search_runs.jsonl")

        for path in (latest_path, run_path):
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)

        with open(jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(compact_report_summary(report, run_path), ensure_ascii=False) + "\n")
        cleanup_old_research_reports(directory, settings["max_research_files"])
        return run_path
    except Exception as exc:
        log_debug(f"Error writing search research report: {exc}", is_error=True)
        return None
