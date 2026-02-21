# Specifications for embeddings status detection, banners, and user-facing copy.
# Implements the plan: handling-stale-embeddings-notes (define-detection-rules,
# design-search-banners, design-embeddings-status-copy, error-and-engine-change-messaging)

from __future__ import annotations

import datetime
from typing import Optional, Dict, Any


# ─── 1. DETECTION RULES AND METADATA ──────────────────────────────────────────

# Backlog size thresholds (used for small vs large backlog behavior)
SMALL_BACKLOG_MAX_NOTES = 200
"""Notes below this count: offer inline 'Index recent notes' action in Search dialog."""

LARGE_BACKLOG_MIN_NOTES = 1000
"""Notes at or above this count: show guidance to Settings → Embeddings instead of inline indexing."""

ESTIMATED_MINUTES_SMALL_JOB_MAX = 5
"""If estimated time for pending notes exceeds this, treat as 'large' and avoid auto-start."""

# Heuristic: small = (pending_notes < SMALL_BACKLOG_MAX_NOTES) AND
#            (estimated_minutes <= ESTIMATED_MINUTES_SMALL_JOB_MAX or unknown)
# large = otherwise

# Metadata keys for checkpoint / config summary (computed lazily; stored in checkpoint when available)
CHECKPOINT_KEYS = {
    "processed_note_ids",
    "total_notes",
    "processed_count",
    "errors",
    "timestamp",
}
"""Checkpoint file keys."""

SUMMARY_KEYS = {
    "pending_notes_count",    # Notes missing embeddings or stale
    "failed_notes_count",     # Notes that failed in last run
    "last_full_run_at",       # ISO timestamp of last successful full run
    "embedded_count",         # Notes with valid embeddings
    "total_searchable_notes", # Total notes matching current filter
}
"""Keys for a lightweight summary used by banners and status labels."""

EMBEDDING_METADATA_KEYS = {
    "engine",      # e.g. "ollama", "voyage"
    "model",       # e.g. "nomic-embed-text:latest"
    "dimension",   # int, embedding vector size
}
"""Keys stored with each embedding or in a global metadata row for staleness checks."""

# Detection rules (logical conditions)
# - MISSING: note not in processed_note_ids (or never embedded)
# - STALE_CONTENT: stored content_hash != current note content hash
# - STALE_ENGINE: embedding was created with different engine/dimension than current config
# - FAILED: note in checkpoint error set or retry list (tracked via errors count / failed_notes)

def is_small_backlog(pending_count: int, estimated_minutes: Optional[float] = None) -> bool:
    """True if backlog is small enough for inline 'Index recent notes' action."""
    if pending_count >= SMALL_BACKLOG_MAX_NOTES:
        return False
    if estimated_minutes is not None and estimated_minutes > ESTIMATED_MINUTES_SMALL_JOB_MAX:
        return False
    return True


def is_large_backlog(pending_count: int) -> bool:
    """True if backlog is large; show guidance to Settings → Embeddings."""
    return pending_count >= LARGE_BACKLOG_MIN_NOTES


# ─── 2. SEARCH DIALOG BANNERS & STATUS ─────────────────────────────────────────

class SearchBannerMessages:
    """Exact banner/status messages and actions for the Search dialog."""

    # When some notes are missing embeddings (search still usable)
    TITLE = "Indexing status: some notes not yet included"
    BODY = (
        "AI search will ignore notes without embeddings. "
        "You can index them now so results cover all your notes."
    )
    ACTION_PRIMARY = "Index recent notes"
    ACTION_SECONDARY_LINK = "Open Embeddings settings"

    # While background update is running
    PROGRESS_TEMPLATE = (
        "Creating embeddings for recent notes ({current}/{total})… "
        "You can keep using Anki; results will improve as indexing finishes."
    )

    # On completion (dismissible toast)
    COMPLETION_TOAST = "Embeddings updated – AI search now covers all indexed notes."

    # Large backlog: no inline action, point to Settings
    LARGE_BACKLOG_TITLE = "Indexing status: many notes not yet included"
    LARGE_BACKLOG_BODY = (
        "Many notes still don't have embeddings. Open Settings → Embeddings "
        "and run 'Create/Update Embeddings' to index everything."
    )
    LARGE_BACKLOG_ACTION = "Open Embeddings settings"


def format_search_progress(current: int, total: int) -> str:
    return SearchBannerMessages.PROGRESS_TEMPLATE.format(current=current, total=total)


# ─── 3. EMBEDDINGS TAB STATUS LABEL & TOOLTIPS ─────────────────────────────────

class EmbeddingsTabMessages:
    """Status label and tooltips for Settings → Embeddings."""

    # Status label: whether embeddings exist, coverage, last run
    NO_EMBEDDINGS = (
        "No embeddings yet. Use 'Test Connection' to verify setup, "
        "then run 'Create/Update Embeddings' to index your notes."
    )
    COVERAGE_TEMPLATE = (
        "Embeddings exist for ~{pct}% of notes ({embedded:,}/{total:,})."
    )
    LAST_RUN_TEMPLATE = "Last updated {when} using {engine}."
    LAST_RUN_UNKNOWN = "Last run: unknown."

    # Full status examples (concatenate as needed):
    # - "Embeddings exist for ~85% of notes (2,300/2,700). Last updated 3 days ago using Ollama."
    # - "Embeddings exist for ~85% of notes (2,300/2,700). Last run: unknown."

    # Tooltips
    TEST_CONNECTION_TOOLTIP = (
        "Quick preflight check: verifies the selected embeddings engine (Voyage, OpenAI, "
        "Cohere, or Ollama) before creating embeddings. Run this first if you changed "
        "URL, model, or API key."
    )
    CREATE_UPDATE_TOOLTIP = (
        "Generate or refresh embeddings for notes that don't have them yet, or that were "
        "embedded with an older engine. Safe to re-run; notes that are already up to date "
        "are skipped. You can stop and resume later using the checkpoint."
    )

    # Large backlog info block (expand status label)
    LARGE_BACKLOG_INFO = (
        "Many notes do not yet have embeddings. Run 'Create/Update Embeddings' when you "
        "have time; you can stop and resume later using the checkpoint."
    )


def format_embeddings_coverage(embedded: int, total: int) -> str:
    if total <= 0:
        return EmbeddingsTabMessages.NO_EMBEDDINGS
    pct = round(100 * embedded / total)
    return EmbeddingsTabMessages.COVERAGE_TEMPLATE.format(pct=pct, embedded=embedded, total=total)


def format_last_run(when: str, engine: str) -> str:
    if when and engine:
        return EmbeddingsTabMessages.LAST_RUN_TEMPLATE.format(when=when, engine=engine)
    return EmbeddingsTabMessages.LAST_RUN_UNKNOWN


def format_relative_time(iso_timestamp: Optional[str]) -> str:
    """Convert ISO timestamp to user-friendly relative time (e.g. '3 days ago').
    Checkpoint stores naive local timestamps; timezone-aware values are stripped to naive for delta."""
    if not iso_timestamp:
        return ""
    try:
        ts = iso_timestamp.replace("Z", "+00:00").strip()
        # Drop timezone suffix (checkpoint uses naive local; e.g. 2026-02-15T18:19:02)
        idx = ts.find("+")
        if idx >= 0:
            ts = ts[:idx].rstrip()
        else:
            idx = ts.find("-", 10)  # Skip date dashes (positions 4, 7)
            if idx >= 0:
                ts = ts[:idx].rstrip()
        dt = datetime.datetime.fromisoformat(ts)
        if dt.tzinfo:
            dt = dt.replace(tzinfo=None)
        delta = datetime.datetime.now() - dt
        days = delta.days
        if days >= 365:
            y = days // 365
            return f"{y} year{'s' if y != 1 else ''} ago"
        if days >= 30:
            m = days // 30
            return f"{m} month{'s' if m != 1 else ''} ago"
        if days >= 7:
            w = days // 7
            return f"{w} week{'s' if w != 1 else ''} ago"
        if days >= 1:
            return f"{days} day{'s' if days != 1 else ''} ago"
        hours = delta.seconds // 3600
        if hours >= 1:
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        mins = delta.seconds // 60
        if mins >= 1:
            return f"{mins} minute{'s' if mins != 1 else ''} ago"
        return "just now"
    except Exception:
        return ""


# ─── 4. ERROR AND ENGINE-CHANGE MESSAGING ─────────────────────────────────────

class ErrorAndEngineMessages:
    """User-facing messages for partial failures and engine-change scenarios."""

    # Partial failures (some notes could not be embedded)
    PARTIAL_FAILURE_EMBEDDINGS_TAB = (
        "Some notes could not be embedded (e.g. API errors). They will be retried "
        "next time you run Create/Update Embeddings."
    )
    PARTIAL_FAILURE_PROGRESS_LOG = (
        "{count} note(s) still missing embeddings — run Create/Update Embeddings again to retry."
    )
    PARTIAL_FAILURE_COMPLETION_DIALOG = (
        "{count} note(s) could not be embedded. Run Create/Update Embeddings again to retry them."
    )

    # Engine change prompt (user changed embedding engine or model)
    ENGINE_CHANGE_PROMPT = (
        "You changed the embeddings engine. To get the best results, re-run "
        "'Create/Update Embeddings' so notes use the new engine."
    )

    # Dimension mismatch (already used in search; keep consistent)
    DIMENSION_MISMATCH_HINT = (
        "Embedding dimension mismatch — notes were embedded with a different engine; "
        "run Create/Update Embeddings with current engine ({engine})"
    )


def format_partial_failure_progress(count: int) -> str:
    return ErrorAndEngineMessages.PARTIAL_FAILURE_PROGRESS_LOG.format(count=count)


def format_partial_failure_completion(count: int) -> str:
    return ErrorAndEngineMessages.PARTIAL_FAILURE_COMPLETION_DIALOG.format(count=count)


def format_dimension_mismatch_hint(engine: str) -> str:
    return ErrorAndEngineMessages.DIMENSION_MISMATCH_HINT.format(engine=engine)
