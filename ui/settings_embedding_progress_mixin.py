# ============================================================================
# Imports
# ============================================================================

import glob
import os
import sqlite3
import subprocess
import time

try:
    import sip
except ImportError:
    try:
        from PyQt6 import sip
    except ImportError:
        try:
            from PyQt5 import sip
        except ImportError:
            sip = None

from aqt import dialogs, mw
from aqt.qt import *
from aqt.utils import showInfo, showText, tooltip

from .dependency_install import _resolve_external_python_exe, install_dependencies
from .settings_constants import (
    ANSWER_CLOUD_PROVIDERS,
    ANSWER_KEY_PROVIDER_PREFIXES,
    EMBEDDING_CLOUD_PROVIDERS,
    EMBEDDING_KEY_PROVIDER_PREFIXES,
)
from .settings_rerank_workers import RerankModelDownloadWorker, RerankModelVerifyWorker
from .theme import (
    get_addon_theme,
    settings_button_style,
    settings_panel_style,
    settings_status_label_style,
    settings_text_style,
)
from .widgets import CollapsibleSection, apply_setting_row_tooltip, settings_field_row, sync_setting_row_tooltips
from ..core.cloud_diagnostics import (
    classify_provider_error,
    provider_status_message,
    test_cloud_answer_connection,
)
from ..core.engine import (
    analyze_note_eligibility,
    clear_checkpoint,
    count_notes_matching_config,
    get_deck_names,
    get_embedding_engine_id,
    get_embedding_for_query,
    get_models_with_fields,
    get_notes_count_per_deck,
    get_notes_count_per_model,
    get_ollama_models,
    load_checkpoint,
    load_embedding_engine_counts,
    load_embeddings_bulk,
    make_embedding_scope_id,
    _normalize_ollama_base_url,
)
from ..core.workers import EmbeddingWorker, RerankCheckWorker
from ..utils import (
    EmbeddingsTabMessages,
    format_partial_failure_completion,
    format_partial_failure_progress,
    get_effective_embedding_config,
    get_embeddings_db_path,
    get_embeddings_storage_path_for_read,
    get_retrieval_config,
    load_config,
    log_debug,
    save_config,
    validate_embedding_config,
)
from ..utils.config import (
    DEFAULT_RERANK_MODEL,
    RERANK_TIMEOUT_SECONDS_DEFAULT,
    RERANK_TIMEOUT_SECONDS_MAX,
    RERANK_TIMEOUT_SECONDS_MIN,
    RERANK_TOP_K_DEFAULT,
    get_rerank_config,
)

_addon_theme = get_addon_theme

class SettingsEmbeddingProgressMixin:
    def _toggle_pause(self, progress_dialog, pause_button, log_text):



        """Toggle pause/resume for embedding process"""



        if progress_dialog._is_paused:



            # Resume



            progress_dialog._is_paused = False



            pause_button.setText("Pause")



            if hasattr(progress_dialog, '_worker') and progress_dialog._worker:



                progress_dialog._worker._is_paused = False



            log_text.append("Resumed processing...")



        else:



            # Pause



            progress_dialog._is_paused = True



            pause_button.setText("\u25b6 Resume")



            if hasattr(progress_dialog, '_worker') and progress_dialog._worker:



                progress_dialog._worker._is_paused = True



            log_text.append("Paused - Click 'Resume' to continue...")


    def _on_embedding_progress_detail(self, progress_dialog, detail):



        detail_label = getattr(progress_dialog, '_detail_label', None)

        progress_bar = getattr(progress_dialog, '_progress_bar', None)

        log_text = getattr(progress_dialog, '_log_text', None)

        if not isinstance(detail, dict):

            return

        checked = int(detail.get("checked") or 0)

        total = int(detail.get("total") or 0)

        batch_size = int(detail.get("batch_size") or 0)

        eta = detail.get("eta") or "calculating..."
        processed = int(detail.get("processed") or 0)
        refreshed = int(detail.get("refreshed") or 0)
        skipped = int(detail.get("skipped") or 0)
        errors = int(detail.get("errors") or 0)
        pending = int(detail.get("pending_count") or 0)

        if progress_bar is not None and total:

            progress_bar.setFormat(f"{checked:,}/{total:,} (%p%)")

        if detail_label is not None:

            error_part = f" | Errors {errors:,}" if errors else ""
            detail_label.setText(
                f"Checked {checked:,}/{total:,} | Updated {refreshed:,} | New {processed:,} | "
                f"Already current {skipped:,} | Pending {pending:,} | ETA {eta}{error_part}"
            )
        QApplication.processEvents()


    def _on_embedding_finished(self, progress_dialog, processed, errors, skipped, refreshed, still_failed_count, note_count):



        """Handle embedding completion"""



        # Invalidate embeddings file cache so next search loads the updated file



        global _embeddings_file_cache, _embeddings_file_cache_path, _embeddings_file_cache_time



        _embeddings_file_cache = None



        _embeddings_file_cache_path = None



        _embeddings_file_cache_time = 0



        status_label = progress_dialog._status_label



        log_text = progress_dialog._log_text



        close_button = progress_dialog._close_button

        pause_button = getattr(progress_dialog, '_pause_button', None)

        if pause_button is not None:

            pause_button.setVisible(False)

        progress_bar = getattr(progress_dialog, '_progress_bar', None)
        detail_label = getattr(progress_dialog, '_detail_label', None)
        if progress_bar is not None:
            progress_bar.setValue(note_count)
            progress_bar.setFormat(f"{note_count:,}/{note_count:,} (100%)")
        if detail_label is not None:
            error_part = f" | Errors {errors:,}" if errors else ""
            detail_label.setText(
                f"Checked {note_count:,}/{note_count:,} | Updated {refreshed:,} | New {processed:,} | "
                f"Already current {skipped:,} | Pending 0 | ETA done{error_part}"
            )







        status_label.setText(
            f"\u2705 Completed! New embeddings: {processed:,}, updated: {refreshed:,}, already current: {skipped:,} ({errors} errors)"
        )



        log_text.append(f"\n\u2705 Embedding scan/update complete!")



        log_text.append(f"New embeddings created: {processed:,} notes")

        if refreshed > 0:

            log_text.append(f"Existing embeddings updated: {refreshed:,} notes")



        if skipped > 0:



            log_text.append(f"Already current: {skipped:,} notes")



        if errors > 0:



            log_text.append(f"Errors: {errors}")



        if still_failed_count > 0:



            log_text.append(f"Warning: {format_partial_failure_progress(still_failed_count)}")







        # Clear checkpoint only when no notes are still missing (so next run is full; missed ones get retried)



        if still_failed_count == 0:



            clear_checkpoint()







        close_button.setVisible(True)

        close_button.setEnabled(True)



        if still_failed_count > 0:



            showInfo(format_partial_failure_completion(still_failed_count))


    def _on_embedding_error(self, progress_dialog, error_msg):



        """Handle embedding error"""



        status_label = progress_dialog._status_label



        log_text = progress_dialog._log_text



        close_button = progress_dialog._close_button

        pause_button = getattr(progress_dialog, '_pause_button', None)

        if pause_button is not None:

            pause_button.setVisible(False)







        status_label.setText(f"\u274c Error: {error_msg}")



        log_text.append(f"\u274c Error: {error_msg}")



        close_button.setVisible(True)

        close_button.setEnabled(True)



        showInfo(f"Error during embedding generation: {error_msg}")


