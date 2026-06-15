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

class SettingsRerankModelMixin:
    def _persist_cached_rerank_model_status(self, model=None, python_path=None):
        try:
            config = load_config()
            sc = config.setdefault("search_config", {})
            sc["rerank_package_verified"] = True
            sc["rerank_package_verified_python_path"] = python_path or self._current_rerank_python_path() or ""
            sc["rerank_model_verified"] = True
            sc["rerank_model_verified_python_path"] = python_path or self._current_rerank_python_path() or ""
            sc["rerank_model_verified_name"] = model or self._selected_rerank_model()
            save_config(config)
        except Exception as exc:
            log_debug(f"Could not persist rerank model status: {exc}")


    def _verify_selected_rerank_model_cache(self):

        """Check whether the selected model can load from cache without downloading."""

        if not getattr(self, "_rerank_available", False):

            self._rerank_model_ready = False
            self._rerank_model_ready_name = ""
            self._rerank_model_checking = False
            self._rerank_model_checking_name = ""
            self._rerank_model_verify_message = "Cross-Encoder package is not ready yet."
            self._update_rerank_status_ui()
            return

        model = self._selected_rerank_model()
        python_path = self._current_rerank_python_path()
        token = (python_path or "", model)

        self._rerank_verify_token = token
        self._rerank_model_ready = False
        self._rerank_model_ready_name = ""
        self._rerank_model_checking = True
        self._rerank_model_checking_name = model
        self._rerank_model_verify_message = "Checking the local model cache only. This will not download files."
        self._update_rerank_tooltip()
        self._update_rerank_status_ui()

        self._rerank_verify_worker = RerankModelVerifyWorker(python_path, model)

        def _on_done(success, checked_model, message):

            try:

                if self._rerank_verify_token != token or self._selected_rerank_model() != checked_model:
                    return

                self._rerank_model_checking = False
                self._rerank_model_checking_name = ""
                self._rerank_model_verify_message = message

                if success:
                    self._rerank_model_ready = True
                    self._rerank_model_ready_name = checked_model
                    self._persist_cached_rerank_model_status(checked_model, python_path)
                else:
                    self._rerank_model_ready = False
                    self._rerank_model_ready_name = ""

                self._update_rerank_tooltip()
                self._update_rerank_status_ui()
                self._rerank_verify_worker = None

            except Exception as exc:
                log_debug(f"Rerank model cache verification completion error: {exc}")

        self._rerank_verify_worker.finished_signal.connect(_on_done)
        self._rerank_verify_worker.start()


    def _selected_rerank_model(self):

        combo = getattr(self, "rerank_model_combo", None)

        if combo is None:

            return DEFAULT_RERANK_MODEL

        model = combo.currentData()

        if model:

            return str(model).strip()

        return (combo.currentText() or DEFAULT_RERANK_MODEL).strip() or DEFAULT_RERANK_MODEL


    def _on_rerank_model_changed(self, *_):

        self._rerank_model_ready = False

        self._rerank_model_ready_name = ""
        self._rerank_model_verify_message = ""

        self._update_rerank_tooltip()

        self._update_rerank_status_ui()
        if getattr(self, "_rerank_available", False):
            self._verify_selected_rerank_model_cache()


    def _on_download_rerank_model(self):

        """Download and warm the selected Cross-Encoder model outside search."""

        model = self._selected_rerank_model()

        python_path = self._current_rerank_python_path()

        self.download_rerank_model_btn.setEnabled(False)

        self.download_rerank_model_btn.setText("Verifying...")

        if hasattr(self, "rerank_download_progress"):

            self.rerank_download_progress.setValue(0)

            self.rerank_download_progress.show()

        if hasattr(self, "rerank_download_status_label"):

            self.rerank_download_status_label.setText("Checking local cache before downloading...")

            self.rerank_download_status_label.show()

        if hasattr(self, "rerank_status_label"):

            self.rerank_status_label.setText("Cross-Encoder package: Ready" if getattr(self, "_rerank_available", False) else "Cross-Encoder package: checking...")

        if hasattr(self, "rerank_model_status_label"):

            self.rerank_model_status_label.setText(f"Selected model: verifying cache for {model}")

        self._rerank_download_worker = RerankModelVerifyWorker(python_path, model)

        def _on_verify_done(success, checked_model, message):

            try:

                if checked_model != model or self._selected_rerank_model() != model:

                    self.download_rerank_model_btn.setEnabled(True)
                    self.download_rerank_model_btn.setText("Verify / Download selected model")
                    self._rerank_download_worker = None
                    return

                self._rerank_model_verify_message = message

                if success:

                    self._rerank_model_ready = True
                    self._rerank_model_ready_name = model
                    self._rerank_model_checking = False
                    self._rerank_model_checking_name = ""
                    self.download_rerank_model_btn.setEnabled(True)
                    self.download_rerank_model_btn.setText("Verify / Download selected model")
                    self._rerank_download_worker = None
                    if hasattr(self, "rerank_download_progress"):
                        self.rerank_download_progress.setValue(100)
                    if hasattr(self, "rerank_download_status_label"):
                        self.rerank_download_status_label.setText(message)
                    self._update_rerank_tooltip()
                    self._update_rerank_status_ui()
                    showInfo(message)
                    return

                if hasattr(self, "rerank_download_status_label"):
                    self.rerank_download_status_label.setText("Local cache is incomplete or unavailable. Downloading missing model files...")

                self._start_rerank_model_download(model, python_path)

            except Exception as exc:

                log_debug(f"Rerank pre-download verification error: {exc}")
                self.download_rerank_model_btn.setEnabled(True)
                self.download_rerank_model_btn.setText("Verify / Download selected model")

        self._rerank_download_worker.finished_signal.connect(_on_verify_done)
        self._rerank_download_worker.start()


    def _start_rerank_model_download(self, model, python_path):

        self.download_rerank_model_btn.setText("Downloading...")

        if hasattr(self, "rerank_model_status_label"):

            self.rerank_model_status_label.setText(f"Selected model: downloading {model}")

        self._rerank_download_worker = RerankModelDownloadWorker(python_path, model)

        def _on_progress(percent, message):

            try:

                if hasattr(self, "rerank_download_progress"):

                    self.rerank_download_progress.setValue(max(0, min(100, int(percent))))

                if hasattr(self, "rerank_download_status_label"):

                    self.rerank_download_status_label.setText(message)

                if hasattr(self, "rerank_status_label"):

                    self.rerank_status_label.setText("Cross-Encoder package: Ready" if getattr(self, "_rerank_available", False) else "Cross-Encoder package: checking...")

                if hasattr(self, "rerank_model_status_label"):

                    self.rerank_model_status_label.setText(f"Selected model: {message}")

            except Exception as exc:

                log_debug(f"Rerank download progress error: {exc}")

        def _on_done(success, message):

            try:

                self.download_rerank_model_btn.setEnabled(True)

                self.download_rerank_model_btn.setText("Verify / Download selected model")

                self._rerank_download_worker = None

                if hasattr(self, "rerank_download_progress"):

                    self.rerank_download_progress.setValue(100 if success else 0)

                if hasattr(self, "rerank_download_status_label"):

                    self.rerank_download_status_label.setText(message)

                if success:

                    self._rerank_model_ready = True

                    self._rerank_model_ready_name = model
                    self._rerank_model_checking = False
                    self._rerank_model_checking_name = ""
                    self._rerank_model_verify_message = message

                    self._rerank_available = self._check_rerank_available(python_path=python_path)

                    self.enable_rerank_cb.setEnabled(self._rerank_available)

                    self._update_rerank_tooltip()

                    self._update_rerank_status_ui()

                    showInfo(message)

                else:

                    self._rerank_model_ready = False

                    self._rerank_model_ready_name = ""
                    self._rerank_model_checking = False
                    self._rerank_model_checking_name = ""
                    self._rerank_model_verify_message = message

                    self._update_rerank_status_ui()

                    showInfo(f"Could not download/warm model:\n\n{message}")

            except Exception as exc:

                log_debug(f"Rerank download completion error: {exc}")

        self._rerank_download_worker.progress_signal.connect(_on_progress)

        self._rerank_download_worker.finished_signal.connect(_on_done)

        self._rerank_download_worker.start()


