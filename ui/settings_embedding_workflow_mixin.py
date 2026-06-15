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
from .settings_embedding_connection_mixin import count_indexed_eligible_notes_for_engine
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
    get_models_with_fields,
    get_notes_count_per_deck,
    get_notes_count_per_model,
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


class EmbeddingPreflightWorker(QThread):
    result_ready = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, ntf, runtime_config):
        super().__init__()
        self.ntf = ntf
        self.runtime_config = runtime_config

    def run(self):
        try:
            started = time.time()
            eligibility = analyze_note_eligibility(self.ntf)
            note_count = eligibility.get("eligible_count", 0)
            checkpoint = load_checkpoint()
            resume_available = False
            current_engine_id = get_embedding_engine_id(self.runtime_config)
            current_scope_id = make_embedding_scope_id(self.ntf)
            eligible_ids = set(eligibility.get("eligible_note_ids") or [])
            indexed_eligible = count_indexed_eligible_notes_for_engine(current_engine_id, eligible_ids)

            if checkpoint and checkpoint.get("engine_id") != current_engine_id:
                checkpoint = None
            if checkpoint and checkpoint.get("scope_id") != current_scope_id:
                log_debug("Ignoring embedding checkpoint because the selected note/deck/field scope changed")
                clear_checkpoint()
                checkpoint = None
            if checkpoint and int(checkpoint.get("total_notes") or 0) != int(note_count):
                log_debug(
                    "Ignoring embedding checkpoint because the selected note scope changed: "
                    f"{checkpoint.get('total_notes')} checkpoint notes vs {note_count} current eligible notes"
                )
                clear_checkpoint()
                checkpoint = None
            if indexed_eligible >= note_count and checkpoint:
                log_debug("Ignoring embedding checkpoint because the current engine already covers all eligible notes")
                clear_checkpoint()
                checkpoint = None
            if checkpoint:
                processed_count = checkpoint.get("processed_count", 0)
                total_notes = checkpoint.get("total_notes", 0)
                resume_available = processed_count > 0 and processed_count < total_notes

            log_debug(
                f"Embedding preflight finished in {time.time() - started:.2f}s; "
                f"eligible={note_count}, indexed={indexed_eligible}"
            )
            self.result_ready.emit({
                "ntf": self.ntf,
                "runtime_config": self.runtime_config,
                "eligibility": eligibility,
                "note_count": note_count,
                "checkpoint": checkpoint,
                "resume_available": resume_available,
            })
        except Exception as exc:
            log_debug(f"Embedding preflight failed: {exc}", is_error=True)
            self.error_signal.emit(str(exc))


class SettingsEmbeddingWorkflowMixin:
    def _create_or_update_embeddings(self):



        """Create or update embeddings for all notes using the selected engine (Voyage, OpenAI, Cohere, or Ollama)."""



        # Persist current UI engine/URL/model so worker uses them (user may have changed without saving dialog)



        config = load_config()



        sc = dict(config.get('search_config') or {})

        # explanation: persist the new API-tab embedding settings before the worker starts.
        sc.update(self._save_embedding_settings())

        config['search_config'] = sc

        runtime_config = self._config_with_current_answer_provider(config)

        valid, validation_message = validate_embedding_config(runtime_config)

        if not valid:

            showInfo(validation_message)

            return

        save_config(config)  # persist embedding settings

        effective_config = get_effective_embedding_config(runtime_config)

        sc = effective_config.get('search_config') or {}

        engine = sc.get('embedding_engine') or 'voyage'
        log_debug(
            "Create/Update Embeddings: deferring provider connection test to the "
            "embedding worker so the button click does not block the UI."
        )







        # Get note type filter config



        # Always base this on the *current* UI selections so user choices



        # (note types, decks, fields) are remembered between sessions,



        # even if they didn't click the main "Save Settings" button.



        config = load_config()



        current_ntf = self._build_ntf_from_ui()
        scope_empty = (
            not current_ntf.get("enabled_decks")
            and not current_ntf.get("enabled_note_types")
            and not current_ntf.get("scope_fields")
        )
        if current_ntf.get('enabled_note_types') == [] or scope_empty:
            restored_ntf = self._resolve_note_type_filter_from_config(config, persist=True)
            if restored_ntf.get('enabled_note_types'):
                current_ntf = restored_ntf
                try:
                    if getattr(self, "_scope_lists_loaded", False):
                        self._apply_scope_config(current_ntf)
                except Exception as exc:
                    log_debug(f"Could not re-apply restored preset before embedding: {exc}")



        config['note_type_filter'] = current_ntf



        # Persist immediately so next Anki restart / addon open uses the same



        # note/deck/field selection.



        save_config(config)



        ntf = current_ntf

        runtime_config['note_type_filter'] = current_ntf

        if hasattr(self, 'create_embedding_btn') and self.create_embedding_btn:
            self.create_embedding_btn.setEnabled(False)
            self.create_embedding_btn.setText("Preparing...")
        QApplication.processEvents()
        preflight_worker = EmbeddingPreflightWorker(current_ntf, runtime_config)
        preflight_worker.result_ready.connect(self._on_embedding_preflight_ready)
        preflight_worker.error_signal.connect(self._on_embedding_preflight_error)
        preflight_worker.finished.connect(lambda: setattr(self, "_embedding_preflight_worker", None))
        self._embedding_preflight_worker = preflight_worker
        preflight_worker.start()
        return







        # Count notes that will be processed



        eligibility = analyze_note_eligibility(ntf)

        note_count = eligibility.get('eligible_count', 0)



        if note_count == 0:



            showInfo("No notes found to process. Check your note type and deck filters.")



            return







        # Check for existing checkpoint (only resume if it was for the same embedding engine)



        checkpoint = load_checkpoint()



        resume_available = False



        current_engine_id = get_embedding_engine_id(runtime_config)
        current_scope_id = make_embedding_scope_id(current_ntf)
        eligible_ids = set(eligibility.get("eligible_note_ids") or [])
        indexed_eligible = self._count_indexed_eligible_notes(current_engine_id, eligible_ids)



        if checkpoint and checkpoint.get('engine_id') != current_engine_id:



            checkpoint = None  # different engine: start fresh, don't offer resume



        if checkpoint and checkpoint.get('scope_id') != current_scope_id:



            log_debug(
                "Ignoring embedding checkpoint because the selected note/deck/field scope changed"
            )



            clear_checkpoint()



            checkpoint = None



        if checkpoint and int(checkpoint.get('total_notes') or 0) != int(note_count):



            log_debug(
                "Ignoring embedding checkpoint because the selected note scope changed: "
                f"{checkpoint.get('total_notes')} checkpoint notes vs {note_count} current eligible notes"
            )



            clear_checkpoint()



            checkpoint = None



        if indexed_eligible >= note_count:



            if checkpoint:



                log_debug(
                    "Ignoring embedding checkpoint because the current engine already covers all eligible notes"
                )



                clear_checkpoint()



                checkpoint = None


        if checkpoint:



            processed_count = checkpoint.get('processed_count', 0)



            total_notes = checkpoint.get('total_notes', 0)



            if processed_count > 0 and processed_count < total_notes:



                resume_available = True



                reply = QMessageBox.question(



                    self,



                    "Resume Embedding Generation?",



                    f"Found a previous checkpoint:\n\n"



                    f"Processed: {processed_count:,} / {total_notes:,} notes\n"



                    f"Timestamp: {checkpoint.get('timestamp', 'unknown')}\n\n"



                    f"Would you like to resume from where you left off?\n\n"



                    f"(Click 'No' to start over)",



                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,



                    QMessageBox.StandardButton.Yes



                )







                if reply == QMessageBox.StandardButton.Cancel:



                    return



                elif reply == QMessageBox.StandardButton.No:



                    # Clear checkpoint and start fresh



                    clear_checkpoint()



                    checkpoint = None



                    resume_available = False







        if not resume_available:



            reply = QMessageBox.question(



                self,



                "Create/Update Embeddings",



                f"This will scan approximately {note_count:,} notes and update only changed or missing embeddings.\n\n"

                f"Currently excluded by filters: {len(eligibility.get('ineligible_notes', [])):,} notes.\n\n"



                f"Unchanged notes should pass quickly.\n\n"



                f"Continue?",



                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,



                QMessageBox.StandardButton.Yes



            )







            if reply != QMessageBox.StandardButton.Yes:



                return







        # Create progress dialog (non-modal so Anki stays responsive)



        progress_dialog = QDialog(self)



        progress_dialog.setWindowTitle("Creating Embeddings")



        progress_dialog.setMinimumWidth(500)



        progress_dialog.setMinimumHeight(350)



        progress_dialog.setModal(False)  # Non-modal so user can continue using Anki



        # Add minimize and maximize buttons



        flags = progress_dialog.windowFlags()



        progress_dialog.setWindowFlags(flags | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)



        progress_layout = QVBoxLayout(progress_dialog)







        # Track pause state



        progress_dialog._is_paused = False



        progress_dialog._pause_lock = False







        status_label = QLabel("Checking notes for embedding updates...")



        status_label.setWordWrap(True)



        progress_layout.addWidget(status_label)







        progress_bar = QProgressBar()



        progress_bar.setRange(0, note_count)



        progress_bar.setValue(0)



        progress_bar.setTextVisible(True)



        progress_bar.setFormat("%p%")



        progress_layout.addWidget(progress_bar)



        detail_label = QLabel("Checked 0/{:,} | Updated 0 | New 0 | Already current 0 | Pending 0 | ETA calculating...".format(note_count))

        detail_label.setWordWrap(True)

        detail_label.setStyleSheet(settings_text_style(_addon_theme(), "subtle"))

        progress_layout.addWidget(detail_label)







        log_text = QTextEdit()



        log_text.setReadOnly(True)



        log_text.setMaximumHeight(200)



        log_text.setFont(QFont("Courier", 9))



        progress_layout.addWidget(log_text)







        # Control buttons



        button_layout = QHBoxLayout()







        pause_button = QPushButton("Pause")



        pause_button.clicked.connect(lambda: self._toggle_pause(progress_dialog, pause_button, log_text))



        button_layout.addWidget(pause_button)







        button_layout.addStretch()







        close_button = QPushButton("Close")



        close_button.setVisible(False)



        close_button.clicked.connect(progress_dialog.close)



        button_layout.addWidget(close_button)







        progress_layout.addLayout(button_layout)







        # Store references for worker thread



        progress_dialog._status_label = status_label



        progress_dialog._progress_bar = progress_bar



        progress_dialog._detail_label = detail_label



        progress_dialog._log_text = log_text



        progress_dialog._close_button = close_button



        progress_dialog._pause_button = pause_button







        progress_dialog.show()



        QApplication.processEvents()







        # Create and start worker thread for embedding (prevents blocking)



        worker = EmbeddingWorker(



            ntf, note_count, checkpoint, resume_available, config=runtime_config



        )







        # Connect worker signals to UI updates



        worker.status_update.connect(status_label.setText)



        worker.progress_update.connect(progress_bar.setValue)



        worker.progress_detail.connect(lambda detail: self._on_embedding_progress_detail(progress_dialog, detail))



        worker.log_message.connect(log_text.append)



        worker.finished_signal.connect(lambda processed, errors, skipped, refreshed, still_failed: self._on_embedding_finished(



            progress_dialog, processed, errors, skipped, refreshed, still_failed, note_count



        ))



        worker.error_signal.connect(lambda msg: self._on_embedding_error(progress_dialog, msg))







        # Store worker reference



        progress_dialog._worker = worker







        # Start worker thread



        worker.start()


    def _reset_embedding_preflight_button(self):
        if hasattr(self, 'create_embedding_btn') and self.create_embedding_btn:
            self.create_embedding_btn.setEnabled(True)
            self.create_embedding_btn.setText("Create/Update Embeddings")


    def _on_embedding_preflight_error(self, error_msg):
        self._reset_embedding_preflight_button()
        showInfo(f"Could not prepare embeddings.\n\nError: {error_msg}")


    def _on_embedding_preflight_ready(self, result):
        self._reset_embedding_preflight_button()
        ntf = result.get("ntf") or {}
        runtime_config = result.get("runtime_config") or {}
        eligibility = result.get("eligibility") or {}
        note_count = int(result.get("note_count") or 0)
        checkpoint = result.get("checkpoint")
        resume_available = bool(result.get("resume_available"))

        if note_count == 0:
            showInfo("No notes found to process. Check your note type and deck filters.")
            return

        if resume_available and checkpoint:
            processed_count = checkpoint.get('processed_count', 0)
            total_notes = checkpoint.get('total_notes', 0)
            reply = QMessageBox.question(
                self,
                "Resume Embedding Generation?",
                f"Found a previous checkpoint:\n\n"
                f"Processed: {processed_count:,} / {total_notes:,} notes\n"
                f"Timestamp: {checkpoint.get('timestamp', 'unknown')}\n\n"
                f"Would you like to resume from where you left off?\n\n"
                f"(Click 'No' to start over)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.No:
                clear_checkpoint()
                checkpoint = None
                resume_available = False

        if not resume_available:
            reply = QMessageBox.question(
                self,
                "Create/Update Embeddings",
                f"This will scan approximately {note_count:,} notes and update only changed or missing embeddings.\n\n"
                f"Currently excluded by filters: {len(eligibility.get('ineligible_notes', [])):,} notes.\n\n"
                f"Unchanged notes should pass quickly.\n\n"
                f"Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._start_embedding_worker(ntf, note_count, checkpoint, resume_available, runtime_config)


    def _start_embedding_worker(self, ntf, note_count, checkpoint, resume_available, runtime_config):
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Creating Embeddings")
        progress_dialog.setMinimumWidth(500)
        progress_dialog.setMinimumHeight(350)
        progress_dialog.setModal(False)
        flags = progress_dialog.windowFlags()
        progress_dialog.setWindowFlags(flags | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)
        progress_layout = QVBoxLayout(progress_dialog)

        progress_dialog._is_paused = False
        progress_dialog._pause_lock = False

        status_label = QLabel("Checking notes for embedding updates...")
        status_label.setWordWrap(True)
        progress_layout.addWidget(status_label)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, note_count)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(True)
        progress_bar.setFormat("%p%")
        progress_layout.addWidget(progress_bar)

        detail_label = QLabel("Checked 0/{:,} | Updated 0 | New 0 | Already current 0 | Pending 0 | ETA calculating...".format(note_count))
        detail_label.setWordWrap(True)
        detail_label.setStyleSheet(settings_text_style(_addon_theme(), "subtle"))
        progress_layout.addWidget(detail_label)

        log_text = QTextEdit()
        log_text.setReadOnly(True)
        log_text.setMaximumHeight(200)
        log_text.setFont(QFont("Courier", 9))
        progress_layout.addWidget(log_text)

        button_layout = QHBoxLayout()
        pause_button = QPushButton("Pause")
        pause_button.clicked.connect(lambda: self._toggle_pause(progress_dialog, pause_button, log_text))
        button_layout.addWidget(pause_button)
        button_layout.addStretch()
        close_button = QPushButton("Close")
        close_button.setVisible(False)
        close_button.clicked.connect(progress_dialog.close)
        button_layout.addWidget(close_button)
        progress_layout.addLayout(button_layout)

        progress_dialog._status_label = status_label
        progress_dialog._progress_bar = progress_bar
        progress_dialog._detail_label = detail_label
        progress_dialog._log_text = log_text
        progress_dialog._close_button = close_button
        progress_dialog._pause_button = pause_button
        progress_dialog.show()
        QApplication.processEvents()

        worker = EmbeddingWorker(ntf, note_count, checkpoint, resume_available, config=runtime_config)
        worker.status_update.connect(status_label.setText)
        worker.progress_update.connect(progress_bar.setValue)
        worker.progress_detail.connect(lambda detail: self._on_embedding_progress_detail(progress_dialog, detail))
        worker.log_message.connect(log_text.append)
        worker.finished_signal.connect(lambda processed, errors, skipped, refreshed, still_failed: self._on_embedding_finished(
            progress_dialog, processed, errors, skipped, refreshed, still_failed, note_count
        ))
        worker.error_signal.connect(lambda msg: self._on_embedding_error(progress_dialog, msg))
        progress_dialog._worker = worker
        worker.start()


    def _bring_browser_to_front(self, browser):


        """Raise the Anki Browser after opening an audit result set."""


        try:


            if browser:


                browser.activateWindow()


                browser.raise_()


        except Exception:


            pass


    def _review_ineligible_notes(self):


        ntf = self._build_ntf_from_ui()


        audit = analyze_note_eligibility(ntf)


        ineligible = audit.get("ineligible_notes", [])


        if not ineligible:


            showInfo(
                "No excluded notes found in the current deck/type scope.\n\n"
                f"Eligible notes: {audit.get('eligible_count', 0):,}"
            )


            return


        reason_lines = [
            f"Eligible notes: {audit.get('eligible_count', 0):,}",
            f"Excluded notes: {len(ineligible):,}",
            f"- Wrong note type: {audit.get('filtered_out_note_type_count', 0):,}",
            f"- No embedding fields selected: {audit.get('no_selected_fields_count', 0):,}",
            f"- Selected fields empty: {audit.get('empty_selected_fields_count', 0):,}",
            "",
            "First excluded notes:",
        ]


        preview = ineligible[:100]


        for note in preview:


            fields = ", ".join(note.get("field_names") or []) or "(none)"


            model_name = note.get("model_name") or "(not in selected note types)"


            reason_lines.append(
                f"- nid:{note['id']} | {note['reason']} | note type: {model_name} | fields: {fields}"
            )


        if len(ineligible) > len(preview):


            reason_lines.append("")


            reason_lines.append(f"...and {len(ineligible) - len(preview):,} more.")


        note_ids = [str(note["id"]) for note in ineligible]


        browser_note_ids = note_ids[:1000]


        search_query = "nid:" + ",".join(browser_note_ids)


        try:


            browser = dialogs.open("Browser", mw)


            browser.form.searchEdit.lineEdit().setText(search_query)


            browser.onSearchActivated()


            QTimer.singleShot(150, lambda b=browser: self._bring_browser_to_front(b))


            tooltip(f"Opened {len(note_ids)} excluded notes in browser")


        except Exception as exc:


            log_debug(f"Could not open excluded notes in browser: {exc}")


        reason_lines.extend([
            "",
            f"Browser opened with first {len(browser_note_ids):,} excluded notes.",
            "",
            "Browser query:",
            search_query,
        ])


        showText("\n".join(reason_lines), title="Excluded Notes Audit")


