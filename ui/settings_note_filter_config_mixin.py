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
from aqt.utils import showInfo, tooltip

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

class SettingsNoteFilterConfigMixin:
    def _apply_note_type_config(self, ntf):



        """Apply note_type_filter config. Migrate fields_to_search -> note_type_fields if needed."""



        self._applying_note_type_config = True



        try:



            # Check if this is the new scope format
            if 'scope_mode' in ntf:
                self._apply_scope_config(ntf)
            else:
                self._apply_note_type_config_impl(ntf)



        finally:



            self._applying_note_type_config = False


    def _apply_note_type_config_impl(self, ntf):



        # Migrate: if fields_to_search exists but not note_type_fields, build note_type_fields



        ntf = dict(ntf)



        if ntf.get('fields_to_search') and not ntf.get('note_type_fields'):



            global_flds = set(f.lower() for f in ntf['fields_to_search'])



            ntf['note_type_fields'] = {}



            for model_name, _c, field_names in get_models_with_fields():



                ntf['note_type_fields'][model_name] = [f for f in field_names if f.lower() in global_flds]



        # Note types



        enabled = ntf.get('enabled_note_types')

        # None is the legacy "include all" value; represent it by checking all
        # current note types instead of showing a separate include-all checkbox.
        enabled_set = None if enabled is None else set(enabled or [])

        for i in range(self.note_types_table.rowCount()):



            it = self.note_types_table.item(i, 0)



            if it:



                name = it.data(Qt.ItemDataRole.UserRole) or it.text().strip()



                it.setCheckState(
                    Qt.CheckState.Checked
                    if (enabled_set is None or name in enabled_set)
                    else Qt.CheckState.Unchecked
                )



        # Search all / use first field



        self.search_all_fields_cb.setChecked(bool(ntf.get('search_all_fields', False)))



        self._on_search_all_fields_toggled()



        self.use_first_field_cb.setChecked(bool(ntf.get('use_first_field_fallback', True)))



        # Fields by note type (default to Text+Extra when neither note_type_fields nor fields_to_search)



        ntf_fields = ntf.get('note_type_fields') or {}



        default_flds = None



        if not ntf_fields and not ntf.get('fields_to_search'):



            default_flds = {'text', 'extra'}



        for model_name, cbs in self._field_cbs.items():



            wanted = set(f.lower() for f in (ntf_fields.get(model_name) or []))



            if not wanted and default_flds:



                wanted = default_flds



            for fn, cb in cbs.items():



                cb.setChecked(fn.lower() in wanted)



        # Decks (block signals so programmatic setCheckState doesn't trigger persist)



        deck_list = ntf.get('enabled_decks')



        # Interpretation:



        #   None        -> include all decks



        #   [] (empty)  -> no decks selected (all unchecked)



        #   [names]     -> only these decks checked



        include_all_d = (deck_list is None)



        self.include_all_decks_cb.blockSignals(True)



        self.include_all_decks_cb.setChecked(include_all_d)



        self.include_all_decks_cb.blockSignals(False)



        self._on_include_all_decks_toggled()



        if hasattr(self, 'decks_list') and self.decks_list:



            self.decks_list.blockSignals(True)



        try:



            if include_all_d:



                self._set_decks_checked(True)



            elif deck_list:



                ds = set(deck_list)



                for it in self._iterate_all_deck_items():



                    deck_name = it.data(0, Qt.ItemDataRole.UserRole) or it.text(0)



                    it.setCheckState(0, Qt.CheckState.Checked if (deck_name in ds) else Qt.CheckState.Unchecked)



            else:



                # Empty list: user chose no decks (all unchecked)



                self._set_decks_checked(False)



        finally:



            if hasattr(self, 'decks_list') and self.decks_list:



                self.decks_list.blockSignals(False)



        self._update_field_groups_enabled()


    def _update_field_groups_enabled(self):



        """Grey out and collapse field sections whose note type is unchecked."""



        if not getattr(self, '_field_groupboxes', None):



            return



        included = set()



        for i in range(self.note_types_table.rowCount()):



            it = self.note_types_table.item(i, 0)



            if it and it.checkState() == Qt.CheckState.Checked:



                name = it.data(Qt.ItemDataRole.UserRole) or it.text().strip()



                included.add(name)



        for model_name, gb in self._field_groupboxes.items():



            is_included = model_name in included

            gb.setEnabled(is_included)

            if hasattr(gb, 'setExpanded'):

                gb.setExpanded(is_included)


    def _note_type_order_from_table(self):
        """Return note type names in the table's current visual sort order."""
        table = getattr(self, 'note_types_table', None)
        if table is None:
            return []

        names = []
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item is None:
                continue
            name = item.data(Qt.ItemDataRole.UserRole) or item.text().strip()
            if name:
                names.append(name)
        return names


    def _sync_field_groups_to_note_type_order(self):
        """Keep field sections in the same order as the note types table."""
        layout = getattr(self, 'fields_by_note_type_layout', None)
        groupboxes = getattr(self, '_field_groupboxes', None)
        if layout is None or not groupboxes:
            return

        ordered_names = self._note_type_order_from_table()
        if not ordered_names:
            return

        while layout.count():
            layout.takeAt(0)

        added = set()
        for model_name in ordered_names:
            gb = groupboxes.get(model_name)
            if gb is None:
                continue
            layout.addWidget(gb)
            added.add(model_name)

        for model_name, gb in groupboxes.items():
            if model_name not in added:
                layout.addWidget(gb)


    def _on_note_type_item_changed(self, item):
        """Persist note type checkbox changes immediately."""
        self._update_field_groups_enabled()
        if item and item.column() == 0:
            self._persist_note_type_filter()


    def _on_sort_note_types_changed(self, index):



        """Handle sort combo box change."""



        data = self.sort_combo.itemData(index)



        if data == "count_desc":



            self.note_types_table.sortByColumn(1, Qt.SortOrder.DescendingOrder)



        elif data == "count_asc":



            self.note_types_table.sortByColumn(1, Qt.SortOrder.AscendingOrder)



        elif data == "name_asc":



            self.note_types_table.sortByColumn(0, Qt.SortOrder.AscendingOrder)



        elif data == "name_desc":



            self.note_types_table.sortByColumn(0, Qt.SortOrder.DescendingOrder)



        self._sync_field_groups_to_note_type_order()


    def _on_search_all_fields_toggled(self):
        en = not self.search_all_fields_cb.isChecked()
        self.fields_by_note_type_scroll.setEnabled(en)
        for cbs in self._field_cbs.values():
            for cb in cbs.values():
                cb.setEnabled(en)
        self._update_field_groups_enabled()
        self._persist_note_type_filter()


    def _on_include_all_decks_toggled(self):



        self.decks_list.setEnabled(not self.include_all_decks_cb.isChecked())



        # Also disable/enable header if needed



        header = self.decks_list.header()



        if header:



            header.setEnabled(not self.include_all_decks_cb.isChecked())



        self._persist_note_type_filter()


    def _persist_note_type_filter(self):



        """Save current Note Types & Fields (decks, note types, fields) to config so changes persist without clicking Save."""



        if getattr(self, '_applying_note_type_config', False):



            return



        try:



            config = load_config()



            config['note_type_filter'] = self._build_ntf_from_ui()



            save_config(config)



        except Exception as e:



            log_debug(f"Error persisting note_type_filter: {e}")


    def _on_deck_item_changed(self, item, column):



        """When user toggles a deck checkbox, persist so settings are saved."""



        if column == 0 and item and item.flags() & Qt.ItemFlag.ItemIsUserCheckable:



            self._persist_note_type_filter()


    def _set_note_types_checked(self, checked, manual=False):


        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked



        for i in range(self.note_types_table.rowCount()):



            it = self.note_types_table.item(i, 0)



            if it:



                it.setCheckState(state)

        if manual:

            self._persist_note_type_filter()


    def _set_decks_checked(self, checked):



        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked



        for item in self._iterate_all_deck_items():



            item.setCheckState(0, state)


    def _get_note_type_fields_from_ui(self):



        out = {}



        for model_name, cbs in self._field_cbs.items():



            sel = [fn for fn, cb in cbs.items() if cb.isChecked()]



            if sel:



                out[model_name] = sel



        return out


    def _get_decks_from_ui(self):



        if self.include_all_decks_cb.isChecked():



            return None



        # Get checked deck names from tree widget



        checked_decks = []



        for item in self._iterate_all_deck_items():



            if item.checkState(0) == Qt.CheckState.Checked:



                # Get full deck name from item data



                deck_name = item.data(0, Qt.ItemDataRole.UserRole)



                if deck_name:



                    checked_decks.append(deck_name)



        return checked_decks


    def _build_ntf_from_ui(self):



        enabled_nt = [self.note_types_table.item(i, 0).data(Qt.ItemDataRole.UserRole) or self.note_types_table.item(i, 0).text().strip() for i in range(self.note_types_table.rowCount()) if self.note_types_table.item(i, 0) and self.note_types_table.item(i, 0).checkState() == Qt.CheckState.Checked]



        # Preserve enabled_decks from config if deck list not yet populated (async load at 150ms)



        if hasattr(self, 'decks_list') and self.decks_list and self.decks_list.topLevelItemCount() > 0:



            enabled_decks = self._get_decks_from_ui()



        else:



            enabled_decks = load_config().get('note_type_filter', {}).get('enabled_decks')



        return {



            'enabled_note_types': enabled_nt,



            'search_all_fields': self.search_all_fields_cb.isChecked(),



            'note_type_fields': self._get_note_type_fields_from_ui() if not self.search_all_fields_cb.isChecked() else {},



            'use_first_field_fallback': self.use_first_field_cb.isChecked(),



            'enabled_decks': enabled_decks,



        }


    def _on_count_notes(self):



        ntf = self._build_ntf_from_ui()



        audit = analyze_note_eligibility(ntf)
        eligible_ids = set(audit.get("eligible_note_ids") or [])
        eligible_count = audit.get("eligible_count", 0)
        excluded_count = max(0, audit.get("total_notes", 0) - eligible_count)

        config = load_config()
        config["note_type_filter"] = ntf
        config = self._config_with_current_answer_provider(config)
        sc = dict(config.get("search_config") or {})
        sc.update(self._save_embedding_settings())
        config["search_config"] = sc
        current_engine_id = get_embedding_engine_id(config)

        db_path = get_embeddings_db_path()
        current_rows = 0
        current_distinct = 0
        additional_chunk_rows = 0
        multi_chunk_notes = 0
        indexed_eligible = 0
        indexed_outside_filters = 0
        missing_count = eligible_count
        engine_lines = []
        db_status = "No embeddings database found."

        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path, timeout=10)
                try:
                    db_status = f"Database: {os.path.basename(db_path)}"
                    per_note_counts = {}
                    for note_id, row_count in conn.execute(
                        "SELECT note_id, COUNT(*) FROM embeddings WHERE engine_id = ? GROUP BY note_id",
                        (current_engine_id,),
                    ):
                        per_note_counts[int(note_id)] = int(row_count)
                    current_rows = sum(per_note_counts.values())
                    current_distinct = len(per_note_counts)
                    additional_chunk_rows = sum(max(0, count - 1) for count in per_note_counts.values())
                    multi_chunk_notes = sum(1 for count in per_note_counts.values() if count > 1)
                    indexed_eligible = len(eligible_ids.intersection(per_note_counts.keys()))
                    indexed_outside_filters = max(0, current_distinct - indexed_eligible)
                    missing_count = max(0, eligible_count - indexed_eligible)
                    for engine_id, rows, notes in conn.execute(
                        "SELECT engine_id, COUNT(*), COUNT(DISTINCT note_id) FROM embeddings GROUP BY engine_id ORDER BY COUNT(*) DESC"
                    ):
                        engine_lines.append(f"- {engine_id}: {rows:,} rows / {notes:,} notes")
                finally:
                    conn.close()
            except Exception as exc:
                db_status = f"Could not read embeddings database: {exc}"

        report = [
            "Embedding Coverage Preview",
            "",
            "Current filters",
            f"- Eligible notes: {eligible_count:,}",
            f"- Excluded notes: {excluded_count:,}",
            f"- Total notes in selected decks: {audit.get('total_notes', 0):,}",
            "",
            "Current embedding engine",
            f"- {current_engine_id}",
            f"- Indexed eligible notes: {indexed_eligible:,} / {eligible_count:,}",
            f"- Missing eligible notes: {missing_count:,}",
            f"- Additional chunk rows: {additional_chunk_rows:,}",
            f"- Notes split into multiple chunks: {multi_chunk_notes:,}",
            f"- Indexed notes outside current filters: {indexed_outside_filters:,}",
            f"- Raw rows for this engine: {current_rows:,}",
            f"- Distinct notes for this engine: {current_distinct:,}",
            "",
            db_status,
        ]
        if engine_lines:
            report.extend(["", "All engines in database", *engine_lines])

        self._show_embedding_coverage_preview("\n".join(report), excluded_count)


    def _show_embedding_coverage_preview(self, report_text, excluded_count):



        dlg = QDialog(self)


        dlg.setWindowTitle("Eligible Notes & Embedding Coverage")


        dlg.resize(520, 430)


        layout = QVBoxLayout(dlg)


        report = QPlainTextEdit()


        report.setReadOnly(True)


        report.setPlainText(report_text)


        try:


            report.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)


        except AttributeError:


            report.setLineWrapMode(QPlainTextEdit.NoWrap)


        layout.addWidget(report)


        buttons = QHBoxLayout()


        review_btn = QPushButton(f"Review excluded notes ({excluded_count:,})")


        review_btn.setEnabled(excluded_count > 0)


        review_btn.setToolTip("Open notes excluded by the current deck, note type, and field filters.")


        try:


            review_btn.setStyleSheet(settings_button_style(get_addon_theme(), "muted"))


        except Exception:


            pass


        def review_excluded_from_preview():


            dlg.accept()


            QTimer.singleShot(0, self._review_ineligible_notes)


        review_btn.clicked.connect(review_excluded_from_preview)


        buttons.addWidget(review_btn)


        buttons.addStretch()


        close_btn = QPushButton("Close")


        close_btn.clicked.connect(dlg.accept)


        buttons.addWidget(close_btn)


        layout.addLayout(buttons)


        dlg.exec()


    def _refresh_note_type_lists(self):



        """Repopulate all lists and preserve checked state where possible."""

        self._note_type_lists_loaded = True
        self._invalidate_note_type_deck_cache()
        self._set_note_type_loading_status(
            "Refreshing note types, fields, and deck counts. Deck counts can take a while on large collections.",
            busy=True,
        )



        checked_nt = [self.note_types_table.item(i, 0).data(Qt.ItemDataRole.UserRole) or self.note_types_table.item(i, 0).text().strip() for i in range(self.note_types_table.rowCount()) if self.note_types_table.item(i, 0) and self.note_types_table.item(i, 0).checkState() == Qt.CheckState.Checked]



        ntf_prev = self._get_note_type_fields_from_ui()



        checked_decks = self._get_decks_from_ui()



        self._populate_note_type_lists()



        for i in range(self.note_types_table.rowCount()):



            it = self.note_types_table.item(i, 0)



            if it:



                name = it.data(Qt.ItemDataRole.UserRole) or it.text().strip()



                if name in checked_nt:



                    it.setCheckState(Qt.CheckState.Checked)



        self._populate_fields_by_note_type()



        for model_name, cbs in self._field_cbs.items():



            wanted = set(f.lower() for f in (ntf_prev.get(model_name) or []))



            for fn, cb in cbs.items():



                if fn.lower() in wanted:



                    cb.setChecked(True)



        self._populate_decks_list(force_refresh=True)



        if checked_decks:



            ds = set(checked_decks)



            for it in self._iterate_all_deck_items():



                deck_name = it.data(0, Qt.ItemDataRole.UserRole) or it.text(0)



                it.setCheckState(0, Qt.CheckState.Checked if (deck_name in ds) else Qt.CheckState.Unchecked)



        else:



            self._set_decks_checked(True)



        self._refresh_preset_combos()

        self._set_note_type_loading_status("Note type and deck data refreshed.", busy=False)
        QTimer.singleShot(3500, lambda: self._set_note_type_loading_status("", busy=False))



        showInfo("Lists refreshed.")


