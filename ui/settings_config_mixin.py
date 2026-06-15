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
from ..core.memory import (
    clear_memory,
    delete_memory_snippets,
    list_memory_snippets,
    load_memory_profile,
    memory_profile_summary,
    prune_memory_snippets,
    rebuild_memory_embeddings,
)
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


class SettingsConfigMixin:
    def _show_memory_summary(self):
        try:
            summary = memory_profile_summary(load_memory_profile())
            lines = [
                f"Profile: {summary.get('profile_id')}",
                f"Created: {summary.get('created_at')}",
                f"Updated: {summary.get('updated_at')}",
                f"Stored keys: {', '.join(summary.get('keys') or []) or '(none)'}",
                f"Deck scopes: {summary.get('enabled_deck_count')}",
                f"Note types: {summary.get('enabled_note_type_count')}",
                f"Selected fields: {summary.get('selected_field_count')}",
                f"Planner mode: {summary.get('agentic_planner_mode') or '(none)'}",
                f"Smart retrieval remembered: {summary.get('enable_agentic_rag')}",
                f"Fact snippets: {summary.get('snippet_count', 0)}",
                f"Embedded snippets: {summary.get('embedded_snippet_count', 0)}",
                f"Snippet retention days: {summary.get('retention_days', 30)}",
                f"Oldest snippet: {summary.get('oldest_snippet_at') or '(none)'}",
                f"Newest snippet: {summary.get('newest_snippet_at') or '(none)'}",
                f"Snippet hits: {summary.get('recent_hit_count', 0)}",
                "",
                "Local Memory stores settings plus compact snippets from final answer context. It does not store full note text, answers, or raw chat history.",
            ]
            showInfo("\n".join(lines))
        except Exception as exc:
            showInfo(f"Could not read local memory:\n{exc}")

    def _clear_profile_memory(self):
        try:
            reply = QMessageBox.question(
                self,
                "Clear local memory",
                "Clear durable local memory, including profile settings and fact snippets?\n\n"
                "Session memory is cleared separately when chat closes.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            clear_memory()
            showInfo("Local Memory cleared.")
        except Exception as exc:
            showInfo(f"Could not clear local memory:\n{exc}")

    def _show_memory_inspector(self):
        try:
            dlg = QDialog(self)
            dlg.setWindowTitle("Local Memory Inspector")
            dlg.resize(980, 560)
            layout = QVBoxLayout(dlg)
            search_row = QHBoxLayout()
            search_input = QLineEdit()
            search_input.setPlaceholderText("Search memory snippets")
            search_btn = QPushButton("Search")
            refresh_btn = QPushButton("Refresh")
            prune_btn = QPushButton("Prune expired")
            rebuild_btn = QPushButton("Rebuild embeddings")
            delete_btn = QPushButton("Delete selected")
            search_row.addWidget(search_input, 1)
            for btn in (search_btn, refresh_btn, prune_btn, rebuild_btn, delete_btn):
                search_row.addWidget(btn)
            layout.addLayout(search_row)

            table = QTableWidget()
            table.setColumnCount(9)
            table.setHorizontalHeaderLabels([
                "ID", "Snippet", "Note ID", "Source query", "Subquery",
                "Created/seen", "Hits", "Answer use", "Memory vector",
            ])
            header_tooltips = {
                7: "Local memory snippets are retrieval hints. They are not cited as answer evidence.",
                8: "Ready means this memory snippet has an embedding for the current embedding engine.",
            }
            for col_idx, tip in header_tooltips.items():
                header_item = table.horizontalHeaderItem(col_idx)
                if header_item:
                    header_item.setToolTip(tip)
            table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
            table.horizontalHeader().setStretchLastSection(True)
            layout.addWidget(table, 1)

            def load_rows():
                rows = list_memory_snippets(
                    query=search_input.text().strip(),
                    limit=300,
                    config=load_config(),
                )
                table.setRowCount(len(rows))
                def answer_use_label(item):
                    return "Citable" if item.get("citable") else "Retrieval hint only"

                def memory_vector_label(status):
                    labels = {
                        "fresh": "Ready",
                        "missing": "Not embedded yet",
                        "stale": "Needs rebuild",
                        "not_checked": "Not checked",
                    }
                    return labels.get(str(status or "").strip().lower(), str(status or ""))

                def cell_tooltip(col_idx, item):
                    if col_idx == 7:
                        return (
                            "This saved memory can help future searches, but answers should cite the original Anki notes instead."
                            if not item.get("citable")
                            else "This row may be used as answer evidence."
                        )
                    if col_idx == 8:
                        status = str(item.get("embedding_status") or "").strip().lower()
                        if status == "fresh":
                            return "This snippet is embedded for the current memory embedding engine."
                        if status == "missing":
                            return "This snippet can still be found by text matching. Rebuild embeddings to add vector search."
                        if status == "stale":
                            return "This snippet has an older or mismatched vector. Rebuild embeddings to refresh it."
                        return "Embedding status was not checked for this snippet."
                    return ""

                for row_idx, item in enumerate(rows):
                    values = [
                        item.get("id"),
                        item.get("snippet"),
                        item.get("note_id"),
                        item.get("source_query"),
                        item.get("subquery_label"),
                        item.get("last_seen_at"),
                        item.get("hit_count"),
                        answer_use_label(item),
                        memory_vector_label(item.get("embedding_status")),
                    ]
                    for col_idx, value in enumerate(values):
                        cell = QTableWidgetItem(str(value if value is not None else ""))
                        tip = cell_tooltip(col_idx, item)
                        if tip:
                            cell.setToolTip(tip)
                        if col_idx == 0:
                            cell.setData(Qt.ItemDataRole.UserRole, item.get("id"))
                        table.setItem(row_idx, col_idx, cell)
                table.resizeColumnsToContents()

            def selected_ids():
                ids = []
                for index in table.selectionModel().selectedRows():
                    item = table.item(index.row(), 0)
                    if item:
                        ids.append(int(item.data(Qt.ItemDataRole.UserRole)))
                return ids

            def run_background(label, op_func, on_success):
                try:
                    from aqt.operations import QueryOp
                    QueryOp(parent=dlg, op=lambda _col: op_func(), success=on_success).run_in_background()
                except Exception:
                    on_success(op_func())

            def prune_action():
                config = load_config()
                days = int((config.get("search_config") or {}).get("memory_retention_days", 30) or 30)
                run_background(
                    "Prune memory",
                    lambda: prune_memory_snippets(retention_days=days),
                    lambda count: (tooltip(f"Pruned {count} snippets"), load_rows()),
                )

            def rebuild_action():
                run_background(
                    "Rebuild memory embeddings",
                    lambda: rebuild_memory_embeddings(config=load_config()),
                    lambda result: (tooltip(f"Embedded {dict(result or {}).get('embedded', 0)} snippets"), load_rows()),
                )

            def delete_action():
                ids = selected_ids()
                if not ids:
                    tooltip("Select memory snippets first")
                    return
                run_background(
                    "Delete memory snippets",
                    lambda: delete_memory_snippets(ids),
                    lambda count: (tooltip(f"Deleted {count} snippets"), load_rows()),
                )

            search_btn.clicked.connect(load_rows)
            refresh_btn.clicked.connect(load_rows)
            search_input.returnPressed.connect(load_rows)
            prune_btn.clicked.connect(prune_action)
            rebuild_btn.clicked.connect(rebuild_action)
            delete_btn.clicked.connect(delete_action)
            load_rows()
            dlg.exec()
        except Exception as exc:
            showInfo(f"Could not open local memory inspector:\n{exc}")

    def _resolve_note_type_filter_from_config(self, config, persist=False):
        """Use the selected preset when the active note filter was accidentally saved empty."""
        ntf = dict(config.get('note_type_filter') or {})
        enabled = ntf.get('enabled_note_types')
        current_name = config.get('current_preset_name')
        presets = config.get('saved_presets') or {}
        preset = presets.get(current_name) if current_name else None
        if enabled == [] and preset and preset.get('enabled_note_types'):
            ntf = dict(preset)
            config['note_type_filter'] = ntf
            log_debug(f"Restored note_type_filter from current preset '{current_name}' because active note types were empty")
            if persist:
                try:
                    save_config(config)
                except Exception as exc:
                    log_debug(f"Could not persist restored note_type_filter preset: {exc}")
        return ntf


    def _load_config_into_ui(self):

        """Atomic configuration loading with strict safety checks."""

        config = load_config()

        search_config = config.get('search_config', {})



        # Guarded setters

        self._safe_set_checked(getattr(self, "enable_query_expansion_cb", None), search_config.get('enable_query_expansion', False))
        self._safe_set_checked(getattr(self, "enable_agentic_rag_cb", None), search_config.get('enable_agentic_rag', False))
        self._safe_set_checked(getattr(self, "enable_profile_memory_cb", None), search_config.get('enable_profile_memory', True))
        if hasattr(self, "memory_retrieval_mode_combo"):
            idx = self.memory_retrieval_mode_combo.findData(search_config.get("memory_retrieval_mode", "auto_hybrid"))
            if idx >= 0:
                self.memory_retrieval_mode_combo.setCurrentIndex(idx)
        if hasattr(self, "memory_retention_days_spin"):
            self.memory_retention_days_spin.setValue(int(search_config.get("memory_retention_days", 30) or 30))
        if hasattr(self, "memory_max_saved_spin"):
            self.memory_max_saved_spin.setValue(int(search_config.get("memory_max_saved_snippets_per_search", 24) or 24))
        if hasattr(self, "memory_max_retrieved_spin"):
            self.memory_max_retrieved_spin.setValue(int(search_config.get("memory_max_retrieved_snippets", 5) or 5))
        self._safe_set_checked(getattr(self, "memory_embedding_enabled_cb", None), search_config.get("memory_embedding_enabled", True))
        if hasattr(self, "agentic_planner_mode_combo"):
            mode = search_config.get("agentic_planner_mode", "deterministic_v1")
            idx = self.agentic_planner_mode_combo.findData(mode)
            if idx >= 0:
                self.agentic_planner_mode_combo.setCurrentIndex(idx)

        self._safe_set_checked(getattr(self, "enable_hyde_cb", None), search_config.get('enable_hyde', False))

        self._safe_set_checked(getattr(self, "enable_rerank_cb", None), search_config.get('enable_rerank', False))

        self._safe_set_checked(getattr(self, "use_context_boost_cb", None), search_config.get('use_context_boost', True))
        self._safe_set_checked(getattr(self, "use_dynamic_batch_size_cb", None), search_config.get('use_dynamic_batch_size', True))

        rerank_config = get_rerank_config(search_config)



        # Value setters (with try/except guards)

        try:

            if hasattr(self, "max_results_spin"):

                self.max_results_spin.setValue(max(5, min(100, search_config.get('max_results', 50))))

            if hasattr(self, "hybrid_weight_spin"):

                self.hybrid_weight_spin.setValue(max(0, min(100, search_config.get('hybrid_embedding_weight', 40))))

            if hasattr(self, "rerank_model_combo"):

                model = rerank_config.get('rerank_model', DEFAULT_RERANK_MODEL)

                idx = self.rerank_model_combo.findData(model)

                if idx >= 0:

                    self.rerank_model_combo.setCurrentIndex(idx)

                else:

                    self.rerank_model_combo.setCurrentText(model)

            if hasattr(self, "rerank_python_path_input"):

                 path = (search_config.get('rerank_python_path') or '').strip()

                 self.rerank_python_path_input.setText(path)

                 if path and hasattr(self, "python_path_widget"):

                     self.python_path_widget.setVisible(True)

        except Exception:

            pass

        if hasattr(self, "_update_agentic_planner_controls"):
            self._update_agentic_planner_controls()

        try:

            retrieval = get_retrieval_config(search_config)
            retrieval_widgets = [
                "mmr_enabled_cb",
                "mmr_lambda_slider",
            ]

            for name in retrieval_widgets:

                w = getattr(self, name, None)

                if w is not None:

                    w.blockSignals(True)

            if hasattr(self, "mmr_enabled_cb"):

                self.mmr_enabled_cb.setChecked(bool(retrieval.get('enable_mmr_diversity', True)))

            if hasattr(self, "mmr_lambda_slider"):

                self.mmr_lambda_slider.setValue(max(0, min(100, int(round(float(retrieval.get('mmr_lambda', 0.75)) * 100)))))

            if hasattr(self, "_update_retrieval_v2_controls"):

                self._update_retrieval_v2_controls(mark_dirty=False)

            if hasattr(self, "retrieval_dirty_label"):

                self.retrieval_dirty_label.hide()
                if hasattr(self, "retrieval_dirty_row"):
                    self.retrieval_dirty_row.hide()

        except Exception as e:

            log_debug(f"Error loading Retrieval V2 settings: {e}", is_error=True)

        finally:

                w = getattr(self, name, None)
                if w is not None:
                    w.blockSignals(False)



        # ... rest of the existing loading logic if needed ...



        # Embedding engine: Voyage, OpenAI, Cohere, or Ollama (load keys, models, batch size)

        # explanation: the old Search-tab provider widgets were removed; load the new API-tab controls instead.
        if not hasattr(self, "voyage_api_key_input"):

            if hasattr(self, "_load_embedding_settings"):

                self._load_embedding_settings()

            return



        engine = search_config.get('embedding_engine') or 'voyage'



        self.voyage_api_key_input.setText((search_config.get('voyage_api_key') or '').strip())



        voyage_model = (search_config.get('voyage_embedding_model') or 'voyage-3.5-lite').strip()



        idx_v = self.voyage_embedding_model_combo.findData(voyage_model)



        if idx_v >= 0:



            self.voyage_embedding_model_combo.setCurrentIndex(idx_v)



        self.openai_embedding_api_key_input.setText((search_config.get('openai_embedding_api_key') or '').strip())



        self.openai_embedding_model_input.setText((search_config.get('openai_embedding_model') or 'text-embedding-3-small').strip())



        self.cohere_api_key_input.setText((search_config.get('cohere_api_key') or '').strip())



        self.cohere_embedding_model_input.setText((search_config.get('cohere_embedding_model') or 'embed-english-v3.0').strip())



        try:



            vb = int(search_config.get('voyage_batch_size', 64))



            self.voyage_batch_size_spin.setValue(max(8, min(256, vb)))



        except (TypeError, ValueError):



            self.voyage_batch_size_spin.setValue(64)



        idx = self.embedding_engine_combo.findData(engine)



        if idx >= 0:



            self.embedding_engine_combo.setCurrentIndex(idx)



        self.ollama_base_url_input.setText((search_config.get('ollama_base_url') or "http://localhost:11434").strip())



        self.ollama_embed_model_combo.setCurrentText((search_config.get('ollama_embed_model') or "nomic-embed-text").strip())



        try:



            ob = int(search_config.get('ollama_batch_size', 64))



            self.ollama_batch_size_spin.setValue(max(8, min(256, ob)))



        except (TypeError, ValueError):



            self.ollama_batch_size_spin.setValue(64)



        self.use_dynamic_batch_size_cb.setChecked(bool(search_config.get('use_dynamic_batch_size', True)))



        self._on_embedding_engine_changed()


    def save_settings(self):

        """Saves settings with radical safety and consistent notification."""

        from aqt.utils import showInfo



        # Silence signals during save

        widgets = ['search_method_combo', 'answer_provider_combo', 'local_backend_combo']

        for name in widgets:

            w = getattr(self, name, None)

            if w: w.blockSignals(True)



        try:

            current_config = load_config()

            current_sc = current_config.get('search_config', {})
            pending_reset_sc = getattr(self, "_pending_reset_search_config", {}) or {}

            current_style = current_config.get('styling', {})



            # 1. Answer Provider Logic

            answer_with = self._safe_get_ui_value('answer_provider_combo', current_config.get('provider', 'api_key'))

            saved_cloud_provider = self._safe_get_ui_value(
                'answer_cloud_provider_combo',
                current_config.get('answer_cloud_provider', current_config.get('provider', 'openai'))
            )



            local_backend = self._safe_get_ui_value(
                'local_backend_combo',
                self._infer_local_backend(current_config.get("provider"), current_sc, current_config)
            )
            raw_local_url = self._safe_get_ui_value(
                'local_llm_url',
                current_sc.get('local_llm_url') or current_sc.get('ollama_base_url') or 'http://localhost:1234/v1'
            )
            local_answer_model = self._safe_get_ui_value(
                'local_llm_model',
                current_sc.get('answer_local_model')
                or current_sc.get('ollama_chat_model')
                or current_sc.get('local_llm_model')
                or 'llama3.2'
            )
            normalized_local_url = self._normalize_local_backend_url(local_backend, raw_local_url)
            ollama_base_url = (
                normalized_local_url
                if local_backend == "ollama"
                else (current_sc.get('ollama_base_url') or 'http://localhost:11434')
            )
            local_openai_url = (
                normalized_local_url
                if local_backend != "ollama"
                else (current_sc.get('local_llm_url') or 'http://localhost:1234/v1')
            )

            if answer_with == "local_server":

                provider_type = "ollama" if local_backend == "ollama" else "local_openai"
                api_key = current_config.get('api_key', '')

            else:

                api_key = self._safe_get_ui_value('api_key_input', current_config.get('api_key', ''))

                provider_type = saved_cloud_provider



            note_type_filter = self._build_ntf_from_ui()
            if note_type_filter.get('enabled_note_types') == []:
                restored_ntf = self._resolve_note_type_filter_from_config(current_config)
                if restored_ntf.get('enabled_note_types'):
                    note_type_filter = restored_ntf
                    log_debug("Save settings kept current preset note_type_filter because UI note type selection was empty")



            preserved_rerank_config = get_rerank_config({**dict(current_sc), **pending_reset_sc})
            preserved_rerank_top_k = preserved_rerank_config.get('rerank_top_k', RERANK_TOP_K_DEFAULT)

            config = {

                'api_key': api_key,

                'provider': provider_type,

                'answer_cloud_provider': saved_cloud_provider,

                'current_preset_name': current_config.get('current_preset_name'),

                'styling': {

                    'question_font_size': self._safe_get_ui_value('question_font_spin', current_style.get('question_font_size', 13)),

                    'answer_font_size': self._safe_get_ui_value('answer_font_spin', current_style.get('answer_font_size', 13)),

                    'notes_font_size': self._safe_get_ui_value('notes_font_spin', current_style.get('notes_font_size', 12)),

                    'label_font_size': self._safe_get_ui_value('label_font_spin', current_style.get('label_font_size', 14)),

                    'window_width': self._safe_get_ui_value('width_spin', current_style.get('window_width', 1100)),

                    'window_height': self._safe_get_ui_value('height_spin', current_style.get('window_height', 800)),

                    'section_spacing': self._safe_get_ui_value('section_spacing_spin', current_style.get('section_spacing', 12)),

                    'layout_mode': self._safe_get_ui_value('layout_combo', current_style.get('layout_mode', 'side_by_side')),

                    'answer_spacing': self._safe_get_ui_value('answer_spacing_combo', current_style.get('answer_spacing', 'normal'))

                },

                'note_type_filter': note_type_filter,

                'review_ask_ai': current_config.get('review_ask_ai') or {'context_source': 'embedding_fields'},

                'search_config': {

                    'local_llm_url': local_openai_url,

                    'answer_local_model': local_answer_model if local_backend != "ollama" else current_sc.get('answer_local_model', ''),

                    'local_llm_model': local_answer_model if local_backend != "ollama" else current_sc.get('local_llm_model', ''),

                    'search_method': self._safe_get_ui_value('search_method_combo', current_sc.get('search_method', 'hybrid')),

                    'enable_query_expansion': self._safe_get_ui_value('enable_query_expansion_cb', current_sc.get('enable_query_expansion', False)),

                    'enable_agentic_rag': self._safe_get_ui_value('enable_agentic_rag_cb', current_sc.get('enable_agentic_rag', False)),
                    'enable_profile_memory': self._safe_get_ui_value('enable_profile_memory_cb', current_sc.get('enable_profile_memory', True)),
                    'memory_retrieval_mode': self._safe_get_ui_value('memory_retrieval_mode_combo', current_sc.get('memory_retrieval_mode', 'auto_hybrid')),
                    'memory_retention_days': max(1, min(3650, int(self._safe_get_ui_value('memory_retention_days_spin', current_sc.get('memory_retention_days', 30)) or 30))),
                    'memory_max_saved_snippets_per_search': max(1, min(100, int(self._safe_get_ui_value('memory_max_saved_spin', current_sc.get('memory_max_saved_snippets_per_search', 24)) or 24))),
                    'memory_max_retrieved_snippets': max(1, min(20, int(self._safe_get_ui_value('memory_max_retrieved_spin', current_sc.get('memory_max_retrieved_snippets', 5)) or 5))),
                    'memory_embedding_enabled': self._safe_get_ui_value('memory_embedding_enabled_cb', current_sc.get('memory_embedding_enabled', True)),
                    'agentic_max_retrieval_passes': max(1, min(5, int(current_sc.get('agentic_max_retrieval_passes', 3) or 3))),
                    'agentic_max_subqueries': max(1, min(12, int(current_sc.get('agentic_max_subqueries', 6) or 6))),
                    'agentic_planner_mode': self._safe_get_ui_value('agentic_planner_mode_combo', current_sc.get('agentic_planner_mode', 'deterministic_v1')),
                    'agentic_planner_model': current_sc.get('agentic_planner_model', ''),
                    'agentic_planner_timeout_seconds': max(3, min(60, int(current_sc.get('agentic_planner_timeout_seconds', 25) or 25))),
                    'agentic_planner_max_tokens': max(100, min(1000, int(current_sc.get('agentic_planner_max_tokens', 350) or 350))),

                    'planner_confidence_threshold': max(0.0, min(1.0, float(current_sc.get('planner_confidence_threshold', 0.6)))),

                    'use_ai_generic_term_detection': bool(current_sc.get('use_ai_generic_term_detection', False)),

                    'enable_hyde': self._safe_get_ui_value('enable_hyde_cb', current_sc.get('enable_hyde', False)),

                    'enable_rerank': self._safe_get_ui_value('enable_rerank_cb', current_sc.get('enable_rerank', False)),

                    'rerank_model': self._selected_rerank_model(),

                    'rerank_top_k': preserved_rerank_top_k,

                    'rerank_timeout_seconds': max(
                        RERANK_TIMEOUT_SECONDS_MIN,
                        min(
                            RERANK_TIMEOUT_SECONDS_MAX,
                            int(current_sc.get('rerank_timeout_seconds', RERANK_TIMEOUT_SECONDS_DEFAULT)),
                        ),
                    ),

                    'use_context_boost': self._safe_get_ui_value('use_context_boost_cb', current_sc.get('use_context_boost', True)),

                    'relevance_threshold_percent': max(0, min(80, int(pending_reset_sc.get('relevance_threshold_percent', current_sc.get('relevance_threshold_percent', 65))))),

                    'max_results': self._safe_get_ui_value('max_results_spin', current_sc.get('max_results', 50)),
                    'hybrid_embedding_weight': self._safe_get_ui_value('hybrid_weight_spin', current_sc.get('hybrid_embedding_weight', 40)),

                    'embedding_engine': self._safe_get_ui_value('embedding_engine_combo', current_sc.get('embedding_engine', 'voyage')),

                    'voyage_api_key': self._safe_get_ui_value('voyage_api_key_input', current_sc.get('voyage_api_key', '')),

                    'voyage_embedding_model': self._safe_get_ui_value('voyage_embedding_model_combo', current_sc.get('voyage_embedding_model', 'voyage-3.5-lite')),

                    'openai_embedding_api_key': self._safe_get_ui_value('openai_embedding_api_key_input', current_sc.get('openai_embedding_api_key', '')),

                    'openai_embedding_model': self._safe_get_ui_value('openai_embedding_model_input', current_sc.get('openai_embedding_model', 'text-embedding-3-small')),

                    'cohere_api_key': self._safe_get_ui_value('cohere_api_key_input', current_sc.get('cohere_api_key', '')),

                    'cohere_embedding_model': self._safe_get_ui_value('cohere_embedding_model_input', current_sc.get('cohere_embedding_model', 'embed-english-v3.0')),

                    'voyage_batch_size': int(self._safe_get_ui_value('voyage_batch_size_spin', current_sc.get('voyage_batch_size', 64))),

                    'ollama_base_url': ollama_base_url,

                    'ollama_embed_model': self._safe_get_ui_value('ollama_embed_model_combo', current_sc.get('ollama_embed_model', "nomic-embed-text")),

                    'ollama_batch_size': int(self._safe_get_ui_value('ollama_batch_size_spin', current_sc.get('ollama_batch_size', 64))),

                    'use_dynamic_batch_size': self._safe_get_ui_value('use_dynamic_batch_size_cb', current_sc.get('use_dynamic_batch_size', True)),

                    'ollama_chat_model': local_answer_model if local_backend == "ollama" else self._safe_get_ui_value('ollama_chat_model_combo', current_sc.get('ollama_chat_model', "llama3.2")),

                    'rerank_python_path': self._safe_get_ui_value('rerank_python_path_input', current_sc.get('rerank_python_path', None)),

                    'retrieval_version': 'v2',

                    'keyword_scoring_method': 'bm25',

                    'enable_mmr_diversity': self._safe_get_ui_value('mmr_enabled_cb', current_sc.get('enable_mmr_diversity', True)),

                    'mmr_lambda': max(0, min(100, int(self._safe_get_ui_value('mmr_lambda_slider', int(float(current_sc.get('mmr_lambda', 0.75)) * 100))))) / 100.0,

                    'mmr_similarity_method': 'token_jaccard',

                }

            }
            config['search_config'].update(pending_reset_sc)



            # explanation: merges the new API-tab embedding fields into search_config.
            config['search_config'].update(self._save_embedding_settings())



            # Preserve other config keys

            for k in ['saved_presets', 'current_preset_name']:

                if k in current_config:

                    config[k] = current_config[k]



            if provider_type == "custom":

                config['api_url'] = self._safe_get_ui_value('api_url_input', current_config.get('api_url', ''))



            if save_config(config):

                if hasattr(self, "_pending_reset_search_config"):

                    self._pending_reset_search_config = {}

                if hasattr(self, "retrieval_dirty_label"):

                    self.retrieval_dirty_label.hide()
                    if hasattr(self, "retrieval_dirty_row"):
                        self.retrieval_dirty_row.hide()

                showInfo("Settings saved successfully!")

                self.accept()

            else:

                showInfo("Error: Could not write config file.")



        except Exception as e:

            log_debug(f"Critical error in save_settings: {e}", is_error=True)

            showInfo(f"Save failed: {e}")

        finally:

            for name in widgets:

                w = getattr(self, name, None)

                if w: w.blockSignals(False)


    def _apply_config_to_ui(self):

        """Populates UI with config values while blocking signals to prevent 'Reversion Disease'."""

        try:
                    try:
                        import sip
                    except ImportError:
                        from PyQt6 import sip

        except ImportError:
                    from PyQt6 import sip



        c = load_config()

        sc = c.get('search_config', {})

        style = c.get('styling', {})



        # Comprehensive list of widgets to silence

        widgets = [

            'search_method_combo', 'max_results_spin', 'hybrid_weight_spin',

            'answer_provider_combo', 'answer_cloud_provider_combo', 'local_backend_combo', 'api_key_input',
            'enable_rerank_cb', 'enable_hyde_cb', 'rerank_model_combo',

            'layout_combo', 'answer_spacing_combo',

            'question_font_spin', 'answer_font_spin', 'notes_font_spin',

            'label_font_spin', 'width_spin', 'height_spin', 'section_spacing_spin',

            'use_context_boost_cb',

            'embedding_same_checkbox', 'embedding_strategy_combo',
            'embedding_cloud_provider_combo', 'embedding_cloud_api_key_input',
            'embedding_local_backend_combo', 'embedding_local_url_input', 'embedding_local_model_input', 'embedding_engine_combo',
            'ollama_chat_model_combo', 'ollama_embed_model_combo',

            'enable_query_expansion_cb', 'enable_agentic_rag_cb', 'enable_profile_memory_cb',

            'mmr_enabled_cb', 'mmr_lambda_slider'

        ]



        for name in widgets:

            w = getattr(self, name, None)

            if w and not sip.isdeleted(w):

                w.blockSignals(True)



        try:

            # 1. Search Method & Logic

            method = sc.get('search_method', 'hybrid')

            if hasattr(self, 'search_method_combo') and not sip.isdeleted(self.search_method_combo):

                idx = self.search_method_combo.findData(method)

                if idx >= 0:

                    self.search_method_combo.setCurrentIndex(idx)

                else:

                    # Fallback to text matching

                    for i in range(self.search_method_combo.count()):

                        if method in self.search_method_combo.itemText(i).lower():

                            self.search_method_combo.setCurrentIndex(i)

                            break



            if hasattr(self, 'max_results_spin') and not sip.isdeleted(self.max_results_spin):

                self.max_results_spin.setValue(sc.get('max_results', 50))

            if hasattr(self, 'hybrid_weight_spin') and not sip.isdeleted(self.hybrid_weight_spin):

                self.hybrid_weight_spin.setValue(sc.get('hybrid_embedding_weight', 40))

            rerank_config = get_rerank_config(sc)

            if hasattr(self, 'rerank_model_combo') and not sip.isdeleted(self.rerank_model_combo):

                model = rerank_config.get('rerank_model', DEFAULT_RERANK_MODEL)

                idx = self.rerank_model_combo.findData(model)

                if idx >= 0:

                    self.rerank_model_combo.setCurrentIndex(idx)

                else:

                    self.rerank_model_combo.setCurrentText(model)

            retrieval = get_retrieval_config(sc)

            if hasattr(self, 'mmr_enabled_cb') and not sip.isdeleted(self.mmr_enabled_cb):

                self.mmr_enabled_cb.setChecked(bool(retrieval.get('enable_mmr_diversity', True)))

            if hasattr(self, 'mmr_lambda_slider') and not sip.isdeleted(self.mmr_lambda_slider):

                self.mmr_lambda_slider.setValue(max(0, min(100, int(round(float(retrieval.get('mmr_lambda', 0.75)) * 100)))))

            if hasattr(self, '_update_retrieval_v2_controls'):

                self._update_retrieval_v2_controls(mark_dirty=False)

            if hasattr(self, 'retrieval_dirty_label') and not sip.isdeleted(self.retrieval_dirty_label):

                self.retrieval_dirty_label.hide()
                if hasattr(self, 'retrieval_dirty_row') and not sip.isdeleted(self.retrieval_dirty_row):
                    self.retrieval_dirty_row.hide()



            # 2. Answer Provider & Local Server

            prov = c.get('provider', 'api_key')
            local_backend = self._infer_local_backend(prov, sc, c)

            if hasattr(self, 'answer_provider_combo') and not sip.isdeleted(self.answer_provider_combo):

                answer_mode = 'local_server' if prov in ['ollama', 'local_openai', 'local_server'] else 'api_key'
                p_idx = self.answer_provider_combo.findData(answer_mode)
                if p_idx >= 0:
                    self.answer_provider_combo.setCurrentIndex(p_idx)

            if hasattr(self, 'answer_cloud_provider_combo') and not sip.isdeleted(self.answer_cloud_provider_combo):

                cloud_provider_id = c.get('answer_cloud_provider') or prov

                cloud_provider = {
                    'anthropic': 'anthropic',
                    'openai': 'openai',
                    'google': 'google',
                    'gemini': 'google',
                    'openrouter': 'openrouter',
                    'custom': 'custom',
                }.get(cloud_provider_id, 'openai')
                cp_idx = self.answer_cloud_provider_combo.findData(cloud_provider)
                if cp_idx >= 0:
                    self.answer_cloud_provider_combo.setCurrentIndex(cp_idx)



            if hasattr(self, 'api_key_input') and not sip.isdeleted(self.api_key_input):

                self.api_key_input.setText(c.get('api_key', ''))



            if hasattr(self, 'local_llm_url') and not sip.isdeleted(self.local_llm_url):

                if hasattr(self, 'local_backend_combo') and not sip.isdeleted(self.local_backend_combo):
                    self._select_combo_data(self.local_backend_combo, local_backend)
                if local_backend == "ollama":
                    self.local_llm_url.setText(sc.get('ollama_base_url', 'http://localhost:11434'))
                else:
                    self.local_llm_url.setText(sc.get('local_llm_url', 'http://localhost:1234/v1'))

            if hasattr(self, 'local_llm_model') and not sip.isdeleted(self.local_llm_model):

                self.local_llm_model.setText(
                    (sc.get('ollama_chat_model') if local_backend == "ollama" else None)
                    or sc.get('answer_local_model')
                    or sc.get('local_llm_model', 'llama3.2')
                )



            # 3. Checkboxes (using radical safety)

            self._safe_set_checked(getattr(self, 'enable_query_expansion_cb', None), sc.get('enable_query_expansion', False))

            self._safe_set_checked(getattr(self, 'enable_agentic_rag_cb', None), sc.get('enable_agentic_rag', False))
            self._safe_set_checked(getattr(self, 'enable_profile_memory_cb', None), sc.get('enable_profile_memory', True))
            if hasattr(self, "memory_retrieval_mode_combo") and not sip.isdeleted(self.memory_retrieval_mode_combo):
                idx = self.memory_retrieval_mode_combo.findData(sc.get("memory_retrieval_mode", "auto_hybrid"))
                if idx >= 0:
                    self.memory_retrieval_mode_combo.setCurrentIndex(idx)
            if hasattr(self, "memory_retention_days_spin") and not sip.isdeleted(self.memory_retention_days_spin):
                self.memory_retention_days_spin.setValue(int(sc.get("memory_retention_days", 30) or 30))
            if hasattr(self, "memory_max_saved_spin") and not sip.isdeleted(self.memory_max_saved_spin):
                self.memory_max_saved_spin.setValue(int(sc.get("memory_max_saved_snippets_per_search", 24) or 24))
            if hasattr(self, "memory_max_retrieved_spin") and not sip.isdeleted(self.memory_max_retrieved_spin):
                self.memory_max_retrieved_spin.setValue(int(sc.get("memory_max_retrieved_snippets", 5) or 5))
            self._safe_set_checked(getattr(self, "memory_embedding_enabled_cb", None), sc.get("memory_embedding_enabled", True))
            if hasattr(self, 'agentic_planner_mode_combo') and not sip.isdeleted(self.agentic_planner_mode_combo):
                mode = sc.get('agentic_planner_mode', 'deterministic_v1')
                idx = self.agentic_planner_mode_combo.findData(mode)
                if idx >= 0:
                    self.agentic_planner_mode_combo.setCurrentIndex(idx)

            self._safe_set_checked(getattr(self, 'enable_hyde_cb', None), sc.get('enable_hyde', False))

            self._safe_set_checked(getattr(self, 'enable_rerank_cb', None), sc.get('enable_rerank', False))

            self._safe_set_checked(getattr(self, 'use_context_boost_cb', None), sc.get('use_context_boost', True))


            # --- Persistent Rerank Path Fix ---

            if hasattr(self, "rerank_python_path_input") and not sip.isdeleted(self.rerank_python_path_input):

                path = sc.get('rerank_python_path', "")

                self.rerank_python_path_input.setText(str(path) if path else "")
                self._apply_cached_rerank_status(sc)



            # 4. Embedding Engine

            if hasattr(self, 'embedding_engine_combo') and not sip.isdeleted(self.embedding_engine_combo):

                engine = sc.get('embedding_engine', 'voyage')

                e_idx = self.embedding_engine_combo.findData(engine)

                if e_idx >= 0: self.embedding_engine_combo.setCurrentIndex(e_idx)



            if hasattr(self, 'voyage_api_key_input') and not sip.isdeleted(self.voyage_api_key_input):

                self.voyage_api_key_input.setText(sc.get('voyage_api_key', ''))



            if hasattr(self, 'voyage_embedding_model_combo') and not sip.isdeleted(self.voyage_embedding_model_combo):

                v_model = sc.get('voyage_embedding_model', 'voyage-3.5-lite')

                v_idx = self.voyage_embedding_model_combo.findData(v_model)

                if v_idx >= 0: self.voyage_embedding_model_combo.setCurrentIndex(v_idx)



            if hasattr(self, 'ollama_base_url_input') and not sip.isdeleted(self.ollama_base_url_input):

                self.ollama_base_url_input.setText(sc.get('ollama_base_url', "http://localhost:11434"))

            if hasattr(self, 'ollama_embed_model_combo') and not sip.isdeleted(self.ollama_embed_model_combo):

                self.ollama_embed_model_combo.setCurrentText(sc.get('ollama_embed_model', "nomic-embed-text"))

            if hasattr(self, 'ollama_chat_model_combo') and not sip.isdeleted(self.ollama_chat_model_combo):

                self.ollama_chat_model_combo.setCurrentText(sc.get('ollama_chat_model', "llama3.2"))


            if hasattr(self, '_load_embedding_settings'):

                self._load_embedding_settings()



            # 5. Styling

            styling_widgets = {

                'question_font_spin': 'question_font_size',

                'answer_font_spin': 'answer_font_size',

                'notes_font_spin': 'notes_font_size',

                'label_font_spin': 'label_font_size',

                'width_spin': 'window_width',

                'height_spin': 'window_height',

                'section_spacing_spin': 'section_spacing'

            }

            for widget_name, config_key in styling_widgets.items():

                w = getattr(self, widget_name, None)

                if w and not sip.isdeleted(w):

                    w.setValue(style.get(config_key, 13 if 'font' in config_key else (1100 if 'width' in config_key else 800)))



            if hasattr(self, 'layout_combo') and not sip.isdeleted(self.layout_combo):

                l_idx = self.layout_combo.findData(style.get('layout_mode', 'side_by_side'))

                if l_idx >= 0: self.layout_combo.setCurrentIndex(l_idx)



            if hasattr(self, 'answer_spacing_combo') and not sip.isdeleted(self.answer_spacing_combo):

                s_idx = self.answer_spacing_combo.findData(style.get('answer_spacing', 'normal'))

                if s_idx >= 0: self.answer_spacing_combo.setCurrentIndex(s_idx)



        except Exception as e:

            log_debug(f"Error during unified UI population: {e}", is_error=True)



        # Unsilencing

        for name in widgets:

            w = getattr(self, name, None)

            if w and not sip.isdeleted(w):

                w.blockSignals(False)



        # Final UI Sync

        if hasattr(self, '_on_answer_provider_changed'): self._on_answer_provider_changed()

        if hasattr(self, '_on_embedding_engine_changed'): self._on_embedding_engine_changed()

        if hasattr(self, '_on_search_method_changed'): self._on_search_method_changed()

        if hasattr(self, '_update_agentic_planner_controls'): self._update_agentic_planner_controls()

        if hasattr(self, '_update_retrieval_v2_controls'): self._update_retrieval_v2_controls(mark_dirty=False)

        if hasattr(self, 'retrieval_dirty_label'): self.retrieval_dirty_label.hide()
        if hasattr(self, 'retrieval_dirty_row'): self.retrieval_dirty_row.hide()


