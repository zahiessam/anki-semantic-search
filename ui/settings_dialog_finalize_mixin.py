"""Focused Settings dialog layout builder mixin."""

import time

from aqt.qt import *

from .widgets import sync_setting_row_tooltips
from ..utils import load_config, log_debug


class SettingsDialogFinalizeMixin:
    """Builds one focused section of the Settings dialog UI."""

    def _finalize_settings_layout(self, main_layout, tabs, search_tab, note_filter_tab, start_time, config):

        # Tab order: API -> Search & Embeddings -> AI Search Scope -> Styling
        tabs.removeTab(tabs.indexOf(search_tab))
        tabs.insertTab(1, search_tab, "\U0001F50D Search & embeddings")

        tabs.removeTab(tabs.indexOf(note_filter_tab))
        tabs.insertTab(2, note_filter_tab, "\U0001F4CB AI search scope")

        for i in range(tabs.count()):
            label = tabs.tabText(i)
            if "Search" in label and "embedding" in label.lower():
                label = "\U0001F50D Search & embeddings"
            else:
                label = label.replace("_", " ")
            tabs.setTabText(i, label)

        if self.open_to_embeddings:
            tabs.setCurrentWidget(search_tab)
            # explanation: focus indexing action because provider setup now lives in API Settings.
            QTimer.singleShot(200, lambda: self.create_embedding_btn.setFocus())

        tabs_elapsed = time.time() - start_time
        log_debug(f"  [Timing] All tabs created: {tabs_elapsed:.3f}s")







        # Initialize embedding status (lazy load to avoid blocking)



        QTimer.singleShot(100, self._refresh_embedding_status)  # Slight delay to avoid race conditions







        # Scroll area: wrap tabs; compact min height to avoid excessive empty space
        scroll_content = QWidget()
        scroll_content.setMinimumHeight(380)
        scroll_content.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        scroll_content_layout = QVBoxLayout(scroll_content)
        scroll_content_layout.setContentsMargins(0, 0, 0, 0)
        scroll_content_layout.addWidget(tabs)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_content)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setMinimumHeight(450)
        scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll_area.setFocusPolicy(Qt.FocusPolicy.WheelFocus)

        self._settings_scroll_area = scroll_area
        self._settings_scroll_content = scroll_content
        self._settings_tabs = tabs
        self._tabs = tabs

        tabs.currentChanged.connect(self._on_settings_tab_changed)

        scroll_content.installEventFilter(self)
        tabs.installEventFilter(self)
        scroll_area.viewport().installEventFilter(self)

        self._apply_settings_control_sizing(tabs)
        sync_setting_row_tooltips(tabs)
        self._install_wheel_scroll_guard(tabs)

        main_layout.addWidget(scroll_area, 1)







        # Final timing



        total_elapsed = time.time() - start_time



        log_debug(f"=== Settings Dialog UI Setup Completed: {total_elapsed:.3f}s total ===")







        # Buttons at bottom (always visible, not inside scroll)



        btn_layout = QHBoxLayout()



        btn_layout.addStretch()  # Push buttons to the right



        save_btn = QPushButton("Save Settings")

        save_btn.setObjectName("saveBtn")

        save_btn.clicked.connect(self.save_settings)



        cancel_btn = QPushButton("Cancel")

        cancel_btn.setObjectName("cancelBtn")



        cancel_btn.clicked.connect(self.close)







        btn_layout.addWidget(save_btn)



        btn_layout.addWidget(cancel_btn)



        main_layout.addLayout(btn_layout)







        # Load existing config
        log_debug(f"  [Debug] About to load config...")

        config = config or load_config()

        log_debug(f"  [Debug] Config loaded")



        if 'api_key' in config:



            self.api_key_input.setText(config['api_key'])



        # Answer provider: local server vs cloud, then the specific cloud provider.

        provider = config.get('provider', 'ollama')

        answer_mode = "local_server" if provider in ["ollama", "local_openai", "local_server"] else "api_key"

        self._select_combo_data(self.answer_provider_combo, answer_mode)

        cloud_provider_id = config.get('answer_cloud_provider') or provider

        self._select_combo_data(self.answer_cloud_provider_combo, {
            "anthropic": "anthropic",
            "openai": "openai",
            "google": "google",
            "gemini": "google",
            "openrouter": "openrouter",
            "custom": "custom",
        }.get(cloud_provider_id, "openai"))



        # Load Local LLM settings

        sc = config.get('search_config') or {}
        local_backend = self._infer_local_backend(provider, sc, config)
        self._select_combo_data(self.local_backend_combo, local_backend)
        if local_backend == "ollama":
            self.local_llm_url.setText((sc.get('ollama_base_url') or "http://localhost:11434").strip())
        else:
            self.local_llm_url.setText(
                (sc.get('local_llm_url') or config.get('local_llm_url') or 'http://localhost:1234/v1').strip()
            )
        self.local_llm_model.setText(
            (sc.get('ollama_chat_model') if local_backend == "ollama" else None)
            or sc.get('answer_local_model')
            or sc.get('local_llm_model')
            or config.get('local_llm_model', 'model-identifier')
        )



        if provider == 'ollama' and hasattr(self, 'ollama_chat_model_combo'):



            sc = config.get('search_config') or {}



            self.ollama_chat_model_combo.setCurrentText((sc.get('ollama_chat_model') or 'llama3.2').strip())



        self._on_answer_provider_changed()







        if 'api_url' in config:



            self.api_url_input.setText(config['api_url'])







        # Apply config to UI (this might involve slow operations)



        apply_start = time.time()



        if 'styling' in config:



            styling = config['styling']



            self.question_font_spin.setValue(styling.get('question_font_size', 13))



            self.answer_font_spin.setValue(styling.get('answer_font_size', 13))



            self.notes_font_spin.setValue(styling.get('notes_font_size', 12))



            self.label_font_spin.setValue(styling.get('label_font_size', 14))



            self.width_spin.setValue(styling.get('window_width', 1100))



            self.height_spin.setValue(styling.get('window_height', 800))



            self.section_spacing_spin.setValue(styling.get('section_spacing', 12))



            mode = styling.get('layout_mode', 'side_by_side')



            idx = self.layout_combo.findData(mode)



            if idx >= 0:



                self.layout_combo.setCurrentIndex(idx)



            spacing_mode = styling.get('answer_spacing', 'normal')



            idx = self.answer_spacing_combo.findData(spacing_mode)



            if idx >= 0:



                self.answer_spacing_combo.setCurrentIndex(idx)



        try:
            ntf = self._resolve_scope_config_from_config(config, persist=True)
        except Exception as e:
            log_debug(f"Error resolving scope config: {e}")
            ntf = {}



        ntf_start = time.time()



        try:
            # Try to apply scope config if available, otherwise fall back to old note type config
            log_debug(f"  [Debug] Checking scope config application: has _apply_scope_config={hasattr(self, '_apply_scope_config')}, _scope_lists_loaded={getattr(self, '_scope_lists_loaded', False)}")
            if hasattr(self, '_apply_scope_config'):
                log_debug(f"  [Debug] Applying scope config during init")
                self._apply_scope_config(ntf)
            elif hasattr(self, '_apply_note_type_config'):
                log_debug(f"  [Debug] Applying note type config during init")
                self._apply_note_type_config(ntf)
        except Exception as e:
            log_debug(f"Error applying scope/note type config: {e}")



        ntf_elapsed = time.time() - ntf_start



        log_debug(f"  [Timing] _apply_scope_config() in __init__: {ntf_elapsed:.3f}s")







        # --- Load Configuration (Guarded) ---

        self._apply_config_to_ui()
