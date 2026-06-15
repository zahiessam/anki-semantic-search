"""Settings dialog helpers for defaults and rerank setup cards."""

from aqt.qt import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from aqt.utils import askUser, showInfo

from .dependency_install import install_dependencies
from .settings_constants import RERANK_MODEL_PRESETS
from .theme import (
    settings_button_style,
    settings_status_label_style,
    settings_text_style,
)
from .widgets import (
    CollapsibleSection,
    settings_button_strip,
    settings_field_row,
    settings_hint_box,
    settings_labeled_action_row,
)
from ..utils.config import (
    DEFAULT_RERANK_MODEL,
    RELEVANCE_THRESHOLD_PERCENT_DEFAULT,
    RERANK_TIMEOUT_SECONDS_DEFAULT,
    RERANK_TOP_K_DEFAULT,
)


class SettingsRerankUiMixin:
    """Owns medical defaults and rerank setup card construction."""

    def reset_to_medical_defaults(self):
        """Apply the current Clinical High-Yield preset to visible Settings controls."""
        if not askUser("Reset all settings to high-yield clinical defaults (Resident-approved)?"):
            return

        self._select_combo_data(self.answer_provider_combo, "local_server")
        self._select_combo_data(self.local_backend_combo, "ollama")
        self.local_llm_url.setText("http://localhost:11434")
        self.local_llm_model.setText("llama3.2:latest")
        if hasattr(self, "ollama_chat_model_combo"):
            self.ollama_chat_model_combo.setCurrentText("llama3.2:latest")

        self._select_combo_data(self.search_method_combo, "hybrid")
        self.max_results_spin.setValue(50)
        self.hybrid_weight_spin.setValue(40)
        self.enable_query_expansion_cb.setChecked(True)
        if hasattr(self, "enable_agentic_rag_cb"):
            self.enable_agentic_rag_cb.setChecked(False)
        if hasattr(self, "enable_profile_memory_cb"):
            self.enable_profile_memory_cb.setChecked(True)
        if hasattr(self, "agentic_planner_mode_combo"):
            self._select_combo_data(self.agentic_planner_mode_combo, "deterministic_v1")
        self.enable_hyde_cb.setChecked(True)
        self.enable_rerank_cb.setChecked(True)
        self.use_context_boost_cb.setChecked(True)
        self._select_combo_data(self.rerank_model_combo, DEFAULT_RERANK_MODEL)

        if hasattr(self, "mmr_enabled_cb"):
            self.mmr_enabled_cb.setChecked(True)
        if hasattr(self, "mmr_lambda_slider"):
            self.mmr_lambda_slider.setValue(75)
        if hasattr(self, "_update_retrieval_v2_controls"):
            self._update_retrieval_v2_controls(mark_dirty=False)

        self._pending_reset_search_config = {
            "relevance_threshold_percent": RELEVANCE_THRESHOLD_PERCENT_DEFAULT,
            "rerank_top_k": RERANK_TOP_K_DEFAULT,
            "rerank_timeout_seconds": RERANK_TIMEOUT_SECONDS_DEFAULT,
            "enable_agentic_rag": False,
            "enable_profile_memory": True,
            "agentic_planner_mode": "deterministic_v1",
            "agentic_planner_model": "",
            "agentic_planner_timeout_seconds": 25,
            "agentic_planner_max_tokens": 350,
            "planner_confidence_threshold": 0.6,
            "enable_context_score_cliff": True,
            "context_score_cliff_threshold": 15.0,
            "context_score_cliff_min_notes": 8,
            "enable_context_anchor_rescue": True,
            "context_score_cliff_anchor_rescue_slots": 3,
            "enable_rescue_specificity_scoring": True,
            "rescue_specificity_threshold": 0.85,
            "rescue_specificity_max_doc_freq": 4,
            "rescue_specificity_max_weight": 0.5,
        }

        if hasattr(self, "embedding_same_checkbox"):
            self.embedding_same_checkbox.setChecked(False)
            self._select_combo_data(self.embedding_strategy_combo, "local")
            self._select_combo_data(self.embedding_local_backend_combo, "ollama")
            self.embedding_local_url_input.setText("http://localhost:11434")
            self.embedding_local_model_input.setText("nomic-embed-text:latest")
            if hasattr(self, "ollama_embed_model_combo"):
                self.ollama_embed_model_combo.setCurrentText("nomic-embed-text:latest")
            if hasattr(self, "_on_same_provider_toggled"):
                self._on_same_provider_toggled()

        self.question_font_spin.setValue(14)
        self.answer_font_spin.setValue(13)
        self.notes_font_spin.setValue(12)
        self.label_font_spin.setValue(14)
        self.section_spacing_spin.setValue(8)
        self._select_combo_data(self.layout_combo, "side_by_side")
        self._select_combo_data(self.answer_spacing_combo, "normal")

        self.include_all_note_types_cb.setChecked(True)
        self.include_all_decks_cb.setChecked(True)
        self.use_first_field_cb.setChecked(True)

        self._on_answer_provider_changed()
        self._on_search_method_changed()
        if hasattr(self, "_update_agentic_planner_controls"):
            self._update_agentic_planner_controls()

        showInfo("Clinical High-Yield preset applied. Click 'Save Settings' to persist it.")

    def _build_rerank_python_card(self, theme):
        """Build the Python environment controls for reranking."""
        section = CollapsibleSection("Python environment", is_expanded=False)
        card = QWidget()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(8)

        hint = QLabel(
            "External Python is recommended on Windows to avoid DLL conflicts with Anki's runtime. Configure once."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(settings_text_style(theme, "hint"))
        card_layout.addWidget(settings_hint_box(theme, hint))

        path_row = QHBoxLayout()
        self.rerank_python_path_input = QLineEdit()
        self.rerank_python_path_input.setPlaceholderText(r"C:\Path\To\Python311\python.exe")
        path_row.addWidget(self.rerank_python_path_input)

        self.autodetect_python_btn = QPushButton("Autodetect")
        self.autodetect_python_btn.clicked.connect(self._on_autodetect_python)
        path_row.addWidget(self.autodetect_python_btn)

        self.browse_rerank_python_btn = QPushButton("Browse")
        self.browse_rerank_python_btn.clicked.connect(self._on_browse_rerank_python)
        path_row.addWidget(self.browse_rerank_python_btn)
        card_layout.addWidget(settings_field_row(theme, label="Python path", layout=path_row))

        self.install_external_btn = QPushButton("Install dependencies")
        self.install_external_btn.setToolTip("Install required packages into the external Python environment.")
        self.install_external_btn.setStyleSheet(settings_button_style(theme, "muted"))
        self.install_external_btn.clicked.connect(self._on_install_into_external_python)

        self.check_rerank_btn = QPushButton("Check environment")
        self.check_rerank_btn.clicked.connect(self._on_check_rerank_again)

        self.install_anki_python_btn = QPushButton("Anki Python fallback")
        self.install_anki_python_btn.setToolTip("Fallback only: external Python is highly recommended for Cross-Encoder setup.")
        self.install_anki_python_btn.setStyleSheet(settings_button_style(theme, "muted"))
        self.install_anki_python_btn.clicked.connect(lambda: install_dependencies(python_exe=None))
        card_layout.addWidget(
            settings_button_strip(
                theme,
                self.install_external_btn,
                self.check_rerank_btn,
                self.install_anki_python_btn,
            )
        )

        self.rerank_status_label = QLabel("Cross-Encoder package: checking...")
        self.rerank_status_label.setStyleSheet(settings_status_label_style(theme, "warning"))
        card_layout.addWidget(self.rerank_status_label)

        section.addWidget(card)
        return section

    def _build_rerank_model_card(self, theme):
        """Build the reranking model controls."""
        section = CollapsibleSection("Re-ranking model", is_expanded=False)
        card = QWidget()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(8)

        self.rerank_model_combo = QComboBox()
        self.rerank_model_combo.setEditable(True)
        for label, model_id in RERANK_MODEL_PRESETS:
            self.rerank_model_combo.addItem(f"{model_id} ({label})", model_id)
        self.rerank_model_combo.setCurrentText(DEFAULT_RERANK_MODEL)
        self.rerank_model_combo.setToolTip("Medical Semantic Sort model used to rerank the top search results.")
        self.rerank_model_combo.currentTextChanged.connect(self._on_rerank_model_changed)
        card_layout.addWidget(settings_labeled_action_row(theme, "Model", self.rerank_model_combo))

        self.rerank_model_status_label = QLabel("Selected model: not verified")
        self.rerank_model_status_label.setStyleSheet(settings_status_label_style(theme, "warning"))
        card_layout.addWidget(self.rerank_model_status_label)

        self.download_rerank_model_btn = QPushButton("Verify / download model")
        self.download_rerank_model_btn.setToolTip(
            "Checks the local cache, downloads missing files, then warms the selected Cross-Encoder model.\n"
            "Search uses cached models only, so rerank timeouts cover scoring rather than model downloads."
        )
        self.download_rerank_model_btn.setStyleSheet(settings_button_style(theme, "muted"))
        self.download_rerank_model_btn.clicked.connect(self._on_download_rerank_model)
        card_layout.addWidget(settings_button_strip(theme, self.download_rerank_model_btn))

        self.rerank_download_progress = QProgressBar()
        self.rerank_download_progress.setRange(0, 100)
        self.rerank_download_progress.setValue(0)
        self.rerank_download_progress.hide()
        card_layout.addWidget(self.rerank_download_progress)

        self.rerank_download_status_label = QLabel("")
        self.rerank_download_status_label.setWordWrap(True)
        self.rerank_download_status_label.setStyleSheet(settings_text_style(theme, "subtle"))
        self.rerank_download_status_label.hide()
        card_layout.addWidget(self.rerank_download_status_label)

        section.addWidget(card)
        return section
