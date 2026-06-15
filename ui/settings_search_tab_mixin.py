"""Focused Settings dialog layout builder mixin."""

from aqt.qt import *

from .theme import settings_button_style, settings_panel_style, settings_text_style
from .widgets import (
    CollapsibleSection,
    settings_child_group,
    settings_checkbox_row,
    settings_compact_group,
    settings_hint_box,
    settings_inline_row,
    settings_page,
    settings_status,
    settings_toolbar,
)
from ..utils import EmbeddingsTabMessages


class SettingsSearchTabMixin:
    """Builds one focused section of the Settings dialog UI."""

    def _build_search_embeddings_tab(self, theme):
        # --- Search Settings Tab (Reorganized by Glutamine) ---

        search_tab = QWidget()
        search_scroll = QScrollArea()
        search_scroll.setWidgetResizable(True)
        search_scroll.setFrameShape(QFrame.Shape.NoFrame)

        search_page, search_layout = settings_page(
            theme,
            "\U0001F50D Search & embeddings",
            "Ask Notes uses hybrid note search by default. Optional: tune result count, relevance, reranking, and retrieval aids.",
        )
        search_scroll.setWidget(search_page)

        main_search_layout = QVBoxLayout(search_tab)
        main_search_layout.setContentsMargins(0, 0, 0, 0)
        main_search_layout.addWidget(search_scroll)



        # Glutamine's Resident Reset Button

        reset_btn_layout = QHBoxLayout()

        self.medical_reset_btn = QPushButton("\u2695\ufe0f Reset to Clinical Defaults")

        self.medical_reset_btn.setToolTip("Sets all search, relevance, and AI settings to high-yield medical defaults (Resident-approved).")

        self.medical_reset_btn.setStyleSheet(settings_button_style(theme, "muted"))

        self.medical_reset_btn.clicked.connect(self.reset_to_medical_defaults)

        self.medical_reset_btn.setMaximumWidth(240)

        reset_btn_layout.addWidget(settings_toolbar(theme, self.medical_reset_btn))

        search_layout.addLayout(reset_btn_layout)



        # --- High-Yield Action Zone (New persistent section for indexing) ---

        index_zone = QFrame()
        self.index_zone = index_zone

        index_zone.setStyleSheet(settings_panel_style(theme, "index"))

        index_layout = QVBoxLayout(index_zone)
        index_layout.setContentsMargins(10, 8, 10, 10)
        index_layout.setSpacing(7)



        self.embedding_status_label = settings_status(theme, "Ready to index...", "info")

        index_layout.addWidget(self.embedding_status_label)



        index_btns = QHBoxLayout()
        index_btns.setSpacing(8)

        self.create_embedding_btn = QPushButton("Create/Update Embeddings")

        self.create_embedding_btn.setToolTip(EmbeddingsTabMessages.CREATE_UPDATE_TOOLTIP)

        self.create_embedding_btn.setStyleSheet(settings_button_style(theme, "primary"))

        self.create_embedding_btn.clicked.connect(self._create_or_update_embeddings)
        self.create_embedding_btn.setMaximumWidth(260)

        index_btns.addWidget(self.create_embedding_btn)

        self.clean_embedding_storage_btn = QPushButton("Clean embedding storage")
        self.clean_embedding_storage_btn.setToolTip(
            "Preview and remove old separate embedding files, deleted-note rows, and true duplicate rows. Keeps valid chunk rows."
        )
        self.clean_embedding_storage_btn.clicked.connect(self._clean_embedding_storage)
        self.clean_embedding_storage_btn.setMaximumWidth(240)
        index_btns.addWidget(self.clean_embedding_storage_btn)
        index_btns.addStretch(1)

        index_layout.addLayout(index_btns)

        search_layout.addWidget(index_zone)



        # explanation: embedding provider setup moved to the API Settings tab.



        # --- 2. RETRIEVAL STRATEGY ---

        retrieval_strategy_section = CollapsibleSection("Retrieval Strategy", is_expanded=True)

        self.enable_agentic_rag_cb = QCheckBox("Smart Retrieval Planner")
        self.enable_agentic_rag_cb.setToolTip(
            "Let Ask Notes plan retrieval, check evidence coverage, and choose the retrieval strategy automatically."
        )
        self.enable_agentic_rag_cb.stateChanged.connect(self._update_agentic_planner_controls)

        smart_group = settings_compact_group(theme, "Smart retrieval planner")
        self.enable_agentic_rag_cb.setText("Enable smart retrieval planner")
        smart_group.content_layout.addWidget(settings_checkbox_row(theme, self.enable_agentic_rag_cb))

        self.agentic_planner_mode_combo = QComboBox()
        self.agentic_planner_mode_combo.addItem("Deterministic V1", "deterministic_v1")
        self.agentic_planner_mode_combo.addItem("Adaptive V2 (experimental)", "adaptive_v2")
        self.agentic_planner_mode_combo.setToolTip(
            "Choose how Smart Retrieval Planner plans retrieval. Adaptive V2 may add a short planning call before search and can be slower if the same local model is also generating answers."
        )
        self.agentic_planner_mode_combo.currentIndexChanged.connect(self._update_agentic_planner_controls)
        smart_group.content_layout.addWidget(settings_inline_row(theme, "Planner mode", self.agentic_planner_mode_combo, 205))

        self.agentic_planner_hint_label = QLabel("Chooses retrieval strategy automatically per search. Adaptive V2 may add a short planning call and can be slower when sharing the answer model.")
        self.agentic_planner_hint_label.setWordWrap(True)
        self.agentic_planner_hint_label.setStyleSheet(settings_text_style(theme, "subtle"))
        smart_group.content_layout.addWidget(settings_hint_box(theme, self.agentic_planner_hint_label))
        retrieval_strategy_section.addWidget(smart_group)

        memory_group = settings_compact_group(theme, "Memory")

        self.enable_profile_memory_cb = QCheckBox("Enable Local Memory")
        self.enable_profile_memory_cb.setToolTip(
            "Remember local profile preferences, search scope, and compact snippets from final answer context. Does not store full note text, answers, or raw chat history."
        )
        self.enable_profile_memory_cb.setText("Enable local memory")
        memory_group.content_layout.addWidget(settings_checkbox_row(theme, self.enable_profile_memory_cb))

        memory_btn_row = QHBoxLayout()
        memory_btn_row.setSpacing(7)
        self.show_memory_summary_btn = QPushButton("Show memory summary")
        self.show_memory_summary_btn.setToolTip("Show local memory keys and fact-snippet counts stored by this add-on.")
        self.show_memory_summary_btn.clicked.connect(self._show_memory_summary)
        self.show_memory_summary_btn.setMaximumWidth(210)
        memory_btn_row.addWidget(self.show_memory_summary_btn)

        self.show_memory_inspector_btn = QPushButton("Inspect memory")
        self.show_memory_inspector_btn.setToolTip("Search and inspect locally stored memory snippets.")
        self.show_memory_inspector_btn.clicked.connect(self._show_memory_inspector)
        self.show_memory_inspector_btn.setMaximumWidth(170)
        memory_btn_row.addWidget(self.show_memory_inspector_btn)

        self.clear_profile_memory_btn = QPushButton("Clear memory")
        self.clear_profile_memory_btn.setToolTip("Delete durable local memory, including profile settings and fact snippets. Session memory is cleared when chat closes.")
        self.clear_profile_memory_btn.clicked.connect(self._clear_profile_memory)
        self.clear_profile_memory_btn.setMaximumWidth(190)
        memory_btn_row.addWidget(self.clear_profile_memory_btn)
        memory_btn_row.addStretch(1)
        memory_group.content_layout.addLayout(memory_btn_row)

        self.memory_retrieval_mode_combo = QComboBox()
        self.memory_retrieval_mode_combo.addItem("Text only", "text")
        self.memory_retrieval_mode_combo.addItem("Auto hybrid", "auto_hybrid")
        self.memory_retrieval_mode_combo.setToolTip("Choose how local memory snippets are retrieved for planning.")
        memory_group.content_layout.addWidget(settings_inline_row(theme, "Memory retrieval", self.memory_retrieval_mode_combo, 170))

        self.memory_retention_days_spin = QSpinBox()
        self.memory_retention_days_spin.setRange(1, 3650)
        self.memory_retention_days_spin.setToolTip("Delete snippets that have not been seen within this many days.")
        memory_group.content_layout.addWidget(settings_inline_row(theme, "Retention days", self.memory_retention_days_spin, 120))

        self.memory_max_saved_spin = QSpinBox()
        self.memory_max_saved_spin.setRange(1, 100)
        self.memory_max_saved_spin.setToolTip("Maximum snippets saved from one answer context.")
        memory_group.content_layout.addWidget(settings_inline_row(theme, "Save per search", self.memory_max_saved_spin, 120))

        self.memory_max_retrieved_spin = QSpinBox()
        self.memory_max_retrieved_spin.setRange(1, 20)
        self.memory_max_retrieved_spin.setToolTip("Maximum memory snippets passed to the retrieval planner.")
        memory_group.content_layout.addWidget(settings_inline_row(theme, "Retrieve per plan", self.memory_max_retrieved_spin, 120))

        self.memory_embedding_enabled_cb = QCheckBox("Enable memory embeddings")
        self.memory_embedding_enabled_cb.setToolTip("Use the configured embedding provider for hybrid local memory retrieval when available.")
        memory_group.content_layout.addWidget(settings_checkbox_row(theme, self.memory_embedding_enabled_cb))

        retrieval_strategy_section.addWidget(memory_group)

        self.search_method_combo = QComboBox()

        self.search_method_combo.addItem("Keyword Only", "keyword")

        self.search_method_combo.addItem("Keyword + Re-rank", "keyword_rerank")

        self.search_method_combo.addItem("Hybrid (RRF)", "hybrid")

        self.search_method_combo.addItem("Embedding Only", "embedding")
        self.search_method_combo.setToolTip(
            "Choose how notes are retrieved: keyword, semantic embeddings, hybrid, or keyword with re-ranking."
        )

        self.search_method_combo.currentIndexChanged.connect(self._on_search_method_changed)

        self.search_method_row = settings_inline_row(theme, "Search strategy", self.search_method_combo, 205)
        retrieval_strategy_section.addWidget(self.search_method_row)
        self.search_method_planner_hint_label = QLabel("Controlled by Smart Retrieval Planner")
        self.search_method_planner_hint_label.setStyleSheet(settings_text_style(theme, "subtle"))
        self.search_method_planner_hint_row = settings_hint_box(theme, self.search_method_planner_hint_label)
        self.search_method_planner_hint_row.hide()
        retrieval_strategy_section.addWidget(self.search_method_planner_hint_row)
        self.search_strategy_section = retrieval_strategy_section

        self.enable_query_expansion_cb = QCheckBox("Query Expansion (AI adds medical synonyms)")
        self.enable_query_expansion_cb.setToolTip(
            "Use AI to add related medical terms and synonyms before searching. Can improve recall."
        )

        retrieval_strategy_section.addWidget(settings_checkbox_row(theme, self.enable_query_expansion_cb))

        self.enable_hyde_cb = QCheckBox("HyDE (AI generates hypothetical document first)")
        self.enable_hyde_cb.setToolTip(
            "Generate a hypothetical answer/document first, then search for notes similar to it."
        )

        self.enable_hyde_row = settings_checkbox_row(theme, self.enable_hyde_cb)

        retrieval_strategy_section.addWidget(self.enable_hyde_row)

        search_layout.addWidget(retrieval_strategy_section)

        # --- 3. CLINICAL ACCURACY ---

        accuracy_section = CollapsibleSection("Clinical Accuracy", is_expanded=False)
        self.accuracy_section = accuracy_section

        accuracy_layout = QVBoxLayout()

        accuracy_layout.setContentsMargins(0, 0, 0, 0)

        accuracy_layout.setSpacing(7)

        self.max_results_spin = QSpinBox()

        self.max_results_spin.setRange(5, 100)
        self.max_results_spin.setToolTip("Maximum visible result rows. AI context is selected dynamically from the best matches.")

        accuracy_layout.addWidget(settings_inline_row(theme, "Max results", self.max_results_spin, 120))

        self.hybrid_weight_spin = QSpinBox()

        self.hybrid_weight_spin.setRange(0, 100)
        self.hybrid_weight_spin.setToolTip(
            "For Hybrid search, controls how much ranking comes from semantic embeddings versus keyword matching. Higher = more semantic; lower = more keyword-based."
        )

        self.hybrid_weight_label = QLabel("Embedding weight")
        self.hybrid_weight_row = settings_inline_row(theme, self.hybrid_weight_label, self.hybrid_weight_spin, 120)
        accuracy_layout.addWidget(self.hybrid_weight_row)

        accuracy_section.addLayout(accuracy_layout)

        search_layout.addWidget(accuracy_section)

        # --- SEARCH RESULT DIVERSITY ---

        retrieval_section = CollapsibleSection("Search Result Diversity (MMR)", is_expanded=False)
        self.retrieval_section = retrieval_section
        retrieval_layout = QVBoxLayout()
        retrieval_layout.setContentsMargins(0, 0, 0, 0)
        retrieval_layout.setSpacing(7)

        self.mmr_enabled_cb = QCheckBox("Diversity filter")
        self.mmr_enabled_cb.setToolTip("Reduce near-duplicate notes using Maximal Marginal Relevance (MMR).")
        retrieval_layout.addWidget(settings_checkbox_row(theme, self.mmr_enabled_cb))

        self.mmr_subcontrols_widget = QWidget()
        mmr_subcontrols_layout = QVBoxLayout(self.mmr_subcontrols_widget)
        mmr_subcontrols_layout.setContentsMargins(12, 0, 0, 0)
        mmr_subcontrols_layout.setSpacing(7)

        mmr_strength_row = QHBoxLayout()
        self.mmr_lambda_slider = QSlider(Qt.Orientation.Horizontal)
        self.mmr_lambda_slider.setRange(0, 100)
        self.mmr_lambda_slider.setSingleStep(5)
        self.mmr_lambda_slider.setPageStep(10)
        self.mmr_lambda_slider.setMaximumWidth(520)
        self.mmr_lambda_slider.setToolTip("Higher = more relevant results. Lower = more varied results.")
        self.mmr_lambda_value_label = QLabel("75%")
        self.mmr_lambda_value_label.setMinimumWidth(42)
        self.mmr_lambda_value_label.setStyleSheet(settings_text_style(theme, "subtle"))
        mmr_strength_row.addWidget(self.mmr_lambda_slider)
        mmr_strength_row.addWidget(self.mmr_lambda_value_label)
        mmr_strength_widget = QFrame()
        mmr_strength_widget.setObjectName("settingsInlineRow")
        mmr_strength_widget.setStyleSheet("QFrame#settingsInlineRow { background: transparent; border: none; }")
        mmr_strength_layout = QHBoxLayout(mmr_strength_widget)
        mmr_strength_layout.setContentsMargins(0, 0, 0, 0)
        mmr_strength_layout.setSpacing(7)
        mmr_strength_label = QLabel("Diversity strength")
        mmr_strength_label.setStyleSheet(settings_text_style(theme, "subtle"))
        mmr_strength_layout.addWidget(mmr_strength_label)
        mmr_strength_layout.addLayout(mmr_strength_row, 1)
        mmr_subcontrols_layout.addWidget(mmr_strength_widget)

        retrieval_layout.addWidget(self.mmr_subcontrols_widget)

        self.retrieval_dirty_label = QLabel("Changes apply to next search")
        self.retrieval_dirty_label.setStyleSheet(settings_text_style(theme, "subtle"))
        self.retrieval_dirty_row = settings_hint_box(theme, self.retrieval_dirty_label)
        self.retrieval_dirty_row.hide()
        retrieval_layout.addWidget(self.retrieval_dirty_row)

        retrieval_section.addLayout(retrieval_layout)

        self.mmr_enabled_cb.stateChanged.connect(lambda *_: self._update_retrieval_v2_controls(mark_dirty=True))
        self.mmr_lambda_slider.valueChanged.connect(lambda *_: self._update_retrieval_v2_controls(mark_dirty=True))


        # --- 4. RESULT RE-RANKING ---

        rerank_section = CollapsibleSection("AI Search Re-Ranking", is_expanded=False)
        self.rerank_section = rerank_section

        # 4a. Main Toggles (Top)
        self.enable_rerank_cb = QCheckBox("Enable AI-powered re-ranking")
        self.enable_rerank_cb.setEnabled(False)
        self.enable_rerank_cb.toggled.connect(lambda *_: self._update_rerank_controls_enabled())
        self.enable_rerank_row = settings_checkbox_row(theme, self.enable_rerank_cb)
        rerank_section.addWidget(self.enable_rerank_row)

        self.use_context_boost_cb = QCheckBox("Consider surrounding context in matches")
        self.use_context_boost_cb.setToolTip(
            "Boost notes that match important surrounding context in the query, not just isolated keywords."
        )
        rerank_section.addWidget(settings_checkbox_row(theme, self.use_context_boost_cb))

        # 4b. Separator
        rerank_separator = QFrame()
        rerank_separator.setFrameShape(QFrame.Shape.HLine if hasattr(QFrame, "Shape") else QFrame.HLine)
        rerank_separator.setFrameShadow(QFrame.Shadow.Sunken if hasattr(QFrame, "Shadow") else QFrame.Sunken)
        rerank_section.addWidget(rerank_separator)

        # 4c. Python environment
        self.python_path_widget = self._build_rerank_python_card(theme)
        python_path_group = settings_child_group(theme)
        python_path_group.content_layout.addWidget(self.python_path_widget)
        rerank_section.addWidget(python_path_group)
        self.python_path_widget.setVisible(True)

        # 4d. Model configuration
        self.rerank_model_card = self._build_rerank_model_card(theme)
        rerank_model_group = settings_child_group(theme)
        rerank_model_group.content_layout.addWidget(self.rerank_model_card)
        rerank_section.addWidget(rerank_model_group)
        self._update_rerank_controls_enabled()

        search_layout.addWidget(rerank_section)

        search_layout.addWidget(retrieval_section)



        # explanation: legacy embedding widgets stay hidden so older helper methods remain safe.
        self.embedding_section = CollapsibleSection("Embeddings (for semantic search)", is_expanded=False)

        self.embedding_section.hide()











        self.ollama_options = QWidget()

        ollama_form = QFormLayout(self.ollama_options)



        # Move "Zombie" Local Widgets here

        self.local_ai_status_label = QLabel("Scanning for local AI...")

        self.local_ai_status_label.setStyleSheet(f"font-weight: bold; color: {theme['warning']};")

        ollama_form.addRow("Detected Provider:", self.local_ai_status_label)



        self.scan_local_btn = QPushButton("Scan & Pull Models")

        self.scan_local_btn.clicked.connect(self._on_scan_and_pull)

        ollama_form.addRow("", self.scan_local_btn)



        self.ollama_base_url_input = QLineEdit()



        self.ollama_base_url_input.setMinimumWidth(380)



        self.ollama_base_url_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)



        self.ollama_base_url_input.setPlaceholderText("http://localhost:11434")



        self.ollama_base_url_input.setToolTip("Ollama server URL. Default: http://localhost:11434")



        ollama_form.addRow("Ollama URL:", self.ollama_base_url_input)



        # Model: editable combo so user can pick from detected list or type a custom model



        model_row = QHBoxLayout()



        self.ollama_embed_model_combo = QComboBox()



        self.ollama_embed_model_combo.setEditable(True)



        self.ollama_embed_model_combo.setMinimumWidth(280)



        self.ollama_embed_model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)



        self.ollama_embed_model_combo.setToolTip("Embedding model. Click 'Refresh models' to load from Ollama, or type a model name (e.g. nomic-embed-text).")



        model_row.addWidget(self.ollama_embed_model_combo)



        self.ollama_refresh_models_btn = QPushButton("\U0001F504 Refresh models")



        self.ollama_refresh_models_btn.setToolTip("Fetch available models from Ollama (requires Ollama to be running)")



        self.ollama_refresh_models_btn.clicked.connect(self._refresh_ollama_models)



        model_row.addWidget(self.ollama_refresh_models_btn)



        ollama_form.addRow("Embed model:", model_row)



        self.ollama_batch_size_spin = QSpinBox()



        self.ollama_batch_size_spin.setRange(8, 256)



        self.ollama_batch_size_spin.setValue(64)



        self.ollama_batch_size_spin.setSuffix(" notes/batch")



        self.ollama_batch_size_spin.setToolTip("Starting batch size. With 'Use dynamic batch size' on, this adapts automatically from response time and notes/sec for best speed.")



        ollama_form.addRow("Batch size:", self.ollama_batch_size_spin)



        self.embedding_section.addWidget(self.ollama_options)







        self.use_dynamic_batch_size_cb = QCheckBox("Use dynamic batch size (adapt to response time for best speed)")



        self.use_dynamic_batch_size_cb.setChecked(True)



        self.use_dynamic_batch_size_cb.setToolTip("When enabled, batch size adapts both ways from response time: decrease if a batch is slow (>15s), increase if fast (<6s), to balance total time and responsiveness.")



        self.embedding_section.addWidget(self.use_dynamic_batch_size_cb)




        # Buttons for embedding operations
        embedding_btn_layout = QHBoxLayout()



        search_layout.addWidget(self.embedding_section)



        return search_tab
