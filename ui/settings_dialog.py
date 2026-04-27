"""Settings dialog UI for configuring search, embeddings, and providers."""

# ============================================================================
# Imports
# ============================================================================

import os

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
from .theme import (
    get_addon_theme,
    settings_button_style,
    settings_dialog_stylesheet,
    settings_panel_style,
    settings_status_label_style,
    settings_text_style,
)
from .widgets import CollapsibleSection, settings_field_row
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
    migrate_embeddings_json_to_db,
)
from ..core.workers import EmbeddingWorker, RerankCheckWorker
from ..utils import (
    EmbeddingsTabMessages,
    format_partial_failure_completion,
    format_partial_failure_progress,
    get_effective_embedding_config,
    get_embeddings_storage_path_for_read,
    load_config,
    log_debug,
    save_config,
    validate_embedding_config,
)


# ============================================================================
# Settings Dialog
# ============================================================================

_addon_theme = get_addon_theme

ANSWER_CLOUD_PROVIDERS = [
    ("Anthropic (Claude)", "anthropic", "sk-ant-..."),
    ("OpenAI (GPT)", "openai", "sk-..."),
    ("Google (Gemini)", "google", "AI..."),
    ("OpenRouter", "openrouter", "sk-or-..."),
    ("Custom / OpenAI-compatible", "custom", "custom key"),
]

EMBEDDING_CLOUD_PROVIDERS = [
    ("Voyage AI (Recommended)", "Voyage AI", "pa-..."),
    ("OpenAI", "OpenAI", "sk-..."),
    ("Cohere", "Cohere", "co-..."),
]

ANSWER_KEY_PROVIDER_PREFIXES = (
    ("sk-ant-", "anthropic"),
    ("sk-or-", "openrouter"),
    ("AI", "google"),
    ("sk-", "openai"),
)

EMBEDDING_KEY_PROVIDER_PREFIXES = (
    ("sk-ant-", None),
    ("sk-or-", None),
    ("pa-", "Voyage AI"),
    ("co-", "Cohere"),
    ("sk-", "OpenAI"),
)


class SettingsDialog(QDialog):



    # --- Lifecycle And Window Setup ---

    def __init__(self, parent=None, open_to_embeddings=False):



        import time



        _t0 = time.time()



        super().__init__(parent)



        self.open_to_embeddings = open_to_embeddings



        self.setWindowTitle("Anki Semantic Search \u2014 Settings")



        self._rerank_check_done = False  # defer rerank check until after show



        # Size: allow small minimum, no max so user can maximize/resize to expand and reduce cramming



        self.setMinimumWidth(750)



        self.setMinimumHeight(550)



        # Open large by default so Search Settings content is less crammed; user can maximize or resize



        screen = QApplication.primaryScreen().geometry()



        w = min(1200, int(screen.width() * 0.96))



        h = min(960, int(screen.height() * 0.92))



        self.resize(w, h)



        # Behave like a normal top-level window so minimize/maximize work



        self.setWindowFlags(



            Qt.WindowType.Window



            | Qt.WindowType.WindowMinimizeButtonHint



            | Qt.WindowType.WindowMaximizeButtonHint



            | Qt.WindowType.WindowCloseButtonHint



        )







        palette = QApplication.palette()



        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128



        theme = _addon_theme(is_dark)



        self.setStyleSheet(settings_dialog_stylesheet(theme))



        log_debug("=== Settings Dialog Opened ===")



        # Store reference to service process



        self.service_process = None



        self.setup_ui()







    def showEvent(self, event):



        """Defer rerank availability check until after window is shown so opening Settings doesn't freeze."""



        super().showEvent(event)



        if not getattr(self, "_rerank_check_scheduled", False):



            self._rerank_check_scheduled = True



            from aqt.qt import QTimer



            QTimer.singleShot(80, self._deferred_check_rerank)







    def _deferred_check_rerank(self):



        """Run _check_rerank_available in a worker thread so Settings never freezes."""



        import time



        config = load_config()



        sc = (config or {}).get("search_config") or {}



        rerank_python = (sc.get("rerank_python_path") or "").strip() or None



        self._rerank_check_worker = RerankCheckWorker(self, rerank_python)



        self._rerank_check_start = time.time()







        def _on_rerank_check_done(available):

            try:

                try:
                    try:
                        import sip
                    except ImportError:
                        from PyQt6 import sip

                except ImportError:
                    from PyQt6 import sip



                self._rerank_available = available

                cb = getattr(self, "enable_rerank_cb", None)

                if cb is not None and not sip.isdeleted(cb):

                    cb.setEnabled(available)



                if hasattr(self, "_update_rerank_tooltip") and not sip.isdeleted(self):

                    self._update_rerank_tooltip()

                    self._update_rerank_status_ui()

                self._rerank_check_worker = None

            except (RuntimeError, AttributeError, ImportError):

                pass







        self._rerank_check_worker.finished_signal.connect(_on_rerank_check_done)



        self._rerank_check_worker.start()







    # --- Defaults And UI Construction ---

    def reset_to_medical_defaults(self):

        """Resets all settings to high-yield medical defaults as suggested by Glutamine."""

        from aqt.utils import askUser

        if not askUser("Reset all settings to high-yield clinical defaults (Resident-approved)?"):

            return



        # 1. API & Provider Defaults

        self.answer_provider_combo.setCurrentIndex(0) # Default to Ollama (Local)

        self.local_llm_url.setText("http://localhost:1234/v1")

        self.local_llm_model.setText("llama3.2")



        # 2. Search & Embedding Defaults (Medical High-Yield)

        self.search_method_combo.setCurrentIndex(2) # Hybrid (RRF) - Best for Med

        self.min_relevance_spin.setValue(55) # Filter out noise

        self.max_results_spin.setValue(50) # Comprehensive for medical use

        self.hybrid_weight_spin.setValue(40) # Slightly favor keywords for drug names/genes

        self.enable_query_expansion_cb.setChecked(True) # AI synonyms are great for medicine

        self.use_ai_generic_term_detection_cb.setChecked(True)



        # 3. Styling Defaults (Dark mode friendly)

        self.question_font_spin.setValue(14)

        self.answer_font_spin.setValue(13)

        self.notes_font_spin.setValue(12)

        self.layout_combo.setCurrentIndex(0) # Side-by-side



        # 4. Note Types & Fields

        self.include_all_note_types_cb.setChecked(True)

        self.include_all_decks_cb.setChecked(True)

        self.use_first_field_cb.setChecked(True)



        from aqt.utils import showInfo

        showInfo("Settings reset to Clinical High-Yield defaults. Click 'Save' to apply.")



    def setup_ui(self):

        # --- RADICAL STABILIZATION: Guard against re-initialization ---

        if hasattr(self, "_ui_initialized") and self._ui_initialized:

            return

        self._ui_initialized = True



        current_config = load_config()

        self.current_config = current_config # Store for access in other methods



        import time

        start_time = time.time()



        log_debug("=== Settings Dialog UI Setup Started ===")



        palette = QApplication.palette()



        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128



        theme = _addon_theme(is_dark)







        # Main layout with proper spacing - fix layout issues



        main_layout = QVBoxLayout(self)



        main_layout.setSpacing(10)



        main_layout.setContentsMargins(15, 15, 15, 15)







        elapsed = time.time() - start_time



        log_debug(f"  [Timing] Layout setup: {elapsed:.3f}s")







        tabs = QTabWidget()



        # Size to content; scroll_content minimum and scroll area give enough height for scrolling when needed



        tabs.setMinimumHeight(0)







        # API Settings Tab



        api_tab = QWidget()
        api_outer_layout = QHBoxLayout(api_tab)
        api_outer_layout.setContentsMargins(0, 0, 0, 0)
        api_outer_layout.setSpacing(0)
        api_body = QWidget()
        api_body.setMinimumWidth(820)
        api_body.setMaximumWidth(1100)
        api_body.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        api_outer_layout.addStretch(1)
        api_outer_layout.addWidget(api_body)
        api_outer_layout.addStretch(1)



        api_layout = QVBoxLayout(api_body)



        api_layout.setSpacing(15)



        api_layout.setContentsMargins(20, 20, 20, 20)







        info = QLabel("\U0001F511 API & AI provider")



        info.setWordWrap(True)



        info.setStyleSheet(settings_text_style(theme, "heading"))



        api_layout.addWidget(info)







        subtitle = QLabel(

            "Choose a Local Server (no key required) or a Cloud API provider and key. "

            "Works with OpenAI, Anthropic, Google, OpenRouter, or custom OpenAI-compatible endpoints."

        )



        subtitle.setWordWrap(True)



        subtitle.setStyleSheet(settings_text_style(theme, "subtitle"))



        api_layout.addWidget(subtitle)







        privacy_note = QLabel(

            "\u26a0\ufe0f AI answers may send selected note content to the chosen provider. "

            "Use a Local Server (Ollama, LM Studio, Jan) to keep everything on your machine. "

            "Cloud providers require an API key and an internet connection \u2014 your note content will be sent to external APIs."

        )



        privacy_note.setWordWrap(True)



        privacy_note.setStyleSheet(settings_panel_style(theme, "warning"))



        api_layout.addWidget(privacy_note)







        # Define sections first (initialize as None or create them)

        self.api_key_section = QWidget()

        self.local_server_section = QWidget()



        self.answer_provider_combo = QComboBox()

        self.answer_provider_combo.blockSignals(True) # Silence during setup

        self.answer_provider_combo.addItem("\U0001f4bb Local Server (Ollama, LM Studio, Jan)", "local_server")

        self.answer_provider_combo.addItem("\u2601\ufe0f Cloud API (OpenAI, Anthropic, Gemini)", "api_key")



        self.answer_provider_combo.setToolTip(

            "Local Server: Best for privacy. Supports Ollama, LM Studio, Jan, etc.\n"

            "Cloud API: Best quality, requires internet and API key."

        )

        # Set initial value based on config

        current_provider = current_config.get('provider', 'openai')

        if current_provider in ['ollama', 'local_openai', 'local_server']:

            self.answer_provider_combo.setCurrentIndex(0) # Local

        else:

            self.answer_provider_combo.setCurrentIndex(1) # Cloud



        self.answer_provider_combo.blockSignals(False) # Re-enable

        api_layout.addWidget(settings_field_row(theme, self.answer_provider_combo, "Answer with:"))







        # API Key section (hidden when Ollama is selected)



        self.api_key_section = QWidget()



        api_key_section_layout = QVBoxLayout(self.api_key_section)



        api_key_section_layout.setContentsMargins(0, 0, 0, 0)



        answer_cloud_section = self._build_cloud_provider_section(
            theme=theme,
            providers=ANSWER_CLOUD_PROVIDERS,
            provider_attr="answer_cloud_provider_combo",
            key_attr="api_key_input",
            show_button_attr="show_key_btn",
            detected_label_attr="provider_label",
            key_placeholder="Paste your answer API key here...",
            show_callback=self.toggle_key_visibility,
            changed_callback=self.detect_provider,
        )
        api_key_section_layout.addWidget(answer_cloud_section)







        url_layout = QHBoxLayout()



        url_label = QLabel("API URL:")



        self.api_url_input = QLineEdit()



        self.api_url_input.setPlaceholderText("https://api.example.com/v1/chat/completions")
        self.api_url_input.textChanged.connect(self.detect_provider)



        url_layout.addWidget(url_label)



        url_layout.addWidget(self.api_url_input)



        self.url_widget = QWidget()



        self.url_widget.setLayout(url_layout)



        self.url_widget.hide()

        self.url_row = settings_field_row(theme, self.url_widget)
        self.url_row.hide()
        api_key_section_layout.addWidget(self.url_row)



        api_layout.addWidget(self.api_key_section)







        # Unified Local Server Section

        self.local_server_section = QWidget()

        local_server_layout = QVBoxLayout(self.local_server_section)
        local_server_layout.setContentsMargins(0, 0, 0, 0)
        local_server_layout.setSpacing(8)



        self.local_llm_url = QLineEdit()

        self.local_llm_url.setPlaceholderText("http://localhost:11434 (Ollama) or http://localhost:1234/v1 (LM Studio)")

        local_server_layout.addWidget(settings_field_row(theme, self.local_llm_url, "Server URL:"))



        model_row = QHBoxLayout()

        self.local_llm_model = QLineEdit()

        self.local_llm_model.setPlaceholderText("e.g. llama3.2 or gemma2")

        model_row.addWidget(self.local_llm_model)

        self.local_server_autodetect_btn = QPushButton("Autodetect")

        self.local_server_autodetect_btn.setToolTip("Find a running local server and select one of its loaded models.")

        self.local_server_autodetect_btn.clicked.connect(self._autodetect_local_server)

        model_row.addWidget(self.local_server_autodetect_btn)



        local_server_layout.addWidget(settings_field_row(theme, layout=model_row, label="Model Name:"))



        self.local_server_test_btn = QPushButton("\U0001F50C Test Connection")

        self.local_server_test_btn.setToolTip("Test connection to your local server. Shows latency and availability.")

        self.local_server_test_btn.clicked.connect(self._test_local_server_connection)

        local_server_layout.addWidget(settings_field_row(theme, self.local_server_test_btn))



        local_guide = QLabel(

            "- <b>Ollama:</b> http://localhost:11434<br>"

            "- <b>LM Studio:</b> http://localhost:1234/v1<br>"

            "- <b>Jan:</b> http://localhost:1337/v1"

        )

        local_guide.setStyleSheet(settings_text_style(theme, "hint"))

        local_server_layout.addWidget(settings_field_row(theme, local_guide))



        self.local_server_section.hide()

        api_layout.addWidget(self.local_server_section)



        # explanation: adds embedding provider controls below the existing answer provider controls.
        embedding_divider = QFrame()

        embedding_divider.setFrameShape(QFrame.Shape.HLine)

        embedding_divider.setFrameShadow(QFrame.Shadow.Sunken)

        api_layout.addWidget(embedding_divider)



        embedding_header = QLabel("\U0001F50D Embedding Provider")

        embedding_header.setStyleSheet(settings_text_style(theme, "section_heading"))

        api_layout.addWidget(embedding_header)



        embedding_subtitle = QLabel("Used for semantic search (finding relevant cards).")

        embedding_subtitle.setWordWrap(True)

        embedding_subtitle.setStyleSheet(settings_text_style(theme, "subtle"))

        api_layout.addWidget(embedding_subtitle)



        self.embedding_same_checkbox = QCheckBox("Use same provider as answering")

        self.embedding_same_checkbox.setChecked(True)

        api_layout.addWidget(settings_field_row(theme, self.embedding_same_checkbox))



        self.embedding_same_summary_label = QLabel()

        self.embedding_same_summary_label.setWordWrap(True)

        self.embedding_same_summary_label.setStyleSheet(settings_text_style(theme, "summary"))

        api_layout.addWidget(self.embedding_same_summary_label)



        self.embedding_independent_section = QWidget()

        embedding_independent_layout = QVBoxLayout(self.embedding_independent_section)

        embedding_independent_layout.setContentsMargins(0, 0, 0, 0)

        embedding_independent_layout.setSpacing(8)



        self.embedding_strategy_combo = QComboBox()

        self.embedding_strategy_combo.addItem("\U0001f4bb Local Server (Ollama, LM Studio, Jan)", "local")

        self.embedding_strategy_combo.addItem("\u2601\ufe0f Cloud API (Voyage, OpenAI, Cohere)", "cloud")

        embedding_independent_layout.addWidget(settings_field_row(theme, self.embedding_strategy_combo, "Embedding with:"))



        self.embedding_local_section = QWidget()

        embedding_local_layout = QVBoxLayout(self.embedding_local_section)

        embedding_local_layout.setContentsMargins(0, 0, 0, 0)

        self.embedding_local_url_input = QLineEdit()

        self.embedding_local_url_input.setPlaceholderText("http://localhost:11434/v1")

        embedding_local_layout.addWidget(settings_field_row(theme, self.embedding_local_url_input, "Server URL:"))

        embedding_local_hint = QLabel("Must expose an /embeddings endpoint")

        embedding_local_hint.setStyleSheet(settings_text_style(theme, "hint"))

        embedding_local_layout.addWidget(settings_field_row(theme, embedding_local_hint))

        embedding_independent_layout.addWidget(self.embedding_local_section)



        self.embedding_cloud_section = self._build_cloud_provider_section(
            theme=theme,
            providers=EMBEDDING_CLOUD_PROVIDERS,
            provider_attr="embedding_cloud_provider_combo",
            key_attr="embedding_cloud_api_key_input",
            show_button_attr="embedding_cloud_show_key_btn",
            detected_label_attr="embedding_cloud_detected_label",
            key_placeholder="Paste your embedding API key here...",
            show_callback=self._toggle_embedding_cloud_key_visibility,
            changed_callback=self._on_embedding_cloud_provider_changed,
        )

        embedding_independent_layout.addWidget(self.embedding_cloud_section)

        api_layout.addWidget(self.embedding_independent_section)

        self._connect_embedding_signals()







        # Collapsible help: "Need help?" toggles visibility



        self._api_help_visible = False



        api_help_btn = QPushButton("Need help? (providers, free options)")



        api_help_btn.setToolTip("Click to show or hide provider links and free options")



        self.info_text = QLabel()



        self.info_text.setWordWrap(True)



        self.info_text.setStyleSheet(settings_panel_style(theme))



        self.info_text.setText(



            "Answer with Ollama (local): no API key \u2014 choose 'Ollama (local)' above and set Chat model. Uses Ollama URL from Search & Embeddings tab.\n\n"



            "API Key (cloud):\n"



            "\u2022 Anthropic (Claude): sk-ant-... \u2192 console.anthropic.com\n"



            "\u2022 OpenAI (GPT): sk-... \u2192 platform.openai.com/api-keys\n"



            "\u2022 Google (Gemini): AI... \u2192 aistudio.google.com/app/apikey (FREE!)\n"



            "\u2022 OpenRouter: sk-or-... \u2192 openrouter.ai/keys\n\n"



            "\U0001F4A1 Free options: Google Gemini or Ollama (local)"



        )



        self.info_text.setVisible(False)



        def _toggle_api_help():



            self._api_help_visible = not self._api_help_visible



            self.info_text.setVisible(self._api_help_visible)



            api_help_btn.setText("\u25b2 Hide help" if self._api_help_visible else "Need help? (providers, free options)")



        api_help_btn.clicked.connect(_toggle_api_help)



        api_layout.addWidget(api_help_btn)



        api_layout.addWidget(self.info_text)



        api_layout.addStretch()







        tabs.addTab(api_tab, "\U0001F511 API Settings")







        # Styling Tab



        style_tab = QWidget()
        style_outer_layout = QHBoxLayout(style_tab)
        style_outer_layout.setContentsMargins(0, 0, 0, 0)
        style_outer_layout.setSpacing(0)
        style_body = QWidget()
        style_body.setMinimumWidth(760)
        style_body.setMaximumWidth(980)
        style_body.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        style_outer_layout.addStretch(1)
        style_outer_layout.addWidget(style_body)
        style_outer_layout.addStretch(1)



        style_layout = QVBoxLayout(style_body)



        style_layout.setSpacing(15)



        style_layout.setContentsMargins(20, 20, 20, 20)







        style_info = QLabel("\U0001F3A8 Appearance")



        style_info.setStyleSheet(settings_text_style(theme, "heading"))



        style_layout.addWidget(style_info)



        style_sub = QLabel("Font sizes, window size, and layout. Optional; defaults work for most users.")



        style_sub.setWordWrap(True)



        style_sub.setStyleSheet(settings_text_style(theme, "subtitle"))



        style_layout.addWidget(style_sub)







        # Font Size Settings



        font_group = QGroupBox("Font Sizes")



        font_layout = QVBoxLayout(font_group)
        font_layout.setSpacing(8)







        question_font_layout = QHBoxLayout()



        question_font_label = QLabel("Question Input Font Size:")



        self.question_font_spin = QSpinBox()



        self.question_font_spin.setRange(10, 20)



        self.question_font_spin.setValue(13)



        self.question_font_spin.setSuffix(" px")



        question_font_layout.addWidget(question_font_label)



        question_font_layout.addStretch()



        question_font_layout.addWidget(self.question_font_spin)



        font_layout.addWidget(settings_field_row(theme, layout=question_font_layout))







        answer_font_layout = QHBoxLayout()



        answer_font_label = QLabel("AI Answer Font Size:")



        self.answer_font_spin = QSpinBox()



        self.answer_font_spin.setRange(10, 20)



        self.answer_font_spin.setValue(13)



        self.answer_font_spin.setSuffix(" px")



        answer_font_layout.addWidget(answer_font_label)



        answer_font_layout.addStretch()



        answer_font_layout.addWidget(self.answer_font_spin)



        font_layout.addWidget(settings_field_row(theme, layout=answer_font_layout))







        notes_font_layout = QHBoxLayout()



        notes_font_label = QLabel("Notes List Font Size:")



        self.notes_font_spin = QSpinBox()



        self.notes_font_spin.setRange(10, 18)



        self.notes_font_spin.setValue(12)



        self.notes_font_spin.setSuffix(" px")



        notes_font_layout.addWidget(notes_font_label)



        notes_font_layout.addStretch()



        notes_font_layout.addWidget(self.notes_font_spin)



        font_layout.addWidget(settings_field_row(theme, layout=notes_font_layout))







        label_font_layout = QHBoxLayout()



        label_font_label = QLabel("Label Font Size:")



        self.label_font_spin = QSpinBox()



        self.label_font_spin.setRange(11, 18)



        self.label_font_spin.setValue(14)



        self.label_font_spin.setSuffix(" px")



        label_font_layout.addWidget(label_font_label)



        label_font_layout.addStretch()



        label_font_layout.addWidget(self.label_font_spin)



        font_layout.addWidget(settings_field_row(theme, layout=label_font_layout))







        style_layout.addWidget(font_group)







        # Window Size Settings



        window_group = QGroupBox("Window Size")



        window_layout = QVBoxLayout(window_group)
        window_layout.setSpacing(8)







        width_layout = QHBoxLayout()



        width_label = QLabel("Default Window Width:")



        self.width_spin = QSpinBox()



        self.width_spin.setRange(800, 1600)



        self.width_spin.setValue(1100)



        self.width_spin.setSuffix(" px")



        width_layout.addWidget(width_label)



        width_layout.addStretch()



        width_layout.addWidget(self.width_spin)



        window_layout.addWidget(settings_field_row(theme, layout=width_layout))







        height_layout = QHBoxLayout()



        height_label = QLabel("Default Window Height:")



        self.height_spin = QSpinBox()



        self.height_spin.setRange(600, 1200)



        self.height_spin.setValue(800)



        self.height_spin.setSuffix(" px")



        height_layout.addWidget(height_label)



        height_layout.addStretch()



        height_layout.addWidget(self.height_spin)



        window_layout.addWidget(settings_field_row(theme, layout=height_layout))







        style_layout.addWidget(window_group)







        # Layout



        layout_group = QGroupBox("Layout")



        layout_layout = QVBoxLayout(layout_group)
        layout_layout.setSpacing(8)



        layout_row = QHBoxLayout()



        layout_label = QLabel("Answer & Notes:")



        self.layout_combo = QComboBox()



        self.layout_combo.addItem("Side-by-side (answer | notes)", "side_by_side")



        self.layout_combo.addItem("Stacked (answer above notes)", "stacked")



        layout_row.addWidget(layout_label)



        layout_row.addStretch()



        layout_row.addWidget(self.layout_combo)



        layout_layout.addWidget(settings_field_row(theme, layout=layout_row))



        style_layout.addWidget(layout_group)







        # Spacing Settings



        spacing_group = QGroupBox("Spacing & Padding")



        spacing_layout = QVBoxLayout(spacing_group)
        spacing_layout.setSpacing(8)







        section_spacing_layout = QHBoxLayout()



        section_spacing_label = QLabel("Section Spacing:")



        self.section_spacing_spin = QSpinBox()



        self.section_spacing_spin.setRange(5, 20)



        self.section_spacing_spin.setValue(12)



        self.section_spacing_spin.setSuffix(" px")



        section_spacing_layout.addWidget(section_spacing_label)



        section_spacing_layout.addStretch()



        section_spacing_layout.addWidget(self.section_spacing_spin)



        spacing_layout.addWidget(settings_field_row(theme, layout=section_spacing_layout))







        answer_spacing_layout = QHBoxLayout()



        answer_spacing_label = QLabel("Answer line spacing:")



        self.answer_spacing_combo = QComboBox()



        self.answer_spacing_combo.addItem("Compact", "compact")



        self.answer_spacing_combo.addItem("Normal", "normal")



        self.answer_spacing_combo.addItem("Comfortable", "comfortable")



        answer_spacing_layout.addWidget(answer_spacing_label)



        answer_spacing_layout.addStretch()



        answer_spacing_layout.addWidget(self.answer_spacing_combo)



        spacing_layout.addWidget(settings_field_row(theme, layout=answer_spacing_layout))







        style_layout.addWidget(spacing_group)



        style_layout.addStretch()







        tabs.addTab(style_tab, "\U0001F3A8 Styling")







        # --- Note Types & Fields Tab ---



        nt_tab = QWidget()



        nt_main = QVBoxLayout(nt_tab)



        nt_main.setSpacing(10)



        nt_main.setContentsMargins(20, 20, 20, 20)







        nt_info = QLabel(



            "\U0001F4CB Choose which note types and decks to search. Select fields to search per note type. "



            "Tip: For shared or public decks, select the note types and decks you want; you can leave all selected to search everything."



        )



        nt_info.setWordWrap(True)



        nt_info.setStyleSheet(f"font-size: 11px; color: {theme['subtext']}; margin-bottom: 6px;")



        nt_info.setStyleSheet(f"font-size: 17px; font-weight: bold; color: {theme['text']}; margin-bottom: 6px;")



        nt_info.setWordWrap(True)



        nt_main.addWidget(nt_info)







        # Side-by-side: left = Note types + Decks (stacked), right = Fields by note type



        main_h_split = QSplitter(Qt.Orientation.Horizontal)



        left_v_split = QSplitter(Qt.Orientation.Vertical)







        # ---- Left column: Note types (top) ----



        nt_group = QGroupBox("Note types to include")



        nt_gl = QVBoxLayout(nt_group)



        nt_btn_row = QHBoxLayout()



        nt_select_btn = QPushButton("Select All")



        nt_select_btn.clicked.connect(lambda: self._set_note_types_checked(True))



        nt_deselect_btn = QPushButton("Deselect All")



        nt_deselect_btn.clicked.connect(lambda: self._set_note_types_checked(False))



        nt_btn_row.addWidget(nt_select_btn)



        nt_btn_row.addWidget(nt_deselect_btn)



        nt_btn_row.addStretch()



        # Sort options



        sort_label = QLabel("Sort by:")



        self.sort_combo = QComboBox()



        self.sort_combo.addItem("Note Count (Desc)", "count_desc")



        self.sort_combo.addItem("Note Count (Asc)", "count_asc")



        self.sort_combo.addItem("Name (A-Z)", "name_asc")



        self.sort_combo.addItem("Name (Z-A)", "name_desc")



        self.sort_combo.currentIndexChanged.connect(self._on_sort_note_types_changed)



        nt_btn_row.addWidget(sort_label)



        nt_btn_row.addWidget(self.sort_combo)



        nt_gl.addLayout(nt_btn_row)



        self.include_all_note_types_cb = QCheckBox("Include all note types")



        self.include_all_note_types_cb.setChecked(True)



        self.include_all_note_types_cb.stateChanged.connect(self._on_include_all_note_types_toggled)



        nt_gl.addWidget(self.include_all_note_types_cb)



        self.note_types_table = QTableWidget()



        self.note_types_table.setColumnCount(2)



        self.note_types_table.setHorizontalHeaderLabels(["Note Type", "Note Count"])



        self.note_types_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)



        self.note_types_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)



        self.note_types_table.setColumnWidth(1, 120)  # Set minimum width for count column



        self.note_types_table.setMinimumHeight(80)



        self.note_types_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)



        self.note_types_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)



        self.note_types_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)



        self.note_types_table.itemChanged.connect(self._update_field_groups_enabled)



        self.note_types_table.setSortingEnabled(True)



        self.note_types_table.sortByColumn(1, Qt.SortOrder.DescendingOrder)  # Sort by count descending by default



        header = self.note_types_table.horizontalHeader()
        header.sortIndicatorChanged.connect(
            lambda _column, _order: QTimer.singleShot(0, self._sync_field_groups_to_note_type_order)
        )



        # Add spacing between columns



        self.note_types_table.setColumnWidth(0, 200)



        nt_gl.addWidget(self.note_types_table)



        left_v_split.addWidget(nt_group)







        # ---- Left column: Decks (bottom) ----



        deck_group = QGroupBox("Decks to search")



        deck_gl = QVBoxLayout(deck_group)



        self.include_all_decks_cb = QCheckBox("Include all decks")



        self.include_all_decks_cb.setChecked(True)



        self.include_all_decks_cb.stateChanged.connect(self._on_include_all_decks_toggled)



        deck_gl.addWidget(self.include_all_decks_cb)



        deck_btn_row = QHBoxLayout()



        deck_select_btn = QPushButton("Select All")



        deck_select_btn.clicked.connect(lambda: self._set_decks_checked(True))



        deck_deselect_btn = QPushButton("Deselect All")



        deck_deselect_btn.clicked.connect(lambda: self._set_decks_checked(False))



        deck_btn_row.addWidget(deck_select_btn)



        deck_btn_row.addWidget(deck_deselect_btn)



        deck_btn_row.addStretch()



        deck_gl.addLayout(deck_btn_row)



        # Use QTreeWidget for hierarchical deck display (like main Anki interface)



        self.decks_list = QTreeWidget()



        self.decks_list.setMinimumHeight(80)



        self.decks_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)



        # Simplified header: only show deck name and total notes



        self.decks_list.setHeaderLabels(["Deck", "Notes"])



        self.decks_list.setRootIsDecorated(True)  # Show expand/collapse arrows



        self.decks_list.setAlternatingRowColors(True)



        # Set column widths: deck name and notes



        self.decks_list.setColumnWidth(0, 260)  # Deck name



        self.decks_list.setColumnWidth(1, 70)   # Notes



        self.decks_list.itemChanged.connect(self._on_deck_item_changed)



        deck_gl.addWidget(self.decks_list)



        left_v_split.addWidget(deck_group)







        left_v_split.setSizes([220, 180])



        left_v_split.setChildrenCollapsible(False)



        left_v_split.setHandleWidth(6)



        main_h_split.addWidget(left_v_split)







        # ---- Right column: Fields by note type ----



        fld_outer = QGroupBox("Fields to search per note type (greyed if note type unchecked)")



        fld_outer_l = QVBoxLayout(fld_outer)



        self.search_all_fields_cb = QCheckBox("Search in all fields (ignore selections below)")



        self.search_all_fields_cb.setChecked(False)



        self.search_all_fields_cb.stateChanged.connect(self._on_search_all_fields_toggled)



        fld_outer_l.addWidget(self.search_all_fields_cb)



        self.use_first_field_cb = QCheckBox("Use first field when no fields selected for a note type")



        self.use_first_field_cb.setChecked(True)



        self.use_first_field_cb.setToolTip("If a note type has no checked fields, use its first field instead of skipping.")



        fld_outer_l.addWidget(self.use_first_field_cb)



        self.fields_by_note_type_scroll = QScrollArea()
        self.fields_by_note_type_scroll.setObjectName("settingsFieldScroll")



        self.fields_by_note_type_scroll.setMinimumHeight(120)



        self.fields_by_note_type_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)



        self.fields_by_note_type_scroll.setWidgetResizable(True)



        self.fields_by_note_type_inner = QWidget()



        self.fields_by_note_type_layout = QVBoxLayout(self.fields_by_note_type_inner)



        self.fields_by_note_type_layout.setAlignment(Qt.AlignmentFlag.AlignTop)



        self.fields_by_note_type_scroll.setWidget(self.fields_by_note_type_inner)



        fld_outer_l.addWidget(self.fields_by_note_type_scroll)



        main_h_split.addWidget(fld_outer)



        self._field_cbs = {}  # model_name -> { field_name: QCheckBox }



        self._field_groupboxes = {}  # model_name -> QGroupBox (for greying when note type unchecked)







        main_h_split.setSizes([380, 420])



        main_h_split.setChildrenCollapsible(False)



        main_h_split.setHandleWidth(6)



        nt_main.addWidget(main_h_split)







        # ---- Count notes, Save/Load/Delete, Refresh ----



        action_row = QHBoxLayout()



        self.count_notes_btn = QPushButton("\U0001F4CA Count notes (with current settings)")



        self.count_notes_btn.clicked.connect(self._on_count_notes)



        action_row.addWidget(self.count_notes_btn)



        action_row.addStretch()



        nt_main.addLayout(action_row)







        save_pl = QHBoxLayout()



        self.preset_name_edit = QLineEdit()



        self.preset_name_edit.setPlaceholderText("e.g. Default, Shared deck")



        self.preset_name_edit.setMaximumWidth(200)



        self.preset_name_edit.setToolTip("Name for this set of note types and decks (for quick switching)")



        save_pl.addWidget(QLabel("Save as preset:"))



        save_pl.addWidget(self.preset_name_edit)



        save_preset_btn = QPushButton("\U0001F4BE Save preset")



        save_preset_btn.clicked.connect(self._on_save_preset)



        save_pl.addWidget(save_preset_btn)



        save_pl.addStretch()



        save_pl.addWidget(QLabel("Load preset:"))



        self.load_preset_combo = QComboBox()



        self.load_preset_combo.setMaximumWidth(180)



        self.load_preset_combo.setEditable(False)



        save_pl.addWidget(self.load_preset_combo)



        load_preset_btn = QPushButton("Load")



        load_preset_btn.clicked.connect(self._on_load_preset)



        save_pl.addWidget(load_preset_btn)



        self.delete_preset_combo = QComboBox()



        self.delete_preset_combo.setMaximumWidth(160)



        save_pl.addWidget(QLabel("Delete:"))



        save_pl.addWidget(self.delete_preset_combo)



        delete_preset_btn = QPushButton("Delete preset")



        delete_preset_btn.clicked.connect(self._on_delete_preset)



        save_pl.addWidget(delete_preset_btn)



        nt_main.addLayout(save_pl)







        refresh_btn = QPushButton("\U0001F504 Refresh lists (detect new note types/fields/decks)")



        refresh_btn.clicked.connect(self._refresh_note_type_lists)



        nt_main.addWidget(refresh_btn)







        # Defer heavy operations to avoid lag (with timing logs)



        QTimer.singleShot(50, lambda: self._populate_note_type_lists_with_timing())



        QTimer.singleShot(100, lambda: self._populate_fields_by_note_type_with_timing())



        QTimer.singleShot(150, lambda: self._populate_decks_list_with_timing())



        QTimer.singleShot(200, lambda: self._refresh_preset_combos_with_timing())







        nt_main.addStretch()







        tabs.addTab(nt_tab, "\U0001F4CB Note Types & Fields")







        # --- Search Settings Tab (Reorganized by Glutamine) ---

        search_tab = QWidget()

        search_scroll = QScrollArea()

        search_scroll.setWidgetResizable(True)

        search_scroll.setFrameShape(QFrame.Shape.NoFrame)

        search_tab_inner = QWidget()

        search_layout = QVBoxLayout(search_tab_inner)

        search_layout.setSpacing(15)

        search_layout.setContentsMargins(24, 16, 24, 16)

        search_scroll.setWidget(search_tab_inner)



        main_search_layout = QVBoxLayout(search_tab)

        main_search_layout.setContentsMargins(0, 0, 0, 0)

        main_search_layout.addWidget(search_scroll)



        search_info = QLabel("\U0001F50D Search & embeddings")

        search_info.setStyleSheet(settings_text_style(theme, "heading"))

        search_layout.addWidget(search_info)



        search_sub = QLabel("Choose how notes are matched (keyword, hybrid, or embedding). Optional: tune result count and relevance. Works with any deck.")

        search_sub.setWordWrap(True)

        search_sub.setStyleSheet(settings_text_style(theme, "subtitle"))

        search_layout.addWidget(search_sub)



        # Glutamine's Resident Reset Button

        reset_btn_layout = QHBoxLayout()

        self.medical_reset_btn = QPushButton("\u2695\ufe0f Reset to Clinical Defaults")

        self.medical_reset_btn.setToolTip("Sets all search, relevance, and AI settings to high-yield medical defaults (Resident-approved).")

        self.medical_reset_btn.setStyleSheet(settings_button_style(theme, "muted"))

        self.medical_reset_btn.clicked.connect(self.reset_to_medical_defaults)

        reset_btn_layout.addWidget(self.medical_reset_btn)

        search_layout.addLayout(reset_btn_layout)



        # --- High-Yield Action Zone (New persistent section for indexing) ---

        index_zone = QFrame()

        index_zone.setStyleSheet(settings_panel_style(theme, "index"))

        index_layout = QVBoxLayout(index_zone)
        index_layout.setContentsMargins(12, 10, 12, 12)
        index_layout.setSpacing(8)



        self.embedding_status_label = QLabel("Ready to index...")

        self.embedding_status_label.setStyleSheet(f"color: {theme['text']}; font-weight: bold; border: none;")

        self.embedding_status_label.setWordWrap(True)

        index_layout.addWidget(self.embedding_status_label)



        index_btns = QHBoxLayout()
        index_btns.setSpacing(8)

        self.create_embedding_btn = QPushButton("Create/Update Embeddings")

        self.create_embedding_btn.setToolTip(EmbeddingsTabMessages.CREATE_UPDATE_TOOLTIP)

        self.create_embedding_btn.setStyleSheet(settings_button_style(theme, "primary"))

        self.create_embedding_btn.clicked.connect(self._create_or_update_embeddings)

        index_btns.addWidget(self.create_embedding_btn)

        self.review_ineligible_btn = QPushButton("Review Ineligible Notes")

        self.review_ineligible_btn.setToolTip("Open notes excluded from embeddings by the current deck, note type, and field filters.")

        self.review_ineligible_btn.setStyleSheet(settings_button_style(theme, "muted"))

        self.review_ineligible_btn.clicked.connect(self._review_ineligible_notes)

        index_btns.addWidget(self.review_ineligible_btn)



        self.test_connection_btn = QPushButton("Test Connection")

        self.test_connection_btn.setToolTip(EmbeddingsTabMessages.TEST_CONNECTION_TOOLTIP)

        self.test_connection_btn.setStyleSheet(settings_button_style(theme, "muted"))

        self.test_connection_btn.clicked.connect(self._test_embedding_connection)

        index_btns.addWidget(self.test_connection_btn)



        index_layout.addLayout(index_btns)

        search_layout.addWidget(index_zone)



        # explanation: embedding provider setup moved to the API Settings tab.



        # --- 2. SEARCH STRATEGY ---

        strategy_section = CollapsibleSection("Search Strategy", is_expanded=False)

        self.search_method_combo = QComboBox()

        self.search_method_combo.addItem("Keyword Only", "keyword")

        self.search_method_combo.addItem("Keyword + Re-rank", "keyword_rerank")

        self.search_method_combo.addItem("Hybrid (RRF)", "hybrid")

        self.search_method_combo.addItem("Embedding Only", "embedding")

        self.search_method_combo.currentIndexChanged.connect(self._on_search_method_changed)

        strategy_section.addWidget(settings_field_row(theme, self.search_method_combo))

        search_layout.addWidget(strategy_section)



        # --- 3. AI-ASSISTED RETRIEVAL ---

        ai_retrieval_section = CollapsibleSection("AI-Assisted Retrieval", is_expanded=False)

        self.enable_query_expansion_cb = QCheckBox("Query Expansion (AI adds medical synonyms)")

        ai_retrieval_section.addWidget(settings_field_row(theme, self.enable_query_expansion_cb))

        self.use_ai_generic_term_detection_cb = QCheckBox("Filter Filler Words (AI detects generic terms)")

        ai_retrieval_section.addWidget(settings_field_row(theme, self.use_ai_generic_term_detection_cb))

        self.enable_hyde_cb = QCheckBox("HyDE (AI generates hypothetical document first)")

        self.enable_hyde_row = settings_field_row(theme, self.enable_hyde_cb)

        ai_retrieval_section.addWidget(self.enable_hyde_row)

        search_layout.addWidget(ai_retrieval_section)



        # --- 4. CLINICAL ACCURACY TUNING ---

        accuracy_section = CollapsibleSection("Clinical Accuracy Tuning", is_expanded=False)

        accuracy_layout = QVBoxLayout()

        accuracy_layout.setContentsMargins(0, 0, 0, 0)

        accuracy_layout.setSpacing(8)

        self.min_relevance_spin = QSpinBox()

        self.min_relevance_spin.setRange(15, 75)

        accuracy_layout.addWidget(settings_field_row(theme, self.min_relevance_spin, "Minimum Relevance Threshold:"))

        self.max_results_spin = QSpinBox()

        self.max_results_spin.setRange(5, 50)

        accuracy_layout.addWidget(settings_field_row(theme, self.max_results_spin, "Max Results Pool:"))

        self.hybrid_weight_spin = QSpinBox()

        self.hybrid_weight_spin.setRange(0, 100)

        self.hybrid_weight_label = QLabel("Embedding Weight:")

        self.hybrid_weight_row = settings_field_row(theme, self.hybrid_weight_spin, self.hybrid_weight_label)

        accuracy_layout.addWidget(self.hybrid_weight_row)

        self.relevance_from_answer_cb = QCheckBox("Relevance from answer (Rerank by AI output)")

        accuracy_layout.addWidget(settings_field_row(theme, self.relevance_from_answer_cb))

        accuracy_section.addLayout(accuracy_layout)

        search_layout.addWidget(accuracy_section)



        # --- 5. RE-RANKING SECTION ---

        rerank_section = CollapsibleSection("Re-Ranking (Advanced Accuracy)", is_expanded=False)

        self.rerank_status_label = QLabel("Cross-Encoder: checking...")
        self.rerank_status_label.setStyleSheet(settings_text_style(theme, "subtle"))
        rerank_section.addWidget(settings_field_row(theme, self.rerank_status_label))

        rerank_hint = QLabel("Recommended: external Python avoids Anki torch/DLL issues.")
        rerank_hint.setWordWrap(True)
        rerank_hint.setStyleSheet(settings_text_style(theme, "subtle"))
        rerank_section.addWidget(settings_field_row(theme, rerank_hint))

        self.python_path_widget = QWidget()
        python_path_layout = QVBoxLayout(self.python_path_widget)
        python_path_layout.setContentsMargins(0, 0, 0, 0)
        python_path_layout.setSpacing(8)

        python_label = QLabel("Use external Python")
        python_label.setStyleSheet("font-weight: bold;")
        python_path_layout.addWidget(python_label)

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
        python_path_layout.addLayout(path_row)

        action_row = QHBoxLayout()
        self.install_external_btn = QPushButton("Install / show command for external Python")
        self.install_external_btn.clicked.connect(self._on_install_into_external_python)
        action_row.addWidget(self.install_external_btn)

        self.check_rerank_btn = QPushButton("Check again")
        self.check_rerank_btn.clicked.connect(self._on_check_rerank_again)
        action_row.addWidget(self.check_rerank_btn)

        self.install_anki_python_btn = QPushButton("Try Anki Python fallback")
        self.install_anki_python_btn.setToolTip("Fallback only: external Python is recommended for Cross-Encoder setup.")
        self.install_anki_python_btn.clicked.connect(lambda: install_dependencies(python_exe=None))
        action_row.addWidget(self.install_anki_python_btn)

        python_path_layout.addLayout(action_row)

        self.python_path_row = settings_field_row(theme, self.python_path_widget, vertical=True)

        rerank_section.addWidget(self.python_path_row)
        self.python_path_widget.setVisible(True)

        self.enable_rerank_cb = QCheckBox("Improve result order with Cross-Encoder")
        self.enable_rerank_cb.setEnabled(False)
        rerank_section.addWidget(settings_field_row(theme, self.enable_rerank_cb))

        self.use_context_boost_cb = QCheckBox("Context-Aware Ranking")
        rerank_section.addWidget(settings_field_row(theme, self.use_context_boost_cb))

        search_layout.addWidget(rerank_section)



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



        legacy_json_path = get_embeddings_storage_path_for_read()

        has_legacy_json = bool(

            legacy_json_path

            and isinstance(legacy_json_path, str)

            and legacy_json_path.endswith(".json")

            and os.path.exists(legacy_json_path)

        )



        if has_legacy_json:

            migrate_json_btn = QPushButton("\U0001F4E6 Legacy migration: JSON \u2192 DB")

            migrate_json_btn.setToolTip(

                "One-time legacy migration for users upgrading from older versions that stored embeddings in a JSON "

                "file. Copies existing embeddings from the old JSON cache into the SQLite database so you don't need "

                "to re-embed. Most users can ignore this."

            )

            migrate_json_btn.clicked.connect(self._migrate_json_to_db)

            embedding_btn_layout.addWidget(migrate_json_btn)



        search_layout.addWidget(self.embedding_section)



        self._tabs = tabs







        tabs.addTab(search_tab, "\U0001F50D Search & Embeddings")



        # Tab order: API \u2192 Search & Embeddings \u2192 Note Types & Fields \u2192 Styling



        tabs.removeTab(tabs.indexOf(search_tab))



        tabs.insertTab(1, search_tab, "\U0001F50D Search & Embeddings")



        tabs.removeTab(tabs.indexOf(nt_tab))



        tabs.insertTab(2, nt_tab, "\U0001F4CB Note Types & Fields")







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



        scroll_content.installEventFilter(self)



        tabs.installEventFilter(self)



        scroll_area.viewport().installEventFilter(self)



        self._apply_settings_control_sizing(tabs)



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



        config = load_config()



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

        self.local_llm_url.setText(config.get('local_llm_url', 'http://localhost:1234/v1'))

        self.local_llm_model.setText(config.get('local_llm_model', 'model-identifier'))



        if provider == 'ollama':



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



        ntf = config.get('note_type_filter', {})



        ntf_start = time.time()



        self._apply_note_type_config(ntf)



        ntf_elapsed = time.time() - ntf_start



        log_debug(f"  [Timing] _apply_note_type_config() in __init__: {ntf_elapsed:.3f}s")







        # --- Load Configuration (Guarded) ---

        self._apply_config_to_ui()



    # --- Config Loading And Preset Application ---

    def _safe_set_checked(self, widget, value):

        """Radical safety: Check if C++ object exists before calling methods."""

        try:

            try:
                    try:
                        import sip
                    except ImportError:
                        from PyQt6 import sip

            except ImportError:
                    from PyQt6 import sip

            if widget is not None and not sip.isdeleted(widget):

                widget.setChecked(bool(value))

        except (RuntimeError, ImportError, AttributeError):

            pass



    def _load_config_into_ui(self):

        """Atomic configuration loading with strict safety checks."""

        config = load_config()

        search_config = config.get('search_config', {})



        # Guarded setters

        self._safe_set_checked(getattr(self, "enable_query_expansion_cb", None), search_config.get('enable_query_expansion', False))

        self._safe_set_checked(getattr(self, "use_ai_generic_term_detection_cb", None), search_config.get('use_ai_generic_term_detection', False))

        self._safe_set_checked(getattr(self, "enable_hyde_cb", None), search_config.get('enable_hyde', False))

        self._safe_set_checked(getattr(self, "enable_rerank_cb", None), search_config.get('enable_rerank', False))

        self._safe_set_checked(getattr(self, "use_context_boost_cb", None), search_config.get('use_context_boost', True))

        self._safe_set_checked(getattr(self, "relevance_from_answer_cb", None), search_config.get('relevance_from_answer', False))

        self._safe_set_checked(getattr(self, "use_dynamic_batch_size_cb", None), search_config.get('use_dynamic_batch_size', True))



        # Value setters (with try/except guards)

        try:

            if hasattr(self, "min_relevance_spin"):

                self.min_relevance_spin.setValue(max(15, min(75, search_config.get('min_relevance_percent', 55))))

            if hasattr(self, "max_results_spin"):

                self.max_results_spin.setValue(max(5, min(50, search_config.get('max_results', 50))))

            if hasattr(self, "hybrid_weight_spin"):

                self.hybrid_weight_spin.setValue(max(0, min(100, search_config.get('hybrid_embedding_weight', 40))))

            if hasattr(self, "rerank_python_path_input"):

                 path = (search_config.get('rerank_python_path') or '').strip()

                 self.rerank_python_path_input.setText(path)

                 if path and hasattr(self, "python_path_widget"):

                     self.python_path_widget.setVisible(True)

        except Exception:

            pass



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







    def _on_apply_hybrid_retrieval(self):



        """One-click RAG-optimized: AI embedding for retrieval, Ollama for answer; Hybrid + Re-rank for best quality with minimal cloud tokens."""



        # explanation: apply the RAG shortcut through the new API-tab embedding controls.
        if hasattr(self, "embedding_same_checkbox"):

            self.embedding_same_checkbox.setChecked(False)

            idx_strategy = self.embedding_strategy_combo.findData("cloud")

            if idx_strategy >= 0:

                self.embedding_strategy_combo.setCurrentIndex(idx_strategy)

            idx_provider = self.embedding_cloud_provider_combo.findData("Voyage AI")

            if idx_provider >= 0:

                self.embedding_cloud_provider_combo.setCurrentIndex(idx_provider)



        idx_ans = self.answer_provider_combo.findData("local_server")



        if idx_ans >= 0:



            self.answer_provider_combo.setCurrentIndex(idx_ans)



        idx_hybrid = self.search_method_combo.findData("hybrid")



        if idx_hybrid >= 0:



            self.search_method_combo.setCurrentIndex(idx_hybrid)



        if hasattr(self, 'enable_rerank_cb') and self.enable_rerank_cb is not None:



            self.enable_rerank_cb.setChecked(True)



        self._on_embedding_engine_changed()



        self._on_answer_provider_changed()



        if hasattr(self, '_on_search_method_changed'):



            self._on_search_method_changed()



        showInfo("RAG-optimized applied: Embeddings = Voyage, Answer = Ollama, Hybrid, Re-rank on. Click Save to apply and run Create/Update Embeddings if needed.")







    def _on_scan_and_pull(self):

        """Autodetect running local AI servers and update UI."""

        from aqt.utils import tooltip

        import requests
        theme = _addon_theme()



        # 1. Probe Ollama (11434)

        ollama_url = "http://localhost:11434"

        try:

            resp = requests.get(f"{ollama_url}/api/tags", timeout=2)

            if resp.status_code == 200:

                models = [m['name'] for m in resp.json().get('models', [])]

                self.local_ai_status_label.setText("Detected: Ollama (Running)")

                self.local_ai_status_label.setStyleSheet(f"font-weight: bold; color: {theme['success']};")

                self.ollama_embed_model_combo.clear()

                self.ollama_embed_model_combo.addItems(models)

                self.ollama_base_url_input.setText(ollama_url)

                tooltip("Found Ollama! Models updated.")

                return

        except: pass



        # 2. Probe LM Studio / OpenAI-Compatible (1234)

        lm_url = "http://localhost:1234"

        try:

            resp = requests.get(f"{lm_url}/v1/models", timeout=2)

            if resp.status_code == 200:

                models = [m['id'] for m in resp.json().get('data', [])]

                self.local_ai_status_label.setText("Detected: LM Studio / Local Server")

                self.local_ai_status_label.setStyleSheet(f"font-weight: bold; color: {theme['success']};")

                self.ollama_embed_model_combo.clear()

                self.ollama_embed_model_combo.addItems(models)

                self.ollama_base_url_input.setText(lm_url)

                tooltip("Found LM Studio! Models updated.")

                return

        except: pass



        self.local_ai_status_label.setText("No local AI found. Defaulting to Cloud or Manual.")

        self.local_ai_status_label.setStyleSheet(f"font-weight: bold; color: {theme['danger']};")

        tooltip("No local AI detected. Ensure Ollama or LM Studio is running.")



    # --- Provider And API Configuration UI ---

    def _on_embedding_engine_changed(self):

        """Compatibility wrapper for the old Search-tab embedding signal."""

        # explanation: provider visibility is now owned by the API-tab embedding controls.
        if hasattr(self, "_on_embedding_strategy_changed"):

            self._on_embedding_strategy_changed()



    def _create_api_config_group(self, provider_id, placeholder, tooltip_key, models=None, key_suffix=""):

        """Standardized UI group for API providers. key_suffix handles cases like 'openai_embedding_api_key'."""

        widget = QWidget()

        form = QFormLayout(widget)

        form.setContentsMargins(0, 5, 0, 5)



        # 1. API Key Row

        key_row = QHBoxLayout()

        key_input = QLineEdit()

        key_input.setEchoMode(QLineEdit.EchoMode.Password)

        key_input.setPlaceholderText(placeholder)

        key_input.setMinimumWidth(280)

        key_input.setToolTip(tooltip_key)



        show_btn = QPushButton("Show")

        show_btn.setMaximumWidth(60)

        show_btn.clicked.connect(lambda: key_input.setEchoMode(

            QLineEdit.EchoMode.Normal if key_input.echoMode() == QLineEdit.EchoMode.Password

            else QLineEdit.EchoMode.Password

        ))



        key_row.addWidget(key_input)

        key_row.addWidget(show_btn)

        form.addRow(f"{provider_id.capitalize()} API Key:", key_row)



        # 2. Model Input/Combo

        model_input = None

        if models:

            model_input = QComboBox()

            for m in models:

                model_input.addItem(m, m)

        else:

            model_input = QLineEdit()

            model_input.setPlaceholderText("e.g. text-embedding-3-small")



        form.addRow("Model:", model_input)



        # Mapping to existing names to prevent "Data Loss"

        key_attr = f"{provider_id}{key_suffix}_api_key_input"

        model_attr = f"{provider_id}_embedding_model_{'combo' if models else 'input'}"



        setattr(self, key_attr, key_input)

        setattr(self, model_attr, model_input)



        return widget

        """Binary UI: Show only the relevant settings for Cloud or Local AI."""

        strategy = self.embedding_engine_combo.currentData() or "cloud"



        is_cloud = (strategy == "cloud")

        is_local = (strategy == "local")



        # 1. Cloud Options (Provider selector + API keys)

        if hasattr(self, "cloud_provider_widget"):

            self.cloud_provider_widget.setVisible(is_cloud)



        cloud_engine = self.cloud_provider_combo.currentData() or "voyage"



        self.voyage_options.setVisible(is_cloud and cloud_engine == "voyage")

        self.openai_options.setVisible(is_cloud and cloud_engine == "openai")

        self.cohere_options.setVisible(is_cloud and cloud_engine == "cohere")

        self.cloud_batch_widget.setVisible(is_cloud)



        # 2. Local Options (Ollama/LM Studio)

        if hasattr(self, "ollama_options"):

            self.ollama_options.setVisible(is_local)



        # 3. Instruction Hint

        if hasattr(self, "embedding_hybrid_hint"):

            self.embedding_hybrid_hint.setVisible(is_cloud)

        if hasattr(self, "apply_hybrid_btn"):

            self.apply_hybrid_btn.setVisible(is_cloud)







    def eventFilter(self, obj, event):



        """Forward mouse wheel gestures to scrolling instead of mutating controls."""



        if event.type() != QEvent.Type.Wheel:



            return super().eventFilter(obj, event)



        if self._is_wheel_guarded_widget(obj):



            guarded_scroll_area = self._nearest_scroll_area(obj) or getattr(self, "_settings_scroll_area", None)



            if guarded_scroll_area and self._scroll_area_by_wheel(guarded_scroll_area, event):



                return True



        scroll_area = getattr(self, "_settings_scroll_area", None)



        tabs = getattr(self, "_settings_tabs", None)



        scroll_content = getattr(self, "_settings_scroll_content", None)



        if not scroll_area or not tabs or scroll_content is None:



            return super().eventFilter(obj, event)



        # Allow scroll when wheel is over scroll content, tabs, viewport, or any descendant



        target = obj



        while target:



            if target == scroll_content or target == tabs or target == scroll_area.viewport():



                break



            target = target.parentWidget() if hasattr(target, "parentWidget") else None



        else:



            return super().eventFilter(obj, event)



        if self._scroll_area_by_wheel(scroll_area, event):



            return True



        return super().eventFilter(obj, event)



    def _install_wheel_scroll_guard(self, root):



        """Prevent touchpad/wheel gestures from changing settings controls."""



        guarded_types = (QAbstractSpinBox, QComboBox, QSlider, QCheckBox)



        self._wheel_guarded_widgets = set()



        for widget in root.findChildren(guarded_types):



            widget.installEventFilter(self)



            self._wheel_guarded_widgets.add(widget)



    def _apply_settings_control_sizing(self, root):



        """Keep Settings controls readable without letting numeric fields stretch too far."""



        for spin in root.findChildren(QAbstractSpinBox):



            spin.setMaximumWidth(170)



            spin.setMinimumWidth(90)



        for combo in root.findChildren(QComboBox):



            combo.setMinimumWidth(180)



    def _is_wheel_guarded_widget(self, obj):



        guarded = getattr(self, "_wheel_guarded_widgets", set())



        target = obj



        while target:



            if target in guarded:



                return True



            target = target.parentWidget() if hasattr(target, "parentWidget") else None



        return False



    def _nearest_scroll_area(self, widget):



        target = widget



        while target:



            if isinstance(target, QScrollArea):



                return target



            target = target.parentWidget() if hasattr(target, "parentWidget") else None



        return None



    def _scroll_area_by_wheel(self, scroll_area, event):



        if not scroll_area or not scroll_area.verticalScrollBar().isVisible():



            return False



        sb = scroll_area.verticalScrollBar()



        delta = event.pixelDelta().y() if hasattr(event, "pixelDelta") and not event.pixelDelta().isNull() else 0



        if not delta:



            delta = event.angleDelta().y() if hasattr(event, "angleDelta") else getattr(event, "delta", 0)



        if not delta:



            return False



        sb.setValue(sb.value() - delta)



        return True







    def _refresh_ollama_models(self):



        """Fetch model list from Ollama and populate the embed model combo."""



        # explanation: answer-provider local test ends here; old embedding-specific Ollama test was removed from this UI.
        return

        base_url = (self.ollama_base_url_input.text() or "http://localhost:11434").strip()



        current = self.ollama_embed_model_combo.currentText().strip() or "nomic-embed-text"



        try:



            names = get_ollama_models(base_url)



            self.ollama_embed_model_combo.clear()



            self.ollama_embed_model_combo.addItems(names)



            if current and current not in names:



                self.ollama_embed_model_combo.insertItem(0, current)



                self.ollama_embed_model_combo.setCurrentIndex(0)



            elif current in names:



                idx = self.ollama_embed_model_combo.findText(current)



                if idx >= 0:



                    self.ollama_embed_model_combo.setCurrentIndex(idx)



            if not names:



                self.ollama_embed_model_combo.setCurrentText(current or "nomic-embed-text")



            if names:



                showInfo(f"Found {len(names)} model(s) at {base_url}. Choose an embedding model (e.g. nomic-embed-text).")



            else:



                showInfo(



                    "No models returned from Ollama. Make sure Ollama is running (ollama serve) and you have pulled at least one model.\n\n"



                    "You can still type an embedding model name manually (e.g. nomic-embed-text)."



                )



        except Exception as e:



            showInfo(



                f"Could not fetch models from {base_url}.\n\n"



                f"Error: {e}\n\n"



                "Check that Ollama is running (ollama serve). You can type a model name manually (e.g. nomic-embed-text)."



            )







    # --- Note Type, Field, And Deck Filters ---

    def _populate_note_type_lists_with_timing(self):



        """Fill note types table with name and count columns (with timing)."""



        # Check if widget still exists (dialog might have been closed)



        try:



            if not hasattr(self, 'note_types_table') or self.note_types_table is None:



                return



            # Check if the C++ object is still valid



            if not sip.isdeleted(self.note_types_table) if hasattr(sip, 'isdeleted') else True:



                import time



                start = time.time()



                self._populate_note_type_lists()



                elapsed = time.time() - start



                log_debug(f"  [Timing] _populate_note_type_lists(): {elapsed:.3f}s")



                # After table is populated, apply saved note/deck/field config



                try:



                    cfg = load_config()



                    ntf = cfg.get('note_type_filter', {})



                    self._apply_note_type_config(ntf)



                except Exception as e:



                    log_debug(f"Error re-applying note_type_filter after note type populate: {e}")



        except RuntimeError:



            # Widget was deleted, ignore



            pass



        except Exception as e:



            log_debug(f"Error in _populate_note_type_lists_with_timing: {e}")







    def _populate_note_type_lists(self):



        """Fill note types table with name and count columns."""



        # Check if widget still exists



        if not hasattr(self, 'note_types_table') or self.note_types_table is None:



            return



        try:



            self.note_types_table.setRowCount(0)



            counts = get_notes_count_per_model()



            for name in sorted(counts.keys()):



                c = counts.get(name, 0)



                row = self.note_types_table.rowCount()



                self.note_types_table.insertRow(row)







                # Name column with checkbox



                name_item = QTableWidgetItem(name)



                name_item.setData(Qt.ItemDataRole.UserRole, name)



                name_item.setFlags(name_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)



                name_item.setCheckState(Qt.CheckState.Unchecked)



                self.note_types_table.setItem(row, 0, name_item)







                # Count column



                count_item = QTableWidgetItem()



                count_item.setData(Qt.ItemDataRole.DisplayRole, c)  # Store numeric value for proper sorting



                count_item.setText(str(c))



                count_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)



                count_item.setFlags(count_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make read-only



                self.note_types_table.setItem(row, 1, count_item)







            # Sort by count descending by default



            self.note_types_table.sortByColumn(1, Qt.SortOrder.DescendingOrder)



        except RuntimeError:



            # Widget was deleted, ignore



            pass



        except Exception as e:



            log_debug(f"Error in _populate_note_type_lists: {e}")







    def _populate_fields_by_note_type(self):



        """Build collapsible field sections per note type and repopulate them."""



        # Check if widget still exists



        if not hasattr(self, 'fields_by_note_type_layout') or self.fields_by_note_type_layout is None:



            return



        try:



            while self.fields_by_note_type_layout.count() > 0:



                it = self.fields_by_note_type_layout.takeAt(0)



                if it and it.widget():



                    it.widget().deleteLater()



            self._field_cbs.clear()



            self._field_groupboxes.clear()



            included = None



            if not self.include_all_note_types_cb.isChecked():



                included = set()



                for i in range(self.note_types_table.rowCount()):



                    it = self.note_types_table.item(i, 0)



                    if it and it.checkState() == Qt.CheckState.Checked:



                        included_name = it.data(Qt.ItemDataRole.UserRole) or it.text().strip()



                        included.add(included_name)



            for model_name, count, field_names in get_models_with_fields():



                is_included = included is None or model_name in included



                gb = CollapsibleSection(f"{model_name}  ({count} notes)", is_expanded=is_included)



                cbs = {}



                for fn in field_names:



                    cb = QCheckBox(fn)



                    cb.installEventFilter(self)



                    if hasattr(self, "_wheel_guarded_widgets"):



                        self._wheel_guarded_widgets.add(cb)



                    cbs[fn] = cb



                    gb.addWidget(cb)



                self._field_cbs[model_name] = cbs



                self._field_groupboxes[model_name] = gb



                self.fields_by_note_type_layout.addWidget(gb)



            self._update_field_groups_enabled()



            self._sync_field_groups_to_note_type_order()



        except RuntimeError:



            # Widget was deleted, ignore



            pass



        except Exception as e:



            log_debug(f"Error in _populate_fields_by_note_type: {e}")







    def _populate_fields_by_note_type_with_timing(self):



        """Populate fields by note type (with timing)."""



        import time



        start = time.time()



        self._populate_fields_by_note_type()



        elapsed = time.time() - start



        if elapsed > 0.1:  # Only log if it takes significant time



            log_debug(f"  [Timing] _populate_fields_by_note_type(): {elapsed:.3f}s")



        # Ensure field checkboxes match saved configuration



        try:



            cfg = load_config()



            ntf = cfg.get('note_type_filter', {})



            self._apply_note_type_config(ntf)



        except Exception as e:



            log_debug(f"Error re-applying note_type_filter after field populate: {e}")







    def _populate_decks_list_with_timing(self):



        """Populate decks list (with timing)."""



        # Check if widget still exists (dialog might have been closed)



        try:



            if not hasattr(self, 'decks_list') or self.decks_list is None:



                return



            import time



            start = time.time()



            self._populate_decks_list()



            elapsed = time.time() - start



            log_debug(f"  [Timing] _populate_decks_list(): {elapsed:.3f}s")



            # Ensure deck checkboxes match saved configuration



            try:



                cfg = load_config()



                ntf = cfg.get('note_type_filter', {})



                self._apply_note_type_config(ntf)



            except Exception as e:



                log_debug(f"Error re-applying note_type_filter after deck populate: {e}")



        except RuntimeError:



            # Widget was deleted, ignore



            pass



        except Exception as e:



            log_debug(f"Error in _populate_decks_list_with_timing: {e}")







    def _populate_decks_list(self):



        import time



        deck_start = time.time()







        # Check if widget still exists



        if not hasattr(self, 'decks_list') or self.decks_list is None:



            return



        try:



            self.decks_list.clear()







            if not mw or not mw.col:



                return







            counts_start = time.time()



            counts = get_notes_count_per_deck()



            counts_elapsed = time.time() - counts_start



            log_debug(f"  [Timing] get_notes_count_per_deck(): {counts_elapsed:.3f}s")







            # Get deck hierarchy and card counts



            deck_names_start = time.time()



            deck_names = get_deck_names()



            deck_names_elapsed = time.time() - deck_names_start



            log_debug(f"  [Timing] get_deck_names(): {deck_names_elapsed:.3f}s")







            # Build hierarchical deck structure



            deck_tree = {}  # parent_name -> [child_decks]



            top_level_decks = []







            for name in deck_names:



                if '::' in name:



                    # Sub-deck



                    parts = name.split('::')



                    parent = '::'.join(parts[:-1])



                    if parent not in deck_tree:



                        deck_tree[parent] = []



                    deck_tree[parent].append(name)



                else:



                    # Top-level deck



                    top_level_decks.append(name)







            # Get card counts for each deck (new, learn, due)

            card_counts = {}

            try:

                # In modern Anki, we don't need to import Scheduler; mw.col.sched is ready.

                # Use the safer get_deck_stats if available, or fallback.

                deck_ids = {name: mw.col.decks.id(name) for name in deck_names if mw.col.decks.by_name(name)}



                for name, deck_id in deck_ids.items():

                    try:

                        # Modern scheduler fallback

                        if hasattr(mw.col.sched, "counts"):

                            counts_info = mw.col.sched.counts(deck_id)

                            card_counts[name] = {

                                'new': getattr(counts_info, 'new', 0),

                                'learn': getattr(counts_info, 'learn', 0),

                                'due': getattr(counts_info, 'review', 0)

                            }

                        else:

                            # If counts() is missing, just use 0s for now

                            card_counts[name] = {'new': 0, 'learn': 0, 'due': 0}

                    except Exception:

                        card_counts[name] = {'new': 0, 'learn': 0, 'due': 0}



            except Exception as e:

                log_debug(f"Error getting card counts: {e}")

                # Fallback: set all to 0

                for name in deck_names:

                    card_counts[name] = {'new': 0, 'learn': 0, 'due': 0}







            # Sort decks



            top_level_decks.sort()



            for parent in deck_tree:



                deck_tree[parent].sort()







            # Create tree items



            def create_deck_item(name, is_parent=False):



                """Create a tree item for a deck (showing only name + total notes)."""



                note_count = counts.get(name, 0)



                # Hide the built-in empty 'Default' deck, which many users don't use



                if name == "Default" and note_count == 0:



                    return None







                # Extract display name (without parent prefix for sub-decks)



                display_name = name.split('::')[-1] if '::' in name else name







                item = QTreeWidgetItem([display_name, str(note_count)])







                # Store full deck name in item data for later retrieval



                item.setData(0, Qt.ItemDataRole.UserRole, name)







                # Make checkable



                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)



                item.setCheckState(0, Qt.CheckState.Unchecked)







                # Style parent decks (bold)



                if is_parent:



                    font = item.font(0)



                    font.setBold(True)



                    item.setFont(0, font)







                return item







            # Add top-level decks



            for deck_name in top_level_decks:



                parent_item = create_deck_item(deck_name, is_parent=(deck_name in deck_tree))



                if parent_item is None:



                    continue



                self.decks_list.addTopLevelItem(parent_item)







                # Add children if any



                if deck_name in deck_tree:



                    for child_name in deck_tree[deck_name]:



                        child_item = create_deck_item(child_name)



                        parent_item.addChild(child_item)







                    # Collapsed by default; user expands on click



                    parent_item.setExpanded(False)







            # Also add decks that are parents but not top-level (nested hierarchies)



            for parent_name in sorted(deck_tree.keys()):



                if parent_name not in top_level_decks:



                    # This is a nested parent, find its position in hierarchy



                    parts = parent_name.split('::')



                    if len(parts) > 1:



                        # Find parent item



                        grandparent_name = '::'.join(parts[:-1])



                        # Search for grandparent in tree



                        for i in range(self.decks_list.topLevelItemCount()):



                            parent_item = self._find_deck_item_recursive(self.decks_list.topLevelItem(i), grandparent_name)



                            if parent_item:



                                child_item = create_deck_item(parent_name, is_parent=True)



                                if child_item is None:



                                    break



                                parent_item.addChild(child_item)



                                # Add its children



                                for child_name in deck_tree[parent_name]:



                                    grandchild_item = create_deck_item(child_name)



                                    child_item.addChild(grandchild_item)



                                child_item.setExpanded(False)



                                break







            deck_elapsed = time.time() - deck_start



            log_debug(f"  [Timing] _populate_decks_list(): {deck_elapsed:.3f}s")







        except RuntimeError:



            # Widget was deleted, ignore



            pass



        except Exception as e:



            log_debug(f"Error in _populate_decks_list: {e}")



            import traceback



            log_debug(traceback.format_exc())







    def _find_deck_item_recursive(self, item, deck_name):



        """Recursively find a deck item by its full name"""



        if item is None:



            return None



        if item.data(0, Qt.ItemDataRole.UserRole) == deck_name:



            return item



        for i in range(item.childCount()):



            found = self._find_deck_item_recursive(item.child(i), deck_name)



            if found:



                return found



        return None







    def _iterate_all_deck_items(self):



        """Generator that yields all deck items (top-level and children)"""



        for i in range(self.decks_list.topLevelItemCount()):



            item = self.decks_list.topLevelItem(i)



            yield item



            # Recursively yield children



            for j in range(item.childCount()):



                yield from self._iterate_all_deck_items_recursive(item.child(j))







    def _iterate_all_deck_items_recursive(self, item):



        """Recursively yield item and all its children"""



        if item:



            yield item



            for i in range(item.childCount()):



                yield from self._iterate_all_deck_items_recursive(item.child(i))







    def _refresh_preset_combos_with_timing(self):



        """Refresh preset combos (with timing)."""



        # Check if widget still exists (dialog might have been closed)



        try:



            if not hasattr(self, 'load_preset_combo') or self.load_preset_combo is None:



                return



            import time



            start = time.time()



            self._refresh_preset_combos()



            elapsed = time.time() - start



            log_debug(f"  [Timing] _refresh_preset_combos(): {elapsed:.3f}s")



        except RuntimeError:



            # Widget was deleted, ignore



            pass



        except Exception as e:



            log_debug(f"Error in _refresh_preset_combos_with_timing: {e}")







    def _refresh_preset_combos(self):



        # Check if widgets still exist



        if not hasattr(self, 'load_preset_combo') or self.load_preset_combo is None:



            return



        try:



            config = load_config()



            presets = config.get('saved_presets') or {}



            current_name = config.get('current_preset_name')



            names = sorted(presets.keys())



            self.load_preset_combo.clear()



            self.load_preset_combo.addItem("-- Select --", None)



            for n in names:



                self.load_preset_combo.addItem(n, n)



            selected_index = self.load_preset_combo.findData(current_name)



            if selected_index >= 0:



                self.load_preset_combo.setCurrentIndex(selected_index)



            if hasattr(self, 'delete_preset_combo') and self.delete_preset_combo is not None:



                try:



                    self.delete_preset_combo.clear()



                    self.delete_preset_combo.addItem("-- Select --", None)



                    for n in names:



                        self.delete_preset_combo.addItem(n, n)



                    delete_index = self.delete_preset_combo.findData(current_name)



                    if delete_index >= 0:



                        self.delete_preset_combo.setCurrentIndex(delete_index)



                except RuntimeError:



                    # Widget was deleted, ignore



                    pass



        except RuntimeError:



            # Widget was deleted, ignore



            pass



        except Exception as e:



            log_debug(f"Error in _refresh_preset_combos: {e}")







    def _apply_note_type_config(self, ntf):



        """Apply note_type_filter config. Migrate fields_to_search -> note_type_fields if needed."""



        self._applying_note_type_config = True



        try:



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



        # Interpretation:



        #   None        -> include all note types



        #   [] (empty)  -> user has not chosen any specific types yet



        #                  (start with none selected to reduce workload)



        include_all_nt = (enabled is None)



        self.include_all_note_types_cb.setChecked(include_all_nt)



        self._on_include_all_note_types_toggled()



        if not include_all_nt and enabled:



            enabled_set = set(enabled)



            for i in range(self.note_types_table.rowCount()):



                it = self.note_types_table.item(i, 0)



                if it:



                    name = it.data(Qt.ItemDataRole.UserRole) or it.text().strip()



                    it.setCheckState(Qt.CheckState.Checked if (name in enabled_set) else Qt.CheckState.Unchecked)



        else:



            self._set_note_types_checked(True)



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



        include_all = self.include_all_note_types_cb.isChecked()



        if include_all:



            included = None



        else:



            included = set()



            for i in range(self.note_types_table.rowCount()):



                it = self.note_types_table.item(i, 0)



                if it and it.checkState() == Qt.CheckState.Checked:



                    name = it.data(Qt.ItemDataRole.UserRole) or it.text().strip()



                    included.add(name)



        for model_name, gb in self._field_groupboxes.items():



            is_included = included is None or model_name in included

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


    def _on_include_all_note_types_toggled(self):



        self.note_types_table.setEnabled(not self.include_all_note_types_cb.isChecked())



        self._update_field_groups_enabled()







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







    def _set_note_types_checked(self, checked):



        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked



        for i in range(self.note_types_table.rowCount()):



            it = self.note_types_table.item(i, 0)



            if it:



                it.setCheckState(state)







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



        include_all_nt = self.include_all_note_types_cb.isChecked()



        enabled_nt = None if include_all_nt else [self.note_types_table.item(i, 0).data(Qt.ItemDataRole.UserRole) or self.note_types_table.item(i, 0).text().strip() for i in range(self.note_types_table.rowCount()) if self.note_types_table.item(i, 0) and self.note_types_table.item(i, 0).checkState() == Qt.CheckState.Checked]



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



        c = count_notes_matching_config(ntf)



        showInfo(f"With current settings, about {c} notes would be searched.")







    def _on_save_preset(self):



        name = self.preset_name_edit.text().strip()



        if not name:



            showInfo("Enter a preset name.")



            return



        config = load_config()



        presets = config.get('saved_presets') or {}



        presets[name] = self._build_ntf_from_ui()



        config['saved_presets'] = presets



        config['current_preset_name'] = name



        if save_config(config):



            self.preset_name_edit.clear()



            self._refresh_preset_combos()



            selected_index = self.load_preset_combo.findData(name)



            if selected_index >= 0:



                self.load_preset_combo.setCurrentIndex(selected_index)



            showInfo(f"Preset '{name}' saved.")







    def _on_load_preset(self):



        name = self.load_preset_combo.currentData()



        if not name:



            showInfo("Select a preset to load.")



            return



        config = load_config()



        presets = config.get('saved_presets') or {}



        if name not in presets:



            showInfo("Preset not found.")



            return



        self._apply_note_type_config(presets[name])



        config['current_preset_name'] = name



        save_config(config)



        selected_index = self.load_preset_combo.findData(name)



        if selected_index >= 0:



            self.load_preset_combo.setCurrentIndex(selected_index)



        if hasattr(self, 'delete_preset_combo') and self.delete_preset_combo is not None:



            delete_index = self.delete_preset_combo.findData(name)



            if delete_index >= 0:



                self.delete_preset_combo.setCurrentIndex(delete_index)



        showInfo(f"Loaded preset '{name}'.")







    def _on_delete_preset(self):



        name = self.delete_preset_combo.currentData()



        if not name:



            showInfo("Select a preset to delete.")



            return



        config = load_config()



        presets = config.get('saved_presets') or {}



        if name in presets:



            del presets[name]



            config['saved_presets'] = presets



            if config.get('current_preset_name') == name:



                config['current_preset_name'] = None



            save_config(config)



            self._refresh_preset_combos()



            showInfo(f"Preset '{name}' deleted.")







    def _refresh_note_type_lists(self):



        """Repopulate all lists and preserve checked state where possible."""



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



        self._populate_decks_list()



        if checked_decks:



            ds = set(checked_decks)



            for it in self._iterate_all_deck_items():



                deck_name = it.data(0, Qt.ItemDataRole.UserRole) or it.text(0)



                it.setCheckState(0, Qt.CheckState.Checked if (deck_name in ds) else Qt.CheckState.Unchecked)



        else:



            self._set_decks_checked(True)



        self._refresh_preset_combos()



        showInfo("Lists refreshed.")







    # --- Answer And Embedding Provider Settings ---

    def _build_cloud_provider_section(
        self,
        theme,
        providers,
        provider_attr,
        key_attr,
        show_button_attr,
        detected_label_attr,
        key_placeholder,
        show_callback,
        changed_callback,
    ):
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        provider_combo = QComboBox()
        for label, provider_id, _prefix in providers:
            provider_combo.addItem(label, provider_id)
        provider_combo.currentIndexChanged.connect(changed_callback)
        setattr(self, provider_attr, provider_combo)
        layout.addWidget(settings_field_row(theme, provider_combo, "Cloud Provider:"))

        key_row = QHBoxLayout()
        key_input = QLineEdit()
        key_input.setEchoMode(QLineEdit.EchoMode.Password)
        key_input.setPlaceholderText(key_placeholder)
        key_input.textChanged.connect(changed_callback)
        setattr(self, key_attr, key_input)
        key_row.addWidget(key_input)

        show_button = QPushButton("Show")
        show_button.setMaximumWidth(80)
        show_button.clicked.connect(show_callback)
        setattr(self, show_button_attr, show_button)
        key_row.addWidget(show_button)
        layout.addWidget(settings_field_row(theme, layout=key_row, label="API Key:"))

        detected_label = QLabel()
        detected_label.setStyleSheet(settings_text_style(theme, "summary"))
        detected_label.hide()
        setattr(self, detected_label_attr, detected_label)
        layout.addWidget(settings_field_row(theme, detected_label))

        return section

    def _selected_provider_label(self, combo):
        index = combo.currentIndex()
        return combo.itemText(index) if index >= 0 else ""

    def _selected_provider_key_hint(self, combo, providers):
        provider_id = combo.currentData()
        for _label, candidate_id, prefix in providers:
            if candidate_id == provider_id:
                return prefix
        return ""

    def _select_combo_data(self, combo, value):
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _detect_provider_from_key(self, api_key, prefixes):
        key = (api_key or "").strip()
        if not key:
            return None
        for prefix, provider_id in prefixes:
            if key.startswith(prefix):
                return provider_id
        return None

    def _key_matches_known_prefix(self, api_key, prefixes):
        key = (api_key or "").strip()
        return bool(key) and any(key.startswith(prefix) for prefix, _provider_id in prefixes)

    def _select_detected_provider(self, combo, provider_id):
        if not combo or not provider_id:
            return False
        idx = combo.findData(provider_id)
        if idx < 0 or idx == combo.currentIndex():
            return idx >= 0
        combo.blockSignals(True)
        try:
            combo.setCurrentIndex(idx)
        finally:
            combo.blockSignals(False)
        return True

    def _custom_provider_hint(self):
        api_url = ""
        if hasattr(self, "api_url_input"):
            api_url = (self.api_url_input.text() or "").strip()
        if not api_url:
            return "Custom / OpenAI-compatible. Enter the provider's chat completions API URL."
        try:
            from urllib.parse import urlparse

            host = urlparse(api_url).netloc
        except Exception:
            host = ""
        return f"Custom / OpenAI-compatible via {host or api_url}"

    def _toggle_password_visibility(self, input_attr, button_attr):
        key_input = getattr(self, input_attr, None)
        show_button = getattr(self, button_attr, None)
        if not key_input or not show_button:
            return
        if key_input.echoMode() == QLineEdit.EchoMode.Password:
            key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            show_button.setText("Hide")
        else:
            key_input.setEchoMode(QLineEdit.EchoMode.Password)
            show_button.setText("Show")

    def detect_provider(self):
        api_key = self.api_key_input.text().strip()
        current_provider_id = self.answer_cloud_provider_combo.currentData() or "openai"
        detected_provider_id = self._detect_provider_from_key(api_key, ANSWER_KEY_PROVIDER_PREFIXES)
        if detected_provider_id and current_provider_id != "custom":
            self._select_detected_provider(self.answer_cloud_provider_combo, detected_provider_id)
        provider_id = self.answer_cloud_provider_combo.currentData() or "openai"
        provider = self._selected_provider_label(self.answer_cloud_provider_combo) or "OpenAI (GPT)"
        is_custom = provider_id == "custom"
        key_hint = self._selected_provider_key_hint(self.answer_cloud_provider_combo, ANSWER_CLOUD_PROVIDERS)
        self.api_key_input.setPlaceholderText(f"Paste your answer API key here ({key_hint})")

        if hasattr(self, "url_row"):
            self.url_row.setVisible(is_custom)
        if hasattr(self, "url_widget"):
            self.url_widget.setVisible(is_custom)

        if not api_key:
            if provider_id == "custom":
                self.provider_label.setText(f"Selected: {self._custom_provider_hint()}")
            else:
                self.provider_label.setText(f"Selected: {provider}")
        elif detected_provider_id and provider_id == detected_provider_id:
            self.provider_label.setText(f"\u2713 Detected: {provider}")
        elif provider_id == "custom":
            self.provider_label.setText(f"\u2713 Using: {self._custom_provider_hint()}")
        else:
            self.provider_label.setText(
                f"\u2713 Using selected provider: {provider}. "
                "If this key is for an unlisted provider, choose Custom / OpenAI-compatible and set the API URL."
            )
        self.provider_label.show()
        return provider







    def toggle_key_visibility(self):
        self._toggle_password_visibility("api_key_input", "show_key_btn")







    # explanation: wires all interactive signals for the embedding section.
    def _connect_embedding_signals(self):
        self.answer_provider_combo.currentIndexChanged.connect(self._on_answer_provider_changed)
        self.answer_cloud_provider_combo.currentIndexChanged.connect(self._update_embedding_same_summary)
        self.api_key_input.textChanged.connect(self._update_embedding_same_summary)
        self.embedding_same_checkbox.stateChanged.connect(self._on_same_provider_toggled)
        self.embedding_strategy_combo.currentIndexChanged.connect(self._on_embedding_strategy_changed)

    # explanation: toggles visibility between same-provider summary and independent embedding fields.
    def _on_same_provider_toggled(self, *args):
        same_provider = self.embedding_same_checkbox.isChecked()
        self.embedding_same_summary_label.setVisible(same_provider)
        self.embedding_independent_section.setVisible(not same_provider)
        self._update_embedding_same_summary()
        self._on_embedding_strategy_changed()

    # explanation: swaps local and cloud embedding sub-fields without saving config.
    def _on_embedding_strategy_changed(self, *args):
        if not hasattr(self, "embedding_strategy_combo"):
            return
        is_independent = not self.embedding_same_checkbox.isChecked()
        strategy = self.embedding_strategy_combo.currentData() or "cloud"
        self.embedding_local_section.setVisible(is_independent and strategy == "local")
        self.embedding_cloud_section.setVisible(is_independent and strategy == "cloud")
        self._on_embedding_cloud_provider_changed()
        self._refresh_embedding_status()

    # explanation: refreshes the embedding cloud provider detection label without saving config.
    def _on_embedding_cloud_provider_changed(self, *args):
        if not hasattr(self, "embedding_cloud_provider_combo"):
            return
        key = (self.embedding_cloud_api_key_input.text() or "").strip()
        detected_provider = self._detect_provider_from_key(key, EMBEDDING_KEY_PROVIDER_PREFIXES)
        if detected_provider:
            self._select_detected_provider(self.embedding_cloud_provider_combo, detected_provider)
        provider = self.embedding_cloud_provider_combo.currentData() or "Voyage AI"
        key_hint = self._selected_provider_key_hint(self.embedding_cloud_provider_combo, EMBEDDING_CLOUD_PROVIDERS)
        self.embedding_cloud_api_key_input.setPlaceholderText(f"Paste your embedding API key here ({key_hint})")
        if not key:
            self.embedding_cloud_detected_label.setText(f"Selected: {provider}")
        elif detected_provider:
            self.embedding_cloud_detected_label.setText(f"\u2713 Detected: {provider}")
        elif self._key_matches_known_prefix(key, EMBEDDING_KEY_PROVIDER_PREFIXES):
            self.embedding_cloud_detected_label.setText(
                "This key looks like an answer-provider key. "
                "Choose Voyage AI, OpenAI, or Cohere for cloud embeddings."
            )
        else:
            self.embedding_cloud_detected_label.setText(
                f"\u2713 Using selected provider: {provider}. "
                "Unlisted answer providers usually cannot be used for embeddings here."
            )
        self.embedding_cloud_detected_label.show()

    # explanation: updates the same-provider summary and warning based on current answer settings.
    def _update_embedding_same_summary(self):
        if not hasattr(self, "embedding_same_summary_label"):
            return
        answer_choice = self.answer_provider_combo.currentData() or ""
        if answer_choice == "local_server":
            self.embedding_same_summary_label.setText("Using: Local Server - same local server settings as above")
            return
        provider_id = self.answer_cloud_provider_combo.currentData() or "openai"
        detected = self._selected_provider_label(self.answer_cloud_provider_combo) or "selected cloud provider"
        if provider_id == "openai":
            self.embedding_same_summary_label.setText("Using: OpenAI - same key as above")
        else:
            self.embedding_same_summary_label.setText(
                f"{detected} cannot create embeddings here. Turn this off and choose an embedding provider below."
            )

    # explanation: shows or hides the independent embedding cloud key.
    def _toggle_embedding_cloud_key_visibility(self):
        self._toggle_password_visibility("embedding_cloud_api_key_input", "embedding_cloud_show_key_btn")

    # explanation: populates all embedding UI fields from config without triggering signals.
    def _load_embedding_settings(self):
        config = load_config()
        sc = config.get("search_config") or {}
        widgets = [
            self.embedding_same_checkbox,
            self.embedding_strategy_combo,
            self.embedding_cloud_provider_combo,
            self.embedding_cloud_api_key_input,
            self.embedding_local_url_input,
        ]
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.embedding_same_checkbox.setChecked(bool(sc.get("embedding_same_as_answer", True)))
            strategy = sc.get("embedding_strategy")
            if not strategy:
                legacy_engine = (sc.get("embedding_engine") or "").strip().lower()
                strategy = "local" if legacy_engine in ("local", "ollama", "local_openai") else "cloud"
            strategy_idx = self.embedding_strategy_combo.findData(strategy)
            if strategy_idx >= 0:
                self.embedding_strategy_combo.setCurrentIndex(strategy_idx)
            provider = sc.get("embedding_cloud_provider")
            if not provider:
                legacy_engine = (sc.get("embedding_engine") or "voyage").strip().lower()
                provider = {"openai": "OpenAI", "cohere": "Cohere"}.get(legacy_engine, "Voyage AI")
            provider_idx = self.embedding_cloud_provider_combo.findData(provider)
            if provider_idx < 0 and provider == "Voyage AI (Recommended)":
                provider_idx = self.embedding_cloud_provider_combo.findData("Voyage AI")
            if provider_idx >= 0:
                self.embedding_cloud_provider_combo.setCurrentIndex(provider_idx)
            cloud_key = sc.get("embedding_cloud_api_key")
            if not cloud_key:
                provider_id = (self.embedding_cloud_provider_combo.currentData() or "Voyage AI").lower()
                if "openai" in provider_id:
                    cloud_key = sc.get("openai_embedding_api_key", "")
                elif "cohere" in provider_id:
                    cloud_key = sc.get("cohere_api_key", "")
                else:
                    cloud_key = sc.get("voyage_api_key", "")
            self.embedding_cloud_api_key_input.setText(cloud_key or "")
            self.embedding_local_url_input.setText(
                sc.get("embedding_local_url")
                or sc.get("local_llm_url")
                or sc.get("ollama_base_url")
                or "http://localhost:11434/v1"
            )
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self._on_same_provider_toggled()

    # explanation: reads all embedding UI fields and writes to config-compatible keys.
    def _save_embedding_settings(self):
        same_provider = self.embedding_same_checkbox.isChecked()
        strategy = self.embedding_strategy_combo.currentData() or "cloud"
        provider = self.embedding_cloud_provider_combo.currentData() or "Voyage AI"
        api_key = (self.embedding_cloud_api_key_input.text() or "").strip()
        local_url = (self.embedding_local_url_input.text() or "").strip()
        values = {
            "embedding_same_as_answer": same_provider,
            "embedding_strategy": strategy,
            "embedding_cloud_provider": provider,
            "embedding_cloud_api_key": api_key,
            "embedding_local_url": local_url,
        }
        if not same_provider:
            if strategy == "local":
                values["embedding_engine"] = "local_openai"
                values["local_llm_url"] = local_url or "http://localhost:11434/v1"
            elif provider == "OpenAI":
                values["embedding_engine"] = "openai"
                values["openai_embedding_api_key"] = api_key
            elif provider == "Cohere":
                values["embedding_engine"] = "cohere"
                values["cohere_api_key"] = api_key
            else:
                values["embedding_engine"] = "voyage"
                values["voyage_api_key"] = api_key
        return values

    # explanation: builds a transient config from the current Answer Provider widgets for validation/tests.
    def _config_with_current_answer_provider(self, config):
        config = dict(config or {})
        sc = dict(config.get("search_config") or {})
        answer_with = self.answer_provider_combo.currentData() or ""
        if answer_with == "local_server":
            current_provider = (config.get("provider") or "").strip().lower()
            config["provider"] = current_provider if current_provider in ("ollama", "local_openai") else "local_openai"
            sc["local_llm_url"] = (self.local_llm_url.text() or "http://localhost:1234/v1").strip()
            sc["local_llm_model"] = (self.local_llm_model.text() or "text-embedding-3-small").strip()
        else:
            api_key = (self.api_key_input.text() or "").strip()
            config["api_key"] = api_key
            config["provider"] = self.answer_cloud_provider_combo.currentData() or "openai"
            if config["provider"] == "custom":
                config["api_url"] = (self.api_url_input.text() or "").strip()
        config["search_config"] = sc
        return config


    def _toggle_voyage_key_visibility(self):



        if self.voyage_api_key_input.echoMode() == QLineEdit.EchoMode.Password:



            self.voyage_api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)



            self.voyage_show_key_btn.setText("Hide")



        else:



            self.voyage_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)



            self.voyage_show_key_btn.setText("Show")







    def _toggle_openai_key_visibility(self):



        if self.openai_embedding_api_key_input.echoMode() == QLineEdit.EchoMode.Password:



            self.openai_embedding_api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)



            self.openai_show_key_btn.setText("Hide")



        else:



            self.openai_embedding_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)



            self.openai_show_key_btn.setText("Show")







    def _toggle_cohere_key_visibility(self):



        if self.cohere_api_key_input.echoMode() == QLineEdit.EchoMode.Password:



            self.cohere_api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)



            self.cohere_show_key_btn.setText("Hide")



        else:



            self.cohere_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)



            self.cohere_show_key_btn.setText("Show")







    def _safe_get_ui_value(self, attr_name, default_value):

        """Radically safe UI reader: returns value only if widget is alive and healthy."""

        try:

            try:
                    try:
                        import sip
                    except ImportError:
                        from PyQt6 import sip

            except ImportError:
                    from PyQt6 import sip



            widget = getattr(self, attr_name, None)

            if widget is None or sip.isdeleted(widget):

                return default_value



            if hasattr(widget, "currentData") and hasattr(widget, "currentText"):

                data = widget.currentData()

                return data if data is not None else widget.currentText().strip()

            if hasattr(widget, "isChecked"):

                return bool(widget.isChecked())

            if hasattr(widget, "value"):

                return widget.value()

            if hasattr(widget, "toPlainText"):

                return widget.toPlainText().strip()

            if hasattr(widget, "text"):

                return widget.text().strip()

            return default_value

        except Exception:

            return default_value



    # --- Settings Save And Reload ---

    def save_settings(self):

        """Saves settings with radical safety and consistent notification."""

        from aqt.utils import showInfo



        # Silence signals during save

        widgets = ['search_method_combo', 'answer_provider_combo']

        for name in widgets:

            w = getattr(self, name, None)

            if w: w.blockSignals(True)



        try:

            current_config = load_config()

            current_sc = current_config.get('search_config', {})

            current_style = current_config.get('styling', {})



            # 1. Answer Provider Logic

            answer_with = self._safe_get_ui_value('answer_provider_combo', current_config.get('provider', 'api_key'))

            saved_cloud_provider = self._safe_get_ui_value(
                'answer_cloud_provider_combo',
                current_config.get('answer_cloud_provider', current_config.get('provider', 'openai'))
            )



            if answer_with == "local_server":

                provider_type = "local_openai"

                if current_config.get("provider") in ["ollama", "local_openai"]:

                    provider_type = current_config.get("provider")

                api_key = current_config.get('api_key', '')

            else:

                api_key = self._safe_get_ui_value('api_key_input', current_config.get('api_key', ''))

                provider_type = saved_cloud_provider



            note_type_filter = self._build_ntf_from_ui()



            relevance_mode = (current_sc.get('relevance_mode') or 'balanced').lower()
            if relevance_mode not in ('focused', 'balanced', 'broad'):
                relevance_mode = 'balanced'

            config = {

                'api_key': api_key,

                'provider': provider_type,

                'answer_cloud_provider': saved_cloud_provider,

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

                'search_config': {

                    'local_llm_url': self._safe_get_ui_value('local_llm_url', current_sc.get('local_llm_url', 'http://localhost:11434')),

                    'local_llm_model': self._safe_get_ui_value('local_llm_model', current_sc.get('local_llm_model', 'llama3.2')),

                    'search_method': self._safe_get_ui_value('search_method_combo', current_sc.get('search_method', 'hybrid')),

                    'enable_query_expansion': self._safe_get_ui_value('enable_query_expansion_cb', current_sc.get('enable_query_expansion', False)),

                    'use_ai_generic_term_detection': self._safe_get_ui_value('use_ai_generic_term_detection_cb', current_sc.get('use_ai_generic_term_detection', False)),

                    'enable_hyde': self._safe_get_ui_value('enable_hyde_cb', current_sc.get('enable_hyde', False)),

                    'enable_rerank': self._safe_get_ui_value('enable_rerank_cb', current_sc.get('enable_rerank', False)),

                    'use_context_boost': self._safe_get_ui_value('use_context_boost_cb', current_sc.get('use_context_boost', True)),

                    'min_relevance_percent': self._safe_get_ui_value('min_relevance_spin', current_sc.get('min_relevance_percent', 55)),

                    'relevance_mode': relevance_mode,

                    'max_results': self._safe_get_ui_value('max_results_spin', current_sc.get('max_results', 50)),

                    'relevance_from_answer': self._safe_get_ui_value('relevance_from_answer_cb', current_sc.get('relevance_from_answer', False)),

                    'hybrid_embedding_weight': self._safe_get_ui_value('hybrid_weight_spin', current_sc.get('hybrid_embedding_weight', 40)),

                    'embedding_engine': self._safe_get_ui_value('embedding_engine_combo', current_sc.get('embedding_engine', 'voyage')),

                    'voyage_api_key': self._safe_get_ui_value('voyage_api_key_input', current_sc.get('voyage_api_key', '')),

                    'voyage_embedding_model': self._safe_get_ui_value('voyage_embedding_model_combo', current_sc.get('voyage_embedding_model', 'voyage-3.5-lite')),

                    'openai_embedding_api_key': self._safe_get_ui_value('openai_embedding_api_key_input', current_sc.get('openai_embedding_api_key', '')),

                    'openai_embedding_model': self._safe_get_ui_value('openai_embedding_model_input', current_sc.get('openai_embedding_model', 'text-embedding-3-small')),

                    'cohere_api_key': self._safe_get_ui_value('cohere_api_key_input', current_sc.get('cohere_api_key', '')),

                    'cohere_embedding_model': self._safe_get_ui_value('cohere_embedding_model_input', current_sc.get('cohere_embedding_model', 'embed-english-v3.0')),

                    'voyage_batch_size': int(self._safe_get_ui_value('voyage_batch_size_spin', current_sc.get('voyage_batch_size', 64))),

                    'ollama_base_url': self._safe_get_ui_value('ollama_base_url_input', current_sc.get('ollama_base_url', "http://localhost:11434")),

                    'ollama_embed_model': self._safe_get_ui_value('ollama_embed_model_combo', current_sc.get('ollama_embed_model', "nomic-embed-text")),

                    'ollama_batch_size': int(self._safe_get_ui_value('ollama_batch_size_spin', current_sc.get('ollama_batch_size', 64))),

                    'use_dynamic_batch_size': self._safe_get_ui_value('use_dynamic_batch_size_cb', current_sc.get('use_dynamic_batch_size', True)),

                    'ollama_chat_model': self._safe_get_ui_value('ollama_chat_model_combo', current_sc.get('ollama_chat_model', "llama3.2")),

                    'rerank_python_path': self._safe_get_ui_value('rerank_python_path_input', current_sc.get('rerank_python_path', None)),

                }

            }



            # explanation: merges the new API-tab embedding fields into search_config.
            config['search_config'].update(self._save_embedding_settings())



            # Preserve other config keys

            for k in ['saved_presets']:

                if k in current_config:

                    config[k] = current_config[k]



            if provider_type == "custom":

                config['api_url'] = self._safe_get_ui_value('api_url_input', current_config.get('api_url', ''))



            if save_config(config):

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







    # --- Rerank Environment And Local Model Checks ---

    def _on_autodetect_python(self):

        """Attempts to find a compatible Python installation on common Windows/Mac paths."""

        from aqt.utils import showInfo, tooltip

        import subprocess

        import os



        # Common Windows paths for Python

        candidates = []

        if os.name == 'nt':

            local_appdata = os.environ.get('LOCALAPPDATA', '')

            base_dir = os.path.join(local_appdata, "Programs", "Python")

            if os.path.exists(base_dir):

                for folder in os.listdir(base_dir):

                    if folder.lower().startswith("python3"):

                        exe = os.path.join(base_dir, folder, "python.exe")

                        if os.path.exists(exe):

                            candidates.append(exe)

            # Check global paths

            for p in [r"C:\Python312\python.exe", r"C:\Python311\python.exe", r"C:\Python310\python.exe"]:

                if os.path.exists(p): candidates.append(p)

        else:

            # Unix/Mac

            for p in ["/usr/bin/python3", "/usr/local/bin/python3", "/opt/homebrew/bin/python3"]:

                if os.path.exists(p): candidates.append(p)



        if not candidates:

            showInfo("No common Python installations found automatically. Please paste the path to your python.exe manually.")

            return



        found_path = None

        for path in candidates:

            try:

                # Check if sentence-transformers is installed in this python

                cmd = [path, "-c", "import sentence_transformers; print('OK')"]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

                if "OK" in result.stdout:

                    found_path = path

                    break

            except Exception:

                continue



        if found_path:

            self.rerank_python_path_input.setText(found_path)

            tooltip(f"Success! Detected 'Rerank-Ready' Python at: {found_path}")

        else:

            # If no ready path, just take the first candidate

            self.rerank_python_path_input.setText(candidates[0])

            showInfo(f"Found Python at {candidates[0]}, but 'sentence-transformers' is not installed there yet. Click 'Install / show command for external Python' to prepare it.")



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

            'answer_provider_combo', 'answer_cloud_provider_combo', 'api_key_input',
            'enable_rerank_cb', 'enable_hyde_cb',

            'min_relevance_spin', 'layout_combo', 'answer_spacing_combo',

            'question_font_spin', 'answer_font_spin', 'notes_font_spin',

            'label_font_spin', 'width_spin', 'height_spin', 'section_spacing_spin',

            'use_context_boost_cb', 'relevance_from_answer_cb',

            'embedding_same_checkbox', 'embedding_strategy_combo',
            'embedding_cloud_provider_combo', 'embedding_cloud_api_key_input',
            'embedding_local_url_input', 'embedding_engine_combo',
            'ollama_chat_model_combo', 'ollama_embed_model_combo',

            'enable_query_expansion_cb', 'use_ai_generic_term_detection_cb'

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

            if hasattr(self, 'min_relevance_spin') and not sip.isdeleted(self.min_relevance_spin):

                self.min_relevance_spin.setValue(sc.get('min_relevance_percent', 55))



            # 2. Answer Provider & Local Server

            prov = c.get('provider', 'api_key')

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

                self.local_llm_url.setText(sc.get('local_llm_url', 'http://localhost:11434'))

            if hasattr(self, 'local_llm_model') and not sip.isdeleted(self.local_llm_model):

                self.local_llm_model.setText(sc.get('local_llm_model', 'llama3.2'))



            # 3. Checkboxes (using radical safety)

            self._safe_set_checked(getattr(self, 'enable_query_expansion_cb', None), sc.get('enable_query_expansion', False))

            self._safe_set_checked(getattr(self, 'use_ai_generic_term_detection_cb', None), sc.get('use_ai_generic_term_detection', False))

            self._safe_set_checked(getattr(self, 'enable_hyde_cb', None), sc.get('enable_hyde', False))

            self._safe_set_checked(getattr(self, 'enable_rerank_cb', None), sc.get('enable_rerank', False))

            self._safe_set_checked(getattr(self, 'use_context_boost_cb', None), sc.get('use_context_boost', True))

            self._safe_set_checked(getattr(self, 'relevance_from_answer_cb', None), sc.get('relevance_from_answer', False))

            # --- Persistent Rerank Path Fix ---

            if hasattr(self, "rerank_python_path_input") and not sip.isdeleted(self.rerank_python_path_input):

                path = sc.get('rerank_python_path', "")

                self.rerank_python_path_input.setText(str(path) if path else "")



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



    def _on_answer_provider_changed(self):

        """Show/hide API key vs Local Server options with radical existence checks."""

        if not hasattr(self, "api_key_section") or not hasattr(self, "local_server_section"):

            return



        provider = self.answer_provider_combo.currentData() or ""



        try:
                    try:
                        import sip
                    except ImportError:
                        from PyQt6 import sip

        except ImportError:
                    from PyQt6 import sip



        if not sip.isdeleted(self.api_key_section):

            self.api_key_section.setVisible(provider == "api_key")

        if not sip.isdeleted(self.local_server_section):

            self.local_server_section.setVisible(provider == "local_server")

        if hasattr(self, "_update_embedding_same_summary"):

            self._update_embedding_same_summary()



    def _refresh_local_models(self):

        """Fetch models from local server (Ollama, LM Studio, etc.)"""

        url = (self.local_llm_url.text() or "").strip()

        if not url:

            showInfo("Please enter a Server URL first.")

            return



        # Auto-correct common mistakes

        if "11434" in url and not url.startswith("http"):

            url = f"http://{url}"



        try:

            # Try Ollama native list first if it looks like Ollama

            if "11434" in url:

                models = get_ollama_models(url)

            else:

                # Try OpenAI compatible /models endpoint

                import requests

                # Handle trailing slashes and /v1

                base = url.rstrip('/')

                models_url = f"{base}/models"

                resp = requests.get(models_url, timeout=5)

                data = resp.json()

                models = [m['id'] for m in data.get('data', [])]



            if models:

                # We don't have a combo anymore, we have a line edit for flexibility,

                # but we'll show a picker or just info.

                from aqt.utils import chooseList

                idx = chooseList("Select a model from your local server:", models)

                if idx >= 0:

                    self.local_llm_model.setText(models[idx])

            else:

                showInfo("Connected but no models found. Make sure a model is loaded in your server.")

        except Exception as e:

            showInfo(f"Could not fetch models: {e}\n\nCheck that your server is running at {url}.")







    def _autodetect_local_server(self):

        """Find a running local AI server and populate URL/model fields."""

        import requests
        from aqt.utils import chooseList, tooltip

        candidates = [
            ("Ollama", "http://localhost:11434", "ollama"),
            ("LM Studio", "http://localhost:1234/v1", "openai"),
            ("Jan", "http://localhost:1337/v1", "openai"),
        ]

        errors = []

        for name, url, kind in candidates:

            try:

                if kind == "ollama":

                    models = get_ollama_models(url)

                else:

                    resp = requests.get(f"{url.rstrip('/')}/models", timeout=2)

                    if resp.status_code != 200:

                        errors.append(f"{name}: HTTP {resp.status_code}")

                        continue

                    data = resp.json()

                    models = [
                        str(item.get("id") or "").strip()
                        for item in data.get("data", [])
                        if isinstance(item, dict) and (item.get("id") or "").strip()
                    ]

                if not models:

                    errors.append(f"{name}: connected, no models found")

                    continue

                self.local_llm_url.setText(url)

                if len(models) == 1:

                    self.local_llm_model.setText(models[0])

                else:

                    idx = chooseList(f"{name} detected. Select a model:", models)

                    self.local_llm_model.setText(models[idx] if idx >= 0 else models[0])

                provider_idx = self.answer_provider_combo.findData("local_server")

                if provider_idx >= 0:

                    self.answer_provider_combo.setCurrentIndex(provider_idx)

                if hasattr(self, "_on_answer_provider_changed"):

                    self._on_answer_provider_changed()

                tooltip(f"Detected {name}. Server and model fields updated.")

                return

            except Exception as exc:

                errors.append(f"{name}: {exc}")

        showInfo(
            "No running local AI server was detected.\n\n"
            "Start Ollama, LM Studio, or Jan, then click Autodetect again.\n\n"
            + "\n".join(errors[:3])
        )



    def _test_local_server_connection(self):

        """Unified test for local server (Ollama, LM Studio, etc.)"""

        url = (self.local_llm_url.text() or "").strip()

        if not url:

            showInfo("Please enter a Server URL first.")

            return



        import time

        import requests

        start = time.time()

        try:

            # Try a simple GET to the base or models endpoint

            # If it's Ollama, use its tags endpoint; otherwise try /models

            test_url = f"{url.rstrip('/')}/api/tags" if "11434" in url else f"{url.rstrip('/')}/models"
            log_debug(f"Testing local server connection: {test_url}")

            resp = requests.get(test_url, timeout=5)

            elapsed = (time.time() - start) * 1000



            if resp.status_code == 200:

                showInfo(f"OK: Connection successful.\n\nLatency: {elapsed:.0f}ms\nServer: {url}")

            else:

                showInfo(f"Warning: Server responded with code {resp.status_code}.\nURL: {test_url}")

        except Exception as e:

            showInfo(f"Error: Connection failed.\n\nError: {e}\n\nMake sure your server is running at {url}")







    def _check_rerank_available(self, extra_path=None, python_path=None):



        """Check if sentence-transformers CrossEncoder is available.



        If python_path is set (path to python.exe), use that Python for the check.



        Else if extra_path is set, run Anki's Python with that folder on sys.path.



        Else run Anki's Python."""



        try:



            import os



            import subprocess



            import sys



            # Prefer user's Python (e.g. Python 3.11) when set



            if python_path:



                python_path = python_path.strip()



                # Allow folder or executable: if folder, append python.exe on Windows



                if os.path.isdir(python_path):



                    python_exe = os.path.join(python_path, "python.exe")



                    if not os.path.isfile(python_exe):



                        python_exe = os.path.join(python_path, "python")



                    python_path = python_exe if os.path.isfile(python_exe) else python_path



                if not os.path.isfile(python_path):



                    return False



                result = subprocess.run(



                    [python_path, "-c", "from sentence_transformers import CrossEncoder; print('ok')"],



                    capture_output=True, text=True, timeout=30,



                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0



                )



                return result.returncode == 0 and 'ok' in (result.stdout or '')



            env = os.environ.copy()



            if extra_path and os.path.isdir(extra_path):



                check_script = (



                    "import sys, os; "



                    "p = os.environ.get('AI_SEARCH_ST_PATH', ''); "



                    "p and sys.path.insert(0, p); "



                    "from sentence_transformers import CrossEncoder; "



                    "print('ok')"



                )



                env['AI_SEARCH_ST_PATH'] = extra_path



            else:



                check_script = "from sentence_transformers import CrossEncoder; print('ok')"



            result = subprocess.run(



                [sys.executable, "-c", check_script],



                capture_output=True, text=True, timeout=15, env=env,



                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0



            )



            return result.returncode == 0 and 'ok' in (result.stdout or '')



        except Exception:



            return False







    def _update_rerank_tooltip(self):



        """Update Cross-Encoder checkbox tooltip with status and (if unavailable) Python path."""



        import sys



        base = "Re-ranks top 15 results with a cross-encoder for 10-30% better relevance.\n"



        if self._rerank_available:



            self.enable_rerank_cb.setToolTip(base + "Ready to use.")



        else:



            self.enable_rerank_cb.setToolTip(



                base + "Not installed. Use an external Python (e.g. Python 3.11) with "



                "sentence-transformers. This is recommended on Windows because Anki's Python may not load torch.\n"
                "Anki's Python: " + sys.executable



            )





    def _update_rerank_status_ui(self):



        """Update the visible Cross-Encoder setup status."""



        label = getattr(self, "rerank_status_label", None)
        theme = _addon_theme()



        if label is None:



            return



        if self._rerank_available:



            label.setText("Cross-Encoder: Ready")



            label.setStyleSheet(settings_status_label_style(theme, "success"))



        else:



            label.setText("Cross-Encoder: Needs setup")



            label.setStyleSheet(settings_status_label_style(theme, "warning"))







    def _on_check_rerank_again(self):



        """Re-check sentence-transformers and update Cross-Encoder checkbox state and tooltip."""



        import sys



        python_path = (self.rerank_python_path_input.text() or '').strip() or None



        self._rerank_available = self._check_rerank_available(python_path=python_path)



        self.enable_rerank_cb.setEnabled(self._rerank_available)



        self._update_rerank_tooltip()



        self._update_rerank_status_ui()



        if self._rerank_available:



            showInfo("sentence-transformers is available. Cross-Encoder re-ranking can be enabled.")



        else:



            msg = (



                "sentence-transformers not found.\n\n"



                "Recommended setup - Use external Python (e.g. Python 3.11):\n"



                "1. Set 'Use external Python' to that python.exe (or its folder).\n"



                "2. Click 'Install / show command for external Python'.\n"



                "3. Run the shown command if needed, then click 'Check again'.\n\n"



                "Advanced option - Use Anki's Python:\n"



                "Clear the external path and use 'Try Anki Python fallback' in Re-Ranking.\n"
                "This can fail on Windows because Anki's Python may not load torch/sentence-transformers.\n\n"



                "Anki's Python: " + sys.executable



            )



            showInfo(msg)







    def _on_browse_rerank_python(self):



        """Let the user pick an external python.exe for Cross-Encoder setup."""



        path, _ = QFileDialog.getOpenFileName(



            self,



            "Select external Python",



            os.path.expanduser("~"),



            "Python executable (*.exe);;All files (*)",



        )



        if path:



            self.rerank_python_path_input.setText(path)





    def _on_install_into_external_python(self):

        """Show install instructions for the selected external Python."""

        path = (self.rerank_python_path_input.text() or '').strip()



        if not path:



            showInfo("Enter an external Python path first, or click Autodetect. Use python.exe or the folder containing it.")



            return



        python_exe = _resolve_external_python_exe(path)



        if not python_exe:



            showInfo(f"Path not found or not a valid Python:\n{path}\n\nEnter the path to python.exe or the folder containing it.")



            return



        install_dependencies(python_exe=python_exe)







    # --- Embedding Status And Indexing Actions ---

    def _on_search_method_changed(self):



        """Show/hide Cloud Embeddings and options based on search method."""



        method = self.search_method_combo.currentData() or "hybrid"



        # explanation: provider controls moved to API Settings; Search tab no longer shows the old accordion.
        if hasattr(self, "embedding_section"):

            self.embedding_section.setVisible(False)



        # HyDE only applies to embedding/hybrid



        hyde_visible = method in ("embedding", "hybrid")

        self.enable_hyde_cb.setVisible(hyde_visible)

        if hasattr(self, "enable_hyde_row"):

            self.enable_hyde_row.setVisible(hyde_visible)



        # Hybrid weight: used in weighted RRF (\xce\xb1_lexical * 1/(k+r_kw) + \xce\xb1_dense * 1/(k+r_emb))



        hybrid_visible = method == "hybrid"

        self.hybrid_weight_label.setVisible(hybrid_visible)



        self.hybrid_weight_spin.setVisible(hybrid_visible)

        if hasattr(self, "hybrid_weight_row"):

            self.hybrid_weight_row.setVisible(hybrid_visible)







    def _refresh_embedding_status(self):

        """Check and display embedding status for the currently selected engine in the UI."""

        try:

            # explanation: derive status from the new API-tab embedding controls.
            config = load_config()

            sc = dict(config.get("search_config") or {})

            if hasattr(self, "_save_embedding_settings"):

                sc.update(self._save_embedding_settings())

            config["search_config"] = sc

            config = self._config_with_current_answer_provider(config)

            valid, message = validate_embedding_config(config)

            effective = get_effective_embedding_config(config)

            effective_sc = effective.get("search_config") or {}

            engine = (effective_sc.get("embedding_engine") or "voyage").strip().lower()



            status_text = ""



            if engine in ('local_openai', 'ollama'):

                base_url = (
                    effective_sc.get("ollama_base_url")
                    or effective_sc.get("local_llm_url")
                    or effective_sc.get("embedding_local_url")
                    or 'http://localhost:11434/v1'
                )

                model = (
                    effective_sc.get("ollama_embed_model")
                    or effective_sc.get("local_llm_model")
                    or 'text-embedding-3-small'
                )

                status_text = (

                    "CONNECTED: Local embeddings\n\n"

                    f"URL: {base_url}\n"

                    f"Model: {model}\n\n"

                    "Ensure the server exposes an /embeddings endpoint."

                )



            elif engine == "voyage":

                api_key = (effective_sc.get("voyage_api_key") or "").strip()

                if not api_key:

                    status_text = (

                        "DISABLED: Voyage AI\n\n"

                        "Please enter your Voyage API key above to enable high-quality medical search."

                    )

                else:

                    status_text = (

                        "READY: Voyage AI (Cloud)\n\n"

                        "API key detected. Click 'Create/Update' below to index your notes."

                    )



            elif engine == "openai":

                api_key = (effective_sc.get("openai_embedding_api_key") or "").strip()

                model = effective_sc.get("openai_embedding_model") or "text-embedding-3-small"

                if not api_key:

                    status_text = (

                        "DISABLED: OpenAI\n\n"

                        "Enter your OpenAI API key above to use OpenAI embeddings."

                    )

                else:

                    status_text = (

                        f"READY: OpenAI (Cloud) - Model: {model}\n\n"

                        "Click 'Create/Update' to start embedding."

                    )



            elif engine == "cohere":

                api_key = (effective_sc.get("cohere_api_key") or "").strip()

                if not api_key:

                    status_text = (

                        "DISABLED: Cohere\n\n"

                        "Enter your Cohere API key above to enable embeddings."

                    )

                else:

                    status_text = (

                        "READY: Cohere (Cloud)\n\n"

                        "API key detected. Ready to create embeddings."

                    )

            elif not valid:

                status_text = message



            if hasattr(self, 'embedding_status_label'):

                self.embedding_status_label.setText(status_text)
                theme = _addon_theme()

                if "READY" in status_text or "CONNECTED" in status_text:

                    self.embedding_status_label.setStyleSheet(settings_status_label_style(theme, "success"))

                else:

                    self.embedding_status_label.setStyleSheet(settings_status_label_style(theme, "error"))

        except Exception as e:

            if hasattr(self, 'embedding_status_label'):

                self.embedding_status_label.setText(f"Error checking status: {str(e)}")



    def _start_embedding_service(self):



        """Start the embedding service in a separate process"""



        import subprocess



        import sys



        import os



        import urllib.request



        import json



        import time







        # Local embedding service is no longer supported.



        # This method is kept only to avoid breaking older configs.



        showInfo(



            "The local embedding service has been removed.\n\n"



            "This addon now uses only the cloud embeddings API (Voyage) for semantic search.\n"



            "You can generate embeddings via the 'Create/Update Embeddings' button."



        )



        return







        # Check if process is already running



        if self.service_process is not None:



            if sys.platform == 'win32':



                # On Windows, we can't easily check if process is running



                # Just check via HTTP below



                pass



            elif hasattr(self.service_process, 'poll') and self.service_process.poll() is None:



                showInfo("Service is already starting. Please wait...")



                return



            else:



                # Process has ended, reset reference



                self.service_process = None







        # Get addon directory



        addon_dir = os.path.dirname(__file__)







        # Try to start embedding_service.py first (real service)



        service_file = os.path.join(addon_dir, "embedding_service.py")



        fallback_file = os.path.join(addon_dir, "simple_embedding_server.py")







        service_script = None



        service_name = None







        if os.path.exists(service_file):



            service_script = service_file



            service_name = "embedding_service.py (Real Service)"



        elif os.path.exists(fallback_file):



            service_script = fallback_file



            service_name = "simple_embedding_server.py (Test Server)"



        else:



            showInfo(



                f"\xe2\x9d\u0152 Cannot find embedding service files!\n\n"



                f"Expected files:\n"



                f"- {service_file}\n"



                f"- {fallback_file}\n\n"



                f"Please make sure the service files are in the addon directory."



            )



            return







        try:



            # Start the service in a new process



            # On Windows, use cmd.exe start to open a new window



            if sys.platform == 'win32':



                # Create a batch file to run the service (handles paths with spaces better)



                import tempfile



                batch_content = f'''@echo off



title Embedding Service



cd /d "{addon_dir}"



echo Starting embedding service...



echo.



"{sys.executable}" "{service_script}"



if errorlevel 1 (



    echo.



    echo Service exited with an error.



    echo Press any key to close this window...



    pause >nul



)



'''



                # Write batch file



                batch_file = os.path.join(addon_dir, "start_embedding_service.bat")



                try:



                    # Ensure directory exists and file can be written



                    os.makedirs(addon_dir, exist_ok=True)



                    with open(batch_file, 'w', encoding='utf-8', newline='\r\n') as f:  # Windows line endings



                        f.write(batch_content)



                    log_debug(f"Created batch file: {batch_file}")







                    # Try multiple methods to start the service, prioritizing simpler ones



                    service_started = False







                    # Method 1: Use VBScript wrapper (handles paths with spaces perfectly)



                    try:



                        vbs_script = os.path.join(addon_dir, "start_service.vbs")



                        vbs_content = f'''Set WshShell = CreateObject("WScript.Shell")



WshShell.CurrentDirectory = "{addon_dir}"



WshShell.Run "cmd /k ""{batch_file}""", 1, False



Set WshShell = Nothing



'''



                        with open(vbs_script, 'w', encoding='utf-8') as f:



                            f.write(vbs_content)







                        # Execute VBScript - this handles paths with spaces automatically



                        subprocess.Popen(['wscript', vbs_script], shell=False)



                        log_debug(f"Started service via VBScript wrapper: {vbs_script}")



                        service_started = True



                        self.service_process = True



                    except Exception as vbs_err:



                        log_debug(f"Method 1 (VBScript) failed: {vbs_err}")







                    # Method 2: Direct batch file execution using subprocess with shell=True



                    if not service_started:



                        try:



                            # Use shell=True which handles path quoting automatically



                            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):



                                subprocess.Popen(



                                    f'cmd /c start cmd /k "{batch_file}"',



                                    shell=True,



                                    creationflags=subprocess.CREATE_NEW_CONSOLE,



                                    cwd=addon_dir



                                )



                            else:



                                subprocess.Popen(



                                    f'cmd /c start cmd /k "{batch_file}"',



                                    shell=True,



                                    cwd=addon_dir



                                )



                            log_debug(f"Started service via batch file (shell=True): {batch_file}")



                            service_started = True



                            self.service_process = True



                        except Exception as batch_err:



                            log_debug(f"Method 2 (batch file shell) failed: {batch_err}")







                    # Method 3: Direct Python execution with shell=True



                    if not service_started:



                        try:



                            # Use shell=True for automatic path handling



                            cmd_str = f'cmd /c start cmd /k "cd /d "{addon_dir}" && "{sys.executable}" "{service_script}""'



                            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):



                                subprocess.Popen(



                                    cmd_str,



                                    shell=True,



                                    creationflags=subprocess.CREATE_NEW_CONSOLE



                                )



                            else:



                                subprocess.Popen(cmd_str, shell=True)



                            log_debug(f"Started service via direct Python execution (shell=True)")



                            service_started = True



                            self.service_process = True



                        except Exception as direct_err:



                            log_debug(f"Method 3 (direct Python shell) failed: {direct_err}")







                    # Method 4: os.startfile (Windows only, simplest but no console)



                    if not service_started and sys.platform == 'win32':



                        try:



                            os.startfile(batch_file)



                            log_debug(f"Started service via os.startfile")



                            service_started = True



                            self.service_process = True



                        except Exception as startfile_err:



                            log_debug(f"Method 4 (os.startfile) failed: {startfile_err}")







                    # Method 5: PowerShell as last resort (only if admin needed)



                    if not service_started:



                        try:



                            # Create PowerShell script with proper escaping



                            ps_script = os.path.join(addon_dir, "start_embedding_service_admin.ps1")



                            # Escape backslashes and quotes properly



                            batch_file_escaped = batch_file.replace('\\', '\\\\').replace("'", "''")



                            addon_dir_escaped = addon_dir.replace('\\', '\\\\').replace("'", "''")



                            python_exe_escaped = sys.executable.replace('\\', '\\\\').replace("'", "''")



                            service_script_escaped = service_script.replace('\\', '\\\\').replace("'", "''")







                            # Use $PSScriptRoot for dynamic path (works regardless of folder name)



                            ps_content = f'''# PowerShell script to start embedding service with admin privileges if needed



# This script uses $PSScriptRoot to get the directory where the script is located (works regardless of folder name)



$ErrorActionPreference = "Continue"



$scriptDir = $PSScriptRoot



$batchFile = Join-Path $scriptDir "start_embedding_service.bat"



$pythonExe = '{python_exe_escaped}'



$serviceScript = Join-Path $scriptDir "embedding_service.py"







# Check if running as admin



$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)







if (-not $isAdmin) {{



    Write-Host "Requesting administrator privileges..."



    $cmd = "Set-Location -LiteralPath '$scriptDir'; & '$pythonExe' '$serviceScript'"



    Start-Process powershell -Verb RunAs -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $cmd



}} else {{



    Write-Host "Running with administrator privileges..."



    Set-Location -LiteralPath $scriptDir



    & $pythonExe $serviceScript



}}



'''



                            with open(ps_script, 'w', encoding='utf-8', newline='\r\n') as f:



                                f.write(ps_content)



                            log_debug(f"Created PowerShell script: {ps_script}")







                            # Execute with shell=True for proper path handling



                            subprocess.Popen(



                                f'powershell -ExecutionPolicy Bypass -File "{ps_script}"',



                                shell=True,



                                cwd=addon_dir



                            )



                            log_debug(f"Started service via PowerShell script")



                            service_started = True



                            self.service_process = True



                        except Exception as ps_err:



                            log_debug(f"Method 5 (PowerShell) failed: {ps_err}")







                    if not service_started:



                        raise Exception("All service startup methods failed. Check debug_log.txt for details.")



                except Exception as batch_error:



                    log_debug(f"Failed to create batch file, trying direct method: {batch_error}")



                    # Fallback: try direct method with explicit window



                    try:



                        # Try direct Python execution in new console window



                        # Use full path to Python and service script



                        python_exe = sys.executable



                        service_path = service_script



                        # Escape paths with spaces properly



                        if ' ' in python_exe:



                            python_exe = f'"{python_exe}"'



                        if ' ' in service_path:



                            service_path = f'"{service_path}"'







                        # Create a command that changes directory and runs Python



                        cmd_str = f'cd /d "{addon_dir}" && {python_exe} {service_path}'



                        log_debug(f"Starting service with command: {cmd_str}")







                        # Use CREATE_NEW_CONSOLE flag to ensure new window



                        if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):



                            subprocess.Popen(



                                ['cmd', '/c', 'start', 'cmd', '/k', cmd_str],



                                shell=False,



                                creationflags=subprocess.CREATE_NEW_CONSOLE



                            )



                        else:



                            subprocess.Popen(



                                ['cmd', '/c', 'start', 'cmd', '/k', cmd_str],



                                shell=True



                            )



                        self.service_process = True



                        log_debug("Service started via direct command method")



                    except Exception as direct_error:



                        log_debug(f"Direct method also failed: {direct_error}")



                        # Last resort: try PowerShell



                        try:



                            ps_cmd = f'Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd \'{addon_dir}\'; & \'{sys.executable}\' \'{service_script}\'"'



                            subprocess.Popen(['powershell', '-Command', ps_cmd], shell=False)



                            self.service_process = True



                            log_debug("Service started via PowerShell method")



                        except Exception as ps_error:



                            log_debug(f"PowerShell method also failed: {ps_error}")



                            raise Exception(f"All service startup methods failed. Last error: {ps_error}")



            else:



                self.service_process = subprocess.Popen(



                    [sys.executable, service_script],



                    cwd=addon_dir



                )







            # Wait a moment for the service to start



            time.sleep(3)







            # Check if service is responding (more reliable than checking process)



            # On Windows, we can't easily track the detached process, so we check HTTP



            if sys.platform == 'win32':



                # For Windows, we check via HTTP instead of process polling



                pass  # Will check below



            elif self.service_process.poll() is not None:



                # Process has already terminated (error starting) - only for non-Windows



                showInfo(



                    f"Error: Failed to start service.\n\n"



                    f"Service: {service_name}\n\n"



                    f"The service process exited immediately. Check the console window for error messages.\n\n"



                    f"Common issues:\n"



                    f"- Missing dependencies (pip install flask sentence-transformers)\n"



                    f"- Port 9000 already in use\n"



                    f"- Python path issues"



                )



                self.service_process = None



                return







            # Test if service is responding



            try:



                test_data = json.dumps({"text": "test"}).encode('utf-8')



                test_req = urllib.request.Request(url, test_data, {"Content-Type": "application/json"})



                urllib.request.urlopen(test_req, timeout=3)







                showInfo(



                    f"\u2705 Service started successfully!\n\n"



                    f"Service: {service_name}\n"



                    f"URL: {url}\n\n"



                    f"A console window has been opened showing the service output.\n"



                    f"Keep this window open while using the embedding service."



                )



                # Refresh status



                QTimer.singleShot(500, self._refresh_embedding_status)



            except Exception as e:



                # Service started but not responding yet



                showInfo(



                    f"\xe2\u0161\xa0\ufe0f Service process started but not responding yet.\n\n"



                    f"Service: {service_name}\n"



                    f"URL: {url}\n\n"



                    f"Please wait a few seconds and click '\U0001F50C Test Connection' to verify.\n\n"



                    f"If the service doesn't start, check the console window for errors."



                )



                # Refresh status after a delay



                QTimer.singleShot(3000, self._refresh_embedding_status)







        except Exception as e:



            showInfo(



                f"\xe2\x9d\u0152 Error starting service!\n\n"



                f"Service: {service_name}\n"



                f"Error: {str(e)}\n\n"



                f"Please check:\n"



                f"- Python is installed and in PATH\n"



                f"- Service file exists: {service_script}\n"



                f"- Required dependencies are installed"



            )



            self.service_process = None







    def _test_embedding_connection(self):



        """Test connection to the selected embedding engine (Voyage, OpenAI, Cohere, or Ollama). Shows pass/fail with latency."""



        import time



        test_text = "Test connection"



        # explanation: test the effective embedding config derived from the new API-tab controls.
        config = load_config()

        sc = dict(config.get("search_config") or {})

        sc.update(self._save_embedding_settings())

        config["search_config"] = sc

        config = self._config_with_current_answer_provider(config)

        valid, validation_message = validate_embedding_config(config)

        if not valid:

            showInfo(validation_message)

            return

        effective_config = get_effective_embedding_config(config)

        sc = effective_config.get("search_config") or {}

        engine = sc.get("embedding_engine") or "voyage"

        config = effective_config



        if hasattr(self, 'embedding_status_label') and self.embedding_status_label:



            self.embedding_status_label.setText("Testing connection...")



            QApplication.processEvents()



        try:



            t0 = time.perf_counter()



            embedding = get_embedding_for_query(test_text, config=config)



            elapsed_ms = int((time.perf_counter() - t0) * 1000)



            dim = len(embedding) if embedding else 0



            if embedding and dim > 0:



                engine_names = {"ollama": "Ollama", "voyage": "Voyage AI", "openai": "OpenAI", "cohere": "Cohere"}



                engine_name = engine_names.get(engine, engine)



                showInfo(



                    f"\u2705 Embedding connection OK \u2014 {engine_name}\n\n"



                    f"Dimension: {dim} | Latency: {elapsed_ms} ms"



                )



                if hasattr(self, 'embedding_status_label') and self.embedding_status_label:



                    self.embedding_status_label.setText(f"\u2705 {engine_name} OK ({elapsed_ms} ms)")



            else:



                showInfo(



                    "\xe2\u0161\xa0\ufe0f Connection succeeded but received an empty embedding.\n\n"



                    "Check your engine settings (URL/model or API key) and try again."



                )



                if hasattr(self, 'embedding_status_label') and self.embedding_status_label:



                    self.embedding_status_label.setText("\xe2\u0161\xa0\ufe0f Empty embedding")



        except Exception as e:



            if hasattr(self, 'embedding_status_label') and self.embedding_status_label:



                self.embedding_status_label.setText("\xe2\x9d\u0152 Test failed")



            if engine == "ollama":



                hint = "Make sure Ollama is running (ollama serve) and the model is pulled (e.g. ollama pull nomic-embed-text)."



            elif engine == "openai":



                hint = "Enter your OpenAI API key above (or set OPENAI_API_KEY) and check internet access."



            elif engine == "cohere":



                hint = "Enter your Cohere API key above (or set COHERE_API_KEY) and check internet access."



            else:



                hint = "Enter your API key above (or set the provider's env var) and check internet access."



            showInfo(



                f"\xe2\x9d\u0152 Embedding test failed!\n\n"



                f"Error: {e}\n\n"



                f"{hint}"



            )







    def _migrate_json_to_db(self):



        """Copy embeddings from legacy JSON file into SQLite DB (no re-embedding)."""



        if hasattr(self, 'embedding_status_label') and self.embedding_status_label:



            self.embedding_status_label.setText("Migrating JSON \u2192 database...")



            QApplication.processEvents()



        try:



            count, err = migrate_embeddings_json_to_db()



            if err:



                showInfo(f"Migration could not complete.\n\n{err}")



                if hasattr(self, 'embedding_status_label') and self.embedding_status_label:



                    self.embedding_status_label.setText("")



                return



            showInfo(f"Migrated {count} embeddings from the old JSON file into the database.\n\nYou can keep or delete the old .json file; new data is now in the .db file.")



            if hasattr(self, 'embedding_status_label') and self.embedding_status_label:



                self.embedding_status_label.setText(f"Migrated {count} embeddings to database.")



            try:



                QTimer.singleShot(100, self._refresh_embedding_status)



            except Exception:



                pass



        except Exception as e:



            showInfo(f"Migration failed: {e}")



            if hasattr(self, 'embedding_status_label') and self.embedding_status_label:



                self.embedding_status_label.setText("")







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



        # For Ollama: verify model is available before starting (Ollama loads models on first request)



        if engine == 'ollama':



            base_url = sc.get('ollama_base_url') or 'http://localhost:11434'



            model = sc.get('ollama_embed_model') or 'nomic-embed-text'



            model_base = model.split(':')[0]  # nomic-embed-text:latest -> nomic-embed-text



            try:



                names = get_ollama_models(base_url.strip())



                if not names:



                    showInfo(



                        "\xe2\x9d\u0152 Ollama returned no models.\n\n"



                        "Make sure Ollama is running (ollama serve) and you have pulled at least one model.\n\n"



                        f"For embeddings, run: ollama pull {model_base}"



                    )



                    return



                # Check if our model (or base name) is available



                if not any(model_base in n or n.startswith(model_base) for n in names):



                    showInfo(



                        f"\xe2\x9d\u0152 Ollama embedding model '{model_base}' not found.\n\n"



                        f"Available models: {', '.join(names[:8])}{'...' if len(names) > 8 else ''}\n\n"



                        f"Run: ollama pull {model_base}"



                    )



                    return



            except Exception as e:



                showInfo(



                    f"Error: Cannot reach Ollama at {base_url}\n\n"



                    f"Error: {e}\n\n"



                    "Make sure Ollama is running (ollama serve)."



                )



                return



        # Quick API check first to avoid running a long job with bad config



        try:



            test_embedding = get_embedding_for_query("Test connection", config=runtime_config)



            if not test_embedding:



                showInfo(



                    "\xe2\x9d\u0152 Embedding engine returned an empty result.\n\n"



                    "Check your engine (URL/model or API key) and try again."



                )



                return



        except Exception as e:



            if engine == 'ollama':



                showInfo(



                    f"\xe2\x9d\u0152 Ollama embedding test failed.\n\n"



                    f"Error: {e}\n\n"



                    "Make sure Ollama is running (ollama serve) and the model is pulled "



                    f"(e.g. ollama pull {model_base})."



                )



            else:



                showInfo(



                    f"Error: Cannot use embeddings API.\n\n"



                    f"Error: {e}\n\n"



                    "Enter your API key for the selected engine above and check internet access."



                )



            return







        # Get note type filter config



        # Always base this on the *current* UI selections so user choices



        # (note types, decks, fields) are remembered between sessions,



        # even if they didn't click the main "Save Settings" button.



        current_ntf = self._build_ntf_from_ui()



        config = load_config()



        config['note_type_filter'] = current_ntf



        # Persist immediately so next Anki restart / addon open uses the same



        # note/deck/field selection.



        save_config(config)



        ntf = current_ntf

        runtime_config['note_type_filter'] = current_ntf







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



        if checkpoint and checkpoint.get('engine_id') != current_engine_id:



            checkpoint = None  # different engine: start fresh, don't offer resume



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



                f"This will generate embeddings for approximately {note_count:,} notes.\n\n"

                f"Currently excluded by filters: {len(eligibility.get('ineligible_notes', [])):,} notes.\n\n"



                f"This may take a while depending on the number of notes.\n\n"



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







        status_label = QLabel("Initializing embedding model...")



        status_label.setWordWrap(True)



        progress_layout.addWidget(status_label)







        progress_bar = QProgressBar()



        progress_bar.setRange(0, note_count)



        progress_bar.setValue(0)



        progress_bar.setTextVisible(True)



        progress_bar.setFormat("%p%")



        progress_layout.addWidget(progress_bar)







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



        close_button.setEnabled(False)



        close_button.clicked.connect(progress_dialog.close)



        button_layout.addWidget(close_button)







        progress_layout.addLayout(button_layout)







        # Store references for worker thread



        progress_dialog._status_label = status_label



        progress_dialog._progress_bar = progress_bar



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



        worker.log_message.connect(log_text.append)



        worker.finished_signal.connect(lambda processed, errors, skipped, still_failed: self._on_embedding_finished(



            progress_dialog, processed, errors, skipped, still_failed, note_count



        ))



        worker.error_signal.connect(lambda msg: self._on_embedding_error(progress_dialog, msg))







        # Store worker reference



        progress_dialog._worker = worker







        # Start worker thread



        worker.start()







    def _review_ineligible_notes(self):


        ntf = self._build_ntf_from_ui()


        audit = analyze_note_eligibility(ntf)


        ineligible = audit.get("ineligible_notes", [])


        if not ineligible:


            showInfo(
                "All notes in the current deck/type scope are eligible for embeddings.\n\n"
                f"Eligible notes: {audit.get('eligible_count', 0):,}"
            )


            return


        reason_lines = [
            f"Eligible notes: {audit.get('eligible_count', 0):,}",
            f"Ineligible notes: {len(ineligible):,}",
            f"- Wrong note type: {audit.get('filtered_out_note_type_count', 0):,}",
            f"- No embedding fields selected: {audit.get('no_selected_fields_count', 0):,}",
            f"- Selected fields empty: {audit.get('empty_selected_fields_count', 0):,}",
            "",
            "First ineligible notes:",
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


            tooltip(f"Opened {len(note_ids)} ineligible notes in browser")


        except Exception as exc:


            log_debug(f"Could not open ineligible notes in browser: {exc}")


        reason_lines.extend([
            "",
            f"Browser opened with first {len(browser_note_ids):,} ineligible notes.",
            "",
            "Browser query:",
            search_query,
        ])


        showText("\n".join(reason_lines), title="Ineligible Notes Audit")


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







    def _on_embedding_finished(self, progress_dialog, processed, errors, skipped, still_failed_count, note_count):



        """Handle embedding completion"""



        # Invalidate embeddings file cache so next search loads the updated file



        global _embeddings_file_cache, _embeddings_file_cache_path, _embeddings_file_cache_time



        _embeddings_file_cache = None



        _embeddings_file_cache_path = None



        _embeddings_file_cache_time = 0



        status_label = progress_dialog._status_label



        log_text = progress_dialog._log_text



        close_button = progress_dialog._close_button







        status_label.setText(
            f"\u2705 Completed! New embeddings: {processed:,}, already present: {skipped:,} ({errors} errors)"
        )



        log_text.append(f"\n\u2705 Embedding generation complete!")



        log_text.append(f"New embeddings created: {processed:,} notes")



        if skipped > 0:



            log_text.append(f"Skipped (already had embeddings): {skipped:,} notes")



        if errors > 0:



            log_text.append(f"Errors: {errors}")



        if still_failed_count > 0:



            log_text.append(f"Warning: {format_partial_failure_progress(still_failed_count)}")







        # Clear checkpoint only when no notes are still missing (so next run is full; missed ones get retried)



        if still_failed_count == 0:



            clear_checkpoint()







        close_button.setEnabled(True)



        message = (
            "Embedding generation complete!\n"
            f"New embeddings created: {processed:,} notes\n"
            f"Already present: {skipped:,} notes\n"
            f"Errors: {errors}"
        )



        if still_failed_count > 0:



            message += f"\n\nWarning: {format_partial_failure_completion(still_failed_count)}"



        showInfo(message)







    def _on_embedding_error(self, progress_dialog, error_msg):



        """Handle embedding error"""



        status_label = progress_dialog._status_label



        log_text = progress_dialog._log_text



        close_button = progress_dialog._close_button







        status_label.setText(f"\u274c Error: {error_msg}")



        log_text.append(f"\u274c Error: {error_msg}")



        close_button.setEnabled(True)



        showInfo(f"Error during embedding generation: {error_msg}")
