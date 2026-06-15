"""Focused Settings dialog layout builder mixin."""

from aqt.qt import *

from .settings_constants import ANSWER_CLOUD_PROVIDERS, EMBEDDING_CLOUD_PROVIDERS
from .theme import settings_button_style, settings_panel_style, settings_text_style
from .widgets import (
    settings_checkbox_row,
    settings_child_group,
    settings_child_row,
    settings_field_row,
    settings_hint_box,
    settings_inline_row,
    settings_labeled_action_row,
    settings_page,
    settings_status,
    settings_toolbar,
)
from ..utils import EmbeddingsTabMessages


class SettingsApiTabMixin:
    """Builds one focused section of the Settings dialog UI."""

    def _build_api_settings_tab(self, theme, current_config):
        # API Settings Tab



        api_tab, api_layout = settings_page(
            theme,
            "\U0001F511 API & AI provider",
            "Choose a Local Server (no key required) or a Cloud API provider and key. "
            "Works with OpenAI, Anthropic, Google, OpenRouter, or custom OpenAI-compatible endpoints.",
            min_width=820,
        )







        privacy_note = QLabel(

            "\u26a0\ufe0f Cloud providers may receive selected note content. "

            "Use a Local Server (Ollama, LM Studio, Jan) to keep content on your machine."

        )



        privacy_note.setWordWrap(True)
        privacy_note.setMaximumWidth(780)



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

        api_layout.addWidget(settings_inline_row(theme, "Answer with", self.answer_provider_combo, 260))

        answer_model_hint = QLabel(
            "This same answer provider/model is used by Ask AI search answers and review-aware Ask AI."
        )
        answer_model_hint.setWordWrap(True)
        answer_model_hint.setStyleSheet(settings_text_style(theme, "hint"))
        api_layout.addWidget(settings_hint_box(theme, answer_model_hint))







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
        answer_cloud_group = settings_child_group(theme)
        answer_cloud_group.content_layout.addWidget(answer_cloud_section)
        api_key_section_layout.addWidget(answer_cloud_group)
        self.answer_cloud_test_btn = self._make_connection_test_button(
            "\U0001F50C Test Cloud API Connection",
            "Test the selected cloud answer provider without sending note content.",
            self._test_cloud_api_connection,
        )
        api_key_section_layout.addWidget(settings_toolbar(theme, self.answer_cloud_test_btn))

        self.answer_cloud_status_label = settings_status(theme, "Cloud API not tested", "info")
        api_key_section_layout.addWidget(self.answer_cloud_status_label)







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

        self.url_row = settings_child_row(theme, self.url_widget)
        self.url_row.hide()
        api_key_section_layout.addWidget(self.url_row)



        api_layout.addWidget(self.api_key_section)







        # Unified Local Server Section

        self.local_server_section = QWidget()

        local_server_layout = QVBoxLayout(self.local_server_section)
        local_server_layout.setContentsMargins(0, 0, 0, 0)
        local_server_layout.setSpacing(7)


        self.local_backend_combo = QComboBox()
        self.local_backend_combo.addItem("Ollama", "ollama")
        self.local_backend_combo.addItem("LM Studio", "lm_studio")
        self.local_backend_combo.addItem("Jan", "jan")
        self.local_backend_combo.addItem("Custom OpenAI-compatible", "custom_openai")
        self.local_backend_combo.setToolTip(
            "Choose which local answer server to use when more than one is available."
        )
        self.local_backend_combo.currentIndexChanged.connect(self._on_local_backend_changed)
        local_server_layout.addWidget(settings_inline_row(theme, "Local backend", self.local_backend_combo, 220))



        self.local_llm_url = QLineEdit()

        self.local_llm_url.setPlaceholderText("http://localhost:11434 (Ollama) or http://localhost:1234/v1 (LM Studio)")
        self.local_llm_url.setToolTip(
            "URL for your selected local answer backend. Ollama uses its native base URL; LM Studio, Jan, and custom servers use an OpenAI-compatible /v1 URL."
        )

        local_server_layout.addWidget(settings_inline_row(theme, "Server URL", self.local_llm_url, 320))



        self.local_llm_model = QLineEdit()

        self.local_llm_model.setPlaceholderText("e.g. llama3.2 or gemma2")
        self.local_llm_model.setToolTip(
            "Type a local answer model manually, or click Find models to choose from the selected server."
        )

        self.local_server_refresh_models_btn = QPushButton("Find models")
        self.local_server_refresh_models_btn.setStyleSheet(settings_button_style(theme, "muted"))

        self.local_server_refresh_models_btn.setToolTip("Fetch models from the shown server, or autodetect a running local server if needed.")

        self.local_server_refresh_models_btn.clicked.connect(self._find_local_answer_models)

        local_server_layout.addWidget(
            settings_labeled_action_row(
                theme,
                "Model",
                self.local_llm_model,
                self.local_server_refresh_models_btn,
                control_width=450,
                tooltip=self.local_server_refresh_models_btn.toolTip(),
            )
        )



        self.local_server_test_btn = self._make_connection_test_button(
            "\U0001F50C Test Connection",
            "Test connection to your local answer server. Shows latency and availability.",
            self._test_local_server_connection,
        )

        local_server_layout.addWidget(settings_toolbar(theme, self.local_server_test_btn))



        local_guide = QLabel(

            "Defaults: Ollama http://localhost:11434/v1   |   "
            "LM Studio http://localhost:1234/v1   |   Jan http://localhost:1337/v1"

        )
        local_guide.setWordWrap(True)

        local_guide.setStyleSheet(settings_text_style(theme, "subtle"))

        local_server_layout.addWidget(local_guide)



        self.local_server_section.hide()

        api_layout.addWidget(self.local_server_section)



        # explanation: adds embedding provider controls below the existing answer provider controls.
        embedding_divider = QFrame()

        embedding_divider.setFrameShape(QFrame.Shape.HLine)

        embedding_divider.setFrameShadow(QFrame.Shadow.Sunken)

        api_layout.addWidget(embedding_divider)



        embedding_header = QLabel("\U0001F50D Embedding provider")

        embedding_header.setStyleSheet(settings_text_style(theme, "section_heading"))

        api_layout.addWidget(embedding_header)



        embedding_subtitle = QLabel("Used for semantic search (finding relevant cards).")

        embedding_subtitle.setWordWrap(True)

        embedding_subtitle.setStyleSheet(settings_text_style(theme, "subtle"))

        api_layout.addWidget(embedding_subtitle)



        self.embedding_same_checkbox = QCheckBox("Use same provider as answering")
        self.embedding_same_checkbox.setToolTip(
            "Use the answer provider for embeddings when compatible. Turn off to choose a separate embedding provider."
        )

        self.embedding_same_checkbox.setChecked(True)

        api_layout.addWidget(settings_checkbox_row(theme, self.embedding_same_checkbox))



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
        self.embedding_strategy_combo.setToolTip(
            "Choose where semantic-search embeddings are created: local server for privacy, or cloud API for hosted embedding models."
        )

        embedding_independent_layout.addWidget(settings_inline_row(theme, "Embedding with", self.embedding_strategy_combo, 260))



        self.embedding_local_section = QWidget()

        embedding_local_layout = QVBoxLayout(self.embedding_local_section)

        embedding_local_layout.setContentsMargins(0, 0, 0, 0)

        self.embedding_local_backend_combo = QComboBox()
        self.embedding_local_backend_combo.addItem("Ollama", "ollama")
        self.embedding_local_backend_combo.addItem("LM Studio", "lm_studio")
        self.embedding_local_backend_combo.addItem("Jan", "jan")
        self.embedding_local_backend_combo.addItem("Custom OpenAI-compatible", "custom_openai")
        self.embedding_local_backend_combo.setToolTip(
            "Choose which local embedding server to use when more than one is available."
        )
        self.embedding_local_backend_combo.currentIndexChanged.connect(self._on_embedding_local_backend_changed)
        embedding_local_layout.addWidget(settings_inline_row(theme, "Local backend", self.embedding_local_backend_combo, 220))

        self.embedding_local_url_input = QLineEdit()

        self.embedding_local_url_input.setPlaceholderText("http://localhost:11434 (Ollama) or http://localhost:1234/v1 (LM Studio)")
        self.embedding_local_url_input.setToolTip(
            "URL for your selected local embedding backend. Ollama uses its native base URL; LM Studio, Jan, and custom servers use an OpenAI-compatible /v1 URL."
        )

        embedding_local_layout.addWidget(settings_inline_row(theme, "Server URL", self.embedding_local_url_input, 320))

        self.embedding_local_model_input = QLineEdit()

        self.embedding_local_model_input.setPlaceholderText("e.g. nomic-embed-text or text-embedding model")
        self.embedding_local_model_input.setToolTip(
            "Type a local embedding model manually, or click Find models to choose from the selected server."
        )

        self.embedding_local_refresh_models_btn = QPushButton("Find models")
        self.embedding_local_refresh_models_btn.setStyleSheet(settings_button_style(theme, "muted"))

        self.embedding_local_refresh_models_btn.setToolTip("Fetch models from the shown embedding server, or autodetect a running local server if needed.")

        self.embedding_local_refresh_models_btn.clicked.connect(self._find_local_embedding_models)

        embedding_local_layout.addWidget(
            settings_labeled_action_row(
                theme,
                "Model",
                self.embedding_local_model_input,
                self.embedding_local_refresh_models_btn,
                control_width=450,
                tooltip=self.embedding_local_refresh_models_btn.toolTip(),
            )
        )

        embedding_local_hint = QLabel("Must expose an /embeddings endpoint")
        embedding_local_hint.setWordWrap(True)

        embedding_local_hint.setStyleSheet(settings_text_style(theme, "subtle"))

        embedding_local_layout.addWidget(embedding_local_hint)

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

        self.embedding_cloud_group = settings_child_group(theme)
        self.embedding_cloud_group.content_layout.addWidget(self.embedding_cloud_section)
        embedding_independent_layout.addWidget(self.embedding_cloud_group)

        api_layout.addWidget(self.embedding_independent_section)

        self.test_connection_btn = self._make_connection_test_button(
            "Test Embedding Connection",
            EmbeddingsTabMessages.TEST_CONNECTION_TOOLTIP,
            self._test_embedding_connection,
        )

        api_layout.addWidget(settings_toolbar(theme, self.test_connection_btn))

        self._connect_embedding_signals()







        # Collapsible help: "Need help?" toggles visibility



        self._api_help_visible = False



        api_help_btn = QPushButton("Need help? (providers, free options)")
        self.api_help_btn = api_help_btn
        api_help_btn.setStyleSheet(settings_button_style(theme, "muted"))



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



        api_layout.addWidget(settings_toolbar(theme, api_help_btn))



        api_layout.addWidget(self.info_text)



        api_layout.addStretch()








        return api_tab
