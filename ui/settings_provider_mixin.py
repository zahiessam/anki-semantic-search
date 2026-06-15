"""Provider/key helper behavior for the Settings dialog."""

from urllib.parse import urlparse

from aqt.qt import QLineEdit

from .settings_constants import ANSWER_CLOUD_PROVIDERS, ANSWER_KEY_PROVIDER_PREFIXES
from .theme import get_addon_theme, settings_text_style
from ..core.engine import _normalize_ollama_base_url


class SettingsProviderMixin:
    """Owns provider detection, local backend URL helpers, and key visibility."""

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

    def _default_local_backend_url(self, backend):
        return {
            "ollama": "http://localhost:11434",
            "lm_studio": "http://localhost:1234/v1",
            "jan": "http://localhost:1337/v1",
        }.get((backend or "").strip(), "")

    def _infer_local_backend(self, provider, search_config=None, config=None):
        if (provider or "").strip().lower() == "ollama":
            return "ollama"
        sc = search_config or {}
        cfg = config or {}
        url = (
            sc.get("local_llm_url")
            or cfg.get("local_llm_url")
            or cfg.get("api_url")
            or ""
        ).strip().lower()
        if "localhost:1234" in url or "127.0.0.1:1234" in url or ":1234" in url:
            return "lm_studio"
        if "localhost:1337" in url or "127.0.0.1:1337" in url or ":1337" in url:
            return "jan"
        return "custom_openai"

    def _infer_local_backend_from_url(self, url, default="custom_openai"):
        value = (url or "").strip().lower()
        if "localhost:11434" in value or "127.0.0.1:11434" in value or ":11434" in value:
            return "ollama"
        if "localhost:1234" in value or "127.0.0.1:1234" in value or ":1234" in value:
            return "lm_studio"
        if "localhost:1337" in value or "127.0.0.1:1337" in value or ":1337" in value:
            return "jan"
        return default

    def _normalize_local_backend_url(self, backend, url):
        value = (url or "").strip()
        if not value:
            value = self._default_local_backend_url(backend) or "http://localhost:1234/v1"
        if "://" not in value:
            value = "http://" + value
        if backend == "ollama":
            return _normalize_ollama_base_url(value)
        value = value.rstrip("/")
        lower_value = value.lower()
        if lower_value.endswith("/chat/completions"):
            value = value[:-17]
            lower_value = value.lower()
        if lower_value.endswith("/models"):
            value = value[:-7]
            lower_value = value.lower()
        if not lower_value.endswith("/v1"):
            value = value + "/v1"
        return value

    def _on_local_backend_changed(self):
        if not hasattr(self, "local_backend_combo") or not hasattr(self, "local_llm_url"):
            return
        backend = self.local_backend_combo.currentData() or "custom_openai"
        default_url = self._default_local_backend_url(backend)
        if not default_url:
            return
        current_url = (self.local_llm_url.text() or "").strip()
        known_defaults = {
            "http://localhost:11434",
            "http://localhost:11434/v1",
            "http://localhost:1234/v1",
            "http://localhost:1337/v1",
            "",
        }
        if current_url in known_defaults:
            self.local_llm_url.setText(default_url)

    def _on_embedding_local_backend_changed(self):
        if not hasattr(self, "embedding_local_backend_combo") or not hasattr(self, "embedding_local_url_input"):
            return
        backend = self.embedding_local_backend_combo.currentData() or "custom_openai"
        default_url = self._default_local_backend_url(backend)
        if not default_url:
            return
        current_url = (self.embedding_local_url_input.text() or "").strip()
        known_defaults = {
            "http://localhost:11434",
            "http://localhost:11434/v1",
            "http://localhost:1234/v1",
            "http://localhost:1337/v1",
            "",
        }
        if current_url in known_defaults:
            self.embedding_local_url_input.setText(default_url)

    def _set_local_backend_combo_from_server(self, combo, name=None, kind=None, url=None):
        if combo is None:
            return
        name_value = (name or "").strip().lower()
        kind_value = (kind or "").strip().lower()
        url_value = (url or "").strip().lower()
        if kind_value == "ollama" or "ollama" in name_value or ":11434" in url_value:
            backend = "ollama"
        elif "jan" in name_value or ":1337" in url_value:
            backend = "jan"
        elif "studio" in name_value or ":1234" in url_value:
            backend = "lm_studio"
        else:
            backend = "custom_openai"
        self._select_combo_data(combo, backend)

    def _set_local_backend_from_server(self, name=None, kind=None, url=None):
        if not hasattr(self, "local_backend_combo"):
            return
        self._set_local_backend_combo_from_server(self.local_backend_combo, name, kind, url)

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
        status_label = getattr(self, "answer_cloud_status_label", None)
        if status_label is not None:
            status_label.setText("Cloud API not tested")
            status_label.setStyleSheet(settings_text_style(get_addon_theme(), "summary"))
        return provider

    def toggle_key_visibility(self):
        self._toggle_password_visibility("api_key_input", "show_key_btn")

    def _toggle_voyage_key_visibility(self):
        self._toggle_password_visibility("voyage_api_key_input", "voyage_show_key_btn")

    def _toggle_openai_key_visibility(self):
        self._toggle_password_visibility("openai_embedding_api_key_input", "openai_show_key_btn")

    def _toggle_cohere_key_visibility(self):
        self._toggle_password_visibility("cohere_api_key_input", "cohere_show_key_btn")
