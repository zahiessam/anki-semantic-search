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


class SettingsEmbeddingConfigMixin:
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


    def _on_embedding_engine_changed(self):

        """Compatibility wrapper for the old Search-tab embedding signal."""

        # explanation: provider visibility is now owned by the API-tab embedding controls.
        if hasattr(self, "_on_embedding_strategy_changed"):

            self._on_embedding_strategy_changed()


    def _connect_embedding_signals(self):
        self.answer_provider_combo.currentIndexChanged.connect(self._on_answer_provider_changed)
        self.answer_cloud_provider_combo.currentIndexChanged.connect(self._update_embedding_same_summary)
        self.api_key_input.textChanged.connect(self._update_embedding_same_summary)
        self.embedding_same_checkbox.stateChanged.connect(self._on_same_provider_toggled)
        self.embedding_strategy_combo.currentIndexChanged.connect(self._on_embedding_strategy_changed)


    def _on_same_provider_toggled(self, *args):
        same_provider = self.embedding_same_checkbox.isChecked()
        self.embedding_same_summary_label.setVisible(same_provider)
        self.embedding_independent_section.setVisible(not same_provider)
        self._update_embedding_same_summary()
        self._on_embedding_strategy_changed()


    def _on_embedding_strategy_changed(self, *args):
        if not hasattr(self, "embedding_strategy_combo"):
            return
        is_independent = not self.embedding_same_checkbox.isChecked()
        strategy = self.embedding_strategy_combo.currentData() or "cloud"
        self.embedding_local_section.setVisible(is_independent and strategy == "local")
        self.embedding_cloud_section.setVisible(is_independent and strategy == "cloud")
        if hasattr(self, "embedding_cloud_group"):
            self.embedding_cloud_group.setVisible(is_independent and strategy == "cloud")
        self._on_embedding_cloud_provider_changed()
        self._refresh_embedding_status()


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


    def _toggle_embedding_cloud_key_visibility(self):
        self._toggle_password_visibility("embedding_cloud_api_key_input", "embedding_cloud_show_key_btn")


    def _load_embedding_settings(self):
        config = load_config()
        sc = config.get("search_config") or {}
        widgets = [
            self.embedding_same_checkbox,
            self.embedding_strategy_combo,
            self.embedding_cloud_provider_combo,
            self.embedding_cloud_api_key_input,
            self.embedding_local_backend_combo,
            self.embedding_local_url_input,
            self.embedding_local_model_input,
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
            embedding_backend = (sc.get("embedding_local_backend") or "").strip()
            if not embedding_backend:
                legacy_engine = (sc.get("embedding_engine") or "").strip().lower()
                embedding_url = (
                    sc.get("embedding_local_url")
                    or sc.get("local_llm_url")
                    or sc.get("ollama_base_url")
                    or ""
                )
                embedding_backend = (
                    "ollama"
                    if legacy_engine == "ollama"
                    else self._infer_local_backend_from_url(embedding_url)
                )
            self._select_combo_data(self.embedding_local_backend_combo, embedding_backend)
            normalized_embedding_url = self._normalize_local_backend_url(
                embedding_backend,
                sc.get("embedding_local_url")
                or sc.get("local_llm_url")
                or sc.get("ollama_base_url")
                or self._default_local_backend_url(embedding_backend),
            )
            self.embedding_local_url_input.setText(
                normalized_embedding_url
            )
            self.embedding_local_model_input.setText(
                sc.get("embedding_local_model")
                or sc.get("ollama_embed_model")
                or "nomic-embed-text"
            )
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self._on_same_provider_toggled()


    def _save_embedding_settings(self):
        same_provider = self.embedding_same_checkbox.isChecked()
        strategy = self.embedding_strategy_combo.currentData() or "cloud"
        provider = self.embedding_cloud_provider_combo.currentData() or "Voyage AI"
        api_key = (self.embedding_cloud_api_key_input.text() or "").strip()
        local_backend = (
            self.embedding_local_backend_combo.currentData()
            if hasattr(self, "embedding_local_backend_combo")
            else "custom_openai"
        ) or "custom_openai"
        local_url = self._normalize_local_backend_url(
            local_backend,
            (self.embedding_local_url_input.text() or "").strip(),
        )
        local_model = (self.embedding_local_model_input.text() or "").strip()
        values = {
            "embedding_same_as_answer": same_provider,
            "embedding_strategy": strategy,
            "embedding_cloud_provider": provider,
            "embedding_cloud_api_key": api_key,
            "embedding_local_backend": local_backend,
            "embedding_local_url": local_url,
            "embedding_local_model": local_model,
        }
        if not same_provider:
            if strategy == "local":
                if local_backend == "ollama":
                    values["embedding_engine"] = "ollama"
                    values["ollama_base_url"] = local_url
                    values["ollama_embed_model"] = local_model or "nomic-embed-text"
                else:
                    values["embedding_engine"] = "local_openai"
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


    def _config_with_current_answer_provider(self, config):
        config = dict(config or {})
        sc = dict(config.get("search_config") or {})
        answer_with = self.answer_provider_combo.currentData() or ""
        if answer_with == "local_server":
            backend = self.local_backend_combo.currentData() if hasattr(self, "local_backend_combo") else None
            backend = backend or self._infer_local_backend(config.get("provider"), sc, config)
            config["provider"] = "ollama" if backend == "ollama" else "local_openai"
            answer_model = (self.local_llm_model.text() or "").strip()
            url = self._normalize_local_backend_url(
                backend,
                (self.local_llm_url.text() or "").strip(),
            )
            if backend == "ollama":
                sc["ollama_base_url"] = url
                sc["ollama_chat_model"] = answer_model or (sc.get("ollama_chat_model") or "llama3.2")
            else:
                sc["local_llm_url"] = url
                sc["answer_local_model"] = answer_model
                sc["local_llm_model"] = answer_model or (sc.get("local_llm_model") or "").strip()
        else:
            api_key = (self.api_key_input.text() or "").strip()
            config["api_key"] = api_key
            config["provider"] = self.answer_cloud_provider_combo.currentData() or "openai"
            if config["provider"] == "custom":
                config["api_url"] = (self.api_url_input.text() or "").strip()
        config["search_config"] = sc
        return config


