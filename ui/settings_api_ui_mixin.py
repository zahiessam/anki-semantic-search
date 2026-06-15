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
from .widgets import (
    CollapsibleSection,
    apply_setting_row_tooltip,
    settings_field_row,
    settings_inline_row,
    settings_labeled_action_row,
    sync_setting_row_tooltips,
)
from ..core.cloud_diagnostics import (
    classify_provider_error,
    provider_status_message,
    test_cloud_answer_connection,
)
from ..core.engine import (
    OLLAMA_FREE_CLOUD_MODEL_CANDIDATES,
    analyze_note_eligibility,
    clear_checkpoint,
    count_notes_matching_config,
    delete_ollama_model,
    get_deck_names,
    get_embedding_engine_id,
    get_embedding_for_query,
    get_models_with_fields,
    get_notes_count_per_deck,
    get_notes_count_per_model,
    get_ollama_model_capabilities,
    get_ollama_models,
    get_ollama_models_with_stale_cloud,
    is_ollama_cloud_model,
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


def _infer_local_backend_from_url_value(url, default="custom_openai"):
    value = (url or "").strip().lower()
    if "localhost:11434" in value or "127.0.0.1:11434" in value or ":11434" in value:
        return "ollama"
    if "localhost:1234" in value or "127.0.0.1:1234" in value or ":1234" in value:
        return "lm_studio"
    if "localhost:1337" in value or "127.0.0.1:1337" in value or ":1337" in value:
        return "jan"
    return default


def _normalize_local_backend_url_value(backend, url):
    value = (url or "").strip()
    if not value:
        if backend == "ollama":
            value = "http://localhost:11434"
        elif backend == "jan":
            value = "http://localhost:1337/v1"
        else:
            value = "http://localhost:1234/v1"
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


def _fetch_local_models_for_backend_value(backend, url, timeout=3):
    backend = (backend or _infer_local_backend_from_url_value(url)).strip()
    normalized_url = _normalize_local_backend_url_value(backend, url)
    if backend == "ollama":
        models, _stale_cloud_models = get_ollama_models_with_stale_cloud(normalized_url)
        return models, normalized_url

    import requests

    models_url = f"{normalized_url.rstrip('/')}/models"
    resp = requests.get(models_url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    models = [
        str(item.get("id") or "").strip()
        for item in data.get("data", [])
        if isinstance(item, dict) and (item.get("id") or "").strip()
    ]
    return sorted(models), normalized_url


class LocalModelDiscoveryWorker(QThread):
    finished_signal = pyqtSignal(object)

    def __init__(self, backend, url):
        super().__init__()
        self.backend = backend
        self.url = url

    def run(self):
        errors = []
        url = (self.url or "").strip()
        if url:
            try:
                normalized_backend = (self.backend or _infer_local_backend_from_url_value(url)).strip()
                normalized_url = _normalize_local_backend_url_value(normalized_backend, url)
                if normalized_backend == "ollama":
                    models, stale_cloud_models = get_ollama_models_with_stale_cloud(normalized_url)
                    model_capabilities = get_ollama_model_capabilities(normalized_url, models)
                else:
                    models, normalized_url = _fetch_local_models_for_backend_value(
                        normalized_backend,
                        normalized_url,
                        timeout=3,
                    )
                    stale_cloud_models = []
                    model_capabilities = {}
                if models:
                    self.finished_signal.emit({
                        "status": "ok",
                        "source": "selected",
                        "server": {
                            "name": self.backend or _infer_local_backend_from_url_value(url),
                            "url": normalized_url,
                            "kind": self.backend or _infer_local_backend_from_url_value(url),
                            "models": models,
                            "stale_cloud_models": stale_cloud_models,
                            "model_capabilities": model_capabilities,
                        },
                        "errors": errors,
                    })
                    return
                errors.append(f"{url}: connected, no models found")
            except Exception as exc:
                errors.append(f"{url}: {exc}")

        candidates = [
            ("Ollama", "http://localhost:11434", "ollama"),
            ("LM Studio", "http://localhost:1234/v1", "openai"),
            ("Jan", "http://localhost:1337/v1", "openai"),
        ]
        detected = []
        for name, candidate_url, kind in candidates:
            try:
                if kind == "ollama":
                    models, stale_cloud_models = get_ollama_models_with_stale_cloud(candidate_url)
                    model_capabilities = get_ollama_model_capabilities(candidate_url, models)
                else:
                    import requests
                    stale_cloud_models = []
                    model_capabilities = {}

                    resp = requests.get(f"{candidate_url.rstrip('/')}/models", timeout=2)
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
                detected.append({
                    "name": name,
                    "url": _normalize_ollama_base_url(candidate_url) + "/v1" if kind == "ollama" else candidate_url,
                    "kind": kind,
                    "models": models,
                    "stale_cloud_models": stale_cloud_models,
                    "model_capabilities": model_capabilities,
                })
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        self.finished_signal.emit({
            "status": "ok" if detected else "none",
            "source": "autodetect",
            "detected": detected,
            "errors": errors,
        })


class SettingsApiUiMixin:
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
        layout.addWidget(settings_inline_row(theme, "Cloud provider", provider_combo, 260))

        key_input = QLineEdit()
        key_input.setEchoMode(QLineEdit.EchoMode.Password)
        key_input.setPlaceholderText(key_placeholder)
        key_input.textChanged.connect(changed_callback)
        setattr(self, key_attr, key_input)

        show_button = QPushButton("Show")
        show_button.setMaximumWidth(80)
        show_button.clicked.connect(show_callback)
        setattr(self, show_button_attr, show_button)
        layout.addWidget(settings_labeled_action_row(theme, "API key", key_input, show_button))

        detected_label = QLabel()
        detected_label.setStyleSheet(settings_text_style(theme, "summary"))
        detected_label.hide()
        setattr(self, detected_label_attr, detected_label)
        layout.addWidget(settings_field_row(theme, detected_label))

        return section


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


    def _fetch_local_models_for_backend(self, backend, url, timeout=5):
        """Return models exposed by one selected local backend and normalized URL used."""
        backend = (backend or self._infer_local_backend_from_url(url)).strip()
        normalized_url = self._normalize_local_backend_url(backend, url)
        if backend == "ollama":
            return get_ollama_models(normalized_url), normalized_url

        import requests

        models_url = f"{normalized_url.rstrip('/')}/models"
        resp = requests.get(models_url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        models = [
            str(item.get("id") or "").strip()
            for item in data.get("data", [])
            if isinstance(item, dict) and (item.get("id") or "").strip()
        ]
        return sorted(models), normalized_url


    def _refresh_local_model_field(self, backend, url, target_field, title, prefer_embedding=False, show_errors=True):
        url = (url or "").strip()
        if not url:
            if show_errors:
                showInfo("Please enter a Server URL first.")
            return False

        try:
            models, normalized_url = self._fetch_local_models_for_backend(backend, url)
            model_capabilities = (
                get_ollama_model_capabilities(normalized_url, models)
                if (backend or "").strip() == "ollama"
                else {}
            )
        except Exception as exc:
            if show_errors:
                showInfo(
                    f"Could not fetch models: {exc}\n\n"
                    f"Check that your local provider is running at {url}."
                )
            return False

        if not models:
            if show_errors:
                showInfo("Connected but no models found. Make sure a model is loaded in your server.")
            return False

        selected = self._choose_local_model(
            title,
            models,
            prefer_embedding=prefer_embedding,
            model_capabilities=model_capabilities,
        )
        if selected:
            target_field.setText(selected)
            try:
                from aqt.utils import tooltip

                tooltip(
                    f"Selected {self._local_model_choice_label(selected, model_capabilities)} from {normalized_url}"
                )
            except Exception:
                pass
            return True
        return False


    def _refresh_local_answer_models(self):
        """Fetch models from the selected local answer server only."""
        backend = (
            self.local_backend_combo.currentData()
            if hasattr(self, "local_backend_combo")
            else "custom_openai"
        ) or "custom_openai"
        self._refresh_local_model_field(
            backend,
            self.local_llm_url.text() if hasattr(self, "local_llm_url") else "",
            self.local_llm_model,
            "Select an answer model from your local server:",
            prefer_embedding=False,
        )


    def _refresh_local_embedding_models(self):
        """Fetch models from the selected local embedding server only."""
        backend = (
            self.embedding_local_backend_combo.currentData()
            if hasattr(self, "embedding_local_backend_combo")
            else "custom_openai"
        ) or "custom_openai"
        self._refresh_local_model_field(
            backend,
            self.embedding_local_url_input.text() if hasattr(self, "embedding_local_url_input") else "",
            self.embedding_local_model_input,
            "Select an embedding model from your local server:",
            prefer_embedding=True,
        )


    def _refresh_local_models(self):
        """Backward-compatible entrypoint for older signal hookups."""
        self._refresh_local_answer_models()


    def _find_local_answer_models(self):
        """Refresh the selected answer server, falling back to local-server autodetect."""
        backend = (
            self.local_backend_combo.currentData()
            if hasattr(self, "local_backend_combo")
            else "custom_openai"
        ) or "custom_openai"
        self._start_local_model_discovery(
            "answer",
            backend,
            self.local_llm_url.text() if hasattr(self, "local_llm_url") else "",
        )


    def _find_local_embedding_models(self):
        """Refresh the selected embedding server, falling back to local-server autodetect."""
        backend = (
            self.embedding_local_backend_combo.currentData()
            if hasattr(self, "embedding_local_backend_combo")
            else "custom_openai"
        ) or "custom_openai"
        self._start_local_model_discovery(
            "embedding",
            backend,
            self.embedding_local_url_input.text() if hasattr(self, "embedding_local_url_input") else "",
        )


    def _start_local_model_discovery(self, target, backend, url):
        button = (
            getattr(self, "embedding_local_refresh_models_btn", None)
            if target == "embedding"
            else getattr(self, "local_server_refresh_models_btn", None)
        )
        if button is not None:
            button.setEnabled(False)
            button.setText("Finding...")
        try:
            tooltip("Looking for local models...")
        except Exception:
            pass
        worker = LocalModelDiscoveryWorker(backend, url)
        worker.finished_signal.connect(
            lambda result, target=target, worker=worker: self._on_local_model_discovery_finished(target, worker, result)
        )
        if not hasattr(self, "_local_model_discovery_workers"):
            self._local_model_discovery_workers = []
        self._local_model_discovery_workers.append(worker)
        worker.start()


    def _on_local_model_discovery_finished(self, target, worker, result):
        if hasattr(self, "_local_model_discovery_workers"):
            try:
                self._local_model_discovery_workers.remove(worker)
            except ValueError:
                pass
        button = (
            getattr(self, "embedding_local_refresh_models_btn", None)
            if target == "embedding"
            else getattr(self, "local_server_refresh_models_btn", None)
        )
        if button is not None:
            button.setEnabled(True)
            button.setText("Find models")

        result = result or {}
        server = result.get("server")
        detected = result.get("detected") or []
        errors = result.get("errors") or []
        if not server and detected:
            server = detected[0]
            if len(detected) > 1:
                from aqt.utils import chooseList

                labels = [
                    f"{item['name']} - {item['url']} ({len(item['models'])} models)"
                    for item in detected
                ]
                idx = chooseList(
                    "Select a local embedding server:" if target == "embedding" else "Select a local answer server:",
                    labels,
                )
                if idx < 0:
                    return
                server = detected[idx]

        if not server:
            showInfo(
                "No running local AI server was detected.\n\n"
                "Start Ollama, LM Studio, or Jan, then click Find models again.\n\n"
                + "\n".join(errors[:3])
            )
            return

        self._offer_remove_stale_ollama_cloud_models(server)

        if target == "embedding":
            self._apply_discovered_embedding_server(server)
        else:
            self._apply_discovered_answer_server(server)


    def _offer_remove_stale_ollama_cloud_models(self, server):
        if (server.get("kind") or "").lower() != "ollama":
            return
        stale_models = [model for model in server.get("stale_cloud_models") or [] if model]
        if not stale_models:
            return

        preview = "\n".join(f"- {model}" for model in stale_models[:12])
        if len(stale_models) > 12:
            preview += f"\n- ...and {len(stale_models) - 12} more"
        reply = QMessageBox.question(
            self,
            "Remove unavailable Ollama Cloud models?",
            "These Ollama Cloud models are installed locally, but they are hidden from selection because "
            "they are not in the verified free-tier list or could not be confirmed as available. "
            "Selecting them may fail with a subscription-required message:\n\n"
            f"{preview}\n\n"
            "Remove them from Ollama now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        removed = []
        failed = []
        failed_models = []
        base_url = _normalize_ollama_base_url(server.get("url") or "http://localhost:11434")
        for model in stale_models:
            try:
                delete_ollama_model(base_url, model)
                removed.append(model)
            except Exception as exc:
                failed_models.append(model)
                failed.append(f"{model}: {exc}")
                log_debug(f"Could not remove stale Ollama Cloud model {model}: {exc}")

        remaining_stale = set(failed_models)
        server["stale_cloud_models"] = [model for model in stale_models if model in remaining_stale]
        server["models"] = [model for model in (server.get("models") or []) if model not in removed]

        if failed:
            showInfo(
                f"Removed {len(removed)} stale Ollama Cloud model(s).\n\n"
                "Some models could not be removed:\n"
                + "\n".join(failed[:8])
            )
        elif removed:
            tooltip(f"Removed {len(removed)} stale Ollama Cloud model(s).")


    def _apply_discovered_answer_server(self, server):
        models = server.get("models") or []
        model_capabilities = server.get("model_capabilities") or {}
        self._set_local_backend_from_server(server.get("name"), server.get("kind"), server.get("url"))
        self.local_llm_url.setText(
            self._normalize_local_backend_url(self.local_backend_combo.currentData(), server["url"])
        )
        self.local_llm_model.setText(
            self._choose_local_model(
                f"{server.get('name') or 'Local server'} detected. Select a chat model:",
                models,
                model_capabilities=model_capabilities,
            )
        )
        provider_idx = self.answer_provider_combo.findData("local_server")
        if provider_idx >= 0:
            self.answer_provider_combo.setCurrentIndex(provider_idx)
        if hasattr(self, "_on_answer_provider_changed"):
            self._on_answer_provider_changed()
        tooltip(
            f"Selected {self._local_model_choice_label(self.local_llm_model.text(), model_capabilities)} "
            f"from {server.get('url')}."
        )


    def _apply_discovered_embedding_server(self, server):
        models = server.get("models") or []
        model_capabilities = server.get("model_capabilities") or {}
        if hasattr(self, "embedding_local_backend_combo"):
            self._set_local_backend_combo_from_server(
                self.embedding_local_backend_combo,
                server.get("name"),
                server.get("kind"),
                server.get("url"),
            )
            backend = self.embedding_local_backend_combo.currentData()
        else:
            backend = self._infer_local_backend_from_url(server.get("url"))
        self.embedding_local_url_input.setText(
            self._normalize_local_backend_url(backend, server["url"])
        )
        self.embedding_local_model_input.setText(
            self._choose_local_model(
                f"{server.get('name') or 'Local server'} detected. Select an embedding model:",
                models,
                prefer_embedding=True,
                model_capabilities=model_capabilities,
            )
        )
        strategy_idx = self.embedding_strategy_combo.findData("local")
        if strategy_idx >= 0:
            self.embedding_strategy_combo.setCurrentIndex(strategy_idx)
        if hasattr(self, "_on_same_provider_toggled"):
            self._on_same_provider_toggled()
        tooltip(
            f"Selected {self._local_model_choice_label(self.embedding_local_model_input.text(), model_capabilities)} "
            f"from {server.get('url')}."
        )


    def _detect_local_servers(self):

        """Return running local AI servers and their available models."""

        import requests

        candidates = [
            ("Ollama", "http://localhost:11434", "ollama"),
            ("LM Studio", "http://localhost:1234/v1", "openai"),
            ("Jan", "http://localhost:1337/v1", "openai"),
        ]

        detected = []

        errors = []

        for name, url, kind in candidates:

            try:

                if kind == "ollama":

                    models, stale_cloud_models = get_ollama_models_with_stale_cloud(url)
                    model_capabilities = get_ollama_model_capabilities(url, models)

                else:
                    stale_cloud_models = []
                    model_capabilities = {}

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

                detected.append({
                    "name": name,
                    "url": _normalize_ollama_base_url(url) + "/v1" if kind == "ollama" else url,
                    "kind": kind,
                    "models": models,
                    "stale_cloud_models": stale_cloud_models,
                    "model_capabilities": model_capabilities,
                })

            except Exception as exc:

                errors.append(f"{name}: {exc}")

        return detected, errors


    def _choose_local_model(self, title, models, prefer_embedding=False, model_capabilities=None):

        from aqt.utils import chooseList
        model_capabilities = model_capabilities or {}

        if not models:

            return ""

        if prefer_embedding:

            preferred_terms = ("embed", "nomic", "bge", "e5", "jina", "gte", "minilm")

            likely_models = [
                model
                for model in models
                if any(term in model.lower() for term in preferred_terms)
            ]

            if len(likely_models) == 1:

                return likely_models[0]

            if len(likely_models) > 1:

                ordered_models = likely_models + [model for model in models if model not in likely_models]

                labels = [
                    f"Recommended embedding model: {self._local_model_choice_label(model, model_capabilities)}"
                    if model in likely_models
                    else self._local_model_choice_label(model, model_capabilities)
                    for model in ordered_models
                ]

                idx = chooseList(title, labels)

                return ordered_models[idx] if idx >= 0 else likely_models[0]

        if len(models) == 1:

            return models[0]

        labels = [self._local_model_choice_label(model, model_capabilities) for model in models]
        idx = chooseList(title, labels)

        return models[idx] if idx >= 0 else models[0]


    def _local_model_choice_label(self, model, model_capabilities=None):
        model = (model or "").strip()
        parts = [model]
        if is_ollama_cloud_model(model):
            if model in OLLAMA_FREE_CLOUD_MODEL_CANDIDATES:
                parts.append("Ollama Cloud - free-tier verified")
            else:
                parts.append("Ollama Cloud - may require subscription")

        capabilities = (model_capabilities or {}).get(model)
        if capabilities is not None:
            parts.append("Image: yes" if "vision" in capabilities else "Image: no")
        elif is_ollama_cloud_model(model):
            parts.append("Image: unknown")

        return f"{parts[0]} ({'; '.join(parts[1:])})" if len(parts) > 1 else parts[0]


    def _autodetect_local_server(self):

        """Find a running local AI server and populate answer URL/model fields."""

        from aqt.utils import tooltip

        detected, errors = self._detect_local_servers()

        if detected:
            server = detected[0]
            if len(detected) > 1:
                from aqt.utils import chooseList

                labels = [
                    f"{server['name']} - {server['url']} ({len(server['models'])} models)"
                    for server in detected
                ]
                idx = chooseList("Select a local answer server:", labels)
                if idx < 0:
                    return
                server = detected[idx]

            models = server["models"]
            self._offer_remove_stale_ollama_cloud_models(server)
            models = server["models"]
            model_capabilities = server.get("model_capabilities") or {}

            self._set_local_backend_from_server(server.get("name"), server.get("kind"), server.get("url"))

            self.local_llm_url.setText(
                self._normalize_local_backend_url(self.local_backend_combo.currentData(), server["url"])
            )

            self.local_llm_model.setText(
                self._choose_local_model(
                    f"{server['name']} detected. Select a chat model:",
                    models,
                    model_capabilities=model_capabilities,
                )
            )

            provider_idx = self.answer_provider_combo.findData("local_server")

            if provider_idx >= 0:

                self.answer_provider_combo.setCurrentIndex(provider_idx)

            if hasattr(self, "_on_answer_provider_changed"):

                self._on_answer_provider_changed()

            tooltip(f"Detected {server['name']}. Server and model fields updated.")

            return

        showInfo(
            "No running local AI server was detected.\n\n"
            "Start Ollama, LM Studio, or Jan, then click Find models again.\n\n"
            + "\n".join(errors[:3])
        )


    def _autodetect_embedding_local_server(self):

        """Find a running local AI server and populate embedding URL/model fields."""

        from aqt.utils import tooltip

        detected, errors = self._detect_local_servers()

        if detected:
            server = detected[0]
            if len(detected) > 1:
                from aqt.utils import chooseList

                labels = [
                    f"{server['name']} - {server['url']} ({len(server['models'])} models)"
                    for server in detected
                ]
                idx = chooseList("Select a local embedding server:", labels)
                if idx < 0:
                    return
                server = detected[idx]

            models = server["models"]
            self._offer_remove_stale_ollama_cloud_models(server)
            models = server["models"]
            model_capabilities = server.get("model_capabilities") or {}

            if hasattr(self, "embedding_local_backend_combo"):
                self._set_local_backend_combo_from_server(
                    self.embedding_local_backend_combo,
                    server.get("name"),
                    server.get("kind"),
                    server.get("url"),
                )
                backend = self.embedding_local_backend_combo.currentData()
            else:
                backend = self._infer_local_backend_from_url(server.get("url"))

            self.embedding_local_url_input.setText(
                self._normalize_local_backend_url(backend, server["url"])
            )

            self.embedding_local_model_input.setText(
                self._choose_local_model(
                    f"{server['name']} detected. Select an embedding model:",
                    models,
                    prefer_embedding=True,
                    model_capabilities=model_capabilities,
                )
            )

            strategy_idx = self.embedding_strategy_combo.findData("local")

            if strategy_idx >= 0:

                self.embedding_strategy_combo.setCurrentIndex(strategy_idx)

            if hasattr(self, "_on_same_provider_toggled"):

                self._on_same_provider_toggled()

            tooltip(f"Detected {server['name']}. Embedding server and model fields updated.")

            return

        showInfo(
            "No running local AI server was detected.\n\n"
            "Start Ollama, LM Studio, or Jan, then click Find models again.\n\n"
            + "\n".join(errors[:3])
        )


    def _make_connection_test_button(self, text, tooltip_text, callback):

        """Create consistently themed test-connection buttons."""

        button = QPushButton(text)

        button.setToolTip(tooltip_text)

        button.setStyleSheet(settings_button_style(_addon_theme(), "muted"))

        button.clicked.connect(callback)

        return button


    def _show_connection_test_result(self, success, title, details="", status_label=None, status_text=None):

        """Show a consistent connection-test result and optionally update a status label."""

        prefix = "OK" if success else "Could not connect"

        message = f"{prefix}: {title}"

        if details:

            message = f"{message}\n\n{details}"

        if status_label is not None:

            status_label.setText(status_text or message.splitlines()[0])

            state = "success" if success else "error"

            status_label.setStyleSheet(settings_status_label_style(_addon_theme(), state))

        showInfo(message)


    def _test_cloud_api_connection(self):
        """Test the selected cloud answer provider without saving settings."""

        config = self._config_with_current_answer_provider(load_config())
        provider = (config.get("provider") or "openai").strip().lower()
        if provider in ("ollama", "local_openai", "local_server"):
            showInfo("Choose Cloud API under 'Answer with' to test a cloud provider.")
            return

        button = getattr(self, "answer_cloud_test_btn", None)
        status_label = getattr(self, "answer_cloud_status_label", None)
        if button is not None:
            button.setEnabled(False)
            button.setText("Testing...")
        if status_label is not None:
            status_label.setText("Testing cloud API connection...")
            status_label.setStyleSheet(settings_status_label_style(_addon_theme(), "warning"))
        QApplication.processEvents()

        try:
            result = test_cloud_answer_connection(
                provider=provider,
                api_key=config.get("api_key", ""),
                api_url=config.get("api_url", ""),
                timeout=8,
            )
            detail_lines = [
                result.detail or provider_status_message(result.status),
                f"Provider: {result.provider}",
                f"Check: {result.check_type}",
            ]
            if result.endpoint:
                detail_lines.append(f"Endpoint: {result.endpoint}")
            if result.http_status:
                detail_lines.append(f"HTTP status: {result.http_status}")
            if result.latency_ms:
                detail_lines.append(f"Latency: {result.latency_ms} ms")

            status_text = (
                f"{result.provider} OK ({result.latency_ms} ms)"
                if result.ok
                else provider_status_message(result.status)
            )
            self._show_connection_test_result(
                success=result.ok,
                title=(
                    f"{result.provider} connection OK"
                    if result.ok
                    else f"{result.provider} connection failed"
                ),
                details="\n".join(detail_lines),
                status_label=status_label,
                status_text=status_text,
            )
        finally:
            if button is not None:
                button.setEnabled(True)
                button.setText("\U0001F50C Test Cloud API Connection")


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

            if "11434" in url:
                test_url = f"{_normalize_ollama_base_url(url)}/api/tags"
            else:
                base = url.rstrip("/")
                if not base.startswith("http"):
                    base = "http://" + base
                if base.endswith("/chat/completions"):
                    base = base[:-17]
                if base.endswith("/models"):
                    base = base[:-7]
                if not base.endswith("/v1"):
                    base = base + "/v1"
                test_url = f"{base}/models"
            log_debug(f"Testing local server connection: {test_url}")

            resp = requests.get(test_url, timeout=5)

            elapsed = (time.time() - start) * 1000



            if resp.status_code == 200:

                self._show_connection_test_result(
                    success=True,
                    title="Answer server connection OK",
                    details=f"Latency: {elapsed:.0f} ms\nServer: {url}",
                )

            else:

                self._show_connection_test_result(
                    success=False,
                    title=f"Answer server responded with code {resp.status_code}",
                    details=f"URL: {test_url}",
                )

        except Exception as e:

            self._show_connection_test_result(
                success=False,
                title="Answer server connection failed",
                details=f"Error: {e}\n\nMake sure your server is running at {url}",
            )


