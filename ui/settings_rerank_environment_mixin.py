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

from .dependency_install import _is_real_python_executable, _resolve_external_python_exe, install_dependencies
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

class SettingsRerankEnvironmentMixin:
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

                if not _is_real_python_executable(python_path):



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



            if not _is_real_python_executable(sys.executable):



                return False



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



        base = "Re-ranks top results with a cross-encoder for better relevance.\n"



        model_ready = bool(getattr(self, "_rerank_model_ready", False))

        if self._rerank_available and model_ready:



            self.enable_rerank_cb.setToolTip(base + "Package and selected model are ready to use.")

        elif self._rerank_available:

            self.enable_rerank_cb.setToolTip(
                base + "Package is installed, but the selected model has not been verified. "
                "Click Verify / Download selected model to verify it before searching."
            )



        else:



            self.enable_rerank_cb.setToolTip(



                base + "Not installed. Use an external Python (e.g. Python 3.11) with "



                "sentence-transformers. This is recommended on Windows because Anki's Python may not load torch.\n"
                "Anki's Python: " + sys.executable



            )

        apply_setting_row_tooltip(
            getattr(self, "enable_rerank_row", None),
            self.enable_rerank_cb.toolTip(),
        )


    def _update_rerank_controls_enabled(self):

        """Enable rerank-only controls when re-ranking itself is enabled."""

        rerank_cb = getattr(self, "enable_rerank_cb", None)
        rerank_enabled = bool(
            rerank_cb is not None
            and rerank_cb.isEnabled()
            and rerank_cb.isChecked()
        )

        context_cb = getattr(self, "use_context_boost_cb", None)
        if context_cb is not None:
            context_cb.setEnabled(rerank_enabled)

        model_card = getattr(self, "rerank_model_card", None)
        if model_card is not None:
            model_card.setEnabled(rerank_enabled)


    def _update_rerank_status_ui(self):



        """Update the visible Cross-Encoder setup status."""



        label = getattr(self, "rerank_status_label", None)
        model_label = getattr(self, "rerank_model_status_label", None)
        theme = _addon_theme()



        if label is None:



            return



        package_ready = bool(self._rerank_available) and not getattr(self, "_rerank_availability_checking", False)

        if self._rerank_available:
            label.setText("Cross-Encoder package: Ready")
            label.setStyleSheet(settings_status_label_style(theme, "success"))
        elif getattr(self, "_rerank_availability_checking", False):
            label.setText("Cross-Encoder package: checking...")
            label.setStyleSheet(settings_status_label_style(theme, "warning"))
        else:
            label.setText("Cross-Encoder package: Needs setup")
            label.setStyleSheet(settings_status_label_style(theme, "warning"))

        package_section = getattr(self, "python_path_widget", None)
        if package_section is not None and hasattr(package_section, "setTitle"):
            package_section.setTitle(
                "Python environment - Ready" if package_ready else "Python environment - Setup"
            )
            if hasattr(package_section, "setExpanded"):
                package_section.setExpanded(not package_ready)

        model_ready = False
        if model_label is not None:
            model = self._selected_rerank_model() if hasattr(self, "_selected_rerank_model") else DEFAULT_RERANK_MODEL
            if getattr(self, "_rerank_model_checking", False) and getattr(self, "_rerank_model_checking_name", "") == model:
                model_label.setText(f"Selected model: checking local cache ({model})")
                model_label.setStyleSheet(settings_status_label_style(theme, "warning"))
            elif getattr(self, "_rerank_model_ready", False) and getattr(self, "_rerank_model_ready_name", "") == model:
                model_label.setText(f"Selected model: Ready ({model})")
                model_label.setStyleSheet(settings_status_label_style(theme, "success"))
                model_ready = True
            else:
                model_label.setText(f"Selected model: Not verified ({model})")
                model_label.setStyleSheet(settings_status_label_style(theme, "warning"))

            message = getattr(self, "_rerank_model_verify_message", "")
            model_label.setToolTip(message or "")

        model_section = getattr(self, "rerank_model_card", None)
        if model_section is not None and hasattr(model_section, "setTitle"):
            model_section.setTitle(
                "Re-ranking model - Ready" if model_ready else "Re-ranking model - Verify"
            )
            if hasattr(model_section, "setExpanded"):
                model_section.setExpanded(not model_ready)

        self._update_rerank_controls_enabled()


    def _current_rerank_python_path(self):

        return (self.rerank_python_path_input.text() or '').strip() or None


    def _apply_cached_rerank_status(self, search_config):
        """Use the last successful rerank check while the background check refreshes it."""
        try:
            python_path = self._current_rerank_python_path() or ""
            model = self._selected_rerank_model()

            package_ready = (
                bool(search_config.get("rerank_package_verified"))
                and (search_config.get("rerank_package_verified_python_path") or "") == python_path
            )
            model_ready = (
                package_ready
                and bool(search_config.get("rerank_model_verified"))
                and (search_config.get("rerank_model_verified_python_path") or "") == python_path
                and (search_config.get("rerank_model_verified_name") or "") == model
            )

            if package_ready:
                self._rerank_available = True
                self._rerank_availability_checking = True
                if hasattr(self, "enable_rerank_cb"):
                    self.enable_rerank_cb.setEnabled(True)
            if model_ready:
                self._rerank_model_ready = True
                self._rerank_model_ready_name = model
                self._rerank_model_verify_message = f"Model was verified previously: {model}"

            self._update_rerank_tooltip()
            self._update_rerank_status_ui()
        except Exception as exc:
            log_debug(f"Could not apply cached rerank status: {exc}")


    def _persist_cached_rerank_package_status(self, python_path=None):
        try:
            config = load_config()
            sc = config.setdefault("search_config", {})
            sc["rerank_package_verified"] = True
            sc["rerank_package_verified_python_path"] = python_path or self._current_rerank_python_path() or ""
            save_config(config)
        except Exception as exc:
            log_debug(f"Could not persist rerank package status: {exc}")


    def _on_check_rerank_again(self):



        """Re-check sentence-transformers and update Cross-Encoder checkbox state and tooltip."""



        import sys



        python_path = self._current_rerank_python_path()

        self._rerank_model_ready = False

        self._rerank_model_ready_name = ""
        self._rerank_model_checking = False
        self._rerank_model_checking_name = ""
        self._rerank_model_verify_message = ""
        self._rerank_availability_checking = True



        self._rerank_available = self._check_rerank_available(python_path=python_path)
        self._rerank_availability_checking = False
        if self._rerank_available:
            self._persist_cached_rerank_package_status(python_path)



        self.enable_rerank_cb.setEnabled(self._rerank_available)



        self._update_rerank_tooltip()



        self._update_rerank_status_ui()



        if self._rerank_available:

            self._verify_selected_rerank_model_cache()


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


