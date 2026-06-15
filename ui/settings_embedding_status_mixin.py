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

class SettingsEmbeddingStatusMixin:
    def _update_agentic_planner_controls(self):
        """Reflect whether the smart planner owns retrieval-method selection."""
        agentic_on = bool(
            getattr(self, "enable_agentic_rag_cb", None)
            and self.enable_agentic_rag_cb.isChecked()
        )
        if hasattr(self, "agentic_planner_mode_combo"):
            self.agentic_planner_mode_combo.setEnabled(agentic_on)

        if hasattr(self, "agentic_planner_hint_label"):
            if not agentic_on:
                hint = "Enable Smart Retrieval Planner to let Ask Notes check evidence coverage and plan retrieval."
            else:
                hint = "Smart Retrieval Planner chooses the retrieval strategy automatically for each search."
            self.agentic_planner_hint_label.setText(hint)

        if hasattr(self, "search_method_combo"):
            manual_strategy_tooltip = (
                "Choose how notes are retrieved: keyword, semantic embeddings, hybrid, or keyword with re-ranking."
            )
            planner_owned_tooltip = "Controlled by Smart Retrieval Planner while Smart Retrieval is enabled."
            self.search_method_combo.setEnabled(not agentic_on)
            if hasattr(self, "search_method_row"):
                self.search_method_row.setEnabled(not agentic_on)
            if agentic_on:
                self.search_method_combo.setToolTip(planner_owned_tooltip)
                if hasattr(self, "search_method_row"):
                    self.search_method_row.setToolTip(planner_owned_tooltip)
                    for label in self.search_method_row.findChildren(QLabel):
                        label.setToolTip(planner_owned_tooltip)
            else:
                self.search_method_combo.setToolTip(manual_strategy_tooltip)
                if hasattr(self, "search_method_row"):
                    self.search_method_row.setToolTip(manual_strategy_tooltip)
                    for label in self.search_method_row.findChildren(QLabel):
                        label.setToolTip(manual_strategy_tooltip)
            if hasattr(self, "search_method_planner_hint_row"):
                self.search_method_planner_hint_row.setVisible(agentic_on)

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

        self._update_agentic_planner_controls()


    def _update_retrieval_v2_controls(self, mark_dirty=False):
        """Update MMR controls and show dirty status only after user changes."""
        mmr_on = bool(getattr(self, "mmr_enabled_cb", None) and self.mmr_enabled_cb.isChecked())

        self.mmr_lambda_slider.setEnabled(mmr_on)
        self.mmr_subcontrols_widget.setEnabled(mmr_on)

        if hasattr(self, "mmr_lambda_value_label"):
            self.mmr_lambda_value_label.setText(f"{self.mmr_lambda_slider.value()}%")
            self.mmr_lambda_value_label.setEnabled(mmr_on)

        if mark_dirty and hasattr(self, "retrieval_dirty_label"):
            self.retrieval_dirty_label.show()
            if hasattr(self, "retrieval_dirty_row"):
                self.retrieval_dirty_row.show()


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

                if engine == "ollama":
                    base_url = (
                        effective_sc.get("ollama_base_url")
                        or effective_sc.get("local_llm_url")
                        or effective_sc.get("embedding_local_url")
                        or 'http://localhost:11434'
                    )

                    model = (
                        effective_sc.get("ollama_embed_model")
                        or effective_sc.get("local_llm_model")
                        or 'nomic-embed-text'
                    )
                else:
                    base_url = (
                        effective_sc.get("local_llm_url")
                        or effective_sc.get("embedding_local_url")
                        or effective_sc.get("ollama_base_url")
                        or 'http://localhost:1234/v1'
                    )

                    model = (
                        effective_sc.get("local_llm_model")
                        or effective_sc.get("embedding_local_model")
                        or 'text-embedding-3-small'
                    )

                status_text = (
                    f"Connected: local embeddings | URL: {base_url} | Model: {model}\n"
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

                    status_text = "Embedding provider ready: Voyage AI | API key detected."



            elif engine == "openai":

                api_key = (effective_sc.get("openai_embedding_api_key") or "").strip()

                model = effective_sc.get("openai_embedding_model") or "text-embedding-3-small"

                if not api_key:

                    status_text = (

                        "DISABLED: OpenAI\n\n"

                        "Enter your OpenAI API key above to use OpenAI embeddings."

                    )

                else:

                    status_text = f"Embedding provider ready: OpenAI - {model}"



            elif engine == "cohere":

                api_key = (effective_sc.get("cohere_api_key") or "").strip()

                if not api_key:

                    status_text = (

                        "DISABLED: Cohere\n\n"

                        "Enter your Cohere API key above to enable embeddings."

                    )

                else:

                    status_text = "Embedding provider ready: Cohere | API key detected."

            elif not valid:

                status_text = message



            if hasattr(self, 'embedding_status_label'):

                self.embedding_status_label.setText(status_text)
                theme = _addon_theme()

                status_lower = status_text.lower()
                if "ready" in status_lower or "connected" in status_lower:

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



                    f"Please wait a few seconds and click 'Test Embedding Connection' to verify.\n\n"



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


