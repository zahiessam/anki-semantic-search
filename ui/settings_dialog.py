"""Settings dialog UI for configuring search, embeddings, and providers."""

# ============================================================================
# Imports
# ============================================================================

from aqt.qt import *
from .settings_api_tab_mixin import SettingsApiTabMixin
from .settings_api_ui_mixin import SettingsApiUiMixin
from .settings_config_mixin import SettingsConfigMixin
from .settings_dialog_finalize_mixin import SettingsDialogFinalizeMixin
from .settings_dialog_layout_mixin import SettingsDialogLayoutMixin
from .settings_embedding_connection_mixin import SettingsEmbeddingConnectionMixin
from .settings_embedding_config_mixin import SettingsEmbeddingConfigMixin
from .settings_embedding_progress_mixin import SettingsEmbeddingProgressMixin
from .settings_embedding_status_mixin import SettingsEmbeddingStatusMixin
from .settings_embedding_workflow_mixin import SettingsEmbeddingWorkflowMixin
from .settings_note_filter_lifecycle_mixin import SettingsNoteFilterLifecycleMixin
from .settings_note_filter_config_mixin import SettingsNoteFilterConfigMixin
from .settings_note_filter_presets_mixin import SettingsNoteFilterPresetsMixin
from .settings_note_filter_tab_mixin import SettingsNoteFilterTabMixin
from .settings_note_filter_tables_mixin import SettingsNoteFilterTablesMixin
from .settings_scope_selector_mixin import SettingsScopeSelectorMixin
from .settings_provider_mixin import SettingsProviderMixin
from .settings_rerank_environment_mixin import SettingsRerankEnvironmentMixin
from .settings_rerank_model_mixin import SettingsRerankModelMixin
from .settings_rerank_ui_mixin import SettingsRerankUiMixin
from .settings_safe_ui_mixin import SettingsSafeUiMixin
from .settings_search_tab_mixin import SettingsSearchTabMixin
from .settings_styling_tab_mixin import SettingsStylingTabMixin
from .settings_wheel_mixin import SettingsWheelMixin
from .theme import get_addon_theme, settings_dialog_stylesheet
from ..core.workers import RerankCheckWorker
from ..utils import load_config, log_debug


# ============================================================================
# Settings Dialog
# ============================================================================

_addon_theme = get_addon_theme

class SettingsDialog(
    SettingsConfigMixin,
    SettingsApiTabMixin,
    SettingsStylingTabMixin,
    SettingsScopeSelectorMixin,
    SettingsNoteFilterTabMixin,
    SettingsSearchTabMixin,
    SettingsDialogFinalizeMixin,
    SettingsDialogLayoutMixin,
    SettingsApiUiMixin,
    SettingsEmbeddingConfigMixin,
    SettingsNoteFilterTablesMixin,
    SettingsNoteFilterConfigMixin,
    SettingsNoteFilterPresetsMixin,
    SettingsRerankEnvironmentMixin,
    SettingsRerankModelMixin,
    SettingsEmbeddingStatusMixin,
    SettingsEmbeddingConnectionMixin,
    SettingsEmbeddingWorkflowMixin,
    SettingsEmbeddingProgressMixin,
    SettingsProviderMixin,
    SettingsNoteFilterLifecycleMixin,
    SettingsSafeUiMixin,
    SettingsWheelMixin,
    SettingsRerankUiMixin,
    QDialog,
):



    # --- Lifecycle And Window Setup ---

    def __init__(self, parent=None, open_to_embeddings=False):



        import time



        _t0 = time.time()



        super().__init__(parent)



        self.open_to_embeddings = open_to_embeddings



        self.setWindowTitle("Anki Semantic Search \u2014 Settings")



        self._rerank_check_done = False  # defer rerank check until after show
        self._rerank_available = False
        self._rerank_availability_checking = True
        self._rerank_model_ready = False
        self._rerank_model_ready_name = ""
        self._rerank_verify_worker = None
        self._rerank_verify_token = None
        self._rerank_model_checking = False
        self._rerank_model_checking_name = ""
        self._rerank_model_verify_message = ""



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



                if not available:
                    config = load_config()
                    sc = (config or {}).get("search_config") or {}
                    cached_python = (sc.get("rerank_package_verified_python_path") or "")
                    current_python = rerank_python or ""
                    if bool(sc.get("rerank_package_verified")) and cached_python == current_python:
                        log_debug(
                            "Startup Cross-Encoder check failed, keeping cached verified "
                            f"status for Python: {current_python or 'Anki Python'}"
                        )
                        available = True

                self._rerank_available = available
                self._rerank_availability_checking = False
                if available:
                    self._persist_cached_rerank_package_status(rerank_python)

                self._rerank_model_ready = bool(getattr(self, "_rerank_model_ready", False))

                cb = getattr(self, "enable_rerank_cb", None)

                if cb is not None and not sip.isdeleted(cb):

                    cb.setEnabled(available)



                if hasattr(self, "_update_rerank_tooltip") and not sip.isdeleted(self):

                    self._update_rerank_tooltip()

                    self._update_rerank_status_ui()

                self._rerank_check_worker = None
                if available:
                    self._verify_selected_rerank_model_cache()

            except (RuntimeError, AttributeError, ImportError):

                pass







        self._rerank_check_worker.finished_signal.connect(_on_rerank_check_done)



        self._rerank_check_worker.start()










    # --- Config Loading And Preset Application ---



















    # --- Provider And API Configuration UI ---


















    # --- Note Type, Field, And Deck Filters ---



































































































































































































    # --- Answer And Embedding Provider Settings ---


    # explanation: wires all interactive signals for the embedding section.

    # explanation: toggles visibility between same-provider summary and independent embedding fields.

    # explanation: swaps local and cloud embedding sub-fields without saving config.

    # explanation: refreshes the embedding cloud provider detection label without saving config.

    # explanation: updates the same-provider summary and warning based on current answer settings.

    # explanation: shows or hides the independent embedding cloud key.

    # explanation: populates all embedding UI fields from config without triggering signals.

    # explanation: reads all embedding UI fields and writes to config-compatible keys.

    # explanation: builds a transient config from the current Answer Provider widgets for validation/tests.


    # --- Settings Save And Reload ---








    # --- Rerank Environment And Local Model Checks ---






































































































    # --- Embedding Status And Indexing Actions ---



























































